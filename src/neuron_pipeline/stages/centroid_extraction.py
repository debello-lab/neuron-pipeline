"""
Synapse / Bouton Centroid Extraction
Neural Reconstruction Pipeline -- Phase 2

For each BOUTON, SYNAPSE, and CONTACT segment this module computes the
3-D centroid position in physical (um) space.

Strategy
--------
Primary  : extract voxel mask via SegmentSurfaceExtractor, compute the
           geometric centroid of the binary mask.
Fallback : use the bounding-box centre (fast, ~equal for compact segments)
           when voxel extraction fails (e.g. empty / too large segment).

Output
------
CentroidTable  -- in-memory dataclass, iterable over CentroidEntry rows.
write_csv()    -- writes centroids.csv for Phase 3 consumption.

CSV columns
-----------
name, seg_id, role, cx_um, cy_um, cz_um, method

  method = 'voxel'          - centroid computed from voxel mask
           'bbox_fallback'  - centroid from bounding-box centre
           'anchor'         - centroid from VAST anchor point (last resort)
"""

import csv
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.segment_classifier import SegmentRegistry, SegmentInfo


@dataclass
class CentroidEntry:
    name: str
    seg_id: int
    role: str                        # BOUTON | SYNAPSE | CONTACT
    cx_um: float
    cy_um: float
    cz_um: float
    method: str                      # voxel | bbox_fallback | anchor


@dataclass
class CentroidTable:
    entries: List[CentroidEntry] = field(default_factory=list)

    # Convenience lookup: name -> CentroidEntry
    _by_name: Dict[str, CentroidEntry] = field(default_factory=dict, repr=False)

    def add(self, entry: CentroidEntry) -> None:
        self.entries.append(entry)
        self._by_name[entry.name] = entry

    def get(self, name: str) -> Optional[CentroidEntry]:
        return self._by_name.get(name)

    def by_role(self, role: str) -> List[CentroidEntry]:
        return [e for e in self.entries if e.role == role]

    def __len__(self) -> int:
        return len(self.entries)

    def write_csv(self, path: str) -> str:
        """Write centroids to a CSV file. Returns the path written."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['name', 'seg_id', 'role', 'cx_um', 'cy_um', 'cz_um', 'method']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                row = asdict(e)
                row['cx_um'] = f"{row['cx_um']:.4f}"
                row['cy_um'] = f"{row['cy_um']:.4f}"
                row['cz_um'] = f"{row['cz_um']:.4f}"
                writer.writerow(row)
        return path


# AXON and POST_SYN get full SWC files in Phase 1 -- their locations are
# captured by the skeleton.  BOUTON / SYNAPSE / CONTACT are compact objects
# whose 3-D position is fully described by a single centroid point.
_TARGET_ROLES = ('BOUTON', 'SYNAPSE', 'CONTACT')


class CentroidExtractor:
    """
    Extract centroid positions for compact annotation segments.

    Workflow for each segment
    -------------------------
    1. Attempt voxel mask extraction via SegmentSurfaceExtractor at MIP level
       `miplevel` (default 0 = full resolution, appropriate for small objects).
    2. Compute centroid as the mean voxel position converted to um.
    3. On failure, fall back to the bounding-box centre.
    4. If the bbox is also invalid, use the VAST anchor point.
    """

    # MIP level 0 (full resolution) is best for small segments.
    # Users may increase this if memory becomes a concern.
    DEFAULT_MIPLEVEL: int = 0
    DEFAULT_PADDING: int = 1          # 1-voxel bbox padding (minimal)

    def __init__(
        self,
        extractor: SegmentSurfaceExtractor,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.extractor = extractor
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract_centroids(
        self,
        registry: SegmentRegistry,
        miplevel: int = DEFAULT_MIPLEVEL,
        padding: int = DEFAULT_PADDING,
        roles: Tuple[str, ...] = _TARGET_ROLES,
    ) -> CentroidTable:
        """
        Compute centroids for all segments whose role is in *roles*.

        Args:
            registry  : SegmentRegistry built by Phase 0.
            miplevel  : MIP level for voxel extraction (0 = full resolution).
            padding   : Padding voxels added around the bounding box.
            roles     : Tuple of role strings to process.

        Returns:
            CentroidTable with one entry per processed segment.
        """
        targets = [
            info
            for info in registry.segments.values()
            if info.role in roles
        ]

        role_counts = {}
        for info in targets:
            role_counts[info.role] = role_counts.get(info.role, 0) + 1
        self.logger.info(
            f"Phase 2: {len(targets)} segments to process -- "
            + ", ".join(f"{r}: {n}" for r, n in sorted(role_counts.items()))
        )

        table = CentroidTable()
        voxel_ok = bbox_fallback = anchor_fallback = 0

        for info in targets:
            entry = self._process_segment(info, miplevel, padding)
            table.add(entry)

            if entry.method == 'voxel':
                voxel_ok += 1
            elif entry.method == 'bbox_fallback':
                bbox_fallback += 1
            else:
                anchor_fallback += 1

        self.logger.info(
            f"Phase 2 complete: {voxel_ok} voxel, {bbox_fallback} bbox-fallback, "
            f"{anchor_fallback} anchor-fallback  ({len(table)} total)"
        )
        return table

    # ------------------------------------------------------------------
    # Per-segment logic
    # ------------------------------------------------------------------

    def _process_segment(
        self,
        info: SegmentInfo,
        miplevel: int,
        padding: int,
    ) -> CentroidEntry:
        """Compute centroid for a single segment, with fallback chain."""
        self.logger.debug(f"  {info.role} {info.name} (seg {info.seg_id})")

        # --- Primary: voxel mask ---
        entry = self._centroid_from_voxels(info, miplevel, padding)
        if entry is not None:
            return entry

        # --- Fallback 1: bounding-box centre ---
        entry = self._centroid_from_bbox(info)
        if entry is not None:
            self.logger.warning(
                f"  {info.name}: voxel extraction failed, using bbox centre"
            )
            return entry

        # --- Fallback 2: VAST anchor point ---
        entry = self._centroid_from_anchor(info)
        self.logger.warning(
            f"  {info.name}: bbox also invalid, using VAST anchor point"
        )
        return entry

    def _centroid_from_voxels(
        self,
        info: SegmentInfo,
        miplevel: int,
        padding: int,
    ) -> Optional[CentroidEntry]:
        """
        Extract voxel mask and compute the mass centroid.

        Returns None on any error so the caller can fall back.
        """
        try:
            result = self.extractor.extract_segment_voxel(
                segment_id=info.seg_id,
                miplevel=miplevel,
                padding=padding,
            )
        except Exception as exc:
            self.logger.debug(f"  {info.name}: voxel extraction raised {exc}")
            return None

        if result is None:
            return None

        mask, bbox_min_vox, voxel_size_um, _ = result

        if mask is None or not np.any(mask):
            return None
        
        if bbox_min_vox is None or voxel_size_um is None:
            return None

        # mask is (Z, Y, X); bbox_min_vox is (minx, miny, minz)
        # voxel_size_um is (sx, sy, sz)
        z_idx, y_idx, x_idx = np.where(mask)

        
        cx_vox = x_idx.mean() + bbox_min_vox[0]
        cy_vox = y_idx.mean() + bbox_min_vox[1]
        cz_vox = z_idx.mean() + bbox_min_vox[2]

        # cx_vox = (x_idx + 0.5).mean() + bbox_min_vox[0]
        # cy_vox = (y_idx + 0.5).mean() + bbox_min_vox[1]
        # cz_vox = (z_idx + 0.5).mean() + bbox_min_vox[2]

        cx_um = float(cx_vox * voxel_size_um[0])
        cy_um = float(cy_vox * voxel_size_um[1])
        cz_um = float(cz_vox * voxel_size_um[2])

        return CentroidEntry(
            name=info.name,
            seg_id=info.seg_id,
            role=info.role,
            cx_um=cx_um,
            cy_um=cy_um,
            cz_um=cz_um,
            method='voxel',
        )

    def _centroid_from_bbox(
        self,
        info: SegmentInfo,
    ) -> Optional[CentroidEntry]:
        """
        Compute centroid from the bounding box centre.

        bbox is (x1, y1, z1, x2, y2, z2) in voxels.
        voxel_size is obtained from the extractor's cached dataset info.
        """
        bbox = info.bbox
        if not bbox or len(bbox) < 6:
            return None
        if bbox[0] < 0 or bbox[3] < 0:
            return None  # VAST reports invalid bbox for empty segments

        di = self.extractor.dataset_info
        if di is None:
            return None

        # Convert nm to um
        sx = di['voxelsizex'] / 1000.0
        sy = di['voxelsizey'] / 1000.0
        sz = di['voxelsizez'] / 1000.0

        cx_um = float((bbox[0] + bbox[3]) / 2.0 * sx)
        cy_um = float((bbox[1] + bbox[4]) / 2.0 * sy)
        cz_um = float((bbox[2] + bbox[5]) / 2.0 * sz)

        return CentroidEntry(
            name=info.name,
            seg_id=info.seg_id,
            role=info.role,
            cx_um=cx_um,
            cy_um=cy_um,
            cz_um=cz_um,
            method='bbox_fallback',
        )

    def _centroid_from_anchor(
        self,
        info: SegmentInfo,
    ) -> CentroidEntry:
        """
        Last-resort: use the VAST anchor point stored in segment_data.

        anchorpoint is (x, y, z) in voxels; convert to um using dataset info.
        If dataset info is unavailable, return zeros (worst-case placeholder).
        """
        try:
            seg_data = self.extractor.vast.get_segment_data(info.seg_id)
        except Exception:
            seg_data = None

        di = self.extractor.dataset_info
        if seg_data is not None and di is not None:
            ap = seg_data.get('anchorpoint', [0, 0, 0])
            sx = di['voxelsizex'] / 1000.0
            sy = di['voxelsizey'] / 1000.0
            sz = di['voxelsizez'] / 1000.0
            cx_um = float(ap[0] * sx)
            cy_um = float(ap[1] * sy)
            cz_um = float(ap[2] * sz)
        else:
            cx_um = cy_um = cz_um = 0.0

        return CentroidEntry(
            name=info.name,
            seg_id=info.seg_id,
            role=info.role,
            cx_um=cx_um,
            cy_um=cy_um,
            cz_um=cz_um,
            method='anchor',
        )

