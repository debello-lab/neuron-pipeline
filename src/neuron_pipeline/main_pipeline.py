"""
Author: Nicolas Randazzo
Entry point for the VASTpyAPI pipeline.

This script is used for the Debello Lab at UC Davis Neuroscience to analyze the auditory cortex of barn owls.
In order to run this script, a VAST instance with a proper segmentation file loaded must be running.

Pipeline phases:
  Phase 0  Segment classification & connectivity table
  Phase 1  Voxel extraction -> cleaning -> skeletonization -> SWC
  Phase 2  Synapse / bouton centroid extraction
  Phase 3  Map centroids to SWC cable locations
  Phase 4  Connectivity CSV + Arbor recipe generation
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import logging
import re
import traceback

import numpy as np
import networkx as nx

from vastpy.control.exporting import VASTControlClass

from neuron_pipeline.stages.segment_classifier import SegmentClassifier, SegmentRegistry
from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner
from neuron_pipeline.stages.skeletonization import SkeletonExtractor, SWCWriter
from neuron_pipeline.stages.centroid_extraction import CentroidExtractor, CentroidTable
from neuron_pipeline.stages.centroid_mapper import CentroidMapper, CableMappingTable
from neuron_pipeline.stages.connectivity_builder import ConnectivityBuilder, ConnectivityOutput

# ---------------------------------------------------------------------------
# Coordinate frame identifiers
# ---------------------------------------------------------------------------
COORD_FRAME_VOXEL_GLOBAL = 'vast_global_voxel_xyz'   # VAST dataset voxel space, corner-of-voxel
COORD_FRAME_VOXEL_LOCAL  = 'local_bbox_voxel_xyz'    # Bounding-box-relative voxel space
COORD_FRAME_ARRAY_INDEX  = 'numpy_array_index_zyx'   # NumPy array indices, ZYX order
COORD_FRAME_PHYSICAL     = 'physical_um_xyz_center'  # Physical µm, XYZ, center-of-voxel

VOXEL_CORNER = 'corner_of_voxel'
VOXEL_CENTER = 'center_of_voxel'

# ---------------------------------------------------------------------------
# Failure / warning tracking
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    segment_name: str
    segment_id: int
    role: str
    phase: str
    stage: str
    reason: str
    details: dict = field(default_factory=dict)

@dataclass
class WarningRecord:
    segment_name: str
    stage: str
    message: str
    severity: str = 'medium'  # 'low' | 'medium' | 'high'

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_pipeline_logger(output_dir: str) -> logging.Logger:
    """Create a single logger writing to both console and pipeline_TIMESTAMP.log."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / "logs" / f"pipeline_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("neuron_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    logger.info(f"Pipeline log: {log_path}")
    return logger

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_failure(logger: logging.Logger, stage: str, name: str, reason: str,
                details: Optional[dict] = None) -> None:
    """Log a processing failure with high visual prominence."""
    logger.error("")
    logger.error("=" * 80)
    logger.error(f"{stage.upper()} FAILED: {name}")
    logger.error(f"  Reason: {reason}")
    if details:
        for k, v in details.items():
            logger.error(f"  {k}: {v}")
    logger.error("=" * 80)
    logger.error("")


def log_warning_box(logger: logging.Logger, stage: str, name: str, message: str,
                    details: Optional[dict] = None) -> None:
    """Log a non-fatal warning with high visual prominence."""
    logger.warning("")
    logger.warning("=" * 80)
    logger.warning(f"{stage.upper()} WARNING: {name}")
    logger.warning(f"  {message}")
    if details:
        for k, v in details.items():
            logger.warning(f"  {k}: {v}")
    logger.warning("=" * 80)
    logger.warning("")


def log_phase_summary(logger: logging.Logger, phase_name: str, total: int,
                      n_succeeded: int, failures: List[FailureRecord],
                      warnings: List[WarningRecord]) -> None:
    """Log an end-of-phase summary table."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{phase_name.upper()} SUMMARY")
    logger.info(f"  Total:      {total}")
    logger.info(f"  Successful: {n_succeeded}")
    logger.info(f"  Failed:     {len(failures)}")
    if failures:
        by_stage: Dict[str, list] = {}
        for f in failures:
            by_stage.setdefault(f.stage, []).append(f)
        logger.info("  Failures by stage:")
        for stage_name, flist in sorted(by_stage.items()):
            logger.info(f"    {stage_name}: {len(flist)}")
        logger.info("  Failed segments:")
        for f in failures:
            logger.info(f"    {f.segment_name:12s} [{f.stage}] {f.reason}")
    if warnings:
        logger.info(f"  Warnings (non-fatal): {len(warnings)}")
        for w in warnings[:5]:
            logger.info(f"    {w.segment_name:12s} [{w.stage}] {w.message}")
        if len(warnings) > 5:
            logger.info(f"    ... and {len(warnings) - 5} more")
    logger.info("=" * 80)
    logger.info("")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_review_queue(
    entries: List[dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Write (or append to) review_queue.csv for segments that need human inspection.

    Each entry represents a segment where automated cleaning produced suspicious
    metrics — either a high closing fraction (closing added >20% of volume) or
    residual fragmentation (multiple components remained after cleaning).

    The file appends on re-runs so successive pipeline runs accumulate the queue.
    Use diag_cleaning_param_sweep.py to inspect flagged segments individually.
    """
    import csv as _csv

    if not entries:
        return

    path = Path(output_dir) / "review_queue.csv"
    fieldnames = [
        'segment_name', 'segment_id', 'role', 'reason',
        'closing_radius_um', 'original_voxels', 'closing_voxels_added',
        'closing_fraction', 'components_after_clean', 'recommended_action',
    ]

    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(entries)

    logger.info(
        f"Review queue: {len(entries)} segment(s) flagged for inspection "
        f"-> {path}"
    )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

#  Phase 1

def run_phase1(
    registry: SegmentRegistry,
    vast: VASTControlClass,
    logger: logging.Logger,
    output_dir: str = "./vast_export",
    miplevel: int = 1,
    padding: int = 1,
    spur_length_um: float = 2.0,
) -> Tuple[Dict[str, Tuple[nx.DiGraph, str]], List[FailureRecord], List[WarningRecord]]:
    """
    Phase 1: Extract and skeletonize every AXON and POST_SYN segment.
    After extraction but before skeletonization, the segments will be cleaned to remove small disconnected components and fill holes. The resulting skeletons are pruned to remove short spurs, and the final SWC files are written to disk.

    Args:
        registry: SegmentRegistry from Phase 0
        vast: Connected VASTControlClass instance
        logger: Shared pipeline logger
        output_dir: Base output directory
        miplevel: MIP level for voxel extraction (0=full, 1=half)
        padding: Padding voxels around bounding box
        spur_length_um: Prune leaf branches shorter than this

    Returns:
        Tuple of (results, failures, warnings) where results maps segment name
        -> (tree, swc_path). Trees are kept in memory for Phase 3.
    """
    extractor = SegmentSurfaceExtractor(vast, output_dir, logger=logger)
    cleaner = VoxelCleaner(logger=logger)
    skel = SkeletonExtractor(logger=logger)
    writer = SWCWriter()

    swc_dir = Path(output_dir) / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (name, info)
        for name, info in registry.segments.items()
        if info.role in ('AXON', 'POST_SYN')
    ]
    logger.info(f"Phase 1: {len(targets)} segments to skeletonize (AXON + POST_SYN)")

    results: Dict[str, Tuple[nx.DiGraph, str]] = {}
    failures: List[FailureRecord] = []
    warnings: List[WarningRecord] = []
    review_queue_entries: List[dict] = []

    # Closing fraction threshold: warn if closing added >20% of original volume.
    # A large fraction means the closing radius is bridging more than surface gaps.
    # Flagged segments are written to review_queue.csv for human inspection.
    CLOSING_FRACTION_WARN = 0.20

    for name, info in targets:
        logger.info(f"--- {info.role} {name} (seg {info.seg_id}) ---")

        try:
            # 1A. Voxel extraction
            mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
                segment_id=info.seg_id,
                miplevel=miplevel,
                padding=padding,
            )

            if mask is None or voxel_size is None or bbox_min is None:
                log_failure(logger, "EXTRACTION", name, "Null return from extractor",
                            {'seg_id': info.seg_id, 'role': info.role})
                failures.append(FailureRecord(name, info.seg_id, info.role,
                                              "phase1", "extraction", "null_return"))
                continue

            original_voxels = int(np.sum(mask))
            if original_voxels == 0:
                log_failure(logger, "EXTRACTION", name, "Empty mask (0 filled voxels)",
                            {'seg_id': info.seg_id})
                failures.append(FailureRecord(name, info.seg_id, info.role,
                                              "phase1", "extraction", "empty_mask"))
                continue

            # 1B. Cleaning -- first pass to count components
            cleaned, stats = cleaner.clean_mask(
                    mask,
                    closing_radius_um=0.05,
                    keep_largest_only=True,
                    fill_holes=True,
                    smooth_iterations=0,
                    voxel_size_um=voxel_size
            )

            # Check how much volume closing added relative to original.
            # A high fraction means the radius is doing more than gap-bridging.
            closing_fraction = stats['closing_voxels_added'] / max(original_voxels, 1)
            if closing_fraction > CLOSING_FRACTION_WARN:
                msg = (
                    f"Closing added {closing_fraction*100:.1f}% of original volume "
                    f"({stats['closing_voxels_added']:,} vox). "
                    f"Closing radius may be too large or segment has large surface gaps. "
                    f"Run: diag_cleaning_param_sweep.py --segment {info.seg_id} --z-slab 20 --no-obj"
                )
                log_warning_box(logger, "CLEANING", name, msg)
                warnings.append(WarningRecord(name, "cleaning", msg, "medium"))
                review_queue_entries.append({
                    'segment_name':         name,
                    'segment_id':           info.seg_id,
                    'role':                 info.role,
                    'reason':               'high_closing_fraction',
                    'closing_radius_um':    0.05,
                    'original_voxels':      original_voxels,
                    'closing_voxels_added': stats['closing_voxels_added'],
                    'closing_fraction':     round(closing_fraction, 4),
                    'components_after_clean': stats['original_components'],
                    'recommended_action':   (
                        f"diag_cleaning_param_sweep.py --segment {info.seg_id} "
                        f"--z-slab 20 --no-obj"
                    ),
                })

            if stats['original_components'] > 1:
                logger.warning(
                    f"  {name}: {stats['original_components']} connected components "
                    f"(sizes: kept={stats['kept_components']}, removed={stats['removed_components']})"
                )

                # Re-clean keeping only the largest component
                cleaned, stats = cleaner.clean_mask(
                    cleaned.mask,
                    closing_radius_um=0.05,
                    keep_largest_only=True,
                    fill_holes=False,
                    smooth_iterations=0,
                    voxel_size_um=voxel_size
                )

            # Validate cleaning did not discard too much
            remaining = int(np.sum(cleaned.mask))
            removed_pct = (original_voxels - remaining) / original_voxels * 100

            if removed_pct > 70.0:
                log_failure(logger, "CLEANING", name,
                            f"Removed {removed_pct:.1f}% of voxels (threshold: 70%)",
                            {'original': original_voxels, 'remaining': remaining})
                failures.append(FailureRecord(name, info.seg_id, info.role,
                                              "phase1", "cleaning", "excessive_removal",
                                              {'removed_pct': f"{removed_pct:.1f}"}))
                continue

            if removed_pct > 40.0:
                log_warning_box(logger, "CLEANING", name,
                                f"Removed {removed_pct:.1f}% of voxels — skeleton quality may be poor",
                                {'original': original_voxels, 'remaining': remaining})
                warnings.append(WarningRecord(name, "cleaning",
                                              f"removed {removed_pct:.1f}%", "medium"))

            n_comp = stats.get('original_components', 1)
            if n_comp > 1:
                log_warning_box(logger, "CLEANING", name,
                                f"{n_comp} connected components found, kept largest")
                warnings.append(WarningRecord(name, "cleaning",
                                              f"{n_comp} components", "low"))
                # Only add to review queue if not already flagged for high closing fraction
                if not any(e['segment_name'] == name for e in review_queue_entries):
                    review_queue_entries.append({
                        'segment_name':         name,
                        'segment_id':           info.seg_id,
                        'role':                 info.role,
                        'reason':               'residual_fragmentation',
                        'closing_radius_um':    0.05,
                        'original_voxels':      original_voxels,
                        'closing_voxels_added': stats['closing_voxels_added'],
                        'closing_fraction':     round(
                            stats['closing_voxels_added'] / max(original_voxels, 1), 4
                        ),
                        'components_after_clean': n_comp,
                        'recommended_action':   (
                            f"diag_cleaning_param_sweep.py --segment {info.seg_id} "
                            f"--z-slab 20 --no-obj"
                        ),
                    })

            # 1C. Skeletonize voxels (includes compression, pruning, soma insertion)
            tree, skel_stats = skel.extract_skeleton(
                mask=cleaned.mask,
                voxel_size_um=voxel_size,
                bbox_min_vox=cleaned.bbox_min_vox,
                spur_length_um=spur_length_um,
            )

            if tree.number_of_nodes() == 0:
                log_failure(logger, "SKELETONIZATION", name,
                            "Empty tree (0 nodes) produced",
                            {'input_voxels': remaining})
                failures.append(FailureRecord(name, info.seg_id, info.role,
                                              "phase1", "skeletonization", "empty_tree",
                                              {'input_voxels': remaining}))
                continue

            # Tag provenance (coord_frame already set by skeletonization)
            tree.graph['voxel_convention'] = VOXEL_CENTER
            tree.graph['bbox_min_vox']     = tuple(int(x) for x in bbox_min)
            tree.graph['voxel_size_um']    = tuple(float(x) for x in voxel_size)

            # 1D. Write SWC
            swc_path = str(swc_dir / f"cell_{name}.swc")
            writer.write_swc(
                tree=tree,
                output_path=swc_path,
                metadata={
                    'segment_id': info.seg_id,
                    'segment_name': name,
                    'role': info.role,
                    'miplevel': miplevel,
                    'bbox_min_vox': str(bbox_min),      # (minx, miny, minz) in voxels
                    'voxel_size_um': str(voxel_size),   # (sx, sy, sz) in µm
                    'coord_frame': COORD_FRAME_PHYSICAL,
                    'compressed_nodes': skel_stats.get('compressed_nodes'),
                    'spurs_pruned': skel_stats.get('branches_pruned'),
                    'total_length_um': f"{skel_stats.get('total_length_um', 0):.2f}",
                },
            )

            results[name] = (tree, swc_path)
            logger.info(f"  -> {swc_path}  ({tree.number_of_nodes()} nodes)")

        except Exception as e:
            log_failure(logger, "UNKNOWN", name, f"Unexpected exception: {e}",
                        {'traceback': traceback.format_exc()})
            failures.append(FailureRecord(name, info.seg_id, info.role,
                                          "phase1", "exception", str(e)))
            continue

    log_phase_summary(logger, "Phase 1", len(targets), len(results), failures, warnings)
    _write_review_queue(review_queue_entries, output_dir, logger)
    return results, failures, warnings

#  Phase 2

def run_phase2(
    registry: SegmentRegistry,
    vast: VASTControlClass,
    logger: logging.Logger,
    output_dir: str = "./vast_export",
    miplevel: int = 0,
) -> CentroidTable:
    """
    Phase 2: Extract centroid positions for all BOUTON, SYNAPSE, and CONTACT
    segments.

    Centroids are computed from the full voxel mask at MIP level *miplevel*
    (default 0 = full resolution, appropriate for compact annotation objects).
    On extraction failure the bounding-box centre is used, with the VAST
    anchor point as the final fallback.

    Args:
        registry   : SegmentRegistry from Phase 0.
        vast       : Connected VASTControlClass instance.
        logger     : Shared pipeline logger.
        output_dir : Base output directory; centroids.csv is written here.
        miplevel   : MIP level for voxel extraction (0 = full resolution).

    Returns:
        CentroidTable -- in-memory table, also written to
        <output_dir>/centroids.csv for Phase 3 consumption.
    """
    surface_extractor = SegmentSurfaceExtractor(vast, output_dir, logger=logger)
    centroid_extractor = CentroidExtractor(surface_extractor, logger=logger)

    table = centroid_extractor.extract_centroids(
        registry=registry,
        miplevel=miplevel
    )

    csv_path = str(Path(output_dir) / "centroids.csv")
    table.write_csv(csv_path)

    logger.info(f"Phase 2: centroids written -> {csv_path} ({len(table)} entries)")

    # Summary by role
    for role in ('BOUTON', 'SYNAPSE', 'CONTACT'):
        entries = table.by_role(role)
        if entries:
            methods: Dict[str, int] = {}
            for e in entries:
                methods[e.method] = methods.get(e.method, 0) + 1
            method_str = ", ".join(f"{m}={n}" for m, n in sorted(methods.items()))
            logger.info(f"  {role}: {len(entries)} entries ({method_str})")

    return table

#  Phase 3

def run_phase3(
    centroid_table: CentroidTable,
    trees: Dict[str, Tuple[nx.DiGraph, str]],
    registry: SegmentRegistry,
    logger: logging.Logger,
    output_dir: str = "./vast_export",
    ok_distance_um: float = 2.0,
    warn_distance_um: float = 5.0,
) -> CableMappingTable:
    """
    Phase 3: Map BOUTON / SYNAPSE / CONTACT centroids onto SWC cable trees.

    For each centroid, finds the nearest point on the relevant skeleton
    and records the edge + arc-fraction position for use in Phase 4.

    Args:
        centroid_table   : Output of Phase 2.
        trees            : Output of Phase 1 -- dict: cell_name -> (tree, swc_path).
        registry         : SegmentRegistry from Phase 0.
        logger           : Shared pipeline logger.
        output_dir       : Base output directory; cable_mappings.csv written here.
        ok_distance_um   : Distance threshold for OK mapping quality.
        warn_distance_um : Warn when nearest cable point exceeds this distance.

    Returns:
        CableMappingTable -- in-memory table, also written to
        <output_dir>/cable_mappings.csv for Phase 4 consumption.
    """
    mapper = CentroidMapper(logger=logger)

    mapping_table = mapper.map_centroids(
        centroid_table=centroid_table,
        trees=trees,
        registry=registry,
        ok_distance_um=ok_distance_um,
        warn_distance_um=warn_distance_um,
    )

    csv_path = str(Path(output_dir) / "cable_mappings.csv")
    mapping_table.write_csv(csv_path)
    logger.info(f"Phase 3: mappings written -> {csv_path}  ({len(mapping_table)} entries)")

    # Summary by role
    for role in ('BOUTON', 'SYNAPSE', 'CONTACT'):
        entries = mapping_table.by_role(role)
        if entries:
            distances = [e.distance_um for e in entries]
            logger.info(
                f"  {role}: {len(entries)} mapped  "
                f"dist min={min(distances):.2f} median={np.median(distances):.2f} "
                f"max={max(distances):.2f} µm"
            )

    return mapping_table

#  Phase 4

def run_phase4(
    registry: SegmentRegistry,
    trees: Dict[str, Tuple[nx.DiGraph, str]],
    mapping_table: CableMappingTable,
    logger: logging.Logger,
    output_dir: str = "./vast_export",
    syn_mechanism: str = "expsyn",
) -> ConnectivityOutput:
    """
    Phase 4: Build connectivity.csv and arbor_recipe.json.

    Consumes:
      - Phase 0 registry  (connectivity / contact wiring tables)
      - Phase 1 trees     (SWC file paths, skeleton graphs)
      - Phase 3 mappings  (cable locations for each centroid)

    Produces:
      <output_dir>/connectivity.csv   — full edge list for analysis
      <output_dir>/arbor_recipe.json  — Arbor-ready network description

    Args:
        registry      : SegmentRegistry from Phase 0.
        trees         : Phase 1 output — cell_name -> (tree, swc_path).
        mapping_table : Phase 3 output — CableMappingTable.
        logger        : Shared pipeline logger.
        output_dir    : Base output directory.
        syn_mechanism : Arbor synapse mechanism name (default: 'expsyn').

    Returns:
        ConnectivityOutput — in-memory connectivity table.
    """
    builder = ConnectivityBuilder(logger=logger)
    connectivity, csv_path, recipe_path = builder.build(
        registry=registry,
        trees=trees,
        mapping_table=mapping_table,
        output_dir=output_dir,
        syn_mechanism=syn_mechanism,
    )

    synapses = [r for r in connectivity.rows if r.connection_type == "synapse"]
    contacts = [r for r in connectivity.rows if r.connection_type == "contact"]
    logger.info(
        f"Phase 4 complete: {len(synapses)} synapses, {len(contacts)} contacts  "
        f"-> {csv_path}  {recipe_path}"
    )
    return connectivity


# Main entry point
def main():
    output_dir = "./vast_export"

    logger = setup_pipeline_logger(output_dir)
    logger.info("Pipeline starting")

    # Phase 0: Classify segments
    classifier = SegmentClassifier()
    registry = classifier.classify_segments()

    if not registry.segments:
        logger.error("Phase 1 aborted: registry is empty after classification")
        return

    # Phase 1: Skeletonize AXON and POST_SYN
    trees, p1_failures, p1_warnings = run_phase1(
        registry=registry,
        vast=classifier.vast,
        logger=logger,
        output_dir=output_dir,
        miplevel=1,
        spur_length_um=2.0,
    )

    # Phase 2: extract BOUTON / SYNAPSE / CONTACT centroids
    centroids = run_phase2(
        registry=registry,
        vast=classifier.vast,
        logger=logger,
        output_dir=output_dir,
        miplevel=1,  # MIP 0 for small markers -- MIP 1 may reduce to too few voxels
    )

    # Phase 3 boundary check
    if not centroids.entries:
        logger.error("Phase 3 aborted: centroid table is empty")
        return
    if not trees:
        logger.error("Phase 3 aborted: no trees produced by Phase 1 (all segments failed)")
        return

    # Warn about centroids whose parent axon failed Phase 1 (expected for bad segmentation)
    missing_parents = set()
    for entry in centroids.entries:
        m = re.match(r'^(A\d+)', entry.name)
        if m:
            axon_name = m.group(1)
            if axon_name not in trees:
                missing_parents.add(axon_name)
    if missing_parents:
        logger.warning(
            f"Phase 3: {len(missing_parents)} parent axon(s) absent from Phase 1 output "
            f"— centroids referencing them will be skipped: {sorted(missing_parents)}"
        )

    # Phase 3: Map centroids onto skeleton cable
    mappings = run_phase3(
        centroid_table=centroids,
        trees=trees,
        registry=registry,
        logger=logger,
        output_dir=output_dir,
        warn_distance_um=5.0,
    )

    # Phase 4 boundary check
    if not mappings.entries:
        logger.warning("Phase 4: mapping table is empty — connectivity output will be empty")

    connectivity = run_phase4(
        registry=registry,
        trees=trees,
        mapping_table=mappings,
        logger=logger,
        output_dir=output_dir,
        syn_mechanism="expsyn",
    )

    logger.info(
        f"Pipeline complete. "
        f"Outputs in {output_dir}/  "
        f"({len(trees)} SWC files, "
        f"{len(connectivity)} connectivity rows)"
    )


if __name__ == "__main__":
    main()
