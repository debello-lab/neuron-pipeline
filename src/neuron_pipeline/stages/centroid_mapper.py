"""
Centroid -> SWC Cable Mapping
Neural Reconstruction Pipeline -- Phase 3

For each BOUTON, SYNAPSE, and CONTACT centroid (from Phase 2) this module
finds the nearest point on the relevant skeleton cable (from Phase 1) and
records the location as:

    (cell_name, edge_u, edge_v, arc_fraction, distance_um)

where
    cell_name    - name of the AXON or POST_SYN cell the centroid belongs to
    edge_u/v     - the two endpoint node IDs of the nearest skeleton edge
    arc_fraction - position along that edge, 0.0 = u-end, 1.0 = v-end
    distance_um  - Euclidean distance from centroid to the nearest cable point

The mapping is used in Phase 4 to build the Arbor connectivity recipe.

Design
------
The skeleton tree produced by Phase 1 stores geometry at two levels:
  - Structural nodes (junctions + endpoints) with node attribute ``pos``
  - Intermediate samples stored on edges as ``edge['points']``

To find the nearest cable point we:
  1. Build a KDTree from ALL sample positions (nodes + edge intermediates).
  2. Query each centroid against the tree (O(log N) per centroid).
  3. Resolve the hit back to its parent edge and arc-fraction.
"""

import csv
import logging
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
from scipy.spatial import KDTree

from neuron_pipeline.stages.centroid_extraction import CentroidEntry, CentroidTable
from neuron_pipeline.stages.segment_classifier import SegmentRegistry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CableMappingEntry:
    # Centroid identity
    centroid_name: str          # e.g. "A1B2"  (BOUTON)
    centroid_role: str          # BOUTON | SYNAPSE | CONTACT
    cx_um: float
    cy_um: float
    cz_um: float

    # Which skeleton this was mapped onto
    cell_name: str              # e.g. "A1"  (AXON or POST_SYN)
    cell_role: str              # AXON | POST_SYN

    # Location on the cable
    edge_u: int                 # upstream node ID of nearest edge
    edge_v: int                 # downstream node ID of nearest edge
    arc_fraction: float         # 0.0 = at node u, 1.0 = at node v
    nearest_x_um: float         # physical position of nearest cable point
    nearest_y_um: float
    nearest_z_um: float
    distance_um: float          # Euclidean distance centroid -> cable

    # Coordinate frame and QC
    coord_frame: str = 'physical_um_xyz_center'
    # Distance QC tier: 'ok' (<ok_distance_um), 'warn', 'suspicious' (>=warn_distance_um)
    qc_distance_flag: str = 'ok'
    # SWC row IDs stamped by SWCWriter onto the tree nodes in Phase 1.
    # None when write_swc() was not called before Phase 3 (e.g. unit tests).
    swc_node_u: Optional[int] = None
    swc_node_v: Optional[int] = None


@dataclass
class CableMappingTable:
    entries: List[CableMappingEntry] = field(default_factory=list)

    def add(self, entry: CableMappingEntry) -> None:
        self.entries.append(entry)

    def by_cell(self, cell_name: str) -> List[CableMappingEntry]:
        return [e for e in self.entries if e.cell_name == cell_name]

    def by_role(self, role: str) -> List[CableMappingEntry]:
        return [e for e in self.entries if e.centroid_role == role]

    def __len__(self) -> int:
        return len(self.entries)

    def write_csv(self, path: str) -> str:
        """Write mappings to CSV. Returns path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            'centroid_name', 'centroid_role', 'cx_um', 'cy_um', 'cz_um',
            'cell_name', 'cell_role',
            'edge_u', 'edge_v', 'arc_fraction',
            'nearest_x_um', 'nearest_y_um', 'nearest_z_um',
            'distance_um',
            'coord_frame', 'qc_distance_flag', 'swc_node_u', 'swc_node_v',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                row = asdict(e)
                for key in ('cx_um', 'cy_um', 'cz_um',
                            'nearest_x_um', 'nearest_y_um', 'nearest_z_um',
                            'distance_um'):
                    row[key] = f"{row[key]:.4f}"
                row['arc_fraction'] = f"{row['arc_fraction']:.6f}"
                writer.writerow(row)
        return path


# ---------------------------------------------------------------------------
# Per-tree index
# ---------------------------------------------------------------------------

class _TreeIndex:
    """
    Spatial index over all sample points of a single skeleton tree.

    Stores every node position AND every intermediate edge point so that
    nearest-cable queries resolve to a specific (edge_u, edge_v, arc_fraction).
    """

    def __init__(self, tree: nx.DiGraph) -> None:
        self._tree = tree
        self._samples: List[Tuple[float, float, float]] = []
        # For each sample: which edge does it belong to, and what fraction?
        self._edge_u:   List[int]   = []
        self._edge_v:   List[int]   = []
        self._arc_frac: List[float] = []
        self._build(tree)
        pts = np.array(self._samples, dtype=np.float64)
        self._kdtree = KDTree(pts) if len(pts) > 0 else None

    # ------------------------------------------------------------------

    def _build(self, tree: nx.DiGraph) -> None:
        """Populate sample list from node positions and edge intermediate points."""
        for u, v, data in tree.edges(data=True):
            u_pos = np.array(tree.nodes[u]['pos'], dtype=np.float64)
            v_pos = np.array(tree.nodes[v]['pos'], dtype=np.float64)
            edge_len = float(data.get('length', 0.0))
            mid_pts  = data.get('points', [])
            mid_rads = data.get('radii',  [])   # noqa: kept for completeness

            # All positions along this edge in order: u -> intermediates -> v
            ordered_positions = [u_pos] + [np.array(p) for p in mid_pts] + [v_pos]

            # Compute cumulative arc lengths
            cum = [0.0]
            for i in range(1, len(ordered_positions)):
                seg = float(np.linalg.norm(ordered_positions[i] - ordered_positions[i - 1]))
                cum.append(cum[-1] + seg)
            total = cum[-1] if cum[-1] > 0 else edge_len  # fall back to stored length

            # Add u node (fraction 0)
            self._add(u_pos, u, v, 0.0)

            # Add intermediate samples
            for k, pt in enumerate(mid_pts):
                frac = (cum[k + 1] / total) if total > 0 else (k + 1) / (len(mid_pts) + 1)
                self._add(np.array(pt), u, v, float(frac))

            # Add v node (fraction 1.0) -- will appear again as u of next edge,
            # that's fine; duplicate positions in kd-tree are harmless
            self._add(v_pos, u, v, 1.0)

    def _add(self, pos: np.ndarray, u: int, v: int, frac: float) -> None:
        self._samples.append(tuple(pos))
        self._edge_u.append(u)
        self._edge_v.append(v)
        self._arc_frac.append(frac)

    # ------------------------------------------------------------------

    def query(
        self, point: np.ndarray
    ) -> Optional[Tuple[int, int, float, np.ndarray, float]]:
        """
        Find the nearest cable point to *point*.

        Returns
        -------
        (edge_u, edge_v, arc_fraction, nearest_pos, distance_um)
        or None if the tree is empty.
        """
        if self._kdtree is None:
            return None
        dist, idx = self._kdtree.query(point)
        u    = self._edge_u[idx]
        v    = self._edge_v[idx]
        frac = self._arc_frac[idx]
        nearest_pos = np.array(self._samples[idx])
        return u, v, frac, nearest_pos, float(dist)


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

# Which skeleton role does each centroid/segment role map onto?
# A BOUTON/SYNAPSE/CONTACT belongs to an axon; the companion POST_SYN
# dendrite is the other side of the synapse.
_CENTROID_TO_SKELETON: Dict[str, Tuple[str, ...]] = {
    'BOUTON':  ('AXON',),
    'SYNAPSE': ('AXON',),
    'CONTACT': ('AXON',),
    'POST_SYN': ('POST_SYN',),
}


class CentroidMapper:
    """
    Map Phase 2 centroids onto Phase 1 skeleton trees.

    For each centroid the relevant parent cell name is derived from the
    segment naming convention (A1B2 -> parent AXON is A1, etc.) using
    the SegmentRegistry built in Phase 0.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def map_centroids(
        self,
        centroid_table: CentroidTable,
        trees: Dict[str, Tuple[nx.DiGraph, str]],
        registry: SegmentRegistry,
        ok_distance_um: float = 2.0,
        warn_distance_um: float = 5.0,
        extra_entries: Optional[List] = None,
    ) -> CableMappingTable:
        """
        Map all centroids in *centroid_table* to their nearest skeleton edge.

        Two passes are performed:

        Pass 1 -- Primary mapping (BOUTON, SYNAPSE, CONTACT -> AXON skeleton):
            Each centroid is projected onto the skeleton of its parent axon cell,
            derived from the segment naming convention via _resolve_parent_cell().

        Pass 2 -- Post-side mapping (SYNAPSE -> POST_SYN skeleton):
            Each SYNAPSE centroid is also projected onto the POST_SYN skeleton
            derived from the synapse name (A1B1P1S1 -> POST_SYN A1B1P1).
            The SYNAPSE coordinate is used as the contact point, not the POST_SYN
            bounding-box centre. This produces the anatomically correct post-side
            cable location for the Arbor recipe.

        Args:
            centroid_table   : Output of Phase 2.
            trees            : Output of Phase 1 -- dict: cell_name -> (tree, swc_path).
            registry         : SegmentRegistry from Phase 0.
            ok_distance_um   : Distance threshold below which a mapping is flagged 'ok'.
            warn_distance_um : Distance threshold above which a mapping is flagged 'suspicious'.
            extra_entries    : Optional additional CentroidEntry objects to include in Pass 1.

        Returns:
            CableMappingTable with entries from both passes.
        """
        # Validate coordinate frames before any spatial work.
        # A mismatch (e.g. voxel vs physical space) would corrupt all outputs silently.
        self._assert_frame_consistency(centroid_table, trees)

        # Build one spatial index per skeleton tree.
        self.logger.info(f"Building spatial indices for {len(trees)} skeleton trees...")
        indices: Dict[str, _TreeIndex] = {
            name: _TreeIndex(tree)
            for name, (tree, _) in trees.items()
        }
        self._log_skeleton_bounds(trees)

        mapping_table = CableMappingTable()

        # ------------------------------------------------------------------
        # Pass 1: map BOUTON / SYNAPSE / CONTACT centroids onto the AXON skeleton.
        # ------------------------------------------------------------------
        all_entries = list(centroid_table.entries) + (extra_entries or [])
        n_ok = n_no_tree = n_no_cell = 0

        for entry in all_entries:
            result = self._map_one(entry, indices, registry, ok_distance_um, warn_distance_um)
            if result == 'no_cell':
                n_no_cell += 1
            elif result == 'no_tree':
                n_no_tree += 1
            else:
                mapping_table.add(result)
                n_ok += 1

        self.logger.info(
            f"Phase 3 Pass 1: {n_ok} mapped, "
            f"{n_no_tree} skipped (no skeleton), "
            f"{n_no_cell} skipped (no parent cell in registry)"
        )

        # ------------------------------------------------------------------
        # Pass 2: map SYNAPSE centroids onto the POST_SYN skeleton.
        # Each SYNAPSE entry produces a second mapping keyed by the POST_SYN
        # cell name (e.g. A1B1P1S1 -> mapped onto A1B1P1 skeleton).
        # _resolve_parent_cell handles POST_SYN role by returning (name, POST_SYN).
        # ------------------------------------------------------------------
        n_post_mapped = n_post_no_tree = 0

        for entry in centroid_table.entries:
            if entry.role != 'SYNAPSE':
                continue

            post_name = self._strip_synapse_suffix(entry.name)
            self.logger.debug(
                f"Post-side pass: {entry.name} -> post_name={post_name}, "
                f"in_indices={post_name in indices if post_name else False}"
            )

            if post_name is None or post_name not in indices:
                n_post_no_tree += 1
                continue

            proxy = CentroidEntry(
                name=post_name,
                seg_id=registry.segments[post_name].seg_id if post_name in registry.segments else -1,
                role='POST_SYN',
                cx_um=entry.cx_um,
                cy_um=entry.cy_um,
                cz_um=entry.cz_um,
                method='synapse_proxy',
            )
            result = self._map_one(proxy, indices, registry, ok_distance_um, warn_distance_um)
            if isinstance(result, CableMappingEntry):
                mapping_table.add(result)
                n_post_mapped += 1
            else:
                n_post_no_tree += 1

        self.logger.info(
            f"Phase 3 Pass 2: {n_post_mapped} SYNAPSE centroids mapped onto "
            f"POST_SYN skeletons, {n_post_no_tree} skipped (no POST_SYN skeleton)"
        )

        # QC summary across all entries from both passes.
        qc_counts: Dict[str, int] = {'ok': 0, 'warn': 0, 'suspicious': 0}
        for e in mapping_table.entries:
            qc_counts[e.qc_distance_flag] = qc_counts.get(e.qc_distance_flag, 0) + 1
        self.logger.info(
            f"Phase 3 QC distance: ok={qc_counts['ok']}, "
            f"warn={qc_counts['warn']}, suspicious={qc_counts['suspicious']}"
        )

        return mapping_table

    # ------------------------------------------------------------------
    # Per-centroid logic
    # ------------------------------------------------------------------

    def _map_one(
        self,
        entry: CentroidEntry,
        indices: Dict[str, _TreeIndex],
        registry: SegmentRegistry,
        ok_distance_um: float,
        warn_distance_um: float,
    ):
        """
        Map a single centroid. Returns a CableMappingEntry or a string error code.
        """
        # Derive the parent cell name from the segment registry
        cell_name, cell_role = self._resolve_parent_cell(entry, registry)
        if cell_name is None or cell_role is None:
            self.logger.debug(
                f"  {entry.name}: cannot resolve parent cell -- skipped"
            )
            return 'no_cell'

        idx = indices.get(cell_name)
        if idx is None:
            self.logger.debug(
                f"  {entry.name}: no skeleton for {cell_name} -- skipped"
            )
            return 'no_tree'

        # Both centroid and skeleton are in physical_um_xyz -- validated by
        # _assert_frame_consistency() before this method is called.
        point = np.array([entry.cx_um, entry.cy_um, entry.cz_um], dtype=np.float64)
        result = idx.query(point)
        if result is None:
            return 'no_tree'

        u, v, arc_frac, nearest_pos, dist = result

        # Three-tier distance QC
        if dist < ok_distance_um:
            qc_flag = 'ok'
        elif dist < warn_distance_um:
            qc_flag = 'warn'
            self.logger.warning(
                f"  {entry.name}: {dist:.2f} µm from cable "
                f"(>ok threshold {ok_distance_um} µm)"
            )
        else:
            qc_flag = 'suspicious'
            self.logger.warning(
                f"  {entry.name}: {dist:.2f} µm from cable -- SUSPICIOUS "
                f"(>{warn_distance_um} µm, check annotation or skeleton)"
            )

        # Look up SWC row IDs stamped by SWCWriter in Phase 1.
        # _TreeIndex stores a reference to the tree; swc_id is None when
        # write_swc() was not called before Phase 3 (e.g. in unit tests).
        swc_node_u = idx._tree.nodes[u].get('swc_id')
        swc_node_v = idx._tree.nodes[v].get('swc_id')

        return CableMappingEntry(
            centroid_name=entry.name,
            centroid_role=entry.role,
            cx_um=entry.cx_um,
            cy_um=entry.cy_um,
            cz_um=entry.cz_um,
            cell_name=cell_name,
            cell_role=cell_role,
            edge_u=int(u),
            edge_v=int(v),
            arc_fraction=float(arc_frac),
            nearest_x_um=float(nearest_pos[0]),
            nearest_y_um=float(nearest_pos[1]),
            nearest_z_um=float(nearest_pos[2]),
            distance_um=dist,
            qc_distance_flag=qc_flag,
            swc_node_u=swc_node_u,
            swc_node_v=swc_node_v,
        )

    # ------------------------------------------------------------------
    # Parent-cell resolution
    # ------------------------------------------------------------------

    def _resolve_parent_cell(
        self,
        entry: CentroidEntry,
        registry: SegmentRegistry,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Derive the parent skeleton cell name from the centroid's segment name.

        Naming convention (from segment_classifier.py patterns):
            BOUTON   A1B2      -> parent AXON  A1
            SYNAPSE  A1B2P1S1  -> parent AXON  A1  (mapped onto the axon)
            CONTACT  A1B2X1    -> parent AXON  A1

        POST_SYN segments (A1B2P1) are skeletonized in Phase 1 as their own
        cell.  Synapses are associated with the AXON side here; the Phase 4
        connectivity builder will link both sides.

        Returns (cell_name, cell_role) or (None, None) if not found.
        """
        seg_info = registry.segments.get(entry.name)
        if seg_info is None:
            return None, None

        role = seg_info.role

        if role == 'BOUTON':
            # A1B2 -> axon = A1
            axon_name = self._axon_prefix(entry.name)
            if axon_name and axon_name in registry.segments:
                return axon_name, 'AXON'

        elif role == 'SYNAPSE':
            # A1B2P1S1 -> axon = A1
            axon_name = self._axon_prefix(entry.name)
            if axon_name and axon_name in registry.segments:
                return axon_name, 'AXON'

        elif role == 'CONTACT':
            # A1B2X1 -> axon = A1
            axon_name = self._axon_prefix(entry.name)
            if axon_name and axon_name in registry.segments:
                return axon_name, 'AXON'
        elif role == 'POST_SYN':
            if entry.name in registry.segments:
                return entry.name, 'POST_SYN'

        return None, None

    @staticmethod
    def _axon_prefix(name: str) -> Optional[str]:
        """Extract the A<n> prefix from any segment name."""
        m = re.match(r'^(A\d+)', name)
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Coordinate frame validation
    # ------------------------------------------------------------------

    def _assert_frame_consistency(
        self,
        centroid_table: 'CentroidTable',
        trees: Dict[str, Tuple[nx.DiGraph, str]],
    ) -> None:
        """
        Assert that all skeleton trees and all centroids share the same
        coordinate frame tag. Raises ValueError on any mismatch so the
        pipeline fails loudly rather than producing silently wrong output.

        The coord_frame tag is set by:
          - Phase 1 (graph_to_tree): tree.graph['coord_frame']
          - Phase 2 (CentroidEntry): entry.coord_frame  (default 'physical_um_xyz')
        """
        expected = 'physical_um_xyz_center'

        for cell_name, (tree, _) in trees.items():
            frame = tree.graph.get('coord_frame')
            if frame is None:
                self.logger.warning(
                    f"Skeleton '{cell_name}' has no coord_frame attribute -- "
                    f"was it produced by an older version of graph_to_tree? "
                    f"Assuming '{expected}'."
                )
            elif frame != expected:
                raise ValueError(
                    f"Skeleton '{cell_name}' coord_frame='{frame}' does not match "
                    f"expected '{expected}'. Coordinate space mismatch would corrupt "
                    f"all synapse placement output."
                )

        for entry in centroid_table.entries:
            frame = getattr(entry, 'coord_frame', None)
            if frame is None:
                self.logger.warning(
                    f"Centroid '{entry.name}' has no coord_frame -- assuming '{expected}'."
                )
            elif frame != expected:
                raise ValueError(
                    f"Centroid '{entry.name}' coord_frame='{frame}' does not match "
                    f"expected '{expected}'. Coordinate space mismatch would corrupt "
                    f"all synapse placement output."
                )

    def _log_skeleton_bounds(
        self,
        trees: Dict[str, Tuple[nx.DiGraph, str]],
    ) -> None:
        """
        Log the physical bounding box of each skeleton tree in µm.

        This gives a visible record of the spatial extent of each cell's
        skeleton, which makes it easy to spot-check whether centroids (logged
        separately via warn_distance_um) fall in a plausible region.
        """
        for cell_name, (tree, _) in trees.items():
            positions = [
                tree.nodes[n]['pos']
                for n in tree.nodes()
                if 'pos' in tree.nodes[n]
            ]
            if not positions:
                self.logger.info(f"  Skeleton '{cell_name}': no node positions found")
                continue
            pts = np.array(positions, dtype=np.float64)
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            self.logger.info(
                f"  Skeleton '{cell_name}' bounds (µm): "
                f"X [{lo[0]:.1f}, {hi[0]:.1f}]  "
                f"Y [{lo[1]:.1f}, {hi[1]:.1f}]  "
                f"Z [{lo[2]:.1f}, {hi[2]:.1f}]"
            )

    @staticmethod
    def _strip_synapse_suffix(name: str) -> Optional[str]:
        """A1B1P1S1 -> A1B1P1, returns None if name does not match."""
        m = re.match(r'^(A\d+B\d+P\d+)S\d+$', name)
        return m.group(1) if m else None