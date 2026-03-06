"""
Voxel Mask to SWC Skeleton Conversion
Neural Reconstruction Pipeline

This module converts cleaned binary voxel masks into SWC morphology files:
- 3D skeletonization (medial axis thinning)
- Radius estimation via Euclidean distance transform
- Topologically-correct graph compression
- Spur pruning and tree construction
- SWC export with Arbor-compatible soma

Author: Nicolas Randazzo
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import networkx as nx
import logging


class SkeletonExtractor:
    """
    Convert binary voxel masks to SWC skeleton format.

    Pipeline:
    1. Skeletonize voxel mask (medial axis thinning via Lee algorithm)
    2. Compute radii via Euclidean distance transform
    3. Build voxel-resolution graph (26-connectivity)
    4. Compress graph using topological junction detection
    5. Prune short leaf branches (spurs)
    6. Convert to rooted directed tree
    7. Insert synthetic 2-sample soma for Arbor compatibility
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def extract_skeleton(
        self,
        mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float],
        bbox_min_vox: Tuple[int, int, int] = (0, 0, 0),
        spur_length_um: float = 2.0,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Extract a compressed skeleton tree from a binary voxel mask.

        Args:
            mask: Boolean 3D array (Z, Y, X).
            voxel_size_um: Voxel dimensions in microns (sx, sy, sz).
            bbox_min_vox: Origin coordinates in voxel space (minx, miny, minz).
            spur_length_um: Prune leaf branches shorter than this (um).

        Returns:
            (tree, stats)
            tree  -- Directed compressed skeleton (NetworkX DiGraph).
                    Node attrs : pos (um tuple), radius (um float), swc_type (int).
                    Edge attrs : length (um), points (list of xyz tuples), radii (list).
            stats -- Extraction statistics dictionary.
        """
        self.logger.info("Starting skeleton extraction...")

        stats = {
            'input_voxel_count': int(np.sum(mask)),
            'skeleton_voxel_count': 0,
            'skeleton_points': 0,
            'compressed_nodes': 0,
            'compressed_edges': 0,
            'branches_pruned': 0,
            'total_length_um': 0.0,
        }

        if not np.any(mask):
            self.logger.error("Input mask is empty")
            return nx.DiGraph(), stats

        # ------------------------------------------------------------------
        # Step 1: Skeletonize (Lee 1994 algorithm -- guaranteed 1-voxel-wide)
        # ------------------------------------------------------------------
        self.logger.info("Computing skeleton...")
        skeleton_mask = skeletonize(mask.astype(bool), method='lee') > 0
        skeleton_voxel_count = int(np.sum(skeleton_mask))
        stats['skeleton_voxel_count'] = skeleton_voxel_count
        self.logger.info(
            f"Skeleton: {skeleton_voxel_count:,} voxels "
            f"({100.0 * skeleton_voxel_count / stats['input_voxel_count']:.2f}% of input)"
        )

        # ------------------------------------------------------------------
        # Step 2: Distance transform for radii
        # ------------------------------------------------------------------
        self.logger.info("Computing distance transform...")
        sz, sy, sx = voxel_size_um
        dist_um = distance_transform_edt(mask.astype(bool), sampling=(sz, sy, sx))

        # ------------------------------------------------------------------
        # Step 3: Extract voxel coordinates and radii
        # ------------------------------------------------------------------
        skel_coords = np.argwhere(skeleton_mask)   # (N, 3) in ZYX order
        if len(skel_coords) == 0:
            self.logger.error("Skeleton is empty after skeletonize()")
            return nx.DiGraph(), stats

        # Minimum radius = half the shortest voxel dimension.
        # This prevents sub-voxel radii (e.g. 5 nm) at skeleton surface voxels.
        radius_floor = max(0.005, min(voxel_size_um) / 2.0)
        radii_um = dist_um[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]]
        radii_um = np.maximum(radii_um, radius_floor)

        stats['skeleton_points'] = len(skel_coords)
        self.logger.info(
            f"Radii: {radii_um.min():.3f} - {radii_um.max():.3f} um "
            f"(median {np.median(radii_um):.3f} um, floor {radius_floor:.3f} um)"
        )

        # ------------------------------------------------------------------
        # Step 4: Build voxel-resolution graph
        # ------------------------------------------------------------------
        G = self.skeleton_to_graph(skeleton_mask, voxel_size_um, bbox_min_vox)

        # ------------------------------------------------------------------
        # Step 5: Topologically-correct graph compression
        #
        # Use topological junction detection instead of raw graph degree. 
        # In a 26-connected voxel graph a diagonal path voxel has degree 3 
        # (back + forward + diagonal) even though it is topologically just 
        # a pass-through. A node is a TRUE junction only if removing it 
        # disconnects its neighbourhood into 2+ components.
        # ------------------------------------------------------------------
        G_compressed = self.compress_graph(G, radii_um)
        stats['compressed_nodes'] = G_compressed.number_of_nodes()
        stats['compressed_edges'] = G_compressed.number_of_edges()

        # ------------------------------------------------------------------
        # Step 6: Spur pruning
        # ------------------------------------------------------------------
        G_pruned, spurs_removed = self.prune_spurs(G_compressed, spur_length_um)
        stats['branches_pruned'] = spurs_removed

        # ------------------------------------------------------------------
        # Step 7: Convert to rooted directed tree
        # ------------------------------------------------------------------
        tree = self.graph_to_tree(G_pruned)

        # ------------------------------------------------------------------
        # Step 8: Insert synthetic 2-sample soma for Arbor compatibility
        # ------------------------------------------------------------------
        tree = self.insert_synthetic_soma(tree)

        stats['total_length_um'] = sum(
            d.get('length', 0.0) for _, _, d in tree.edges(data=True)
        )
        self.logger.info(
            f"Final tree: {tree.number_of_nodes()} nodes, "
            f"{stats['total_length_um']:.2f} um total cable"
        )
        return tree, stats

    # -------------------------------------------------------------------------
    # Step 4: Voxel-resolution graph
    # -------------------------------------------------------------------------

    def skeleton_to_graph(
        self,
        skeleton_mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float],
        bbox_min_vox: Tuple[int, int, int] = (0, 0, 0),
    ) -> nx.Graph:
        """
        Build a voxel-resolution undirected graph from the skeleton mask.

        Node attrs : pos (um tuple), voxel_pos (ZYX ndarray).
        Edge attrs : length (um).
        """
        self.logger.info("Building voxel-resolution skeleton graph...")
        skel_coords = np.argwhere(skeleton_mask)
        if len(skel_coords) == 0:
            return nx.Graph()

        kd = KDTree(skel_coords)
        G = nx.Graph()
        sz, sy, sx = voxel_size_um
        minx, miny, minz = bbox_min_vox

        for i, coord in enumerate(skel_coords):
            z, y, x = coord
            G.add_node(i, pos=((x + minx) * sx, (y + miny) * sy, (z + minz) * sz), voxel_pos=coord)

        for i, coord in enumerate(skel_coords):
            for j in kd.query_ball_point(coord, r=1.8):   # sqrt(3) + margin = 26-conn
                if i < j:
                    pi = G.nodes[i]['pos']
                    pj = G.nodes[j]['pos']
                    G.add_edge(i, j, length=float(np.linalg.norm(
                        np.array(pi) - np.array(pj)
                    )))

        for n in G.nodes():
            G.nodes[n]['degree'] = G.degree(n)

        self.logger.info(
            f"Voxel graph: {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )
        return G

    # -------------------------------------------------------------------------
    # Step 5: Topologically-correct compression
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_topological_junction(G: nx.Graph, node: int) -> bool:
        """
        Return True if node is a topologically genuine junction or endpoint.

        A node is structural if and only if removing it splits its immediate
        neighbourhood into 2+ connected components (true branch point) or it
        has <= 1 neighbour (endpoint / isolated).

        This correctly handles the 26-connectivity artefact where a straight
        diagonal path voxel has graph-degree 3 but is NOT a real junction.
        """
        neighbors = list(G.neighbors(node))
        n = len(neighbors)
        if n <= 1:
            return True   # endpoint or isolated node
        if n == 2:
            return False  # unambiguously a chain node
        # For degree ≥ 3: check whether neighbours stay connected without node
        subgraph = G.subgraph(neighbors)
        return nx.number_connected_components(subgraph) >= 2

    def compress_graph(
        self,
        G: nx.Graph,
        radii_um: np.ndarray,
        sample_spacing_um: float = 1.0,
    ) -> nx.Graph:
        """
        Collapse degree-2 chains into single edges using topological detection.

        After compression every node is a topological endpoint (degree 1) or
        a genuine branch point (neighbourhood splits on removal).  Each edge
        stores intermediate sample points for SWC output.

        Args:
            G               : Voxel-resolution undirected graph.
            radii_um        : Radius per node (indexed same as G node IDs).
            sample_spacing_um : Spacing between intermediate samples (um).

        Returns:
            Compressed graph.
            Node attrs : pos, radius, degree.
            Edge attrs : length, points, radii.
        """
        if G.number_of_nodes() == 0:
            return nx.Graph()

        self.logger.info("Classifying structural nodes (topological)...")

        # Classify using topological test -- not raw degree
        structural: set = set()
        for n in G.nodes():
            if self._is_topological_junction(G, n):
                structural.add(n)

        # Pure cycle -- break at an arbitrary node
        if not structural:
            structural.add(next(iter(G.nodes())))

        self.logger.info(
            f"Structural nodes: {len(structural):,} / {G.number_of_nodes():,} "
            f"({100.0 * len(structural) / G.number_of_nodes():.1f}%)"
        )

        # Build compressed graph with structural nodes as vertices
        CG = nx.Graph()
        for n in structural:
            CG.add_node(
                n,
                pos=G.nodes[n]['pos'],
                radius=float(radii_um[n]) if n < len(radii_um) else 1.0,
            )

        # Trace chains between pairs of structural nodes
        visited_edges: set = set()

        for start in structural:
            for neighbor in list(G.neighbors(start)):
                if (start, neighbor) in visited_edges or (neighbor, start) in visited_edges:
                    continue

                chain = [start]
                chain_length = 0.0
                current = start
                nxt = neighbor

                # Walk along degree-2 (non-structural) nodes
                while nxt not in structural:
                    chain_length += G[current][nxt]['length']
                    chain.append(nxt)
                    visited_edges.add((current, nxt))
                    others = [nb for nb in G.neighbors(nxt) if nb != current]
                    if not others:
                        break
                    current = nxt
                    nxt = others[0]

                # Add the final edge that leads to (or IS) the second structural node
                if nxt in structural:
                    chain_length += G[current][nxt]['length']
                    chain.append(nxt)
                    visited_edges.add((current, nxt))

                    end = nxt
                    if not CG.has_edge(start, end):
                        pts, rads = self._sample_chain(chain, G, radii_um, sample_spacing_um)
                        CG.add_edge(
                            start, end,
                            length=chain_length,
                            points=pts,
                            radii=rads,
                        )

        # Refresh stored degree
        for n in CG.nodes():
            CG.nodes[n]['degree'] = CG.degree(n)

        self.logger.info(
            f"Compressed: {G.number_of_nodes():,} -> {CG.number_of_nodes():,} nodes  "
            f"({G.number_of_edges():,} -> {CG.number_of_edges():,} edges)"
        )
        return CG

    def _sample_chain(
        self,
        chain: List[int],
        G: nx.Graph,
        radii_um: np.ndarray,
        spacing: float,
    ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Return evenly-spaced intermediate samples along a voxel chain."""
        if len(chain) < 2:
            return [], []

        positions = [np.array(G.nodes[n]['pos']) for n in chain]
        node_radii = [
            float(radii_um[n]) if n < len(radii_um) else 1.0
            for n in chain
        ]

        # Cumulative arc length
        cum = [0.0]
        for i in range(1, len(positions)):
            cum.append(cum[-1] + float(np.linalg.norm(positions[i] - positions[i - 1])))

        total = cum[-1]
        if total < spacing:
            return [], []   # chain too short for any intermediate sample

        pts: List[Tuple[float, float, float]] = []
        rads: List[float] = []
        seg = 0
        dist = spacing

        while dist < total:
            while seg < len(cum) - 1 and cum[seg + 1] < dist:
                seg += 1
            if seg >= len(cum) - 1:
                break
            seg_len = cum[seg + 1] - cum[seg]
            t = (dist - cum[seg]) / seg_len if seg_len > 0 else 0.0
            pt = positions[seg] * (1 - t) + positions[seg + 1] * t
            rd = node_radii[seg] * (1 - t) + node_radii[seg + 1] * t
            pts.append(tuple(pt))
            rads.append(float(rd))
            dist += spacing

        return pts, rads

    # -------------------------------------------------------------------------
    # Step 6: Spur pruning
    # -------------------------------------------------------------------------

    def prune_spurs(
        self,
        G: nx.Graph,
        spur_length_um: float = 2.0,
    ) -> Tuple[nx.Graph, int]:
        """
        Iteratively remove leaf branches shorter than *spur_length_um*.

        Args:
            G              : Compressed undirected graph.
            spur_length_um : Length threshold (um).

        Returns:
            (pruned_graph, number_of_spurs_removed)
        """
        G = G.copy()
        removed = 0
        changed = True

        while changed:
            changed = False
            if G.number_of_nodes() <= 2:
                break
            to_remove = [
                n for n in G.nodes()
                if G.degree(n) == 1
                and G[n][next(iter(G.neighbors(n)))]['length'] < spur_length_um
            ]
            for n in to_remove:
                if n in G and G.number_of_nodes() > 2:
                    G.remove_node(n)
                    removed += 1
                    changed = True

        for n in G.nodes():
            G.nodes[n]['degree'] = G.degree(n)

        self.logger.info(f"Spur pruning: removed {removed} branches < {spur_length_um} um")
        return G, removed

    # -------------------------------------------------------------------------
    # Step 7: Graph -> rooted directed tree
    # -------------------------------------------------------------------------

    def graph_to_tree(
        self,
        G: nx.Graph,
        root_node: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Convert a compressed undirected graph to a BFS-rooted directed tree.

        Root selection (when *root_node* is None):
        1. Find all endpoints (degree 1).
        2. Among endpoints, pick the one that is one end of the graph diameter
           (longest shortest-path).  This avoids selecting a bouton swelling
           as the root, since it would typically lie mid-cable.

        Args:
            G         : Compressed undirected skeleton graph.
            root_node : Optional explicit root (graph node ID).

        Returns:
            Directed tree (NetworkX DiGraph) with same node/edge attributes.
        """
        if G.number_of_nodes() == 0:
            return nx.DiGraph()

        if root_node is None:
            endpoints = [n for n in G.nodes() if G.nodes[n].get('degree', G.degree(n)) == 1]
            if endpoints:
                max_dist, root_node = 0, endpoints[0]
                for i, n1 in enumerate(endpoints):
                    for n2 in endpoints[i + 1:]:
                        try:
                            d = nx.shortest_path_length(G, n1, n2, weight='length')
                            if d > max_dist:
                                max_dist, root_node = d, n1
                        except nx.NetworkXNoPath:
                            continue
                self.logger.info(f"Root: node {root_node} (diameter endpoint)")
            else:
                root_node = max(G.nodes(), key=lambda n: G.nodes[n].get('degree', G.degree(n)))
                self.logger.info(f"Root: node {root_node} (highest degree, no endpoints)")

        tree = nx.DiGraph()
        for n in G.nodes():
            tree.add_node(n, **G.nodes[n])

        visited = {root_node}
        queue = [root_node]
        while queue:
            parent = queue.pop(0)
            for nb in G.neighbors(parent):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    tree.add_edge(parent, nb, **G.get_edge_data(parent, nb))

        self.logger.info(
            f"Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges"
        )
        return tree

    # -------------------------------------------------------------------------
    # Step 8: Synthetic soma insertion
    # -------------------------------------------------------------------------

    def insert_synthetic_soma(self, tree: nx.DiGraph) -> nx.DiGraph:
        """
        Insert a 2-sample synthetic soma stub at the root for Arbor compatibility.

        Arbor's SWC parser rejects morphologies where the soma is described
        by a single sample.  Since no biological soma location is known for
        axon-only / dendrite-only segments, this inserts a second soma sample
        displaced one root-radius along the first outgoing edge direction.

        Sets node attribute ``swc_type``:
            1  -- the two soma samples (root + synthetic neighbour)
            0  -- all other nodes (compartment type = undefined)

        Args:
            tree: Directed skeleton tree from graph_to_tree().

        Returns:
            Modified tree (edited in-place and returned).
        """
        if tree.number_of_nodes() == 0:
            return tree

        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        if not roots:
            return tree
        root = roots[0]

        root_pos = np.array(tree.nodes[root]['pos'])
        root_r = float(tree.nodes[root].get('radius', 1.0))
        new_id = max(tree.nodes()) + 1
        children = list(tree.successors(root))

        if children:
            first_child = children[0]
            child_pos = np.array(tree.nodes[first_child]['pos'])
            direction = child_pos - root_pos
            dist = float(np.linalg.norm(direction))
            direction = direction / dist if dist > 0 else np.array([1.0, 0.0, 0.0])
            soma_pos = tuple(root_pos + direction * root_r)

            tree.add_node(new_id, pos=soma_pos, radius=root_r, swc_type=1)
            old_edge = dict(tree[root][first_child])
            tree.remove_edge(root, first_child)
            tree.add_edge(root, new_id, length=root_r, points=[], radii=[])
            old_edge['length'] = max(old_edge.get('length', 0.0) - root_r, 0.0)
            tree.add_edge(new_id, first_child, **old_edge)
        else:
            soma_pos = tuple(root_pos + np.array([root_r, 0.0, 0.0]))
            tree.add_node(new_id, pos=soma_pos, radius=root_r, swc_type=1)
            tree.add_edge(root, new_id, length=root_r, points=[], radii=[])

        tree.nodes[root]['swc_type'] = 1
        for n in tree.nodes():
            tree.nodes[n].setdefault('swc_type', 0)

        self.logger.info(f"Synthetic soma: root={root}, stub={new_id}")
        return tree


# =============================================================================
# SWC Writer
# =============================================================================

class SWCWriter:
    """
    Write a compressed skeleton tree to SWC format.

    SWC columns: id  type  x  y  z  radius  parent_id
    type: 1=soma, 2=axon, 3=dendrite, 4=apical dendrite, 0=undefined
    Requirement: parent_id < id for all non-root nodes.

    Intermediate edge samples are interleaved between structural nodes so
    that the file faithfully represents the cable geometry at the requested
    spatial resolution.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def write_swc(
        self,
        tree: nx.DiGraph,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Write a compressed skeleton tree to an SWC file.

        Node attributes read:
            pos      (tuple)  -- (x, y, z) in um
            radius   (float)  -- radius in um
            swc_type (int)    -- SWC compartment tag

        Edge attributes read:
            points   (list)   -- intermediate (x, y, z) samples
            radii    (list)   -- radii at those samples

        Args:
            tree        : Compressed directed skeleton tree.
            output_path : Destination .swc path.
            metadata    : Optional dict written to file header.

        Returns:
            output_path (str)
        """
        self.logger.info(f"Writing SWC: {output_path}")

        if tree.number_of_nodes() == 0:
            raise ValueError("Skeleton tree is empty -- nothing to write.")

        # Explicit Unix line endings via newline='\n'
        with open(output_path, 'w', newline='\n') as f:
            f.write("# SWC format skeleton\n")
            f.write("# Generated by VAST Neural Reconstruction Pipeline\n")
            f.write("# WARNING: Soma is synthetic. No biological soma location is known.\n")
            if metadata:
                for key, val in metadata.items():
                    f.write(f"# {key}: {val}\n")
            f.write("# Columns: id, type, x, y, z, radius, parent_id\n")
            f.write("#\n")
            self._write_tree_swc(f, tree)

        n_samples = (
            1  # root itself
            + sum(1 + len(d.get('points', [])) for _, _, d in tree.edges(data=True))
        )
        self.logger.info(f"Wrote {n_samples} SWC samples -> {output_path}")
        return output_path

    # ------------------------------------------------------------------

    def _write_tree_swc(self, f, tree: nx.DiGraph) -> None:
        """
        BFS traversal of the tree.

        For each edge (parent -> child) the intermediate samples stored on
        that edge are emitted between the parent structural node and the
        child structural node, maintaining parent_id < id throughout.
        """
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        if not roots:
            self.logger.warning("No root found; using first node.")
            roots = [next(iter(tree.nodes()))]
        root = roots[0]

        swc_id = 1
        queue = [(root, -1)]   # (graph_node, parent_swc_id)

        while queue:
            node, parent_swc_id = queue.pop(0)

            pos    = tree.nodes[node]['pos']
            radius = tree.nodes[node].get('radius', 1.0)
            stype  = tree.nodes[node].get('swc_type', 0)

            f.write(
                f"{swc_id} {stype} "
                f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{radius:.6f} {parent_swc_id}\n"
            )
            node_swc_id = swc_id
            swc_id += 1

            for child in tree.successors(node):
                edge       = tree[node][child]
                edge_pts   = edge.get('points', [])
                edge_radii = edge.get('radii', [])

                prev_id = node_swc_id
                for pt, rad in zip(edge_pts, edge_radii):
                    f.write(
                        f"{swc_id} 0 "
                        f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                        f"{rad:.6f} {prev_id}\n"
                    )
                    prev_id = swc_id
                    swc_id += 1

                queue.append((child, prev_id))


# =============================================================================
# Convenience wrappers
# =============================================================================

def quick_extract_skeleton(
    mask: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    bbox_min_vox: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """One-call skeleton extraction.  Returns (tree, stats)."""
    return SkeletonExtractor().extract_skeleton(mask, voxel_size_um, bbox_min_vox)


def quick_write_swc(
    tree: nx.DiGraph,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """One-call SWC writer.  Returns output_path."""
    return SWCWriter().write_swc(tree, output_path, metadata=metadata)