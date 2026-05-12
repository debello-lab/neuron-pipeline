"""
Voxel Mask to SWC Skeleton Conversion
Neural Reconstruction Pipeline

Skeletonization method: TEASAR via kimimaro
    kimimaro.skeletonize() operates on binary voxel masks and returns a
    compressed path graph (vertices + edges) directly.  It does not produce
    a thinned voxel mask -- the output is already a sparse set of structural
    nodes (endpoints and branch points) connected by edges.  No voxel-graph
    construction or chain compression is required after this step.

    If kimimaro proves unsuitable (e.g. fails to resolve thin processes on
    this dataset), the next candidate is mesh-based skeletonization (e.g.
    CGAL mean curvature flow).  The graph-to-tree and SWC stages below are
    method-agnostic and would not need to change in that case -- only the
    front end (_skeletonize_teasar) would be replaced.

Pipeline:
    1. TEASAR skeletonization (kimimaro) -> compressed graph
    2. Radius estimation via Euclidean distance transform
    3. Collapse bouton swelling clusters
    4. Spur pruning
    5. Convert graph to rooted directed tree
    6. Insert synthetic 2-sample soma for Arbor compatibility
    7. SWC export

Coordinate conventions (applied consistently throughout):
    - Mask axis order:        (Z, Y, X)
    - kimimaro anisotropy:    (sz, sy, sx)  -- must match mask axis order
    - kimimaro vertex order:  (Z, Y, X) local physical µm (after *= anisotropy)
    - Node pos attribute:     (x_um, y_um, z_um) physical microns, XYZ
    - coord_frame tag:        'physical_um_xyz_center'

Author: Nicolas Randazzo
"""

import numpy as np
import kimimaro
import networkx as nx
import logging

from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, List, Optional, Tuple


class SkeletonExtractor:
    """
    Convert binary voxel masks to SWC skeleton format.

    Pipeline:
        1. TEASAR skeletonization via kimimaro
        2. EDT-based radius estimation
        3. Collapse bouton swelling clusters
        4. Spur pruning
        5. Graph to rooted directed tree
        6. Synthetic soma insertion for Arbor compatibility
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
        teasar_scale: float = 1.5,
        teasar_const_um: float = 0.1,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Extract a skeleton tree from a binary voxel mask.

        Args:
            mask            : Boolean 3D array (Z, Y, X).
            voxel_size_um   : Voxel dimensions in microns (sx, sy, sz).
            bbox_min_vox    : Global voxel origin of this mask (minx, miny, minz).
            spur_length_um  : Prune leaf branches shorter than this (um).
            teasar_scale    : TEASAR invalidation sphere scale factor.
                              invalidation_radius = scale * DBF(x) + const
                              Larger = more aggressive branch suppression.
                              Start at 1.5; increase if spurious branches remain.
            teasar_const_um : TEASAR invalidation sphere constant (um).
                              Provides a minimum invalidation radius regardless
                              of local DBF value.  Start at 1.0.

        Returns:
            (tree, stats)

            tree  -- Directed skeleton tree (NetworkX DiGraph).

            - Node attrs: pos (um XYZ tuple), radius (um float), swc_type (int).

            - Edge attrs: length (um float), points (list), radii (list).

            stats -- Extraction statistics dictionary.
        """
        self.logger.info("[extract_skeleton] Starting TEASAR skeleton extraction...")

        stats: Dict[str, Any] = {
            'input_voxel_count': int(np.sum(mask)),
            'skeleton_vertices': 0,
            'skeleton_edges': 0,
            'nodes_after_bouton_collapse': 0,
            'branches_pruned': 0,
            'components_dropped': 0,
            'bouton_clusters_collapsed': 0,
            'bouton_nodes_removed': 0,
            'bouton_radius_threshold_um': 0.0,
            'total_length_um': 0.0,
            'teasar_scale': teasar_scale,
            'teasar_const_um': teasar_const_um,
            'coord_frame': 'physical_um_xyz_center',
        }

        if not np.any(mask):
            self.logger.error("[extract_skeleton] Input mask is empty")
            return nx.DiGraph(), stats

        # ------------------------------------------------------------------
        # Step 1: TEASAR skeletonization + radius estimation
        # ------------------------------------------------------------------
        G = self._skeletonize_teasar(
            mask, voxel_size_um, bbox_min_vox, teasar_scale, teasar_const_um
        )

        if G.number_of_nodes() == 0:
            return nx.DiGraph(), stats

        stats['skeleton_vertices'] = G.number_of_nodes()
        stats['skeleton_edges'] = G.number_of_edges()

        # ------------------------------------------------------------------
        # Step 2: Collapse bouton swelling clusters
        #
        # TEASAR can produce dense vertex clusters inside bulbous regions
        # (boutons, spine heads) where many short paths converge.  Replace
        # each such cluster with a single centroid node.
        # ------------------------------------------------------------------
        G, collapse_stats = self.collapse_high_radius_clusters(G)
        stats['bouton_clusters_collapsed'] = collapse_stats['clusters_collapsed']
        stats['bouton_nodes_removed']      = collapse_stats['nodes_removed']
        stats['bouton_radius_threshold_um'] = collapse_stats['radius_threshold_um']
        stats['nodes_after_bouton_collapse'] = G.number_of_nodes()

        # ------------------------------------------------------------------
        # Step 3: Spur pruning
        #
        # kimimaro has its own dust/spur threshold, but pruning here gives
        # explicit control in physical units and acts as a second-pass filter.
        # ------------------------------------------------------------------
        G, spurs_removed = self.prune_spurs(G, spur_length_um)
        stats['branches_pruned'] = spurs_removed

        # ------------------------------------------------------------------
        # Step 4: Convert to rooted directed tree
        # ------------------------------------------------------------------
        tree, n_dropped = self.graph_to_tree(G)
        stats['components_dropped'] = n_dropped

        # ------------------------------------------------------------------
        # Step 5: Insert synthetic 2-sample soma for Arbor compatibility
        # ------------------------------------------------------------------
        tree = self.insert_synthetic_soma(tree)

        stats['total_length_um'] = sum(
            d.get('length', 0.0) for _, _, d in tree.edges(data=True)
        )
        self.logger.info(
            f"[extract_skeleton] Final tree: {tree.number_of_nodes()} nodes, "
            f"{stats['total_length_um']:.2f} um total cable"
        )
        return tree, stats

    # -------------------------------------------------------------------------
    # Step 1: TEASAR skeletonization + radius estimation
    # -------------------------------------------------------------------------

    def _skeletonize_teasar(
        self,
        mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float],
        bbox_min_vox: Tuple[int, int, int],
        teasar_scale: float,
        teasar_const_um: float,
    ) -> nx.Graph:
        """
        Run TEASAR via kimimaro and return an undirected NetworkX graph with
        physical-space node positions and EDT-derived radii.

        kimimaro returns vertices in local physical µm -- it applies
        ``vertices *= anisotropy`` internally before returning, so each
        coordinate is already scaled to physical units relative to the
        local mask origin.  Global positions are computed by adding the
        bbox origin in physical µm.

        Coordinate mapping per node (vertex index i):
            vz_phys, vy_phys, vx_phys = vertices[i]   # local physical µm, ZYX
            x_um = vx_phys + minx * sx
            y_um = vy_phys + miny * sy
            z_um = vz_phys + minz * sz
            pos  = (x_um, y_um, z_um)                 # global physical µm, XYZ

        Args:
            mask            : Boolean 3D array (Z, Y, X).
            voxel_size_um   : (sx, sy, sz).
            bbox_min_vox    : (minx, miny, minz) global voxel origin.
            teasar_scale    : TEASAR invalidation scale factor.
            teasar_const_um : TEASAR invalidation constant (um).

        Returns:
            Undirected NetworkX graph.
            Node attrs : pos (um XYZ tuple), radius (um float), degree (int).
            Edge attrs : length (um float), points ([]), radii ([]).
            Empty graph on failure.
        """
        sx, sy, sz = voxel_size_um
        minx, miny, minz = bbox_min_vox

        # anisotropy must match mask axis order (Z, Y, X)
        anisotropy = (sz, sy, sx)

        self.logger.info(
            f"[teasar] Running kimimaro (scale={teasar_scale}, "
            f"const={teasar_const_um} um, anisotropy={anisotropy})"
        )

        skels = kimimaro.skeletonize(
            mask.astype(np.uint8),
            teasar_params={
                'scale':                    teasar_scale,
                'const':                    teasar_const_um,
                'pdrf_scale':               100000,
                'pdrf_exponent':            4,
                'soma_invalidation_scale':  0.5,
                'soma_invalidation_const':  0,
            },
            anisotropy=anisotropy,
            dust_threshold=0,   # small components removed upstream in voxel_cleaning
            progress=False,
        )

        # kimimaro keys output by label value; boolean mask produces label 1
        if 1 not in skels or len(skels[1].vertices) == 0:
            self.logger.error("[teasar] kimimaro returned empty skeleton")
            return nx.Graph()

        skel     = skels[1]
        vertices = skel.vertices   # (N, 3) float, local physical µm, ZYX order
        edges    = skel.edges      # (M, 2) int index pairs

        self.logger.info(
            f"[teasar] Raw skeleton: {len(vertices)} vertices, {len(edges)} edges"
        )

        # --- Radius estimation via EDT ---------------------------------------
        # sampling=(sz, sy, sx) must match mask axis order (Z, Y, X)
        dist_um = distance_transform_edt(mask.astype(bool), sampling=anisotropy)

        radius_floor = max(0.005, min(voxel_size_um) / 2.0)

        # vertices are in local physical µm; convert back to voxel indices for EDT lookup
        vox_idx_z = np.clip(np.round(vertices[:, 0] / sz).astype(int), 0, mask.shape[0] - 1)
        vox_idx_y = np.clip(np.round(vertices[:, 1] / sy).astype(int), 0, mask.shape[1] - 1)
        vox_idx_x = np.clip(np.round(vertices[:, 2] / sx).astype(int), 0, mask.shape[2] - 1)

        radii_um = dist_um[vox_idx_z, vox_idx_y, vox_idx_x]
        radii_um = np.maximum(radii_um, radius_floor)

        self.logger.info(
            f"[teasar] Radii: {radii_um.min():.3f} - {radii_um.max():.3f} um "
            f"(median {np.median(radii_um):.3f} um, floor {radius_floor:.3f} um)"
        )

        # --- Build NetworkX graph --------------------------------------------
        G = nx.Graph()

        for i, (vz_phys, vy_phys, vx_phys) in enumerate(vertices):
            pos_x = float(vx_phys) + minx * sx
            pos_y = float(vy_phys) + miny * sy
            pos_z = float(vz_phys) + minz * sz
            G.add_node(i, pos=(pos_x, pos_y, pos_z), radius=float(radii_um[i]))

        for u, v in edges:
            pu = np.array(G.nodes[int(u)]['pos'])
            pv = np.array(G.nodes[int(v)]['pos'])
            G.add_edge(
                int(u), int(v),
                length=float(np.linalg.norm(pu - pv)),
                points=[],
                radii=[],
            )

        for n in G.nodes():
            G.nodes[n]['degree'] = G.degree(n)

        self.logger.info(
            f"[teasar] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    # -------------------------------------------------------------------------
    # Step 2: Collapse bouton swelling clusters
    # -------------------------------------------------------------------------


    def collapse_high_radius_clusters(
        self,
        G: nx.Graph,
        radius_factor: float = 1.5,
        min_cluster_nodes: int = 3,
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
    
        """
        Collapse dense node clusters caused by bouton swellings.

        TEASAR can produce tightly connected subgraphs of nodes with
        abnormally large radii inside bulbous regions.  This replaces each
        such cluster with a single centroid node, preserving external
        connectivity.

        Clustering uses single-linkage: two candidate nodes are linked if
        their distance is less than the larger of their two radii.

        Args:
            G                 : Undirected skeleton graph.
                                Node attrs: pos, radius, degree.
                                Edge attrs: length, points, radii.
            radius_factor     : Nodes with radius > radius_factor * median
                                are candidates.
            min_cluster_nodes : Minimum cluster size to collapse.

        Returns:
            (collapsed_graph, stats)
            stats keys: radius_threshold_um, median_radius_um,
                        candidates_found, clusters_found,
                        clusters_collapsed, nodes_removed.
        """
        zero_stats: Dict[str, Any] = {
            'radius_threshold_um': 0.0,
            'median_radius_um':    0.0,
            'candidates_found':    0,
            'clusters_found':      0,
            'clusters_collapsed':  0,
            'nodes_removed':       0,
        }

        if G.number_of_nodes() < min_cluster_nodes:
            return G.copy(), zero_stats

        all_radii      = np.array([G.nodes[n]['radius'] for n in G.nodes()])
        median_radius  = float(np.median(all_radii))
        radius_threshold = radius_factor * median_radius

        self.logger.info(
            f"[bouton_collapse] median_radius={median_radius:.3f} um, "
            f"threshold={radius_threshold:.3f} um (factor={radius_factor})"
        )

        candidates = [n for n in G.nodes() if G.nodes[n]['radius'] > radius_threshold]

        if not candidates:
            self.logger.info("[bouton_collapse] No nodes above threshold -- skipping")
            zero_stats['median_radius_um']    = median_radius
            zero_stats['radius_threshold_um'] = radius_threshold
            return G.copy(), zero_stats

        if len(candidates) == G.number_of_nodes():
            self.logger.warning(
                f"[bouton_collapse] All {G.number_of_nodes()} nodes above threshold "
                "-- skipping (segment may be uniformly large)"
            )
            zero_stats['median_radius_um']    = median_radius
            zero_stats['radius_threshold_um'] = radius_threshold
            zero_stats['candidates_found']    = len(candidates)
            return G.copy(), zero_stats

        # --- Graph-connectivity clustering ------------------------------------
        # KDTree spatial proximity fails here: kimimaro bouton nodes are
        # connected by skeleton edges but their physical positions can be
        # farther apart than their EDT radii, so distance-based linkage
        # misses the cluster.  Use the skeleton graph itself instead --
        # two candidate nodes belong to the same cluster iff they are
        # connected through the subgraph of candidates.
        cand_set    = set(candidates)
        all_clusters = [list(c) for c in nx.connected_components(G.subgraph(cand_set))]
        clusters     = [c for c in all_clusters if len(c) >= min_cluster_nodes]

        self.logger.info(
            f"[bouton_collapse] {len(candidates)} candidates, "
            f"{len(all_clusters)} cluster(s), "
            f"{len(clusters)} >= {min_cluster_nodes} nodes"
        )

        if not clusters:
            stats = dict(zero_stats)
            stats['median_radius_um']    = median_radius
            stats['radius_threshold_um'] = radius_threshold
            stats['candidates_found']    = len(candidates)
            stats['clusters_found']      = len(all_clusters)
            return G.copy(), stats

        # --- Batch collapse ---------------------------------------------------
        orig_components = nx.number_connected_components(G)

        member_to_centroid: Dict[int, int] = {}
        centroid_info: Dict[int, Tuple[Tuple[float, float, float], float]] = {}
        next_id = max(G.nodes()) + 1

        for cluster in clusters:
            cid = next_id
            next_id += 1
            pos = tuple(float(x) for x in np.mean(
                [G.nodes[n]['pos'] for n in cluster], axis=0
            ))
            rad = float(max(G.nodes[n]['radius'] for n in cluster))
            for n in cluster:
                member_to_centroid[n] = cid
            centroid_info[cid] = (pos, rad)

        G_out = nx.Graph()

        for n, attrs in G.nodes(data=True):
            if n not in member_to_centroid:
                G_out.add_node(n, **attrs)

        for cid, (pos, rad) in centroid_info.items():
            G_out.add_node(cid, pos=pos, radius=rad, degree=0)

        for u, v, data in G.edges(data=True):
            u_mapped = member_to_centroid.get(u, u)
            v_mapped = member_to_centroid.get(v, v)

            if u_mapped == v_mapped:
                continue    # intra-cluster edge -- drop

            pos_u = centroid_info[u_mapped][0] if u_mapped in centroid_info else G.nodes[u]['pos']
            pos_v = centroid_info[v_mapped][0] if v_mapped in centroid_info else G.nodes[v]['pos']
            new_length = float(np.linalg.norm(np.array(pos_u) - np.array(pos_v)))

            # Filter intermediate samples inside cluster spheres
            new_points: List[Tuple[float, float, float]] = []
            new_radii:  List[float] = []
            for pt, r in zip(data.get('points', []), data.get('radii', [])):
                pt_arr = np.array(pt)
                inside = any(
                    float(np.linalg.norm(pt_arr - np.array(centroid_info[cid][0]))) < centroid_info[cid][1]
                    for cid in (u_mapped, v_mapped)
                    if cid in centroid_info
                )
                if not inside:
                    new_points.append(pt)
                    new_radii.append(r)

            edge_key = (min(u_mapped, v_mapped), max(u_mapped, v_mapped))
            if G_out.has_edge(*edge_key):
                if new_length < G_out[edge_key[0]][edge_key[1]]['length']:
                    G_out[edge_key[0]][edge_key[1]].update(
                        length=new_length, points=new_points, radii=new_radii
                    )
            else:
                G_out.add_edge(
                    u_mapped, v_mapped,
                    length=new_length, points=new_points, radii=new_radii,
                )

        for n in G_out.nodes():
            G_out.nodes[n]['degree'] = G_out.degree(n)

        new_components = nx.number_connected_components(G_out)
        if new_components != orig_components:
            self.logger.warning(
                f"[bouton_collapse] Connectivity changed: "
                f"{orig_components} -> {new_components} components"
            )

        nodes_removed = sum(len(c) for c in clusters) - len(clusters)
        self.logger.info(
            f"[bouton_collapse] {len(clusters)} cluster(s) collapsed, "
            f"{nodes_removed} nodes removed"
        )

        return G_out, {
            'radius_threshold_um': radius_threshold,
            'median_radius_um':    median_radius,
            'candidates_found':    len(candidates),
            'clusters_found':      len(all_clusters),
            'clusters_collapsed':  len(clusters),
            'nodes_removed':       nodes_removed,
        }

    # -------------------------------------------------------------------------
    # Step 3: Spur pruning
    # -------------------------------------------------------------------------

    def prune_spurs(
        self,
        G: nx.Graph,
        spur_length_um: float = 2.0,
    ) -> Tuple[nx.Graph, int]:
        """
        Iteratively remove leaf branches shorter than spur_length_um.

        kimimaro has its own internal spur suppression via the invalidation
        sphere, but pruning here provides explicit control in physical units
        and acts as a second-pass filter.

        Args:
            G              : Undirected skeleton graph.
            spur_length_um : Prune threshold (um).

        Returns:
            (pruned_graph, number_of_spurs_removed)
        """
        def _spur_path_length(G: nx.Graph, leaf: int) -> Tuple[Optional[float], int]:
            # Walk from leaf toward the interior, accumulating edge lengths.
            # Returns (cumulative_length, terminal_node) when the walk hits a
            # branch point (degree >= 3) -- that spur is a pruning candidate.
            # Returns (None, terminal) when the walk reaches another leaf:
            # a leaf-to-leaf path is the main axis cable and must never be pruned.
            length = 0.0
            prev, cur = None, leaf
            while True:
                nbrs = [n for n in G.neighbors(cur) if n != prev]
                if not nbrs:
                    return None, cur        # isolated node -- skip
                nxt = nbrs[0]
                length += G[cur][nxt]['length']
                if G.degree(nxt) >= 3:
                    return length, nxt      # valid spur: ends at branch point
                if G.degree(nxt) == 1:
                    return None, nxt        # leaf-to-leaf: main axis, never prune
                prev, cur = cur, nxt

        G = G.copy()
        removed = 0
        changed = True

        while changed:
            changed = False
            if G.number_of_nodes() <= 2:
                break
            spurs = []
            for n in list(G.nodes()):
                if G.degree(n) == 1:
                    spur_len, _ = _spur_path_length(G, n)
                    if spur_len is not None and spur_len < spur_length_um:
                        spurs.append((spur_len, n))
            spurs.sort()                    # prune shortest first
            for _, n in spurs:
                if n in G and G.number_of_nodes() > 2:
                    G.remove_node(n)
                    removed += 1
                    changed = True

        for n in G.nodes():
            G.nodes[n]['degree'] = G.degree(n)

        self.logger.info(f"[prune_spurs] Removed {removed} branches < {spur_length_um} um")
        return G, removed

    # -------------------------------------------------------------------------
    # Step 4: Graph -> rooted directed tree
    # -------------------------------------------------------------------------

    def graph_to_tree(
        self,
        G: nx.Graph,
        root_node: Optional[int] = None,
    ) -> Tuple[nx.DiGraph, int]:
        """
        Convert an undirected skeleton graph to a BFS-rooted directed tree.

        The skeleton graph may be disconnected if a thin process falls below
        the TEASAR invalidation radius or spur pruning cuts a narrow neck.
        Only the connected component containing the root is kept; other
        components are dropped and counted.

        This conversion forces a graph into a tree.  Cycles are implicitly
        broken by BFS (the first path to each node is kept).  If the graph
        is a proper tree (no cycles) after TEASAR + pruning, BFS produces an
        exact result.  Any dropped edges or components are logged so the
        caller can inspect them.

        Root selection (when root_node is None):
            1. Find all endpoints (degree 1).
            2. Pick the endpoint that is one end of the graph diameter
               (longest shortest-path between any two endpoints).
               This avoids rooting at a bouton swelling mid-cable.
            3. If no endpoints exist, pick the highest-degree node.

        Args:
            G         : Undirected skeleton graph.
            root_node : Optional explicit root (graph node ID).

        Returns:
            (tree, n_dropped)
            tree      -- Directed tree (NetworkX DiGraph).
                         coord_frame graph attr: 'physical_um_xyz_center'
            n_dropped -- Number of disconnected components dropped (0 = ok).
        """
        if G.number_of_nodes() == 0:
            return nx.DiGraph(), 0

        if root_node is None:
            endpoints = [n for n in G.nodes() if G.nodes[n].get('degree', G.degree(n)) == 1]
            if endpoints:
                max_dist, root_node = 0.0, endpoints[0]
                for i, n1 in enumerate(endpoints):
                    for n2 in endpoints[i + 1:]:
                        try:
                            d = nx.shortest_path_length(G, n1, n2, weight='length')
                            if d > max_dist:
                                max_dist, root_node = d, n1
                        except nx.NetworkXNoPath:
                            continue
                self.logger.info(f"[graph_to_tree] Root: node {root_node} (diameter endpoint)")
            else:
                root_node = max(G.nodes(), key=lambda n: G.nodes[n].get('degree', G.degree(n)))
                self.logger.info(f"[graph_to_tree] Root: node {root_node} (highest degree, no endpoints)")

        n_components = nx.number_connected_components(G)
        n_dropped    = n_components - 1

        if n_dropped > 0:
            root_component = nx.node_connected_component(G, root_node)
            dropped_nodes  = G.number_of_nodes() - len(root_component)
            self.logger.warning(
                f"[graph_to_tree] Graph has {n_components} components -- "
                f"keeping root component ({len(root_component)} nodes), "
                f"dropping {n_dropped} component(s) ({dropped_nodes} nodes)."
            )
            G = G.subgraph(root_component)

        tree = nx.DiGraph()
        for n in G.nodes():
            tree.add_node(n, **G.nodes[n])

        visited = {root_node}
        queue   = [root_node]
        while queue:
            parent = queue.pop(0)
            for nb in G.neighbors(parent):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    tree.add_edge(parent, nb, **G.get_edge_data(parent, nb))

        # Coordinate frame tag -- asserted by downstream centroid mapper
        tree.graph['coord_frame'] = 'physical_um_xyz_center'

        self.logger.info(
            f"[graph_to_tree] Tree: {tree.number_of_nodes()} nodes, "
            f"{tree.number_of_edges()} edges"
        )
        return tree, n_dropped

    # -------------------------------------------------------------------------
    # Step 5: Synthetic soma insertion
    # -------------------------------------------------------------------------

    def insert_synthetic_soma(self, tree: nx.DiGraph) -> nx.DiGraph:
        """
        Insert a 2-sample synthetic soma stub at the root for Arbor compatibility.

        Arbor's SWC parser rejects morphologies where the soma is represented
        by a single sample.  Since no biological soma location is known for
        axon-only or dendrite-only segments, a second soma sample is inserted
        displaced one root-radius along the direction of the first outgoing edge.

        Sets node attribute swc_type:
            1 -- the two soma samples (root + synthetic neighbour)
            0 -- all other nodes (compartment type = undefined)

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

        if len(roots) > 1:
            self.logger.warning(
                f"[soma_insert] Tree has {len(roots)} root nodes (in_degree==0); "
                "expected 1.  Only the first root receives a soma stub."
            )

        root     = roots[0]
        root_pos = np.array(tree.nodes[root]['pos'])
        root_r   = float(tree.nodes[root].get('radius', 1.0))
        new_id   = max(tree.nodes()) + 1
        children = list(tree.successors(root))

        if children:
            first_child = children[0]
            child_pos   = np.array(tree.nodes[first_child]['pos'])
            direction   = child_pos - root_pos
            dist        = float(np.linalg.norm(direction))
            direction   = direction / dist if dist > 0 else np.array([1.0, 0.0, 0.0])
            soma_pos    = tuple(root_pos + direction * root_r)

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

        self.logger.info(f"[soma_insert] root={root}, stub={new_id}")
        return tree


# =============================================================================
# SWC Writer
# =============================================================================

class SWCWriter:
    """
    Write a directed skeleton tree to SWC format.

    SWC columns: id  type  x  y  z  radius  parent_id
    type: 1=soma, 2=axon, 3=dendrite, 4=apical dendrite, 0=undefined

    Structural nodes (from the skeleton graph) are written at junctions and
    endpoints.  Intermediate samples stored on each edge (edge attr 'points')
    are interleaved between them to preserve cable geometry.  kimimaro edges
    carry no intermediate samples by default, so 'points' will typically be
    empty -- the writer handles both cases.
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
        Write a skeleton tree to an SWC file.

        Node attributes read:
            pos      (tuple)  -- (x, y, z) in um
            radius   (float)  -- radius in um
            swc_type (int)    -- SWC compartment tag

        Edge attributes read:
            points   (list)   -- intermediate (x, y, z) samples (may be empty)
            radii    (list)   -- radii at those samples

        Args:
            tree        : Directed skeleton tree.
            output_path : Destination .swc path.
            metadata    : Optional dict written to file header.

        Returns:
            output_path (str)
        """
        self.logger.info(f"[write_swc] Writing: {output_path}")

        if tree.number_of_nodes() == 0:
            raise ValueError("Skeleton tree is empty -- nothing to write.")

        with open(output_path, 'w', newline='\n') as f:
            f.write("# SWC format skeleton\n")
            f.write("# Generated by VAST Neural Reconstruction Pipeline\n")
            f.write("# Skeletonization method: TEASAR (kimimaro)\n")
            f.write("# WARNING: Soma is synthetic. No biological soma location is known.\n")
            if metadata:
                for key, val in metadata.items():
                    f.write(f"# {key}: {val}\n")
            f.write("# Columns: id, type, x, y, z, radius, parent_id\n")
            f.write("#\n")
            self._write_tree_swc(f, tree)

        n_samples = (
            1
            + sum(1 + len(d.get('points', [])) for _, _, d in tree.edges(data=True))
        )
        self.logger.info(f"[write_swc] Wrote {n_samples} samples -> {output_path}")
        return output_path

    def _write_tree_swc(self, f, tree: nx.DiGraph) -> None:
        """
        BFS traversal writing structural nodes and interleaved edge samples.

        parent_id < id is maintained throughout by processing nodes in BFS
        order and emitting edge intermediate samples before the child node.
        """
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        if not roots:
            self.logger.warning("[write_swc] No root found; using first node.")
            roots = [next(iter(tree.nodes()))]
        root = roots[0]

        swc_id = 1
        queue  = [(root, -1)]   # (graph_node, parent_swc_id)

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
            # Stamp SWC row ID for downstream resolvers (e.g. centroid mapper)
            tree.nodes[node]['swc_id'] = node_swc_id
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