"""
Voxel Mask to SWC Skeleton Conversion
Neural Reconstruction Pipeline

This module converts cleaned binary voxel masks into SWC morphology files:
- 3D skeletonization (medial axis thinning)
- Radius estimation via Euclidean distance transform
- Graph extraction from skeleton voxels
- Tree construction and SWC export

Based on the voxel_pipeline.md specification.

Author: Nicolas Randazzo
Date: 2026-02-11
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import networkx as nx
import logging


class SkeletonExtractor:
    """
    Convert binary voxel masks to SWC skeleton format.
    
    Pipeline:
    1. Skeletonize voxel mask (medial axis)
    2. Compute radii via distance transform
    3. Extract graph from skeleton voxels
    4. Convert graph to rooted tree
    5. Export to SWC format
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the skeleton extractor.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_skeleton(
        self,
        mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float],
        bbox_min_vox: Tuple[int, int, int] = (0, 0, 0),
        spur_length_um: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract skeleton from binary mask.
        
        Args:
            mask: Boolean 3D array (Z, Y, X)
            voxel_size_um: Voxel dimensions in microns (sx, sy, sz)
            bbox_min_vox: Origin coordinates in voxel space (minx, miny, minz)
            spur_length_um: Minimum branch length in microns (shorter branches pruned)
            
        Returns:
            Tuple of (skeleton_points, radii_um, stats)
            - skeleton_points: (N, 3) array of XYZ coordinates in microns
            - radii_um: (N,) array of radii in microns
            - stats: Dictionary with extraction statistics
        """
        self.logger.info("Starting skeleton extraction...")
        
        stats = {
            'input_voxel_count': int(np.sum(mask)),
            'skeleton_voxel_count': 0,
            'skeleton_points': 0,
            'branches_pruned': 0,
            'total_length_um': 0.0
        }
        
        if not np.any(mask):
            self.logger.error("Input mask is empty")
            return np.array([]), np.array([]), stats
        
        # Step 1: Skeletonize
        self.logger.info("Computing skeleton...")
        skeleton_mask = skeletonize(mask.astype(bool)) > 0
        skeleton_voxel_count = np.sum(skeleton_mask)
        stats['skeleton_voxel_count'] = int(skeleton_voxel_count)
        
        self.logger.info(f"Skeleton has {skeleton_voxel_count:,} voxels "
                        f"({100.0 * skeleton_voxel_count / stats['input_voxel_count']:.2f}% of input)")
        
        # Step 2: Compute distance transform (for radii)
        self.logger.info("Computing distance transform...")
        sz, sy, sx = voxel_size_um
        dist_um = distance_transform_edt(mask, sampling=(sz, sy, sx))
        
        # Step 3: Extract skeleton points and radii
        self.logger.info("Extracting skeleton points...")
        skel_coords = np.argwhere(skeleton_mask)  # Returns (N, 3) in ZYX order
        
        if len(skel_coords) == 0:
            self.logger.error("Skeleton is empty")
            return np.array([]), np.array([]), stats
        
        # Convert to physical coordinates
        # Note: skel_coords is in (Z, Y, X) order from argwhere
        skeleton_points = np.zeros((len(skel_coords), 3), dtype=np.float64)
        skeleton_points[:, 0] = (skel_coords[:, 2] + bbox_min_vox[0]) * sx  # X
        skeleton_points[:, 1] = (skel_coords[:, 1] + bbox_min_vox[1]) * sy  # Y
        skeleton_points[:, 2] = (skel_coords[:, 0] + bbox_min_vox[2]) * sz  # Z
        
        # Get radii at skeleton points
        radii_um = dist_um[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]]
        
        stats['skeleton_points'] = len(skeleton_points)
        
        self.logger.info(f"Extracted {len(skeleton_points):,} skeleton points")
        self.logger.info(f"Radius range: {radii_um.min():.3f} - {radii_um.max():.3f} um "
                        f"(median: {np.median(radii_um):.3f} um)")
        
        # TODO: Implement graph extraction and spur pruning
        # For now, just return the raw skeleton points
        
        return skeleton_points, radii_um, stats
    
    def skeleton_to_graph(
        self,
        skeleton_mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float]
    ) -> nx.Graph:
        """
        Convert skeleton voxel mask to graph representation.
        
        Args:
            skeleton_mask: Boolean 3D array of skeleton voxels
            voxel_size_um: Voxel dimensions in microns
            
        Returns:
            NetworkX graph where:
            - Nodes have 'pos' attribute (physical coordinates in um)
            - Nodes have 'degree' attribute (number of neighbors)
            - Edges have 'length' attribute (physical distance in um)
        """
        self.logger.info("Building skeleton graph...")
        
        # Get skeleton coordinates
        skel_coords = np.argwhere(skeleton_mask)
        
        if len(skel_coords) == 0:
            return nx.Graph()
        
        # Build KD-tree for finding neighbors
        tree = cKDTree(skel_coords)
        
        # Create graph
        G = nx.Graph()
        
        sz, sy, sx = voxel_size_um
        
        # Add nodes with physical positions
        for i, coord in enumerate(skel_coords):
            z, y, x = coord
            pos_um = (x * sx, y * sy, z * sz)
            G.add_node(i, pos=pos_um, voxel_pos=coord)
        
        # Find neighbors within 26-connectivity (sqrt3 = 1.73 voxels)
        for i, coord in enumerate(skel_coords):
            # Query neighbors within connectivity distance
            neighbors = tree.query_ball_point(coord, r=1.8)  # sqrt3 + small margin
            
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    # Calculate physical distance
                    pos_i = G.nodes[i]['pos']
                    pos_j = G.nodes[j]['pos']
                    dist_um = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                    
                    G.add_edge(i, j, length=dist_um)
        
        self.logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Compute degree for each node
        for node in G.nodes():
            G.nodes[node]['degree'] = G.degree(node)
        
        return G
    
    def graph_to_tree(
        self,
        G: nx.Graph,
        root_node: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Convert skeleton graph to rooted tree (for SWC export).
        
        Args:
            G: Undirected skeleton graph
            root_node: Optional root node ID (if None, choose automatically)
            
        Returns:
            Directed graph (tree) rooted at specified node
        """
        if G.number_of_nodes() == 0:
            return nx.DiGraph()
        
        # Choose root if not specified
        if root_node is None:
            # Find endpoints (degree 1) and choose one
            endpoints = [n for n in G.nodes() if G.nodes[n]['degree'] == 1]
            
            if endpoints:
                # Use diameter: find pair of endpoints farthest apart
                max_dist = 0
                best_pair = (endpoints[0], endpoints[0])
                
                for i, n1 in enumerate(endpoints):
                    for n2 in endpoints[i+1:]:
                        try:
                            path_length = nx.shortest_path_length(
                                G, n1, n2, weight='length'
                            )
                            if path_length > max_dist:
                                max_dist = path_length
                                best_pair = (n1, n2)
                        except nx.NetworkXNoPath:
                            continue
                
                root_node = best_pair[0]
                self.logger.info(f"Auto-selected root node {root_node} (diameter endpoint)")
            else:
                # No endpoints, use node with highest degree (branching point)
                root_node = max(G.nodes(), key=lambda n: G.nodes[n]['degree'])
                self.logger.info(f"Auto-selected root node {root_node} (highest degree)")
        
        # Build directed tree from root using BFS
        tree = nx.DiGraph()
        
        # Copy node attributes
        for node in G.nodes():
            tree.add_node(node, **G.nodes[node])
        
        # BFS to build tree
        visited = {root_node}
        queue = [root_node]
        
        while queue:
            parent = queue.pop(0)
            
            for neighbor in G.neighbors(parent):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Add directed edge from parent to child
                    edge_data = G.get_edge_data(parent, neighbor)
                    tree.add_edge(parent, neighbor, **edge_data)
        
        self.logger.info(f"Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")
        
        return tree


class SWCWriter:
    """
    Write skeleton data to SWC format.
    
    SWC Format:
    - Each row: id, type, x, y, z, radius, parent_id
    - type: 0=undefined, 1=soma, 2=axon, 3=dendrite, 4=apical dendrite
    - parent_id: -1 for root, otherwise refers to parent node id
    - Requirement: parent_id < id (parent before child)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the SWC writer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def write_swc(
        self,
        skeleton_points: np.ndarray,
        radii_um: np.ndarray,
        output_path: str,
        tree: Optional[nx.DiGraph] = None,
        compartment_type: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Write skeleton to SWC file.
        
        Args:
            skeleton_points: (N, 3) array of XYZ coordinates in microns
            radii_um: (N,) array of radii in microns
            output_path: Path to output .swc file
            tree: Optional directed tree (if None, create simple chain)
            compartment_type: SWC type tag (0=undefined, 1=soma, 2=axon, 3=dendrite)
            metadata: Optional metadata to include in header
            
        Returns:
            Path to written file
        """
        self.logger.info(f"Writing SWC file: {output_path}")
        
        if len(skeleton_points) == 0:
            self.logger.error("Cannot write empty skeleton")
            raise ValueError("Skeleton is empty")
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("# SWC format skeleton\n")
            f.write("# Generated by VAST Neural Reconstruction Pipeline\n")
            
            if metadata:
                f.write(f"# Segment ID: {metadata.get('segment_id', 'unknown')}\n")
                f.write(f"# Segment name: {metadata.get('segment_name', 'unknown')}\n")
            
            f.write("# Columns: id, type, x, y, z, radius, parent_id\n")
            f.write("#\n")
            
            # Write nodes
            if tree is not None:
                # Use tree structure
                self._write_tree_swc(f, tree, radii_um, compartment_type)
            else:
                # Simple linear chain (fallback)
                self._write_linear_swc(f, skeleton_points, radii_um, compartment_type)
        
        self.logger.info(f"Wrote {len(skeleton_points)} skeleton points to {output_path}")
        
        return output_path
    
    def _write_tree_swc(
        self,
        f,
        tree: nx.DiGraph,
        radii_um: np.ndarray,
        compartment_type: int
    ):
        """Write SWC from tree structure (proper parent-child relationships)."""
        # Find root (node with no incoming edges)
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        
        if not roots:
            self.logger.warning("No root found in tree, using arbitrary node")
            roots = [list(tree.nodes())[0]]
        
        root = roots[0]
        
        # Traverse tree in breadth-first order to ensure parent_id < id
        node_to_swc_id = {}
        swc_id = 1
        
        queue = [(root, -1)]  # (node, parent_swc_id)
        
        while queue:
            node, parent_swc_id = queue.pop(0)
            
            # Assign SWC ID
            node_to_swc_id[node] = swc_id
            
            # Get position and radius
            pos = tree.nodes[node]['pos']
            radius = radii_um[node] if node < len(radii_um) else 1.0
            
            # Write SWC line
            f.write(f"{swc_id} {compartment_type} "
                   f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                   f"{radius:.6f} {parent_swc_id}\n")
            
            # Add children to queue
            for child in tree.successors(node):
                queue.append((child, swc_id))
            
            swc_id += 1
    
    def _write_linear_swc(
        self,
        f,
        skeleton_points: np.ndarray,
        radii_um: np.ndarray,
        compartment_type: int
    ):
        """Write SWC as simple linear chain (fallback when no tree structure)."""
        for i, (point, radius) in enumerate(zip(skeleton_points, radii_um)):
            swc_id = i + 1
            parent_id = i if i > 0 else -1  # Root has parent_id = -1
            
            f.write(f"{swc_id} {compartment_type} "
                   f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                   f"{radius:.6f} {parent_id}\n")


def quick_extract_skeleton(
    mask: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    bbox_min_vox: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick convenience function for skeleton extraction.
    
    Args:
        mask: Boolean 3D array
        voxel_size_um: Voxel dimensions in microns
        bbox_min_vox: Origin coordinates
        
    Returns:
        Tuple of (skeleton_points, radii_um)
    """
    extractor = SkeletonExtractor()
    skeleton_points, radii_um, _ = extractor.extract_skeleton(
        mask, voxel_size_um, bbox_min_vox
    )
    return skeleton_points, radii_um


def quick_write_swc(
    skeleton_points: np.ndarray,
    radii_um: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Quick convenience function for SWC writing.
    
    Args:
        skeleton_points: (N, 3) array of XYZ coordinates in microns
        radii_um: (N,) array of radii in microns
        output_path: Path to output file
        metadata: Optional metadata
        
    Returns:
        Path to written file
    """
    writer = SWCWriter()
    return writer.write_swc(
        skeleton_points, radii_um, output_path,
        metadata=metadata
    )