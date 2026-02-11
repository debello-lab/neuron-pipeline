
import numpy as np
import struct
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import networkx as nx
from skimage import morphology
from scipy import ndimage
import pandas as pd
import time
import socket


from extract_surfaces import SegmentSurfaceExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VAST_Pipeline")

class Skeletonizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def skeletonize_volume(self, volume: np.ndarray, resolution: Tuple[float, float, float]) -> nx.Graph:
        """
        Skeletonize a binary volume and return a NetworkX graph.
        volume: 3D numpy array (binary)
        resolution: (x, y, z) voxel size in physical units
        """
        logger.info("Skeletonizing volume...")
        skeleton = morphology.skeletonize(volume)
        
        # Convert skeleton voxels to graph
        # This is a non-trivial step. 
        # A simple approach: 
        # 1. Get all True voxel coordinates.
        # 2. Add edges between adjacent True voxels.
        # 3. Prune/Smooth?
        
        z, y, x = np.nonzero(skeleton)
        nodes = np.column_stack((x, y, z)) # Local coords
        
        if len(nodes) == 0:
            return nx.Graph()

        G = nx.Graph()
        # Add nodes with physical coordinates
        for i, (vx, vy, vz) in enumerate(nodes):
            G.add_node(i, x=vx*resolution[0], y=vy*resolution[1], z=vz*resolution[2], 
                       vx=vx, vy=vy, vz=vz) # Keep voxel coords for mapping
        
        # Add edges (26-connectivity)
        # Using KDTree for fast neighbor finding is one way, or direct array check.
        # Since we have the voxel coords, we can iterate.
        # Optimization: usage of skan or similar library is best, but we'll stick to basic loop or scipy.
        
        # Let's use a dictionary map for voxel -> node_id
        vox_to_id = {tuple(n): i for i, n in enumerate(nodes)}
        
        for i, (vx, vy, vz) in enumerate(nodes):
            # Check 26 neighbors
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0: continue
                        neighbor = (vx+dx, vy+dy, vz+dz)
                        if neighbor in vox_to_id:
                            j = vox_to_id[neighbor]
                            if i < j: # Avoid duplicate edges
                                dist = np.sqrt((dx*resolution[0])**2 + 
                                               (dy*resolution[1])**2 + 
                                               (dz*resolution[2])**2)
                                G.add_edge(i, j, weight=dist)
        
        return G

    def save_swc(self, graph: nx.Graph, filename: str, soma_node=None):
        """Save graph to SWC format."""
        # Simple DFS/BFS to order nodes for SWC (parent < id)
        # If no soma, pick random or specific start.
        # For separate components, we might need multiple trees or arbitrary roots.
        
        lines = []
        # SWC: id type x y z radius parent
        # We need to reindex nodes 1..N
        
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        
        node_counter = 1
        
        with open(self.output_dir / filename, 'w') as f:
            f.write("# VAST Export\n")
            f.write("# id type x y z radius parent\n")
            
            for comp in components:
                # Pick a root
                root = list(comp.nodes())[0] # TODO: heuristics for soma
                
                # BFS to order
                tree_edges = list(nx.bfs_edges(comp, root))
                tree_nodes = [root] + [v for u, v in tree_edges]
                
                # Map old ID to new ID
                old_to_new = {}
                
                for node in tree_nodes:
                    new_id = node_counter
                    old_to_new[node] = new_id
                    node_counter += 1
                    
                    data = comp.nodes[node]
                    
                    # Find parent
                    parent_id = -1
                    # In BFS edges (u, v), u is parent of v
                    # We need to look up who is the parent in our traversal
                    # Or just use the bfs_predecessors
                    pass # TODO: fix parent logic
                
                # Re-do with traversal that gives parents
                preds = dict(nx.bfs_predecessors(comp, root))
                
                for node in tree_nodes:
                    new_id = old_to_new[node]
                    parent_old = preds.get(node)
                    parent_new = old_to_new[parent_old] if parent_old is not None else -1
                    
                    data = comp.nodes[node]
                    # type 0=undef, 1=soma, 2=axon, 3=dendrite, 7=custom
                    f.write(f"{new_id} 0 {data['x']:.3f} {data['y']:.3f} {data['z']:.3f} 1.0 {parent_new}\n")


def main():
    print("main")


if __name__ == "__main__":
    main()