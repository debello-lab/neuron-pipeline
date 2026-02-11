
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

# Import existing modules
from VastControlClass_exporting import VASTControlClass
from extract_surfaces import SegmentSurfaceExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VAST_Pipeline")

class VASTClient(VASTControlClass):
    """
    Extended VAST Control Class with Annotation support.
    """
    def __init__(self):
        super().__init__()

    def get_anno_layer_count(self, layer_nr: int) -> int:
        """Get number of objects in an annotation layer."""
        payload = self.bytes_from_uint32(layer_nr)
        msg_type, data = self.send_command(self.GETANNOLAYERNROFOBJECTS, payload)
        if msg_type != 1:
            return -1
        parsed = self.parse_payload(data)
        if parsed['uints']:
            return parsed['uints'][0]
        return 0

    def get_anno_layer_objects(self, layer_nr: int) -> List[Dict[str, Any]]:
        """
        Get all objects from an annotation layer.
        Note: The API for GETANNOLAYEROBJECTDATA (71) is not fully documented in the provided file.
        I will assume it returns a list of object properties.
        """
        # This is a placeholder. Without the exact API spec for command 71, 
        # we might need to iterate or parse a complex blob.
        # For now, let's try to get object NAMES as a proxy for existence.
        
        objects = []
        count = self.get_anno_layer_count(layer_nr)
        if count <= 0:
            return objects

        # Try Getting Names
        payload = self.bytes_from_uint32(layer_nr)
        msg_type, data = self.send_command(self.GETANNOLAYEROBJECTNAMES, payload)
        if msg_type == 1:
            parsed = self.parse_payload(data)
            names = parsed['text']
            # We assume IDs are 0..Count-1 or we need to find how IDs are returned.
            # Usually VAST lists match the count.
            for i, name in enumerate(names):
                 objects.append({'id': i, 'name': name, 'layer_nr': layer_nr})
        
        return objects

    def get_annotation_object_data(self, layer_nr: int, object_id: int) -> Optional[Dict]:
        """
        Get data for a specific annotation object.
        Command 71/90? 
        The provided file lists GETANNOOBJECT = 90. Let's try that.
        """
        # Payload: likely layer_nr + object_id? or just object_id (unique)?
        # VAST Lite usually uses ID.
        # Let's assume Unique ID if possible, or Layer+Index.
        # If we use GETANNOOBJECT (90):
        # Implementation depends on API.
        pass

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



def list_segments(client: VASTClient):
    """List all segments with their basic info."""
    logger.info("Fetching segment list...")
    # Get all names
    names = client.get_all_segment_names()
    if not names:
        logger.warning("No segment names found.")
        return

    # Get all data (for bounding boxes/sizes)
    # Note: get_all_segment_data might be heavy if many segments, but usually fine.
    seg_data = client.get_all_segment_data()
    
    # Create a map of ID -> Data
    data_map = {item['id']: item for item in seg_data} if seg_data else {}
    
    logger.info(f"Found {len(names)} segments (including background).")
    
    print(f"{'ID':<5} {'Name':<30} {'BBox Vol (Approx)':<15} {'Children'}")
    print("-" * 70)
    
    count = 0
    for i, name in enumerate(names):
        if i == 0: continue # Skip background
        
        info = data_map.get(i)
        bbox_vol = "N/A"
        children = "N/A"
        
        if info:
            bb = info['boundingbox'] # x1,y1,z1, x2,y2,z2
            vol = (bb[3]-bb[0]) * (bb[4]-bb[1]) * (bb[5]-bb[2])
            bbox_vol = f"{vol:,}"
            children = info['hierarchy'][1] # child
            
        # Only print valid ones or first 20
        if info and vol > 0:
            print(f"{i:<5} {name[:30]:<30} {bbox_vol:<15} {children}")
            count += 1
            if count > 20:
                print("... (more segments exist)")
                break

def main():
    try:
        
        # Connect to examine segments
        client = VASTClient()
        if client.connect("127.0.0.1", 22081, 10):
            list_segments(client)
            client.disconnect()
            
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()