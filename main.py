import numpy as np
from skimage import measure
from scipy import ndimage, sparse
from math import floor, ceil
import re
from typing import Tuple, List, Dict, Optional
import time
import logging
from VASTControlClass import VASTControlClass


# class SurfaceExtractor:
#     """
#         Python port of MATLAB function extractsurfaces from VAST.
#         Extracts 3D surfaces from segmentation or screenshot data.
#     """

#     def __init__(self, vast_control: VASTControlClass, export_params: dict, region_params: dict):
#         """
#         Initialize the surface extractor.
        
#         Args:
#             vast_control: VASTControlClass instance for data access
#             export_params: Dictionary containing export configuration
#             region_params: Dictionary containing region bounds (xmin, xmax, ymin, ymax, zmin, zmax)
#         """
#         self.vast = vast_control
#         self.export_params = export_params
#         self.region_params = region_params
#         self.canceled = False
        
#         # Will be populated during extraction
#         self.param = {**export_params}
#         self.objects = None
#         self.names = []
#         self.data = None

#     def _setup_screenshot_names(self, param: dict) -> List[str]:
#         """Setup names for screenshot extraction modes"""
#         extract_which = param.get('extractwhich', 5)
        
#         if extract_which == 5:  # RGB 50%
#             return ['Red Layer', 'Green Layer', 'Blue Layer']
#         elif extract_which == 6:  # Brightness 50%
#             param['lev'] = 128
#             return ['Brightness 128']
#         elif extract_which == 7:  # 16 levels
#             param['lev'] = list(range(8, 257, 16))
#             return [f'B{lev:03d}' for lev in param['lev']]
#         elif extract_which == 8:  # 32 levels
#             param['lev'] = list(range(4, 257, 8))
#             return [f'B{lev:03d}' for lev in param['lev']]
#         elif extract_which == 9:  # 64 levels
#             param['lev'] = list(range(2, 257, 4))
#             return [f'B{lev:03d}' for lev in param['lev']]
#         else:
#             return []

#     def _get_child_tree_ids_seg(self, data: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
#         """
#         Recursively get all child IDs in the segment hierarchy
        
#         Args:
#             data: Segment data matrix
#             parent_ids: Array of parent segment indices
            
#         Returns:
#             Array of all child indices
#         """
#         children = []
#         for pid in parent_ids:
#             # Find all segments where parent (column 14/index 13) equals this segment's ID
#             child_mask = data[:, 13] == data[pid, 0]
#             child_indices = np.where(child_mask)[0]
#             if len(child_indices) > 0:
#                 children.extend(child_indices)
#                 # Recursively get children of children
#                 children.extend(self._get_child_tree_ids_seg(data, child_indices))
        
#         return np.array(children, dtype=int) if children else np.array([], dtype=int)
 
#     def _expand_bounding_boxes(self, bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
#         """
#         Expand bounding boxes to encompass both input boxes
        
#         Args:
#             bbox1: First bounding box array (N x 6)
#             bbox2: Second bounding box array (N x 6)
            
#         Returns:
#             Expanded bounding boxes
#         """
#         if bbox1.ndim == 1:
#             bbox1 = bbox1.reshape(1, -1)
#         if bbox2.ndim == 1:
#             bbox2 = bbox2.reshape(1, -1)
        
#         result = bbox1.copy()
        
#         # For uninitialized boxes (-1), use bbox2
#         uninit_mask = bbox1[:, 0] == -1
#         result[uninit_mask] = bbox2[uninit_mask]
        
#         # For initialized boxes, take min/max
#         init_mask = ~uninit_mask
#         if np.any(init_mask):
#             result[init_mask, 0] = np.minimum(bbox1[init_mask, 0], bbox2[init_mask, 0])
#             result[init_mask, 1] = np.minimum(bbox1[init_mask, 1], bbox2[init_mask, 1])
#             result[init_mask, 2] = np.minimum(bbox1[init_mask, 2], bbox2[init_mask, 2])
#             result[init_mask, 3] = np.maximum(bbox1[init_mask, 3], bbox2[init_mask, 3])
#             result[init_mask, 4] = np.maximum(bbox1[init_mask, 4], bbox2[init_mask, 4])
#             result[init_mask, 5] = np.maximum(bbox1[init_mask, 5], bbox2[init_mask, 5])
        
#         return result

#     def _get_block_mesh(self, seg_id: int, tx: int, ty: int, tz: int, param: dict) -> Tuple[np.ndarray, np.ndarray]:
#         """Retrieve mesh for specific block from storage."""
#         key = (seg_id, tx, ty, tz)
#         f = param['farray'].get(key, np.array([]))
#         v = param['varray'].get(key, np.array([]))

#         # Return copies to avoid modifying originals during merge
#         if f is not None and len(f) > 0:
#             return f.copy(), v.copy()
#         return np.array([]), np.array([])

#     def _merge_object_meshes(self, seg_id: int, param: dict) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Merge all block meshes for a single segment using X→Y→Z consolidation.
#         Based on MATLAB extractsurfaces.m lines 827-891.

#         Returns:
#             (faces, vertices) - Final consolidated mesh
#         """
#         nr_x_tiles = param['nr_x_tiles']
#         nr_y_tiles = param['nr_y_tiles']
#         nr_z_tiles = param['nr_z_tiles']

#         # Storage for incremental Z-merge optimization
#         completed_faces = []
#         completed_verts = []
#         vert_offset = 0

#         # Initialize final mesh variables
#         final_f, final_v = None, None

#         # Z-axis loop (iterate through planes)
#         for tz in range(nr_z_tiles):
#             plane_f, plane_v = None, None

#             # Y-axis loop (iterate through rows in plane)
#             for ty in range(nr_y_tiles):
#                 row_f, row_v = None, None

#                 # X-axis loop (merge blocks in row)
#                 for tx in range(nr_x_tiles):
#                     block_f, block_v = self._get_block_mesh(seg_id, tx, ty, tz, param)

#                     if len(block_v) == 0:
#                         continue

#                     if row_v is None:
#                         row_f, row_v = block_f, block_v
#                     else:
#                         row_f, row_v = self._merge_meshes(row_f, row_v, block_f, block_v)

#                 # Merge row into plane
#                 if row_v is not None and len(row_v) > 0:
#                     if plane_v is None:
#                         plane_f, plane_v = row_f, row_v
#                     else:
#                         plane_f, plane_v = self._merge_meshes(plane_f, plane_v, row_f, row_v)

#             # Merge plane into final mesh
#             if plane_v is not None and len(plane_v) > 0:
#                 if tz == 0:
#                     final_f, final_v = plane_f, plane_v
#                 else:
#                     final_f, final_v = self._merge_meshes(final_f, final_v, plane_f, plane_v)

#                     # OPTIMIZATION: Extract completed portion (MATLAB lines 879-888)
#                     # Vertices at max Z won't be merged with future planes
#                     if final_v is not None and len(final_v) > 0 and tz < nr_z_tiles - 1:
#                         max_z = np.max(final_v[:, 2])
#                         z_threshold = max_z - 0.01  # Small epsilon for floating point

#                         # Find first vertex below threshold
#                         vcut_candidates = np.where(final_v[:, 2] < z_threshold)[0]
#                         if len(vcut_candidates) > 0:
#                             vcut = vcut_candidates[0]

#                             # Find first face using vertex >= vcut
#                             face_uses_later = np.any(final_f >= vcut, axis=1)
#                             fcut_candidates = np.where(face_uses_later)[0]

#                             if len(fcut_candidates) > 0 and vcut > 0:
#                                 fcut = fcut_candidates[0]

#                                 # Extract completed portion
#                                 completed_verts.append(final_v[:vcut, :])
#                                 completed_faces.append(final_f[:fcut, :] + vert_offset)

#                                 # Keep only incomplete portion
#                                 vert_offset += vcut
#                                 final_v = final_v[vcut:, :]
#                                 final_f = final_f[fcut:, :] - vcut

#         # Combine completed + final portions
#         if final_v is None or len(final_v) == 0:
#             return np.array([]), np.array([])

#         if len(completed_verts) > 0:
#             all_verts = np.vstack(completed_verts + [final_v])
#             all_faces = np.vstack(completed_faces + [final_f + vert_offset])
#         else:
#             all_verts = final_v
#             all_faces = final_f

#         return all_faces, all_verts

#     def _get_segment_colors(self, param: dict, data: np.ndarray) -> np.ndarray:
#         """Extract RGB colors for segments from VAST data matrix."""
#         colors = np.zeros((len(param['objects']), 3), dtype=np.uint8)

#         if param.get('objectcolors', 1) == 1:
#             # Use VAST segment colors
#             for i, seg_id in enumerate(param['objects'][:, 0]):
#                 seg_idx = np.where(data[:, 0] == seg_id)[0]
#                 if len(seg_idx) > 0:
#                     # Get collapse target segment to inherit color
#                     inherit_idx = int(data[seg_idx[0], 17])  # Column 18 (0-indexed)
#                     if inherit_idx < len(data):
#                         # RGB in columns 3-5 (0-indexed: 2-4)
#                         colors[i, :] = data[inherit_idx, 2:5].astype(np.uint8)
#         else:
#             # Fallback: random colors
#             colors = np.random.randint(0, 255, (len(param['objects']), 3), dtype=np.uint8)

#         return colors

#     def _sanitize_filename(self, name: str) -> str:
#         """Remove invalid filesystem characters."""
#         invalid_chars = ' ?*\\/|:"<>'
#         for char in invalid_chars:
#             name = name.replace(char, '_')
#         return name

#     def _apply_output_transforms(self, vertices: np.ndarray, param: dict) -> np.ndarray:
#         """Apply Z-inversion and output offset."""
#         v = vertices.copy()

#         if param.get('invertz', 0) == 1:
#             v[:, 2] = -v[:, 2]

#         v[:, 0] += param.get('outputoffsetx', 0)
#         v[:, 1] += param.get('outputoffsety', 0)
#         v[:, 2] += param.get('outputoffsetz', 0)

#         return v

#     def _write_obj_with_mtl(self, vertices, faces, obj_path, mtl_filename,
#                             material_name, object_name, invert_normals=False):
#         """Write OBJ file with MTL link."""
#         with open(obj_path, 'w') as f:
#             f.write(f"mtllib {mtl_filename}\n")
#             f.write(f"usemtl {material_name}\n")

#             # Write vertices
#             for v in vertices:
#                 f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

#             f.write(f"g {object_name}\n")

#             # Write faces (convert to 1-based indexing)
#             for face in faces:
#                 if invert_normals:
#                     f.write(f"f {int(face[1])+1} {int(face[0])+1} {int(face[2])+1}\n")
#                 else:
#                     f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")

#             f.write("g\n")

#     def _write_mtl(self, mtl_path, material_name, color_rgb):
#         """Write MTL material file."""
#         color = color_rgb / 255.0  # Normalize to [0,1]

#         with open(mtl_path, 'w') as f:
#             f.write(f"newmtl {material_name}\n")
#             f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
#             f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
#             f.write(f"Ks 0.5 0.5 0.5\n")
#             f.write(f"d 1.000\n")
#             f.write(f"Ns 32\n")

#     def export_meshes(self):
#         """
#         Phase 5-6: Merge block meshes and write OBJ files.
#         Call this after extract_surfaces() completes.
#         """
#         import os

#         param = self.param

#         # Create output directory
#         os.makedirs(param['targetfolder'], exist_ok=True)

#         # Get segment colors
#         print("Extracting segment colors...")
#         colors = self._get_segment_colors(param, self.data)

#         # Process each object
#         total_objects = len(param['objects'])

#         for seg_nr, seg_id in enumerate(param['objects'][:, 0]):
#             if self.canceled:
#                 break

#             seg_id = int(seg_id)
#             print(f"Processing object {seg_nr + 1}/{total_objects}: "
#                   f"{self.names[seg_id] if seg_id < len(self.names) else seg_id}")

#             # Merge all blocks for this segment
#             print("  Merging blocks...")
#             merged_faces, merged_verts = self._merge_object_meshes(seg_id, param)

#             if len(merged_verts) == 0:
#                 print("  (empty - skipping)")
#                 continue

#             # Apply output transformations
#             merged_verts = self._apply_output_transforms(merged_verts, param)

#             # Generate filenames
#             obj_name = self._sanitize_filename(self.names[seg_id])
#             prefix = param.get('targetfileprefix', 'Segment_')
#             obj_filename = f"{prefix}{seg_id:04d}_{obj_name}.obj"
#             mtl_filename = f"{prefix}{seg_id:04d}_{obj_name}.mtl"
#             material_name = f"mat_{seg_id:04d}"

#             obj_path = os.path.join(param['targetfolder'], obj_filename)
#             mtl_path = os.path.join(param['targetfolder'], mtl_filename)

#             # Write OBJ file
#             print(f"  Writing {obj_filename}")
#             invert = (param.get('invertz', 0) == 0)  # Invert normals if NOT inverting Z
#             self._write_obj_with_mtl(merged_verts, merged_faces, obj_path, mtl_filename,
#                                      material_name, obj_name, invert_normals=invert)

#             # Write MTL file
#             self._write_mtl(mtl_path, material_name, colors[seg_nr])

#             print(f"  ✓ {len(merged_verts)} vertices, {len(merged_faces)} faces")

#         print(f"\n{'='*60}")
#         print(f"Export complete! {seg_nr + 1} objects written to:")
#         print(f"  {param['targetfolder']}")
#         print(f"{'='*60}")

#     def _merge_meshes(self, f1, v1, f2, v2):
#         """
#         Merge two meshes defined by (f1, v1) and (f2, v2).
        
#         Merges meshes by identifying and merging duplicate vertices in the overlapping region,
#         then re-indexing face indices appropriately.
        
#         Args:
#             f1: Face indices array for mesh 1 (Nx3)
#             v1: Vertex coordinates array for mesh 1 (Mx3)
#             f2: Face indices array for mesh 2 (Nx3)
#             v2: Vertex coordinates array for mesh 2 (Mx3)
        
#         Returns:
#             tuple: (merged_faces, merged_vertices) where faces are re-indexed
#         """
#         # Handle empty meshes
#         if v1.size == 0:
#             return f2.copy(), v2.copy()
        
#         if v2.size == 0:
#             return f1.copy(), v1.copy()
        
#         # Convert to numpy arrays if needed
#         f1 = np.asarray(f1, dtype=np.int32)
#         v1 = np.asarray(v1, dtype=np.float32)
#         f2 = np.asarray(f2, dtype=np.int32)
#         v2 = np.asarray(v2, dtype=np.float32)
        
#         nrofvertices1 = v1.shape[0]
#         nrofvertices2 = v2.shape[0]
        
#         # Adjust f2 indices by number of vertices in v1
#         f2 = f2 + nrofvertices1
        
#         # Find overlapping vertex region
#         minv1 = np.min(v1, axis=0)
#         maxv1 = np.max(v1, axis=0)
#         minv2 = np.min(v2, axis=0)
#         maxv2 = np.max(v2, axis=0)
        
#         ovmin = np.maximum(minv1, minv2)
#         ovmax = np.minimum(maxv1, maxv2)
        
#         # Find vertices in overlap zone for v1
#         mask1 = ((v1[:, 0] >= ovmin[0]) & (v1[:, 0] <= ovmax[0]) &
#                 (v1[:, 1] >= ovmin[1]) & (v1[:, 1] <= ovmax[1]) &
#                 (v1[:, 2] >= ovmin[2]) & (v1[:, 2] <= ovmax[2]))
#         ov1_indices = np.where(mask1)[0]
        
#         # Find vertices in overlap zone for v2
#         mask2 = ((v2[:, 0] >= ovmin[0]) & (v2[:, 0] <= ovmax[0]) &
#                 (v2[:, 1] >= ovmin[1]) & (v2[:, 1] <= ovmax[1]) &
#                 (v2[:, 2] >= ovmin[2]) & (v2[:, 2] <= ovmax[2]))
#         ov2_indices = np.where(mask2)[0]
        
#         # If no overlap, concatenate meshes
#         if ov2_indices.size == 0:
#             f = np.vstack([f1, f2])
#             v = np.vstack([v1, v2])
#             return f, v
        
#         # Find matching vertices in overlapping regions
#         deletevertex = np.zeros(nrofvertices2, dtype=bool)
        
#         # Use faster loopless version with intersect equivalent
#         # Find common vertices in overlap zones
#         ov1_verts = v1[ov1_indices]
#         ov2_verts = v2[ov2_indices]
        
#         # Find intersecting vertices (common coordinates)
#         common_rows = []
#         i1a_list = []
#         i2a_list = []
        
#         for i, v_ov1_idx in enumerate(ov1_indices):
#             v_ov1 = v1[v_ov1_idx]
#             # Check if this vertex exists in ov2
#             for j, v_ov2_idx in enumerate(ov2_indices):
#                 v_ov2 = v2[v_ov2_idx]
#                 if np.allclose(v_ov1, v_ov2, atol=1e-6):
#                     i1a_list.append(v_ov1_idx)
#                     i2a_list.append(v_ov2_idx)
#                     common_rows.append(v_ov1)
#                     break
        
#         # Link duplicate vertices
#         if len(i2a_list) > 0:
#             i1a = np.array(i1a_list, dtype=np.int32)
#             i2a = np.array(i2a_list, dtype=np.int32)
            
#             # Remap f2 to use v1 vertices for common vertices
#             aov2 = i2a
#             kov2 = aov2 + nrofvertices1
            
#             # For each face in f2, replace vertices that match
#             for idx in range(len(aov2)):
#                 v1_idx = i1a[idx]
#                 v2_idx = kov2[idx]
#                 # Replace all occurrences of v2_idx in f2 with v1_idx
#                 f2[f2 == v2_idx] = v1_idx
#                 deletevertex[aov2[idx]] = True
        
#         # Re-index faces in f2 to account for deleted vertices
#         z = np.arange(nrofvertices1, dtype=np.int32)
#         z = np.append(z, np.zeros(nrofvertices2, dtype=np.int32))
        
#         zp = nrofvertices1
#         for sp in range(nrofvertices2):
#             if not deletevertex[sp]:
#                 z[nrofvertices1 + sp] = zp
#                 zp += 1
        
#         f2d = z[f2]
        
#         # Ensure f2d is 2D array
#         if f2d.ndim == 1:
#             f2d = f2d.reshape(1, -1)
        
#         # Delete unused vertices from v2 and concatenate
#         v2d = v2[~deletevertex]
        
#         f = np.vstack([f1, f2d])
#         v = np.vstack([v1, v2d])
        
#         return f, v

#     def extractsurfaces(self):



def main():
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect")
    
    # Get dataset info
    info = vast.get_info()
    if not info:
        print("No dataset loaded")
        vast.disconnect()
        return
    vast.disconnect()

    region_params = {
    'xmin': 0,
    'xmax': info['datasizex'] - 1,
    'ymin': 0,
    'ymax': info['datasizey'] - 1,
    'zmin': 0,
    'zmax': info['datasizez'] - 1,
    }

    export_params = {     
        # Mip level and sampling
        'miplevel': 2,  # 0=full res, higher=lower res
        'slicestep': 1,  # Use every nth slice
        
        # Mip region constraint (optional)
        'usemipregionconstraint': False,
        'mipregionmip': info['nrofmiplevels'] - 1,
        'mipregionpadding': 1,
        
        # Processing block size (use reasonable sizes that work at any mip level)
        'blocksizex': 512,
        'blocksizey': 512,
        'blocksizez': 64,
        'overlap': 1,
        
        # Scaling and units
        'xscale': 0.001,
        'yscale': 0.001,
        'zscale': 0.001,
        'xunit': info['voxelsizex'],  # nm
        'yunit': info['voxelsizey'],
        'zunit': info['voxelsizez'],
        
        # Output offset
        'outputoffsetx': 0,
        'outputoffsety': 0,
        'outputoffsetz': 0,
        
        # Options
        'invertz': True,
        'erodedilate': False,
        'closesurfaces': True,
        
        # Export mode
        'extractwhich': 3,  # 1=all segments uncollapsed, 2=collapsed, 
                           # 3=selected+children uncollapsed, 4=selected+children collapsed
                           # 5=RGB isosurfaces, 6=brightness isosurface, 
                           # 7/8/9=multi-level brightness, 10=one per color
        
        # File output
        'targetfileprefix': 'Segment_',
        'targetfolder': './vast_export',
        'fileformat': 1,  # 1=OBJ/MTL, 2=PLY
        'includefoldernames': True,
        'objectcolors': 1,  # 1=VAST colors, 2=volume-based colormap
        'max_object_number': 100,
        
        # Advanced
        'skipmodelgeneration': False,
        'disablenetwarnings': True,
        'write3dsmaxloader': False,
        'savesurfacestats': False,
        'surfacestatsfile': 'surfacestats.txt',
    }
    # extractor = SurfaceExtractor(vast, export_params, region_params)

    # Phase 1-4: Extract surfaces from blocks
    print("Phase 1-4: Extracting surfaces from blocks...")
    # extractor.extract_surfaces()

    # Check for cancellation
    # if extractor.canceled:
    #     print("Extraction canceled by user")
    #     vast.disconnect()
    #     return

    # Phase 5-6: Merge and export
    print("\nPhase 5-6: Merging meshes and writing files...")
    # extractor.export_meshes()

    print("\nAll done!")
    vast.disconnect()


if __name__ == "__main__":
    main()
    
    main()