"""
Voxel and Mesh Cleaning Operations
Neural Reconstruction Pipeline

This module provides cleaning operations for both voxel masks and surface meshes:
- Remove noise components (specks from accidental clicks)
- Fill internal holes
- Smooth surfaces
- Remove spurious branches

Author: Nicolas Randazzo
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
import logging


class VoxelCleaner:
    """
    Clean binary voxel masks for skeletonization.
    
    Removes artifacts that would corrupt skeleton quality:
    - Small disconnected components (specks from annotation errors)
    - Internal holes (can create false branches)
    - Surface roughness (optional smoothing)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the cleaner.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def clean_mask(
        self,
        mask: np.ndarray,
        min_component_voxels: int = 5000,
        keep_largest_only: bool = True,
        fill_holes: bool = True,
        smooth_iterations: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Clean a binary voxel mask.
        
        Args:
            mask: Boolean 3D array (Z, Y, X)
            min_component_voxels: Minimum size for components to keep (if keep_largest_only=False)
            keep_largest_only: If True, keep only the largest component (recommended for single neurons)
            fill_holes: Fill internal holes in the mask
            smooth_iterations: Number of morphological smoothing iterations (0 = no smoothing)
            
        Returns:
            Tuple of (cleaned_mask, stats)
            - cleaned_mask: Boolean 3D array
            - stats: Dictionary with cleaning statistics
        """
        self.logger.info("Starting mask cleaning...")
        
        original_voxel_count = np.sum(mask)
        
        stats = {
            'original_voxel_count': int(original_voxel_count),
            'original_components': 0,
            'kept_components': 0,
            'removed_components': 0,
            'holes_filled': False,
            'smoothing_iterations': smooth_iterations,
            'final_voxel_count': 0,
            'voxels_added': 0,
            'voxels_removed': 0
        }
        
        # Step 1: Component filtering
        self.logger.info("Identifying connected components...")
        labeled_mask = label(mask, connectivity=1)  # 6-connectivity
        num_components = labeled_mask.max()
        stats['original_components'] = num_components
        
        if num_components == 0:
            self.logger.warning("Input mask is empty")
            return mask.copy(), stats
        
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0  # Exclude background
        
        self.logger.info(f"Found {num_components} connected components")
        self.logger.info(f"Component sizes: min={component_sizes[1:].min():,}, "
                        f"max={component_sizes[1:].max():,}, "
                        f"median={int(np.median(component_sizes[1:])):,}")
        
        # Determine which components to keep
        if keep_largest_only:
            largest_component_id = component_sizes.argmax()
            keep_ids = [largest_component_id]
            self.logger.info(f"Keeping only largest component (ID {largest_component_id}, "
                           f"{component_sizes[largest_component_id]:,} voxels)")
        else:
            keep_ids = np.where(component_sizes >= min_component_voxels)[0]
            self.logger.info(f"Keeping {len(keep_ids)} components with >={min_component_voxels:,} voxels")
        
        cleaned_mask = np.isin(labeled_mask, keep_ids)
        
        stats['kept_components'] = len(keep_ids)
        stats['removed_components'] = num_components - len(keep_ids)
        
        # Step 2: Fill holes
        if fill_holes:
            self.logger.info("Filling internal holes...")
            pre_fill_voxels = np.sum(cleaned_mask)
            cleaned_mask = binary_fill_holes(cleaned_mask)
            post_fill_voxels = np.sum(cleaned_mask)
            voxels_added = post_fill_voxels - pre_fill_voxels
            
            stats['holes_filled'] = True
            stats['voxels_added'] = int(voxels_added)
            
            if voxels_added > 0:
                self.logger.info(f"Filled holes: +{int(voxels_added):,} voxels")
        
        # Step 3: Morphological smoothing (optional)
        if smooth_iterations > 0:
            self.logger.info(f"Applying {smooth_iterations} smoothing iterations...")
            pre_smooth_voxels = np.sum(cleaned_mask)
            
            # Opening operation (erosion + dilation) to smooth surface
            for i in range(smooth_iterations):
                cleaned_mask = binary_erosion(cleaned_mask)
                cleaned_mask = binary_dilation(cleaned_mask)
            
            post_smooth_voxels = np.sum(cleaned_mask)
            voxel_change = post_smooth_voxels - pre_smooth_voxels
            
            self.logger.info(f"Smoothing changed voxel count by {voxel_change:+,}")
        
        final_voxel_count = np.sum(cleaned_mask)
        stats['final_voxel_count'] = int(final_voxel_count)
        stats['voxels_removed'] = int(original_voxel_count - final_voxel_count + stats['voxels_added'])
        
        self.logger.info(f"Cleaning complete: {int(original_voxel_count):,} -> {int(final_voxel_count):,} voxels "
                        f"({100.0 * final_voxel_count / original_voxel_count:.1f}% retained)")
        
        return cleaned_mask, stats
    
    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Quick utility to extract only the largest connected component.
        
        Args:
            mask: Boolean 3D array
            
        Returns:
            Boolean 3D array with only largest component
        """
        labeled_mask = label(mask, connectivity=1)
        if labeled_mask.max() == 0:
            return mask.copy()
        
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        largest_id = component_sizes.argmax()
        
        return labeled_mask == largest_id


class MeshCleaner:
    """
    Clean surface meshes (vertices and faces).
    
    Operations:
    - Remove duplicate vertices
    - Remove degenerate faces
    - Remove isolated vertices
    - Optionally decimate mesh
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the cleaner.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def clean_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        remove_duplicates: bool = True,
        remove_degenerate: bool = True,
        remove_isolated: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Clean a surface mesh.
        
        Args:
            vertices: (N, 3) array of vertex coordinates
            faces: (M, 3) array of triangle indices
            remove_duplicates: Remove duplicate vertices and remap faces
            remove_degenerate: Remove degenerate triangles (zero area)
            remove_isolated: Remove vertices not referenced by any face
            
        Returns:
            Tuple of (cleaned_vertices, cleaned_faces, stats)
        """
        self.logger.info("Starting mesh cleaning...")
        
        stats = {
            'original_vertices': len(vertices),
            'original_faces': len(faces),
            'duplicate_vertices_removed': 0,
            'degenerate_faces_removed': 0,
            'isolated_vertices_removed': 0,
            'final_vertices': 0,
            'final_faces': 0
        }
        
        clean_verts = vertices.copy()
        clean_faces = faces.copy()
        
        # Step 1: Remove degenerate faces
        if remove_degenerate:
            self.logger.info("Removing degenerate faces...")
            # Face is degenerate if any two vertices are the same
            valid_faces = (
                (clean_faces[:, 0] != clean_faces[:, 1]) &
                (clean_faces[:, 1] != clean_faces[:, 2]) &
                (clean_faces[:, 2] != clean_faces[:, 0])
            )
            
            degenerate_count = len(clean_faces) - np.sum(valid_faces)
            clean_faces = clean_faces[valid_faces]
            stats['degenerate_faces_removed'] = int(degenerate_count)
            
            if degenerate_count > 0:
                self.logger.info(f"Removed {degenerate_count} degenerate faces")
        
        # Step 2: Remove duplicate vertices
        if remove_duplicates:
            self.logger.info("Removing duplicate vertices...")
            # Find unique vertices and create remapping
            unique_verts, inverse_indices = np.unique(
                clean_verts, axis=0, return_inverse=True
            )
            
            duplicate_count = len(clean_verts) - len(unique_verts)
            
            if duplicate_count > 0:
                # Remap face indices
                clean_faces = inverse_indices[clean_faces]
                clean_verts = unique_verts
                stats['duplicate_vertices_removed'] = int(duplicate_count)
                self.logger.info(f"Removed {duplicate_count} duplicate vertices")
        
        # Step 3: Remove isolated vertices
        if remove_isolated:
            self.logger.info("Removing isolated vertices...")
            # Find which vertices are actually used
            used_vertices = np.unique(clean_faces.ravel())
            
            isolated_count = len(clean_verts) - len(used_vertices)
            
            if isolated_count > 0:
                # Create mapping from old to new indices
                new_indices = np.full(len(clean_verts), -1, dtype=np.int64)
                new_indices[used_vertices] = np.arange(len(used_vertices))
                
                # Remap faces and extract used vertices
                clean_faces = new_indices[clean_faces]
                clean_verts = clean_verts[used_vertices]
                
                stats['isolated_vertices_removed'] = int(isolated_count)
                self.logger.info(f"Removed {isolated_count} isolated vertices")
        
        stats['final_vertices'] = len(clean_verts)
        stats['final_faces'] = len(clean_faces)
        
        self.logger.info(f"Mesh cleaning complete:")
        self.logger.info(f"  Vertices: {stats['original_vertices']:,} -> {stats['final_vertices']:,}")
        self.logger.info(f"  Faces: {stats['original_faces']:,} -> {stats['final_faces']:,}")
        
        return clean_verts, clean_faces, stats


def quick_clean_mask(
    mask: np.ndarray,
    keep_largest: bool = True,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Quick convenience function for basic mask cleaning.
    
    Args:
        mask: Boolean 3D array
        keep_largest: Keep only largest component
        fill_holes: Fill internal holes
        
    Returns:
        Cleaned boolean mask
    """
    cleaner = VoxelCleaner()
    cleaned, _ = cleaner.clean_mask(
        mask,
        keep_largest_only=keep_largest,
        fill_holes=fill_holes
    )
    return cleaned


def quick_clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick convenience function for basic mesh cleaning.
    
    Args:
        vertices: (N, 3) vertex array
        faces: (M, 3) face array
        
    Returns:
        Tuple of (cleaned_vertices, cleaned_faces)
    """
    cleaner = MeshCleaner()
    clean_verts, clean_faces, _ = cleaner.clean_mesh(vertices, faces)
    return clean_verts, clean_faces