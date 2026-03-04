"""
Surface Extraction for VAST Segmentation Data

This module extracts 3D surface meshes from individual segments in VAST datasets.
Designed for neurons ranging from thousands to tens of millions of voxels.

Functions for extracting obj/ply meshes as well as voxel masks for skeletonization.

Author: Nicolas Randazzo
"""

import numpy as np
from skimage import measure
from typing import Optional, Tuple, Dict, Any
import sys
import logging
from datetime import datetime
from pathlib import Path
from vastpy.control.exporting import VASTControlClass


class SegmentSurfaceExtractor:
    """
    Robust single-segment surface extraction with memory-efficient block processing.
    
    Key features:
    - Handles segments from 7K to 55M+ voxels
    - Automatic memory management and block sizing
    - Preserves morphological features for skeletonization
    - Comprehensive logging and error handling
    - Production-grade quality assurance
    """
    
    def __init__(self, vast: VASTControlClass, output_dir: str = "./output"):
        """
        Initialize the extractor.
        
        Args:
            vast: Connected VASTControlClass instance
            output_dir: Base directory for output files
        """
        self.vast = vast
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Cache dataset info
        self.dataset_info: Dict[str, Any] = None  # type: ignore
        self._load_dataset_info()
        assert self.dataset_info is not None, "Dataset info must be loaded"
        
        # Memory management settings (can be tuned)
        self.max_block_voxels = 512 ** 3  # ~134M voxels max per block (conservative)
        self.block_overlap = 2  # Voxels of overlap for mesh stitching
        
    def _setup_logging(self):
        """Configure logging with both file and console output."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"extraction_{timestamp}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"SurfaceExtractor_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers if re-instantiated
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler - detailed logs
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fh_formatter)
        
        # Console handler - important info only
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(ch_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Logging initialized: {log_file}")
    
    def _load_dataset_info(self):
        """Load and cache dataset information from VAST."""
        self.logger.info("Loading dataset information...")
        
        info = self.vast.get_info()
        if not info:
            raise RuntimeError("Failed to get dataset info from VAST. Is a dataset loaded?")
        
        self.dataset_info = info
        self.logger.info(f"Dataset: {info['datasizex']}x{info['datasizey']}x{info['datasizez']} voxels")
        self.logger.info(f"Voxel size: {info['voxelsizex']:.3f} x {info['voxelsizey']:.3f} x {info['voxelsizez']:.3f} nm")
        
        # Get MIP scale factors for segmentation layer
        layer_info  = self.vast.get_selected_layernr()
        if layer_info is not None:
            selected_layer, selected_em_layer, selected_segment_layer = layer_info
            if selected_layer:
                seg_layer = selected_layer
                self.mip_factors = self.vast.get_mipmap_scale_factors(seg_layer)
                self.logger.debug(f"Segmentation layer: {seg_layer}")
            else:
                self.mip_factors = None
                self.logger.warning("Could not retrieve MIP scale factors")
        
    
    def _set_socket_timeout(self, timeout_seconds):
        """Temporarily change socket timeout for large data transfers."""
        if self.vast.client_socket:
            self.vast.client_socket.settimeout(timeout_seconds)

    def get_segment_metadata(self, segment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a segment.
        
        Args:
            segment_id: Segment ID to query
            
        Returns:
            Dictionary with segment metadata or None if not found
        """
        self.logger.debug(f"Retrieving metadata for segment {segment_id}")
        
        seg_data = self.vast.get_segment_data(segment_id)
        if seg_data is None:
            self.logger.error(f"Segment {segment_id} not found in dataset")
            return None
        
        bbox = seg_data['boundingbox']
        
        # Check for valid/empty segment
        if bbox[0] < 0 or bbox[3] < 0:
            self.logger.warning(f"Segment {segment_id} has invalid bounding box - likely empty")
            return None
        
        # Get segment name if available
        name = self.vast.get_segment_name(segment_id)
        if name is None:
            name = f"Segment_{segment_id}"
        
        if 'col1' in seg_data:
            col1 = seg_data['col1']
            r = (col1 >> 0) & 0xFF
            g = (col1 >> 8) & 0xFF
            b = (col1 >> 16) & 0xFF
            color = np.array([r, g, b], dtype=np.uint8)
        elif 'color' in seg_data:
            color = seg_data['color']
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)
        self.logger.debug(f"Segment {segment_id} color: RGB{tuple(color)}")
        
        metadata = {
            'id': segment_id,
            'name': name,
            'bbox': bbox,
            'bbox_size': [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]],
            'color': color
        }
        
        self.logger.info(f"Segment '{name}' (ID {segment_id})")
        self.logger.info(f"  Bounding box: X=[{bbox[0]},{bbox[3]}] Y=[{bbox[1]},{bbox[4]}] Z=[{bbox[2]},{bbox[5]}]")
        self.logger.info(f"  Size: {metadata['bbox_size'][0]} x {metadata['bbox_size'][1]} x {metadata['bbox_size'][2]} voxels")
        
        return metadata
    
    def estimate_memory_requirements(self, bbox_size: list, miplevel: int = 0) -> Dict[str, float]:
        """
        Estimate memory requirements for extraction.
        
        Args:
            bbox_size: [width, height, depth] in voxels at MIP 0
            miplevel: MIP level to extract at
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Adjust for MIP level
        mip_scale = [1, 1, 1]
        if miplevel > 0 and self.mip_factors is not None and miplevel <= len(self.mip_factors):
            mip_scale = self.mip_factors[miplevel - 1]
        
        adj_size = [
            bbox_size[0] // mip_scale[0],
            bbox_size[1] // mip_scale[1],
            bbox_size[2] // mip_scale[2]
        ]
        
        # TODO: Convert to int64 to avoid possible overflow 
        total_voxels = adj_size[0] * adj_size[1] * adj_size[2]
        
        # Memory estimates (conservative)
        # - Segmentation data: 4 bytes/voxel (uint32)
        # - Binary volume: 4 bytes/voxel (float32 for marching cubes)
        # - Marching cubes temporary: ~2x binary volume
        # - Mesh data: highly variable, estimate 100 bytes/surface voxel (conservative)
        
        seg_data_gb = float(total_voxels * 4) / (1024**3)
        binary_gb = float(total_voxels * 4) / (1024**3)
        mc_temp_gb = binary_gb * 2
        
        # Estimate surface voxels as ~10% of volume (very rough)
        surface_voxels = total_voxels * 0.1
        mesh_gb = (surface_voxels * 100) / (1024**3)
        
        peak_gb = seg_data_gb + binary_gb + mc_temp_gb + mesh_gb
        
        estimates = {
            'total_voxels': total_voxels,
            'seg_data_gb': seg_data_gb,
            'binary_gb': binary_gb,
            'marching_cubes_temp_gb': mc_temp_gb,
            'mesh_estimated_gb': mesh_gb,
            'peak_estimated_gb': peak_gb,
            'needs_blocking': peak_gb > 20  # Conservative threshold for 32GB systems
        }
        
        self.logger.debug(f"Memory estimate: {peak_gb:.2f} GB peak")
        
        return estimates
    
    def _calculate_block_dimensions(
        self,
        bbox_size: list,
        miplevel: int,
        max_voxels_per_block: int
        ) -> Tuple[int, int, int]:
        """
        Calculate optimal block dimensions for processing.
        
        Args:
            bbox_size: [width, height, depth] of full region
            miplevel: MIP level
            max_voxels_per_block: Maximum voxels per block (uses self.max_block_voxels if None)
            
        Returns:
            Tuple of (blocks_x, blocks_y, blocks_z) - number of blocks in each dimension
        """
        if max_voxels_per_block is None:
            max_voxels_per_block = self.max_block_voxels
        
        # Adjust size for MIP level
        mip_scale = [1, 1, 1]
        if miplevel > 0 and self.mip_factors is not None and miplevel <= len(self.mip_factors):
            mip_scale = self.mip_factors[miplevel - 1]
        
        adj_size = [
            bbox_size[0] // mip_scale[0],
            bbox_size[1] // mip_scale[1],
            bbox_size[2] // mip_scale[2]
        ]
        
        total_voxels = adj_size[0] * adj_size[1] * adj_size[2]
        
        # If fits in one block, return 1,1,1
        if total_voxels <= max_voxels_per_block:
            return 1, 1, 1
        
        # Calculate number of blocks needed
        # Start with cube root to get balanced splitting
        blocks_needed = np.ceil(total_voxels / max_voxels_per_block)
        blocks_per_dim = int(np.ceil(blocks_needed ** (1/3)))
        
        # Adjust based on actual dimensions to minimize wasted blocks
        blocks_x = max(1, int(np.ceil(adj_size[0] / (adj_size[0] / blocks_per_dim))))
        blocks_y = max(1, int(np.ceil(adj_size[1] / (adj_size[1] / blocks_per_dim))))
        blocks_z = max(1, int(np.ceil(adj_size[2] / (adj_size[2] / blocks_per_dim))))
        
        self.logger.info(f"Block division: {blocks_x} x {blocks_y} x {blocks_z} = {blocks_x*blocks_y*blocks_z} blocks")
        self.logger.info(f"Average block size: ~{total_voxels/(blocks_x*blocks_y*blocks_z):,.0f} voxels")
        
        return blocks_x, blocks_y, blocks_z

    def _merge_meshes(
        self,
        verts1: Optional[np.ndarray],
        faces1: Optional[np.ndarray],
        verts2: Optional[np.ndarray],
        faces2: Optional[np.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge two meshes into one.
        
        Args:
            verts1, faces1: First mesh (can be None/empty)
            verts2, faces2: Second mesh (can be None/empty)
            
        Returns:
            Tuple of (merged_vertices, merged_faces)
        """

        assert faces2 is not None and faces1 is not None
        # Handle empty cases
        if verts1 is None or (isinstance(verts1, np.ndarray) and len(verts1) == 0):
            if verts2 is None or (isinstance(verts2, np.ndarray) and len(verts2) == 0):
                return np.array([]), np.array([])
            assert verts2 is not None
            return verts2.copy(), faces2.copy()
        
        if verts2 is None or (isinstance(verts2, np.ndarray) and len(verts2) == 0):
            assert verts1 is not None
            return verts1.copy(), faces1.copy()
        
        # Merge vertices
        assert verts1 is not None and verts2 is not None
        merged_verts = np.vstack([verts1, verts2])
        
        # Offset face indices for second mesh
        offset_faces = faces2 + len(verts1)
        merged_faces = np.vstack([faces1, offset_faces])
        
        return merged_verts, merged_faces

    def _extract_single_block(
        self,
        segment_id: int,
        metadata: Dict[str, Any],
        miplevel: int,
        block_bounds: Tuple[int, int, int, int, int, int],
        global_bounds: Tuple[int, int, int, int, int, int],
        close_surfaces: bool,
        mip_scale: list
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract surface for a single block.
        
        Args:
            segment_id: Segment ID
            metadata: Segment metadata
            miplevel: MIP level
            block_bounds: (minx, maxx, miny, maxy, minz, maxz) for this block
            global_bounds: Dataset bounds for clamping
            close_surfaces: Whether to close boundaries
            mip_scale: MIP scale factors
            
        Returns:
            Tuple of (vertices, faces) for this block
        """
        minx, maxx, miny, maxy, minz, maxz = block_bounds
        max_x_bound, max_y_bound, max_z_bound = global_bounds[1], global_bounds[3], global_bounds[5]
        
        # Clamp to dataset bounds
        minx = max(global_bounds[0], minx)
        maxx = min(max_x_bound, maxx)
        miny = max(global_bounds[2], miny)
        maxy = min(max_y_bound, maxy)
        minz = max(global_bounds[4], minz)
        maxz = min(max_z_bound, maxz)
        
        self.logger.debug(f"  Block region: X=[{minx},{maxx}] Y=[{miny},{maxy}] Z=[{minz},{maxz}]")
        
        # Load block data with timeout
        volume_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
        estimated_timeout = max(60, int(volume_size / 10_000_000) * 10)
        self._set_socket_timeout(estimated_timeout)
        
        try:
            seg_image = self.vast.get_seg_image_rle_decoded(
                miplevel, minx, maxx, miny, maxy, minz, maxz,
                surfonlyflag=0, flipflag=0
            )
        finally:
            self._set_socket_timeout(10)
        
        if seg_image is None:
            self.logger.warning(f"Failed to load block data")
            return None, None
        
        # Create binary volume
        binary_volume = (seg_image == segment_id).astype(np.float32)
        voxel_count = np.sum(binary_volume)
        
        if voxel_count == 0:
            self.logger.debug(f"  Block empty (no voxels)")
            return None, None
        
        self.logger.debug(f"  Block has {int(voxel_count):,} voxels")
        
        del seg_image
        
        # Add boundary padding if closing surfaces
        offset_adjust = np.array([0, 0, 0])
        if close_surfaces:
            padded = np.zeros((
                binary_volume.shape[0] + 2,
                binary_volume.shape[1] + 2,
                binary_volume.shape[2] + 2
            ), dtype=np.float32)
            padded[1:-1, 1:-1, 1:-1] = binary_volume
            binary_volume = padded
            offset_adjust = np.array([-1, -1, -1])
        
        # Run marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                binary_volume,
                level=0.5,
                step_size=1
            )
        except Exception as e:
            self.logger.warning(f"  Marching cubes failed for block: {str(e)}")
            return None, None
        
        del binary_volume
        
        # Transform coordinates to global space
        verts = verts + offset_adjust
        verts[:, 0] += minx
        verts[:, 1] += miny
        verts[:, 2] += minz
        
        # Scale by voxel size and MIP factor
        voxel_size = np.array([
            self.dataset_info['voxelsizex'],
            self.dataset_info['voxelsizey'],
            self.dataset_info['voxelsizez']
        ])
        
        verts[:, 0] *= voxel_size[0] * mip_scale[0]
        verts[:, 1] *= voxel_size[1] * mip_scale[1]
        verts[:, 2] *= voxel_size[2] * mip_scale[2]
        
        # Convert to micrometers
        if voxel_size[0] > 1:
            verts *= 0.001
        
        return verts, faces

    def _extract_with_blocks(
        self,
        segment_id: int,
        metadata: Dict[str, Any],
        miplevel: int,
        close_surfaces: bool
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract surface using block-based processing for large segments.
        
        Args:
            segment_id: Segment ID
            metadata: Segment metadata
            miplevel: MIP level
            close_surfaces: Whether to close boundaries
            
        Returns:
            Tuple of (vertices, faces) or (None, None) on failure
        """
        self.logger.info("Using block-based extraction for large segment")
        
        bbox = metadata['bbox']
        
        # Adjust bounding box for MIP level
        mip_scale = [1, 1, 1]
        if miplevel > 0 and self.mip_factors is not None and miplevel <= len(self.mip_factors):
            mip_scale = self.mip_factors[miplevel - 1]
        
        minx = bbox[0] >> miplevel
        maxx = bbox[3] >> miplevel
        miny = bbox[1] >> miplevel
        maxy = bbox[4] >> miplevel
        minz = bbox[2]
        maxz = bbox[5]
        
        if miplevel > 0 and mip_scale[2] != 1:
            minz = minz // mip_scale[2]
            maxz = maxz // mip_scale[2]
        
        # Get dataset bounds
        mip_scale_val = mip_scale if isinstance(mip_scale, list) else [1, 1, 1]
        max_x_bound = (self.dataset_info['datasizex'] >> miplevel) - 1
        max_y_bound = (self.dataset_info['datasizey'] >> miplevel) - 1
        max_z_bound = self.dataset_info['datasizez'] - 1
        if miplevel > 0 and mip_scale_val[2] != 1:
            max_z_bound = max_z_bound // mip_scale_val[2]
        
        # Add padding for marching cubes
        padding = 2 if close_surfaces else 1
        minx = max(0, minx - padding)
        miny = max(0, miny - padding)
        minz = max(0, minz - padding)
        maxx = min(max_x_bound, maxx + padding)
        maxy = min(max_y_bound, maxy + padding)
        maxz = min(max_z_bound, maxz + padding)
        
        global_bounds = (0, max_x_bound, 0, max_y_bound, 0, max_z_bound)
        
        # Calculate block dimensions
        region_size = [maxx - minx + 1, maxy - miny + 1, maxz - minz + 1]
        blocks_x, blocks_y, blocks_z = self._calculate_block_dimensions(region_size, miplevel, self.max_block_voxels)
        
        total_blocks = blocks_x * blocks_y * blocks_z
        self.logger.info(f"Processing {total_blocks} blocks total")
        
        # Calculate block sizes
        block_size_x = int(np.ceil(region_size[0] / blocks_x))
        block_size_y = int(np.ceil(region_size[1] / blocks_y))
        block_size_z = int(np.ceil(region_size[2] / blocks_z))
        
        # Set translation to isolate this segment
        self.vast.set_seg_translation([segment_id], [segment_id])
        
        try:
            # Process blocks
            merged_verts = None
            merged_faces = None
            blocks_processed = 0
            blocks_with_data = 0
            
            for iz in range(blocks_z):
                for iy in range(blocks_y):
                    for ix in range(blocks_x):
                        blocks_processed += 1
                        
                        # Calculate block bounds with overlap
                        bminx = minx + ix * block_size_x - (self.block_overlap if ix > 0 else 0)
                        bmaxx = min(maxx, minx + (ix + 1) * block_size_x + self.block_overlap)
                        bminy = miny + iy * block_size_y - (self.block_overlap if iy > 0 else 0)
                        bmaxy = min(maxy, miny + (iy + 1) * block_size_y + self.block_overlap)
                        bminz = minz + iz * block_size_z - (self.block_overlap if iz > 0 else 0)
                        bmaxz = min(maxz, minz + (iz + 1) * block_size_z + self.block_overlap)
                        
                        self.logger.info(f"Processing block {blocks_processed}/{total_blocks} " +
                                    f"({ix},{iy},{iz})")
                        
                        # Extract block
                        block_verts, block_faces = self._extract_single_block(
                            segment_id, metadata, miplevel,
                            (bminx, bmaxx, bminy, bmaxy, bminz, bmaxz),
                            global_bounds, close_surfaces, mip_scale
                        )
                        
                        if block_verts is not None and len(block_verts) > 0:
                            blocks_with_data += 1
                            merged_verts, merged_faces = self._merge_meshes(
                                merged_verts, merged_faces,
                                block_verts, block_faces
                            )
                            self.logger.info(f"  Block contributed {len(block_verts):,} vertices, " +
                                        f"{len(block_faces) if block_faces is not None else 0:,} faces")
                            self.logger.info(f"  Total so far: {len(merged_verts):,} vertices, " +
                                        f"{len(merged_faces):,} faces")
            
            self.logger.info(f"Block processing complete: {blocks_with_data}/{total_blocks} blocks had data")
            
            if merged_verts is None or len(merged_verts) == 0:
                self.logger.error("No geometry generated from any block")
                return None, None
            
            assert merged_faces is not None
            self.logger.info(f"Final mesh: {len(merged_verts):,} vertices, {len(merged_faces):,} faces")
            
            return merged_verts, merged_faces
            
        finally:
            # Always clear translation
            self.vast.set_seg_translation([], [])

    #######################################################
    # Extracting mesh
    #######################################################

    def extract_segment(
        self,
        segment_id: int,
        miplevel: int = 0,
        close_surfaces: bool = True,
        output_format: str = 'obj',
        output_filename: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Extract surface mesh for a single segment.
        
        Args:
            segment_id: Segment ID to extract
            miplevel: MIP level (0 = full resolution)
            close_surfaces: Close mesh at volume boundaries
            output_format: 'obj' or 'ply'
            output_filename: Custom filename (auto-generated if None)
            
        Returns:
            Tuple of (vertices, faces, output_path)
            - vertices: (N, 3) array of XYZ coordinates in physical units (micrometers)
            - faces: (M, 3) array of triangle indices (0-indexed)
            - output_path: Path to saved file
            Returns (None, None, None) if extraction fails
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting extraction for segment {segment_id}")
        self.logger.info("=" * 80)
        
        try:
            # Get segment metadata
            metadata = self.get_segment_metadata(segment_id)
            if metadata is None:
                self.logger.error(f"Cannot extract segment {segment_id} - metadata retrieval failed")
                return None, None, None
            
            # Estimate memory requirements
            mem_est = self.estimate_memory_requirements(metadata['bbox_size'], miplevel)
            self.logger.info(f"Estimated peak memory: {mem_est['peak_estimated_gb']:.2f} GB")
            
            # Choose extraction method based on size
            if mem_est['needs_blocking']:
                self.logger.warning("Large segment detected - using block processing")
                vertices, faces = self._extract_with_blocks(
                    segment_id,
                    metadata,
                    miplevel,
                    close_surfaces
                )
            else:
                # Extract surface with full volume method
                vertices, faces = self._extract_full_volume(
                    segment_id, 
                    metadata, 
                    miplevel, 
                    close_surfaces
                )
            
            if vertices is None or len(vertices) == 0:
                self.logger.error("Surface extraction produced no geometry")
                return None, None, None
            
            assert faces is not None
            # Save to file
            output_path = self._save_mesh(
                vertices, 
                faces, 
                metadata, 
                output_format, 
                output_filename
            )
            
            # Log success
            self.logger.info("=" * 80)
            self.logger.info("Extraction completed successfully!")
            self.logger.info(f"  Vertices: {len(vertices):,}")
            self.logger.info(f"  Faces: {len(faces) if faces is not None else 0:,}")
            self.logger.info(f"  Output: {output_path}")
            self.logger.info("=" * 80)
            
            return vertices, faces, output_path
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return None, None, None
    
    def _extract_full_volume(
        self,
        segment_id: int,
        metadata: Dict[str, Any],
        miplevel: int,
        close_surfaces: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract surface from full volume (no blocking).
        
        Args:
            segment_id: Segment ID
            metadata: Segment metadata
            miplevel: MIP level
            close_surfaces: Whether to close boundaries
            
        Returns:
            Tuple of (vertices, faces) or (None, None) on failure
        """
        bbox = metadata['bbox']
        
        # Adjust bounding box for MIP level
        mip_scale = [1, 1, 1]
        if miplevel > 0 and self.mip_factors is not None and miplevel <= len(self.mip_factors):
            mip_scale = self.mip_factors[miplevel - 1]
        
        minx = bbox[0] >> miplevel
        maxx = bbox[3] >> miplevel
        miny = bbox[1] >> miplevel
        maxy = bbox[4] >> miplevel
        minz = bbox[2]
        maxz = bbox[5]
        
        if miplevel > 0 and mip_scale[2] != 1:
            minz = minz // mip_scale[2]
            maxz = maxz // mip_scale[2]
        
        mip_scale_val = mip_scale if isinstance(mip_scale, list) else [1, 1, 1]
        max_x_bound = (self.dataset_info['datasizex'] >> miplevel) - 1
        max_y_bound = (self.dataset_info['datasizey'] >> miplevel) - 1
        max_z_bound = self.dataset_info['datasizez'] - 1
        if miplevel > 0 and mip_scale_val[2] != 1:
            max_z_bound = max_z_bound // mip_scale_val[2]

        # Add padding for marching cubes, clamped to dataset bounds
        padding = 2 if close_surfaces else 1
        minx = max(0, minx - padding)
        miny = max(0, miny - padding)
        minz = max(0, minz - padding)
        maxx = min(max_x_bound, maxx + padding)  # CLAMP
        maxy = min(max_y_bound, maxy + padding)  # CLAMP
        maxz = min(max_z_bound, maxz + padding)  # CLAMP
        
        self.logger.info(f"Loading volume at MIP {miplevel}")
        self.logger.info(f"  Region: X=[{minx},{maxx}] Y=[{miny},{maxy}] Z=[{minz},{maxz}]")
        
        # Set translation to isolate this segment
        self.vast.set_seg_translation([segment_id], [segment_id])
        
        # Load segmentation data
        # Increase timeout for large transfers (estimate: 1 second per 10M voxels)
        volume_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
        estimated_timeout = max(60, int(volume_size / 10_000_000) * 10)  # At least 60 seconds
        self.logger.debug(f"Setting socket timeout to {estimated_timeout}s for {volume_size:,} voxel request")
        self._set_socket_timeout(estimated_timeout)

        try:
            seg_image = self.vast.get_seg_image_rle_decoded(
                miplevel, minx, maxx, miny, maxy, minz, maxz,
                surfonlyflag=0, flipflag=0
            )
        finally:
            # Always clear translation
            self.vast.set_seg_translation([], [])
            # Reset to default timeout
            self._set_socket_timeout(10)
        
        if seg_image is None:
            self.logger.error("Failed to load segmentation data from VAST")
            return None, None
        
        self.logger.info(f"Loaded volume shape: {seg_image.shape}")
        
        # Create binary volume
        self.logger.info("Creating binary volume...")
        binary_volume = (seg_image == segment_id).astype(np.float32)
        
        voxel_count = np.sum(binary_volume)
        if voxel_count == 0:
            self.logger.error(f"Segment {segment_id} not found in loaded region")
            return None, None
        
        self.logger.info(f"Found {int(voxel_count):,} voxels for segment {segment_id}")
        
        # Free up memory
        del seg_image
        
        # Add boundary padding if closing surfaces
        offset_adjust = np.array([0, 0, 0])
        if close_surfaces:
            self.logger.debug("Adding boundary padding for closed surfaces")
            padded = np.zeros((
                binary_volume.shape[0] + 2,
                binary_volume.shape[1] + 2,
                binary_volume.shape[2] + 2
            ), dtype=np.float32)
            padded[1:-1, 1:-1, 1:-1] = binary_volume
            binary_volume = padded
            offset_adjust = np.array([-1, -1, -1])
        
        # Run marching cubes
        self.logger.info("Running marching cubes algorithm...")
        try:
            verts, faces, normals, values = measure.marching_cubes(
                binary_volume, 
                level=0.5,
                step_size=1
            )
        except Exception as e:
            self.logger.error(f"Marching cubes failed: {str(e)}")
            return None, None
        
        self.logger.info(f"Marching cubes complete: {len(verts):,} vertices, {len(faces):,} faces")
        
        # Free up memory
        del binary_volume
        
        # Transform coordinates
        self.logger.info("Transforming coordinates to physical space...")
        
        # Undo padding offset
        verts = verts + offset_adjust
        
        # Add region offset
        verts[:, 0] += minx
        verts[:, 1] += miny
        verts[:, 2] += minz
        
        # Scale by voxel size and MIP factor
        voxel_size = np.array([
            self.dataset_info['voxelsizex'],
            self.dataset_info['voxelsizey'],
            self.dataset_info['voxelsizez']
        ])
        
        verts[:, 0] *= voxel_size[0] * mip_scale[0]
        verts[:, 1] *= voxel_size[1] * mip_scale[1]
        verts[:, 2] *= voxel_size[2] * mip_scale[2]
        
        # Convert to micrometers (assuming voxel_size is in nm)
        if voxel_size[0] > 1:  # Likely nanometers
            verts *= 0.001
            self.logger.debug("Converted coordinates from nm to um")
        
        return verts, faces
    
    ########################################################
    # Extracting voxel data
    ########################################################
    
    def extract_segment_voxel(
        self,
        segment_id: int,
        miplevel: int = 0,
        padding: int = 1,
        output_filename: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]], Optional[Tuple[float, float, float]], Optional[str]]:
        """
        Extract voxel mask for a single segment (for skeletonization pipeline).
        
        This method extracts the raw binary voxel mask without any cleaning operations.
        The mask can then be passed to cleaning and skeletonization functions.
        
        Args:
            segment_id: Segment ID to extract
            miplevel: MIP level (0 = full resolution, higher = coarser)
            padding: Voxels of padding around bounding box
            output_filename: Optional filename for saving mask as .npy
            
        Returns:
            Tuple of (mask, bbox_min_vox, voxel_size_um, output_path)
            - mask: bool[z, y, x] - Boolean 3D array of segment voxels
            - bbox_min_vox: (minx, miny, minz) - Origin coordinates in voxel space
            - voxel_size_um: (sx, sy, sz) - Voxel dimensions in microns (includes MIP scaling)
            - output_path: Path to saved .npy file (if output_filename provided) or None
            Returns (None, None, None, None) if extraction fails
            
        Example:
            >>> extractor = SegmentSurfaceExtractor(vast)
            >>> mask, origin, voxel_size, path = extractor.extract_segment_voxel(
            ...     segment_id=1, 
            ...     miplevel=1,
            ...     output_filename="segment_001_mask.npy"
            ... )
            >>> # mask is ready for cleaning and skeletonization
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting voxel extraction for segment {segment_id}")
        self.logger.info("=" * 80)
        
        try:
            # Get segment metadata
            metadata = self.get_segment_metadata(segment_id)
            if metadata is None:
                self.logger.error(f"Cannot extract segment {segment_id} - metadata retrieval failed")
                return None, None, None, None
            
            # Extract the mask
            mask, bbox_min_vox, voxel_size_um = self._extract_full_volume_mask(
                segment_id, 
                metadata, 
                miplevel,
                padding
            )
            
            if mask is None:
                self.logger.error("Voxel mask extraction failed")
                return None, None, None, None
            
            voxel_count = np.sum(mask)
            if voxel_count == 0:
                self.logger.error(f"Segment {segment_id} has no voxels in extracted region")
                return None, None, None, None
            
            self.logger.info(f"Extracted mask: {mask.shape} with {int(voxel_count):,} voxels")
            self.logger.info(f"  Origin (voxels): {bbox_min_vox}")
            self.logger.info(f"  Voxel size (um): {voxel_size_um}")
            
            # Save to file if requested
            output_path = None
            if output_filename:
                output_path = self._save_voxel_mask(
                    mask,
                    bbox_min_vox,
                    voxel_size_um,
                    metadata,
                    output_filename
                )
            
            # Log success
            self.logger.info("=" * 80)
            self.logger.info("Voxel extraction completed successfully!")
            self.logger.info(f"  Mask shape: {mask.shape}")
            self.logger.info(f"  Voxel count: {int(voxel_count):,}")
            if output_path:
                self.logger.info(f"  Output: {output_path}")
            self.logger.info("=" * 80)
            
            return mask, bbox_min_vox, voxel_size_um, output_path
            
        except Exception as e:
            self.logger.error(f"Voxel extraction failed: {str(e)}", exc_info=True)
            return None, None, None, None

    def _extract_full_volume_mask(
        self,
        segment_id: int,
        metadata: Dict[str, Any],
        miplevel: int,
        padding: int = 1
    ) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[float, float, float]]:
        """
        Extract the binary voxel mask for a segment at a specific MIP level.
        
        This is a low-level extraction method that returns the raw binary mask
        without any cleaning operations. The mask is suitable for further processing
        in the skeletonization pipeline.
        
        Args:
            segment_id: Segment ID to extract
            metadata: Segment metadata from get_segment_metadata()
            miplevel: MIP level (0 = full resolution)
            padding: Voxels of padding to add around bounding box (default: 1)
            
        Returns:
            Tuple of (mask, bbox_min_vox, voxel_size_um)
            - mask: bool[z, y, x] - Boolean mask of the segment
            - bbox_min_vox: (minx, miny, minz) - Minimum x/y/z used to fetch the volume in voxels
            - voxel_size_um: (sx, sy, sz) - Voxel size in microns, including mip scaling
            
        Note:
            - Z dimension may not be MIP-scaled depending on dataset configuration
            - Coordinates are in (Z, Y, X) order following numpy convention
            - Returns (None, None, None) on failure
        """
        bbox = metadata['bbox']
        
        # Adjust bounding box for MIP level
        mip_scale = [1, 1, 1]
        if miplevel > 0 and self.mip_factors is not None and miplevel <= len(self.mip_factors):
            mip_scale = self.mip_factors[miplevel - 1]
        
        minx = bbox[0] >> miplevel
        maxx = bbox[3] >> miplevel
        miny = bbox[1] >> miplevel
        maxy = bbox[4] >> miplevel
        minz = bbox[2]
        maxz = bbox[5]
        
        if miplevel > 0 and mip_scale[2] != 1:
            minz = minz // mip_scale[2]
            maxz = maxz // mip_scale[2]
        
        mip_scale_val = mip_scale if isinstance(mip_scale, list) else [1, 1, 1]
        max_x_bound = (self.dataset_info['datasizex'] >> miplevel) - 1
        max_y_bound = (self.dataset_info['datasizey'] >> miplevel) - 1
        max_z_bound = self.dataset_info['datasizez'] - 1
        if miplevel > 0 and mip_scale_val[2] != 1:
            max_z_bound = max_z_bound // mip_scale_val[2]

        # Add padding around bounding box, clamped to dataset bounds
        minx = max(0, minx - padding)
        miny = max(0, miny - padding)
        minz = max(0, minz - padding)
        maxx = min(max_x_bound, maxx + padding)  # CLAMP
        maxy = min(max_y_bound, maxy + padding)  # CLAMP
        maxz = min(max_z_bound, maxz + padding)  # CLAMP
        
        self.logger.info(f"Loading volume at MIP {miplevel}")
        self.logger.info(f"  Region: X=[{minx},{maxx}] Y=[{miny},{maxy}] Z=[{minz},{maxz}]")
        
        # Set translation to isolate this segment
        self.vast.set_seg_translation([segment_id], [segment_id])
        
        # Load segmentation data
        # Increase timeout for large transfers (estimate: 1 second per 10M voxels)
        volume_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
        estimated_timeout = max(60, int(volume_size / 10_000_000) * 10)  # At least 60 seconds
        self.logger.debug(f"Setting socket timeout to {estimated_timeout}s for {volume_size:,} voxel request")
        self._set_socket_timeout(estimated_timeout)

        try:
            seg_image = self.vast.get_seg_image_rle_decoded(
                miplevel, minx, maxx, miny, maxy, minz, maxz,
                surfonlyflag=0, flipflag=0
            )
        finally:
            # Always clear translation
            self.vast.set_seg_translation([], [])
            # Reset to default timeout
            self._set_socket_timeout(10)
        
        if seg_image is None:
            self.logger.error("Failed to load segmentation data from VAST")
            return np.array([]), (0, 0, 0), (0.0, 0.0, 0.0) #Tuple[np.ndarray, Tuple[int, int, int], Tuple[float, float, float]]
        
        self.logger.info(f"Loaded volume shape: {seg_image.shape}")
        
        # Create binary volume
        self.logger.info("Creating binary volume...")
        mask = (seg_image == segment_id).astype(bool)
        
        # Store the min bounds in voxels
        bbox_min_vox = (minx, miny, minz)
        
        # Calculate voxel size in microns, including mip scaling
        # Base voxel sizes from dataset info (in nm)
        base_voxel_size = np.array([
            self.dataset_info['voxelsizex'],
            self.dataset_info['voxelsizey'],
            self.dataset_info['voxelsizez']
        ])
        
        # Apply MIP scaling
        voxel_size_nm = base_voxel_size * mip_scale_val
        
        # Convert to microns
        voxel_size_um = tuple(voxel_size_nm * 0.001)

        return mask, bbox_min_vox, voxel_size_um
    
    def _save_voxel_mask(
        self,
        mask: np.ndarray,
        bbox_min_vox: Tuple[int, int, int],
        voxel_size_um: Tuple[float, float, float],
        metadata: Dict[str, Any],
        custom_filename: Optional[str]
    ) -> str:
        """
        Save voxel mask and metadata to .npz file.
        
        Args:
            mask: Boolean voxel mask
            bbox_min_vox: Origin coordinates in voxel space
            voxel_size_um: Voxel dimensions in microns
            metadata: Segment metadata
            custom_filename: Custom filename or None
            
        Returns:
            Path to saved file
        """
        # Generate filename
        if custom_filename:
            base_name = custom_filename
            if not base_name.endswith('.npz'):
                base_name += '.npz'
        else:
            # Clean segment name for filename
            clean_name = metadata['name'].replace(' ', '_')
            # Remove special characters
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ['_', '-'])
            base_name = f"seg_{metadata['id']:04d}_{clean_name}_mask.npz"
        
        # Create output directory structure
        voxels_dir = self.output_dir / "voxels"
        voxels_dir.mkdir(exist_ok=True)
        
        output_path = voxels_dir / base_name
        
        # Save mask and metadata
        np.savez_compressed(
            output_path,
            mask=mask,
            bbox_min_vox=bbox_min_vox,
            voxel_size_um=voxel_size_um,
            segment_id=metadata['id'],
            segment_name=metadata['name']
        )
        
        self.logger.info(f"Saved voxel mask: {output_path}")
        
        return str(output_path)

    #######################################################
    # Saving meshes 
    #######################################################

    def _save_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        metadata: Dict[str, Any],
        output_format: str,
        custom_filename: Optional[str]
    ) -> str:
        """
        Save mesh to file.
        
        Args:
            vertices: Vertex array
            faces: Face array
            metadata: Segment metadata
            output_format: File format
            custom_filename: Custom filename or None
            
        Returns:
            Path to saved file
        """
        # Generate filename
        if custom_filename:
            base_name = custom_filename
        else:
            # Clean segment name for filename
            clean_name = metadata['name'].replace(' ', '_')
            # Remove special characters
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ['_', '-'])
            base_name = f"seg_{metadata['id']:04d}_{clean_name}"
        
        # Create output directory structure
        meshes_dir = self.output_dir / "meshes"
        meshes_dir.mkdir(exist_ok=True)
        
        if output_format.lower() == 'obj':
            output_path = meshes_dir / f"{base_name}.obj"
            self._write_obj(output_path, vertices, faces, metadata)
        elif output_format.lower() == 'ply':
            output_path = meshes_dir / f"{base_name}.ply"
            self._write_ply(output_path, vertices, faces, metadata)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.logger.info(f"Saved mesh: {output_path}")
        
        return str(output_path)
    
    def _write_obj(
        self,
        filepath: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        metadata: Dict[str, Any]
    ):
        """Write mesh in OBJ format with MTL material."""
        obj_path = filepath
        mtl_path = filepath.with_suffix('.mtl')
        
        object_name = metadata['name'].replace(' ', '_')
        material_name = f"mat_{metadata['id']}"
        
        # Write OBJ file
        with open(obj_path, 'w') as f:
            # Header
            f.write(f"# VAST Surface Export - Production Pipeline\n")
            f.write(f"# Segment: {metadata['name']} (ID: {metadata['id']})\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
            f.write(f"# Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Coordinates in micrometers\n")
            f.write(f"mtllib {mtl_path.name}\n")
            f.write(f"o {object_name}\n")
            f.write(f"usemtl {material_name}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write(f"\ng {object_name}\n")
            
            # Write faces (OBJ uses 1-indexed)
            for face in faces:
                f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
        
        # Write MTL file
        color = np.array(metadata['color']) / 255.0
        with open(mtl_path, 'w') as f:
            f.write(f"# Material for {metadata['name']}\n")
            f.write(f"newmtl {material_name}\n")
            f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            f.write(f"Ks 0.5 0.5 0.5\n")
            f.write(f"Ns 32\n")
            f.write(f"d 1.0\n")
    
    def _write_ply(
        self,
        filepath: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        metadata: Dict[str, Any]
    ):
        """Write mesh in PLY format."""
        color = metadata['color']
        
        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment VAST Surface Export - {metadata['name']}\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices with color
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def main():
    print("=" * 80)
    print("VAST Surface Extraction")
    print("=" * 80)
    
    # Connect to VAST
    vast = VASTControlClass()
    print("\nConnecting to VAST...")
    
    if not vast.connect("127.0.0.1", 22081, timeout=10):
        print("ERROR: Failed to connect to VAST")
        print("  - Ensure VAST is running")
        print("  - Enable API in VAST Preferences")
        return
    
    print("Connected to VAST")
    try:
        if(len(sys.argv) > 1 and sys.argv[1] == "skeleton"):

                extractor = SegmentSurfaceExtractor(vast, output_dir="./vast_export")

                segment_id = 1
                    
                print(f"\nExtracting segment {segment_id}...")
                print("This may take several minutes for large neurons...\n")
                
                mask, origin, voxel_size, output_path = extractor.extract_segment_voxel(
                    segment_id=segment_id,
                    miplevel=1,  # Half resolution
                    padding=1,
                    output_filename=f"segment_{segment_id:04d}_mask.npy"
                )
        
        else:
            # Create extractor
            extractor = SegmentSurfaceExtractor(vast, output_dir="./vast_export")
            
            num_segments = vast.get_number_of_segments()
            if num_segments is None:
                print("ERROR: Could not retrieve number of segments")
                return
            for segment_id in range(1, num_segments + 1):
                
                print(f"\nExtracting segment {segment_id}...")
                print("This may take several minutes for large neurons...\n")
                
                vertices, faces, output_path = extractor.extract_segment(
                    segment_id=segment_id,
                    miplevel=1,  # Full resolution
                    close_surfaces=False,
                    output_format='obj'
                )
                
                if vertices is not None and faces is not None:
                    print("\n" + "=" * 80)
                    print("SUCCESS!")
                    print("=" * 80)
                    print(f"Mesh saved to: {output_path}")
                    print(f"Vertices: {len(vertices):,}")
                    print(f"Faces: {len(faces):,}")

                else:
                    print("\nExtraction failed - check logs for details")
                    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        vast.disconnect()
        print("\nDisconnected from VAST")


if __name__ == "__main__":
    main()