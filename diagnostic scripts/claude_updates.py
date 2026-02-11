import numpy as np
from skimage import measure
from scipy import ndimage
from math import floor, ceil
import re
from typing import Tuple, List, Dict, Optional
import time
import logging
from VASTControlClass import VASTControlClass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SurfaceExtractor:
    """
    Python port of MATLAB extractsurfaces function.
    Extracts 3D surfaces from segmentation or screenshot data.
    """
    
    def __init__(self, vast_control: VASTControlClass, export_params: Dict, region_params: Dict):
        """
        Initialize the surface extractor.
        
        Args:
            vast_control: VASTControlClass instance for data access
            export_params: Dictionary containing export configuration
            region_params: Dictionary containing region bounds (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        self.vast = vast_control
        self.export_params = export_params
        self.region_params = region_params
        self.canceled = False
        
        # Will be populated during extraction
        self.param = {}
        self.objects = None
        self.names = []
        self.data = None
        
    def extract_surfaces(self):
        """Main extraction function - Part 1: Setup and initialization"""
        
        # Check connection (implement as needed for your GUI)
        if not self._check_connection():
            return
        
        # Disable network warnings if requested
        if self.export_params.get('disablenetwarnings', 0) == 1:
            self.vast.set_error_popups_enabled(74, False)  # network connection error
            self.vast.set_error_popups_enabled(75, False)  # unexpected data
            self.vast.set_error_popups_enabled(76, False)  # unexpected length
        
        self._update_message('Exporting Surfaces ...', 'Loading Metadata ...')
        
        param = self.export_params.copy()
        rparam = self.region_params.copy()
        
        # Determine if extracting segmentation or screenshots
        extract_seg = True
        if 5 <= param.get('extractwhich', 1) <= 10:
            extract_seg = False
        
        # Load segmentation data or setup screenshot extraction
        if extract_seg:
            # Get all segment data
            data, res = self.vast.get_all_segment_data_matrix()
            names, res = self.vast.get_all_segment_names()
            seg_layer_name = self._get_selected_seg_layer_name()
            
            # Remove 'Background' (first entry)
            names = names[1:]
            
            max_object_number = int(np.max(data[:, 0]))
            
            # Get selected layer info
            selected_layer_nr, selected_em_layer_nr, selected_seg_layer_nr, res = \
                self.vast.get_selected_layer_nr()
            
            mip_data_size = self.vast.get_data_size_at_mip(
                param['miplevel'], 
                selected_seg_layer_nr
            )
        else:
            # Setup for screenshot extraction
            names = self._setup_screenshot_names(param)
            selected_layer_nr, selected_em_layer_nr, selected_seg_layer_nr, res = \
                self.vast.get_selected_layer_nr()
        
        # Get mipmap scale factors
        z_scale = 1
        z_min = rparam['zmin']
        z_max = rparam['zmax']
        
        if param['miplevel'] > 0:
            if extract_seg:
                mip_scale_matrix, res = self.vast.get_mipmap_scale_factors(selected_seg_layer_nr)
            else:
                mip_scale_matrix, res = self.vast.get_mipmap_scale_factors(selected_em_layer_nr)
            
            z_scale = mip_scale_matrix[param['miplevel'], 2]
            
            if z_scale != 1:
                z_min = floor(z_min / z_scale)
                z_max = floor(z_max / z_scale)
        
        # Calculate coordinate bounds at current mip level
        x_min = rparam['xmin'] >> param['miplevel']
        x_max = (rparam['xmax'] >> param['miplevel']) - 1
        y_min = rparam['ymin'] >> param['miplevel']
        y_max = (rparam['ymax'] >> param['miplevel']) - 1
        
        # Get mip scale factors
        if param['miplevel'] > 0:
            mip_fact_x = mip_scale_matrix[param['miplevel'], 0]
            mip_fact_y = mip_scale_matrix[param['miplevel'], 1]
            mip_fact_z = mip_scale_matrix[param['miplevel'], 2]
        else:
            mip_fact_x = 1
            mip_fact_y = 1
            mip_fact_z = 1
        
        # Validate volume dimensions
        if ((x_min == x_max or y_min == y_max or z_min == z_max) and 
            param.get('closesurfaces', 0) == 0):
            self._show_error(
                'ERROR: The surface script needs a volume which is at least '
                'two pixels wide in each direction. Please adjust "Render from area" '
                'values, or enable "Close surface sides".'
            )
            self._update_message('Canceled.')
            return
        
        # Store processed parameters
        self.param = {
            'extract_seg': extract_seg,
            'names': names,
            'xmin': x_min,
            'xmax': x_max,
            'ymin': y_min,
            'ymax': y_max,
            'zmin': z_min,
            'zmax': z_max,
            'mipfactx': mip_fact_x,
            'mipfacty': mip_fact_y,
            'mipfactz': mip_fact_z,
            **param
        }
        
        if extract_seg:
            self.data = data
            self.param['max_object_number'] = max_object_number
            self.param['mip_data_size'] = mip_data_size
            self.param['seg_layer_name'] = seg_layer_name
        
        print(f"Initialization complete. Extract mode: {'segmentation' if extract_seg else 'screenshots'}")
        
        # Continue to Part 2
        self._process_objects_and_blocks(extract_seg, data, names, mip_scale_matrix)
        
    def _setup_screenshot_names(self, param: Dict) -> List[str]:
        """Setup names for screenshot extraction modes"""
        extract_which = param.get('extractwhich', 5)
        
        if extract_which == 5:  # RGB 50%
            return ['Red Layer', 'Green Layer', 'Blue Layer']
        elif extract_which == 6:  # Brightness 50%
            param['lev'] = 128
            return ['Brightness 128']
        elif extract_which == 7:  # 16 levels
            param['lev'] = list(range(8, 257, 16))
            return [f'B{lev:03d}' for lev in param['lev']]
        elif extract_which == 8:  # 32 levels
            param['lev'] = list(range(4, 257, 8))
            return [f'B{lev:03d}' for lev in param['lev']]
        elif extract_which == 9:  # 64 levels
            param['lev'] = list(range(2, 257, 4))
            return [f'B{lev:03d}' for lev in param['lev']]
        else:
            return []
    
    def _check_connection(self) -> bool:
        """Check if connection to VAST is valid"""
        # Implement based on your GUI/connection requirements
        return True
    
    def _get_selected_seg_layer_name(self) -> str:
        """Get the name of the selected segmentation layer"""
        # Implement via VAST API
        return self.vast.get_selected_seg_layer_name()
    
    def _update_message(self, *messages):
        """Update progress message (implement for your GUI)"""
        print(' | '.join(messages))
        time.sleep(0.01)
    
    def _show_error(self, message: str):
        """Show error message (implement for your GUI)"""
        print(f"ERROR: {message}")
    
    def _process_objects_and_blocks(self, extract_seg: bool, data: np.ndarray, 
                                     names: List[str], mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 2: Process object selection, compute folder names, and calculate block divisions
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
            data: Segment data matrix (for segmentation mode)
            names: List of object/segment names
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        
        if extract_seg:
            # ===== COMPUTE FULL NAMES (including folder hierarchy) =====
            if param.get('includefoldernames', 0) == 1:
                logging.info("Computing full names with folder hierarchy...")
                full_names = names.copy()
                for i in range(len(data)):
                    j = i
                    # Traverse up the hierarchy (column 14 is parent ID)
                    while data[j, 13] != 0:  # Check if parent is not 0
                        j = int(data[j, 13])
                        full_names[i] = names[j] + '.' + full_names[i]
                names = full_names
                logging.debug(f"Sample full name: {names[0] if names else 'N/A'}")
            
            # ===== COMPUTE LIST OF OBJECTS TO EXPORT =====
            extract_which = param.get('extractwhich', 1)
            logging.info(f"Computing object list (extractwhich={extract_which})...")
            
            if extract_which == 1:
                # All segments individually, uncollapsed
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                self.vast.set_seg_translation(None, None)
                logging.info(f"Mode 1: Exporting {len(objects)} individual segments")
                
            elif extract_which == 2:
                # All segments, collapsed as in VAST
                objects = np.unique(data[:, 17])  # Column 18 in MATLAB (0-indexed: 17)
                objects = np.column_stack([objects, data[objects.astype(int), 1]]).astype(np.uint32)
                self.vast.set_seg_translation(data[:, 0], data[:, 17])
                logging.info(f"Mode 2: Exporting {len(objects)} collapsed segments")
                
            elif extract_which == 3:
                # Selected segment and children, uncollapsed
                selected = np.where((data[:, 1] & 65536) > 0)[0]
                if len(selected) == 0:
                    objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                    logging.warning("No segments selected, exporting all")
                else:
                    selected = np.concatenate([selected, self._get_child_tree_ids_seg(data, selected)])
                    objects = np.column_stack([selected, data[selected, 1]]).astype(np.uint32)
                    logging.info(f"Mode 3: Exporting {len(objects)} selected segments + children")
                self.vast.set_seg_translation(data[selected, 0], data[selected, 0])
                
            elif extract_which == 4:
                # Selected segment and children, collapsed as in VAST
                selected = np.where((data[:, 1] & 65536) > 0)[0]
                if len(selected) == 0:
                    # None selected: choose all, collapsed
                    selected = data[:, 0].astype(int)
                    objects = np.unique(data[:, 17])
                    logging.warning("No segments selected, exporting all collapsed")
                else:
                    selected = np.concatenate([selected, self._get_child_tree_ids_seg(data, selected)])
                    objects = np.unique(data[selected.astype(int), 17])
                    logging.info(f"Mode 4: Exporting {len(objects)} collapsed selected segments + children")
                
                objects = np.column_stack([objects, data[objects.astype(int), 1]]).astype(np.uint32)
                self.vast.set_seg_translation(data[selected.astype(int), 0], data[selected.astype(int), 17])
            
            else:
                logging.error(f"Unknown extractwhich value: {extract_which}")
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
            
            self.objects = objects
            param['objects'] = objects
        
        # ===== COMPUTE NUMBER OF BLOCKS/TILES IN VOLUME =====
        logging.info("Computing block divisions...")
        
        x_min, x_max = param['xmin'], param['xmax']
        y_min, y_max = param['ymin'], param['ymax']
        z_min, z_max = param['zmin'], param['zmax']
        
        block_size_x = param.get('blocksizex', 256)
        block_size_y = param.get('blocksizey', 256)
        block_size_z = param.get('blocksizez', 256)
        overlap = param.get('overlap', 0)
        slice_step = param.get('slicestep', 1)
        
        # Calculate X tiles
        nr_x_tiles = 0
        tile_x1 = x_min
        while tile_x1 <= x_max:
            tile_x1 += block_size_x - overlap
            nr_x_tiles += 1
        
        # Calculate Y tiles
        nr_y_tiles = 0
        tile_y1 = y_min
        while tile_y1 <= y_max:
            tile_y1 += block_size_y - overlap
            nr_y_tiles += 1
        
        # Calculate Z tiles
        nr_z_tiles = 0
        tile_z1 = z_min
        
        if slice_step == 1:
            slice_numbers = list(range(z_min, z_max + 1))
            while tile_z1 <= z_max:
                tile_z1 += block_size_z - overlap
                nr_z_tiles += 1
        else:
            slice_numbers = list(range(z_min, z_max + 1, slice_step))
            nr_z_tiles = ceil(len(slice_numbers) / (block_size_z - overlap))
            
            # Compute slice numbers for each block
            block_slice_numbers = []
            j = 0
            for p in range(0, len(slice_numbers), block_size_z - overlap):
                pe = min(p + block_size_z, len(slice_numbers))
                block_slice_numbers.append(slice_numbers[p:pe])
                j += 1
            param['block_slice_numbers'] = block_slice_numbers
        
        param['nr_x_tiles'] = nr_x_tiles
        param['nr_y_tiles'] = nr_y_tiles
        param['nr_z_tiles'] = nr_z_tiles
        param['slice_numbers'] = slice_numbers
        
        print(f"Block division: {nr_x_tiles} x {nr_y_tiles} x {nr_z_tiles} = {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")
        logging.info(f"Total blocks to process: {nr_x_tiles * nr_y_tiles * nr_z_tiles}")
        
        # Continue to Part 3 (MIP region constraint)
        if param.get('usemipregionconstraint', 0) == 1:
            self._compute_mip_region_constraint(extract_seg, mip_scale_matrix)
        
        # Continue to Part 4 (Initialize storage arrays)
        self._initialize_storage_arrays(extract_seg)
    
    def _get_child_tree_ids_seg(self, data: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
        """
        Recursively get all child IDs in the segment hierarchy
        
        Args:
            data: Segment data matrix
            parent_ids: Array of parent segment indices
            
        Returns:
            Array of all child indices
        """
        children = []
        for pid in parent_ids:
            # Find all segments where parent (column 14/index 13) equals this segment's ID
            child_mask = data[:, 13] == data[pid, 0]
            child_indices = np.where(child_mask)[0]
            if len(child_indices) > 0:
                children.extend(child_indices)
                # Recursively get children of children
                children.extend(self._get_child_tree_ids_seg(data, child_indices))
        
        return np.array(children, dtype=int) if children else np.array([], dtype=int)
    
    def _compute_mip_region_constraint(self, extract_seg: bool, mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 3: Compute MIP region constraint to skip empty blocks
        
        This loads a lower-resolution version of the data, creates a binary mask,
        dilates it by padding, and marks which blocks contain data.
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        rparam = {k: param[k] for k in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] if k in param}
        
        logging.info("Computing MIP region constraint...")
        print("Loading constraint mask at lower resolution...")
        
        # Calculate export region at constraint mip level
        mip_region_mip = param.get('mipregionmip', 0)
        
        c_x_min = rparam['xmin'] >> mip_region_mip
        c_x_max = (rparam['xmax'] >> mip_region_mip) - 1
        c_y_min = rparam['ymin'] >> mip_region_mip
        c_y_max = (rparam['ymax'] >> mip_region_mip) - 1
        c_z_min = rparam['zmin']
        c_z_max = rparam['zmax']
        
        if mip_region_mip > 0 and mip_scale_matrix is not None:
            c_z_scale = mip_scale_matrix[mip_region_mip][2]
            c_z_min = floor(c_z_min / c_z_scale)
            c_z_max = floor(c_z_max / c_z_scale)
        
        # Load complete region at constraint mip level
        if extract_seg:
            # Load segmentation data
            print(f"Loading segmentation at mip {mip_region_mip} for constraint...")
            logging.info(f"Loading segmentation constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
            
            slice_step = param.get('slicestep', 1)
            
            if slice_step == 1:
                mc_seg_image, values, numbers, bboxes, res = \
                    self.vast.get_seg_image_rle_decoded_bboxes(
                        mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, 0
                    )
            else:
                # Load slice by slice for non-unit step
                slices = list(range(c_z_min, c_z_max + 1, slice_step))
                mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                
                for idx, slice_num in enumerate(slices):
                    mc_seg_slice, values, numbers, bboxes, res = \
                        self.vast.get_seg_image_rle_decoded_bboxes(
                            mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                            slice_num, slice_num, 0
                        )
                    mc_seg_image[:, :, idx] = mc_seg_slice
        else:
            # Load screenshot data
            print(f"Loading screenshots at mip {mip_region_mip} for constraint...")
            logging.info(f"Loading screenshot constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
            
            slice_step = param.get('slicestep', 1)
            
            if slice_step == 1:
                mc_seg_image, res = self.vast.get_screenshot_image(
                    mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, 0
                )
                # Convert from (y, x, z, c) to (x, y, z) and make binary
                mc_seg_image = np.transpose(np.sum(mc_seg_image, axis=3) > 0, (1, 0, 2))
            else:
                slices = list(range(c_z_min, c_z_max + 1, slice_step))
                mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                
                for idx, slice_num in enumerate(slices):
                    mc_seg_slice, res = self.vast.get_screenshot_image(
                        mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                        slice_num, slice_num, 0
                    )
                    mc_seg_image[:, :, idx] = mc_seg_slice
                
                mc_seg_image = np.transpose(np.sum(mc_seg_image, axis=3) > 0, (1, 0, 2))
        
        # Convert to binary mask (VAST pre-translates so nonzero = exported objects)
        print("Processing constraint mask...")
        mc_seg_image = (mc_seg_image > 0).astype(np.uint8)
        
        # Dilate mask by region padding
        padding = param.get('mipregionpadding', 1)
        if padding > 0:
            kernel_size = padding * 2 + 1
            strel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
            mc_seg_image = ndimage.binary_dilation(mc_seg_image, structure=strel).astype(np.uint8)
            logging.info(f"Dilated mask by {padding} pixels")
        
        # Generate 3D matrix of block loading flags
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']
        mc_load_flags = np.zeros((nr_x_tiles, nr_y_tiles, nr_z_tiles), dtype=bool)
        
        # Calculate scale factors between constraint mip and export mip
        if mip_scale_matrix is not None:
            c_mip_fact_x = mip_scale_matrix[mip_region_mip][0] / mip_scale_matrix[param['miplevel']][0]
            c_mip_fact_y = mip_scale_matrix[mip_region_mip][1] / mip_scale_matrix[param['miplevel']][1]
            c_mip_fact_z = mip_scale_matrix[mip_region_mip][2] / mip_scale_matrix[param['miplevel']][2]
        else:
            c_mip_fact_x = c_mip_fact_y = c_mip_fact_z = 1
        
        # Iterate through all blocks and check if they contain data
        x_min, y_min, z_min = param['xmin'], param['ymin'], param['zmin']
        x_max, y_max, z_max = param['xmax'], param['ymax'], param['zmax']
        block_size_x = param.get('blocksizex', 256)
        block_size_y = param.get('blocksizey', 256)
        block_size_z = param.get('blocksizez', 256)
        overlap = param.get('overlap', 0)
        
        print("Checking blocks for data...")
        tile_z1 = z_min
        for tz in range(nr_z_tiles):
            tile_z2 = min(tile_z1 + block_size_z - 1, z_max)
            tile_y1 = y_min
            
            for ty in range(nr_y_tiles):
                tile_y2 = min(tile_y1 + block_size_y - 1, y_max)
                tile_x1 = x_min
                
                for tx in range(nr_x_tiles):
                    tile_x2 = min(tile_x1 + block_size_x - 1, x_max)
                    
                    # Compute tile coords on constraint mip
                    c_min_x = max(0, int(floor((tile_x1 - x_min) / c_mip_fact_x)))
                    c_max_x = min(mc_seg_image.shape[0] - 1, int(ceil((tile_x2 - x_min) / c_mip_fact_x)))
                    c_min_y = max(0, int(floor((tile_y1 - y_min) / c_mip_fact_y)))
                    c_max_y = min(mc_seg_image.shape[1] - 1, int(ceil((tile_y2 - y_min) / c_mip_fact_y)))
                    c_min_z = max(0, int(floor((tile_z1 - z_min) / c_mip_fact_z)))
                    c_max_z = min(mc_seg_image.shape[2] - 1, int(ceil((tile_z2 - z_min) / c_mip_fact_z)))
                    
                    # Crop region from constraint mip
                    crop_region = mc_seg_image[c_min_x:c_max_x+1, c_min_y:c_max_y+1, c_min_z:c_max_z+1]
                    
                    # Flag will be True if any voxels in crop region are nonzero
                    mc_load_flags[tx, ty, tz] = np.max(crop_region) > 0
                    
                    tile_x1 += block_size_x - overlap
                tile_y1 += block_size_y - overlap
            tile_z1 += block_size_z - overlap
        
        blocks_to_process = np.sum(mc_load_flags)
        blocks_to_skip = mc_load_flags.size - blocks_to_process
        print(f"Constraint check complete: {blocks_to_process} blocks to process, {blocks_to_skip} blocks to skip")
        logging.info(f"MIP constraint reduces workload by {100 * blocks_to_skip / mc_load_flags.size:.1f}%")
        
        param['mc_load_flags'] = mc_load_flags
    
    def _initialize_storage_arrays(self, extract_seg: bool):
        """
        Part 4: Initialize storage arrays for faces and vertices
        
        Creates cell/dict structures to store mesh data for each object in each block.
        For mode 10 (unique colors), uses sparse indexing.
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
        """
        param = self.param
        
        logging.info("Initializing storage arrays...")
        
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']
        
        if extract_seg:
            # Storage for segmentation objects
            max_obj_num = param['max_object_number']
            
            # Initialize as nested dictionaries: farray[obj][tx][ty][tz] = faces
            # Using dict for sparse storage (most cells will be empty)
            param['farray'] = {}
            param['varray'] = {}
            param['object_volume'] = np.zeros(len(param['objects']), dtype=np.int64)
            
            print(f"Initialized storage for {max_obj_num} potential objects across {nr_x_tiles}x{nr_y_tiles}x{nr_z_tiles} blocks")
            logging.info(f"Storage size: {len(param['objects'])} objects x {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")
            
        else:
            # Storage for screenshot-based extraction
            extract_which = param.get('extractwhich', 5)
            
            if extract_which == 10:
                # Up to 2^24 unique color objects
                # Use sparse matrix for indexing: fvindex[color, block_idx] = block_number
                total_colors = 256 * 256 * 256
                total_blocks = (nr_x_tiles + 1) * (nr_y_tiles + 1) * (nr_z_tiles + 1)
                
                # Sparse matrix would be huge, use dict instead
                param['fvindex'] = {}  # Will map (color, block_idx) -> block_number
                param['farray'] = {}   # Will map block_number -> faces
                param['varray'] = {}   # Will map block_number -> vertices
                param['object_volume'] = np.zeros(total_colors, dtype=np.int64)
                
                print(f"Initialized storage for unique color extraction (up to {total_colors} colors)")
                logging.info("Using sparse storage for color-based extraction")
                
            else:
                # RGB layers or brightness levels (small number of objects)
                num_objects = len(param['objects'])
                
                param['farray'] = {}
                param['varray'] = {}
                param['object_volume'] = np.zeros(num_objects, dtype=np.int64)
                
                print(f"Initialized storage for {num_objects} screenshot-based objects")
                logging.info(f"Storage for {num_objects} objects x {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")
    
    def _extract_block_surfaces(self, extract_seg: bool, mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 5: Main extraction loop - process each block and extract isosurfaces
        
        Iterates through all blocks in the volume, loads data, and extracts surfaces
        using marching cubes algorithm.
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        data = self.data if extract_seg else None
        
        x_min, y_min, z_min = param['xmin'], param['ymin'], param['zmin']
        x_max, y_max, z_max = param['xmax'], param['ymax'], param['zmax']
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']
        block_size_x = param.get('blocksizex', 256)
        block_size_y = param.get('blocksizey', 256)
        block_size_z = param.get('blocksizez', 256)
        overlap = param.get('overlap', 0)
        
        mc_load_flags = param.get('mc_load_flags', None)
        use_constraint = mc_load_flags is not None
        
        print("=" * 60)
        print("Starting surface extraction...")
        print("=" * 60)
        
        block_nr = 0  # For mode 10 color tracking
        total_blocks = nr_x_tiles * nr_y_tiles * nr_z_tiles
        processed_blocks = 0
        skipped_blocks = 0
        
        # Main triple loop through blocks
        tile_z1 = z_min
        for tz in range(nr_z_tiles):
            if self.canceled:
                break
                
            tile_z2 = min(tile_z1 + block_size_z - 1, z_max)
            tile_zs = tile_z2 - tile_z1 + 1
            
            tile_y1 = y_min
            for ty in range(nr_y_tiles):
                if self.canceled:
                    break
                    
                tile_y2 = min(tile_y1 + block_size_y - 1, y_max)
                tile_ys = tile_y2 - tile_y1 + 1
                
                tile_x1 = x_min
                for tx in range(nr_x_tiles):
                    if self.canceled:
                        break
                        
                    tile_x2 = min(tile_x1 + block_size_x - 1, x_max)
                    tile_xs = tile_x2 - tile_x1 + 1
                    
                    # Check if we should skip this block
                    if use_constraint and not mc_load_flags[tx, ty, tz]:
                        skipped_blocks += 1
                        tile_x1 += block_size_x - overlap
                        continue
                    
                    # Process this block
                    processed_blocks += 1
                    if processed_blocks % 10 == 0 or processed_blocks == 1:
                        progress = 100 * processed_blocks / (total_blocks - skipped_blocks)
                        print(f"Processing block ({tx+1},{ty+1},{tz+1}) of ({nr_x_tiles},{nr_y_tiles},{nr_z_tiles}) - {progress:.1f}% complete")
                    
                    if extract_seg:
                        self._process_segmentation_block(
                            tx, ty, tz, tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2,
                            tile_xs, tile_ys, tile_zs, data, mip_scale_matrix
                        )
                    else:
                        self._process_screenshot_block(
                            tx, ty, tz, tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2,
                            tile_xs, tile_ys, tile_zs, mip_scale_matrix, block_nr
                        )
                        block_nr += 1
                    
                    tile_x1 += block_size_x - overlap
                tile_y1 += block_size_y - overlap
            tile_z1 += block_size_z - overlap
        
        if extract_seg:
            # Clear segment translation
            self.vast.set_seg_translation(None, None)
        
        print("=" * 60)
        if self.canceled:
            print("Extraction CANCELED")
        else:
            print(f"Extraction complete! Processed {processed_blocks} blocks, skipped {skipped_blocks}")
        print("=" * 60)
        
        logging.info(f"Block processing complete: {processed_blocks} processed, {skipped_blocks} skipped")

    def _process_segmentation_block(self, tx: int, ty: int, tz: int,
                                     tile_x1: int, tile_x2: int, 
                                     tile_y1: int, tile_y2: int,
                                     tile_z1: int, tile_z2: int,
                                     tile_xs: int, tile_ys: int, tile_zs: int,
                                     data: np.ndarray, mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 5a: Process a single segmentation block
        
        Loads segmentation data for this block, optionally applies erosion/dilation,
        adds boundary slices for surface closing, then extracts surfaces for each object.
        
        Args:
            tx, ty, tz: Block indices
            tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2: Block boundaries
            tile_xs, tile_ys, tile_zs: Block sizes
            data: Segment data matrix
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        
        logging.debug(f"Loading segmentation block ({tx},{ty},{tz})...")
        
        # Determine erosion/dilation padding
        mip_data_size = param.get('mip_data_size', None)
        if param.get('erodedilate', 0) == 1 and mip_data_size is not None:
            edx = np.ones((3, 2), dtype=int)
            if tile_x1 == 0:
                edx[0, 0] = 0
            if tile_x2 >= mip_data_size[0]:
                edx[0, 1] = 0
            if tile_y1 == 0:
                edx[1, 0] = 0
            if tile_y2 >= mip_data_size[1]:
                edx[1, 1] = 0
            if tile_z1 == 0:
                edx[2, 0] = 0
            if tile_z2 >= mip_data_size[2]:
                edx[2, 1] = 0
        else:
            edx = np.zeros((3, 2), dtype=int)
        
        # Load segmentation data
        slice_step = param.get('slicestep', 1)
        max_object_number = param.get('max_object_number', 0)
        
        if slice_step == 1:
            seg_image, values, numbers, bboxes, res = \
                self.vast.get_seg_image_rle_decoded_bboxes(
                    param['miplevel'],
                    tile_x1 - edx[0, 0], tile_x2 + edx[0, 1],
                    tile_y1 - edx[1, 0], tile_y2 + edx[1, 1],
                    tile_z1 - edx[2, 0], tile_z2 + edx[2, 1],
                    0
                )
        else:
            # Load slice-by-slice for non-unit step
            block_slice_numbers = param.get('block_slice_numbers', [[]])[tz]
            bs = block_slice_numbers.copy()
            
            # Add padding slices if needed for erosion/dilation
            if edx[2, 0] == 1:
                bs = [bs[0] - slice_step] + bs
            if edx[2, 1] == 1:
                bs = bs + [bs[-1] + slice_step]
            
            seg_image = np.zeros((tile_x2 - tile_x1 + 1 + edx[0, 0] + edx[0, 1],
                                 tile_y2 - tile_y1 + 1 + edx[1, 0] + edx[1, 1],
                                 len(bs)), dtype=np.uint16)
            
            num_arr = np.zeros(max_object_number, dtype=np.int32)
            bbox_arr = np.full((max_object_number, 6), -1, dtype=float)
            first_block_slice = bs[0]
            
            for i, slice_num in enumerate(bs):
                s_seg_image, s_values, s_numbers, s_bboxes, res = \
                    self.vast.get_seg_image_rle_decoded_bboxes(
                        param['miplevel'],
                        tile_x1 - edx[0, 0], tile_x2 + edx[0, 1],
                        tile_y1 - edx[1, 0], tile_y2 + edx[1, 1],
                        slice_num, slice_num, 0
                    )
                seg_image[:, :, i] = s_seg_image
                
                # Remove zero values
                if s_values is not None and len(s_values) > 0:
                    mask = s_values != 0
                    s_values = s_values[mask]
                    s_numbers = s_numbers[mask]
                    s_bboxes = s_bboxes[mask]
                    
                    # Adjust bboxes for z
                    s_bboxes[:, [2, 5]] = s_bboxes[:, [2, 5]] + i
                    
                    if len(s_values) > 0:
                        num_arr[s_values] += s_numbers
                        bbox_arr[s_values] = self._expand_bounding_boxes(bbox_arr[s_values], s_bboxes)
            
            values = np.where(num_arr > 0)[0]
            numbers = num_arr[values]
            bboxes = bbox_arr[values]
        
        # Apply erosion/dilation if requested
        if param.get('erodedilate', 0) == 1:
            strel = np.ones((2, 2, 2), dtype=np.uint8)
            # Opening = erosion followed by dilation (removes 1-voxel thin objects)
            seg_image = ndimage.binary_opening(seg_image, structure=strel).astype(np.uint16)
            
            # Crop padding
            seg_image = seg_image[
                edx[0, 0]:(seg_image.shape[0] - edx[0, 1]) if edx[0, 1] > 0 else seg_image.shape[0],
                edx[1, 0]:(seg_image.shape[1] - edx[1, 1]) if edx[1, 1] > 0 else seg_image.shape[1],
                edx[2, 0]:(seg_image.shape[2] - edx[2, 1]) if edx[2, 1] > 0 else seg_image.shape[2]
            ]
            
            # Adjust bboxes
            if bboxes is not None and len(bboxes) > 0:
                if edx[0, 0] > 0:
                    bb = bboxes[:, [0, 3]] - 1
                    bb[bb == 0] = 1
                    bboxes[:, [0, 3]] = bb
                if edx[0, 1] > 0:
                    bb = bboxes[:, [0, 3]]
                    bb[bb > seg_image.shape[0]] = seg_image.shape[0]
                    bboxes[:, [0, 3]] = bb
                
                if edx[1, 0] > 0:
                    bb = bboxes[:, [1, 4]] - 1
                    bb[bb == 0] = 1
                    bboxes[:, [1, 4]] = bb
                if edx[1, 1] > 0:
                    bb = bboxes[:, [1, 4]]
                    bb[bb > seg_image.shape[1]] = seg_image.shape[1]
                    bboxes[:, [1, 4]] = bb
                
                if edx[2, 0] > 0:
                    bb = bboxes[:, [2, 5]] - 1
                    bb[bb == 0] = 1
                    bboxes[:, [2, 5]] = bb
                if edx[2, 1] > 0:
                    bb = bboxes[:, [2, 5]]
                    bb[bb > seg_image.shape[2]] = seg_image.shape[2]
                    bboxes[:, [2, 5]] = bb
        
        # Process segmentation data
        logging.debug(f"Processing {len(values)} objects in block ({tx},{ty},{tz})...")
        
        # Remove zero values from consideration
        if values is not None and len(values) > 0:
            mask = values != 0
            values = values[mask]
            numbers = numbers[mask]
            bboxes = bboxes[mask]
        
        if values is None or len(values) == 0:
            logging.debug(f"No objects in block ({tx},{ty},{tz})")
            return
        
        # Adjust for surface closing (add boundary slices)
        x_vofs = y_vofs = z_vofs = 0
        tt_xs, tt_ys, tt_zs = tile_xs, tile_ys, tile_zs
        
        if param.get('closesurfaces', 0) == 1:
            if tx == 0:
                seg_image = np.concatenate([np.zeros((1, seg_image.shape[1], seg_image.shape[2]), dtype=seg_image.dtype), 
                                           seg_image], axis=0)
                bboxes[:, 0] += 1
                bboxes[:, 3] += 1
                x_vofs -= 1
                tt_xs += 1
            if ty == 0:
                seg_image = np.concatenate([np.zeros((seg_image.shape[0], 1, seg_image.shape[2]), dtype=seg_image.dtype), 
                                           seg_image], axis=1)
                bboxes[:, 1] += 1
                bboxes[:, 4] += 1
                y_vofs -= 1
                tt_ys += 1
            if tz == 0:
                seg_image = np.concatenate([np.zeros((seg_image.shape[0], seg_image.shape[1], 1), dtype=seg_image.dtype), 
                                           seg_image], axis=2)
                bboxes[:, 2] += 1
                bboxes[:, 5] += 1
                z_vofs -= 1
                tt_zs += 1
            if tx == param['nr_x_tiles'] - 1:
                seg_image = np.concatenate([seg_image,
                                           np.zeros((1, seg_image.shape[1], seg_image.shape[2]), dtype=seg_image.dtype)], axis=0)
                tt_xs += 1
            if ty == param['nr_y_tiles'] - 1:
                seg_image = np.concatenate([seg_image,
                                           np.zeros((seg_image.shape[0], 1, seg_image.shape[2]), dtype=seg_image.dtype)], axis=1)
                tt_ys += 1
            if tz == param['nr_z_tiles'] - 1:
                seg_image = np.concatenate([seg_image,
                                           np.zeros((seg_image.shape[0], seg_image.shape[1], 1), dtype=seg_image.dtype)], axis=2)
                tt_zs += 1
        
        # Extract surfaces for each segment
        first_block_slice = param.get('block_slice_numbers', [[tile_z1]])[tz][0] if slice_step > 1 else tile_z1
        
        for seg_nr, seg in enumerate(values):
            if self.canceled:
                break
            
            # Get bounding box and expand by 1 voxel
            bbx = bboxes[seg_nr].copy()
            bbx += np.array([-1, -1, -1, 1, 1, 1])
            bbx[0] = max(1, bbx[0])
            bbx[1] = max(1, bbx[1])
            bbx[2] = max(1, bbx[2])
            bbx[3] = min(tt_xs, bbx[3])
            bbx[4] = min(tt_ys, bbx[4])
            bbx[5] = min(tt_zs, bbx[5])
            
            # Ensure at least 2 pixels in each direction
            if bbx[0] == bbx[3]:
                if bbx[0] > 1:
                    bbx[0] -= 1
                else:
                    bbx[3] += 1
            if bbx[1] == bbx[4]:
                if bbx[1] > 1:
                    bbx[1] -= 1
                else:
                    bbx[4] += 1
            if bbx[2] == bbx[5]:
                if bbx[2] > 1:
                    bbx[2] -= 1
                else:
                    bbx[5] += 1
            
            # Extract subsegment (convert to 0-indexed)
            bbx_int = bbx.astype(int) - 1  # MATLAB is 1-indexed
            subseg = seg_image[bbx_int[0]:bbx_int[3]+1, bbx_int[1]:bbx_int[4]+1, bbx_int[2]:bbx_int[5]+1]
            subseg = (subseg == seg).astype(float)
            
            # Extract isosurface using marching cubes
            try:
                verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)
                
                if len(verts) > 0:
                    # Adjust coordinates for bounding box offset
                    verts[:, 0] += bbx_int[1] + y_vofs  # Y in MATLAB order
                    verts[:, 1] += bbx_int[0] + x_vofs  # X in MATLAB order
                    verts[:, 2] += bbx_int[2] + z_vofs  # Z
                    
                    # Adjust for tile position
                    verts[:, 0] += tile_y1
                    verts[:, 1] += tile_x1
                    
                    if slice_step == 1:
                        verts[:, 2] += tile_z1
                    else:
                        verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice
                    
                    # Scale to physical units
                    verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']
                    verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']
                    verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']
                    
                    # Store faces and vertices (convert faces to 1-indexed for OBJ format later)
                    param['farray'][(int(seg), tx, ty, tz)] = faces
                    param['varray'][(int(seg), tx, ty, tz)] = verts
                    
            except Exception as e:
                logging.warning(f"Failed to extract surface for segment {seg} in block ({tx},{ty},{tz}): {e}")
    
    def _expand_bounding_boxes(self, bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
        """
        Expand bounding boxes to encompass both input boxes
        
        Args:
            bbox1: First bounding box array (N x 6)
            bbox2: Second bounding box array (N x 6)
            
        Returns:
            Expanded bounding boxes
        """
        if bbox1.ndim == 1:
            bbox1 = bbox1.reshape(1, -1)
        if bbox2.ndim == 1:
            bbox2 = bbox2.reshape(1, -1)
        
        result = bbox1.copy()
        
        # For uninitialized boxes (-1), use bbox2
        uninit_mask = bbox1[:, 0] == -1
        result[uninit_mask] = bbox2[uninit_mask]
        
        # For initialized boxes, take min/max
        init_mask = ~uninit_mask
        if np.any(init_mask):
            result[init_mask, 0] = np.minimum(bbox1[init_mask, 0], bbox2[init_mask, 0])
            result[init_mask, 1] = np.minimum(bbox1[init_mask, 1], bbox2[init_mask, 1])
            result[init_mask, 2] = np.minimum(bbox1[init_mask, 2], bbox2[init_mask, 2])
            result[init_mask, 3] = np.maximum(bbox1[init_mask, 3], bbox2[init_mask, 3])
            result[init_mask, 4] = np.maximum(bbox1[init_mask, 4], bbox2[init_mask, 4])
            result[init_mask, 5] = np.maximum(bbox1[init_mask, 5], bbox2[init_mask, 5])
        
        return result

    def _process_screenshot_block(self, tx: int, ty: int, tz: int,
                                   tile_x1: int, tile_x2: int,
                                   tile_y1: int, tile_y2: int,
                                   tile_z1: int, tile_z2: int,
                                   tile_xs: int, tile_ys: int, tile_zs: int,
                                   mip_scale_matrix: Optional[np.ndarray],
                                   block_nr: int):
        """
        Part 5b: Process a single screenshot block
        
        Loads screenshot/image data for this block, extracts RGB layers or brightness levels,
        and generates surfaces for each layer/level or unique color.
        
        Args:
            tx, ty, tz: Block indices
            tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2: Block boundaries
            tile_xs, tile_ys, tile_zs: Block sizes
            mip_scale_matrix: Mipmap scaling factors
            block_nr: Block number (for mode 10 color tracking)
        """
        param = self.param
        
        logging.debug(f"Loading screenshot block ({tx},{ty},{tz})...")
        
        # Load screenshot data
        slice_step = param.get('slicestep', 1)
        
        if slice_step == 1:
            scs_image, res = self.vast.get_screenshot_image(
                param['miplevel'],
                tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2, 1, 1
            )
            
            # Handle single slice case (returns 3D instead of 4D)
            if tile_z1 == tile_z2 and scs_image.ndim == 3:
                scs_image = scs_image.reshape(scs_image.shape[0], scs_image.shape[1], 1, scs_image.shape[2])
        else:
            # Load slice-by-slice
            block_slice_numbers = param.get('block_slice_numbers', [[]])[tz]
            bs = block_slice_numbers
            
            scs_image = np.zeros((tile_y2 - tile_y1 + 1, tile_x2 - tile_x1 + 1, len(bs), 3), dtype=np.uint8)
            first_block_slice = bs[0]
            
            for i, slice_num in enumerate(bs):
                scs_slice, res = self.vast.get_screenshot_image(
                    param['miplevel'],
                    tile_x1, tile_x2, tile_y1, tile_y2,
                    slice_num, slice_num, 1, 1
                )
                scs_image[:, :, i, :] = scs_slice
        
        logging.debug(f"Processing screenshot block ({tx},{ty},{tz})...")
        
        # Convert from (y, x, z, c) to (x, y, z, c) and extract channels
        scs_image = np.transpose(scs_image, (1, 0, 2, 3))
        r_cube = scs_image[:, :, :, 0]
        g_cube = scs_image[:, :, :, 1]
        b_cube = scs_image[:, :, :, 2]
        
        # Handle surface closing
        x_vofs = y_vofs = z_vofs = 0
        tt_xs, tt_ys, tt_zs = tile_xs, tile_ys, tile_zs
        
        if param.get('closesurfaces', 0) == 1:
            if tx == 0:
                r_cube = np.concatenate([np.zeros((1, r_cube.shape[1], r_cube.shape[2]), dtype=r_cube.dtype), r_cube], axis=0)
                g_cube = np.concatenate([np.zeros((1, g_cube.shape[1], g_cube.shape[2]), dtype=g_cube.dtype), g_cube], axis=0)
                b_cube = np.concatenate([np.zeros((1, b_cube.shape[1], b_cube.shape[2]), dtype=b_cube.dtype), b_cube], axis=0)
                x_vofs = -1
                tt_xs += 1
            if ty == 0:
                r_cube = np.concatenate([np.zeros((r_cube.shape[0], 1, r_cube.shape[2]), dtype=r_cube.dtype), r_cube], axis=1)
                g_cube = np.concatenate([np.zeros((g_cube.shape[0], 1, g_cube.shape[2]), dtype=g_cube.dtype), g_cube], axis=1)
                b_cube = np.concatenate([np.zeros((b_cube.shape[0], 1, b_cube.shape[2]), dtype=b_cube.dtype), b_cube], axis=1)
                y_vofs = -1
                tt_ys += 1
            if tz == 0:
                r_cube = np.concatenate([np.zeros((r_cube.shape[0], r_cube.shape[1], 1), dtype=r_cube.dtype), r_cube], axis=2)
                g_cube = np.concatenate([np.zeros((g_cube.shape[0], g_cube.shape[1], 1), dtype=g_cube.dtype), g_cube], axis=2)
                b_cube = np.concatenate([np.zeros((b_cube.shape[0], b_cube.shape[1], 1), dtype=b_cube.dtype), b_cube], axis=2)
                z_vofs = -1
                tt_zs += 1
            if tx == param['nr_x_tiles'] - 1:
                r_cube = np.concatenate([r_cube, np.zeros((1, r_cube.shape[1], r_cube.shape[2]), dtype=r_cube.dtype)], axis=0)
                g_cube = np.concatenate([g_cube, np.zeros((1, g_cube.shape[1], g_cube.shape[2]), dtype=g_cube.dtype)], axis=0)
                b_cube = np.concatenate([b_cube, np.zeros((1, b_cube.shape[1], b_cube.shape[2]), dtype=b_cube.dtype)], axis=0)
                tt_xs += 1
            if ty == param['nr_y_tiles'] - 1:
                r_cube = np.concatenate([r_cube, np.zeros((r_cube.shape[0], 1, r_cube.shape[2]), dtype=r_cube.dtype)], axis=1)
                g_cube = np.concatenate([g_cube, np.zeros((g_cube.shape[0], 1, g_cube.shape[2]), dtype=g_cube.dtype)], axis=1)
                b_cube = np.concatenate([b_cube, np.zeros((b_cube.shape[0], 1, b_cube.shape[2]), dtype=b_cube.dtype)], axis=1)
                tt_ys += 1
            if tz == param['nr_z_tiles'] - 1:
                r_cube = np.concatenate([r_cube, np.zeros((r_cube.shape[0], r_cube.shape[1], 1), dtype=r_cube.dtype)], axis=2)
                g_cube = np.concatenate([g_cube, np.zeros((g_cube.shape[0], g_cube.shape[1], 1), dtype=g_cube.dtype)], axis=2)
                b_cube = np.concatenate([b_cube, np.zeros((b_cube.shape[0], b_cube.shape[1], 1), dtype=b_cube.dtype)], axis=2)
                tt_zs += 1
        
        first_block_slice = param.get('block_slice_numbers', [[tile_z1]])[tz][0] if slice_step > 1 else tile_z1
        
        # Extract isosurfaces based on mode
        extract_which = param.get('extractwhich', 5)
        
        if extract_which == 10:
            # Extract unique colors as individual objects
            self._extract_unique_colors(
                r_cube, g_cube, b_cube, tx, ty, tz,
                tile_x1, tile_y1, tile_z1,
                x_vofs, y_vofs, z_vofs,
                first_block_slice, slice_step, block_nr
            )
        else:
            # Extract RGB layers or brightness levels
            self._extract_layers_or_levels(
                r_cube, g_cube, b_cube, tx, ty, tz,
                tile_x1, tile_y1, tile_z1,
                x_vofs, y_vofs, z_vofs,
                first_block_slice, slice_step, extract_which
            )
    
    def _extract_unique_colors(self, r_cube: np.ndarray, g_cube: np.ndarray, b_cube: np.ndarray,
                               tx: int, ty: int, tz: int,
                               tile_x1: int, tile_y1: int, tile_z1: int,
                               x_vofs: int, y_vofs: int, z_vofs: int,
                               first_block_slice: int, slice_step: int, block_nr: int):
        """
        Extract unique colors as individual 3D objects (mode 10)
        
        Args:
            r_cube, g_cube, b_cube: RGB data cubes
            tx, ty, tz: Block indices
            tile_x1, tile_y1, tile_z1: Tile positions
            x_vofs, y_vofs, z_vofs: Offset for surface closing
            first_block_slice: First slice number in this block
            slice_step: Slice stepping factor
            block_nr: Block number for indexing
        """
        param = self.param
        
        # Combine RGB into single color value (R << 16 | G << 8 | B)
        col_cube = (r_cube.astype(np.int32) << 16) + (g_cube.astype(np.int32) << 8) + b_cube.astype(np.int32)
        
        # Count unique colors (histogram)
        unique_colors, counts = np.unique(col_cube[col_cube != 0], return_counts=True)
        
        if len(unique_colors) == 0:
            return
        
        # Update volume counts
        for color, count in zip(unique_colors, counts):
            param['object_volume'][color] += count
        
        logging.debug(f"Found {len(unique_colors)} unique colors in block ({tx},{ty},{tz})")
        
        # Extract surface for each unique color
        for col_idx, color in enumerate(unique_colors):
            if self.canceled:
                break
            
            # Create binary mask for this color
            subseg = (col_cube == color).astype(float)
            
            # Skip if dimensions are invalid
            if subseg.ndim != 3:
                continue
            
            # Extract isosurface
            try:
                verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)
                
                if len(verts) > 0:
                    # Adjust coordinates
                    verts[:, 0] += y_vofs
                    verts[:, 1] += x_vofs
                    verts[:, 2] += z_vofs
                    
                    verts[:, 0] += tile_y1
                    verts[:, 1] += tile_x1
                    
                    if slice_step == 1:
                        verts[:, 2] += tile_z1
                    else:
                        verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice
                    
                    # Scale to physical units
                    verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']
                    verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']
                    verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']
                    
                    # Store with block indexing
                    idx = tz * param['nr_y_tiles'] * param['nr_x_tiles'] + ty * param['nr_x_tiles'] + tx
                    param['fvindex'][(int(color), idx)] = block_nr
                    param['farray'][block_nr] = faces
                    param['varray'][block_nr] = verts
                    
            except Exception as e:
                logging.warning(f"Failed to extract color {color:06X} in block ({tx},{ty},{tz}): {e}")
    
    def _extract_layers_or_levels(self, r_cube: np.ndarray, g_cube: np.ndarray, b_cube: np.ndarray,
                                  tx: int, ty: int, tz: int,
                                  tile_x1: int, tile_y1: int, tile_z1: int,
                                  x_vofs: int, y_vofs: int, z_vofs: int,
                                  first_block_slice: int, slice_step: int, extract_which: int):
        """
        Extract RGB layers or brightness levels (modes 5-9)
        
        Args:
            r_cube, g_cube, b_cube: RGB data cubes
            tx, ty, tz: Block indices
            tile_x1, tile_y1, tile_z1: Tile positions
            x_vofs, y_vofs, z_vofs: Offset for surface closing
            first_block_slice: First slice number in this block
            slice_step: Slice stepping factor
            extract_which: Extraction mode (5=RGB, 6-9=brightness levels)
        """
        param = self.param
        
        # Compute brightness cube if needed
        if extract_which in [6, 7, 8, 9]:
            cube = ((r_cube.astype(np.int32) + g_cube.astype(np.int32) + b_cube.astype(np.int32)) / 3).astype(np.uint8)
        
        # Process each object/layer
        objects = param.get('objects', [])
        
        for obj_idx in range(len(objects)):
            if self.canceled:
                break
            
            # Determine which subsegment to extract
            if extract_which == 5:  # RGB layers
                if obj_idx == 0:
                    subseg = (r_cube > 128).astype(float)
                elif obj_idx == 1:
                    subseg = (g_cube > 128).astype(float)
                elif obj_idx == 2:
                    subseg = (b_cube > 128).astype(float)
                else:
                    continue
            elif extract_which in [6, 7, 8, 9]:  # Brightness levels
                lev = param.get('lev', [128])
                if obj_idx >= len(lev):
                    continue
                subseg = (cube > lev[obj_idx]).astype(float)
            else:
                continue
            
            # Skip if dimensions are invalid
            if subseg.ndim != 3:
                continue
            
            # Extract isosurface
            try:
                verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)
                
                if len(verts) > 0:
                    # Adjust coordinates
                    verts[:, 0] += y_vofs
                    verts[:, 1] += x_vofs
                    verts[:, 2] += z_vofs
                    
                    verts[:, 0] += tile_y1
                    verts[:, 1] += tile_x1
                    
                    if slice_step == 1:
                        verts[:, 2] += tile_z1
                    else:
                        verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice
                    
                    # Scale to physical units
                    verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']
                    verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']
                    verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']
                    
                    # Store faces and vertices
                    param['farray'][(obj_idx, tx, ty, tz)] = faces
                    param['varray'][(obj_idx, tx, ty, tz)] = verts
                    
            except Exception as e:
                logging.warning(f"Failed to extract object {obj_idx} in block ({tx},{ty},{tz}): {e}")


# Example usage:
if __name__ == "__main__":
    # Initialize VAST control
    vast = VASTControlClass()
    
    # Setup export parameters
    export_params = {
        'extractwhich': 1,  # 1=all segments, 2=collapsed, 3=selected+children, etc.
        'miplevel': 0,
        'blocksizex': 256,
        'blocksizey': 256,
        'blocksizez': 256,
        'overlap': 0,
        'closesurfaces': 0,
        'slicestep': 1,
        'disablenetwarnings': 1,
        'xscale': 1.0,
        'yscale': 1.0,
        'zscale': 1.0,
        'xunit': 1.0,
        'yunit': 1.0,
        'zunit': 1.0,
    }
    
    # Setup region parameters
    region_params = {
        'xmin': 0,
        'xmax': 512,
        'ymin': 0,
        'ymax': 512,
        'zmin': 0,
        'zmax': 100,
    }
    
    # Create extractor and run
    extractor = SurfaceExtractor(vast, export_params, region_params)
    extractor.extract_surfaces()