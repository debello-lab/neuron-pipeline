import numpy as np
from skimage import measure
from scipy import ndimage, sparse
from math import floor, ceil
import re
from typing import Tuple, List, Dict, Optional
import time
import logging
from VCC import VASTControlClass

class SurfaceExtractor:
    """
        Python port of MATLAB function extractsurfaces from VAST.
        Extracts 3D surfaces from segmentation or screenshot data.
    """

    def __init__(self, vast_control: VASTControlClass, export_params: dict, region_params: dict):
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
        self.param = {**export_params}
        self.objects = None
        self.names = []
        self.data = None

    def _setup_screenshot_names(self, param: dict) -> List[str]:
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

    def extract_surfaces(self):
        """
        Extract 3D surfaces from VAST segmentation or screenshot data.

        Args:
            params: Dictionary with export parameters
        """
        # Use the VAST instance passed in constructor (already connected)
        # Don't create a new connection
        if not self.vast:
            print("ERROR: No VAST connection provided")
            return None

        if self.export_params.get('disablenetwarnings', 0) == 1:
            self.vast.set_error_popups_enabled(74, False)  # network connection error
            self.vast.set_error_popups_enabled(75, False)  # unexpected data
            self.vast.set_error_popups_enabled(76, False)  # unexpected length

        print('Exporting Surfaces ...','Loading Metadata ...')

        param = self.export_params.copy()
        rparam = self.region_params.copy()

        # Determine if extracting from segmentation or screenshots
        extract_seg = True
        if 5 <= param.get('extractwhich', 1) <= 10:
            extract_seg = False


        if extract_seg:
            # ===== SEGMENTATION EXPORT =====
            data, res = self.vast.get_all_segment_data_matrix()
            if len(data) == 0 or data is None:
                print("Did not get data from segment data matrix")
                return
            # Extract data matrix from return tuple if needed and convert to numpy array
            if isinstance(data, tuple):
                data = data[0]  # First element is the data matrix
            data = np.array(data)  # Convert to NumPy array for indexing
            names = self.vast.get_all_segment_names()
            if names is None: 
                print("Did not get segment names")
                return
            seg_layer_name = self.vast.get_all_segment_names()
            if seg_layer_name is None: 
                print("Did not get segment layer name")
                return
            
            names = names[1:] # Remove background name
 
            max_object_number = int(np.max(data[0]))
            
            # Get segment data
            all_seg_data = self.vast.get_all_segment_data()
            if all_seg_data is None:
                print("Failed to get segment data")
                return
            names = self.vast.get_all_segment_names()
            
            # Get selected layer
            selected_layers = self.vast.get_selected_layer_nr()
            if not selected_layers or selected_layers.get('selected_segment_layer', -1) < 0:
                print("No segment layer selected")
                return
            
            selected_seg_layer_nr = selected_layers['selected_segment_layer']
            if selected_seg_layer_nr < 0:
                print("No segment layer selected")
                return
            selected_em_layer_nr = selected_layers.get('selected_em_layer', -1)

            mip_data_size = self.vast.get_data_size_at_mip(param['miplevel'], selected_seg_layer_nr)
        else:
            # ===== SCREENSHOT EXPORT =====
            data = [[]]
            names = []
            seg_layer_name = []
            max_object_number = 0
            mip_data_size = None
            names = self._setup_screenshot_names(param)

            selected_layers = self.vast.get_selected_layer_nr()

            if not selected_layers or selected_layers.get('selected_segment_layer', -1) < 0:
                print("No segment layer selected")
                return
            
            selected_seg_layer_nr = selected_layers['selected_segment_layer']

            if selected_seg_layer_nr < 0:
                print("No segment layer selected")
                return
            selected_em_layer_nr = selected_layers.get('selected_em_layer', -1)


        # Initialize Z scaling
        z_scale = 1
        z_min = rparam['zmin']
        z_max = rparam['zmax']

        if param['miplevel'] > 0:
            if extract_seg:
                mip_scale_matrix = self.vast.get_mipmap_scale_factors(selected_seg_layer_nr)
            else:
                mip_scale_matrix = self.vast.get_mipmap_scale_factors(selected_em_layer_nr)
            
            z_scale = mip_scale_matrix[param['miplevel']][2]
            
            if z_scale != 1:
                z_min = floor(z_min / z_scale)
                z_max = floor(z_max / z_scale)
        else:
            mip_scale_matrix = None

        # Apply mip level scaling
        
        x_min = rparam['xmin'] >> param['miplevel']
        x_max = (rparam['xmax'] >> param['miplevel']) - 1
        y_min = rparam['ymin'] >> param['miplevel']
        y_max = (rparam['ymax'] >> param['miplevel']) - 1

        # Get mip scale factors
        if mip_scale_matrix is not None:
            mip_scale_matrix = np.array(mip_scale_matrix)
        if param['miplevel'] > 0 and mip_scale_matrix is not None:
            mip_fact_x = mip_scale_matrix[param['miplevel']][0]
            mip_fact_y = mip_scale_matrix[param['miplevel']][1]
            mip_fact_z = mip_scale_matrix[param['miplevel']][2]
        else:
            mip_fact_x = 1
            mip_fact_y = 1
            mip_fact_z = 1
        print(f"DEBUG: Mip factors: X={mip_fact_x}, Y={mip_fact_y}, Z={mip_fact_z}")
        # Validate volume dimensions
        if ((x_min == x_max or y_min == y_max or z_min == z_max) and 
            param.get('closesurfaces', 0) == 0):
            print(
                'ERROR: The surface script needs a volume which is at least ',
                'two pixels wide in each direction. Please adjust "Render from area" ',
                'values, or enable "Close surface sides".',
            )
            print('Canceled.')
            return
        
        # Store processed parametersQ
        self.param = {
            **param,
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
            'mipfactz': mip_fact_z
        }

        # Update param to reference self.param so local variable has all keys
        param = self.param

        objects = None

        if extract_seg:
            self.data = data
            self.param['max_object_number'] = max_object_number
            self.param['mip_data_size'] = mip_data_size
            self.param['seg_layer_name'] = seg_layer_name

            # Create mapping from segment ID to name for export_meshes
            # data[:, 0] contains segment IDs, names array corresponds to rows
            self.seg_id_to_name = {}
            if len(names) > 0 and len(data) > 0:
                for i, row in enumerate(data):
                    seg_id = int(row[0])
                    if i < len(names):
                        self.seg_id_to_name[seg_id] = names[i]
                    else:
                        self.seg_id_to_name[seg_id] = f"Segment_{seg_id}"
        
        print(f"Initialization complete. Extract mode: {'segmentation' if extract_seg else 'screenshots'}")

        if extract_seg: # ===== COMPUTE FULL NAMES (including folder hierarchy) =====
            # Convert data to numpy array for easier indexing
            data = np.array(data)
            if param.get('includefoldernames', 0) == 1:
                logging.info("Computing full names with folder hierarchy...")
                full_names = names.copy()
                id_to_index = {int(data[i, 0]): i for i in range(len(data))}
                for i in range(len(data)):
                    j = i
                    # Traverse up the hierarchy (column 13 is parent ID, 0-indexed)
                    parent_id = int(data[j, 13])
                    while parent_id != 0 and parent_id != 4294967295:  # Check if parent is not 0
                        # Find the index of the parent in the data array
                        if parent_id in id_to_index:
                            j = id_to_index[parent_id]
                            full_names[i] = names[j] + '.' + full_names[i]
                            parent_id = int(data[j, 13])
                        else:
                            logging.warning(f"Parent ID {parent_id} not found for object at index {i}")
                            break
                names = full_names
                logging.debug(f"Sample full name: {names[0] if names else 'N/A'}")

            extract_which = param.get('extractwhich', 1)
            logging.info(f"Computing object list (extractwhich={extract_which})...")

            if extract_which == 1:
                # All segments individually, uncollapsed
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                self.vast.set_seg_translation([], [])
                logging.info(f"Mode 1: Exporting {len(objects)} individual segments")
            elif extract_which == 2:
                # All segments, collapsed as in VAST
                objects = np.unique(data[:, 17])  # Column 18 in MATLAB (0-indexed: 17)
                objects = np.column_stack([objects, data[objects.astype(int), 1]]).astype(np.uint32)
                self.vast.set_seg_translation(data[:, 0].astype(int).tolist(), data[:, 17].astype(int).tolist())
                logging.info(f"Mode 2: Exporting {len(objects)} collapsed segments")               
            elif extract_which == 3:
                # Selected segment and children, uncollapsed
                selected = np.where((data[:, 1].astype(np.uint32) & 65536) > 0)[0]
                if len(selected) == 0:
                    objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                    logging.warning("No segments selected, exporting all")
                else:
                    selected = np.concatenate([selected, self._get_child_tree_ids_seg(data, selected)])
                    objects = np.column_stack([selected, data[selected, 1]]).astype(np.uint32)
                    logging.info(f"Mode 3: Exporting {len(objects)} selected segments + children")
                self.vast.set_seg_translation(data[selected, 0].astype(int).tolist(), data[selected, 0].astype(int).tolist())         
            elif extract_which == 4:
                # Selected segment and children, collapsed as in VAST
                selected = np.where((data[:, 1].astype(np.uint32) & 65536) > 0)[0]
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
                self.vast.set_seg_translation(data[selected.astype(int), 0].astype(int).tolist(), data[selected.astype(int), 17].astype(int).tolist())           
            else:
                logging.error(f"Unknown extractwhich value: {extract_which}")
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
            
            self.objects = objects
            param['objects'] = objects
            self.param['objects'] = objects  # Update instance param with computed objects

        # ===== COMPUTE NUMBER OF BLOCKS/TILES IN VOLUME =====
        logging.info("Computing block divisions...")

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

        print(f"DEBUG: Region bounds: X={x_min}-{x_max}, Y={y_min}-{y_max}, Z={z_min}-{z_max}")
        print(f"DEBUG: Block sizes: {block_size_x}x{block_size_y}x{block_size_z}, overlap={overlap}")

        if slice_step == 1:
            slice_numbers = list(range(z_min, z_max + 1))
            while tile_z1 <= z_max:
                tile_z1 += block_size_z - overlap
                nr_z_tiles += 1
        else:
            # Use every nth slice
            slice_numbers = list(range(z_min, z_max + 1, param['slicestep']))
            nr_z_tiles = ceil(len(slice_numbers) / (block_size_z - overlap))

            block_slice_numbers = []
            j = 0
            for p in range(0, len(slice_numbers), block_size_z - overlap):
                pe = min(p + block_size_z, len(slice_numbers))
                block_slice_numbers.append(slice_numbers[p:pe])
                j += 1
            param['block_slice_numbers'] = block_slice_numbers
        
        # Store in params for later use
        param['nr_x_tiles'] = nr_x_tiles
        param['nr_y_tiles'] = nr_y_tiles
        param['nr_z_tiles'] = nr_z_tiles
        param['slice_numbers'] = slice_numbers

        total_blocks = nr_x_tiles * nr_y_tiles * nr_z_tiles
        print(f"Block division: {nr_x_tiles} x {nr_y_tiles} x {nr_z_tiles} = {total_blocks} blocks")
        logging.info(f"Total blocks to process: {total_blocks}")

        if total_blocks == 0:
            print("ERROR: No blocks to process! Check region bounds and block sizes.")
            return

        if param.get('usemipregionconstraint', 0) == 1:
            rparam = {k: param[k] for k in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] if k in param}
            logging.info("Computing MIP region constraint...")
            print("Loading constraint mask at lower resolution...")
            mip_region_mip = param.get('usemipregionconstraint', 0)

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
            
            if extract_seg:
                print(f"Loading segmentation at mip {mip_region_mip} for constraint...")
                logging.info(f"Loading segmentation constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
                slice_step = param.get('slicestep', 1)
            
                if slice_step == 1:
                    mc_seg_image, values, numbers, bboxes = \
                        self.vast.get_seg_image_rle_decoded_bboxes(
                            mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, False
                        )
                else:
                    # Load slice by slice for non-unit step
                    slices = list(range(c_z_min, c_z_max + 1, slice_step))
                    mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                    
                    for idx, slice_num in enumerate(slices):
                        mc_seg_slice, values, numbers, bboxes = \
                            self.vast.get_seg_image_rle_decoded_bboxes(
                                mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                                slice_num, slice_num, False
                            )
                        mc_seg_image[:, :, idx] = mc_seg_slice
            else:
                # Load screenshot data
                print(f"Loading screenshots at mip {mip_region_mip} for constraint...")
                logging.info(f"Loading screenshot constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
                
                slice_step = param.get('slicestep', 1)
                
                if slice_step == 1:
                    mc_seg_image = self.vast.get_screenshot_image(
                        mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, False
                    )
                    # Convert from (y, x, z, c) to (x, y, z) and make binary
                    if mc_seg_image is not None:
                        mc_seg_image = np.transpose(np.sum(mc_seg_image, axis=3) > 0, (1, 0, 2))
                    else:
                        mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, c_z_max - c_z_min + 1))
                else:
                    slices = list(range(c_z_min, c_z_max + 1, slice_step))
                    mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                    
                    for idx, slice_num in enumerate(slices):
                        mc_seg_slice = self.vast.get_screenshot_image(
                            mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                            slice_num, slice_num, False
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
                        c_max_x = min(mc_seg_image.shape[0] - 1, int(ceil((tile_x2 - x_min) / c_mip_fact_x))) if len(mc_seg_image.shape) > 0 else 0
                        c_min_y = max(0, int(floor((tile_y1 - y_min) / c_mip_fact_y)))
                        c_max_y = min(mc_seg_image.shape[1] - 1, int(ceil((tile_y2 - y_min) / c_mip_fact_y))) if len(mc_seg_image.shape) > 1 else 0
                        c_min_z = max(0, int(floor((tile_z1 - z_min) / c_mip_fact_z)))
                        c_max_z = min(mc_seg_image.shape[2] - 1, int(ceil((tile_z2 - z_min) / c_mip_fact_z))) if len(mc_seg_image.shape) > 2 else 0
                        
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

        logging.info("Initializing storage arrays...")

        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']

        if extract_seg:
            # Storage for segmentation objects
            max_obj_num = self.param['max_object_number']

            param['farray'] = {}
            param['varray'] = {}
            param['object_volume'] = np.zeros(len(param['objects']), dtype=np.int64)

            print(f"Initialized storage for {max_obj_num} potential objects across {nr_x_tiles}x{nr_y_tiles}x{nr_z_tiles} blocks")
            logging.info(f"Storage size: {len(param['objects'])} objects x {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")

        else:
            extract_which = param.get('extractwhich', 5)

            if extract_which == 10:
                # Up to 2^24 unique color objects
                # Use sparse matrix for indexing: fvindex[color, block_idx] = block_number
                total_colors = 256 * 256 * 256
                total_blocks = (nr_x_tiles + 1) * (nr_y_tiles + 1) * (nr_z_tiles + 1)

                # Sparse matrix would be huge, use dict instead
                param['fvindex'] = sparse.csr_matrix((total_colors, total_blocks))
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

        x_min, y_min, z_min = rparam['xmin'], rparam['ymin'], rparam['zmin']
        x_max, y_max, z_max = rparam['xmax'], rparam['ymax'], rparam['zmax']
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

        # Ensure data is numpy array for processing
        if extract_seg and isinstance(data, list):
            data = np.array(data)
        
        # Ensure mip_scale_matrix is numpy array if present
        if mip_scale_matrix is not None and isinstance(mip_scale_matrix, list):
            mip_scale_matrix = np.array(mip_scale_matrix)
        
        block_nr = 0  # For mode 10 color tracking
        total_blocks = nr_x_tiles * nr_y_tiles * nr_z_tiles
        processed_blocks = 0
        skipped_blocks = 0

        print(f"DEBUG: Starting block processing loop. Tiles: {nr_x_tiles}x{nr_y_tiles}x{nr_z_tiles}")
        print(f"DEBUG: Total objects to extract: {len(param['objects'])}")

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
                        max_object_number = self.param.get('max_object_number', 0)
                        
                        if slice_step == 1:
                            seg_image, values, numbers, bboxes = self.vast.get_seg_image_rle_decoded_bboxes(
                                    param['miplevel'],
                                    tile_x1 - edx[0, 0], tile_x2 + edx[0, 1],
                                    tile_y1 - edx[1, 0], tile_y2 + edx[1, 1],
                                    tile_z1 - edx[2, 0], tile_z2 + edx[2, 1],
                                    False
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
                                s_seg_image, s_values, s_numbers, s_bboxes = self.vast.get_seg_image_rle_decoded_bboxes(
                                        param['miplevel'],
                                        tile_x1 - edx[0, 0], tile_x2 + edx[0, 1],
                                        tile_y1 - edx[1, 0], tile_y2 + edx[1, 1],
                                        slice_num, slice_num, False
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
                            if seg_image is not None and seg_image.ndim >= 3:
                                sh = seg_image.shape  # type: ignore
                                x_end = int(sh[0] - edx[0, 1]) if edx[0, 1] > 0 else int(sh[0])  # type: ignore
                                y_end = int(sh[1] - edx[1, 1]) if edx[1, 1] > 0 else int(sh[1])  # type: ignore
                                z_end = int(sh[2] - edx[2, 1]) if edx[2, 1] > 0 else int(sh[2])  # type: ignore
                                seg_image = seg_image[edx[0, 0]:x_end, edx[1, 0]:y_end, edx[2, 0]:z_end]
                            
                            # Adjust bboxes
                            if bboxes is not None and len(bboxes) > 0:
                                if edx[0, 0] > 0:
                                    bb = bboxes[:, [0, 3]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [0, 3]] = bb
                                if edx[0, 1] > 0:
                                    bb = bboxes[:, [0, 3]]
                                    sh0 = int(seg_image.shape[0]) if seg_image is not None else 0  # type: ignore
                                    bb[bb > sh0] = sh0
                                    bboxes[:, [0, 3]] = bb
                                
                                if edx[1, 0] > 0:
                                    bb = bboxes[:, [1, 4]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [1, 4]] = bb
                                if edx[1, 1] > 0:
                                    bb = bboxes[:, [1, 4]]
                                    sh1 = int(seg_image.shape[1]) if seg_image is not None else 0  # type: ignore
                                    bb[bb > sh1] = sh1
                                    bboxes[:, [1, 4]] = bb
                                
                                if edx[2, 0] > 0:
                                    bb = bboxes[:, [2, 5]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [2, 5]] = bb
                                if edx[2, 1] > 0:
                                    bb = bboxes[:, [2, 5]]
                                    sh2 = int(seg_image.shape[2]) if seg_image is not None else 0  # type: ignore
                                    bb[bb > sh2] = sh2
                                    bboxes[:, [2, 5]] = bb
                        
                        # Process segmentation data
                        logging.debug(f"Processing {len(values)} objects in block ({tx},{ty},{tz})...")
                        
                        # Remove zero values from consideration
                        if values is not None:
                            # Ensure values is an array
                            if not isinstance(values, np.ndarray):
                                if isinstance(values, (int, float)):
                                    values = np.array([values])
                                else:
                                    values = np.array(values)
                            if len(values) > 0:
                                # Ensure numbers and bboxes are also arrays
                                if not isinstance(numbers, np.ndarray):
                                    numbers = np.array(numbers) if isinstance(numbers, list) else np.array([numbers])
                                if not isinstance(bboxes, np.ndarray):
                                    bboxes = np.array(bboxes) if isinstance(bboxes, list) else np.array([bboxes])
                                    if bboxes.ndim == 1:
                                        bboxes = bboxes.reshape(1, -1)
                                
                                mask = values != 0
                                values = values[mask]
                                numbers = numbers[mask]
                                bboxes = bboxes[mask]
                            else:
                                values = None
                        
                        if values is None or len(values) == 0:
                            logging.debug(f"No objects in block ({tx},{ty},{tz})")
                            tile_x1 += block_size_x - overlap
                            continue
                        
                        # Adjust for surface closing (add boundary slices)
                        x_vofs = y_vofs = z_vofs = 0
                        tt_xs, tt_ys, tt_zs = tile_xs, tile_ys, tile_zs
                        
                        if param.get('closesurfaces', 0) == 1 and seg_image is not None:
                            sh = seg_image.shape  # type: ignore
                            if tx == 0:
                                seg_image = np.concatenate([np.zeros((1, int(sh[1]), int(sh[2])), dtype=seg_image.dtype),  # type: ignore
                                                        seg_image], axis=0)
                                bboxes[:, 0] += 1
                                bboxes[:, 3] += 1
                                x_vofs -= 1
                                tt_xs += 1
                            if ty == 0:
                                sh = seg_image.shape  # type: ignore
                                seg_image = np.concatenate([np.zeros((int(sh[0]), 1, int(sh[2])), dtype=seg_image.dtype),  # type: ignore
                                                        seg_image], axis=1)
                                bboxes[:, 1] += 1
                                bboxes[:, 4] += 1
                                y_vofs -= 1
                                tt_ys += 1
                            if tz == 0:
                                sh = seg_image.shape  # type: ignore
                                seg_image = np.concatenate([np.zeros((int(sh[0]), int(sh[1]), 1), dtype=seg_image.dtype),  # type: ignore
                                                        seg_image], axis=2)
                                bboxes[:, 2] += 1
                                bboxes[:, 5] += 1
                                z_vofs -= 1
                                tt_zs += 1
                            if tx == param['nr_x_tiles'] - 1:
                                sh = seg_image.shape  # type: ignore
                                seg_image = np.concatenate([seg_image,
                                                        np.zeros((1, int(sh[1]), int(sh[2])), dtype=seg_image.dtype)], axis=0)  # type: ignore
                                tt_xs += 1
                            if ty == param['nr_y_tiles'] - 1:
                                sh = seg_image.shape  # type: ignore
                                seg_image = np.concatenate([seg_image,
                                                        np.zeros((int(sh[0]), 1, int(sh[2])), dtype=seg_image.dtype)], axis=1)  # type: ignore
                                tt_ys += 1
                            if tz == param['nr_z_tiles'] - 1:
                                sh = seg_image.shape  # type: ignore
                                seg_image = np.concatenate([seg_image,
                                                        np.zeros((int(sh[0]), int(sh[1]), 1), dtype=seg_image.dtype)], axis=2)  # type: ignore
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
                            # seg_image is (X, Y, Z) ordered
                            # bbx is [xmin, ymin, zmin, xmax, ymax, zmax] (1-indexed)
                            bbx_int = bbx.astype(int) - 1  # Convert to 0-indexed
                            subseg = seg_image[bbx_int[0]:bbx_int[3]+1, 
                                               bbx_int[1]:bbx_int[4]+1, 
                                               bbx_int[2]:bbx_int[5]+1]
                            subseg = (subseg == seg).astype(float)

                            # Skip if no valid surface can be extracted
                            if subseg.min() == subseg.max() or np.any(np.array(subseg.shape) < 2):
                                continue

                            # Extract isosurface using marching cubes
                            # NOTE: Do NOT transpose - seg_image is already (X, Y, Z)
                            try:
                                verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)

                                if len(verts) > 0:
                                    # ================================================
                                    # CRITICAL: Match MATLAB's coordinate output order
                                    # ================================================
                                    # MATLAB isosurface with input V(x,y,z) returns vertices as (Y, X, Z)
                                    # Python marching_cubes with input V[x,y,z] returns vertices as (X, Y, Z)
                                    # Swap columns 0 and 1 to convert (X,Y,Z) -> (Y,X,Z) like MATLAB
                                    verts[:, [0, 1]] = verts[:, [1, 0]]
                                    
                                    # Reverse face winding to maintain correct normals after the swap
                                    faces = faces[:, [0, 2, 1]]
                                    
                                    # ================================================
                                    # Apply offsets - matching MATLAB exactly
                                    # ================================================
                                    # After swap: verts[:,0]=Y, verts[:,1]=X, verts[:,2]=Z
                                    
                                    # MATLAB code:
                                    #   v(:,1)=v(:,1)+bbx(2)-1+yvofs;  % v(:,1) is Y, bbx(2) is ymin
                                    #   v(:,2)=v(:,2)+bbx(1)-1+xvofs;  % v(:,2) is X, bbx(1) is xmin
                                    #   v(:,3)=v(:,3)+bbx(3)-1+zvofs;  % v(:,3) is Z
                                    # bbx_int is 0-indexed, so bbx(2)-1 in MATLAB = bbx_int[1] in Python
                                    verts[:, 0] += bbx_int[1] + y_vofs  # Y offset (column 0 after swap)
                                    verts[:, 1] += bbx_int[0] + x_vofs  # X offset (column 1 after swap)
                                    verts[:, 2] += bbx_int[2] + z_vofs  # Z offset

                                    # Add tile position
                                    # MATLAB: v(:,1)=v(:,1)+tiley1-1;  v(:,2)=v(:,2)+tilex1-1;
                                    # tile coords in Python are 0-indexed already
                                    verts[:, 0] += tile_y1  # Y (column 0)
                                    verts[:, 1] += tile_x1  # X (column 1)

                                    if slice_step == 1:
                                        verts[:, 2] += tile_z1
                                    else:
                                        verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice

                                    # Scale to physical units
                                    # MATLAB: v(:,1)*yscale*yunit*mipfacty (Y)
                                    #         v(:,2)*xscale*xunit*mipfactx (X)
                                    verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']  # Y
                                    verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']  # X
                                    verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']  # Z

                                    # Track volume for this segment
                                    voxel_count = int(np.sum(subseg > 0))
                                    obj_idx = np.where(param['objects'][:, 0] == int(seg))[0]
                                    if len(obj_idx) > 0:
                                        param['object_volume'][obj_idx[0]] += voxel_count

                                    # Store faces and vertices
                                    param['farray'][(int(seg), tx, ty, tz)] = faces
                                    param['varray'][(int(seg), tx, ty, tz)] = verts
                                    
                            except Exception as e:
                                logging.warning(f"Failed to extract surface for segment {seg} in block ({tx},{ty},{tz}): {e}")
                    else:
                        logging.debug(f"Loading screenshot block ({tx},{ty},{tz})...")
        
                        # Load screenshot data
                        slice_step = param.get('slicestep', 1)
                        
                        if slice_step == 1:
                            scs_image = self.vast.get_screenshot_image(
                                param['miplevel'],
                                tile_x1, tile_x2, tile_y1, tile_y2, tile_z1, tile_z2, True, True
                            )
                            
                            # Handle single slice case (returns 3D instead of 4D)
                            if tile_z1 == tile_z2 and scs_image is not None and scs_image.ndim == 3:
                                scs_image = scs_image.reshape(scs_image.shape[0], scs_image.shape[1], 1, scs_image.shape[2])
                        else:
                            # Load slice-by-slice
                            block_slice_numbers = param.get('block_slice_numbers', [[]])[tz]
                            bs = block_slice_numbers
                            
                            scs_image = np.zeros((tile_y2 - tile_y1 + 1, tile_x2 - tile_x1 + 1, len(bs), 3), dtype=np.uint8)
                            first_block_slice = bs[0]
                            
                            for i, slice_num in enumerate(bs):
                                scs_slice = self.vast.get_screenshot_image(param['miplevel'], tile_x1, tile_x2, tile_y1, tile_y2, slice_num, slice_num, True, True) 
                                if scs_slice is not None:
                                    scs_image[:, :, i, :] = scs_slice
                        
                        logging.debug(f"Processing screenshot block ({tx},{ty},{tz})...")
                        
                        # Convert from (y, x, z, c) to (x, y, z, c) and extract channels
                        if scs_image is not None:
                            scs_image = np.transpose(scs_image, (1, 0, 2, 3))
                        else:
                            return
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
                           # Combine RGB into single color value (R << 16 | G << 8 | B)
                            col_cube = (r_cube.astype(np.int32) << 16) + (g_cube.astype(np.int32) << 8) + b_cube.astype(np.int32)
                            
                            # Count unique colors (histogram)
                            unique_colors, counts = np.unique(col_cube[col_cube != 0], return_counts=True)
                            
                            if len(unique_colors) == 0:
                                tile_x1 += block_size_x - overlap
                                continue
                            
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
                                if np.sum(subseg) == 0 or np.sum(subseg) == subseg.size:
                                    # All zeros or all ones - skip
                                    continue

                                # NOTE: Do NOT transpose - col_cube is already (X, Y, Z)

                                # Extract isosurface
                                try:
                                    verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)

                                    if len(verts) > 0:
                                        # ================================================
                                        # CRITICAL: Match MATLAB's coordinate output order
                                        # ================================================
                                        # Swap columns 0 and 1 to convert (X,Y,Z) -> (Y,X,Z) like MATLAB
                                        verts[:, [0, 1]] = verts[:, [1, 0]]
                                        
                                        # Reverse face winding to maintain correct normals
                                        faces = faces[:, [0, 2, 1]]

                                        # After swap: verts[:,0]=Y, verts[:,1]=X, verts[:,2]=Z
                                        verts[:, 0] += y_vofs  # Y offset
                                        verts[:, 1] += x_vofs  # X offset
                                        verts[:, 2] += z_vofs  # Z offset

                                        verts[:, 0] += tile_y1  # Y
                                        verts[:, 1] += tile_x1  # X

                                        if slice_step == 1:
                                            verts[:, 2] += tile_z1
                                        else:
                                            verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice

                                        # Scale to physical units
                                        # verts[:,0] is Y, verts[:,1] is X
                                        verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']  # Y
                                        verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']  # X
                                        verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']  # Z

                                        # Store with block indexing
                                        idx = tz * param['nr_y_tiles'] * param['nr_x_tiles'] + ty * param['nr_x_tiles'] + tx
                                        param['fvindex'][(int(color), idx)] = block_nr
                                        param['farray'][block_nr] = faces
                                        param['varray'][block_nr] = verts
                                        
                                except Exception as e:
                                    logging.warning(f"Failed to extract color {color:06X} in block ({tx},{ty},{tz}): {e}")
                        else:
                            # Extract RGB layers or brightness levels
                            # Compute brightness cube if needed
                            cube = None
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

                                # NOTE: Do NOT transpose - cube data is already (X, Y, Z)

                                # Extract isosurface
                                try:
                                    verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)

                                    if len(verts) > 0:
                                        # ================================================
                                        # CRITICAL: Match MATLAB's coordinate output order
                                        # ================================================
                                        # Swap columns 0 and 1 to convert (X,Y,Z) -> (Y,X,Z) like MATLAB
                                        verts[:, [0, 1]] = verts[:, [1, 0]]
                                        
                                        # Reverse face winding to maintain correct normals
                                        faces = faces[:, [0, 2, 1]]

                                        # After swap: verts[:,0]=Y, verts[:,1]=X, verts[:,2]=Z
                                        verts[:, 0] += y_vofs  # Y offset
                                        verts[:, 1] += x_vofs  # X offset
                                        verts[:, 2] += z_vofs  # Z offset

                                        verts[:, 0] += tile_y1  # Y
                                        verts[:, 1] += tile_x1  # X

                                        if slice_step == 1:
                                            verts[:, 2] += tile_z1
                                        else:
                                            verts[:, 2] = ((verts[:, 2] - 0.5) * slice_step) + 0.5 + first_block_slice

                                        # Scale to physical units
                                        # verts[:,0] is Y, verts[:,1] is X
                                        verts[:, 0] *= param.get('yscale', 1.0) * param.get('yunit', 1.0) * param['mipfacty']  # Y
                                        verts[:, 1] *= param.get('xscale', 1.0) * param.get('xunit', 1.0) * param['mipfactx']  # X
                                        verts[:, 2] *= param.get('zscale', 1.0) * param.get('zunit', 1.0) * param['mipfactz']  # Z

                                        # Store faces and vertices
                                        param['farray'][(obj_idx, tx, ty, tz)] = faces
                                        param['varray'][(obj_idx, tx, ty, tz)] = verts
                                        
                                except Exception as e:
                                    logging.warning(f"Failed to extract object {obj_idx} in block ({tx},{ty},{tz}): {e}")
                        block_nr += 1
                    
                    tile_x1 += block_size_x - overlap
                tile_y1 += block_size_y - overlap
            tile_z1 += block_size_z - overlap
        
        if extract_seg:
            self.vast.set_seg_translation([],[])
        
        print("="*60)
        if self.canceled:
            print("Extraction CANCELED")
        else:
            print(f"Extraction complete! Processed {processed_blocks} blocks, skipped {skipped_blocks}")
        print("=" * 60)

        logging.info(f"Block processing complete: {processed_blocks} processed, {skipped_blocks} skipped")

        # Update self.param with all changes made during extraction
        self.param = param

    def _get_block_mesh(self, seg_id: int, tx: int, ty: int, tz: int, param: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve mesh for specific block from storage."""
        key = (seg_id, tx, ty, tz)
        f = param['farray'].get(key, np.array([]))
        v = param['varray'].get(key, np.array([]))

        # Return copies to avoid modifying originals during merge
        if f is not None and len(f) > 0:
            return f.copy(), v.copy()
        return np.array([]), np.array([])

    def _merge_object_meshes(self, seg_id: int, param: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge all block meshes for a single segment using X→Y→Z consolidation.
        Based on MATLAB extractsurfaces.m lines 827-891.

        Returns:
            (faces, vertices) - Final consolidated mesh
        """
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']

        # Storage for incremental Z-merge optimization
        completed_faces = []
        completed_verts = []
        vert_offset = 0

        # Initialize final mesh variables
        final_f, final_v = None, None

        # Z-axis loop (iterate through planes)
        for tz in range(nr_z_tiles):
            plane_f, plane_v = None, None

            # Y-axis loop (iterate through rows in plane)
            for ty in range(nr_y_tiles):
                row_f, row_v = None, None

                # X-axis loop (merge blocks in row)
                for tx in range(nr_x_tiles):
                    block_f, block_v = self._get_block_mesh(seg_id, tx, ty, tz, param)

                    if len(block_v) == 0:
                        continue

                    if row_v is None:
                        row_f, row_v = block_f, block_v
                    else:
                        row_f, row_v = self._merge_meshes(row_f, row_v, block_f, block_v)

                # Merge row into plane
                if row_v is not None and len(row_v) > 0:
                    if plane_v is None:
                        plane_f, plane_v = row_f, row_v
                    else:
                        plane_f, plane_v = self._merge_meshes(plane_f, plane_v, row_f, row_v)

            # Merge plane into final mesh
            if plane_v is not None and len(plane_v) > 0:
                if final_v is None:
                    # First plane with data
                    final_f, final_v = plane_f, plane_v
                else:
                    # Merge with existing data
                    final_f, final_v = self._merge_meshes(final_f, final_v, plane_f, plane_v)

                    # OPTIMIZATION: Extract completed portion (MATLAB lines 879-888)
                    # Vertices at max Z won't be merged with future planes
                    if final_v is not None and len(final_v) > 0 and tz < nr_z_tiles - 1:
                        max_z = np.max(final_v[:, 2])
                        z_threshold = max_z - 0.01  # Small epsilon for floating point

                        # Find first vertex below threshold
                        vcut_candidates = np.where(final_v[:, 2] < z_threshold)[0]
                        if len(vcut_candidates) > 0:
                            vcut = vcut_candidates[0]

                            # Find first face using vertex >= vcut
                            face_uses_later = np.any(final_f >= vcut, axis=1)
                            fcut_candidates = np.where(face_uses_later)[0]

                            if len(fcut_candidates) > 0 and vcut > 0:
                                fcut = fcut_candidates[0]

                                # Extract completed portion
                                completed_verts.append(final_v[:vcut, :])
                                completed_faces.append(final_f[:fcut, :] + vert_offset)

                                # Keep only incomplete portion
                                vert_offset += vcut
                                final_v = final_v[vcut:, :]
                                final_f = final_f[fcut:, :] - vcut

        # Combine completed + final portions
        if final_v is None or len(final_v) == 0:
            return np.array([]), np.array([])

        if len(completed_verts) > 0:
            all_verts = np.vstack(completed_verts + [final_v])
            all_faces = np.vstack(completed_faces + [final_f + vert_offset])
        else:
            all_verts = final_v
            all_faces = final_f

        return all_faces, all_verts

    def _get_segment_colors(self, param: dict, data: np.ndarray) -> np.ndarray:
        """Extract RGB colors for segments from VAST data matrix."""
        colors = np.zeros((len(param['objects']), 3), dtype=np.uint8)

        if param.get('objectcolors', 1) == 1:
            # Use VAST segment colors
            for i, seg_id in enumerate(param['objects'][:, 0]):
                seg_idx = np.where(data[:, 0] == seg_id)[0]
                if len(seg_idx) > 0:
                    # Get collapse target segment to inherit color
                    inherit_idx = int(data[seg_idx[0], 17])  # Column 18 (0-indexed)
                    if inherit_idx < len(data):
                        # RGB in columns 3-5 (0-indexed: 2-4)
                        colors[i, :] = data[inherit_idx, 2:5].astype(np.uint8)
        else:
            # Fallback: random colors
            colors = np.random.randint(0, 255, (len(param['objects']), 3), dtype=np.uint8)

        return colors

    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid filesystem characters."""
        invalid_chars = ' ?*\\/|:"<>'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name

    def _apply_output_transforms(self, vertices: np.ndarray, param: dict) -> np.ndarray:
        """Apply Z-inversion and output offset."""
        v = vertices.copy()

        if param.get('invertz', 0) == 1:
            v[:, 2] = -v[:, 2]

        v[:, 0] += param.get('outputoffsetx', 0)
        v[:, 1] += param.get('outputoffsety', 0)
        v[:, 2] += param.get('outputoffsetz', 0)

        return v

    def _write_obj_with_mtl(self, vertices, faces, obj_path, mtl_filename,
                            material_name, object_name, invert_normals=False):
        """Write OBJ file with MTL link."""
        with open(obj_path, 'w') as f:
            f.write(f"mtllib {mtl_filename}\n")
            f.write(f"usemtl {material_name}\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write(f"g {object_name}\n")

            # Write faces (convert to 1-based indexing)
            for face in faces:
                if invert_normals:
                    f.write(f"f {int(face[1])+1} {int(face[0])+1} {int(face[2])+1}\n")
                else:
                    f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")

            f.write("g\n")

    def _write_mtl(self, mtl_path, material_name, color_rgb):
        """Write MTL material file."""
        color = color_rgb / 255.0  # Normalize to [0,1]

        with open(mtl_path, 'w') as f:
            f.write(f"newmtl {material_name}\n")
            f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            f.write(f"Ks 0.5 0.5 0.5\n")
            f.write(f"d 1.000\n")
            f.write(f"Ns 32\n")

    def export_meshes(self):
        """
        Phase 5-6: Merge block meshes and write OBJ files.
        Call this after extract_surfaces() completes.
        """
        import os

        param = self.param

        # Verify extract_surfaces was called and completed
        if 'objects' not in param:
            print("ERROR: extract_surfaces() must be called before export_meshes()")
            print("No objects found to export.")
            return

        # Create output directory
        os.makedirs(param['targetfolder'], exist_ok=True)

        # Get segment colors
        print("Extracting segment colors...")
        colors = self._get_segment_colors(param, self.data)

        # Process each object
        total_objects = len(param['objects'])

        for seg_nr, seg_id in enumerate(param['objects'][:, 0]):
            if self.canceled:
                break

            seg_id = int(seg_id)
            obj_name = self.seg_id_to_name.get(seg_id, f"Segment_{seg_id}")
            print(f"Processing object {seg_nr + 1}/{total_objects}: {obj_name}")

            # Merge all blocks for this segment
            print("  Merging blocks...")
            merged_faces, merged_verts = self._merge_object_meshes(seg_id, param)

            if len(merged_verts) == 0:
                print("  (empty - skipping)")
                continue

            # Apply output transformations
            merged_verts = self._apply_output_transforms(merged_verts, param)

            # Generate filenames
            obj_name_sanitized = self._sanitize_filename(obj_name)
            prefix = param.get('targetfileprefix', 'Segment_')
            obj_filename = f"{prefix}{seg_id:04d}_{obj_name}.obj"
            mtl_filename = f"{prefix}{seg_id:04d}_{obj_name}.mtl"
            material_name = f"mat_{seg_id:04d}"

            obj_path = os.path.join(param['targetfolder'], obj_filename)
            mtl_path = os.path.join(param['targetfolder'], mtl_filename)

            # Write OBJ file
            print(f"  Writing {obj_filename}")
            invert = (param.get('invertz', 0) == 0)  # Invert normals if NOT inverting Z
            self._write_obj_with_mtl(merged_verts, merged_faces, obj_path, mtl_filename,
                                     material_name, obj_name, invert_normals=invert)

            # Write MTL file
            self._write_mtl(mtl_path, material_name, colors[seg_nr])

            print(f"  ✓ {len(merged_verts)} vertices, {len(merged_faces)} faces")

        print(f"\n{'='*60}")
        print(f"Export complete! {seg_nr + 1} objects written to:")
        print(f"  {param['targetfolder']}")
        print(f"{'='*60}")

    def _merge_meshes(self, f1, v1, f2, v2):
        """
        Merge two meshes defined by (f1, v1) and (f2, v2).
        
        Merges meshes by identifying and merging duplicate vertices in the overlapping region,
        then re-indexing face indices appropriately.
        
        Args:
            f1: Face indices array for mesh 1 (Nx3)
            v1: Vertex coordinates array for mesh 1 (Mx3)
            f2: Face indices array for mesh 2 (Nx3)
            v2: Vertex coordinates array for mesh 2 (Mx3)
        
        Returns:
            tuple: (merged_faces, merged_vertices) where faces are re-indexed
        """
        # Handle None inputs
        if v1 is None or f1 is None:
            if v2 is None or f2 is None:
                return np.array([]), np.array([])
            return f2.copy() if hasattr(f2, 'copy') else f2, v2.copy() if hasattr(v2, 'copy') else v2

        if v2 is None or f2 is None:
            return f1.copy() if hasattr(f1, 'copy') else f1, v1.copy() if hasattr(v1, 'copy') else v1

        # Handle empty meshes
        if v1.size == 0:
            return f2.copy(), v2.copy()

        if v2.size == 0:
            return f1.copy(), v1.copy()
        
        # Convert to numpy arrays if needed
        f1 = np.asarray(f1, dtype=np.int32)
        v1 = np.asarray(v1, dtype=np.float32)
        f2 = np.asarray(f2, dtype=np.int32)
        v2 = np.asarray(v2, dtype=np.float32)
        
        nrofvertices1 = v1.shape[0]
        nrofvertices2 = v2.shape[0]
        
        # Adjust f2 indices by number of vertices in v1
        f2 = f2 + nrofvertices1
        
        # Find overlapping vertex region
        minv1 = np.min(v1, axis=0)
        maxv1 = np.max(v1, axis=0)
        minv2 = np.min(v2, axis=0)
        maxv2 = np.max(v2, axis=0)
        
        ovmin = np.maximum(minv1, minv2)
        ovmax = np.minimum(maxv1, maxv2)
        
        # Find vertices in overlap zone for v1
        mask1 = ((v1[:, 0] >= ovmin[0]) & (v1[:, 0] <= ovmax[0]) &
                (v1[:, 1] >= ovmin[1]) & (v1[:, 1] <= ovmax[1]) &
                (v1[:, 2] >= ovmin[2]) & (v1[:, 2] <= ovmax[2]))
        ov1_indices = np.where(mask1)[0]
        
        # Find vertices in overlap zone for v2
        mask2 = ((v2[:, 0] >= ovmin[0]) & (v2[:, 0] <= ovmax[0]) &
                (v2[:, 1] >= ovmin[1]) & (v2[:, 1] <= ovmax[1]) &
                (v2[:, 2] >= ovmin[2]) & (v2[:, 2] <= ovmax[2]))
        ov2_indices = np.where(mask2)[0]
        
        # If no overlap, concatenate meshes
        if ov2_indices.size == 0:
            print(f"    DEBUG: No overlap found. v1: {v1.shape[0]} verts, v2: {v2.shape[0]} verts")
            f = np.vstack([f1, f2])
            v = np.vstack([v1, v2])
            return f, v

        print(f"    DEBUG: Overlap region - v1: {ov1_indices.size} verts, v2: {ov2_indices.size} verts")
        
        # Find matching vertices in overlapping regions
        deletevertex = np.zeros(nrofvertices2, dtype=bool)
        
        # Use faster loopless version with intersect equivalent
        # Find common vertices in overlap zones
        ov1_verts = v1[ov1_indices]
        ov2_verts = v2[ov2_indices]
        
        # Find intersecting vertices (common coordinates)
        common_rows = []
        i1a_list = []
        i2a_list = []
        
        for i, v_ov1_idx in enumerate(ov1_indices):
            v_ov1 = v1[v_ov1_idx]
            # Check if this vertex exists in ov2
            for j, v_ov2_idx in enumerate(ov2_indices):
                v_ov2 = v2[v_ov2_idx]
                # Use larger tolerance to account for floating-point precision after coordinate
                # transformations (mipmap scaling, physical units, etc.)
                # With mipfact=4, overlap=2, scaled coords can differ by several units
                if np.allclose(v_ov1, v_ov2, atol=5.0):
                    i1a_list.append(v_ov1_idx)
                    i2a_list.append(v_ov2_idx)
                    common_rows.append(v_ov1)
                    break
        
        # Link duplicate vertices
        if len(i2a_list) > 0:
            print(f"    DEBUG: Merged {len(i2a_list)} duplicate vertices")
            i1a = np.array(i1a_list, dtype=np.int32)
            i2a = np.array(i2a_list, dtype=np.int32)
            
            # Remap f2 to use v1 vertices for common vertices
            aov2 = i2a
            kov2 = aov2 + nrofvertices1
            
            # For each face in f2, replace vertices that match
            for idx in range(len(aov2)):
                v1_idx = i1a[idx]
                v2_idx = kov2[idx]
                # Replace all occurrences of v2_idx in f2 with v1_idx
                f2[f2 == v2_idx] = v1_idx
                deletevertex[aov2[idx]] = True
        else:
            print(f"    WARNING: Overlap region found but NO vertices matched! This will cause gaps.")
            print(f"    Overlap region: min={ovmin}, max={ovmax}")
            if ov1_indices.size > 0 and ov2_indices.size > 0:
                # Show a sample of distances
                sample_dist = np.linalg.norm(v1[ov1_indices[0]] - v2[ov2_indices[0]])
                print(f"    Sample distance between overlap verts: {sample_dist:.6f}")

        # Re-index faces in f2 to account for deleted vertices
        z = np.arange(nrofvertices1, dtype=np.int32)
        z = np.append(z, np.zeros(nrofvertices2, dtype=np.int32))
        
        zp = nrofvertices1
        for sp in range(nrofvertices2):
            if not deletevertex[sp]:
                z[nrofvertices1 + sp] = zp
                zp += 1
        
        f2d = z[f2]
        
        # Ensure f2d is 2D array
        if f2d.ndim == 1:
            f2d = f2d.reshape(1, -1)
        
        # Delete unused vertices from v2 and concatenate
        v2d = v2[~deletevertex]
        
        f = np.vstack([f1, f2d])
        v = np.vstack([v1, v2d])
        
        return f, v

    def vertface2obj_mtllink_invnormal(v, f, filename, objectname, mtlfilename, materialname):
        """
        Saves mesh as OBJ with inverted normals (flipped winding order).
        """
        with open(filename, 'w') as fid:
            fid.write(f"mtllib {mtlfilename}\n")
            fid.write(f"usemtl {materialname}\n")
            
            # Write vertices
            for row in v:
                fid.write(f"v {row[0]:f} {row[1]:f} {row[2]:f}\n")
                
            fid.write(f"g {objectname}\n")
            
            # Write faces with flipped order: f(i,2), f(i,1), f(i,3)
            # Note: assuming f contains 1-based indices
            for row in f:
                fid.write(f"f {int(row[1])} {int(row[0])} {int(row[2])}\n")
                
            fid.write("g\n")

    def vertface2obj_mtllink(v, f, filename, objectname, mtlfilename, materialname):
        """
        Saves mesh as OBJ with standard winding order.
        """
        with open(filename, 'w') as fid:
            # Header
            fid.write(f"mtllib {mtlfilename}\n")
            fid.write(f"usemtl {materialname}\n")
            
            # Write all vertices efficiently
            # Creates a single large string block for all vertices
            v_lines = "".join(f"v {row[0]:f} {row[1]:f} {row[2]:f}\n" for row in v)
            fid.write(v_lines)
            
            fid.write(f"g {objectname}\n")
            
            # Write all faces efficiently
            # Note: assuming f contains 1-based indices
            f_lines = "".join(f"f {int(row[0])} {int(row[1])} {int(row[2])}\n" for row in f)
            fid.write(f_lines)
            
            fid.write("g\n")

    def export_skeleton(anno_object_id, output_file=None):
        """
        Export skeleton as SWC or OBJ line mesh.
        
        Returns: (nodes, edges) where nodes has columns [id, x, y, z, radius, parent_id]
        """
        
        info = self.vast.get_info()
        if not info:
            return None
        
        # Select the annotation object
        if not self.vast.set_selected_anno_object_nr(anno_object_id):
            print(f"Failed to select annotation object {anno_object_id}")
            return None
        
        # Get node data
        node_data = self.vast.get_ao_node_data()
        if node_data is None:
            print("Failed to get node data")
            return None
        
        # Extract and scale coordinates
        scale = np.array([info['voxelsizex'], info['voxelsizey'], info['voxelsizez']])
        
        # Build node list: [id, x, y, z, radius, parent_id]
        nodes = []
        for i in range(len(node_data)):
            node_id = int(node_data[i, 0])
            x = node_data[i, 12] * scale[0]
            y = node_data[i, 13] * scale[1]
            # Z coordinate needs to be extracted separately if stored
            z = 0  # You may need to get Z from anchorpoint or object data
            radius = node_data[i, 11]
            parent_id = int(node_data[i, 5])  # parent index
            
            nodes.append([node_id, x, y, z, radius, parent_id])
        
        nodes = np.array(nodes)
        
        # Build edge list
        edges = []
        for i in range(len(nodes)):
            parent_idx = int(nodes[i, 5])
            if parent_idx >= 0 and parent_idx < len(nodes):
                edges.append([i, parent_idx])
        
        if output_file:
            if output_file.endswith('.swc'):
                save_swc(output_file, nodes)
            elif output_file.endswith('.obj'):
                save_skeleton_obj(output_file, nodes, edges)
            print(f"Saved {output_file}")
        
        return nodes, edges

    def save_obj(filename, vertices, faces):
        """Save mesh as Wavefront OBJ."""
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def save_material_file(filename, material_name, color, alpha, roughness=0.5):
        """Save MTL (material) file for OBJ."""
        with open(filename, 'w') as f:
            f.write(f"newmtl {material_name}\n")
            f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # Ambient
            f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # Diffuse
            f.write(f"Ks 0.5 0.5 0.5\n")  # Specular
            f.write(f"d {alpha:.3f}\n")  # Transparency
            f.write(f"Ns 32\n")  # Shininess

    def save_skeleton_obj(filename, nodes, edges):
        """Save skeleton as OBJ line mesh."""
        with open(filename, 'w') as f:
            # Write vertices
            for node in nodes:
                f.write(f"v {node[1]} {node[2]} {node[3]}\n")
            
            # Write edges as lines
            for edge in edges:
                f.write(f"l {edge[0]+1} {edge[1]+1}\n")

    def save_swc(filename, nodes):
        """Save skeleton in SWC format (standard neuron format)."""
        with open(filename, 'w') as f:
            f.write("# SWC format skeleton\n")
            f.write("# id type x y z radius parent\n")
            for node in nodes:
                node_id = int(node[0]) + 1  # SWC uses 1-based
                parent_id = int(node[5]) + 1 if node[5] >= 0 else -1
                f.write(f"{node_id} 0 {node[1]:.3f} {node[2]:.3f} {node[3]:.3f} {node[4]:.3f} {parent_id}\n")

    def save_stl(filename, vertices, faces):
        """Save mesh as binary STL."""
        import struct
        
        with open(filename, 'wb') as f:
            f.write(b'\0' * 80)
            f.write(struct.pack('<I', len(faces)))
            
            for face in faces:
                v0, v1, v2 = vertices[face]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))

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

    export_params = {
        'miplevel': 1,           # Use mip level 2 for faster testing
        'extractwhich': 1,       # Selected segment and children, uncollapsed
        'blocksizex': 128,
        'blocksizey': 128,
        'blocksizez': 64,
        'overlap': 1,
        'slicestep': 1,
        'xunit': 5.0,            # Voxel size in nm
        'yunit': 5.0,
        'zunit': 50.0,
        'xscale': 0.001,
        'yscale': 0.001,
        'zscale': 0.001,
        'targetfolder': './test_output_meshes/',
        'targetfileprefix': 'test_',
        'fileformat': 1,         # 1 = OBJ
        'closesurfaces': 1,
        'invertz': 0,
        'includefoldernames': 0,
        'disablenetwarnings': 0,
        'erodedilate': 0,
        'usemipregionconstraint': 0,
        'outputoffsetx': 0,
        'outputoffsety': 0, 
        'outputoffsetz': 0,
        'objectcolors': 1,       # Use VAST colors
    }
    
    # Small region for testing
    # Note: At mip level 2, these coordinates will be divided by 4
    # So these full-res coordinates become [250-377, 250-377, 0-20] at mip 2
    region_params = {
        'xmin': 0,
        'xmax': 2713,
        'ymin': 0,
        'ymax': 2713,
        'zmin': 0,
        'zmax': 119,
    }
    extractor = SurfaceExtractor(vast, export_params, region_params)

    # Phase 1-4: Extract surfaces from blocks
    print("Phase 1-4: Extracting surfaces from blocks...")
    extractor.extract_surfaces()

    # Check for cancellation
    if extractor.canceled:
        print("Extraction canceled by user")
        vast.disconnect()
        return

    # Phase 5-6: Merge and export
    print("\nPhase 5-6: Merging meshes and writing files...")
    extractor.export_meshes()

    print("\nAll done!")
    vast.disconnect()






if __name__ == "__main__":
    main()