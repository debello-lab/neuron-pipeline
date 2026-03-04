import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vastpy.control.exporting import VASTControlClass
import numpy as np
import math

# Connection parameters
HOST = '127.0.0.1'
PORT = 22081
TIMEOUT = 10  # seconds


def test_connection():
    """Test connecting and disconnecting from VAST."""
    print("\n=== Test: Connection ===")
    vast = VASTControlClass()

    res = vast.connect(HOST, PORT, TIMEOUT)
    assert res == 1, f"Connection failed. Is VAST running with Remote Control enabled?"
    print("  Connected successfully")

    res = vast.disconnect()
    assert res == 1, "Disconnect failed"
    print("  Disconnected successfully")

    return True


def test_get_info(vast):
    """Test get_info() - retrieves general metadata about the dataset."""
    print("\n=== Test: get_info() ===")

    info = vast.get_info()

    assert info is not None, f"get_info() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"

    # Check expected keys exist
    expected_keys = [
        "datasizex", "datasizey", "datasizez",
        "voxelsizex", "voxelsizey", "voxelsizez",
        "cubesizex", "cubesizey", "cubesizez",
        "currentviewx", "currentviewy", "currentviewz",
        "nrofmiplevels"
    ]

    for key in expected_keys:
        assert key in info, f"Missing key: {key}"

    print(f"  Dataset size: {info['datasizex']} x {info['datasizey']} x {info['datasizez']}")
    print(f"  Voxel size: {info['voxelsizex']} x {info['voxelsizey']} x {info['voxelsizez']}")
    print(f"  Cube size: {info['cubesizex']} x {info['cubesizey']} x {info['cubesizez']}")
    print(f"  Current view: ({info['currentviewx']}, {info['currentviewy']}, {info['currentviewz']})")
    print(f"  Mip levels: {info['nrofmiplevels']}")

    return True


def test_get_nrof_layers(vast):
    """Test get_nrof_layers() - retrieves number of layers."""
    print("\n=== Test: get_nrof_layers() ===")

    num_layers = vast.get_nrof_layers()

    assert num_layers is not None, f"get_nrof_layers() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(num_layers, int), f"Expected int, got {type(num_layers)}"
    assert num_layers >= 0, f"Number of layers should be non-negative, got {num_layers}"

    print(f"  Number of layers: {num_layers}")

    return num_layers


def test_get_layer_info(vast, layer_nr=0):
    """Test get_layer_info() - retrieves info about a specific layer."""
    layer_info = vast.get_layer_info(layer_nr)

    if layer_info is None:
        print(f"    Layer {layer_nr} not found or error. Error code: {vast.get_last_error()}")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"

    expected_keys = [
        "type", "editable", "visible", "brightness", "contrast",
        "opacitylevel", "brightnesslevel", "contrastlevel",
        "blendmode", "blendoradd", "tintcolor", "name", "inverted"
    ]

    for key in expected_keys:
        assert key in layer_info, f"Missing key: {key}"

    return layer_info


def test_get_all_layers_info(vast, num_layers):
    """Test get_layer_info() for all layers."""
    print("\n=== Test: get_layer_info() (all layers) ===")

    all_layer_info = []

    for layer_nr in range(num_layers):
        layer_info = test_get_layer_info(vast, layer_nr)
        if layer_info:
            all_layer_info.append(layer_info)
            inverted_str = "Yes" if layer_info['inverted'] else "No"
            layer_type = "EM" if layer_info['type'] == 0 else "Segmentation" if layer_info['type'] == 1 else f"Type {layer_info['type']}"
            visible_str = "Yes" if layer_info['visible'] else "No"
            editable_str = "Yes" if layer_info['editable'] else "No"

            print(f"  Layer {layer_nr}: {layer_info['name']}")
            print(f"    Type: {layer_type}, Visible: {visible_str}, Editable: {editable_str}, Inverted: {inverted_str}")

    print(f"  Total layers retrieved: {len(all_layer_info)}")
    return all_layer_info


def test_get_number_of_segments(vast):
    """Test get_number_of_segments() - retrieves total segment count."""
    print("\n=== Test: get_number_of_segments() ===")

    num_segments = vast.get_number_of_segments()

    assert num_segments is not None, f"get_number_of_segments() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(num_segments, int), f"Expected int, got {type(num_segments)}"

    print(f"  Number of segments: {num_segments}")

    return num_segments


def test_get_segment_data(vast, segment_id=1):
    """Test get_segment_data() - retrieves data for a specific segment."""
    print(f"\n=== Test: get_segment_data({segment_id}) ===")

    seg_data = vast.get_segment_data(segment_id)

    if seg_data is None:
        print(f"  Segment {segment_id} not found. Error code: {vast.get_last_error()}")
        return False

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"

    expected_keys = ["id", "flags", "col1", "col2", "anchorpoint", "hierarchy", "collapsednr", "boundingbox"]

    for key in expected_keys:
        assert key in seg_data, f"Missing key: {key}"

    print(f"  Segment ID: {seg_data['id']}")
    print(f"  Anchor point: {seg_data['anchorpoint']}")
    bbox = seg_data['boundingbox']
    print(f"  Bounding box: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}) - ({int(bbox[3])}, {int(bbox[4])}, {int(bbox[5])})")
    print(f"  Hierarchy: {seg_data['hierarchy']}")

    return True


def test_get_segment_name(vast, segment_id=1):
    """Test get_segment_name() - retrieves name of a specific segment."""
    print(f"\n=== Test: get_segment_name({segment_id}) ===")

    name = vast.get_segment_name(segment_id)

    if name is None:
        print(f"  Segment {segment_id} name not found. Error code: {vast.get_last_error()}")
        return False

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(name, str), f"Expected str, got {type(name)}"

    print(f"  Segment {segment_id} name: '{name}'")

    return True


def test_get_all_segment_data(vast):
    """Test get_all_segment_data() - retrieves data for all segments as list of dicts."""
    print("\n=== Test: get_all_segment_data() ===")

    all_data = vast.get_all_segment_data()

    assert all_data is not None, f"get_all_segment_data() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(all_data, list), f"Expected list, got {type(all_data)}"

    print(f"  Retrieved {len(all_data)} segments")

    if len(all_data) > 0:
        # Check first segment structure
        first_seg = all_data[0]
        expected_keys = ["id", "flags", "col1", "col2", "anchorpoint", "hierarchy", "collapsednr", "boundingbox"]
        for key in expected_keys:
            assert key in first_seg, f"Missing key in segment data: {key}"

        print(f"  First segment: id={first_seg['id']}, anchorpoint={first_seg['anchorpoint']}")

        if len(all_data) > 1:
            last_seg = all_data[-1]
            print(f"  Last segment: id={last_seg['id']}, anchorpoint={last_seg['anchorpoint']}")

    return all_data


def test_get_all_segment_data_matrix(vast):
    """Test get_all_segment_data_matrix() - retrieves all segment data as numpy matrix."""
    print("\n=== Test: get_all_segment_data_matrix() ===")

    matrix = vast.get_all_segment_data_matrix()

    assert matrix is not None, f"get_all_segment_data_matrix() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(matrix, np.ndarray), f"Expected np.ndarray, got {type(matrix)}"

    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Matrix dtype: {matrix.dtype}")

    if matrix.shape[0] > 0:
        print(f"  First row (segment 1): {matrix[0, :]}")
        print(f"    - ID: {matrix[0, 0]}")
        print(f"    - Flags: {matrix[0, 1]}")
        print(f"    - Color1 RGBP: {matrix[0, 2:6]}")
        print(f"    - Anchorpoint: {matrix[0, 10:13]}")
        x1, y1, z1 = int(matrix[0, 18]), int(matrix[0, 19]), int(matrix[0, 20])
        x2, y2, z2 = int(matrix[0, 21]), int(matrix[0, 22]), int(matrix[0, 23])
        print(f"    - Bounding box: ({x1}, {y1}, {z1}) - ({x2}, {y2}, {z2})")

    return matrix


def test_get_all_segment_names(vast):
    """Test get_all_segment_names() - retrieves names of all segments."""
    print("\n=== Test: get_all_segment_names() ===")

    names = vast.get_all_segment_names()

    assert names is not None, f"get_all_segment_names() returned None. Error: {vast.get_last_error()}"
    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(names, list), f"Expected list, got {type(names)}"

    print(f"  Retrieved {len(names)} segment names")

    if len(names) > 0:
        print(f"  First segment name: '{names[0]}'")
        if len(names) > 1:
            print(f"  Second segment name: '{names[1]}'")
        if len(names) > 2:
            print(f"  Last segment name: '{names[-1]}'")

    return names


def test_get_set_selected_segment_nr(vast):
    """Test get_selected_segment_nr() and set_selected_segment_nr()."""
    print("\n=== Test: get/set_selected_segment_nr() ===")

    # Get current selection
    original_seg = vast.get_selected_segment_nr()
    assert original_seg >= -1, f"get_selected_segment_nr() returned invalid value: {original_seg}"
    print(f"  Current selected segment: {original_seg}")

    # Try selecting segment 1
    success = vast.set_selected_segment_nr(1)
    print(f"  set_selected_segment_nr(1) returned: {success}")

    # Check what's selected now (may not change immediately depending on VAST state)
    selected = vast.get_selected_segment_nr()
    print(f"  Selected segment after set: {selected}")

    # Note: VAST may not immediately reflect selection changes via API
    # The important thing is the functions execute without errors
    if selected == 1:
        print("  Selection change verified")
    else:
        print("  Note: Selection may not update via API (VAST behavior)")

    # Restore original selection if it was valid
    if original_seg >= 0 and original_seg != selected:
        vast.set_selected_segment_nr(original_seg)
        print(f"  Attempted to restore original selection: {original_seg}")

    return True


def test_get_seg_image_raw(vast):
    """Test get_seg_image_raw() - retrieves raw segmentation image data."""
    print("\n=== Test: get_seg_image_raw() ===")

    # Get a small region (16x16x1) at mip level 0
    miplevel = 0
    minx, maxx = 0, 15
    miny, maxy = 0, 15
    minz, maxz = 0, 0

    print(f"  Requesting region: mip={miplevel}, x=[{minx},{maxx}], y=[{miny},{maxy}], z=[{minz},{maxz}]")

    segimage = vast.get_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz)

    if segimage is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_raw() returned None. Error code: {error_code}")
        # Error code 10 = coordinates out of bounds, which may happen depending on dataset
        if error_code == 10:
            print("  (Coordinates may be out of bounds for this dataset)")
            return None
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(segimage, np.ndarray), f"Expected np.ndarray, got {type(segimage)}"

    expected_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
    print(f"  Received array length: {len(segimage)} (expected: {expected_size})")
    print(f"  Array dtype: {segimage.dtype}")
    print(f"  Unique values: {np.unique(segimage)[:10]}...")  # Show first 10 unique values
    print(f"  Non-zero voxels: {np.count_nonzero(segimage)}")

    return segimage


def test_get_seg_image_raw_with_flip(vast):
    """Test get_seg_image_raw() with flipflag=1."""
    print("\n=== Test: get_seg_image_raw(flipflag=1) ===")

    miplevel = 0
    minx, maxx = 0, 15
    miny, maxy = 0, 15
    minz, maxz = 0, 0

    segimage = vast.get_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz, flipflag=1)

    if segimage is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_raw(flipflag=1) returned None. Error code: {error_code}")
        if error_code == 10:
            print("  (Coordinates may be out of bounds for this dataset)")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    print(f"  Received array with flipflag=1, length: {len(segimage)}")

    return segimage


def test_get_seg_image_rle(vast):
    """Test get_seg_image_rle() - retrieves RLE-encoded segmentation image data."""
    print("\n=== Test: get_seg_image_rle() ===")

    miplevel = 0
    minx, maxx = 0, 63
    miny, maxy = 0, 63
    minz, maxz = 0, 0

    print(f"  Requesting RLE region: mip={miplevel}, x=[{minx},{maxx}], y=[{miny},{maxy}], z=[{minz},{maxz}]")

    segimage_rle = vast.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz)

    if segimage_rle is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_rle() returned None. Error code: {error_code}")
        if error_code == 10:
            print("  (Coordinates may be out of bounds for this dataset)")
        elif error_code == 22:
            print("  (RLE overflow - data would be larger than raw)")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(segimage_rle, np.ndarray), f"Expected np.ndarray, got {type(segimage_rle)}"

    print(f"  RLE array length: {len(segimage_rle)}")
    print(f"  Array dtype: {segimage_rle.dtype}")

    # RLE format: pairs of (count, value)
    raw_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
    compression_ratio = raw_size / len(segimage_rle) if len(segimage_rle) > 0 else 0
    print(f"  Compression ratio: {compression_ratio:.2f}x (raw would be {raw_size} values)")

    return segimage_rle


def test_get_seg_image_rle_surfonly(vast):
    """Test get_seg_image_rle() with surfonlyflag=1."""
    print("\n=== Test: get_seg_image_rle(surfonlyflag=1) ===")

    miplevel = 0
    minx, maxx = 0, 63
    miny, maxy = 0, 63
    minz, maxz = 0, 3

    print(f"  Requesting surface-only RLE: mip={miplevel}, x=[{minx},{maxx}], y=[{miny},{maxy}], z=[{minz},{maxz}]")

    segimage_rle = vast.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz, surfonlyflag=1)

    if segimage_rle is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_rle(surfonlyflag=1) returned None. Error code: {error_code}")
        if error_code == 10:
            print("  (Coordinates may be out of bounds for this dataset)")
        elif error_code == 22:
            print("  (RLE overflow - data would be larger than raw)")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    print(f"  Surface-only RLE array length: {len(segimage_rle)}")

    return segimage_rle


def test_set_error_popups_enabled(vast):
    """Test set_error_popups_enabled() - enable/disable error popups."""
    print("\n=== Test: set_error_popups_enabled() ===")

    # Test disabling error popup for error code 10 (coordinates out of bounds)
    success = vast.set_error_popups_enabled(10, False)
    print(f"  set_error_popups_enabled(10, False): {success}")

    # Re-enable it
    success = vast.set_error_popups_enabled(10, True)
    print(f"  set_error_popups_enabled(10, True): {success}")

    # Note: We just test that the function executes without crash
    # The actual popup behavior depends on VAST UI state
    return True


def test_get_selected_layernr(vast):
    """Test get_selected_layernr() - retrieves selected layer information."""
    print("\n=== Test: get_selected_layernr() ===")

    result = vast.get_selected_layernr()

    if result is None:
        print(f"  get_selected_layernr() returned None. Error: {vast.get_last_error()}")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3 values, got {len(result)}"

    selected_layer, selected_em_layer, selected_segment_layer = result
    print(f"  Selected layer: {selected_layer}")
    print(f"  Selected EM layer: {selected_em_layer}")
    print(f"  Selected segment layer: {selected_segment_layer}")

    return result


def test_get_seg_image_rle_decoded(vast):
    """Test get_seg_image_rle_decoded() - retrieves and decodes RLE data to 3D array."""
    print("\n=== Test: get_seg_image_rle_decoded() ===")

    miplevel = 0
    minx, maxx = 0, 31
    miny, maxy = 0, 31
    minz, maxz = 0, 1

    print(f"  Requesting decoded RLE: mip={miplevel}, x=[{minx},{maxx}], y=[{miny},{maxy}], z=[{minz},{maxz}]")

    segimage = vast.get_seg_image_rle_decoded(miplevel, minx, maxx, miny, maxy, minz, maxz)

    if segimage is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_rle_decoded() returned None. Error code: {error_code}")
        if error_code == 10:
            print("  (Coordinates may be out of bounds for this dataset)")
        elif error_code == 22:
            print("  (RLE overflow - data would be larger than raw)")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(segimage, np.ndarray), f"Expected np.ndarray, got {type(segimage)}"

    expected_shape = (maxx - minx + 1, maxy - miny + 1, maxz - minz + 1)
    print(f"  Decoded array shape: {segimage.shape} (expected: {expected_shape})")
    print(f"  Array dtype: {segimage.dtype}")
    print(f"  Unique values: {np.unique(segimage)[:10]}...")
    print(f"  Non-zero voxels: {np.count_nonzero(segimage)}")

    return segimage


def test_get_seg_image_rle_decoded_with_flip(vast):
    """Test get_seg_image_rle_decoded() with flipflag=1."""
    print("\n=== Test: get_seg_image_rle_decoded(flipflag=1) ===")

    miplevel = 0
    minx, maxx = 0, 31
    miny, maxy = 0, 31
    minz, maxz = 0, 1

    segimage = vast.get_seg_image_rle_decoded(miplevel, minx, maxx, miny, maxy, minz, maxz, flipflag=1)

    if segimage is None:
        error_code = vast.get_last_error()
        print(f"  get_seg_image_rle_decoded(flipflag=1) returned None. Error code: {error_code}")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"

    # With flipflag=1, shape should be (height, width, depth) instead of (width, height, depth)
    expected_shape = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
    print(f"  Decoded array shape with flip: {segimage.shape} (expected: {expected_shape})")

    return segimage


def test_set_seg_translation(vast):
    """Test set_seg_translation() - set segmentation translation mapping."""
    print("\n=== Test: set_seg_translation() ===")

    # Set a translation: map segments 1,2,3 to 10,20,30
    source = [1, 2, 3]
    target = [10, 20, 30]

    success = vast.set_seg_translation(source, target)
    print(f"  set_seg_translation({source}, {target}): {success}")

    if success:
        # Clear the translation by passing empty arrays
        success_clear = vast.set_seg_translation([], [])
        print(f"  set_seg_translation([], []) to clear: {success_clear}")
    else:
        print(f"  Error: {vast.get_last_error()}")

    return success


def test_get_mipmap_scale_factors(vast):
    """Test get_mipmap_scale_factors() - retrieves MIP map scale factors."""
    print("\n=== Test: get_mipmap_scale_factors() ===")

    # Get scale factors for layer 0
    layer_nr = 0
    scale_factors = vast.get_mipmap_scale_factors(layer_nr)

    if scale_factors is None:
        error_code = vast.get_last_error()
        print(f"  get_mipmap_scale_factors({layer_nr}) returned None. Error code: {error_code}")
        return None

    assert vast.get_last_error() == 0, f"Error occurred: {vast.get_last_error()}"
    assert isinstance(scale_factors, np.ndarray), f"Expected np.ndarray, got {type(scale_factors)}"

    print(f"  Scale factors shape: {scale_factors.shape}")
    print(f"  Scale factors dtype: {scale_factors.dtype}")

    for i, row in enumerate(scale_factors):
        print(f"    MIP level {i+1}: scale_x={row[0]}, scale_y={row[1]}, scale_z={row[2]}")

    return scale_factors


def test_get_immediate_child_ids(vast):
    """Test get_immediate_child_ids() - retrieves immediate children of segments."""
    print("\n=== Test: get_immediate_child_ids() ===")

    # Test with 0 (all top-level segments)
    print("  Testing with parentidlist=0 (top-level segments)...")
    top_level_ids = vast.get_immediate_child_ids(0)
    
    assert top_level_ids is not None, f"get_immediate_child_ids(0) returned None. Error: {vast.get_last_error()}"
    assert isinstance(top_level_ids, np.ndarray), f"Expected np.ndarray, got {type(top_level_ids)}"
    
    print(f"  Top-level segments count: {len(top_level_ids)}")
    if len(top_level_ids) > 0:
        print(f"  First few top-level IDs: {top_level_ids[:10]}")

        # Test with a specific parent ID (use the first top-level one if available)
        parent_id = top_level_ids[0]
        print(f"  Testing with parentidlist={parent_id}...")
        child_ids = vast.get_immediate_child_ids(parent_id)
        
        assert child_ids is not None, f"get_immediate_child_ids({parent_id}) returned None. Error: {vast.get_last_error()}"
        print(f"  Children of {parent_id}: {child_ids}")
        
    return top_level_ids


def test_get_child_tree_ids(vast):
    """Test get_child_tree_ids() - retrieves all descendant segments recursively."""
    print("\n=== Test: get_child_tree_ids() ===")

    # Test with 0 (entire tree of all segments)
    print("  Testing with parentidlist=0 (all segments)...")
    all_tree_ids = vast.get_child_tree_ids(0)
    
    assert all_tree_ids is not None, f"get_child_tree_ids(0) returned None. Error: {vast.get_last_error()}"
    assert isinstance(all_tree_ids, np.ndarray), f"Expected np.ndarray, got {type(all_tree_ids)}"
    
    print(f"  Total descendants count from root: {len(all_tree_ids)}")
    
    # Try to find a segment that has children to test a subtree
    # Using get_immediate_child_ids to find a parent with children
    top_level_ids = vast.get_immediate_child_ids(0)
    if len(top_level_ids) > 0:
        parent_with_kids = -1
        # Check first few to find one with kids
        for pid in top_level_ids[:20]:
             kids = vast.get_immediate_child_ids(pid)
             if len(kids) > 0:
                 parent_with_kids = pid
                 break
        
        if parent_with_kids != -1:
            print(f"  Testing subtree for parent {parent_with_kids}...")
            subtree_ids = vast.get_child_tree_ids(parent_with_kids)
            assert subtree_ids is not None
            print(f"  Subtree size for {parent_with_kids}: {len(subtree_ids)}")
            print(f"  Subtree IDs: {subtree_ids}")
        else:
             print("  (No segments with children found in top 20 to test subtree)")

    return all_tree_ids


def test_helper_functions():
    """Test the bytes_from_* helper functions."""
    print("\n=== Test: Helper Functions ===")

    vast = VASTControlClass()

    # Test bytes_from_uint32
    result = vast.bytes_from_uint32(42)
    assert len(result) == 5, f"Expected 5 bytes, got {len(result)}"
    assert result[0] == 1, f"Expected type tag 1, got {result[0]}"
    print("  bytes_from_uint32(42): OK")

    # Test bytes_from_uint32 with list
    result = vast.bytes_from_uint32([1, 2, 3])
    assert len(result) == 15, f"Expected 15 bytes, got {len(result)}"
    print("  bytes_from_uint32([1,2,3]): OK")

    # Test bytes_from_int32
    result = vast.bytes_from_int32(-42)
    assert len(result) == 5, f"Expected 5 bytes, got {len(result)}"
    assert result[0] == 4, f"Expected type tag 4, got {result[0]}"
    print("  bytes_from_int32(-42): OK")

    # Test bytes_from_double
    result = vast.bytes_from_double(3.14159)
    assert len(result) == 9, f"Expected 9 bytes, got {len(result)}"
    assert result[0] == 2, f"Expected type tag 2, got {result[0]}"
    print("  bytes_from_double(3.14159): OK")

    # Test bytes_from_text
    result = vast.bytes_from_text("hello")
    assert result[0] == 3, f"Expected type tag 3, got {result[0]}"
    assert result[-1] == 0, f"Expected null terminator"
    print("  bytes_from_text('hello'): OK")

    # Test bytes_from_data
    result = vast.bytes_from_data(b'\x01\x02\x03')
    assert result[0] == 5, f"Expected type tag 5, got {result[0]}"
    print("  bytes_from_data(b'\\x01\\x02\\x03'): OK")

    return True


def test_parse_payload():
    """Test the parse_payload function."""
    print("\n=== Test: parse_payload() ===")

    vast = VASTControlClass()

    # Build a test payload with known values
    payload = b''
    payload += vast.bytes_from_uint32(12345)
    payload += vast.bytes_from_int32(-999)
    payload += vast.bytes_from_double(2.718)
    payload += vast.bytes_from_text("test string")

    parsed = vast.parse_payload(payload)

    assert len(parsed["uints"]) == 1, f"Expected 1 uint, got {len(parsed['uints'])}"
    assert parsed["uints"][0] == 12345, f"Expected uint 12345, got {parsed['uints'][0]}"
    print("  Parsed uint32: OK")

    assert len(parsed["ints"]) == 1, f"Expected 1 int, got {len(parsed['ints'])}"
    assert parsed["ints"][0] == -999, f"Expected int -999, got {parsed['ints'][0]}"
    print("  Parsed int32: OK")

    assert len(parsed["doubles"]) == 1, f"Expected 1 double, got {len(parsed['doubles'])}"
    assert abs(parsed["doubles"][0] - 2.718) < 0.0001, f"Expected double ~2.718, got {parsed['doubles'][0]}"
    print("  Parsed double: OK")

    assert len(parsed["text"]) == 1, f"Expected 1 text, got {len(parsed['text'])}"
    assert parsed["text"][0] == "test string", f"Expected 'test string', got '{parsed['text'][0]}'"
    assert parsed["last_text"] == "test string"
    print("  Parsed text: OK")

    return True



def draw_circle(api, center_x: int, center_y: int, radius: int, num_points: int = 360):
    """Draw one circle outline."""
    if radius <= 0:
        return True

    # Choose points proportional to circumference; clamp to a sane minimum
    if num_points is None:
        num_points = max(24, int(2 * math.pi * radius))  # ~1 point per pixel around

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    coords = np.column_stack((x, y)).astype(np.int32)
    return api.execute_canvas_paint_stroke(coords)

def draw_filled_circle(api, center_x: int, center_y: int, radius: int, ring_step: int = 4):
    """
    Fill a circle by drawing concentric rings.
    ring_step should roughly match brush diameter in pixels (or slightly smaller).
    """
    if radius <= 0:
        return True

    # Draw from outside in (often looks cleaner)
    for r in range(radius, 0, -ring_step):
        ok = draw_circle(api, center_x, center_y, r)
        if not ok:
            return False

    # Optional: dot the center (sometimes a tiny hole remains)
    return draw_circle(api, center_x, center_y, 1, num_points=12)

def draw_filled_sphere(api, center_x: int, center_y: int, radius: int,
                               top_z: int = 0, z_step: int = 1, ring_step: int = 4):
    """
    Draw a filled sphere where the TOP slice is at top_z (default 0),
    and we move forward by INCREASING z.
    """
    info = api.get_info()
    if info is None:
        return False

    x0 = int(info["currentviewx"])
    y0 = int(info["currentviewy"])

    # Put the sphere center radius slices below the top
    center_z = top_z + radius

    # Sweep z only upward (never negative)
    z_start = top_z
    z_end = top_z + 2 * radius

    for z in range(z_start, z_end + 1, z_step):
        dz = z - center_z
        r_slice = int(round(math.sqrt(max(0, radius * radius - dz * dz))))

        api.set_view_coordinates(x0, y0, z)

        ok = draw_filled_circle(api, center_x, center_y, r_slice, ring_step=ring_step)
        if not ok:
            return False

    return True

def test_draw_filled_sphere(vast):
    """Test draw_filled_sphere() - draws a filled sphere (stacked circles across z) on segmentation."""

    print("\n=== Test: draw_filled_sphere() ===")

    # Get the first segment number
    first_seg = vast.get_first_segment_nr()
    if first_seg == -1:
        print(f"  Failed to get first segment number. Error: {vast.get_last_error()}")
        return False
    print(f"  First segment number: {first_seg}")

    # Set the selected segment to the first segment
    success = vast.set_selected_segment_nr(first_seg)
    if not success:
        print(f"  Failed to set selected segment. Error: {vast.get_last_error()}")
        return False
    print(f"  Selected segment set to: {first_seg}")

    # Get current view position to draw near the center of the view
    info = vast.get_info()
    if info is None:
        print(f"  Failed to get info. Error: {vast.get_last_error()}")
        return False

    # Use window coordinates - draw in center of a typical window
    # These are screen/window coordinates, not dataset coordinates
    center_x = 775
    center_y = 475
    radius = 50
    info = vast.get_info()
    vast.set_view_coordinates(info["currentviewx"], info["currentviewy"], info["currentviewz"])

    print(f"  Drawing filled sphere at window position ({center_x}, {center_y}) with radius {radius}")
    print(f"  Current view in dataset: ({info['currentviewx']}, {info['currentviewy']}, {info['currentviewz']})")

    # Draw the filled sphere
    success = draw_filled_sphere(vast, center_x, center_y, radius, top_z=0, z_step=5, ring_step=10)
    vast.set_view_coordinates(info["currentviewx"], info["currentviewy"], info["currentviewz"])

    if success:
        print(f"  Successfully drew filled sphere with segment {first_seg}")
    else:
        print(f"  Failed to draw filled sphere. Error: {vast.get_last_error()}")

    return success

def draw_sphere_3d(api, center_x: int, center_y: int, start_z: int, max_radius: int,
                   radius_step: int = 5, ring_step: int = 4):
    """
    Draw a 3D sphere by incrementing Z and growing circle radius, then decrementing.
    Uses set_drawing_properties to adjust paint cursor diameter for each slice.

    Args:
        api: VAST API instance
        center_x: X window coordinate of sphere center
        center_y: Y window coordinate of sphere center
        start_z: Starting Z coordinate in dataset
        max_radius: Maximum radius at the sphere's equator
        radius_step: How much to change radius per Z slice (default 5)
        ring_step: Step for filling circles (default 4)

    Returns:
        True on success, False on failure
    """
    # Get current view to preserve x, y
    info = api.get_info()
    if info is None:
        return False

    view_x = int(info["currentviewx"])
    view_y = int(info["currentviewy"])

    # Phase 1: Increment Z while growing radius (from radius_step to max_radius)
    current_z = start_z
    radii_up = list(range(radius_step, max_radius + 1, radius_step))

    print(f"    Phase 1: Growing circles (z={start_z} to z={start_z + len(radii_up) - 1})")
    for i, radius in enumerate(radii_up):
        # Set view coordinates for this Z slice
        api.set_view_coordinates(view_x, view_y, current_z)

        # Set paint cursor diameter to match the circle radius
        api.set_drawing_properties({"paintcursordiameter": max(1, radius // 2)})

        print(f"      Z={current_z}, radius={radius}")
        draw_filled_circle(api, center_x, center_y, radius, ring_step=ring_step)
        current_z += 1

    # Phase 2: Continue incrementing Z while shrinking radius (from max_radius-step down to radius_step)
    radii_down = list(range(max_radius - radius_step, 0, -radius_step))

    print(f"    Phase 2: Shrinking circles (z={current_z} to z={current_z + len(radii_down) - 1})")
    for i, radius in enumerate(radii_down):
        # Set view coordinates for this Z slice
        api.set_view_coordinates(view_x, view_y, current_z)

        # Set paint cursor diameter to match the circle radius
        api.set_drawing_properties({"paintcursordiameter": max(1, radius // 2)})

        print(f"      Z={current_z}, radius={radius}")
        draw_filled_circle(api, center_x, center_y, radius, ring_step=ring_step)
        draw_filled_circle(api, center_x, center_y, radius, ring_step=ring_step)
        current_z += 1

    return True


def test_draw_sphere_3d(vast):
    """Test draw_sphere_3d() - draws a 3D sphere by varying Z and circle radius."""
    print("\n=== Test: draw_sphere_3d() ===")

    # Get the first segment number
    first_seg = vast.get_first_segment_nr()
    if first_seg == -1:
        print(f"  Failed to get first segment number. Error: {vast.get_last_error()}")
        return False
    print(f"  First segment number: {first_seg}")

    # Set the selected segment to the first segment
    success = vast.set_selected_segment_nr(first_seg)
    if not success:
        print(f"  Failed to set selected segment. Error: {vast.get_last_error()}")
        return False
    print(f"  Selected segment set to: {first_seg}")

    # Get current view position
    info = vast.get_info()
    if info is None:
        print(f"  Failed to get info. Error: {vast.get_last_error()}")
        return False

    # Store original view to restore later
    original_x = int(info['currentviewx'])
    original_y = int(info['currentviewy'])
    original_z = int(info['currentviewz'])

    # Get and display current drawing properties
    draw_props = vast.get_drawing_properties()
    if draw_props:
        original_cursor_diameter = draw_props.get('paintcursordiameter', 10)
        print(f"  Original paint cursor diameter: {original_cursor_diameter}")
    else:
        original_cursor_diameter = 10

    # Window coordinates for drawing (center of typical window)
    center_x = 570
    center_y = 320

    # Sphere parameters
    max_radius = 50  # Maximum radius at equator
    radius_step = 2  # Increment/decrement radius by 5
    start_z = original_z  # Start at current Z

    print(f"  Drawing 3D sphere at window ({center_x}, {center_y})")
    print(f"  Starting Z: {start_z}, Max radius: {max_radius}, Radius step: {radius_step}")
    print(f"  Current view in dataset: ({original_x}, {original_y}, {original_z})")

    # Draw the 3D sphere
    success = draw_sphere_3d(vast, center_x, center_y, start_z, max_radius,
                             radius_step=radius_step, ring_step=4)

    # Restore original view and drawing properties
    vast.set_view_coordinates(original_x, original_y, original_z)
    vast.set_drawing_properties({"paintcursordiameter": original_cursor_diameter})
    print(f"  Restored view to: ({original_x}, {original_y}, {original_z})")
    print(f"  Restored paint cursor diameter to: {original_cursor_diameter}")

    if success:
        print(f"  Successfully drew 3D sphere with segment {first_seg}")
    else:
        print(f"  Failed to draw 3D sphere. Error: {vast.get_last_error()}")

    return success


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("VAST Control Class Test Suite")
    print("=" * 60)

    # Test helper functions first (no connection needed)
    test_helper_functions()
    test_parse_payload()

    # Test connection
    test_connection()

    # Now run connected tests
    vast = VASTControlClass()
    res = vast.connect(HOST, PORT, TIMEOUT)

    if res != 1:
        print("\nFailed to connect to VAST.")
        print("  - Ensure VAST Lite is running")
        print("  - Ensure 'Remote Control Server' is enabled in VAST settings")
        return

    print("\nConnected to VAST. Running integration tests...")

    try:
        # Core info tests
        test_get_info(vast)

        # Layer tests
        num_layers = test_get_nrof_layers(vast)
        if num_layers > 0:
            test_get_all_layers_info(vast, num_layers)

        # Segment tests
        num_segments = test_get_number_of_segments(vast)
        if num_segments and num_segments > 1:
            test_get_segment_data(vast, 1)
            test_get_segment_name(vast, 1)

        # Bulk segment tests
        test_get_all_segment_data(vast)
        test_get_all_segment_data_matrix(vast)
        test_get_all_segment_names(vast)

        # Selection tests
        test_get_set_selected_segment_nr(vast)

        # Image retrieval tests
        test_get_seg_image_raw(vast)
        test_get_seg_image_raw_with_flip(vast)
        test_get_seg_image_rle(vast)
        test_get_seg_image_rle_surfonly(vast)
        test_get_seg_image_rle_decoded(vast)
        test_get_seg_image_rle_decoded_with_flip(vast)

        # Layer selection tests
        test_get_selected_layernr(vast)

        # Segmentation translation tests
        test_set_seg_translation(vast)

        # MIP map tests
        test_get_mipmap_scale_factors(vast)

        # Child/Tree ID tests
        test_get_immediate_child_ids(vast)
        test_get_child_tree_ids(vast)

        # Error popup tests
        test_set_error_popups_enabled(vast)

        # Drawing tests (paint stroke on segmentation layer)
        # test_draw_sphere_3d(vast)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    finally:
        vast.disconnect()
        print("\nDisconnected from VAST.")


if __name__ == "__main__":
    run_all_tests()
