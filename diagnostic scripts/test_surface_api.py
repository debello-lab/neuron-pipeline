"""
Test script for all VAST API functions used by surface extraction (exportSkeleton.py).
Run this with VAST open and a dataset loaded.
"""
from VASTControlClass import VASTControlClass

def test_surface_extraction_api():
    vast = VASTControlClass()
    
    print("=" * 60)
    print("VAST Surface Extraction API Test")
    print("=" * 60)
    
    # 1. Connection
    print("\n[1] Testing connect()...")
    if not vast.connect():
        print("   FAIL: Could not connect to VAST")
        return
    print("   PASS: Connected to VAST")
    
    results = {}
    
    # 2. set_error_popups_enabled
    print("\n[2] Testing set_error_popups_enabled()...")
    try:
        res = vast.set_error_popups_enabled(74, False)
        results['set_error_popups_enabled'] = 'PASS' if res else 'FAIL'
        print(f"   {'PASS' if res else 'FAIL'}: set_error_popups_enabled(74, False) = {res}")
    except Exception as e:
        results['set_error_popups_enabled'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 3. get_info
    print("\n[3] Testing get_info()...")
    try:
        info = vast.get_info()
        if info and 'datasizex' in info:
            results['get_info'] = 'PASS'
            print(f"   PASS: Dataset size = {info['datasizex']}x{info['datasizey']}x{info['datasizez']}")
        else:
            results['get_info'] = 'FAIL: empty result'
            print("   FAIL: Empty or invalid info returned")
    except Exception as e:
        results['get_info'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 4. get_selected_layer_nr
    print("\n[4] Testing get_selected_layer_nr()...")
    try:
        layer_info = vast.get_selected_layer_nr()
        if layer_info and 'selected_layer' in layer_info:
            results['get_selected_layer_nr'] = 'PASS'
            print(f"   PASS: selected_layer={layer_info.get('selected_layer')}, "
                  f"selected_segment_layer={layer_info.get('selected_segment_layer')}")
        else:
            results['get_selected_layer_nr'] = 'FAIL: empty result'
            print(f"   FAIL: {layer_info}")
    except Exception as e:
        results['get_selected_layer_nr'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 5. get_all_segment_data
    print("\n[5] Testing get_all_segment_data()...")
    try:
        seg_data = vast.get_all_segment_data()
        if seg_data is not None:
            results['get_all_segment_data'] = 'PASS'
            print(f"   PASS: Got {len(seg_data)} segments")
        else:
            results['get_all_segment_data'] = 'FAIL: None returned'
            print("   FAIL: None returned")
    except Exception as e:
        results['get_all_segment_data'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 6. get_all_segment_data_matrix
    print("\n[6] Testing get_all_segment_data_matrix()...")
    try:
        matrix, success = vast.get_all_segment_data_matrix()
        if success == 1:
            results['get_all_segment_data_matrix'] = 'PASS'
            print(f"   PASS: Matrix shape = {matrix.shape}")
        else:
            results['get_all_segment_data_matrix'] = 'FAIL'
            print(f"   FAIL: success={success}")
    except Exception as e:
        results['get_all_segment_data_matrix'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 7. get_all_segment_names
    print("\n[7] Testing get_all_segment_names()...")
    try:
        names = vast.get_all_segment_names()
        if names is not None:
            results['get_all_segment_names'] = 'PASS'
            print(f"   PASS: Got {len(names)} segment names")
            if len(names) > 0:
                print(f"   First few names: {names[:min(3, len(names))]}")
        else:
            results['get_all_segment_names'] = 'FAIL'
            print("   FAIL: None returned")
    except Exception as e:
        results['get_all_segment_names'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 8. get_mipmap_scale_factors
    print("\n[8] Testing get_mipmap_scale_factors()...")
    try:
        scale_factors = vast.get_mipmap_scale_factors(1)
        if scale_factors:
            results['get_mipmap_scale_factors'] = 'PASS'
            print(f"   PASS: Got {len(scale_factors)} mip levels")
            for i, sf in enumerate(scale_factors):
                print(f"      Mip {i+1}: {sf}")
        else:
            results['get_mipmap_scale_factors'] = 'FAIL'
            print("   FAIL: Empty result")
    except Exception as e:
        results['get_mipmap_scale_factors'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 9. get_data_size_at_mip
    print("\n[9] Testing get_data_size_at_mip()...")
    try:
        size = vast.get_data_size_at_mip(1, 0)  # layer 1, mip 0
        if size:
            results['get_data_size_at_mip'] = 'PASS'
            print(f"   PASS: Size at mip 0 = {size}")
        else:
            results['get_data_size_at_mip'] = 'FAIL'
            print("   FAIL: None returned")
    except Exception as e:
        results['get_data_size_at_mip'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 10. set_seg_translation
    print("\n[10] Testing set_seg_translation()...")
    try:
        res = vast.set_seg_translation([], [])
        results['set_seg_translation'] = 'PASS' if res else 'FAIL'
        print(f"   {'PASS' if res else 'FAIL'}: set_seg_translation([], []) = {res}")
    except Exception as e:
        results['set_seg_translation'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 11. get_seg_image_rle_decoded_bboxes
    print("\n[11] Testing get_seg_image_rle_decoded_bboxes()...")
    try:
        seg_image, values, counts, bboxes = vast.get_seg_image_rle_decoded_bboxes(
            0, 0, 63, 0, 63, 0, 0)  # mip 0, small area, single slice
        if seg_image is not None:
            results['get_seg_image_rle_decoded_bboxes'] = 'PASS'
            print(f"   PASS: Image shape = {seg_image.shape}, "
                  f"unique vals = {len(values)}, bboxes = {len(bboxes)}")
        else:
            results['get_seg_image_rle_decoded_bboxes'] = 'FAIL'
            print("   FAIL: None returned")
    except Exception as e:
        results['get_seg_image_rle_decoded_bboxes'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 12. get_screenshot_image
    print("\n[12] Testing get_screenshot_image()...")
    try:
        img = vast.get_screenshot_image(0, 0, 63, 0, 63, 0, 0)
        if img is not None:
            results['get_screenshot_image'] = 'PASS'
            print(f"   PASS: Screenshot shape = {img.shape}, dtype = {img.dtype}")
        else:
            results['get_screenshot_image'] = 'FAIL'
            print("   FAIL: None returned")
    except Exception as e:
        results['get_screenshot_image'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 13. get_anno_layer_nr_of_objects (for skeleton functions)
    print("\n[13] Testing get_anno_layer_nr_of_objects()...")
    try:
        num_objects, first_obj = vast.get_anno_layer_nr_of_objects()
        results['get_anno_layer_nr_of_objects'] = 'PASS'
        print(f"   PASS: num_objects={num_objects}, first_obj={first_obj}")
    except Exception as e:
        results['get_anno_layer_nr_of_objects'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 14. set_selected_anno_object_nr (if we have anno objects)
    print("\n[14] Testing set_selected_anno_object_nr()...")
    try:
        if num_objects > 0:
            res = vast.set_selected_anno_object_nr(first_obj)
            results['set_selected_anno_object_nr'] = 'PASS' if res else 'FAIL'
            print(f"   {'PASS' if res else 'FAIL'}: set_selected_anno_object_nr({first_obj})")
        else:
            results['set_selected_anno_object_nr'] = 'SKIP: no anno objects'
            print("   SKIP: No annotation objects available")
    except Exception as e:
        results['set_selected_anno_object_nr'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # 15. get_ao_node_data
    print("\n[15] Testing get_ao_node_data()...")
    try:
        node_data = vast.get_ao_node_data()
        if node_data is not None:
            results['get_ao_node_data'] = 'PASS'
            print(f"   PASS: Node data shape = {node_data.shape}")
        else:
            results['get_ao_node_data'] = 'SKIP: No nodes returned (may require skeleton selected)'
            print("   SKIP: None returned (may require selected skeleton)")
    except Exception as e:
        results['get_ao_node_data'] = f'ERROR: {e}'
        print(f"   ERROR: {e}")
    
    # Disconnect
    print("\n[16] Testing disconnect()...")
    vast.disconnect()
    print("   PASS: Disconnected")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if 'FAIL' in v or 'ERROR' in v)
    skipped = sum(1 for v in results.values() if 'SKIP' in v)
    
    for func, status in results.items():
        icon = "✓" if status == 'PASS' else ("⊘" if 'SKIP' in status else "✗")
        print(f"  {icon} {func}: {status}")
    
    print(f"\nTotal: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    test_surface_extraction_api()
