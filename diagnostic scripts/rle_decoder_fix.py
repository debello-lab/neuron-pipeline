"""
Patched RLE decoder for VastControlClass to fix overflow issues.

This file contains a monkey-patch fix for the RLE decoding overflow problem.
Use this until the main VastControlClass is updated.
"""

import numpy as np
from typing import Optional


def get_seg_image_rle_decoded_fixed(
    vast_instance,
    miplevel: int,
    minx: int,
    maxx: int,
    miny: int,
    maxy: int,
    minz: int,
    maxz: int,
    surfonlyflag: int = 0,
    flipflag: int = 0,
    immediateflag: int = 0,
    requestloadflag: int = 0
) -> Optional[np.ndarray]:
    """
    Fixed version of get_seg_image_rle_decoded with proper overflow handling.
    
    Changes from original:
    1. Use int64 for all indexing operations
    2. Explicit type casting to prevent overflow
    3. Added bounds checking
    4. Better error reporting
    """
    # Get RLE data using the original method
    segimage_rle = vast_instance.get_seg_image_rle(
        miplevel, minx, maxx, miny, maxy, minz, maxz,
        surfonlyflag, immediateflag, requestloadflag
    )
    
    if segimage_rle is None:
        return None
    
    # Calculate dimensions
    width = maxx - minx + 1
    height = maxy - miny + 1
    depth = maxz - minz + 1
    total_size = int(width) * int(height) * int(depth)  # Force int64
    
    print(f"[RLE Decoder] Allocating array for {total_size:,} voxels")
    print(f"[RLE Decoder] Dimensions: {width} x {height} x {depth}")
    print(f"[RLE Decoder] RLE data length: {len(segimage_rle):,} values ({len(segimage_rle)//2:,} runs)")
    
    # Allocate output array
    segimage = np.zeros(total_size, dtype=np.uint16)
    
    # Decode RLE with proper int64 handling
    dp = np.int64(0)  # destination pointer - explicitly int64
    decoded_voxels = np.int64(0)
    max_run_length = 0
    
    # RLE format: [value, count, value, count, ...]
    for sp in range(0, len(segimage_rle), 2):
        if sp + 1 >= len(segimage_rle):
            break
        
        val = np.uint16(segimage_rle[sp])
        num = np.int64(segimage_rle[sp + 1])  # CRITICAL: cast to int64
        
        # Track statistics
        if num > max_run_length:
            max_run_length = num
        decoded_voxels += num
        
        # Bounds checking
        if dp + num > total_size:
            print(f"[RLE Decoder] ERROR: Run would exceed array bounds!")
            print(f"  Position: {dp:,}, Run length: {num:,}, Total size: {total_size:,}")
            print(f"  This would write to position {dp + num:,}")
            # Truncate to fit
            num = total_size - dp
            if num <= 0:
                break
        
        # Write run
        try:
            segimage[int(dp):int(dp + num)] = val
        except Exception as e:
            print(f"[RLE Decoder] ERROR writing run: {e}")
            print(f"  dp={dp}, num={num}, val={val}")
            break
        
        dp += num
    
    print(f"[RLE Decoder] Decoded {decoded_voxels:,} / {total_size:,} voxels ({100*decoded_voxels/total_size:.1f}%)")
    print(f"[RLE Decoder] Max run length: {max_run_length:,}")
    print(f"[RLE Decoder] Final position: {dp:,}")
    
    if dp < total_size:
        print(f"[RLE Decoder] WARNING: Only filled {100*dp/total_size:.1f}% of array")
    
    # Reshape to 3D array (Fortran order for MATLAB compatibility)
    segimage = segimage.reshape((width, height, depth), order='F')
    
    if flipflag == 1:
        # Permute: swap first two dimensions [height, width, depth]
        segimage = np.transpose(segimage, (1, 0, 2))
    
    return segimage


def diagnose_segment_data(vast_instance, segment_id: int):
    """
    Diagnostic tool to check what's happening with segment data loading.
    """
    print("\n" + "="*80)
    print(f"DIAGNOSTIC: Segment {segment_id}")
    print("="*80)
    
    # Check if segment exists
    print("\n[1] Checking segment metadata...")
    seg_data = vast_instance.get_segment_data(segment_id)
    if not seg_data:
        print(f"  ERROR: Segment {segment_id} not found!")
        return
    
    print(f"  Name: {seg_data.get('name', 'UNKNOWN')}")
    bbox = seg_data.get('boundingbox', [])
    print(f"  Bounding box: {bbox}")
    
    if len(bbox) >= 6:
        bbox_volume = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
        print(f"  Bounding box volume: {bbox_volume:,} voxels")
    
    # Test loading without translation
    print("\n[2] Testing data load WITHOUT translation...")
    minx, maxx = bbox[0], bbox[3]
    miny, maxy = bbox[1], bbox[4]
    minz, maxz = bbox[2], bbox[5]
    
    # Add padding
    minx = max(0, minx - 2)
    maxx = maxx + 2
    miny = max(0, miny - 2)
    maxy = maxy + 2
    minz = max(0, minz - 2)
    maxz = maxz + 2
    
    print(f"  Loading region: X=[{minx},{maxx}] Y=[{miny},{maxy}] Z=[{minz},{maxz}]")
    
    # Clear any translation
    vast_instance.set_seg_translation([], [])
    
    # Load with fixed decoder
    segimage = get_seg_image_rle_decoded_fixed(
        vast_instance,
        miplevel=0,
        minx=minx, maxx=maxx,
        miny=miny, maxy=maxy,
        minz=minz, maxz=maxz,
        surfonlyflag=0,
        flipflag=0
    )
    
    if segimage is None:
        print("  ERROR: Failed to load data!")
        return
    
    print(f"\n[3] Analyzing loaded data...")
    print(f"  Array shape: {segimage.shape}")
    print(f"  Array dtype: {segimage.dtype}")
    
    # Check what segment IDs are present
    unique_ids = np.unique(segimage)
    print(f"  Unique segment IDs: {unique_ids}")
    
    # Count voxels for our segment
    if segment_id in unique_ids:
        count = np.sum(segimage == segment_id)
        print(f"  Voxels for segment {segment_id}: {count:,}")
    else:
        print(f"  WARNING: Segment {segment_id} not found in loaded data!")
        print(f"  Present segments: {unique_ids}")
    
    # Now test WITH translation
    print("\n[4] Testing data load WITH translation...")
    vast_instance.set_seg_translation([segment_id], [segment_id])
    
    segimage_translated = get_seg_image_rle_decoded_fixed(
        vast_instance,
        miplevel=0,
        minx=minx, maxx=maxx,
        miny=miny, maxy=maxy,
        minz=minz, maxz=maxz,
        surfonlyflag=0,
        flipflag=0
    )
    
    vast_instance.set_seg_translation([], [])
    
    if segimage_translated is not None:
        unique_ids_trans = np.unique(segimage_translated)
        print(f"  Unique IDs with translation: {unique_ids_trans}")
        count_trans = np.sum(segimage_translated == segment_id)
        print(f"  Voxels for segment {segment_id}: {count_trans:,}")
    
    print("\n" + "="*80)


def extract_segment_with_fix(vast_instance, segment_id: int, output_dir: str = "./output_fixed"):
    """
    Extract a segment using the fixed RLE decoder.
    """
    from pathlib import Path
    from skimage import measure
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EXTRACTING SEGMENT {segment_id} WITH FIXED DECODER")
    print(f"{'='*80}\n")
    
    # Get segment info
    seg_data = vast_instance.get_segment_data(segment_id)
    if not seg_data:
        print(f"ERROR: Segment {segment_id} not found")
        return None, None, None
    
    bbox = seg_data['boundingbox']
    print(f"Segment: {seg_data.get('name', 'UNKNOWN')}")
    print(f"Bounding box: {bbox}")
    
    # Prepare region with padding
    minx = max(0, bbox[0] - 2)
    maxx = bbox[3] + 2
    miny = max(0, bbox[1] - 2)
    maxy = bbox[4] + 2
    minz = max(0, bbox[2] - 2)
    maxz = bbox[5] + 2
    
    print(f"Loading region: X=[{minx},{maxx}] Y=[{miny},{maxy}] Z=[{minz},{maxz}]")
    
    # Set translation to isolate this segment
    vast_instance.set_seg_translation([segment_id], [segment_id])
    
    # Load with fixed decoder
    print("\nLoading segmentation data...")
    segimage = get_seg_image_rle_decoded_fixed(
        vast_instance,
        miplevel=0,
        minx=minx, maxx=maxx,
        miny=miny, maxy=maxy,
        minz=minz, maxz=maxz,
        surfonlyflag=0,
        flipflag=0
    )
    
    # Clear translation
    vast_instance.set_seg_translation([], [])
    
    if segimage is None:
        print("ERROR: Failed to load data")
        return None, None, None
    
    # Create binary volume
    print("\nCreating binary volume...")
    binary_volume = (segimage == segment_id).astype(np.float32)
    voxel_count = np.sum(binary_volume)
    
    print(f"Found {int(voxel_count):,} voxels for segment {segment_id}")
    
    if voxel_count == 0:
        print("ERROR: No voxels found for this segment!")
        return None, None, None
    
    # Add boundary padding
    print("Adding boundary padding...")
    padded = np.zeros((
        binary_volume.shape[0] + 2,
        binary_volume.shape[1] + 2,
        binary_volume.shape[2] + 2
    ), dtype=np.float32)
    padded[1:-1, 1:-1, 1:-1] = binary_volume
    binary_volume = padded
    
    # Run marching cubes
    print("Running marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0.5)
    
    print(f"Generated {len(verts):,} vertices, {len(faces):,} faces")
    
    # Transform coordinates
    print("Transforming coordinates...")
    
    # Undo padding
    verts = verts - 1
    
    # Add region offset
    verts[:, 0] += minx
    verts[:, 1] += miny
    verts[:, 2] += minz
    
    # Get voxel size
    info = vast_instance.get_info()
    voxel_size = np.array([
        info['voxelsizex'],
        info['voxelsizey'],
        info['voxelsizez']
    ])
    
    # Scale to physical coordinates
    verts *= voxel_size
    
    # Convert to micrometers
    verts *= 0.001
    
    # Save OBJ
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    
    obj_path = meshes_dir / f"seg_{segment_id:04d}_FIXED.obj"
    
    print(f"\nSaving to {obj_path}...")
    with open(obj_path, 'w') as f:
        f.write(f"# Fixed extraction for segment {segment_id}\n")
        f.write(f"# Voxels: {int(voxel_count):,}\n")
        f.write(f"# Vertices: {len(verts):,}, Faces: {len(faces):,}\n")
        f.write(f"o Segment_{segment_id}\n\n")
        
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write(f"\ng Segment_{segment_id}\n")
        
        for face in faces:
            f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"  Voxels: {int(voxel_count):,}")
    print(f"  Vertices: {len(verts):,}")
    print(f"  Faces: {len(faces):,}")
    print(f"  Output: {obj_path}")
    print(f"{'='*80}\n")
    
    return verts, faces, str(obj_path)


if __name__ == "__main__":
    from VastControlClass_exporting import VASTControlClass
    
    print("VAST RLE Decoder Fix - Diagnostic Tool")
    print("="*80)
    
    # Connect to VAST
    vast = VASTControlClass()
    print("\nConnecting to VAST...")
    if not vast.connect("127.0.0.1", 22081, timeout=10):
        print("ERROR: Could not connect to VAST")
        exit(1)
    
    print("Connected!\n")
    
    segment_id = 1  # Your sphere/saucer segment
    
    try:
        # Run diagnostics
        diagnose_segment_data(vast, segment_id)
        
        # Extract with fixed decoder
        input("\nPress Enter to extract with fixed decoder...")
        vertices, faces, output_path = extract_segment_with_fix(vast, segment_id)
        
    finally:
        vast.disconnect()
        print("\nDisconnected from VAST")