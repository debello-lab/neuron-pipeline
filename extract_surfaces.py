"""
Surface Extraction from VAST Segmentation Data.

This module provides a simplified interface for extracting 3D surface meshes
from segmentation data in VAST using marching cubes.
"""
import numpy as np
from skimage import measure
from typing import Optional, Tuple
import os
from VastControlClass_exporting import VASTControlClass


def extract_single_segment(
    vast: VASTControlClass,
    segment_id: int,
    miplevel: int = 0,
    output_path: Optional[str] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    close_surfaces: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a surface mesh for a single segment.

    Args:
        vast: Connected VASTControlClass instance
        segment_id: The segment ID to extract
        miplevel: MIP level to use (0 = full resolution)
        output_path: Optional path to save OBJ file
        voxel_size: Optional (x, y, z) voxel size for scaling. If None, uses VAST info.
        close_surfaces: If True, close the mesh at volume boundaries

    Returns:
        Tuple of (vertices, faces) numpy arrays
        vertices shape: (N, 3) - XYZ coordinates
        faces shape: (M, 3) - triangle indices (0-indexed)
    """
    # Get segment data to find bounding box
    seg_data = vast.get_segment_data(segment_id)
    if seg_data is None:
        raise ValueError(f"Segment {segment_id} not found")

    bbox = seg_data['boundingbox']  # [x1, y1, z1, x2, y2, z2]

    # Check for valid bounding box
    if bbox[0] < 0 or bbox[3] < 0:
        raise ValueError(f"Segment {segment_id} has no valid bounding box - may be empty")

    # Get dataset info for scaling
    info = vast.get_info()
    if info is None:
        raise RuntimeError("Could not get dataset info from VAST")

    # Get voxel size
    if voxel_size is None:
        voxel_size = (info['voxelsizex'], info['voxelsizey'], info['voxelsizez'])

    # Get MIP scale factors
    mip_scale = [1, 1, 1]
    if miplevel > 0:
        # Get the selected segment layer
        layer_info = vast.get_selected_layernr()
        if layer_info:
            seg_layer = layer_info[2]  # selected_segment_layer
            mip_factors = vast.get_mipmap_scale_factors(seg_layer)
            if mip_factors is not None and miplevel <= len(mip_factors):
                mip_scale = mip_factors[miplevel - 1]  # 0-indexed after first row removed

    # Adjust bounding box for MIP level
    minx = bbox[0] >> miplevel
    maxx = (bbox[3] >> miplevel)
    miny = bbox[1] >> miplevel
    maxy = (bbox[4] >> miplevel)
    minz = bbox[2]
    maxz = bbox[5]

    # Apply Z scaling if needed
    if miplevel > 0 and mip_scale[2] != 1:
        minz = minz // mip_scale[2]
        maxz = maxz // mip_scale[2]

    # Add 1 voxel padding for marching cubes
    minx = max(0, minx - 1)
    miny = max(0, miny - 1)
    minz = max(0, minz - 1)
    maxx = maxx + 1
    maxy = maxy + 1
    maxz = maxz + 1

    print(f"Loading segment {segment_id} at mip {miplevel}")
    print(f"Region: X={minx}-{maxx}, Y={miny}-{maxy}, Z={minz}-{maxz}")

    # Set translation so only this segment is visible
    vast.set_seg_translation([segment_id], [segment_id])

    # Get the segmentation data
    seg_image = vast.get_seg_image_rle_decoded(
        miplevel, minx, maxx, miny, maxy, minz, maxz,
        surfonlyflag=0, flipflag=0
    )

    # Clear translation
    vast.set_seg_translation([], [])

    if seg_image is None:
        raise RuntimeError("Failed to load segmentation data")

    print(f"Loaded volume: {seg_image.shape}")

    # Create binary mask for this segment
    binary_volume = (seg_image == segment_id).astype(np.float32)

    # Check if segment exists in the loaded region
    voxel_count = np.sum(binary_volume)
    if voxel_count == 0:
        raise ValueError(f"Segment {segment_id} not found in loaded region")

    print(f"Found {int(voxel_count)} voxels for segment {segment_id}")

    # Add boundary padding if closing surfaces
    if close_surfaces:
        # Pad with zeros on all sides
        padded = np.zeros((
            binary_volume.shape[0] + 2,
            binary_volume.shape[1] + 2,
            binary_volume.shape[2] + 2
        ), dtype=np.float32)
        padded[1:-1, 1:-1, 1:-1] = binary_volume
        binary_volume = padded
        # Adjust offset for padding
        offset_adjust = np.array([-1, -1, -1])
    else:
        offset_adjust = np.array([0, 0, 0])

    # Run marching cubes
    print("Running marching cubes...")
    try:
        verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0.5)
    except Exception as e:
        raise RuntimeError(f"Marching cubes failed: {e}")

    print(f"Extracted {len(verts)} vertices, {len(faces)} faces")

    # Apply coordinate transformations
    # Adjust for padding offset
    verts = verts + offset_adjust

    # Add region offset
    verts[:, 0] += minx
    verts[:, 1] += miny
    verts[:, 2] += minz

    # Scale to physical coordinates
    verts[:, 0] *= voxel_size[0] * mip_scale[0]
    verts[:, 1] *= voxel_size[1] * mip_scale[1]
    verts[:, 2] *= voxel_size[2] * mip_scale[2]

    # Convert to micrometers if voxel size is in nm
    if voxel_size[0] > 1:  # Likely nanometers
        verts *= 0.001  # Convert to micrometers

    # Save to OBJ if path provided
    if output_path:
        write_obj(output_path, verts, faces, f"Segment_{segment_id}")
        print(f"Saved mesh to {output_path}")

    return verts, faces


def extract_selected_segment(
    vast: VASTControlClass,
    miplevel: int = 0,
    output_path: Optional[str] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract surface mesh for the currently selected segment in VAST.

    Args:
        vast: Connected VASTControlClass instance
        miplevel: MIP level to use (0 = full resolution)
        output_path: Optional path to save OBJ file
        **kwargs: Additional arguments passed to extract_single_segment

    Returns:
        Tuple of (vertices, faces) numpy arrays
    """
    segment_id = vast.get_selected_segment_nr()
    if segment_id is None or segment_id < 1:
        raise ValueError("No segment selected in VAST")

    return extract_single_segment(vast, segment_id, miplevel, output_path, **kwargs)


def write_obj(filepath: str, vertices: np.ndarray, faces: np.ndarray, object_name: str = "mesh"):
    """
    Write mesh to OBJ file format.

    Args:
        filepath: Output file path
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices (0-indexed)
        object_name: Name for the object in OBJ file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"# VAST Surface Export\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n")
        f.write(f"o {object_name}\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"g {object_name}\n")

        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")


def write_obj_with_mtl(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int],
    object_name: str = "mesh"
):
    """
    Write mesh to OBJ file with accompanying MTL material file.

    Args:
        filepath: Output file path for OBJ
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices (0-indexed)
        color: RGB tuple (0-255) for material color
        object_name: Name for the object
    """
    base_path = os.path.splitext(filepath)[0]
    mtl_path = base_path + ".mtl"
    mtl_filename = os.path.basename(mtl_path)
    material_name = f"mat_{object_name}"

    # Write OBJ file
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"# VAST Surface Export\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n")
        f.write(f"mtllib {mtl_filename}\n")
        f.write(f"usemtl {material_name}\n")
        f.write(f"o {object_name}\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"g {object_name}\n")

        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")

    # Write MTL file
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {r:.3f} {g:.3f} {b:.3f}\n")
        f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")
        f.write(f"Ks 0.5 0.5 0.5\n")
        f.write(f"d 1.0\n")
        f.write(f"Ns 32\n")


def main():
    """Example usage of surface extraction."""
    # Connect to VAST
    vast = VASTControlClass()

    print("Connecting to VAST...")
    if not vast.connect("127.0.0.1", 22081, timeout=10):
        print("Failed to connect to VAST. Make sure VAST is running and API is enabled.")
        return

    print("Connected successfully!")

    try:
        # Get dataset info
        info = vast.get_info()
        if not info:
            print("No dataset loaded in VAST")
            return

        print(f"Dataset size: {info['datasizex']}x{info['datasizey']}x{info['datasizez']}")
        print(f"Voxel size: {info['voxelsizex']}x{info['voxelsizey']}x{info['voxelsizez']} nm")

        # Get selected segment
        selected_seg = vast.get_selected_segment_nr()
        if selected_seg is None or selected_seg < 1:
            print("No segment selected. Please select a segment in VAST.")
            # Try to find any segment with data
            num_segments = vast.get_number_of_segments()
            if num_segments and num_segments > 1:
                # Try segment 1 (0 is background)
                selected_seg = 1
                print(f"Using segment {selected_seg} as fallback")
            else:
                print("No segments available")
                return
        else:
            print(f"Selected segment: {selected_seg}")

        # Create output directory
        output_dir = "./vast_export"
        os.makedirs(output_dir, exist_ok=True)

        # Extract surface
        output_file = os.path.join(output_dir, f"segment_{selected_seg}.obj")

        print(f"\nExtracting surface for segment {selected_seg}...")
        vertices, faces = extract_single_segment(
            vast,
            segment_id=selected_seg,
            miplevel=0,  # Full resolution
            output_path=output_file,
            close_surfaces=True
        )

        print(f"\nExtraction complete!")
        print(f"Vertices: {len(vertices)}")
        print(f"Faces: {len(faces)}")
        print(f"Output: {output_file}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

    finally:
        vast.disconnect()
        print("\nDisconnected from VAST")


if __name__ == "__main__":
    main()
