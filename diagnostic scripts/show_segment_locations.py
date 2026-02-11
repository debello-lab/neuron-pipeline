"""
Quick script to show where segments are located in the VAST dataset.
This will help you choose good test coordinates.
"""
from VASTControlClass import VASTControlClass
import numpy as np

def show_segment_locations():
    print("=" * 60)
    print("Finding Segment Locations")
    print("=" * 60)

    # Connect to VAST
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect to VAST")
        return

    print("Connected to VAST\n")

    # Get dataset info
    info = vast.get_info()
    print(f"Dataset size: {info['datasizex']}x{info['datasizey']}x{info['datasizez']}")

    # Get all segment data
    segment_matrix, success = vast.get_all_segment_data_matrix()
    if not success or len(segment_matrix) == 0:
        print("Failed to get segment data")
        vast.disconnect()
        return

    # Get segment names
    names = vast.get_all_segment_names()
    if names and len(names) > len(segment_matrix):
        names = names[1:]  # Remove background

    print(f"\nFound {len(segment_matrix)} segments (excluding background)\n")
    print("=" * 60)
    print("Segment Bounding Boxes (at full resolution):")
    print("=" * 60)

    # Columns 18-23 are bounding box: minx, maxx, miny, maxy, minz, maxz
    for i, row in enumerate(segment_matrix[:10]):  # Show first 10
        seg_id = int(row[0])
        name = names[i] if names and i < len(names) else f"Segment_{seg_id}"

        minx, maxx = int(row[18]), int(row[21])
        miny, maxy = int(row[19]), int(row[22])
        minz, maxz = int(row[20]), int(row[23])

        print(f"\n{i+1}. {name} (ID={seg_id}):")
        print(f"   X: [{minx:5d} - {maxx:5d}]  (width: {maxx-minx:5d})")
        print(f"   Y: [{miny:5d} - {maxy:5d}]  (width: {maxy-miny:5d})")
        print(f"   Z: [{minz:5d} - {maxz:5d}]  (slices: {maxz-minz:5d})")

    if len(segment_matrix) > 10:
        print(f"\n... and {len(segment_matrix) - 10} more segments")

    # Calculate overall bounding box
    all_minx = np.min(segment_matrix[:, 18])
    all_maxx = np.max(segment_matrix[:, 19])
    all_miny = np.min(segment_matrix[:, 20])
    all_maxy = np.max(segment_matrix[:, 21])
    all_minz = np.min(segment_matrix[:, 22])
    all_maxz = np.max(segment_matrix[:, 23])

    print("\n" + "=" * 60)
    print("Overall Bounding Box (all segments):")
    print("=" * 60)
    print(f"X: [{int(all_minx):5d} - {int(all_maxx):5d}]")
    print(f"Y: [{int(all_miny):5d} - {int(all_maxy):5d}]")
    print(f"Z: [{int(all_minz):5d} - {int(all_maxz):5d}]")

    # Suggest test region
    center_x = int((all_minx + all_maxx) / 2)
    center_y = int((all_miny + all_maxy) / 2)
    center_z = int((all_minz + all_maxz) / 2)

    print("\n" + "=" * 60)
    print("Suggested Test Region (512x512x20 centered on segments):")
    print("=" * 60)
    print("region_params = {")
    print(f"    'xmin': {max(0, center_x - 256)},")
    print(f"    'xmax': {min(int(info['datasizex']), center_x + 256)},")
    print(f"    'ymin': {max(0, center_y - 256)},")
    print(f"    'ymax': {min(int(info['datasizey']), center_y + 256)},")
    print(f"    'zmin': {max(0, center_z - 10)},")
    print(f"    'zmax': {min(int(info['datasizez']), center_z + 10)},")
    print("}")

    vast.disconnect()
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    show_segment_locations()
