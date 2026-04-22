from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
import sys
from vastpy.control.exporting import VASTControlClass


def extract_surface(extractor, segment_id):
    print(f"\nExtracting segment {segment_id}...")
    print("This may take several minutes for large neurons...\n")
    vertices, faces, output_path = extractor.extract_segment(
        segment_id=segment_id,
        miplevel=0,
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


def main():
    print("=" * 80)
    print("VAST Surface Extraction")
    print("=" * 80)

    args = sys.argv[1:]

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
        extractor = SegmentSurfaceExtractor(vast, output_dir="./vast_export")

        if len(args) == 1 and args[0] == "skeleton":
            # Voxel extraction for segment 1
            segment_id = 1
            print(f"\nExtracting voxel mask for segment {segment_id}...")
            print("This may take several minutes for large neurons...\n")
            extractor.extract_segment_voxel(
                segment_id=segment_id,
                miplevel=1,
                padding=1,
                output_filename=f"segment_{segment_id:04d}_mask.npy"
            )

        elif len(args) == 1:
            # Single segment id
            segment_id = int(args[0])
            extract_surface(extractor, segment_id)

        elif len(args) == 2:
            # Range of segment ids (inclusive)
            start_id = int(args[0])
            end_id = int(args[1])
            for segment_id in range(start_id, end_id + 1):
                extract_surface(extractor, segment_id)

        else:
            # No arguments -- extract all segments
            num_segments = vast.get_number_of_segments()
            if num_segments is None:
                print("ERROR: Could not retrieve number of segments")
                return
            for segment_id in range(1, num_segments + 1):
                extract_surface(extractor, segment_id)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        vast.disconnect()
        print("\nDisconnected from VAST")


if __name__ == "__main__":
    main()