from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
import sys
from vastpy.control.exporting import VASTControlClass


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
        
        else: # Surface extraction to .obj files
            # Create extractor
            extractor = SegmentSurfaceExtractor(vast, output_dir="./vast_export")
            
            num_segments = vast.get_number_of_segments()
            if num_segments is None:
                print("ERROR: Could not retrieve number of segments")
                return
            for segment_id in range(1, num_segments):
                
                print(f"\nExtracting segment {segment_id}...")
                print("This may take several minutes for large neurons...\n")
                
                vertices, faces, output_path = extractor.extract_segment(
                    segment_id=segment_id,
                    miplevel=1,  # half resolution
                    close_surfaces=True,
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