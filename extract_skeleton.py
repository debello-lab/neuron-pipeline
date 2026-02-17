"""


Author: Nicolas Randazzo
Date: 2026-02-11
"""

import sys
import numpy as np
from pathlib import Path

from VastControlClass_exporting import VASTControlClass
from extract_surfaces import SegmentSurfaceExtractor
from voxel_cleaning import VoxelCleaner
from skeletonization import SkeletonExtractor, SWCWriter


def extract_and_skeletonize_segment(
    segment_id: int,
    miplevel: int = 1,
    output_dir: str = "./vast_export"
):
    """
    Complete pipeline: VAST segment -> cleaned voxels -> skeleton -> SWC.
    
    Args:
        segment_id: Segment ID to process
        miplevel: MIP level (0=full resolution, 1=half, etc.)
        output_dir: Output directory
    """
    print("=" * 80)
    print(f"VOXEL -> SKELETON -> SWC PIPELINE")
    print(f"Segment {segment_id} at MIP level {miplevel}")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Connect to VAST and extract voxel mask
    # ========================================================================
    print("\n[1/5] Connecting to VAST and extracting voxel mask...")
    
    vast = VASTControlClass()
    if not vast.connect("127.0.0.1", 22081, timeout=10):
        print("ERROR: Failed to connect to VAST")
        return
    
    try:
        extractor = SegmentSurfaceExtractor(vast, output_dir=output_dir)
        
        # Extract raw voxel mask
        mask, bbox_min_vox, voxel_size_um, mask_path = extractor.extract_segment_voxel(
            segment_id=segment_id,
            miplevel=miplevel,
            padding=1,
            output_filename=f"segment_{segment_id:04d}_raw_mask.npz"
        )
        
        if mask is None:
            print("ERROR: Voxel extraction failed")
            return
        
        print(f" Extracted mask: {mask.shape}")
        print(f"  Voxel count: {np.sum(mask):,}")
        print(f"  Voxel size: {voxel_size_um} um")
        print(f"  Origin: {bbox_min_vox} voxels")
        
        # ====================================================================
        # Step 2: Clean the mask
        # ====================================================================
        print("\n[2/5] Cleaning voxel mask...")
        
        cleaner = VoxelCleaner(logger=extractor.logger)
        cleaned_mask, clean_stats = cleaner.clean_mask(
            mask,
            keep_largest_only=True,
            fill_holes=True,
            smooth_iterations=0  # No smoothing for now
        )
        
        print(f" Cleaning complete:")
        print(f"  Components: {clean_stats['original_components']} -> {clean_stats['kept_components']}")
        print(f"  Voxels: {clean_stats['original_voxel_count']:,} -> {clean_stats['final_voxel_count']:,}")
        print(f"  Removed: {clean_stats['removed_components']} components, "
              f"{clean_stats['voxels_removed']:,} voxels")
        
        # Save cleaned mask
        cleaned_mask_path = Path(output_dir) / "voxels" / f"segment_{segment_id:04d}_cleaned_mask.npz"
        np.savez_compressed(
            cleaned_mask_path,
            mask=cleaned_mask,
            bbox_min_vox=bbox_min_vox,
            voxel_size_um=voxel_size_um,
            segment_id=segment_id,
            clean_stats=clean_stats
        )
        print(f"  Saved: {cleaned_mask_path}")
        
        # ====================================================================
        # Step 3: Skeletonize
        # ====================================================================
        print("\n[3/5] Extracting skeleton...")
        
        skeleton_extractor = SkeletonExtractor(logger=extractor.logger)
        skeleton_points, radii_um, skel_stats = skeleton_extractor.extract_skeleton(
            cleaned_mask,
            voxel_size_um,
            bbox_min_vox,
            spur_length_um=2.0  # Prune branches < 2 um
        )
        
        if len(skeleton_points) == 0:
            print("ERROR: Skeleton extraction failed")
            return
        
        print(f" Skeleton extracted:")
        print(f"  Skeleton voxels: {skel_stats['skeleton_voxel_count']:,}")
        print(f"  Skeleton points: {skel_stats['skeleton_points']:,}")
        print(f"  Radius range: {radii_um.min():.3f} - {radii_um.max():.3f} um")
        print(f"  Median radius: {np.median(radii_um):.3f} um")
        
        # ====================================================================
        # Step 4: Build skeleton graph (optional - for advanced processing)
        # ====================================================================
        print("\n[4/5] Building skeleton graph...")
        
        # First get the skeleton mask from the cleaned mask
        from skimage.morphology import skeletonize
        skeleton_mask = skeletonize(cleaned_mask.astype(np.uint8)) > 0
        
        # Build graph
        graph = skeleton_extractor.skeleton_to_graph(skeleton_mask, voxel_size_um)
        
        print(f" Graph built:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        # Analyze graph structure
        endpoints = [n for n in graph.nodes() if graph.nodes[n]['degree'] == 1]
        branchpoints = [n for n in graph.nodes() if graph.nodes[n]['degree'] >= 3]
        
        print(f"  Endpoints: {len(endpoints)}")
        print(f"  Branch points: {len(branchpoints)}")
        
        # Convert to tree
        tree = skeleton_extractor.graph_to_tree(graph)
        
        print(f"  Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")
        
        # ====================================================================
        # Step 5: Export to SWC
        # ====================================================================
        print("\n[5/5] Writing SWC file...")
        
        swc_dir = Path(output_dir) / "swc"
        swc_dir.mkdir(parents=True, exist_ok=True)
        swc_path = swc_dir / f"segment_{segment_id:04d}.swc"
        
        writer = SWCWriter(logger=extractor.logger)
        
        metadata = {
            'segment_id': segment_id,
            'segment_name': f'Segment_{segment_id}',
            'mip_level': miplevel,
            'voxel_size_um': voxel_size_um,
            'clean_stats': clean_stats,
            'skel_stats': skel_stats
        }
        
        writer.write_swc(
            skeleton_points,
            radii_um,
            str(swc_path),
            tree=tree,
            compartment_type=0,  # 0 = undefined
            metadata=metadata
        )
        
        print(f" SWC file written: {swc_path}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"Input: Segment {segment_id} from VAST")
        print(f"Output files:")
        print(f"  - Raw mask:     {mask_path}")
        print(f"  - Cleaned mask: {cleaned_mask_path}")
        print(f"  - SWC skeleton: {swc_path}")
        print()
        print(f"Statistics:")
        print(f"  - Original voxels: {clean_stats['original_voxel_count']:,}")
        print(f"  - Cleaned voxels:  {clean_stats['final_voxel_count']:,}")
        print(f"  - Skeleton voxels: {skel_stats['skeleton_voxel_count']:,}")
        print(f"  - Skeleton points: {skel_stats['skeleton_points']:,}")
        print(f"  - Graph endpoints: {len(endpoints)}")
        print(f"  - Branch points:   {len(branchpoints)}")
        print("=" * 80)
        
    finally:
        vast.disconnect()
        print("\nDisconnected from VAST")


def batch_process_segments(
    segment_ids: list,
    miplevel: int = 1,
    output_dir: str = "./vast_export"
):
    """
    Process multiple segments through the pipeline.
    
    Args:
        segment_ids: List of segment IDs to process
        miplevel: MIP level
        output_dir: Output directory
    """
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING: {len(segment_ids)} segments")
    print(f"{'=' * 80}\n")
    
    for i, seg_id in enumerate(segment_ids, 1):
        print(f"\n{'*' * 80}")
        print(f"Processing segment {i}/{len(segment_ids)}: ID {seg_id}")
        print(f"{'*' * 80}\n")
        
        try:
            extract_and_skeletonize_segment(seg_id, miplevel, output_dir)
        except Exception as e:
            print(f"\nERROR processing segment {seg_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}")
    print(f"BATCH COMPLETE: Processed {len(segment_ids)} segments")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Example 1: Single segment
    if len(sys.argv) > 1:
        segment_id = int(sys.argv[1])
        miplevel = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        extract_and_skeletonize_segment(segment_id, miplevel)
    else:
        # Default: process segment 1
        print("Usage: python pipeline_example.py <segment_id> [mip_level]")
        print("\nRunning with default: segment 1, MIP level 1\n")
        extract_and_skeletonize_segment(segment_id=1, miplevel=1)
    
    # Example 2: Batch processing (commented out)
    # batch_process_segments([1, 2, 3, 4, 5], miplevel=1)