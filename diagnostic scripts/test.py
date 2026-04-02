import sys
import logging
from pathlib import Path


from vastpy.control.exporting import VASTControlClass
from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner
from neuron_pipeline.stages.skeletonization import SkeletonExtractor, SWCWriter

# Configuration vars
SEGMENT_ID   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MIPLEVEL     = 1        # 0 = full resolution, 1 = half resolution
PADDING      = 2        # extra voxels around bounding box
SPUR_UM      = 0.5      # prune leaf branches shorter than this (micrometers)
OUTPUT_DIR   = Path(__file__).parent.parent / "diag_output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("diag")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    swc_dir = OUTPUT_DIR / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Connect to VAST
    log.info("Connecting to VAST...")
    vast = VASTControlClass()
    if not vast.connect(host="127.0.0.1", port=22081, timeout=100):
        log.error("Failed to connect to VAST. Ensure VAST is running and API is enabled.")
        return
    log.info("Connected.")

    extractor = SegmentSurfaceExtractor(vast, str(OUTPUT_DIR))
    cleaner   = VoxelCleaner()
    skel      = SkeletonExtractor()
    writer    = SWCWriter()

    name = str(SEGMENT_ID)
    log.info(f"--- segment {SEGMENT_ID} ---")

    # 1A. Voxel extraction
    log.info(f"Extracting voxel mask for segment {SEGMENT_ID} (MIP {MIPLEVEL})...")
    mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
        segment_id=SEGMENT_ID,
        miplevel=MIPLEVEL,
        padding=PADDING,
    )

    if mask is None or voxel_size is None or bbox_min is None:
        log.error(f"Voxel extraction failed for segment {SEGMENT_ID}")
        return

    # 1B. Cleaning -- first pass to count components
    cleaned, stats = cleaner.clean_mask(
        mask,
        closing_radius_um=2,
        keep_largest_only=True,
        fill_holes=True,
        smooth_iterations=0,
        voxel_size_um=voxel_size,
    )

    if stats['original_components'] > 1:
        log.warning(
            f"  {name}: {stats['original_components']} connected components "
            f"(sizes: kept={stats['kept_components']}, removed={stats['removed_components']})"
        )

        # Re-clean keeping only the largest component
        cleaned, stats = cleaner.clean_mask(
            cleaned.mask,
            closing_radius_um=2,
            keep_largest_only=True,
            fill_holes=False,
            smooth_iterations=0,
            voxel_size_um=voxel_size,
        )

    # 1C. Skeletonize voxels (includes compression, pruning, soma insertion)
    tree, skel_stats = skel.extract_skeleton(
        mask=cleaned.mask,
        voxel_size_um=voxel_size,
        bbox_min_vox=cleaned.bbox_min_vox,
        spur_length_um=SPUR_UM,
    )
    if tree.number_of_nodes() == 0:
        log.error(f"Skeletonization produced empty tree for segment {SEGMENT_ID}")
        return

    # 1D. Write SWC
    swc_path = str(swc_dir / f"cell_{name}.swc")
    writer.write_swc(
        tree=tree,
        output_path=swc_path,
        metadata={
            'segment_id': SEGMENT_ID,
            'segment_name': name,
            'miplevel': MIPLEVEL,
            'bbox_min_vox': str(bbox_min),      # (minx, miny, minz) in voxels
            'voxel_size_um': str(voxel_size),   # (sx, sy, sz) in µm
            'coord_frame': 'physical_um_xyz',
            'compressed_nodes': skel_stats.get('compressed_nodes'),
            'spurs_pruned': skel_stats.get('branches_pruned'),
            'total_length_um': f"{skel_stats.get('total_length_um', 0):.2f}",
        },
    )

    log.info(f"-> {swc_path}  ({tree.number_of_nodes()} nodes)")


if __name__ == "__main__":
    main()
