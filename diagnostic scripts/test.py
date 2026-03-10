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

    # 1. Connect to VAST
    log.info("Connecting to VAST...")
    vast = VASTControlClass()
    vast.connect(host="127.0.0.1", port=22081, timeout=100)
    log.info("Connected.")

    # 2. Extract voxel mask
    log.info(f"Extracting voxel mask for segment {SEGMENT_ID} (MIP {MIPLEVEL})...")
    extractor = SegmentSurfaceExtractor(vast, str(OUTPUT_DIR))
    mask, bbox_min, voxel_size, _ = extractor.extract_segment(
        segment_id=SEGMENT_ID,
        miplevel=MIPLEVEL,
        padding=PADDING,
    )

    if mask is None or voxel_size is None or bbox_min is None:
        log.error("Voxel extraction failed. Is the segment ID valid?")
        return

    log.info(
        f"Mask shape: {mask.shape}  |  voxel size: {voxel_size} um  "
        f"|  bbox_min: {bbox_min}  |  filled voxels: {int(mask.sum()):,}"
    )

    # 3. Clean mask
    log.info("Cleaning mask...")
    cleaner = VoxelCleaner()
    cleaned, clean_stats = cleaner.clean_mask(
        mask,
        keep_largest_only=True,
        fill_holes=True,
        smooth_iterations=0,
    )
    log.info(f"Clean stats: {clean_stats}")

    # 4. Skeletonize
    log.info("Skeletonizing...")
    skel = SkeletonExtractor()
    tree, skel_stats = skel.extract_skeleton(
        mask=cleaned,
        voxel_size_um=voxel_size,
        bbox_min_vox=bbox_min,
        spur_length_um=SPUR_UM,
    )

    if tree.number_of_nodes() == 0:
        log.error("Skeletonization produced an empty tree.")
        return

    log.info(f"Skeleton stats: {skel_stats}")

    # 5. Write SWC
    swc_path = OUTPUT_DIR / f"seg_{SEGMENT_ID}_mip{MIPLEVEL}.swc"
    writer = SWCWriter()
    writer.write_swc(
        tree=tree,
        output_path=str(swc_path),
        metadata={
            "segment_id":   SEGMENT_ID,
            "miplevel":     MIPLEVEL,
            "voxel_size_um": voxel_size,
            "bbox_min_vox": bbox_min,
            **{k: v for k, v in skel_stats.items()},
        },
    )
    log.info(f"Done. SWC written -> {swc_path}")


if __name__ == "__main__":
    main()
