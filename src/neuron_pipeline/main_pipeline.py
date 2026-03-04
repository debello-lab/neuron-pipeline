"""
Author: Nicolas Randazzo
Entry point for the VASTpyAPI pipeline.

This script is used for the Debello Lab at UC Davis Neuroscience to analyze the auditory cortex of barn owls.
In order to run this script, a VAST instance with a proper segmentation file loaded must be running.

Pipeline phases:
  Phase 0  Segment classification & connectivity table
  Phase 1  Voxel extraction -> cleaning -> skeletonization -> SWC
  Phase 2  Synapse / bouton centroid extraction        
  Phase 3  Map centroids to SWC cable locations        (TODO)
  Phase 4  Connectivity CSV + Arbor recipe generation  (TODO)
"""
from pathlib import Path
from typing import Dict, Tuple
import logging

import networkx as nx

from vastpy.control.exporting import VASTControlClass

from neuron_pipeline.stages.segment_classifier import SegmentClassifier, SegmentRegistry
from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner
from neuron_pipeline.stages.skeletonization import SkeletonExtractor, SWCWriter
from neuron_pipeline.stages.centroid_extraction import CentroidExtractor, CentroidTable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


#  Phase 1 

def run_phase1(
    registry: SegmentRegistry,
    vast: VASTControlClass,
    output_dir: str = "./vast_export",
    miplevel: int = 1,
    padding: int = 1,
    spur_length_um: float = 2.0,
) -> Dict[str, Tuple[nx.DiGraph, str]]:
    """
    Phase 1: Extract and skeletonize every AXON and POST_SYN segment.

    Args:
        registry: SegmentRegistry from Phase 0
        vast: Connected VASTControlClass instance
        output_dir: Base output directory
        miplevel: MIP level for voxel extraction (0=full, 1=half)
        padding: Padding voxels around bounding box
        spur_length_um: Prune leaf branches shorter than this

    Returns:
        Dict mapping segment name -> (tree, swc_path).
        Trees are kept in memory for Phase 3 (centroid -> SWC mapping).
    """
    extractor = SegmentSurfaceExtractor(vast, output_dir)
    cleaner = VoxelCleaner()
    skel = SkeletonExtractor()
    writer = SWCWriter()

    swc_dir = Path(output_dir) / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (name, info)
        for name, info in registry.segments.items()
        if info.role in ('AXON', 'POST_SYN')
    ]
    logger.info(f"Phase 1: {len(targets)} segments to skeletonize (AXON + POST_SYN)")

    results: Dict[str, Tuple[nx.DiGraph, str]] = {}

    for name, info in targets:
        logger.info(f"--- {info.role} {name} (seg {info.seg_id}) ---")

        # 1A. Voxel extraction
        mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
            segment_id=info.seg_id,
            miplevel=miplevel,
            padding=padding,
        )
        if mask is None:
            logger.error(f"  Voxel extraction failed -- skipping {name}")
            continue

        # 1B. Cleaning -- first pass to count components
        cleaned, stats = cleaner.clean_mask(
            mask, keep_largest_only=False, fill_holes=True, smooth_iterations=0
        )
        if stats['original_components'] > 1:
            logger.warning(
                f"  {name}: {stats['original_components']} connected components "
                f"(sizes: kept={stats['kept_components']}, removed={stats['removed_components']})"
            )
            # Re-clean keeping only the largest component
            cleaned, stats = cleaner.clean_mask(
                mask, keep_largest_only=True, fill_holes=True, smooth_iterations=0
            )

        # 1C. Skeletonize (includes compression, pruning, soma insertion)
        tree, skel_stats = skel.extract_skeleton(
            mask=cleaned,
            voxel_size_um=voxel_size,
            bbox_min_vox=bbox_min,
            spur_length_um=spur_length_um,
        )
        if tree.number_of_nodes() == 0:
            logger.error(f"  Skeletonization produced empty tree -- skipping {name}")
            continue

        # 1D. Write SWC
        swc_path = str(swc_dir / f"cell_{name}.swc")
        writer.write_swc(
            tree=tree,
            output_path=swc_path,
            metadata={
                'segment_id': info.seg_id,
                'segment_name': name,
                'role': info.role,
                'miplevel': miplevel,
                'compressed_nodes': skel_stats.get('compressed_nodes'),
                'spurs_pruned': skel_stats.get('branches_pruned'),
                'total_length_um': f"{skel_stats.get('total_length_um', 0):.2f}",
            },
        )

        results[name] = (tree, swc_path)
        logger.info(f"  -> {swc_path}  ({tree.number_of_nodes()} nodes)")

    logger.info(f"Phase 1 complete: {len(results)}/{len(targets)} SWC files written")
    return results





def run_phase2(
    registry: SegmentRegistry,
    vast: VASTControlClass,
    output_dir: str = "./vast_export",
    miplevel: int = 0,
) -> CentroidTable:
    """
    Phase 2: Extract centroid positions for all BOUTON, SYNAPSE, and CONTACT
    segments.

    Centroids are computed from the full voxel mask at MIP level *miplevel*
    (default 0 = full resolution, appropriate for compact annotation objects).
    On extraction failure the bounding-box centre is used, with the VAST
    anchor point as the final fallback.

    Args:
        registry   : SegmentRegistry from Phase 0.
        vast       : Connected VASTControlClass instance.
        output_dir : Base output directory; centroids.csv is written here.
        miplevel   : MIP level for voxel extraction (0 = full resolution).

    Returns:
        CentroidTable — in-memory table, also written to
        <output_dir>/centroids.csv for Phase 3 consumption.
    """
    surface_extractor = SegmentSurfaceExtractor(vast, output_dir)
    centroid_extractor = CentroidExtractor(surface_extractor)

    table = centroid_extractor.extract_centroids(
        registry=registry,
        miplevel=miplevel
    )

    csv_path = str(Path(output_dir) / "centroids.csv")
    table.write_csv(csv_path)

    logger.info(f"Phase 2: centroids written -> {csv_path} ({len(table)} entries)")

    # Summary by role
    for role in ('BOUTON', 'SYNAPSE', 'CONTACT'):
        entries = table.by_role(role)
        if entries:
            methods = {}
            for e in entries:
                methods[e.method] = methods.get(e.method, 0) + 1
            method_str = ", ".join(f"{m}={n}" for m, n in sorted(methods.items()))
            logger.info(f"  {role}: {len(entries)} entries ({method_str})")

    return table


# Main entry point 

def main():
    # Phase 0: Classify segments
    classifier = SegmentClassifier()
    registry = classifier.classify_segments()

    # Phase 1: Skeletonize AXON and POST_SYN
    trees = run_phase1(
        registry=registry,
        vast=classifier.vast,
        output_dir="./vast_export",
        miplevel=1,
        spur_length_um=2.0,
    )

    # Phase 2: extract BOUTON / SYNAPSE / CONTACT centroids
    centroids = run_phase2(
        registry=registry,
        vast=classifier.vast,
        output_dir="./vast_export",
        miplevel=0,  # MIP 0 for small markers — MIP 1 may reduce to too few voxels
    )
    print(centroids)

    # Phase 3: map centroids to SWCs
    # trees dict will be passed to Phase 3 for centroid -> SWC mapping


# if __name__ == "__main__":
#     main()
