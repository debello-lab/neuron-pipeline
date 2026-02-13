"""
Author: Nicolas Randazzo
Date: [When finished]
Entry point for the VASTpyAPI pipeline.

This script is used for the Debello Lab at UC Davis Neuroscience to analyze the auditory cortex of barn owls.
In order to run this script, a VAST instance with a proper segmentation file loaded must be running.

The order of operations is as follows:
1. Extract surfaces from the segmentation into .obj file format.
2. Clean meshes.
3. Skeletonize each neuron and retain connectivity information.
4. Check for errors in the skeletonization process.
5. Perform analysis on the skeletonized neurons.
6. Save the results to a file.
7. Use NEURD to do further analysis on the meshed/skeletonized neurons.
8. Pass skeletonized neurons to Arbor for simulation.

"""

def main():
    # Step 1: Extract surfaces. 
    # Decide whether the extraction of surfaces is for meshes, or for further skeletonization
    # Using SegmentSurfaceExtractor class

    # 1A. For meshes, use the extract_segment method
    #   voxel -> marching cubes -> OBJ
    #   For meshes, return the .obj/.mtl
    # 1B. For skeletonization, use the extract_segment_voxel method
    #   voxel mask -> clean -> skeletonize -> SWC.
    

    # Step 2: Clean meshes.
    #   2A Clean voxel mask (required for SWC):
    #       connected-component filtering (kills specks)
    #       optional hole fill (careful)
    #       optional small morphological closing
    #   2B Clean mesh (only if exporting OBJ):
    #       remove disconnected components
    #       non-manifold repair / degenerate faces
    #       optional smoothing

    # Step 3: Skeletonize
    #   Use 3D thinning/medial axis skeletonizer?
    #   Use MCF CGAL?
    #   Returns skeleton graph

    # Step 4: Convert Skeleton graph to SWC tree
    # Need to define these rules
    #   root selection (soma if present; otherwise diameter endpoint)
    #   graph compression (don’t emit every voxel; resample chains)
    #   cycle handling (if cycles appear, break them deterministically)
    #   ordering (parent_id < id)
    #   units (convert voxel indices → physical microns using voxel size + bbox offset)
    # Skeleton output is a graph (voxels connected by adjacency).
    # SWC requires a rooted tree with ordered node IDs.
    # Returns SWC tree

    # Step 5: Map VAST markers to SWC locations
    # For connectivity scaffolding
    # Should output 
    #   bouton marker centroid → nearest SWC segment (segment + t)
    #   synapse marker centroid → nearest SWC segment on pre + post cells

    # Step 6: Emit artifacts
    # Produce three outputs per dataset:
    #   cell_<id>.swc
    #   synapses.csv (pre_cell, post_cell, pre_loc, post_loc, marker_id, etc.)
    #   optional cell_<id>.obj for QA
        
    print("main")


if __name__ == "__main__":
    main()