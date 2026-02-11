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
    print("main")


if __name__ == "__main__":
    main()