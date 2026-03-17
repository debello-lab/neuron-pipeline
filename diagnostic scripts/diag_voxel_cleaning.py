"""
Diagnostic: Voxel Cleaning Stage

Extracts a voxel mask from VAST and exports OBJ surface meshes at each
cleaning stage so results can be inspected visually (e.g. in MeshLab).

Tests
-----
1. Full neuron  — all cleaning stages on the complete mask
2. Single slice — middle Z-slice window, same stages (skipped if too thin)

Cleaning stages
---------------
  00_raw         : marching cubes on the raw, uncleaned mask
  01_closing     : after morphological closing (bridges annotation gaps)
  02_components  : after keeping only the largest connected component
  03_holefill    : after filling enclosed interior voids

Usage
-----
    python "diagnostic scripts/diag_voxel_cleaning.py" [segment_id]
"""

import sys
import logging
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from vastpy.control.exporting import VASTControlClass
from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner, MeshCleaner

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEGMENT_ID       = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MIPLEVEL         = 1        # 0 = full res, 1 = half res
PADDING          = 2        # extra voxels around bounding box
SLICE_HALF_WIDTH = 10       # Z half-width for single-slice test
OUTPUT_DIR       = Path(__file__).parent.parent / "diag_output" / "voxel_cleaning"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("diag_voxel_cleaning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_obj(mask: np.ndarray, voxel_size_um: tuple, filepath: Path) -> tuple[int, int]:
    """
    Run marching cubes on a boolean mask and write the result as an OBJ file.

    Vertices are scaled to physical micron coordinates using voxel_size_um
    (sx, sy, sz).  The mask is expected in (Z, Y, X) order; marching_cubes
    returns verts in (Z, Y, X) order which we remap to (X, Y, Z) for OBJ.

    Returns (n_verts, n_faces).
    """
    if not mask.any():
        log.warning(f"  Mask is empty — skipping {filepath.name}")
        return 0, 0

    verts, faces, _, _ = marching_cubes(mask, level=0.5)

    # voxel_size_um is (sx, sy, sz) = (voxelsizex, voxelsizey, voxelsizez).
    sx, sy, sz = voxel_size_um
    verts_um = verts * np.array([sy, sx, sz])  # (Y*sy, X*sx, Z*sz)

    # Clean the mesh before saving.
    mesh_cleaner = MeshCleaner(logger=log)
    verts_um, faces, _ = mesh_cleaner.clean_mesh(verts_um, faces)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"# Voxel cleaning diagnostic — {filepath.stem}\n")
        for v in verts_um:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    log.info(f"  Saved {filepath.name}  ({len(verts_um):,} verts, {len(faces):,} faces)")
    return len(verts_um), len(faces)


def run_stage_tests(
    mask: np.ndarray,
    voxel_size_um: tuple,
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Step through each cleaning stage on *mask*, exporting an OBJ surface
    after each step.  Each stage is applied to the original raw mask so the
    exports are independent and directly comparable.
    """
    cleaner = VoxelCleaner(logger=log)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 00: raw (no cleaning)
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 00 — raw mask ({int(mask.sum()):,} filled voxels)")
    mask_to_obj(mask, voxel_size_um, output_dir / f"{prefix}_00_raw.obj")

    # ------------------------------------------------------------------
    # Stage 01: morphological closing only
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 01 — closing")
    m01, stats = cleaner.clean_mask(
        mask,
        closing_radius=2,
        keep_largest_only=False,
        fill_holes=False,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    log.info(f"  closing_voxels_added={stats['closing_voxels_added']:,}  "
             f"components={stats['original_components']}")
    mask_to_obj(m01, voxel_size_um, output_dir / f"{prefix}_01_closing.obj")

    # ------------------------------------------------------------------
    # Stage 02: closing + keep largest component
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 02 — closing + component filter")
    m02, stats = cleaner.clean_mask(
        mask,
        closing_radius=2,
        keep_largest_only=True,
        fill_holes=False,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    log.info(f"  removed_components={stats['removed_components']}  "
             f"kept={stats['kept_components']}")
    mask_to_obj(m02, voxel_size_um, output_dir / f"{prefix}_02_components.obj")

    # ------------------------------------------------------------------
    # Stage 03: closing + component filter + hole fill
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 03 — closing + component filter + hole fill")
    m03, stats = cleaner.clean_mask(
        mask,
        closing_radius=2,
        keep_largest_only=True,
        fill_holes=True,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    log.info(f"  hole_fill_voxels_added={stats['hole_fill_voxels_added']:,}  "
             f"final={stats['final_voxel_count']:,} voxels")
    mask_to_obj(m03, voxel_size_um, output_dir / f"{prefix}_03_holefill.obj")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Connect to VAST
    log.info("Connecting to VAST...")
    vast = VASTControlClass()
    if not vast.connect(host="127.0.0.1", port=22081, timeout=100):
        log.error("Failed to connect to VAST. Is it running and accessible?")
        return
    log.info("Connected.")

    # 2. Extract voxel mask
    log.info(f"Extracting voxel mask for segment {SEGMENT_ID} (MIP {MIPLEVEL})...")
    extractor = SegmentSurfaceExtractor(vast, str(OUTPUT_DIR))
    mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
        segment_id=SEGMENT_ID,
        miplevel=MIPLEVEL,
        padding=PADDING,
    )

    if mask is None or voxel_size is None or bbox_min is None:
        log.error("Voxel extraction failed — is the segment ID valid?")
        return

    log.info(
        f"Mask shape: {mask.shape}  |  voxel size: {voxel_size} um  "
        f"|  bbox_min: {bbox_min}  |  filled voxels: {int(mask.sum()):,}"
    )

    # ------------------------------------------------------------------
    # Test 1: Full neuron — cleaning stages on the complete mask
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("TEST 1: Full neuron")
    log.info("=" * 60)
    run_stage_tests(
        mask=mask,
        voxel_size_um=voxel_size,
        output_dir=OUTPUT_DIR / "full",
        prefix="full",
    )

    # ------------------------------------------------------------------
    # Test 2: Single slice — middle Z window of the mask
    #
    # There is no dedicated slice extraction method in extract_surfaces,
    # so we slice the numpy array directly.
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("TEST 2: Single slice")
    log.info("=" * 60)

    n_z = mask.shape[0]
    if n_z < 3:
        log.warning(f"Mask has only {n_z} Z-slices — skipping single-slice test.")
    else:
        z_center = n_z // 2
        z_lo = max(0, z_center - SLICE_HALF_WIDTH)
        z_hi = min(n_z, z_center + SLICE_HALF_WIDTH)
        slice_mask = mask[z_lo:z_hi, :, :]
        log.info(
            f"Slicing Z[{z_lo}:{z_hi}] (center={z_center})  "
            f"slice shape={slice_mask.shape}  "
            f"filled voxels={int(slice_mask.sum()):,}"
        )
        if not slice_mask.any():
            log.warning("Slice contains no filled voxels — skipping single-slice test.")
        else:
            run_stage_tests(
                mask=slice_mask,
                voxel_size_um=voxel_size,
                output_dir=OUTPUT_DIR / "slice",
                prefix="slice",
            )

    log.info("Done. Outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
