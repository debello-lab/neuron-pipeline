"""
Diagnostic: Voxel Cleaning Stage

Extracts a voxel mask from VAST and exports OBJ surface meshes at each
cleaning stage so results can be inspected visually (e.g. in MeshLab).

Also runs quantitative checks at each stage:
  - Hole census      : per-slice 2D hole count and voxel totals
  - 3D vs 2D fill    : detects shell gaps (3D fill escapes but 2D doesn't)
  - Surface ratio    : surface voxels / total voxels (drops as holes fill)

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
from typing import Dict, List, Tuple

import numpy as np
from skimage.measure import marching_cubes, label
from scipy.ndimage import binary_fill_holes as fill_holes_3d

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
# Quantitative checks
# ---------------------------------------------------------------------------

def _fill_holes_2d(slc: np.ndarray) -> np.ndarray:
    """Fill holes in a single 2D slice."""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(slc)


def hole_census(mask: np.ndarray) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Count holes per Z-slice using 2D fill.

    Returns
    -------
    per_slice : list of (z_index, n_distinct_holes, n_hole_voxels)
    total_hole_voxels : int
        Sum of hole voxels across all slices.
    """
    per_slice = []
    total_hole_voxels = 0
    for z in range(mask.shape[0]):
        slc = mask[z]
        filled = _fill_holes_2d(slc)
        holes = filled & ~slc
        n_hole_voxels = int(holes.sum())
        n_holes = int(label(holes).max()) if n_hole_voxels > 0 else 0
        per_slice.append((z, n_holes, n_hole_voxels))
        total_hole_voxels += n_hole_voxels
    return per_slice, total_hole_voxels


def fill_discrepancy(mask: np.ndarray) -> Dict[str, int]:
    """
    Compare 3D binary_fill_holes against per-slice 2D fills.

    If 2D finds many hole voxels but 3D finds few/none, the shell has
    gaps that let the 3D flood fill escape to the exterior.  After
    closing, the two numbers should converge.

    Returns
    -------
    dict with keys:
        filled_3d      : voxels added by 3D fill
        filled_2d      : voxels added by summing per-slice 2D fills
        discrepancy    : filled_2d - filled_3d  (>0 means shell has gaps)
    """
    # 3D fill
    filled_3d_mask = fill_holes_3d(mask)
    voxels_3d = int(filled_3d_mask.sum() - mask.sum())

    # 2D per-slice fill
    voxels_2d = 0
    for z in range(mask.shape[0]):
        filled_slice = _fill_holes_2d(mask[z])
        voxels_2d += int((filled_slice & ~mask[z]).sum())

    return {
        "filled_3d": voxels_3d,
        "filled_2d": voxels_2d,
        "discrepancy": voxels_2d - voxels_3d,
    }


def surface_voxel_ratio(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute the fraction of filled voxels that sit on the surface.

    A surface voxel is any True voxel with at least one False
    face-neighbour (6-connectivity).  As holes are filled and the
    surface smoothed, this ratio should decrease.

    Returns
    -------
    dict with keys:
        total_voxels   : int
        surface_voxels : int
        surface_ratio  : float  (surface / total, 0-1)
    """
    from scipy.ndimage import binary_erosion

    total = int(mask.sum())
    if total == 0:
        return {"total_voxels": 0, "surface_voxels": 0, "surface_ratio": 0.0}

    # Interior = eroded mask (every face-neighbour is also True)
    interior = binary_erosion(mask)  # default struct = 6-connectivity cross
    surface = mask & ~interior
    n_surface = int(surface.sum())

    return {
        "total_voxels": total,
        "surface_voxels": n_surface,
        "surface_ratio": n_surface / total,
    }


def run_quantitative_checks(
    mask: np.ndarray,
    stage_label: str,
) -> Dict[str, object]:
    """
    Run all three quantitative checks on a mask and log the results.

    Returns a dict with all metrics for later comparison.
    """
    log.info(f"  [{stage_label}] Running quantitative checks...")

    # 1. Hole census
    per_slice, total_2d_holes = hole_census(mask)
    slices_with_holes = sum(1 for _, nh, _ in per_slice if nh > 0)
    max_holes_in_slice = max((nh for _, nh, _ in per_slice), default=0)

    log.info(
        f"    Hole census:  {total_2d_holes:,} hole voxels across "
        f"{slices_with_holes}/{mask.shape[0]} slices  "
        f"(max {max_holes_in_slice} distinct holes in one slice)"
    )

    # 2. 3D vs 2D fill discrepancy
    disc = fill_discrepancy(mask)
    log.info(
        f"    Fill discrepancy:  3D={disc['filled_3d']:,}  2D={disc['filled_2d']:,}  "
        f"gap={disc['discrepancy']:,}"
    )
    if disc["discrepancy"] > 0:
        log.warning(
            f"    Shell has gaps: 2D fill finds {disc['discrepancy']:,} more "
            f"voxels than 3D fill (3D flood escapes through holes in the shell)"
        )
    elif disc["filled_3d"] > 0:
        log.info(
            f"    Shell is sealed: 3D fill found {disc['filled_3d']:,} enclosed voxels"
        )
    else:
        log.info("    No fillable voids detected by either method")

    # 3. Surface ratio
    surf = surface_voxel_ratio(mask)
    log.info(
        f"    Surface ratio:  {surf['surface_voxels']:,} / {surf['total_voxels']:,} "
        f"= {surf['surface_ratio']:.3f}"
    )

    return {
        "stage": stage_label,
        "total_voxels": int(mask.sum()),
        "hole_voxels_2d": total_2d_holes,
        "slices_with_holes": slices_with_holes,
        "filled_3d": disc["filled_3d"],
        "filled_2d": disc["filled_2d"],
        "fill_discrepancy": disc["discrepancy"],
        "surface_voxels": surf["surface_voxels"],
        "surface_ratio": surf["surface_ratio"],
    }


def log_comparison_table(all_metrics: List[Dict]) -> None:
    """
    Log a summary table comparing metrics across all cleaning stages.
    """
    log.info("")
    log.info("=" * 90)
    log.info("STAGE COMPARISON")
    log.info("-" * 90)
    log.info(
        f"  {'Stage':<20s}  {'Voxels':>10s}  {'2D Holes':>10s}  "
        f"{'3D Fill':>10s}  {'Discrep':>10s}  {'Surf Ratio':>10s}"
    )
    log.info("-" * 90)
    for m in all_metrics:
        log.info(
            f"  {m['stage']:<20s}  {m['total_voxels']:>10,d}  "
            f"{m['hole_voxels_2d']:>10,d}  {m['filled_3d']:>10,d}  "
            f"{m['fill_discrepancy']:>10,d}  {m['surface_ratio']:>10.4f}"
        )
    log.info("=" * 90)

    # Interpretation hints
    first, last = all_metrics[0], all_metrics[-1]
    if first["fill_discrepancy"] > 0 and last["fill_discrepancy"] == 0:
        log.info("  Shell gaps fully sealed by closing + hole fill.")
    elif last["fill_discrepancy"] > 0:
        log.warning(
            f"  Shell still has gaps after all stages "
            f"(discrepancy={last['fill_discrepancy']:,}).  "
            f"Consider increasing closing_radius_um."
        )
    if last["surface_ratio"] < first["surface_ratio"]:
        log.info(
            f"  Surface ratio improved: {first['surface_ratio']:.4f} -> "
            f"{last['surface_ratio']:.4f}"
        )
    log.info("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_npz(
    mask: np.ndarray,
    voxel_size_um: tuple,
    bbox_min_vox: tuple,
    segment_id: int,
    segment_name: str,
    filepath: Path,
) -> None:
    """
    Save a voxel mask and spatial metadata to a compressed .npz file.

    Saved arrays match the format written by extract_surfaces._save_voxel_mask
    so the file can be loaded by other pipeline tools (e.g. view_mask_slices.py).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        mask=mask,
        bbox_min_vox=np.array(bbox_min_vox),
        voxel_size_um=np.array(voxel_size_um),
        segment_id=segment_id,
        segment_name=segment_name,
    )
    log.info(f"  Saved {filepath.name}  ({int(mask.sum()):,} filled voxels)")


def mask_to_obj(
    mask: np.ndarray,
    voxel_size_um: tuple,
    filepath: Path,
) -> Tuple[int, int]:
    """
    Run marching cubes on a boolean mask and write the result as an OBJ file.

    Vertices are scaled to physical micron coordinates using voxel_size_um
    (sx, sy, sz).  The mask is in (Z, Y, X) order; marching_cubes returns
    verts in the same (Z, Y, X) index order.

    Returns (n_verts, n_faces).
    """
    if not mask.any():
        log.warning(f"  Mask is empty — skipping {filepath.name}")
        return 0, 0

    verts, faces, _, _ = marching_cubes(mask, level=0.5)

    # marching_cubes returns (Z, Y, X) index coordinates.
    # Scale each axis by its physical voxel dimension.
    sx, sy, sz = voxel_size_um
    verts_um = verts * np.array([sy, sx, sz])   # (Z*sz, Y*sy, X*sx)

    # Reorder to (X, Y, Z) for OBJ output convention
    # verts_um = verts_um[:, [2, 1, 0]]           # (X*sx, Y*sy, Z*sz)

    # Clean the mesh before saving
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
    bbox_min_vox: tuple,
    segment_id: int,
    segment_name: str,
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Step through each cleaning stage on *mask*, exporting an OBJ surface
    and running quantitative checks after each step.
    """
    cleaner = VoxelCleaner(logger=log)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict] = []

    # ------------------------------------------------------------------
    # Stage 00: raw (no cleaning)
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 00 — raw mask ({int(mask.sum()):,} filled voxels)")
    mask_to_obj(mask, voxel_size_um, output_dir / f"{prefix}_00_raw.obj")
    mask_to_npz(mask, voxel_size_um, bbox_min_vox, segment_id, segment_name, output_dir / f"{prefix}_00_raw.npz")
    all_metrics.append(run_quantitative_checks(mask, "00_raw"))

    # ------------------------------------------------------------------
    # Stage 01: morphological closing only
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 01 — closing")
    vd01, stats01 = cleaner.clean_mask(
        mask,
        closing_radius_um=2,
        keep_largest_only=False,
        fill_holes=False,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    if not vd01.mask.any():
        log.warning(f"[{prefix}] Stage 01 produced empty mask — skipping remaining stages")
        return
    log.info(
        f"  closing_voxels_added={stats01['closing_voxels_added']:,}  "
        f"components={stats01['original_components']}"
    )
    mask_to_obj(vd01.mask, voxel_size_um, output_dir / f"{prefix}_01_closing.obj")
    mask_to_npz(vd01.mask, voxel_size_um, bbox_min_vox, segment_id, segment_name, output_dir / f"{prefix}_01_closing.npz")
    all_metrics.append(run_quantitative_checks(vd01.mask, "01_closing"))

    # ------------------------------------------------------------------
    # Stage 02: closing + keep largest component
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 02 — closing + component filter")
    vd02, stats02 = cleaner.clean_mask(
        mask,
        closing_radius_um=2,
        keep_largest_only=True,
        fill_holes=False,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    log.info(
        f"  removed_components={stats02['removed_components']}  "
        f"kept={stats02['kept_components']}"
    )
    mask_to_obj(vd02.mask, voxel_size_um, output_dir / f"{prefix}_02_components.obj")
    mask_to_npz(vd02.mask, voxel_size_um, bbox_min_vox, segment_id, segment_name, output_dir / f"{prefix}_02_components.npz")
    all_metrics.append(run_quantitative_checks(vd02.mask, "02_components"))

    # ------------------------------------------------------------------
    # Stage 03: closing + component filter + hole fill
    # ------------------------------------------------------------------
    log.info(f"[{prefix}] Stage 03 — closing + component filter + hole fill")
    vd03, stats03 = cleaner.clean_mask(
        mask,
        closing_radius_um=2,
        keep_largest_only=True,
        fill_holes=True,
        smooth_iterations=0,
        voxel_size_um=voxel_size_um,
    )
    log.info(
        f"  hole_fill_voxels_added={stats03['hole_fill_voxels_added']:,}  "
        f"final={stats03['final_voxel_count']:,} voxels"
    )
    mask_to_obj(vd03.mask, voxel_size_um, output_dir / f"{prefix}_03_holefill.obj")
    mask_to_npz(vd03.mask, voxel_size_um, bbox_min_vox, segment_id, segment_name, output_dir / f"{prefix}_03_holefill.npz")
    all_metrics.append(run_quantitative_checks(vd03.mask, "03_holefill"))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    log_comparison_table(all_metrics)


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
    seg_name = f"seg_{SEGMENT_ID}"
    run_stage_tests(
        mask=mask,
        voxel_size_um=voxel_size,
        bbox_min_vox=bbox_min,
        segment_id=SEGMENT_ID,
        segment_name=seg_name,
        output_dir=OUTPUT_DIR / "full",
        prefix="full",
    )

    # ------------------------------------------------------------------
    # Test 2: Single slice — middle Z window of the mask
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
                bbox_min_vox=bbox_min,
                segment_id=SEGMENT_ID,
                segment_name=seg_name,
                output_dir=OUTPUT_DIR / "slice",
                prefix="slice",
            )

    log.info("Done. Outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()