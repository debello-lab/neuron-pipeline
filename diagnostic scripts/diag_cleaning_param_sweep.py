"""
Diagnostic: VoxelCleaner Closing Radius Parameter Sweep

Runs clean_mask() across a range of closing_radius_um values and records
metrics for each, helping identify the smallest radius that reliably bridges
annotation gaps without over-smoothing morphology.

Two test modes run in sequence:
  1. Synthetic  — a pure-NumPy cylinder with planted voids of known sizes.
                  Validates that each radius closes the gap sizes it should.
  2. Real       — a segment extracted from VAST (if --segment is provided).
                  Tests on actual data to confirm the chosen radius works.

Gap sizes planted in the synthetic segment:
  Small  : 5x5x5  voxels   (~0.40 µm at 80 nm/vox, ~0.20 µm at 40 nm/vox)
  Medium : 10x10x10 voxels (~0.80 µm at 80 nm/vox, ~0.40 µm at 40 nm/vox)
  Large  : 15x15x15 voxels (~1.20 µm at 80 nm/vox, ~0.60 µm at 40 nm/vox)

Outputs
-------
  sweep_results/<label>_sweep.csv            — metric table per radius
  sweep_results/<label>_radius_<r>_final.obj — cleaned mesh at each radius

Usage
-----
    # Synthetic only
    python "diagnostic scripts/diag_cleaning_param_sweep.py"

    # Synthetic + real segment
    python "diagnostic scripts/diag_cleaning_param_sweep.py" --segment 42

    # Custom radii
    python "diagnostic scripts/diag_cleaning_param_sweep.py" --radii 0.5,1.0,1.5,2.0

    # Real segment at MIP 0
    python "diagnostic scripts/diag_cleaning_param_sweep.py" --segment 42 --mip 0
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import marching_cubes

from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner, MeshCleaner

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RADII_UM = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
DEFAULT_MIPLEVEL = 1
DEFAULT_PADDING  = 2
OUTPUT_DIR       = Path(__file__).parent.parent / "diag_output" / "param_sweep"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("diag_sweep")


# ---------------------------------------------------------------------------
# Synthetic segment construction
# ---------------------------------------------------------------------------

def make_cylinder_with_gaps(
    voxel_size_um: Tuple[float, float, float] = (0.08, 0.08, 0.08),
    radius_vox: int = 20,
    length_vox: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a solid cylinder along the Z-axis with scattered voids of three sizes.

    The cylinder is axis-aligned (Z) and padded by 5 voxels on all sides.
    Voids are planted at known (z, y, x) positions so that we can predict
    which closing radius should seal them.

    Gap sizes (edge length of cubic void):
      small  = 5 vox  → ~0.40 µm at 80 nm/vox
      medium = 10 vox → ~0.80 µm at 80 nm/vox
      large  = 15 vox → ~1.20 µm at 80 nm/vox

    Parameters
    ----------
    voxel_size_um : (sx, sy, sz) — used only for logging
    radius_vox    : cylinder radius in voxels
    length_vox    : cylinder length along Z in voxels
    seed          : random seed for void placement

    Returns
    -------
    mask : bool array of shape (length_vox + 10, 2*radius_vox + 10, 2*radius_vox + 10)
           dtype bool, axis order (Z, Y, X)
    """
    pad = 5
    nz  = length_vox + 2 * pad
    ny  = 2 * radius_vox + 2 * pad
    nx  = 2 * radius_vox + 2 * pad
    cy  = ny // 2  # cylinder centre in Y
    cx  = nx // 2  # cylinder centre in X

    mask = np.zeros((nz, ny, nx), dtype=bool)

    # Fill cylinder: for each Z-slice, set voxels within radius
    zz = np.arange(pad, pad + length_vox)
    yy, xx = np.ogrid[:ny, :nx]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_vox ** 2
    mask[pad:pad + length_vox, :, :] = circle[np.newaxis, :, :]

    # Plant voids at fixed positions (known gap sizes)
    # Each entry: (z_start, y_start, x_start, edge_voxels, label)
    void_specs = [
        # small voids (~0.40 µm at 80 nm): should be sealed at r ≥ 0.5 µm
        (20,  cy - 2, cx - 2, 5,  "small"),
        (80,  cy - 2, cx - 2, 5,  "small"),
        (140, cy - 2, cx - 2, 5,  "small"),
        # medium voids (~0.80 µm at 80 nm): should be sealed at r ≥ 1.0 µm
        (35,  cy - 5, cx - 5, 10, "medium"),
        (100, cy - 5, cx - 5, 10, "medium"),
        (160, cy - 5, cx - 5, 10, "medium"),
        # large voids (~1.20 µm at 80 nm): should be sealed at r ≥ 1.5 µm
        (55,  cy - 7, cx - 7, 15, "large"),
        (120, cy - 7, cx - 7, 15, "large"),
    ]

    for z0, y0, x0, e, lbl in void_specs:
        z0 += pad  # shift into padded coordinate space
        mask[z0:z0 + e, y0:y0 + e, x0:x0 + e] = False

    voxel_count = int(mask.sum())
    sx, sy, sz = voxel_size_um
    log.info(
        f"[synthetic] Cylinder shape: {mask.shape}  "
        f"voxel size: ({sx*1000:.0f}, {sy*1000:.0f}, {sz*1000:.0f}) nm  "
        f"filled voxels: {voxel_count:,}"
    )
    log.info(
        f"[synthetic] Planted voids — "
        f"3x small (~{5*sx*1000:.0f} nm), "
        f"3x medium (~{10*sx*1000:.0f} nm), "
        f"2x large (~{15*sx*1000:.0f} nm)"
    )
    return mask

def create_test_segment_with_gaps(
    voxel_size_um: Tuple[float, float, float] = (0.04, 0.04, 0.04),
    outer_radius_vox: int = 30,
    inner_radius_vox: int = 25,
    length_vox: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Create a hollow cylinder with internal gaps, matching the structure
    and conventions of make_cylinder_with_gaps.

    Returns
    -------
    mask : bool array of shape (Z, Y, X) with padding
    """
    pad = 5
    nz = length_vox + 2 * pad
    ny = 2 * outer_radius_vox + 2 * pad
    nx = 2 * outer_radius_vox + 2 * pad

    cy = ny // 2
    cx = nx // 2

    mask = np.zeros((nz, ny, nx), dtype=bool)

    # Build outer cylinder
    yy, xx = np.ogrid[:ny, :nx]
    outer_circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= outer_radius_vox ** 2
    inner_circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= inner_radius_vox ** 2

    # Fill hollow cylinder
    shell = outer_circle & ~inner_circle
    mask[pad:pad + length_vox, :, :] = shell[np.newaxis, :, :]

    # Add random gaps
    rng = np.random.default_rng(seed)
    for _ in range(20):
        z = rng.integers(pad + 10, pad + length_vox - 10)
        y = rng.integers(10, ny - 10)
        x = rng.integers(10, nx - 10)
        mask[z:z+5, y:y+5, x:x+5] = False

    return mask


# ---------------------------------------------------------------------------
# OBJ export (mirrors diag_voxel_cleaning.py)
# ---------------------------------------------------------------------------

def mask_to_obj(
    mask: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    filepath: Path,
) -> None:
    if not mask.any():
        log.warning(f"  Mask is empty — skipping {filepath.name}")
        return

    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    sx, sy, sz = voxel_size_um
    verts_um = verts * np.array([sy, sx, sz])

    mesh_cleaner = MeshCleaner(logger=log)
    verts_um, faces, _ = mesh_cleaner.clean_mesh(verts_um, faces)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"# param sweep — {filepath.stem}\n")
        for v in verts_um:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    log.info(f"  Saved {filepath.name}  ({len(verts_um):,} verts, {len(faces):,} faces)")


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    mask: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    radii_um: List[float],
    label: str,
    output_dir: Path,
    export_obj: bool = True,
) -> List[Dict]:
    """
    Run clean_mask() for each radius in *radii_um* and collect metrics.

    Parameters
    ----------
    mask          : raw (uncleaned) boolean mask, shape (Z, Y, X)
    voxel_size_um : (sx, sy, sz) in µm
    radii_um      : list of closing radii to test
    label         : short identifier used in filenames (e.g. "synthetic", "seg_42")
    output_dir    : directory for OBJ and CSV outputs

    Returns
    -------
    rows : list of metric dicts, one per radius
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaner = VoxelCleaner(logger=log)

    sx, sy, sz = voxel_size_um
    avg_vox_um = (sx + sy) / 2.0
    original_voxels = int(mask.sum())
    bbox_vol = mask.size

    log.info(f"\n{'='*70}")
    log.info(f"SWEEP: {label}  ({original_voxels:,} original voxels)")
    log.info(f"{'='*70}")

    rows: List[Dict] = []

    for radius_um in radii_um:
        radius_vox = max(1, int(np.round(radius_um / avg_vox_um)))
        log.info(f"\n--- radius = {radius_um} µm  (~{radius_vox} vox) ---")

        # Guard: estimate scipy's working memory before attempting the closing.
        # binary_closing pads the input by the struct radius on each axis and
        # allocates ~5 full-sized arrays internally (bool + several float64).
        # For anisotropic voxels the struct radius differs per axis:
        #   rx = ry = radius_vox  (from avg_vox_um, which uses XY)
        #   rz = ceil(radius_um / sz)
        sz = voxel_size_um[2]
        rz_vox = max(1, int(np.ceil(radius_um / sz)))
        mz, my, mx = mask.shape
        padded_voxels = (mz + 2 * rz_vox) * (my + 2 * radius_vox) * (mx + 2 * radius_vox)
        estimated_gb = padded_voxels * 5 * 8 / 1e9  # 5 arrays × 8 bytes (float64)
        max_gb = 4.0  # skip if estimated working set exceeds this
        if estimated_gb > max_gb:
            log.warning(
                f"  SKIP radius={radius_um} µm: estimated scipy working memory "
                f"{estimated_gb:.1f} GB exceeds {max_gb} GB limit "
                f"(struct rxy={radius_vox} vox, rz={rz_vox} vox on mask {mask.shape})"
            )
            continue

        cleaned, stats = cleaner.clean_mask(
            mask,
            closing_radius_um=radius_um,
            voxel_size_um=voxel_size_um,
            keep_largest_only=True,
            fill_holes=True,
            smooth_iterations=0,
        )

        final_voxels  = stats["final_voxel_count"]
        fill_fraction = final_voxels / bbox_vol if bbox_vol > 0 else 0.0

        row = {
            "label":                  label,
            "closing_radius_um":      radius_um,
            "closing_radius_vox":     radius_vox,
            "original_voxels":        original_voxels,
            "closing_voxels_added":   stats["closing_voxels_added"],
            "hole_fill_voxels_added": stats["hole_fill_voxels_added"],
            "final_voxels":           final_voxels,
            "fill_fraction":          round(fill_fraction, 5),
            "voxels_added_total":     final_voxels - original_voxels,
            "pct_change":             round(
                100.0 * (final_voxels - original_voxels) / max(original_voxels, 1), 2
            ),
        }
        rows.append(row)

        log.info(
            f"  original={original_voxels:,}  "
            f"closing+{stats['closing_voxels_added']:,}  "
            f"holefill+{stats['hole_fill_voxels_added']:,}  "
            f"final={final_voxels:,}  ({row['pct_change']:+.1f}%)"
        )

        # Export OBJ for visual inspection (skip if --no-obj)
        if export_obj:
            obj_path = output_dir / f"{label}_radius_{radius_um:.1f}_final.obj"
            mask_to_obj(cleaned.mask, voxel_size_um, obj_path)

    # Print comparison table to log
    _log_sweep_table(rows, label)

    # Write CSV
    csv_path = output_dir / f"{label}_sweep.csv"
    _write_csv(rows, csv_path)

    return rows


def _log_sweep_table(rows: List[Dict], label: str) -> None:
    log.info(f"\n{'='*85}")
    log.info(f"SWEEP SUMMARY: {label}")
    log.info(f"{'-'*85}")
    log.info(
        f"  {'radius_um':>10s}  {'radius_vox':>10s}  {'orig_vox':>10s}  "
        f"{'closing+':>10s}  {'holefill+':>10s}  {'final':>10s}  {'pct_chg':>8s}"
    )
    log.info(f"{'-'*85}")
    for r in rows:
        log.info(
            f"  {r['closing_radius_um']:>10.2f}  {r['closing_radius_vox']:>10d}  "
            f"{r['original_voxels']:>10,d}  {r['closing_voxels_added']:>10,d}  "
            f"{r['hole_fill_voxels_added']:>10,d}  {r['final_voxels']:>10,d}  "
            f"{r['pct_change']:>+7.1f}%"
        )
    log.info(f"{'='*85}\n")


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", type=int, default=None,
                   help="Segment ID to extract from VAST (omit for synthetic only)")
    p.add_argument("--mip",     type=int, default=DEFAULT_MIPLEVEL,
                   help=f"MIP level for VAST extraction (default: {DEFAULT_MIPLEVEL})")
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING,
                   help=f"Bbox padding in voxels (default: {DEFAULT_PADDING})")
    p.add_argument("--radii",   type=str, default=None,
                   help="Comma-separated closing radii in µm "
                        f"(default: {','.join(str(r) for r in DEFAULT_RADII_UM)})")
    p.add_argument("--no-obj",  action="store_true", default=False,
                   help="Skip OBJ mesh export (faster — CSV metrics only)")
    p.add_argument("--z-slab",  type=int, default=None, metavar="N",
                   help="Crop the real segment to N Z-slices around the centre "
                        "before sweeping. Recommended for large masks where the "
                        "full volume would OOM (e.g. --z-slab 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    radii_um = (
        [float(r.strip()) for r in args.radii.split(",")]
        if args.radii
        else DEFAULT_RADII_UM
    )
    log.info(f"Radii to sweep: {radii_um} µm")

    all_results: List[Dict] = []

    # ------------------------------------------------------------------
    # Test 1: Synthetic segment
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("TEST 1: Synthetic cylinder with planted voids")
    log.info("=" * 60)

    # Use 80 nm isotropic to match MIP 1 (default test case)
    synth_voxel_size = (0.08, 0.08, 0.08)
    synth_mask = make_cylinder_with_gaps(voxel_size_um=synth_voxel_size)
    # synth_mask = create_test_segment_with_gaps()

    synth_rows = run_sweep(
        mask=synth_mask,
        voxel_size_um=synth_voxel_size,
        radii_um=radii_um,
        label="synthetic",
        output_dir=OUTPUT_DIR / "synthetic",
        export_obj=not args.no_obj,
    )
    all_results.extend(synth_rows)

    # Interpret: which radii sealed the planted voids?
    _interpret_synthetic(synth_rows, synth_voxel_size)

    # ------------------------------------------------------------------
    # Test 2: Real segment from VAST (optional)
    # ------------------------------------------------------------------
    if args.segment is not None:
        log.info("\n" + "=" * 60)
        log.info(f"TEST 2: Real segment {args.segment} (MIP {args.mip})")
        log.info("=" * 60)

        try:
            from vastpy.control.exporting import VASTControlClass
            from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
        except ImportError as e:
            log.error(f"Could not import VAST or pipeline modules: {e}")
            return

        log.info("Connecting to VAST...")
        vast = VASTControlClass()
        if not vast.connect(host="127.0.0.1", port=22081, timeout=100):
            log.error("Failed to connect to VAST. Ensure VAST is running and API is enabled.")
            return
        log.info("Connected.")

        extractor = SegmentSurfaceExtractor(vast, str(OUTPUT_DIR))
        log.info(f"Extracting segment {args.segment} at MIP {args.mip}...")
        mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
            segment_id=args.segment,
            miplevel=args.mip,
            padding=args.padding,
        )

        if mask is None or voxel_size is None or bbox_min is None:
            log.error(f"Extraction failed for segment {args.segment} — is the ID valid?")
            return

        log.info(
            f"Mask shape: {mask.shape}  voxel size: {voxel_size} µm  "
            f"filled voxels: {int(mask.sum()):,}"
        )

        # Optional Z-slab crop for parameter tuning on large masks.
        # Crops to N slices centred on the Z midpoint of the filled voxels
        # (not the bbox midpoint) so the slab is guaranteed to contain data.
        if args.z_slab is not None:
            filled_z = np.where(mask.any(axis=(1, 2)))[0]
            if len(filled_z) == 0:
                log.error("Mask has no filled voxels — cannot crop slab.")
                return
            z_center = int(filled_z[len(filled_z) // 2])
            half = args.z_slab // 2
            z_lo = max(0, z_center - half)
            z_hi = min(mask.shape[0], z_lo + args.z_slab)
            z_lo = max(0, z_hi - args.z_slab)  # clamp if near top edge
            mask = mask[z_lo:z_hi, :, :]
            log.info(
                f"Z-slab crop: Z[{z_lo}:{z_hi}] ({args.z_slab} slices centred on "
                f"Z={z_center})  filled voxels after crop: {int(mask.sum()):,}"
            )
            if not mask.any():
                log.error("Z-slab contains no filled voxels — try a larger --z-slab.")
                return

        real_label = f"seg_{args.segment}_mip{args.mip}"
        if args.z_slab is not None:
            real_label += f"_zslab{args.z_slab}"
        real_rows = run_sweep(
            mask=mask,
            voxel_size_um=voxel_size,
            radii_um=radii_um,
            label=real_label,
            output_dir=OUTPUT_DIR / real_label,
            export_obj=not args.no_obj,
        )
        all_results.extend(real_rows)

    log.info(f"\nAll outputs written to: {OUTPUT_DIR}")


def _interpret_synthetic(rows: List[Dict], voxel_size_um: Tuple[float, float, float]) -> None:
    """
    Log an interpretation of which radii sealed the planted voids.

    The synthetic cylinder starts with known total voxels. After cleaning,
    voxels_added_total reflects gap closure + hole fill. A sharp increase
    between consecutive radii indicates the threshold at which a void size
    is being bridged.
    """
    sx = voxel_size_um[0]
    small_um  = 5  * sx
    medium_um = 10 * sx
    large_um  = 15 * sx

    log.info("\n--- Synthetic interpretation ---")
    log.info(
        f"  Planted void sizes: "
        f"small={small_um*1000:.0f} nm, "
        f"medium={medium_um*1000:.0f} nm, "
        f"large={large_um*1000:.0f} nm"
    )
    log.info(
        f"  Expected closure thresholds: "
        f"small at r≥{small_um/2:.2f} µm, "
        f"medium at r≥{medium_um/2:.2f} µm, "
        f"large at r≥{large_um/2:.2f} µm"
    )

    if len(rows) < 2:
        return

    log.info("  Voxels added vs radius (look for jumps):")
    prev = rows[0]
    for row in rows[1:]:
        delta = row["voxels_added_total"] - prev["voxels_added_total"]
        flag = "  <-- jump" if delta > 500 else ""
        log.info(
            f"    {prev['closing_radius_um']:.1f} → {row['closing_radius_um']:.1f} µm : "
            f"Δvoxels_added = {delta:+,d}{flag}"
        )
        prev = row


if __name__ == "__main__":
    main()
