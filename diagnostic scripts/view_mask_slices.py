"""
Voxel Mask ASCII Viewer — Diagnostic Utility
Neural Reconstruction Pipeline

Displays a .npz voxel mask slice-by-slice in the terminal using only
numpy and the standard library. No display, no matplotlib required.
Works over SSH.

The mask is stored as (X, Y, Z) in the .npz; this script transposes it
to (Z, Y, X) on load so --axis 0 slices along Z as expected.

Each slice is shown in a fixed-size viewport. Pan around with W/S/A/D.

Usage
-----
    python view_mask_slices.py path/to/seg_001_mask.npz

    # Jump to a specific slice
    python view_mask_slices.py seg.npz --start 42

    # Slice along Y or X axis
    python view_mask_slices.py seg.npz --axis 1

    # Print fill histogram and exit (find which slices have content)
    python view_mask_slices.py seg.npz --summary-only

Controls (while viewing)
------------------------
    Enter / N   : next slice
    P           : previous slice
    F           : skip forward 10 slices
    B           : skip back 10 slices
    W / S       : pan viewport up / down
    A / D       : pan viewport left / right
    R           : reset viewport to center on content
    Q           : quit
"""

import argparse
import msvcrt
import sys
import os
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILLED_CHAR = '#'
EMPTY_CHAR  = '.'

# Axis labels: (col_axis_name, row_axis_name, slice_axis_name)
# After transposing the stored (X,Y,Z) → (Z,Y,X):
#   axis=0 slices along Z → each slice is a (Y,X) plane
#   axis=1 slices along Y → each slice is a (Z,X) plane
#   axis=2 slices along X → each slice is a (Z,Y) plane
AXIS_LABELS = {
    0: ('X', 'Y', 'Z'),
    1: ('X', 'Z', 'Y'),
    2: ('Y', 'Z', 'X'),
}

# Default viewport size in voxels (before stride subsampling)
DEFAULT_VIEW_ROWS = 60
DEFAULT_VIEW_COLS = 160   # will be further capped by --width after x_stride


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mask(npz_path: str):
    """
    Load a voxel mask saved by SegmentSurfaceExtractor._save_voxel_mask.

    The .npz stores the array in (X, Y, Z) order (the natural order from the
    VAST API).  We transpose here to (Z, Y, X) so that axis=0 corresponds to
    Z-slices throughout the rest of this script.

    Returns
    -------
    mask         : bool[Z, Y, X]
    bbox_min_vox : (minx, miny, minz)
    voxel_size   : (sx, sy, sz) um
    seg_id       : int
    seg_name     : str
    """
    data = np.load(npz_path, allow_pickle=True)
    raw  = data['mask'].astype(bool)   # stored as (X, Y, Z)
    mask = raw.transpose(2, 1, 0)      # → (Z, Y, X)

    bbox_min_vox = tuple(data['bbox_min_vox'])
    voxel_size   = tuple(data['voxel_size_um'])
    seg_id       = int(data['segment_id'])
    seg_name     = str(data['segment_name'])
    return mask, bbox_min_vox, voxel_size, seg_id, seg_name


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def get_slice(mask: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a 2D boolean slice (rows, cols) along the given axis."""
    return np.take(mask, index, axis=axis)


def nearest_filled_center(sl: np.ndarray, vr: int, vc: int, view_rows: int, view_cols: int):
    """
    If the current viewport contains no filled voxels, find the filled voxel
    nearest to the viewport center and return (new_vr, new_vc) that centers
    the viewport on it.  Returns (vr, vc) unchanged if the viewport already
    shows content or the slice is entirely empty.
    """
    rows, cols = np.where(sl)
    if len(rows) == 0:
        return vr, vc   # empty slice — nothing to snap to

    # Check whether any filled voxel already falls inside the viewport
    r_min = vr;         r_max = vr + view_rows - 1
    c_min = vc;         c_max = vc + view_cols - 1
    visible = ((rows >= r_min) & (rows <= r_max) &
               (cols >= c_min) & (cols <= c_max))
    if visible.any():
        return vr, vc   # content already visible — don't auto-pan

    # Find the filled voxel closest to the viewport center
    cr = vr + view_rows // 2
    cc = vc + view_cols // 2
    dist2 = (rows - cr) ** 2 + (cols - cc) ** 2
    nearest = np.argmin(dist2)
    new_vr = int(rows[nearest]) - view_rows // 2
    new_vc = int(cols[nearest]) - view_cols // 2
    return new_vr, new_vc


def slice_stats(sl: np.ndarray) -> dict:
    filled = int(sl.sum())
    total  = sl.size
    return {
        'filled':   filled,
        'total':    total,
        'fill_pct': 100.0 * filled / total if total > 0 else 0.0,
    }


def content_center(mask: np.ndarray, axis: int):
    """
    Return (row_center, col_center) of the filled-voxel centroid projected
    across all slices along `axis`.  Used to initialise the viewport.
    """
    sl_shape = [s for i, s in enumerate(mask.shape) if i != axis]
    n_rows, n_cols = sl_shape[0], sl_shape[1]
    projection = mask.any(axis=axis)        # 2D bool (rows, cols)
    rows, cols = np.where(projection)
    if len(rows) == 0:
        return n_rows // 2, n_cols // 2
    return int(rows.mean()), int(cols.mean())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_slice_ascii(
    sl: np.ndarray,
    r_min: int, r_max: int,
    c_min: int, c_max: int,
    max_display_cols: int = 100,
    x_stride: int = 2,
    y_stride: int = 1,
) -> list[str]:
    """
    Render the viewport region of a 2D boolean slice as ASCII strings.

    sl            : bool[rows, cols]
    r_min/r_max   : viewport row bounds (inclusive, clamped to sl)
    c_min/c_max   : viewport col bounds (inclusive, clamped to sl)
    max_display_cols : maximum number of characters per line
    x_stride      : subsample columns to compensate for character aspect ratio
    y_stride      : subsample rows to reduce vertical height

    Rows are printed top = high physical row (flipped so Y increases upward).
    """
    # Clamp to actual array bounds
    r_min = max(r_min, 0);  r_max = min(r_max, sl.shape[0] - 1)
    c_min = max(c_min, 0);  c_max = min(c_max, sl.shape[1] - 1)

    region = sl[r_min:r_max + 1:y_stride, c_min:c_max + 1:x_stride]

    # Further subsample if still too wide
    if region.shape[1] > max_display_cols:
        extra = (region.shape[1] + max_display_cols - 1) // max_display_cols
        region = region[:, ::extra]

    # Flip rows: high row index = high Y = top of printed image
    region = region[::-1, :]

    return [''.join(FILLED_CHAR if v else EMPTY_CHAR for v in row)
            for row in region]


# ---------------------------------------------------------------------------
# Summary / histogram
# ---------------------------------------------------------------------------

def print_summary(mask, bbox_min_vox, voxel_size, seg_id, seg_name, axis):
    z, y, x = mask.shape   # after transpose: (Z, Y, X)
    total_vox = int(mask.sum())
    col_ax, row_ax, slice_ax = AXIS_LABELS[axis]

    print("=" * 60)
    print(f"  Segment : {seg_name}  (id={seg_id})")
    print(f"  Shape   : Z={z}  Y={y}  X={x}  (after XYZ→ZYX transpose)")
    print(f"  Voxels  : {total_vox:,}  ({100.0 * total_vox / mask.size:.1f}% fill)")
    print(f"  Vox sz  : x={voxel_size[0]:.4f} um  y={voxel_size[1]:.4f} um  z={voxel_size[2]:.4f} um")
    phys = (x * voxel_size[0], y * voxel_size[1], z * voxel_size[2])
    print(f"  Phys sz : x={phys[0]:.2f} um  y={phys[1]:.2f} um  z={phys[2]:.2f} um")
    print(f"  Origin  : minx={bbox_min_vox[0]}  miny={bbox_min_vox[1]}  minz={bbox_min_vox[2]}  (global vox)")
    print(f"  Viewing : slices along {slice_ax}  (rows={row_ax}, cols={col_ax})")
    print("=" * 60)


def fill_histogram(mask: np.ndarray, axis: int, bar_width: int = 40) -> list[str]:
    counts = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
    max_count = int(counts.max()) if counts.max() > 0 else 1
    _, _, slice_ax = AXIS_LABELS[axis]
    lines = [f"  Fill histogram along {slice_ax}-axis  (max={max_count:,} vox/slice)",
             "  " + "-" * (bar_width + 22)]
    for i, c in enumerate(counts):
        bar = '#' * int(bar_width * c / max_count)
        lines.append(f"  {slice_ax}={i:>4}  {bar:<{bar_width}}  {int(c):>8,}")
    return lines


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def run_viewer(
    mask, seg_name,
    axis: int = 0,
    start: int = 0,
    max_display_cols: int = 100,
    x_stride: int = 2,
    y_stride: int = 1,
    view_rows: int = DEFAULT_VIEW_ROWS,
    view_cols: int = DEFAULT_VIEW_COLS,
    show_histogram: bool = False,
):
    n_slices = mask.shape[axis]
    col_ax, row_ax, slice_ax = AXIS_LABELS[axis]

    # Slice dimensions (rows, cols) of a single slice
    sl_shape = list(mask.shape)
    sl_shape.pop(axis)
    n_rows, n_cols = sl_shape[0], sl_shape[1]

    # Initialise viewport centered on filled content
    rc, cc = content_center(mask, axis)
    vr = rc - view_rows // 2   # viewport top row in voxel space
    vc = cc - view_cols // 2   # viewport left col in voxel space

    pan_step_r = view_rows // 3
    pan_step_c = view_cols // 3

    if show_histogram:
        for line in fill_histogram(mask, axis):
            print(line)
        print()
        print("  Press any key to start viewer...", end='', flush=True)
        msvcrt.getwch()

    idx = max(0, min(start, n_slices - 1))
    status_msg = ''

    while True:
        # Clamp viewport to array bounds
        vr = max(0, min(vr, n_rows - view_rows))
        vc = max(0, min(vc, n_cols - view_cols))
        r_min = vr;          r_max = vr + view_rows - 1
        c_min = vc;          c_max = vc + view_cols - 1

        sl    = get_slice(mask, axis, idx)
        stats = slice_stats(sl)
        lines = render_slice_ascii(
            sl, r_min, r_max, c_min, c_max,
            max_display_cols=max_display_cols,
            x_stride=x_stride,
            y_stride=y_stride,
        )

        # Clear screen (ANSI: cursor home + erase display)
        print('\033[H\033[2J', end='', flush=True)

        # Header
        status_str = f"  [{status_msg}]" if status_msg else ''
        print(f"  {seg_name}  |  {slice_ax}={idx}/{n_slices-1}"
              f"  |  filled={stats['filled']:,} ({stats['fill_pct']:.1f}%)"
              f"{status_str}")
        print(f"  viewport: {row_ax} [{r_min}:{r_max}]  {col_ax} [{c_min}:{c_max}]"
              f"  (slice shape {n_rows}x{n_cols})")
        print(f"  {col_ax} →")

        for li, line in enumerate(lines):
            phys_row = r_max - li * y_stride
            side = f"{row_ax}={phys_row:>4} |" if li == 0 else (
                   f"     {r_min:>4} |" if li == len(lines) - 1 else
                   f"           |")
            print(f"  {side} {line}")

        print()
        print("  ←→/NP=slice  ↑↓/WS=pan↑↓  AD=pan←→  F/B=±10  C=center  R=reset  Q=quit", end='', flush=True)

        status_msg = ''   # clear after it has been displayed for one frame

        # Read one keypress without waiting for Enter
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):
            # Extended key (arrows, page keys): read the scan code byte
            scan = msvcrt.getwch()
            cmd = {
                'M': 'next',   # →
                'K': 'prev',   # ←
                'H': 'w',      # ↑ → pan up
                'P': 's',      # ↓ → pan down
                'I': 'f',      # Page Up
                'Q': 'b',      # Page Down
                'G': 'r',      # Home → reset
            }.get(scan, '')
        elif ch == '\x03':     # Ctrl-C
            break
        else:
            cmd = ch.lower()

        if cmd in ('\r', '\n', 'n', 'next'):
            idx = min(idx + 1, n_slices - 1)
        elif cmd in ('p', 'prev'):
            idx = max(idx - 1, 0)
        elif cmd == 'f':
            idx = min(idx + 10, n_slices - 1)
        elif cmd == 'b':
            idx = max(idx - 10, 0)
        elif cmd == 'c':
            new_vr, new_vc = nearest_filled_center(
                get_slice(mask, axis, idx), vr, vc, view_rows, view_cols
            )
            if new_vr == vr and new_vc == vc:
                # Either already on content, or the slice is empty
                sl_check = get_slice(mask, axis, idx)
                if not sl_check.any():
                    status_msg = 'empty slice'
            vr, vc = new_vr, new_vc
        if cmd == 'w':
            vr += pan_step_r
        elif cmd == 's':
            vr -= pan_step_r
        elif cmd == 'a':
            vc -= pan_step_c
        elif cmd == 'd':
            vc += pan_step_c
        elif cmd == 'r':
            rc, cc = content_center(mask, axis)
            vr = rc - view_rows // 2
            vc = cc - view_cols // 2
        elif cmd == 'q':
            break

    print('\033[H\033[2J', end='', flush=True)
    print("Viewer closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASCII terminal viewer for .npz voxel masks (no display needed)."
    )
    parser.add_argument("npz_path", help="Path to the .npz mask file")
    parser.add_argument("--axis",      type=int, default=0, choices=[0, 1, 2],
                        help="Axis to slice along: 0=Z (default), 1=Y, 2=X")
    parser.add_argument("--start",     type=int, default=0,
                        help="First slice index (default=0)")
    parser.add_argument("--width",     type=int, default=100,
                        help="Max character columns per line (default=100)")
    parser.add_argument("--x-stride",  type=int, default=2,
                        help="Column subsampling for aspect ratio (default=2)")
    parser.add_argument("--y-stride",  type=int, default=1,
                        help="Row subsampling to reduce height (default=1, try 2)")
    parser.add_argument("--view-rows", type=int, default=DEFAULT_VIEW_ROWS,
                        help=f"Viewport height in voxels (default={DEFAULT_VIEW_ROWS})")
    parser.add_argument("--view-cols", type=int, default=DEFAULT_VIEW_COLS,
                        help=f"Viewport width in voxels before stride (default={DEFAULT_VIEW_COLS})")
    parser.add_argument("--histogram", action="store_true",
                        help="Show fill histogram before the viewer")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary + histogram then exit")
    args = parser.parse_args()

    if not Path(args.npz_path).exists():
        print(f"ERROR: file not found: {args.npz_path}")
        sys.exit(1)

    mask, bbox_min_vox, voxel_size, seg_id, seg_name = load_mask(args.npz_path)
    print_summary(mask, bbox_min_vox, voxel_size, seg_id, seg_name, args.axis)

    if args.summary_only:
        for line in fill_histogram(mask, args.axis):
            print(line)
        return

    run_viewer(
        mask, seg_name,
        axis=args.axis,
        start=args.start,
        max_display_cols=args.width,
        x_stride=args.x_stride,
        y_stride=args.y_stride,
        view_rows=args.view_rows,
        view_cols=args.view_cols,
        show_histogram=args.histogram,
    )


if __name__ == "__main__":
    main()
