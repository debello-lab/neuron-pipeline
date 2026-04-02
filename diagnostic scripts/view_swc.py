"""
SWC Skeleton Viewer

Loads an SWC file and renders the skeleton as a true interactive 3D plot
in the browser using Plotly. Rotate, pan, and zoom with the mouse.

Usage
-----
    python "diagnostic scripts/view_swc.py"                        # newest .swc in diag_output/
    python "diagnostic scripts/view_swc.py" path/to/file.swc
    python "diagnostic scripts/view_swc.py" file.swc --color-by radius
"""

import sys
import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# SWC compartment type colours and display names
_TYPE_COLOR = {
    0: '#aaaaaa',  # undefined
    1: '#e41a1c',  # soma
    2: '#377eb8',  # axon
    3: '#4daf4a',  # dendrite
    4: '#984ea3',  # apical dendrite
}
_TYPE_LABEL = {0: 'undefined', 1: 'soma', 2: 'axon', 3: 'dendrite', 4: 'apical'}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_swc(path: Path):
    """
    Parse an SWC file.

    Returns
    -------
    nodes : dict  {id -> {'x','y','z','r','type','parent'}}
    meta  : dict  {key -> value string} from # key: value header lines
    """
    meta  = {}
    nodes = {}

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                m = re.match(r'^#\s*([\w_]+)\s*:\s*(.+)$', line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid    = int(parts[0])
            stype  = int(parts[1])
            x, y, z, r = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            parent = int(parts[6])
            nodes[nid] = {'x': x, 'y': y, 'z': z, 'r': r, 'type': stype, 'parent': parent}

    return nodes, meta


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def _build_segments_by_type(nodes):
    """
    Group parent->child line segments by SWC type.

    Returns
    -------
    by_type : dict  { stype -> {'xs':[], 'ys':[], 'zs':[], 'radii':[], 'hover':[]} }
        xs/ys/zs contain interleaved (start, end, None) triples so a single
        Scatter3d trace draws all segments for that type.
    """
    by_type = {}

    for nid, node in nodes.items():
        pid = node['parent']
        if pid == -1 or pid not in nodes:
            continue
        p     = nodes[pid]
        stype = node['type']

        bucket = by_type.setdefault(stype, {'xs': [], 'ys': [], 'zs': [], 'radii': [], 'hover': []})
        bucket['xs']    += [p['x'],    node['x'],    None]
        bucket['ys']    += [p['y'],    node['y'],    None]
        bucket['zs']    += [p['z'],    node['z'],    None]
        bucket['radii'] += [node['r'], node['r'],    None]
        bucket['hover'] += [
            f"id {nid}  r={node['r']:.4f} µm",
            f"id {nid}  r={node['r']:.4f} µm",
            None,
        ]

    return by_type


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _title_str(meta, swc_path):
    parts = [Path(swc_path).name]
    if 'segment_id' in meta:
        parts.append(f"seg {meta['segment_id']}")
    if 'total_length_um' in meta:
        try:
            parts.append(f"{float(meta['total_length_um']):.1f} µm cable")
        except ValueError:
            pass
    if 'branches_pruned' in meta:
        parts.append(f"{meta['branches_pruned']} spurs pruned")
    return '  |  '.join(parts)


def plot_swc_type(nodes, meta, swc_path):
    """One Scatter3d trace per SWC compartment type, coloured distinctly."""
    by_type = _build_segments_by_type(nodes)
    traces  = []

    for stype in sorted(by_type):
        b     = by_type[stype]
        color = _TYPE_COLOR.get(stype, '#666666')
        label = _TYPE_LABEL.get(stype, f'type {stype}')
        traces.append(go.Scatter3d(
            x=b['xs'], y=b['ys'], z=b['zs'],
            mode='lines',
            line=dict(color=color, width=3),
            name=label,
            hovertext=b['hover'],
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)
    _apply_layout(fig, meta, swc_path)
    fig.show()


def plot_swc_radius(nodes, meta, swc_path):
    """Single trace with branch colour mapped to radius (plasma scale)."""
    by_type = _build_segments_by_type(nodes)

    xs, ys, zs, radii, hover = [], [], [], [], []
    for b in by_type.values():
        xs    += b['xs']
        ys    += b['ys']
        zs    += b['zs']
        radii += b['radii']
        hover += b['hover']

    # Replace None with NaN so the colorscale array stays aligned
    r_arr = np.array([r if r is not None else float('nan') for r in radii])

    trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(
            color=r_arr,
            colorscale='Plasma',
            width=3,
            colorbar=dict(title='Radius (µm)', thickness=15, len=0.6),
        ),
        hovertext=hover,
        hoverinfo='text',
        name='skeleton',
    )

    fig = go.Figure(data=[trace])
    _apply_layout(fig, meta, swc_path)
    fig.show()


def _apply_layout(fig, meta, swc_path):
    fig.update_layout(
        title=dict(text=_title_str(meta, swc_path), font=dict(size=14)),
        scene=dict(
            xaxis_title='X (µm)',
            yaxis_title='Y (µm)',
            zaxis_title='Z (µm)',
            aspectmode='data',          # preserve physical proportions
            bgcolor='#111111',
            xaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333', color='#cccccc'),
            yaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333', color='#cccccc'),
            zaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333', color='#cccccc'),
        ),
        paper_bgcolor='#0d0d0d',
        font=dict(color='#cccccc'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#444444', borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(nodes, meta, swc_path):
    print(f"\n{'=' * 60}")
    print(f"  {Path(swc_path).name}")
    print(f"{'=' * 60}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  samples (nodes): {len(nodes):,}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='View an SWC skeleton file in 3D.')
    parser.add_argument(
        'swc', nargs='?',
        help='Path to .swc file. Defaults to the newest file in diag_output/.',
    )
    parser.add_argument(
        '--color-by', choices=['type', 'radius'], default='type',
        help='Colour branches by SWC compartment type (default) or radius.',
    )
    args = parser.parse_args()

    if args.swc:
        swc_path = Path(args.swc)
    else:
        diag_dir  = Path(__file__).parent.parent / 'diag_output'
        swc_files = sorted(diag_dir.glob('**/*.swc'), key=lambda p: p.stat().st_mtime)
        if not swc_files:
            print(f"No .swc files found in {diag_dir}")
            sys.exit(1)
        swc_path = swc_files[-1]
        print(f"No file specified — loading newest: {swc_path.name}")

    if not swc_path.exists():
        print(f"File not found: {swc_path}")
        sys.exit(1)

    nodes, meta = parse_swc(swc_path)
    if not nodes:
        print("No samples parsed — is this a valid SWC file?")
        sys.exit(1)

    print_summary(nodes, meta, str(swc_path))

    if args.color_by == 'radius':
        plot_swc_radius(nodes, meta, str(swc_path))
    else:
        plot_swc_type(nodes, meta, str(swc_path))


if __name__ == '__main__':
    main()
