"""
trace_synapse.py  --  End-to-end diagnostic for a single synapse.

Usage:
    python trace_synapse.py <synapse_name>
    e.g.:  python trace_synapse.py A1B2P1S1

For the named synapse this script traces every coordinate transformation
step and reports the final cable-mapping QC result for both the presynaptic
(axon) side and the postsynaptic (dendrite) side.

Expected output structure
-------------------------
[SYNAPSE SIDE]  A1B2P1S1  ->  axon A1
  Mask  : shape=(Z,Y,X)  bbox_min=(x,y,z)  voxel_size=(sx,sy,sz)  filled=N voxels
  Step 1 local  centroid (voxels): x=...  y=...  z=...
  Step 2 global centroid (voxels): x=...  y=...  z=...
  Step 3 physical centroid  (µm) : x=...  y=...  z=...
  Skeleton: coord_frame=physical_um_xyz  nodes=N
    bounds X [lo, hi]  Y [lo, hi]  Z [lo, hi]  µm
  KDTree query:
    edge_u=N  edge_v=N  arc_frac=0.XXXXXX  dist=X.XX µm  qc=ok
    swc_node_u=N  swc_node_v=N

[POST-SYN SIDE]  A1B2P1  ->  post-syn A1B2P1
  ... (same structure)

RESULT: PASS | WARN | FAIL
"""

import sys
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from vastpy.control.exporting import VASTControlClass
from neuron_pipeline.stages.segment_classifier import SegmentClassifier
from neuron_pipeline.stages.extract_surfaces import SegmentSurfaceExtractor
from neuron_pipeline.stages.voxel_cleaning import VoxelCleaner
from neuron_pipeline.stages.skeletonization import SkeletonExtractor, SWCWriter
from neuron_pipeline.stages.centroid_mapper import _TreeIndex

# ── configuration ────────────────────────────────────────────────────────────
VAST_HOST     = "127.0.0.1"
VAST_PORT     = 22081
MIPLEVEL_SKEL = 1       # MIP level for skeleton extraction (axon / post-syn)
MIPLEVEL_CEN  = 1       # MIP level for centroid extraction (synapse, full res)
PADDING       = 2
SPUR_UM       = 2.0
OK_UM         = 2.0
WARN_UM       = 5.0
OUTPUT_DIR    = Path(__file__).parent.parent / "diag_output"

# Synapse name pattern:  A<n>B<n>P<n>S<n>
_SYNAPSE_RE = re.compile(r'^(A\d+)(B\d+)(P\d+)(S\d+)$')

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("trace_synapse")


# ── helpers ───────────────────────────────────────────────────────────────────

def _derive_names(synapse_name: str):
    """Return (axon_name, bouton_name, post_syn_name) or raise."""
    m = _SYNAPSE_RE.match(synapse_name)
    if not m:
        raise ValueError(
            f"'{synapse_name}' does not match synapse pattern A<n>B<n>P<n>S<n>"
        )
    axon_name    = m.group(1)
    bouton_name  = m.group(1) + m.group(2)
    post_syn_name = m.group(1) + m.group(2) + m.group(3)
    return axon_name, bouton_name, post_syn_name


def _extract_mask(extractor, seg_id, seg_name, miplevel, label):
    """Extract and describe a voxel mask. Returns (mask, bbox_min, voxel_size)."""
    print(f"\n  Extracting mask for {label} (seg_id={seg_id}, MIP {miplevel})...")
    mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
        segment_id=seg_id,
        miplevel=miplevel,
        padding=PADDING,
    )
    if mask is None or bbox_min is None or voxel_size is None:
        raise RuntimeError(f"Voxel extraction failed for {seg_name} (seg_id={seg_id})")
    print(
        f"  Mask  : shape={mask.shape}  "
        f"bbox_min={bbox_min}  voxel_size={tuple(round(v,4) for v in voxel_size)}  "
        f"filled={int(mask.sum()):,} voxels"
    )
    return mask, bbox_min, voxel_size


def _trace_centroid(mask, bbox_min_vox, voxel_size_um):
    """Reproduce _centroid_from_voxels step-by-step, printing each stage."""
    z_idx, y_idx, x_idx = np.where(mask)

    # Step 1 — local (relative to sub-volume)
    local_cx = x_idx.mean()
    local_cy = y_idx.mean()
    local_cz = z_idx.mean()
    print(
        f"  Step 1 local  centroid (voxels): "
        f"x={local_cx:.3f}  y={local_cy:.3f}  z={local_cz:.3f}"
    )

    # Step 2 — global (add bbox origin)
    global_cx = local_cx + bbox_min_vox[0]
    global_cy = local_cy + bbox_min_vox[1]
    global_cz = local_cz + bbox_min_vox[2]
    print(
        f"  Step 2 global centroid (voxels): "
        f"x={global_cx:.3f}  y={global_cy:.3f}  z={global_cz:.3f}"
    )

    # Step 3 — physical µm (corner-of-voxel convention)
    cx_um = global_cx * voxel_size_um[0]
    cy_um = global_cy * voxel_size_um[1]
    cz_um = global_cz * voxel_size_um[2]
    print(
        f"  Step 3 physical centroid  (µm) : "
        f"x={cx_um:.4f}  y={cy_um:.4f}  z={cz_um:.4f}"
    )
    return cx_um, cy_um, cz_um


def _get_or_build_skeleton(extractor, cleaner, skel_extractor, writer,
                            seg_id, seg_name, miplevel, label):
    """
    Load a cached SWC from diag_output/swc/cell_<name>.swc if present.
    Otherwise run a fresh Phase 1 extraction so swc_id stamps are written.
    Returns the skeleton tree (nx.DiGraph).
    """
    swc_dir  = OUTPUT_DIR / "swc"
    swc_path = swc_dir / f"cell_{seg_name}.swc"

    if swc_path.exists():
        print(
            f"  Skeleton: loaded from cache {swc_path}\n"
            f"  NOTE: swc_node_u/v will be None (swc_id not stamped on cached trees).\n"
            f"  Re-run without cached SWC to get SWC row IDs."
        )
        # Re-extract fresh so we get a stamped tree for this session
        # (a cached SWC on disk does not give us a Python tree object with swc_id)

    print(f"\n  Building skeleton for {label} (seg_id={seg_id}, MIP {miplevel})...")
    mask, bbox_min, voxel_size, _ = extractor.extract_segment_voxel(
        segment_id=seg_id, miplevel=miplevel, padding=PADDING,
    )
    if mask is None:
        raise RuntimeError(f"Mask extraction failed for {seg_name}")

    cleaned, _ = cleaner.clean_mask(
        mask, keep_largest_only=False, fill_holes=True, smooth_iterations=0,
    )
    tree, skel_stats = skel_extractor.extract_skeleton(
        mask=cleaned,
        voxel_size_um=voxel_size,
        bbox_min_vox=bbox_min,
        spur_length_um=SPUR_UM,
    )
    if tree.number_of_nodes() == 0:
        raise RuntimeError(f"Empty skeleton for {seg_name}")

    # Write SWC — this stamps swc_id onto every node via SWCWriter
    swc_dir.mkdir(parents=True, exist_ok=True)
    writer.write_swc(
        tree=tree,
        output_path=str(swc_path),
        metadata={
            "segment_id":   seg_id,
            "segment_name": seg_name,
            "miplevel":     miplevel,
            "bbox_min_vox": str(bbox_min),
            "voxel_size_um": str(voxel_size),
            "coord_frame":  "physical_um_xyz",
        },
    )

    # Print skeleton summary
    positions = [tree.nodes[n]['pos'] for n in tree.nodes() if 'pos' in tree.nodes[n]]
    pts = np.array(positions)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    print(
        f"  Skeleton: coord_frame={tree.graph.get('coord_frame', 'UNSET')}  "
        f"nodes={tree.number_of_nodes()}  edges={tree.number_of_edges()}\n"
        f"    bounds X [{lo[0]:.1f}, {hi[0]:.1f}]  "
        f"Y [{lo[1]:.1f}, {hi[1]:.1f}]  "
        f"Z [{lo[2]:.1f}, {hi[2]:.1f}]  µm"
    )
    return tree


def _trace_query(tree, cx_um, cy_um, cz_um):
    """Run KDTree query and print all mapping fields. Returns qc_flag."""
    idx = _TreeIndex(tree)
    result = idx.query(np.array([cx_um, cy_um, cz_um], dtype=np.float64))
    if result is None:
        print("  KDTree query: FAILED (empty tree index)")
        return 'suspicious'

    u, v, arc_frac, nearest_pos, dist = result

    if dist < OK_UM:
        qc_flag = 'ok'
    elif dist < WARN_UM:
        qc_flag = 'warn'
    else:
        qc_flag = 'suspicious'

    swc_u = tree.nodes[u].get('swc_id')
    swc_v = tree.nodes[v].get('swc_id')
    swc_u_str = str(swc_u) if swc_u is not None else "None (write_swc not called)"
    swc_v_str = str(swc_v) if swc_v is not None else "None (write_swc not called)"

    print(
        f"  KDTree query:\n"
        f"    edge_u={u}  edge_v={v}  arc_frac={arc_frac:.6f}  "
        f"dist={dist:.4f} µm  qc={qc_flag}\n"
        f"    nearest (µm): x={nearest_pos[0]:.4f}  "
        f"y={nearest_pos[1]:.4f}  z={nearest_pos[2]:.4f}\n"
        f"    swc_node_u={swc_u_str}  swc_node_v={swc_v_str}"
    )
    return qc_flag


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_synapse.py <synapse_name>")
        print("  e.g. python trace_synapse.py A1B2P1S1")
        sys.exit(1)

    synapse_name = sys.argv[1]
    axon_name, bouton_name, post_syn_name = _derive_names(synapse_name)

    print(f"\nTracing synapse: {synapse_name}")
    print(f"  Derived names — axon: {axon_name}  bouton: {bouton_name}  post-syn: {post_syn_name}")


    classifier = SegmentClassifier()
    registry   = classifier.classify_segments()

    for name in (synapse_name, bouton_name, axon_name, post_syn_name):
        if name not in registry.segments:
            print(f"ERROR: '{name}' not found in segment registry. Check the name.")
            sys.exit(1)

    syn_seg_id      = registry.segments[synapse_name].seg_id
    bouton_seg_id   = registry.segments[bouton_name].seg_id
    axon_seg_id     = registry.segments[axon_name].seg_id
    post_syn_seg_id = registry.segments[post_syn_name].seg_id

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor     = SegmentSurfaceExtractor(classifier.vast, str(OUTPUT_DIR))
    cleaner       = VoxelCleaner()
    skel_extractor = SkeletonExtractor()
    writer        = SWCWriter()

    # ── PRE-SYNAPTIC (BOUTON -> AXON) ─────────────────────────────────────────
    # The pipeline uses the BOUTON centroid for the axon-side cable location,
    # not the synapse centroid. The synapse sits at the cleft and can be several
    # µm from the axon cable — that is expected biology, not a coordinate error.
    print(f"\n{'='*60}")
    print(f"[PRE-SYNAPTIC SIDE]  bouton {bouton_name}  ->  axon {axon_name}")
    print(f"  (The pipeline maps the BOUTON centroid onto the axon cable.)")
    print(f"  (The synapse centroid is shown for reference only.)")
    print(f"{'='*60}")

    bouton_mask, bouton_bbox, bouton_vsize = _extract_mask(
        extractor, bouton_seg_id, bouton_name, MIPLEVEL_CEN, bouton_name
    )
    bouton_cx, bouton_cy, bouton_cz = _trace_centroid(bouton_mask, bouton_bbox, bouton_vsize)

    axon_tree = _get_or_build_skeleton(
        extractor, cleaner, skel_extractor, writer,
        axon_seg_id, axon_name, MIPLEVEL_SKEL, axon_name
    )
    print(f"\n  --- Bouton centroid -> axon (the mapping the pipeline uses) ---")
    qc_pre = _trace_query(axon_tree, bouton_cx, bouton_cy, bouton_cz)

    print(f"\n  --- Synapse centroid -> axon (reference only; expected to be further) ---")
    syn_mask, syn_bbox, syn_vsize = _extract_mask(
        extractor, syn_seg_id, synapse_name, MIPLEVEL_CEN, synapse_name
    )
    syn_cx, syn_cy, syn_cz = _trace_centroid(syn_mask, syn_bbox, syn_vsize)
    _trace_query(axon_tree, syn_cx, syn_cy, syn_cz)

    # ── POST-SYN SIDE ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[POST-SYN SIDE]  {post_syn_name}  ->  dendrite {post_syn_name}")
    print(f"{'='*60}")

    ps_mask, ps_bbox, ps_vsize = _extract_mask(
        extractor, post_syn_seg_id, post_syn_name, MIPLEVEL_SKEL, post_syn_name
    )
    ps_cx, ps_cy, ps_cz = _trace_centroid(ps_mask, ps_bbox, ps_vsize)

    post_tree = _get_or_build_skeleton(
        extractor, cleaner, skel_extractor, writer,
        post_syn_seg_id, post_syn_name, MIPLEVEL_SKEL, post_syn_name
    )
    qc_post = _trace_query(post_tree, ps_cx, ps_cy, ps_cz)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  Pre  (bouton -> axon)   qc = {qc_pre}  [pipeline-relevant]")
    print(f"  Post (post-syn -> dendrite)  qc = {qc_post}  [pipeline-relevant]")

    flags = {qc_pre, qc_post}
    if 'suspicious' in flags:
        verdict = "FAIL"
        reason  = "one or more distances exceed warn threshold — likely coordinate mismatch or sparse skeleton"
    elif 'warn' in flags:
        verdict = "WARN"
        reason  = "one or more distances in marginal range — worth inspecting"
    else:
        verdict = "PASS"
        reason  = "both mappings within ok threshold"

    print(f"\n  {verdict}: {reason}")
    print()

    # Load the SWC
    axon_swc = pd.read_csv("diag_output/swc/cell_A1.swc", 
                        sep=' ', comment='#', 
                        names=['id','type','x','y','z','r','parent'])

    # Plot the skeleton
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(axon_swc['x'], axon_swc['y'], axon_swc['z'], 'b-', linewidth=0.5)

    # Plot the synapse centroid
    syn_x, syn_y, syn_z = 8.39, 18.83, 43.39  # From the script output
    ax.scatter([syn_x], [syn_y], [syn_z], c='r', s=100, marker='o')

    # Plot the nearest cable point
    ax.scatter([8.39], [18.83], [43.39], c='g', s=50, marker='x')

    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Z (µm)')
    plt.title('Synapse -> Axon Mapping')
    plt.show()


if __name__ == "__main__":
    main()
