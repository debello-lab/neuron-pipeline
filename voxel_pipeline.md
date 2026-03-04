# Voxel Mask -> Skeleton -> SWC (Arbor-ready) Export Plan

This is a practical pipeline for generating **`.swc` morphology files** from VAST segmentation data **without** going through `.obj` meshes.

It is designed to be robust to:
- **“specks”** caused by accidental annotation clicks
- **messy internal mesh artifacts** (irrelevant in voxel space)

---

## Why not OBJ -> SWC?

- `.obj` is a **surface**.
- `.swc` is a **tree** of centerline samples with **parent pointers** and **radii**.

So OBJ -> SWC is not a file conversion. It requires **skeletonization** and **graph extraction**.

Because you have access to voxel data (via VAST A.7 transfer functions), the most reliable path is:

> **voxel mask -> clean -> skeletonize -> graph -> SWC**

---

## What you already have in `extract_surfaces.py`

Your surface extraction code already does the critical setup for skeletonization:

- pulls a segmentation block from VAST using `get_seg_image_rle_decoded(...)`
- isolates a single segment via `set_seg_translation(...)`
- builds a **binary mask**: `binary_volume = (seg_image == segment_id)`

That `binary_volume` is exactly the input skeletonization needs.

---

## High-level pipeline

### 1) Export a **binary voxel mask**
Instead of running marching cubes immediately, return the mask:

- `mask` (3D bool)
- `origin` / bbox offsets
- voxel sizes (including mip scaling)

### 2) Clean the mask (prevents skeleton corruption)
**Minimum recommended cleaning:**
1. Remove tiny disconnected components (“specks”)  
2. Optionally fill small holes  
3. Spur pruning can happen after skeletonization

### 3) Skeletonize in voxel space
Use a 3D thinning/medial-axis skeletonizer (baseline: `skeletonize_3d`).

### 4) Compute radius along skeleton
Use an Euclidean distance transform (EDT) on the cleaned mask.

### 5) Convert skeleton graph -> SWC tree
Skeleton output is a **graph** (voxels connected by adjacency).  
SWC requires a **rooted tree** with ordered node IDs.

### 6) Write SWC
Assign IDs so **parent_id < id** and output:
`id, tag, x, y, z, radius, parent_id`

---

## Step-by-step details

## 1) Add a “mask export” path

Create a method like:

- `_extract_full_volume_mask(segment_id, ...) -> (mask, bbox_min_xyz_vox, voxel_size_um)`

This should share the exact bbox/mip logic you already use for meshes, but stop after:

- `binary_volume = (seg_image == segment_id)`

Return:
- `mask`: `bool[z, y, x]` (or whatever ordering you use)
- `bbox_min_vox`: the min x/y/z used to fetch the volume
- `voxel_size_um`: (sx, sy, sz) in microns, including mip scaling

---

## 2) Clean the mask (remove specks)

**Goal:** stop noise components from creating fake skeleton branches.

### Minimal cleaning function (conceptual)

```python
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

def clean_mask(mask, min_component_voxels=5000, keep_largest=True):
    lab = label(mask, connectivity=1)
    if lab.max() == 0:
        return mask

    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background

    if keep_largest:
        keep = counts.argmax()
        cleaned = (lab == keep)
    else:
        keep_ids = np.where(counts >= min_component_voxels)[0]
        cleaned = np.isin(lab, keep_ids)

    cleaned = binary_fill_holes(cleaned)
    return cleaned
```

**Rule of thumb:** “keep largest component” is usually the safest default for individual segment exports.

---

## 3) Skeletonize the volume

Baseline skeletonization:

```python
from skimage.morphology import skeletonize_3d

skel = skeletonize_3d(cleaned_mask.astype(np.uint8)) > 0
```

This produces a **1-voxel-thick** skeleton.

---

## 4) Radius via distance transform (EDT)

### Why EDT?
The EDT gives, at each voxel, distance to the boundary.  
Along the skeleton, that distance is a reasonable **local radius estimate** (also useful for bouton detection).

### Use physical spacing (anisotropic voxels)

```python
from scipy.ndimage import distance_transform_edt

# sx, sy, sz are voxel sizes in microns (include mip scaling!)
dist_um = distance_transform_edt(cleaned_mask, sampling=(sz, sy, sx))
# note: sampling order must match mask axis order
```

Then for each skeleton point:
- `radius_um = dist_um[z, y, x]`

---

## 5) Skeleton voxels -> SWC tree

Skeletons are graphs, not trees. You must:

1. Build adjacency (26-neighborhood typically)
2. Identify:
   - endpoints: degree == 1
   - branchpoints: degree >= 3
3. Compress the voxel skeleton into a graph:
   - nodes = endpoints + branchpoints
   - edges = chains between them
4. Choose a root:
   - if soma exists: root at soma
   - else: pick an endpoint on the graph diameter
5. Traverse from root outward and emit SWC nodes **in parent-before-child order**.

### Spur pruning (recommended)
After graph extraction (or during), remove short leaf branches:
- prune any leaf edge shorter than **L microns** (typical: ~0.5–2 um depending on voxel size)

This removes leftover noise branches even after component filtering.

---

## 6) Writing SWC

Arbor’s SWC checks require:
- unique IDs
- `parent_id < id`
- parent refers to an existing node ID

Emit nodes in traversal order:
- root first (`parent_id = -1`)
- then children

### Tags
SWC reserved tags:
- `1`: soma
- `2`: axon
- `3`: basal dendrite
- `4`: apical dendrite

If you don’t yet classify compartments, you can set everything to `0` or `3` temporarily and refine later.
(But if you include a soma, Arbor’s default interpretation expects ≥2 soma samples; if you don’t have soma data, you can omit soma entirely.)

---

## Integration plan (minimal refactor)

### A) Keep mesh export as-is
Your `.obj` pipeline stays for visualization.

### B) Add SWC export alongside it
Implement a new entry point, e.g.:

- `extract_segment_swc(segment_id, mip=..., padding=..., out_path=...)`

Internals:
1. `mask, bbox_min, voxel_size = _extract_full_volume_mask(...)`
2. `mask = clean_mask(mask)`
3. `skel = skeletonize_3d(mask)`
4. `dist_um = distance_transform_edt(mask, sampling=...)`
5. `graph = skeleton_to_graph(skel)`
6. `tree = graph_to_tree(graph, root=...)`
7. `swc = tree_to_swc(tree, dist_um, bbox_min, voxel_size)`
8. write file

---

## Handling huge objects (block extraction note)

Your current surface extractor supports block extraction for large segments.

Skeletonization across blocks is harder (skeleton continuity across seams).

A pragmatic skeleton strategy:
- skeletonize at a **coarser mip level** so the full bbox fits in memory
- optionally resample radii from higher resolution around the final skeleton points

This avoids complicated “stitch skeleton across blocks” logic.

---

## Quality report (recommended)

For each exported neuron, log:
- number of connected components before/after cleaning
- number of spur branches removed
- total cable length (before/after)
- min/median/max radius along skeleton
- # branchpoints and endpoints

This makes the pipeline defensible and debuggable.

---

## Where bouton + synapse markers fit (not in SWC)

SWC stores morphology only.  
Boutons and synapses should be exported into a **sidecar feature/connectivity table**:

- bouton marker -> nearest SWC segment (segment_id + fraction)
- synapse marker -> nearest SWC segment on presynaptic skeleton **and** postsynaptic skeleton
- edge list row: `(pre_cell, pre_loc) -> (post_cell, post_loc)`

This is the bridge into Arbor’s “interconnectivity” model.
