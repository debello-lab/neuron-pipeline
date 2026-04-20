# Architecture

This document describes the architecture of **ms-neuron-pipeline**, a Python project that (1) provides an open-source interface for exporting data from **VAST Lite** without MATLAB, and (2) runs a multi-stage pipeline to produce downstream morphology artifacts that can be used in neuron digital simulation software.

The project is intentionally split into two layers:

- **`vastpy/` (library layer):** a reusable Python API for interacting with VAST Lite (remote control/export).
- **`neuron_pipeline/` (application layer):** the master's project pipeline that orchestrates export, cleaning, skeletonization, and analysis steps.

This separation makes `vastpy` independently reusable for other researchers while keeping the project code focused on the end-to-end workflow and evaluation.

---

## Goals and constraints

### Goals
- Export 3D segment representations from VAST Lite (voxel masks and/or meshes) in a way that is scriptable and reproducible.
- Convert exported data into structures usable by downstream tools and workflows:
  - skeleton graphs and SWC morphologies
  - marker/centroid tables for boutons/synapses/contacts
  - (planned) mapping markers to SWC nodes for annotation/connectivity analyses
- Provide a workflow that reduces reliance on GUI/manual steps and minimizes MATLAB dependence.

### Constraints / assumptions
- **VAST Lite must be running** and have the target project/segmentation loaded before the pipeline executes.
- The pipeline assumes the VAST remote-control/export interface is reachable (host/port configuration is handled by the VAST control classes).
- Coordinate conventions and voxel-to-world conversions must be consistent across phases (e.g., handling of voxel centers and offsets).

---

## Repository structure
```
ms-neuron-pipeline/
+--- src/
|   +--- vastpy/
|   | +--- control/
|   |   | +--- exporting.py 
|   |   | +--- VASTControlClass.py
|   +--- neuron_pipeline/
|   | +--- main_pipeline.py
|   | +--- stages/
|   |   | +--- centroid_extraction.py
|   |   | +--- centroid_mapper.py
|   |   | +--- connectivity_builder.py
|   |   | +--- extract_surfaces.py
|   |   | +--- segment_classifier.py
|   |   | +--- skeletonization.py
|   |   | +--- voxel_cleaning.py
+--- scripts/
|   +--- run_pipeline.py
+--- docs/
|   +--- architecture.md
|   +--- setup.md
+--- LICENSE
+--- pyproject.toml
+--- README.md
```


---

## Data flow overview

1. **Phase 0 -- Classification**
    - Identify and register segments of interest by type (e.g., AXON, POST_SYN, BOUTON, SYNAPSE/CONTACT).
    - Output: a `SegmentRegistry` that becomes the shared “truth” for subsequent phases.

2. **Phase 1 -- Extraction, Skeletonization + SWC**
    - Export segment voxel masks or surfaces from VAST Lite.
    - Clean voxel representations to remove small components / artifacts.
    - Skeletonize the cleaned representation and convert it into an SWC tree.
    - Output: SWC files (and any intermediate graphs/logs needed for QC).

3. **Phase 2 -- Centroid extraction for markers**
    - Export marker voxel masks (boutons/synapses/contacts) at an appropriate mip level.
    - Compute centroids (binary centroid: equal weight per voxel).
    - Output: `centroids.csv` — one row per BOUTON/SYNAPSE/CONTACT, used as input to Phase 3.

4. **Phase 3 -- Marker-to-SWC mapping**
    - For each centroid, find the nearest point on the parent skeleton cable (KDTree over all structural nodes + edge intermediate samples).
    - Record the nearest edge (edge_u, edge_v), arc fraction along that edge, physical position, and Euclidean distance from centroid to cable.
    - Output: `cable_mappings.csv` — consumed by Phase 4.

5. **Phase 4 -- Connectivity export**
    - Join SYNAPSE and CONTACT annotations with their pre- and post-synaptic cable locations from Phase 3.
    - Build full edge list: `(pre_cell, pre_edge, pre_arc_frac, post_cell, post_edge, post_arc_frac, synapse/contact identifiers)`.
    - Write Arbor-ready network recipe (JSON) and a human-readable connectivity report.
    - Output: `connectivity.csv`, `connectivity_summary.csv`, `connectivity_report.txt`, `arbor_recipe.json`.

6. **Phase 5 -- Arbor Integration (planned)**
    - Validate simulation runs against the generated recipe.

The main artifacts written to disk are under `./vast_export/`:

| File | Phase | Description |
|------|-------|-------------|
| `swc/cell_<name>.swc` | 1 | Skeleton morphology in SWC format |
| `centroids.csv` | 2 | Centroid positions for BOUTON/SYNAPSE/CONTACT segments |
| `cable_mappings.csv` | 3 | Centroids mapped to nearest skeleton cable location |
| `connectivity.csv` | 4 | Full synapse/contact edge list |
| `connectivity_summary.csv` | 4 | Per-(axon, post_syn) pair synapse counts and distance stats |
| `connectivity_report.txt` | 4 | Human-readable connectivity summary with QC warnings |
| `arbor_recipe.json` | 4 | Arbor-ready network description |
| `review_queue.csv` | 1 | Segments flagged for manual inspection (high closing fraction or residual fragmentation) |
| `logs/pipeline_<timestamp>.log` | all | Full pipeline log (DEBUG level) |

`./vast_export/` is automatically generated by the pipeline if it does not exist.

---

## Key components

### 1) `vastpy` (library layer)

The `vastpy` layer provides a Python interface to drive VAST Lite exports without MATLAB scripts.

**Capabilities:**
- Connect to the VAST remote control interface.
- Request/export:
  - voxel masks (for skeletonization and marker extraction)
  - meshes (for visualization or surface-based workflows)
  - metadata needed for coordinate conversion and alignment (voxel size, offsets, bounding boxes, mip level)
- Manipulate VAST Lite software.


---

### 2) `SegmentClassifier` + `SegmentRegistry` (Phase 0)

These two classes decide what segments to process and how to treat them with a heuristic classification process using regex.

Classification defines the downstream branching logic (e.g., which segments get SWCs vs centroids only).

**Responsibilities:**
- Establish the set of target segments.
- Assign each segment a type (axon, post-synaptic, bouton, synapse/contact markers, etc.).
- Provide a central `SegmentRegistry` object used by later phases.

---

### 3) `SegmentSurfaceExtractor` (export surface/voxels)

The `SegmentSurfaceExtractor` exports a segment representation from VAST in the format needed by downstream steps.

**Responsibilities:**
- Export either:
  - **mesh** (`.obj/.mtl`) for visualization / optional mesh cleaning workflows
  - **voxel mask** (3D boolean array or serialized form) for skeletonization and centroid extraction
- Save intermediate results into `vast_export/meshes`

---

### 4) `VoxelCleaner` (voxel cleaning stage)

Improves voxel representations before skeletonization/centroid computation. Can not fix superbly imperfect segmentations.

**Responsibilities:**
- Remove tiny disconnected components.
- Fill small holes when appropriate (with constraints to avoid topology-breaking changes).
- (Optional) spur pruning / smoothing steps.

**Design note:** voxel cleaning is a distinct stage to allow controlled experiments (e.g., evaluate effect on skeleton quality).

---

### 5) `SkeletonExtractor` + `SWCWriter` (Phase 1)

Converts a cleaned voxel mask to a skeleton representation and save it in SWC format.

**Responsibilities:**
- Skeletonize 3D voxel masks (Lee method).
- Represent the skeleton as a graph (typically voxel-adjacency-based) internally.
- Convert graph to an SWC tree:
  - ensure a single root
  - maintain parent-child structure
  - assign node coordinates and radii 
- Write SWC to disk.

**Important constraint:** coordinate conventions must match those used by centroid extraction (see coordinate section below).

---

### 6) `CentroidExtractor` + `CentroidTable` (Phase 2)

Computes marker centroids for small structures (boutons/synapses/contacts).

**Responsibilities:**
- Export voxel masks for marker segments (often at MIP 0 for small structures).
- Compute binary centroids (each voxel contributes equally; not intensity-weighted).
- Save results to a structured table (CSV).

---

### 7) `CentroidMapper` + `CableMappingTable` (Phase 3)

Maps each centroid to the nearest point on its parent skeleton cable.

**Responsibilities:**
- Build a KDTree from all skeleton sample positions (structural nodes + intermediate edge samples).
- Query each centroid and resolve the hit to a `(edge_u, edge_v, arc_fraction)` location.
- Flag mappings with large distances (`qc_distance_flag`: `ok` / `warn` / `suspicious`).

---

### 8) `ConnectivityBuilder` + `ConnectivityOutput` (Phase 4)

Joins annotation wiring from the registry with Phase 1/3 cable locations.

**Responsibilities:**
- Look up pre- and post-synaptic cable positions from the mapping table.
- Write `connectivity.csv` and `arbor_recipe.json`.
- Write `connectivity_summary.csv` (per-pair synapse counts, distance stats, QC tallies) and `connectivity_report.txt` (human-readable, includes missing-connection warnings).

---

## CSV output reference

All coordinates are in physical micrometres (µm), XYZ axis order, center-of-voxel convention
(`coord_frame = 'physical_um_xyz_center'`).  Every CSV that carries coordinates also carries a
`coord_frame` column so Phase 3 can assert consistency before building KDTrees.

---

### `centroids.csv` (Phase 2)

One row per BOUTON, SYNAPSE, or CONTACT segment.

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Segment name (e.g. `A1B2`) |
| `seg_id` | int | VAST segment ID |
| `role` | str | `BOUTON` / `SYNAPSE` / `CONTACT` |
| `cx_um` | float | Centroid X in µm |
| `cy_um` | float | Centroid Y in µm |
| `cz_um` | float | Centroid Z in µm |
| `method` | str | How the centroid was computed: `voxel` (from mask), `bbox_fallback` (from bounding-box centre), `anchor` (VAST anchor point, last resort) |
| `coord_frame` | str | Always `physical_um_xyz_center` |

**Interpreting `method`:** `voxel` entries are the most accurate.  `bbox_fallback` and `anchor`
entries are approximate and should be treated with lower confidence when mapping to skeleton cables.

---

### `cable_mappings.csv` (Phase 3)

One row per centroid, recording its nearest point on the parent skeleton.

| Column | Type | Description |
|--------|------|-------------|
| `centroid_name` | str | Matches `name` in `centroids.csv` |
| `centroid_role` | str | `BOUTON` / `SYNAPSE` / `CONTACT` |
| `cx_um`, `cy_um`, `cz_um` | float | Centroid position (µm) |
| `cell_name` | str | Parent skeleton cell (e.g. `A1`) |
| `cell_role` | str | `AXON` or `POST_SYN` |
| `edge_u`, `edge_v` | int | Graph node IDs of the two endpoints of the nearest skeleton edge |
| `arc_fraction` | float | Position along edge: 0.0 = at node u, 1.0 = at node v |
| `nearest_x_um`, `nearest_y_um`, `nearest_z_um` | float | Physical position of the nearest cable point (µm) |
| `distance_um` | float | Euclidean distance from centroid to nearest cable point (µm) |
| `coord_frame` | str | Always `physical_um_xyz_center` |
| `qc_distance_flag` | str | `ok` (< 2 µm), `warn` (2–5 µm), `suspicious` (≥ 5 µm) |
| `swc_node_u`, `swc_node_v` | int/empty | SWC row IDs corresponding to `edge_u` / `edge_v`; empty when not available |

**Interpreting `arc_fraction`:** Use it with `edge_u` / `edge_v` to locate the synapse on the
cable for Arbor input.  The physical position is pre-computed as `nearest_*_um`.  The
`qc_distance_flag` is the primary quality indicator; `suspicious` rows should be reviewed before
using them in simulations.

---

### `connectivity.csv` (Phase 4)

One row per confirmed synapse (connection_type = `synapse`) or putative contact (`contact`).

| Column | Type | Description |
|--------|------|-------------|
| `connection_type` | str | `synapse` or `contact` |
| `pre_cell` | str | Presynaptic axon cell name |
| `pre_swc` | str | Path to the axon SWC file |
| `pre_edge_u`, `pre_edge_v` | int | Graph node IDs of the nearest axon edge |
| `pre_arc_frac` | float | Arc fraction along the axon edge (0–1) |
| `pre_nearest_x_um`, `pre_nearest_y_um`, `pre_nearest_z_um` | float | Axon cable position (µm) |
| `distance_pre_um` | float | Distance from BOUTON centroid to axon cable (µm) |
| `post_cell` | str | Postsynaptic cell name (empty for contacts) |
| `post_swc` | str | Path to the post-synaptic SWC file (empty for contacts) |
| `post_edge_u`, `post_edge_v` | int | Graph node IDs of the nearest post-synaptic edge |
| `post_arc_frac` | float | Arc fraction along the post-synaptic edge |
| `post_nearest_x_um`, `post_nearest_y_um`, `post_nearest_z_um` | float | Post-synaptic cable position (µm) |
| `distance_post_um` | float | Distance from SYNAPSE centroid to post-synaptic cable (µm); -1 if unresolved |
| `synapse_name`, `synapse_seg_id` | str/int | SYNAPSE annotation identifier |
| `bouton_name`, `bouton_seg_id` | str/int | BOUTON annotation identifier |
| `pre_qc_flag` | str | `ok` / `warn` / `suspicious` for the presynaptic mapping |
| `pre_swc_u`, `pre_swc_v` | int/empty | SWC row IDs of `pre_edge_u` / `pre_edge_v` |
| `post_qc_flag` | str | `ok` / `warn` / `suspicious` for the postsynaptic mapping |
| `post_swc_u`, `post_swc_v` | int/empty | SWC row IDs of `post_edge_u` / `post_edge_v` |

**Reading this file:** Filter to `connection_type = synapse` for confirmed connections.
`contact` rows represent co-apposed segments without a confirmed SYNAPSE annotation.
Rows where either `pre_qc_flag` or `post_qc_flag` is `suspicious` should be verified before
inclusion in a simulation.

---

### `connectivity_summary.csv` (Phase 4)

One row per (axon, post_syn) pair — a compact overview of connection strength and mapping quality.

| Column | Description |
|--------|-------------|
| `axon`, `post_syn` | Paired cell names |
| `n_synapses` | Number of confirmed synapses between the pair |
| `bouton_names` | Semicolon-separated BOUTON names |
| `synapse_names` | Semicolon-separated SYNAPSE names |
| `pre_dist_mean_um`, `pre_dist_min_um`, `pre_dist_max_um` | Distance stats for axon-side mappings |
| `post_dist_mean_um`, `post_dist_min_um`, `post_dist_max_um` | Distance stats for post-synaptic-side mappings (`n/a` if unresolved) |
| `pre_qc_ok`, `pre_qc_warn`, `pre_qc_suspicious` | QC flag counts for the axon side |
| `post_qc_ok`, `post_qc_warn`, `post_qc_suspicious` | QC flag counts for the post-synaptic side |

---

### `review_queue.csv` (Phase 1, appended on each run)

Segments where automated cleaning produced suspicious metrics and warrant human inspection.

| Column | Description |
|--------|-------------|
| `segment_name`, `segment_id`, `role` | Segment identity |
| `reason` | `high_closing_fraction` or `residual_fragmentation` |
| `closing_radius_um` | Closing radius applied |
| `original_voxels` | Voxel count before cleaning |
| `closing_voxels_added` | Voxels added by morphological closing |
| `closing_fraction` | `closing_voxels_added / original_voxels`; values > 0.20 trigger a flag |
| `components_after_clean` | Number of connected components remaining after cleaning |
| `recommended_action` | Suggested diagnostic command |

Use `diag_cleaning_param_sweep.py` with the suggested command to inspect flagged segments.

---

## Coordinate systems and conventions

This pipeline combines multiple coordinate spaces:

1. **Local voxel indices** (within an exported mask bounding box)
2. **Global voxel indices** (offset by the segment's bounding box minimum in the full volume)
3. **World units** (voxel index × voxel_size in microns)

Key requirements:
- All phases must use the same convention for converting voxel indices to physical coordinates.
- If a voxel-center convention is used (e.g., +0.5 offset), it must be consistent across SWC writing and centroid extraction.

A mismatch of even half a voxel is usually small in absolute terms, but consistency is important for mapping markers to morphologies.

---

## Execution / orchestration

### `neuron_pipeline.main_pipeline`
Defines the high-level control flow and exposes a CLI via `argparse`:

- instantiate classifier and build registry (Phase 0, always runs)
- run phase 1 and write SWCs
- run phase 2 and write centroids
- run phase 3 cable mapping
- run phase 4 connectivity export + Arbor recipe

**Key CLI flags:**

| Flag | Description |
|------|-------------|
| `--phases N [N ...]` | Run only specific phase numbers (0–4) |
| `--from-phase N` | Resume from phase N, loading earlier outputs from disk |
| `--resume` | Phase 1 only: skip segments that already have an `.swc` file |
| `--output-dir DIR` | Base output directory (default: `./vast_export`) |
| `--miplevel-skel N` | MIP level for Phase 1 voxel extraction (default: 1) |
| `--miplevel-centroid N` | MIP level for Phase 2 centroid extraction (default: 1) |
| `--spur-length-um F` | Phase 1 spur-pruning threshold in µm (default: 2.0) |
| `--warn-distance-um F` | Phase 3 mapping-distance warning threshold in µm (default: 5.0) |
| `--syn-mechanism NAME` | Arbor synapse mechanism name for Phase 4 (default: `expsyn`) |
| `--segment NAME [...]` | Process only the named segments |
| `--log-level LEVEL` | Console log verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### `scripts/run_pipeline.py`
Runs the `main_pipeline` (i.e., a wrapper). This is the recommended entrypoint for running the pipeline from the repo root.

---

## Extensibility points

- Add new segment types or classification rules in `segment_classifier.py`.
- Add new pipeline stages under `neuron_pipeline/stages/` and call them from `main_pipeline.py`.
- Extend `vastpy` with additional export functions (keeping the API stable and documented).

---

## Testing strategy (current / planned)

Current testing is minimal while the pipeline is under heavy development. The intended test layers:

1. **Unit tests** for:
   - voxel cleaning invariants
   - centroid extraction correctness on small synthetic volumes
   - SWC structural invariants (single parent per node, connectedness, root presence)

2. **Integration tests** (smoke tests) that:
   - run a small subset of segments end-to-end
   - validate that SWCs are written and loadable by downstream tools

3. **Regression checks** for coordinate consistency:
   - centroid-to-SWC mapping residual distributions
   - sanity checks on typical distances per marker type

---

## Known limitations

- Requires VAST Lite to be open/running with the proper data loaded.
- Performance depends on export throughput and voxel mask sizes; large segments can be slow and memory-heavy.
- Skeletonization quality can vary with voxel cleaning parameters and topology of the structure.

---