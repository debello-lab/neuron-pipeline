# ms-neuron-pipeline

A Python pipeline that automates the complete workflow from **VAST Lite** segmentation annotations to simulation-ready neuron morphologies for **Arbor-sim**. Developed as a UC Davis CS Master's project for the Debello Lab.

This repository contains two components:

- **`vastpy`** — a reusable Python interface to VAST Lite, replacing the previous MATLAB-dependent export workflow. Handles RLE-decoded voxel mask retrieval, mesh export, and metadata access.
- **`neuron_pipeline`** — the multi-stage processing pipeline built on `vastpy`: segment classification → voxel extraction → cleaning → skeletonization → SWC export → centroid extraction → cable mapping → connectivity export.

> **Status:** Active development (MS project, UC Davis).  
> **Target audience:** Connectomics / EM reconstruction researchers using VAST Lite.

---

## Prerequisites

- Python 3.10+
- **VAST Lite** must be running with the target segmentation project loaded before any pipeline command. The Remote Control API server must be enabled on port `22081`.

---

## Installation

```bat
install.bat
```

This creates a `.venv`, installs all dependencies from `requirements.txt`, and installs `neuron_pipeline` and `vastpy` in editable mode.

To activate the environment in a new terminal:

```bat
.venv\Scripts\activate
```

---

## Running the pipeline

```bat
scripts\run_pipeline.bat
```

Or directly:

```
python -m neuron_pipeline.main_pipeline [OPTIONS]
```

**Common flags:**

| Flag | Description |
|------|-------------|
| `--phases 1` | Run only Phase 1 |
| `--from-phase 2` | Skip Phase 1 using existing outputs on disk |
| `--segment A1 A2` | Process only named segments |
| `--spur-length-um 2.0` | Spur pruning threshold (µm) |
| `--resume` | Skip segments that already have a `.swc` on disk |
| `--output-dir ./vast_export` | Output directory (default) |

**Single-segment diagnostic:**

```
python "diagnostic scripts/test.py" <segment_id>
```

**Clear generated outputs:**

```
python scripts/clear_outputs.py
```

---

## Pipeline phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Segment classification — parses VAST naming conventions to assign roles (AXON, POST\_SYN, BOUTON, SYNAPSE, CONTACT) and builds a shared registry | Implemented |
| 1 | Voxel extraction → morphological cleaning → TEASAR skeletonization → SWC export | Implemented |
| 2 | Centroid extraction for bouton / synapse / contact markers | Implemented |
| 3 | Map centroids to nearest cable edge (KD-tree, arc-fraction output) | Implemented |
| 4 | Connectivity CSV + Arbor recipe JSON | Implemented |

---

## Output layout

All outputs are written to `./vast_export/` (created automatically):

```
vast_export/
├── swc/                     SWC skeleton files (cell_<name>.swc)
├── stats/
│   └── skeleton_stats.csv   Per-segment skeletonization metrics
├── centroids.csv            Bouton / synapse / contact centroids
├── cable_mappings.csv       Centroid-to-skeleton cable locations
├── connectivity.csv         Full pre/post synaptic edge list
├── connectivity_summary.csv Per axon–POST_SYN pair statistics
├── connectivity_report.txt  Plain-text connectivity report with QC flags
├── arbor_recipe.json        Arbor-ready network description
├── arbor_recipe.py          Arbor Python recipe class (ReconstructedRecipe)
├── review_queue.csv         Segments flagged for manual inspection
├── pipeline_report.txt      Single-page summary of the full run
└── logs/
    └── pipeline_<timestamp>.log
```

Diagnostic outputs (single-segment runs) go to `./diag_output/`.

---

## Repository layout

```
ms-neuron-pipeline/
├── src/
│   ├── vastpy/                  Library layer (reusable)
│   │   └── control/
│   │       ├── exporting.py     RLE decoding, voxel/mesh export
│   │       └── VASTControlClass.py
│   └── neuron_pipeline/         Application layer
│       ├── main_pipeline.py     Pipeline orchestrator
│       ├── templates/
│       │   └── arbor_recipe_template.py
│       └── stages/
│           ├── segment_classifier.py
│           ├── extract_surfaces.py
│           ├── voxel_cleaning.py
│           ├── skeletonization.py
│           ├── centroid_extraction.py
│           ├── centroid_mapper.py
│           └── connectivity_builder.py
├── scripts/
│   ├── run_pipeline.bat
│   ├── clear_outputs.py
│   └── run_extract_surfaces.py
├── diagnostic scripts/
├── vast_export/                 Generated outputs (gitignored)
├── install.bat
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Using `vastpy` standalone

`vastpy` can be used independently of the pipeline to access VAST Lite programmatically:

```python
from vastpy.control.exporting import VASTControlClass

vast = VASTControlClass(host="127.0.0.1", port=22081)
```

`exporting.py` provides optimized voxel mask retrieval (RLE decoding) and mesh export.  
`VASTControlClass.py` provides broader VAST remote-control functions.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named neuron_pipeline` | Run `pip install -e .` inside the activated venv |
| `ConnectionRefusedError` on pipeline start | VAST Lite is not running, or the Remote Control API is not enabled on port 22081 |
| `ImportError: vastpy.control.exporting` | Check that `exporting.py` filename matches the import exactly |

---

## Citation

If you use this software in academic work, please cite:

> Nicolas Randazzo, *ms-neuron-pipeline: VAST Lite export + neuron skeletonization pipeline*, UC Davis, 2026.

---

## License

GNU GPLv3 — see [LICENSE](LICENSE).  
VAST Lite remains under its original license; this repository does not redistribute VAST Lite.

---

## AI use disclosure

This project uses AI tools for assistance (debugging, refactoring suggestions, documentation drafting). All code and research decisions are reviewed and validated by the author.
