# Setup

This document explains how to set up and run **ms-neuron-pipeline** in a local development environment.

The project uses a `src/` layout and contains two Python packages:

- **`vastpy`**: reusable Python interface for interacting with VAST Lite
- **`neuron_pipeline`**: the end-to-end pipeline for classification, export, cleaning, skeletonization, and centroid extraction

---

## Requirements

### Software
- **Python 3.10+**
- **VAST Lite 1.5.0** (or the version your lab workflow depends on)
- A local clone of this repository

### Assumptions
Before running the pipeline:

- **VAST Lite must be open**
- The target segmentation/project must already be loaded in VAST
- The VAST remote-control/export workflow used by `vastpy` must be available and configured as expected by your local setup
- The Python environment must have all required dependencies installed

---

## Repository layout

Expected structure:

```text
ms-neuron-pipeline/
+--- src/
|   +--- vastpy/
|   | +--- control/
|   |   | +--- exporting.py 
|   |   | +--- VASTControlClass.py
|   +--- neuron_pipeline/
|   | +--- main_pipeline.py
|   | +--- templates/
|   |   | +--- arbor_recipe_template.py
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

## 1) Install (recommended)

From the repository root:

```bat
install.bat
```

This script checks for Python 3.10+, creates `.venv`, installs all dependencies from `requirements.txt`, and installs `neuron_pipeline` and `vastpy` in editable mode.

To activate the environment in a new terminal:

```bat
.venv\Scripts\activate
```

## 1b) Manual install (alternative)

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Windows CMD
```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 3) Verify package imports

From the repository root, test that the packages import successfully:
```
python -c "import neuron_pipeline; import vastpy; print('Imports OK')"
```
If this fails, check the following:
- pyproject.toml exists in the repo root
- \_\_init\_\_.py exists in:
    - src/neuron_pipeline/
    - src/neuron_pipeline/stages/
    - src/vastpy/
    - src/vastpy/control/

- the module filename matches the import path

Example:

- if your code imports:
    ```python 
    from vastpy.control.exporting import VASTControlClass
    ```
    then the file must be:
    ```
    src/vastpy/control/exporting.py
    ```

## 4) Prepare VAST Lite

Before running the pipeline:
- Open VAST Lite
- Load the target segmentation/project
- Confirm the remote-control/export interface is in the expected state for your local VAST setup
- Confirm any paths, project files, or ports used by your control classes are correct

This project does not launch VAST Lite automatically. VAST must already be running.

## 5) Run the pipeline

```bat
scripts\run_pipeline.bat
```

Or directly:

```
python -m neuron_pipeline.main_pipeline [OPTIONS]
```

Common flags:

**Phase control** (mutually exclusive)

| Flag | Default | Description |
|------|---------|-------------|
| `--phases N [N ...]` | all | Run only the listed phase numbers (0–4). E.g. `--phases 1 2` |
| `--from-phase N` | — | Run phases N through 4, loading earlier outputs from disk. Phase 0 always re-runs. |

**Output / I/O**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `./vast_export` | Base output directory |
| `--resume` | off | Phase 1: skip segments that already have a `.swc` on disk |

**Segment filter** (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--segment NAME [NAME ...]` | Process only the named segments. E.g. `--segment A1 A1B2P1` |
| `--skip NAME [NAME ...]` | Skip the named segments. E.g. `--skip A1P1` |

**Phase 1 — skeletonization**

| Flag | Default | Description |
|------|---------|-------------|
| `--miplevel-skel N` | `1` | MIP level for voxel extraction (0 = full res) |
| `--spur-length-um F` | `2.0` | Spur-pruning threshold in µm |

**Phase 2 — centroid extraction**

| Flag | Default | Description |
|------|---------|-------------|
| `--miplevel-centroid N` | `1` | MIP level for centroid voxel extraction |

**Phase 3 — cable mapping**

| Flag | Default | Description |
|------|---------|-------------|
| `--warn-distance-um F` | `5.0` | Warning threshold for centroid-to-cable distance in µm |

**Phase 4 — connectivity**

| Flag | Default | Description |
|------|---------|-------------|
| `--syn-mechanism NAME` | `expsyn` | Arbor synapse mechanism name written into the recipe |

**Logging**

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` | Console verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. File log is always `DEBUG`. |

Single-segment diagnostic:

```
python "diagnostic scripts/test.py" <segment_id>
```

## 6) Expected outputs

By default, the pipeline writes outputs under: `vast_export/`

```
vast_export/
├── swc/                        SWC skeleton files (cell_<name>.swc)
├── stats/
│   └── skeleton_stats.csv      Per-segment skeletonization metrics
├── centroids.csv               Bouton / synapse / contact centroids
├── cable_mappings.csv          Centroid-to-skeleton cable locations
├── connectivity.csv            Full pre/post synaptic edge list
├── connectivity_summary.csv    Per axon–POST_SYN pair statistics
├── connectivity_report.txt     Plain-text connectivity report with QC flags
├── arbor_recipe.json           Arbor-ready network description
├── review_queue.csv            Segments flagged for manual inspection
├── pipeline_report.txt         Single-page run summary
└── logs/
    └── pipeline_<timestamp>.log
```

## Common problems
**No module named neuron_pipeline**
Cause:
- project not installed in editable mode
- running from the wrong directory

Fix:
```
python -m pip install -e .
```
Then run again from the repo root.

---

**No module named 'vastpy.control.exporting'**

Cause:
- import path does not match filename

Fix one of the following:

Option A: rename file
```bat 
rename src\vastpy\control\exporting.py exporting.py
```
Option B: change import
```python
from vastpy.control.exporting import VASTControlClass
```

---

**Script runs but VAST export fails**
Cause:
- VAST Lite is not open
- the correct segmentation/project is not loaded
- remote-control setup is not in the expected state

Fix:
- open VAST manually
- load the correct data
- verify local VAST control/export assumptions
---

**Imports work in one terminal but not another**
Cause:
- virtual environment not activated
- different Python interpreter being used

Fix:
```bat
where python
python -V
```
Make sure the interpreter is the one inside .venv.

---

### Development workflow

Recommended workflow:
1. Activate the virtual environment
2. Pull latest changes
3. Work on a feature branch
4. Run the pipeline or a small test subset
5. Commit incremental changes with clear messages

Example branch names:
- feature/repo-restructure
- feature/vastpy-packaging
- feature/swc-export
- fix/import-paths
- fix/centroid-coordinate-alignment
---
### Recommended .gitignore

At minimum:
```gitignore
# Virtual environment
.venv/

# Python cache
__pycache__/
*.pyc

# Build / packaging artifacts
*.egg-info/
build/
dist/

# Pipeline outputs
vast_export/
```
---

## Quick setup checklist

- Clone the repository
- Run `install.bat`
- Activate `.venv\Scripts\activate` in any new terminal
- Confirm `python -c "import neuron_pipeline; import vastpy; print('OK')"` passes
- Open VAST Lite and load the target segmentation
- Run `scripts\run_pipeline.bat`
- Confirm outputs appear under `vast_export/`