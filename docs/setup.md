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
  LICENSE
  pyproject.toml
  README.md
  docs/
    architecture.md
    setup.md
  scripts/
    run_pipeline.py
  src/
    neuron_pipeline/
      __init__.py
      main_pipeline.py
      stages/
        centroid_extraction.py
        extract_surfaces.py
        segment_classifier.py
        skeletonization.py
        voxel_cleaning.py
    vastpy/
      __init__.py
      control/
        __init__.py
        VASTControlClass.py
        exporting.py
```

## 1) Create and activate a virtual environment
### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows CMD
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
After activation, confirm Python is using the virtual environment:

```bat
python -V
where python
```

## 2) Install the project in editable mode

From the repository root:
```
python -m pip install --upgrade pip
python -m pip install -e .
```
Editable install is recommended because:
- the project uses a src/ layout
- it allows imports like from neuron_pipeline.main_pipeline import main
- it lets you edit source files without reinstalling each time

## 3) Verify package imports

From the repository root, test that the packages import successfully:
```
python -c "import neuron_pipeline; import vastpy; print('Imports OK')"
```
If this fails, check the following:
- pyproject.toml exists in the repo root
- __init__.py exists in:
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

From the repository root:
```bat
python .\scripts\run_pipeline.py
```

This script should call:
```python
from neuron_pipeline.main_pipeline import main

if __name__ == "__main__":
    main()
```

## 6) Expected outputs

By default, the pipeline writes outputs under: `vast_export/`

Typical output folders/files include:
```
vast_export/
  logs/
  meshes/
  swc/
  voxels/
  centroids.csv
```
These outputs may include:
- exported voxel masks
- exported meshes
- generated SWC files
- centroid tables for boutons/synapses/contacts
- logs or debug artifacts

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

Use this as a minimal setup checklist:

- Clone the repository
- Create and activate .venv
- Run python -m pip install -e .
- Confirm import neuron_pipeline works
- Open VAST Lite
- Load the correct project/segmentation
- Run python .\scripts\run_pipeline.py
- Confirm outputs appear under vast_export/