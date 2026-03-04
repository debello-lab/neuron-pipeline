# ms-neuron-pipeline

A Python-based neuron reconstruction pipeline for exporting 3D segment data from **VAST Lite** and producing downstream artifacts such as cleaned voxel masks, skeleton graphs, and **SWC** files for simulation/analysis workflows (e.g., Arbor / NEURD integration).

This project is to reduce manual MATLAB-based exporting and produce standardized SWCs + marker tables for simulation and connectivity analysis.

This repository contains two components:

- **`vastpy`**: an open-source Python interface for interacting with VAST Lite without MATLAB dependencies (exporting meshes/voxels/metadata).
- **`neuron_pipeline`**: the master’s project pipeline built on top of `vastpy` (classification → extraction → cleaning → skeletonization → SWC + centroid mapping).

> Status: **Active development** (MS project, UC Davis).  
> Target audience: connectomics / EM reconstruction researchers using VAST Lite.

---

## Repository layout
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

## Quick start (recommended)

### 1 Create / activate venv

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (CMD):
```
python -m venv .venv
.\.venv\Scripts\activate
```

### 2 Install editable
```
python -m pip install -U pip
python -m pip install -e .
```

### 3 Run the pipeline
```
python .\scripts\run_pipeline.py
```

---

## Pipeline overview

The pipeline currently runs in phases:

- Phase 0 — Segment classification
    - identifies segment types (e.g., AXON, POST_SYN, BOUTON, SYNAPSE/CONTACT markers)

- Phase 1 — Skeletonization
    - exports voxel masks / surfaces
    - cleans voxel representation
    - extracts a skeleton graph and writes SWC

- Phase 2 — Centroid extraction
    - extracts centroids for small markers (boutons/synapses/contacts)

- Phase 3 — Centroid → SWC mapping (planned / in-progress)
    - maps centroids to nearest SWC nodes for connectivity/annotation

Outputs are written under:

- ./vast_export/meshes/

- ./vast_export/voxels/

- ./vast_export/swc/

- ./vast_export/centroids.csv

See: [docs/architecture.md](docs/architecture.md)

---

## Using `vastpy` as a standalone library

If you only want the VAST Lite interface (without the pipeline), you can import vastpy directly after installation:

```python
from vastpy.control.exporting import VASTControlClass

vast = VASTControlClass(host="127.0.0.1", port=22082)
# Example: export a segment's voxel mask / mesh (functions depend on your current API)
# vast.export_segment_voxels(segment_id=123, out_dir="out/voxels")
# vast.export_segment_mesh(segment_id=123, out_dir="out/meshes")
```
Inside of vastpy/control/ is also `VASTControlClass.py`, where there are more functions dedicated to controlling the VAST Lite software. `exporting.py` is primarily used due to the optimized socket functionality and fully worked surface extraction.


---

## Configuration

The pipeline currently assumes:

- VAST Lite is running and accessible via the remote control interface

- The Remote Control API Server is enabled and open on Port 22081

- There is at least a segmentation (.vsseg) file open in VAST Lite

See: [docs/setup.md](docs/setup.md)

---

## Development Notes

### Common issues

- `No module named neuron_pipeline`

    - You probably forgot `pip install -e .`

- **Import errors for** `vastpy.control.exporting`

    - Ensure the module filename matches the import (e.g., `exporting.py`)

---

## Citation

If you use this software in academic work, please cite:

> Nicolas Randazzo, ms-neuron-pipeline: VAST Lite export + neuron skeletonization pipeline, 2026.

---

## License
This repository uses the GNU GPLv3 License. See [LICENSE](LICENSE)

VAST Lite: remains under its original license (this repo does not redistribute VAST Lite itself)

---

## AI use disclosure

This project may use AI tools for assistance (debugging, refactoring suggestions, documentation drafting).
All code and research decisions are reviewed and validated by the author.