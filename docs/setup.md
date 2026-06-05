# Setup

This document explains how to set up and run **ms-neuron-pipeline** in a local development environment.

The project uses a `src/` layout and contains two Python packages:

- **`vastpy`**: reusable Python interface for interacting with VAST Lite
- **`neuron_pipeline`**: the end-to-end pipeline for classification, export, cleaning, skeletonization, and centroid extraction

Phase 1 skeletonization uses **CGAL Mean Curvature Flow** via compiled C++ executables (in `cpp/`) and the **MASCAF** Python library. This requires extra setup steps before the pipeline can run.

---

## Requirements

### Software

| Tool | Required version | Notes |
|------|-----------------|-------|
| Python | **3.12+** | 3.14 recommended |
| [uv](https://docs.astral.sh/uv/) | any recent | Python package manager |
| CMake | 3.20+ | For building C++ executables |
| Visual Studio Build Tools | 2022+ | MSVC C++ compiler |
| [vcpkg](https://vcpkg.io/) | any recent | C++ package manager (CGAL, Eigen3) |
| VAST Lite | 1.5.0+ | Must be running before the pipeline |

### Assumptions

- VAST Lite is open with the target segmentation loaded
- The VAST remote-control/export interface is configured for your local setup
- All dependencies (Python + C++) are installed as described below

---

## Repository layout

```text
ms-neuron-pipeline/
├── cpp/                    C++ source for CGAL executables
│   ├── CMakeLists.txt
│   ├── vcpkg.json          vcpkg manifest (cgal, eigen3)
│   ├── mesh_skeletonize.cpp
│   ├── mesh_repair.cpp
│   └── mesh_simplify.cpp
├── src/
│   ├── vastpy/control/
│   │   ├── exporting.py
│   │   └── VASTControlClass.py
│   └── neuron_pipeline/
│       ├── main_pipeline.py
│       ├── templates/
│       └── stages/
├── docs/
├── scripts/
├── pyproject.toml
├── .env                    Local environment variables (not committed)
└── uv.lock
```

---

## 1) Install Python dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```powershell
# Install uv if not already present
pip install uv

# Install all Python dependencies (creates .venv automatically)
uv sync
```

Alternatively, create the environment and activate it manually:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
```

> **Note:** `install.bat` also works for a quick setup but uses pip rather than uv.
> For full dependency locking, prefer `uv sync`.

Verify imports:

```powershell
uv run python -c "import neuron_pipeline; import vastpy; from mascaf import CGALOperator; print('OK')"
```

---

## 2) Build the CGAL C++ executables

The CGAL skeletonizer requires three compiled executables:
- `mesh_skeletonize.exe` — MCF skeleton extraction (primary)
- `mesh_repair.exe` — watertight mesh repair utility
- `mesh_simplify.exe` — mesh edge-collapse simplification

### Prerequisites

**vcpkg** manages CGAL and Eigen3. Install it if you haven't already:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

Set `VCPKG_ROOT` (add this to your system environment variables for persistence):

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"   # adjust to your vcpkg location
```

### Configure and build

From the repository root (use the literal path to your vcpkg toolchain):

```powershell
cmake -S . -B cpp\build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

cmake --build cpp\build --config Release
```

> **Important:** Do not quote the value when setting `VCPKG_ROOT` with `set` in cmd.exe —
> `set VCPKG_ROOT=C:\vcpkg` (no quotes). With `%VCPKG_ROOT%`, embedded quotes break the path.
> In PowerShell, `$env:VCPKG_ROOT` expands cleanly.

vcpkg reads `vcpkg.json` at the repo root and installs CGAL + Eigen3 into `cpp\build\vcpkg_installed\`
on the first run (slow, ~20–40 min). Subsequent builds reuse the cache.

**Reusing an existing vcpkg_installed directory** (skip the download):

If CGAL is already installed in another local build with the same `vcpkg.json` baseline
(`0297aaaf`), pass its path with `-DVCPKG_INSTALLED_DIR`:

```powershell
cmake -S . -B cpp\build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_INSTALLED_DIR="C:\path\to\other-build\vcpkg_installed"

cmake --build cpp\build --config Release
```

After a successful build, binaries are at `cpp\build\Release\`.

---

## 3) Create a `.env` file

Create `.env` at the repository root (already in `.gitignore`):

```
MASCAF_CGAL_BIN_DIR=./cpp/build/Release
```

This tells `MCFSkeletonizer` where to find the compiled executables. All pipeline runs
should use `uv run --env-file .env` to pick it up automatically:

```powershell
uv run --env-file .env python -m neuron_pipeline.main_pipeline [OPTIONS]
```

---

## 4) Prepare VAST Lite

Before running the pipeline:

- Open VAST Lite
- Load the target segmentation/project
- Confirm the remote-control/export interface is in the expected state

The pipeline does not launch VAST automatically. VAST must already be running at
`127.0.0.1:22081`.

---

## 5) Run the pipeline

```powershell
uv run --env-file .env python -m neuron_pipeline.main_pipeline [OPTIONS]
```

Or via the batch script (does not load `.env`):

```bat
scripts\run_pipeline.bat
```

### Common flags

**Phase control** (mutually exclusive)

| Flag | Default | Description |
|------|---------|-------------|
| `--phases N [N ...]` | all | Run only the listed phase numbers (0–4). E.g. `--phases 1 2` |
| `--from-phase N` | — | Run phases N through 4, loading Phase 1 outputs from disk |

**Output / I/O**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `./vast_export` | Base output directory |
| `--resume` | off | Phase 1: skip segments that already have a `.swc` on disk |

**Segment filter** (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--segment NAME [NAME ...]` | Process only the named segments |
| `--skip NAME [NAME ...]` | Skip the named segments |

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
| `--syn-mechanism NAME` | `expsyn` | Arbor synapse mechanism name |

**Logging**

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` | Console verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Single-segment diagnostic (no VAST required, uses pre-existing SWC):

```powershell
python "diagnostic scripts/test.py" <segment_id>
```

---

## 6) Expected outputs

By default, the pipeline writes outputs under `vast_export/`:

```
vast_export/
├── swc/                        SWC skeleton files (cell_<name>.swc)
├── meshes/                     Watertight OBJ mesh per segment (Phase 1 MCF)
├── polylines/                  CGAL MCF polylines per segment (Phase 1 MCF)
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

`meshes/` and `polylines/` contain intermediate MCF outputs useful for inspection and debugging.

---

## Quick setup checklist

- [ ] Install Python 3.12+
- [ ] Install uv: `pip install uv`
- [ ] Install vcpkg and set `VCPKG_ROOT`
- [ ] Build C++ executables: `cmake -S . -B cpp\build ... && cmake --build cpp\build --config Release`
- [ ] Create `.env` with `MASCAF_CGAL_BIN_DIR=./cpp/build/Release`
- [ ] Run `uv sync` to install Python dependencies
- [ ] Verify: `uv run --env-file .env python -c "from mascaf import CGALOperator; print('OK')"`
- [ ] Open VAST Lite and load the target segmentation
- [ ] Run: `uv run --env-file .env python -m neuron_pipeline.main_pipeline --phases 1 --segment A1`

---

## Common problems

**CMake cannot find CGAL**

```
CMake Error: could not find CGALConfig.cmake
```

Cause: vcpkg toolchain not passed to cmake, or `VCPKG_ROOT` set with quotes.

Fix: use the literal path (no env var expansion in cmd.exe with `set`):
```powershell
cmake -S cpp -B cpp\build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="C:\your\vcpkg\scripts\buildsystems\vcpkg.cmake"
```

---

**`MASCAF_CGAL_BIN_DIR` not set**

```
RuntimeError: MCFSkeletonizer: set bin_dir or MASCAF_CGAL_BIN_DIR env var
```

Cause: `.env` file missing or pipeline run without `--env-file .env`.

Fix: ensure `.env` exists at the repo root and run via:
```powershell
uv run --env-file .env python -m neuron_pipeline.main_pipeline ...
```

---

**`No module named 'pyvista'` or `mascaf` import error**

Cause: Python environment out of sync.

Fix:
```powershell
uv sync
```

---

**`mesh_skeletonize.exe` not found**

Cause: C++ build not completed, or `MASCAF_CGAL_BIN_DIR` points to wrong directory.

Fix: complete the cmake build and confirm `cpp\build\Release\mesh_skeletonize.exe` exists,
then update `.env` accordingly.

---

**Marching-cubes mesh is not watertight**

The segment will be logged as a skeletonization failure and appear in `review_queue.csv`.
This affects roughly 5% of segments with very thin or topologically complex geometry.
Run `cpp\build\Release\mesh_repair.exe <input.obj> <output.obj>` to diagnose the mesh pathology.

---

**No module named neuron_pipeline**

Cause: project not installed in editable mode, or running outside the venv.

Fix:
```powershell
uv run python -c "import neuron_pipeline; print('OK')"
```

If that fails: `uv sync` then retry.

---

**Script runs but VAST export fails**

Cause: VAST Lite is not open, or the correct segmentation is not loaded.

Fix: open VAST manually, load the correct data, verify it is reachable at `127.0.0.1:22081`.
