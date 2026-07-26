# AGENTS.md — Motion Amplification for 4D Gaussian Splatting

> This file is written for AI coding agents. It assumes you know nothing about the project. All facts below are derived from the actual repository contents (`README.md`, `docs/`, `pyproject.toml`, source files, and the orchestrator planning documents).

## 1. Project overview

This repository is Bartosz Moczkowski's thesis project at the Technology University of Lodz: **per-part motion amplification of 4D Gaussian Splatting (4DGS) reconstructions**.

The high-level pipeline is:

```
synthetic multi-cam capture (NVIDIA Omniverse / Isaac Sim)
  → 4DGS reconstruction (canonical Gaussians + deformation field)
  → motion segmentation (cluster Gaussians into rigid motion groups)
  → per-segment Eulerian motion amplification (render_amp.py)
```

Synthetic data is the enabler: real captures have no ground-truth camera poses or per-part labels, so quantitative evaluation (ARI / IoU) is only possible on scenes authored in Omniverse.

The codebase is a fork of [4DGaussians](https://github.com/hustvl/4DGaussians). Most of the root-level 4DGS code is upstream; the author's additions are concentrated in:

- `render_amp.py`, `motion_amp/renderer.py`
- `ampUI.py`, `cameras.py`, `run_renders_auto.py`
- `omniverse_pipeline/`
- `motion_seg/`
- `orchestrator/`

## 2. Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12.12 (locked via `.python-version` and root `pyproject.toml`) |
| Package manager | `uv` (root and orchestrator both use `pyproject.toml` + `uv.lock`) |
| ML framework | PyTorch 2.6+ with CUDA 12.6 (root `pyproject.toml` pins to the `pytorch-cu126` index) |
| CUDA rasterizer | `diff-gaussian-rasterization` and `simple-knn` as editable workspace submodules |
| Synthetic capture | NVIDIA Isaac Sim 6.0.1 (native Windows install), USD / `pxr` |
| Container runtime | Docker Desktop on Windows, `nvidia/cuda:12.4.1-devel` base image, `nvcr.io/nvidia/isaac-sim:6.0.1` image |
| Config / orchestration | Pydantic, YAML presets, custom DAG scheduler |
| Remote control | MCP (Model Context Protocol) over HTTP/SSE with bearer-token auth |
| UI | Streamlit (`ampUI.py` for the standalone workflow; `orchestrator/ui/` for the orchestrator) |
| Testing | `pytest` (only in `orchestrator/tests/`) |

## 3. Repository layout

| Path | Purpose |
|------|---------|
| `train.py`, `render.py`, `scene/`, `gaussian_renderer/`, `utils/`, `arguments/` | Upstream 4DGS core: training, rendering, Gaussian model, deformation network (HexPlane), dataset readers, losses, configs |
| `render_amp.py` | **Motion-amplified rendering**. Extracts per-frame Gaussian parameters, applies FFT-based amplification (Eulerian / absolute / segmented), then renders a video |
| `motion_amp/renderer.py` | Low-level helper used by `render_amp.py`; returns raw pre-rasterization Gaussian parameters |
| `ampUI.py` | Standalone Streamlit UI for `render_amp.py` |
| `cameras.py` | USB multi-camera recorder (OpenCV) |
| `run_renders_auto.py` | Benchmark harness that loops methods/VRAM modes/models and writes `results.csv` |
| `omniverse_pipeline/` | Isaac Sim capture + USD prep + conversion to 4DGS `multipleview` format |
| `motion_seg/` | Motion segmentation: rigidity-graph clustering (`segment_rigid.py`) + MultiBodySync adapter (`mbs_infer.py`) + evaluation |
| `orchestrator/` | Three-layer pipeline system: DAG execution (`pipeline/`), HTTP MCP server (`mcp_server/`), Streamlit UI (`ui/`) |
| `submodules/` | `depth-diff-gaussian-rasterization`, `simple-knn`, `multibody-sync-4dgs` |
| `data/` | D-NeRF synthetic scenes (`bouncingballs`, `lego`, `mutant`, ...) + generated `multipleview` scenes (e.g. `pump01`) |
| `output/` | Trained 4DGS models (not committed due to size) |
| `docs/` | Compiled documentation (`overview.md`, `motion-segmentation.md`, `omniverse-pipeline.md`, `orchestrator.md`, ...) |
| `.claude_notes/` | Chronological working notes (primary detailed record) |
| `LLFF/`, `lpipsPyTorch/`, `viseron/` | Bundled reference tools, LPIPS implementation, and an unrelated NVR setup respectively |

## 4. Build and runtime setup

### 4.1 Root project (training / rendering / motion amp)

The root project is a `uv` workspace. It requires Python 3.12.12 and a CUDA-capable GPU.

```bash
# Create the venv and install all default dependencies (including editable CUDA submodules)
uv sync
```

Key editable workspace members (declared in root `pyproject.toml`):

- `submodules/depth-diff-gaussian-rasterization` → package `diff_gaussian_rasterization`
- `submodules/simple-knn` → package `simple_knn`
- `orchestrator` → package `pipeline`

These submodules are **CUDA extensions** built by `torch.utils.cpp_extension`. They compile on first `uv sync`. The Dockerfile hard-codes `TORCH_CUDA_ARCH_LIST="8.6+PTX"` for the author's RTX 3090; adjust if you target a different GPU.

### 4.2 Orchestrator extras

The orchestrator is opt-in via root extras:

```bash
# Layer 1: DAG engine, artifacts, container manager, tests
uv sync --extra orchestrator

# Layer 2: HTTP MCP server
uv sync --extra orchestrator-mcp

# Layer 3: Streamlit UI
uv sync --extra orchestrator-ui
```

### 4.3 Docker images

- **`cuda`** — built from repo `Dockerfile`, tag `4dgs-motion-amp-cuda:latest`. Used for train / render / seg_extract / amp / Option-A segmentation.
- **`isaac`** — pulled from `nvcr.io/nvidia/isaac-sim:6.0.1` (requires NGC login/EULA). Used for CPU-only USD prep (`prep_split`, `prep_motion`).

### 4.4 Native Isaac Sim requirement

`capture.isaac` **cannot run inside Docker** because Vulkan (required by Isaac Sim's RTX renderer) is unsupported under WSL2/Docker Desktop on Windows. It runs as a native Windows subprocess against the author's Isaac Sim install:

```
Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat
```

Override with the environment variable `PIPELINE_ISAAC_NATIVE_PYTHON`.

First-time Windows machine setup is documented in `orchestrator/planning/WINDOWS_SETUP.md`.

## 5. Key commands

### 5.1 Training

```bash
python train.py -s data/dnerf/lego -m output/dnerf/lego --configs arguments/dnerf/lego.py
python train.py -s data/multipleview/pump01 -m output/multipleview/pump01 --configs arguments/multipleview/default.py
```

Training runs in two stages: coarse (static Gaussians) then fine (deformation network enabled). Final artifacts are `point_cloud/iteration_N/point_cloud.ply` + `deformation.pth`.

### 5.2 Standard rendering

```bash
python render.py -s data/dnerf/lego -m output/dnerf/lego --iteration 20000 --skip_train
```

### 5.3 Motion-amplified rendering

```bash
python render_amp.py -m output/dnerf/lego --configs arguments/dnerf/lego.py \
    --amp_factors 2 -1 -1 -1 -1 -1 -1 -1 \
    --freq_low 0.0 --freq_high 1.0 \
    --method eulerian --video_path out.mp4
```

### 5.4 Motion segmentation (reference scripts)

```bash
# Option B: rigidity-graph clustering (CPU, default)
uv run python -m motion_seg.extract_trajectories --model_path output/multipleview/pump01 --configs arguments/multipleview/pump01.py
uv run python -m motion_seg.segment_rigid --trajectories output/multipleview/pump01/trajectories.npz --out output/multipleview/pump01/segmentation.npz
uv run python -m motion_seg.evaluate_segmentation --pred output/multipleview/pump01/segmentation.npz --gt data/multipleview/pump01/gt_segmentation.npz

# Convenience wrapper
./motion_seg/run.sh pump01
```

### 5.5 Omniverse pipeline (reference scripts)

```bash
# USD prep (plain Python)
python omniverse_pipeline/split_mesh.py --in "CONJUNTO BOMBAS.usd" --out CONJUNTO_BOMBAS_segmented.usd --group CONJUNTO_BOMBAS
python omniverse_pipeline/add_motion.py --in CONJUNTO_BOMBAS_segmented.usd --out CONJUNTO_BOMBAS_animated.usd

# Capture (must use Isaac Sim's own Python, native Windows)
Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat omniverse_pipeline/omni_capture.py --config omniverse_pipeline/capture_config_pump.yaml

# Convert to 4DGS multipleview format
python omniverse_pipeline/omni_to_4dgs.py --capture <capture_dir> --out . --name pump01
```

### 5.6 Orchestrator

```bash
# Run the full automated DAG (from Python; see orchestrator/planning/ARCHITECTURE.md for the API)
python -c "from pipeline.api import run_pipeline; ..."

# Start the MCP server
$env:PIPELINE_MCP_TOKEN = "<generate-with-secrets.token_urlsafe(32)>"
python -m mcp_server
# Default: http://127.0.0.1:8765/mcp

# Start the orchestrator UI
uv run streamlit run orchestrator/ui/app.py
# Default: http://localhost:8501
```

## 6. Testing strategy

- **No top-level tests** exist for the upstream 4DGS core.
- **All automated tests live in `orchestrator/tests/`** and are configured in `orchestrator/pyproject.toml`.

```bash
cd orchestrator

# Sandbox tests (fake Docker, no GPU needed)
pytest -q

# Real GPU + Docker checks (opt-in, can be slow due to image build/pull)
$env:PIPELINE_TEST_GPU = "1"
pytest -q -s tests/test_containers_gpu.py

# Real Isaac image/container checks
$env:PIPELINE_TEST_GPU = "1"
$env:PIPELINE_TEST_ISAAC = "1"
pytest -q -s tests/test_containers_gpu.py

# End-to-end Isaac prep/capture/amp chain on real hardware
$env:PIPELINE_TEST_ISAAC = "1"
pytest -q -s tests/test_stages_isaac_gpu.py
```

- `motion_seg/segment_rigid.py` has a built-in `--selftest` that verifies on a synthetic 7-body scene with no GPU (expected ARI ≈ 0.999).
- `omniverse_pipeline/rig.py` also supports `--selftest`.
- GPU/Isaac tests auto-skip unless the corresponding environment flags are set.

## 7. Development conventions

These conventions are locked in `orchestrator/planning/INSTRUCTIONS.md` and apply especially to the orchestrator, but the mindset is useful across the repo:

- **Copy the logic in, don't call the original script.** `omniverse_pipeline/`, `motion_seg/`, and repo-root scripts are reference/testing code. Orchestrator stages must not shell out to them or `sys.path`-hack imports. Verified logic is vendored into `orchestrator/pipeline/vendored/{host,cuda,isaac}/`.
- **Config is the single source of truth.** New experiments are declared as YAML presets under `orchestrator/pipeline/config/presets/` (layered via `extends:`), not as new `.sh` files or scattered `arguments/*.py` overrides.
- **Path translation lives in exactly one module:** `orchestrator/pipeline/paths.py`. Do not hardcode `Q:\`, `/workspace`, or `/omniverse` elsewhere.
- **Light package imports.** Do not import `torch`, `docker`, `pynvml`, or `psutil` at module scope inside the orchestrator; import them inside functions to keep Layer 1 importable in sandbox tests.
- **One task at a time.** The orchestrator is tracked in `orchestrator/planning/TASKS.md` and per-task specs under `orchestrator/planning/tasks/`. Update a task's status header as you work.
- **Every task ends with verification.** CPU-only work must be verifiable in the sandbox; GPU work gets a real-hardware checklist or test.
- **Keep old `.sh` scripts working** until orchestrator parity is reached.

## 8. Security considerations

- **MCP server bearer token.** The HTTP MCP server requires `PIPELINE_MCP_TOKEN`. Generate it with `secrets.token_urlsafe(32)` and treat it like an API key. Do not commit tokens or hardcode defaults.
- **Docker socket access.** The orchestrator drives Docker Desktop directly. Running it grants container-management privileges equivalent to the user account.
- **Path traversal.** The orchestrator resolves external artifact paths (`raw_mesh`, `gt_segmentation`, capture directories). Do not pass untrusted paths into `run_pipeline`/`run_stage` without validation.
- **Native subprocess execution.** `capture.isaac` executes Isaac Sim's `python.bat` as a native Windows subprocess. Ensure `PIPELINE_ISAAC_NATIVE_PYTHON` points to a trusted binary.
- **CUDA extension builds.** The editable submodules compile native CUDA code at install time. Builds happen inside the local repo; do not point the build at untrusted source trees.

## 9. Common pitfalls / gotchas

- **`docker build` has no GPU**, so `torch.utils.cpp_extension` cannot auto-detect compute capability. The Dockerfile sets `TORCH_CUDA_ARCH_LIST="8.6+PTX"` for an RTX 3090. Missing this causes `IndexError: list index out of range` during the build.
- **The `cuda` Dockerfile builds the venv in `/opt/build`, not `/workspace`**, because `/workspace` is bind-mounted from the live repo at runtime and would shadow anything built there. Do not move the build back into `/workspace`.
- **`requirements.txt` at the repo root is stale** (torch 1.13.1, mmcv 1.6.0). The authoritative dependency set is root `pyproject.toml` + `uv.lock`.
- **`motion_seg/checkpoint-best.pth.tar` is not used.** The orchestrator expects the MultiBodySync checkpoint at `submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar`, which must be downloaded manually.
- **Option-A segmentation (`mbs_infer.py`) is written but not yet verified on real GPU data.** Option B (`segment_rigid.py`) is the current default.
- **The pump01 scene is the primary real-hardware benchmark:** 107 rigid parts, 10 cameras, 60 frames, mm-scale periodic motion.

## 10. Where to read more

- `README.md` — short project intro and file attribution
- `docs/README.md` — documentation index
- `docs/overview.md` — project goal, repo map, current status
- `docs/motion-segmentation.md` — segmentation design and results
- `docs/omniverse-pipeline.md` — synthetic-data pipeline details
- `docs/orchestrator.md` — orchestrator architecture and milestone history
- `orchestrator/planning/ARCHITECTURE.md` — single source of truth for orchestrator design
- `orchestrator/planning/TASKS.md` — task board and dependency graph
- `orchestrator/planning/WINDOWS_SETUP.md` — one-time machine setup
- `.claude_notes/` — chronological working notes
