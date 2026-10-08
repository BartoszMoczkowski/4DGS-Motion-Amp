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

(2026-10-05 scope note: `render_amp.py`'s `amp_factors` are per *parameter channel*, not per part — there is no per-motion-segment amplification in `render_amp.py` itself; per-part amplification flows through the orchestrator's seg pipeline (`seg_extract → segment.* → amp`).)

Synthetic data is the enabler: real captures have no ground-truth camera poses or per-part labels, so quantitative evaluation (ARI / IoU) is only possible on scenes authored in Omniverse.

The codebase is a fork of [4DGaussians](https://github.com/hustvl/4DGaussians), reorganized as a `uv` workspace of separately installable packages. Most of the 4DGS code in `core/` is upstream; the author's additions are concentrated in:

- `core/render_amp.py`, `core/motion_amp/renderer.py`
- `amp-ui/amp_ui/` (`ampUI.py`, `cameras.py`, `run_renders_auto.py`)
- `omniverse-pipeline/omniverse_pipeline/`
- `motion-seg/motion_seg/`
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
| UI | Streamlit (`amp-ui/amp_ui/ampUI.py` for the standalone workflow; `orchestrator/ui/` for the orchestrator) |
| Testing | `pytest` (only in `orchestrator/tests/`) |

## 3. Repository layout

| Path | Purpose |
|------|---------|
| `core/` (package `4dgs-core`) | Upstream 4DGS core + author's motion amp: `train.py`, `render.py`, `render_amp.py`, `export_perframe_3DGS.py`, `merge_many_4dgs.py`, `scene/`, `gaussian_renderer/`, `utils/`, `arguments/` (configs), `lpipsPyTorch/`, `motion_amp/`. torch/CUDA heavy |
| `core/render_amp.py` | **Motion-amplified rendering**. Extracts per-frame Gaussian parameters, applies FFT-based amplification (Eulerian / absolute / chunked variants), then renders a video. 2026-10-05: all four methods were reimplemented as mean-anchored Eulerian *displacement* amplification (`out[t] = mean + a·filter(v[t] − mean)`; previously `eulerian*` amplified frame-to-frame velocity, so effective gain was ≪ `a`). `amp_factors` is per **parameter channel** (8 slots), not per motion part — the "segmented" method names mean Gaussian-chunked memory processing, not segmentation masks; per-part amplification flows through the orchestrator's seg pipeline |
| `core/motion_amp/renderer.py` | Low-level helper used by `render_amp.py`; returns raw pre-rasterization Gaussian parameters |
| `amp-ui/` (package `amp-ui`) | `amp_ui/ampUI.py` (standalone Streamlit UI for `render_amp.py`), `amp_ui/cameras.py` (USB multi-camera recorder, OpenCV), `amp_ui/run_renders_auto.py` (benchmark harness writing `results.csv`) |
| `omniverse-pipeline/` (package `omniverse-pipeline`) | Isaac Sim capture + USD prep + conversion to 4DGS `multipleview` format; scripts in `omniverse-pipeline/omniverse_pipeline/` |
| `scene-gen/` | `gen_scenes.py` — parametric grid generator of pump test scenes (base motion amplitude × amplification multiplier; metallic per-part materials). Plain python + usd-core; reuses `omniverse_pipeline.add_motion` / `compose_scene`. Outputs to `omniverse-pipeline/data/scenes/grid/`. Also `frames_to_mp4.py` (capture frames → mp4 preview) and `run_grid_4dgs.py` (batch-runs captured grid cells through the orchestrator's convert/train/render DAG + a Gaussian-count sweep with densification frozen; results in `runs/grid_4dgs_results.csv`), and `run_grid_seg.py` (baseline Option-B rigidity-graph segmentation over the trained grid/sweep models via the orchestrator's seg_extract/segment.rigid/seg_eval stages; results in `runs/grid_seg_results.csv`) |
| `motion-seg/` (package `motion-seg`) | Motion segmentation (pure CPU base): rigidity-graph clustering (`segment_rigid.py`) + MultiBodySync adapter (`mbs_infer.py`) + evaluation; Python package in `motion-seg/motion_seg/` |
| `orchestrator/` | Three-layer pipeline system: DAG execution (`pipeline/`), HTTP MCP server (`mcp_server/`), Streamlit UI (`ui/`) |
| `submodules/` | `depth-diff-gaussian-rasterization`, `simple-knn`, `multibody-sync-4dgs` |
| `data/` | D-NeRF synthetic scenes (`bouncingballs`, `lego`, `mutant`, ...) + generated `multipleview` scenes (e.g. `pump01`) |
| `output/` | Trained 4DGS models (not committed due to size) |
| `docs/` | Compiled documentation (`overview.md`, `motion-segmentation.md`, `omniverse-pipeline.md`, `orchestrator.md`, ...) |
| `.claude_notes/` | Chronological working notes (primary detailed record) |
| `LLFF/`, `viseron/` | Bundled reference tools and an unrelated NVR setup respectively |

## 4. Build and runtime setup

### 4.1 Workspace packages

**A plain `uv sync` at the repo root is the default install** — the root project depends on every workspace member, so one sync installs the whole project into `.venv` (GPU torch from the cu126 index by default):

```bash
# Everything (default, GPU)
uv sync

# CPU-only torch: sync normally, then swap the torch build in place
# (uv 0.9.13 cannot express "unconditional cu126 default + opt-in cpu fork"
# in one lockfile — see .claude_notes T24 entry)
uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Single package only (special environments, e.g. the Docker build)
uv sync --package 4dgs-core

# Orchestrator test tooling (pytest) lives in the root `test` dependency group
uv run --group test pytest -q   # from orchestrator/
```

Key editable workspace members (declared in root `pyproject.toml`):

- `submodules/depth-diff-gaussian-rasterization` → package `diff_gaussian_rasterization`
- `submodules/simple-knn` → package `simple_knn`
- `submodules/multibody-sync-4dgs` → package `mbs-bootstrap` (a shim that puts the submodule on `sys.path`; MBS's own flat top-level modules are intentionally NOT installed — its `utils` would collide with core's)
- `orchestrator` → package `pipeline`
- `scene-gen` → package `scene-gen` (flat py-modules + `identity/` package)

These submodules are **CUDA extensions** built by `torch.utils.cpp_extension`. They compile on first `uv sync --package 4dgs-core`. The Dockerfile hard-codes `TORCH_CUDA_ARCH_LIST="8.6+PTX"` for the author's RTX 3090; adjust if you target a different GPU.

### 4.2 Orchestrator

The orchestrator is its own workspace package (`pipeline`) with opt-in extras:

```bash
# Layer 1: DAG engine, artifacts, container manager, tests
uv sync --package pipeline

# Layer 2: HTTP MCP server
uv sync --package pipeline --extra mcp

# Layer 3: Streamlit UI
uv sync --package pipeline --extra ui
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

This is the **accepted remaining multi-environment case** (T24): after the uv workspace restructure, the root `.venv` covers every other workflow, but Isaac Sim's own `python.bat` can never share the project venv (it bundles its own Python 3.11 runtime with `pxr`/`omni`). USD-authoring scripts in `omniverse-pipeline` (`split_mesh.py`, `add_motion.py`, `omni_capture.py`) and `scene-gen` (`gen_scenes.py`, `gen_cubes_dataset_all.py`, `build_gt_motion_classes.py`, `export_cube_gt_pointcloud.py`) therefore run under the Isaac Python; everything else runs in the workspace venv.

First-time Windows machine setup is documented in `orchestrator/planning/WINDOWS_SETUP.md`.

## 5. Key commands

### 5.1 Training

```bash
uv run --package 4dgs-core python core/train.py -s data/dnerf/lego -m output/dnerf/lego --configs core/arguments/dnerf/lego.py
uv run --package 4dgs-core python core/train.py -s data/multipleview/pump01 -m output/multipleview/pump01 --configs core/arguments/multipleview/default.py
```

Training runs in two stages: coarse (static Gaussians) then fine (deformation network enabled). Final artifacts are `point_cloud/iteration_N/point_cloud.ply` + `deformation.pth`.

### 5.2 Standard rendering

```bash
uv run --package 4dgs-core python core/render.py -m output/dnerf/lego --iteration 20000 --skip_train
```

### 5.3 Motion-amplified rendering

```bash
uv run --package 4dgs-core python core/render_amp.py -m output/dnerf/lego --configs core/arguments/dnerf/lego.py \
    --amp_factors 2 -1 -1 -1 -1 -1 -1 -1 \
    --freq_low 0.0 --freq_high 1.0 \
    --method eulerian --video_path out.mp4
```

Since 2026-10-05 `render_amp.py` validates its arguments (`validate_amp_args`: `amp_factors` must be exactly 8 values with `-1` as the only allowed negative / skip sentinel; `--freq_low`/`--freq_high` length 1 or 8 with broadcast) and `load_config` imports `mmengine.config.Config` first with an `mmcv` fallback (the pinned `mmcv==2.2.0` no longer ships `Config`). Unknown `--method` values now raise instead of silently rendering unamplified.

### 5.4 Motion segmentation (reference scripts)

```bash
# Option B: rigidity-graph clustering (CPU, default); extract_trajectories is GPU (needs the `core` extra)
uv run --package motion-seg --extra core python -m motion_seg.extract_trajectories --model_path output/multipleview/pump01 --configs core/arguments/multipleview/pump01.py
uv run --package motion-seg python -m motion_seg.segment_rigid --trajectories output/multipleview/pump01/trajectories.npz --out output/multipleview/pump01/segmentation.npz
uv run --package motion-seg python -m motion_seg.evaluate_segmentation --pred output/multipleview/pump01/segmentation.npz --gt data/multipleview/pump01/gt_segmentation.npz

# Convenience wrapper
./motion-seg/motion_seg/run.sh pump01
```

### 5.5 Omniverse pipeline (reference scripts)

```bash
# USD prep (plain Python)
python omniverse-pipeline/omniverse_pipeline/split_mesh.py --in "CONJUNTO BOMBAS.usd" --out CONJUNTO_BOMBAS_segmented.usd --group CONJUNTO_BOMBAS
python omniverse-pipeline/omniverse_pipeline/add_motion.py --in CONJUNTO_BOMBAS_segmented.usd --out CONJUNTO_BOMBAS_animated.usd

# Capture (must use Isaac Sim's own Python, native Windows)
Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat omniverse-pipeline/omniverse_pipeline/omni_capture.py --config omniverse-pipeline/omniverse_pipeline/capture_config_pump.yaml

# Convert to 4DGS multipleview format
python omniverse-pipeline/omniverse_pipeline/omni_to_4dgs.py --capture <capture_dir> --out . --name pump01
```

### 5.6 Orchestrator

```bash
# Run the full automated DAG (from Python; see orchestrator/planning/ARCHITECTURE.md for the API)
python -c "from pipeline.api import run_pipeline; ..."

# Start the MCP server
$env:PIPELINE_MCP_TOKEN = "<generate-with-secrets.token_urlsafe(32)>"
uv run --package pipeline --extra mcp python -m mcp_server
# Default: http://127.0.0.1:8765/mcp

# Start the orchestrator UI
uv run --package pipeline --extra ui streamlit run orchestrator/ui/app.py
# Default: http://localhost:8501

# Standalone motion-amp UI (reference workflow)
uv run --package amp-ui streamlit run amp-ui/amp_ui/ampUI.py
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

- `motion-seg/motion_seg/segment_rigid.py` has a built-in `--selftest` that verifies on synthetic 7-body scenes with no GPU. 2026-10-05: the selftest was reworked — the legacy disjoint-parts scene (which never exercised edge-cutting) is kept as Case A (sanity, bar ARI > 0.99); the primary Case B uses adjacent, jittered parts whose kNN graph bridges part boundaries, genuinely requiring rigidity edge-cutting (bar ARI ≥ 0.99; fixture scores ARI 1.0, cutting nothing yields ≈ 0.84).
- `omniverse-pipeline/omniverse_pipeline/rig.py` also supports `--selftest`.
- GPU/Isaac tests auto-skip unless the corresponding environment flags are set.

## 7. Development conventions

These conventions are locked in `orchestrator/planning/INSTRUCTIONS.md` and apply especially to the orchestrator, but the mindset is useful across the repo:

- **Copy the logic in, don't call the original script.** `omniverse-pipeline/omniverse_pipeline/`, `motion-seg/motion_seg/`, and `core/` scripts are reference/testing code. Orchestrator stages must not shell out to them or `sys.path`-hack imports. Verified logic is vendored into `orchestrator/pipeline/vendored/{host,cuda,isaac}/`. The vendored copies are kept identical to the reference apart from intentional divergences documented in the vendored file headers (e.g. `vendored/cuda/amp.py` after the 2026-10-05 motion-amp fixes; `vendored/host/seg_eval.py`/`metrics.py` re-synced to the fixed scoring conventions on 2026-10-06). Keep reference and vendored copies in sync when fixing bugs — several 2026-10-05 review bugs existed in both.
- **Config is the single source of truth.** New experiments are declared as YAML presets under `orchestrator/pipeline/config/presets/` (layered via `extends:`), not as new `.sh` files or scattered `core/arguments/*.py` overrides.
- **Path translation lives in exactly one module:** `orchestrator/pipeline/paths.py`. Do not hardcode `Q:\`, `/workspace`, or `/omniverse` elsewhere.
- **Light package imports.** Do not import `torch`, `docker`, `pynvml`, or `psutil` at module scope inside the orchestrator; import them inside functions to keep Layer 1 importable in sandbox tests.
- **One task at a time.** The orchestrator is tracked in `orchestrator/planning/TASKS.md` and per-task specs under `orchestrator/planning/tasks/`. Update a task's status header as you work.
- **Every task ends with verification.** CPU-only work must be verifiable in the sandbox; GPU work gets a real-hardware checklist or test.
- **Keep old `.sh` scripts working** until orchestrator parity is reached.

## 8. Security considerations

- **MCP server bearer token.** The HTTP MCP server requires `PIPELINE_MCP_TOKEN`. Generate it with `secrets.token_urlsafe(32)` and treat it like an API key. Do not commit tokens or hardcode defaults.
- **Docker socket access.** The orchestrator drives Docker Desktop directly. Running it grants container-management privileges equivalent to the user account.
- **Path traversal.** The orchestrator resolves external artifact paths (`raw_mesh`, `gt_segmentation`, capture directories). Since 2026-10-05 these are gated at seeding time (`validate_external_artifact_path`: absolute, must exist, confined to the runs/repo/assets roots) and at MCP serving time (`resolve_servable_artifact_path`); `run_id` and preset names are charset- and confinement-validated (`validate_run_id` / `validate_preset_name`), and `stop_container` refuses containers without the `pipeline.managed` label. Do not pass untrusted paths into `run_pipeline`/`run_stage` regardless.
- **Native subprocess execution.** `capture.isaac` executes Isaac Sim's `python.bat` as a native Windows subprocess. Ensure `PIPELINE_ISAAC_NATIVE_PYTHON` points to a trusted binary.
- **CUDA extension builds.** The editable submodules compile native CUDA code at install time. Builds happen inside the local repo; do not point the build at untrusted source trees.

## 9. Common pitfalls / gotchas

- **`uv sync` at the root installs everything** (the root project depends on all workspace members). `uv sync --package <name>` is now the exception, for special environments (the Docker build). After any member pyproject change, re-run `uv lock` and check `uv lock --check` before committing.
- **The CUDA extensions are linux-only dependencies** (`diff-gaussian-rasterization` / `simple-knn` carry `sys_platform == 'linux'` markers in `core/pyproject.toml`): there is no CUDA toolkit on the Windows host, so they only build inside the Docker image. Consequently core's renderer modules (`scene`, `gaussian_renderer`, `train`, `render_amp`, ...) are not importable in a Windows venv — they never were. `scripts/smoke_workspace_imports.py` treats them as expected skips off-linux.
- **torchvision is pinned by direct wheel URL on win32/linux** (root `[tool.uv.sources]`) because the cu126 index publishes the cp312 win_amd64 wheel of 0.28.0+cu126 without a `#sha256` fragment, which trips uv's hash verification; the URL pins also keep torch at the validated 2.13.0+cu126 on every desktop platform.
- **`docker build` has no GPU**, so `torch.utils.cpp_extension` cannot auto-detect compute capability. The Dockerfile sets `TORCH_CUDA_ARCH_LIST="8.6+PTX"` for an RTX 3090. Missing this causes `IndexError: list index out of range` during the build.
- **The `cuda` Dockerfile builds the venv in `/opt/build`, not `/workspace`**, because `/workspace` is bind-mounted from the live repo at runtime and would shadow anything built there. Do not move the build back into `/workspace`.
- **The pre-uv `requirements.txt` files were deleted in T24** (root, torch 1.13.1 era; `submodules/multibody-sync-4dgs`, `open3d==0.11.2` era). The authoritative dependency set is the workspace `pyproject.toml` files + `uv.lock`. The mmcv pin remains a cu121/torch2.4 wheel URL (no cu126/torch≥2.5 cp312 wheel exists on download.openmmlab.com as of 2026-10-06 — known risk, Linux/Docker only).
- **`motion-seg/motion_seg/checkpoint-best.pth.tar` is not used.** The MultiBodySync checkpoint lives at `submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar` (downloaded from the Google Drive link in `orchestrator/planning/WINDOWS_SETUP.md` §7; gitignored, not vendored).
- **Option-A segmentation (`mbs_infer.py`) now runs on real data** (checkpoint downloaded to the expected path; ran on the 7 grid/sweep pump models via `scene-gen/run_grid_seg.py --impl mbs`), but segments poorly on these scenes (ARI ≈ 0, 2–8 clusters vs 107 GT parts — MotNet is out-of-distribution for mm-scale 4DGS trajectories, see `orchestrator/planning/WINDOWS_SETUP.md` §7). 2026-10-05 caveat: the published ARI ≈ 0 numbers substantially measure an evaluation artifact — the preset labels only 4 000 points, so ≥98% of points are labeled −1 and scored as one giant segment (`drop_floaters=False`); the out-of-distribution conclusion may hold but is not isolated by those numbers. Re-score the existing `segmentation_mbs.npz` artifacts with `--drop-floaters` before citing them. Option B (`segment_rigid.py`) remains the default.
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
