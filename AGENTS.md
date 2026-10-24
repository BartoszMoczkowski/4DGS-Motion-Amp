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
| `scene-gen/` | `gen_scenes.py` — parametric grid generator of pump test scenes (base motion amplitude × amplification multiplier; metallic per-part materials). Plain python + usd-core; reuses `omniverse_pipeline.add_motion` / `compose_scene`. Outputs to `omniverse-pipeline/data/scenes/grid/`. Also `frames_to_mp4.py` (capture frames → mp4 preview). The old grid/sweep batch drivers (`run_grid_4dgs.py` / `run_grid_seg.py`) were deleted in T27 — the grid is now one declarative spec + `amp exp run experiments/pump_grid_v2.yaml` (see §5.6) |
| `motion-seg/` (package `motion-seg`) | Motion segmentation (pure CPU base): rigidity-graph clustering (`segment_rigid.py`) + MultiBodySync adapter (`mbs_infer.py`) + evaluation; Python package in `motion-seg/motion_seg/` |
| `orchestrator/` | Three-layer pipeline system: DAG execution (`pipeline/`), HTTP MCP server (`mcp_server/`), Streamlit UI (`ui/`) |
| `submodules/` | `depth-diff-gaussian-rasterization`, `simple-knn`, `multibody-sync-4dgs` |
| `data/` | D-NeRF synthetic scenes (`bouncingballs`, `lego`, `mutant`, ...) + generated `multipleview` scenes (e.g. `pump01`) |
| `runs/` | Canonical orchestrator run tree (`runs/<run_id>/` with `manifest.json`, `config_snapshot.json`, `logs/`, and stage output dirs) |
| `experiments/` | Declarative experiment specifications (`pump_grid_v2.yaml`, `spinning_cubes.yaml`, ...) executed via `amp exp run` |
| `output/` | Legacy trained model directory (from upstream 4DGaussians direct scripts; superseded by `runs/`, migration guide in `docs/migration_legacy_output.md`) |
| `scratch/` | Unsupported one-off scripts, diagnostic harnesses, and retired ad-hoc code (see `scratch/README.md`) |
| `docs/` | Compiled documentation (`overview.md`, `motion-segmentation.md`, `omniverse-pipeline.md`, `orchestrator.md`, `migration_legacy_output.md`, ...) |
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

The unified entry point is the `amp` CLI (T25/T26; installed by `uv sync`, run via `uv run amp`):

```bash
amp --help   # groups: capture, train, render, seg, exp, orch, bench, mlflow; report (T29) reserved
```

```bash
# Local MLflow tracking (T28): every run logs to the local server (manifest stays source of
# truth; joined by the pipeline.run_id tag). Server down -> warning only, runs never fail.
uv run amp mlflow serve     # SQLite store runs/mlflow/mlflow.db, UI http://127.0.0.1:5000
uv run amp mlflow status    # resolved URI, reachability, run count
uv run amp mlflow prune --older-than 30d
# Env: PIPELINE_MLFLOW_TRACKING_URI (file:///... for offline), PIPELINE_MLFLOW_DISABLE=1,
# PIPELINE_MLFLOW_ROOT. Details: docs/orchestrator.md "MLflow tracking (T28)".
```

Every pipeline-executing command supports `--dry-run` (prints the resolved stage + config slice, executes nothing). The raw `python <script>.py` invocations below remain only as the in-container exec paths the Docker stages use (see `pipeline/stages/cuda_common.py`) — they are no longer the user path.

### 5.1 Training

```bash
uv run amp train pump01 --scene data/multipleview/pump01   # scene seeded as external artifact
uv run amp train pump01 --dry-run                          # resolved train.default config slice
```

Runs `run_pipeline(preset, only=["train.default"])` (coarse static stage, then fine deformation stage). Final artifacts are `point_cloud/iteration_N/point_cloud.ply` + `deformation.pth` under the run's `train_out/`. (Raw `core/train.py -s ... -m ... --configs ...` persists inside the `cuda` container, exec'd by the `train.default` stage.)

### 5.2 Standard rendering

```bash
uv run amp render <run_id> [--force]
```

### 5.3 Motion-amplified rendering

```bash
uv run amp render <run_id> --amp-factors "2 -1 -1 -1 -1 -1 -1 -1" --method eulerian \
    --freq-low 0.0 --freq-high 1.0 --video out.mp4 [--low-vram] [--frozen-cam]
# slot order: pos3d pos2d scale rotation opacity SHs color cov3D
```

Presence of `--amp-factors` switches the stage from `render.default` to `amp.default`; factors/freqs map onto `amp.channels` config overrides (T26). Validation mirrors `render_amp.py`'s `validate_amp_args` (exactly 8 values, `-1` the only negative / skip sentinel; freq lists length 1 or 8). click can't take argparse-style space-separated option values: quote the list (`"2 -1 ..."`), use `--amp-factors="..."` when it starts with a negative, or repeat the flag per value. Since 2026-10-05 `render_amp.py` validates its arguments (`validate_amp_args`) and `load_config` imports `mmengine.config.Config` first with an `mmcv` fallback (the pinned `mmcv==2.2.0` no longer ships `Config`). Unknown `--method` values raise instead of silently rendering unamplified.

### 5.4 Motion segmentation

```bash
# Full chain (replaces the deleted motion-seg/motion_seg/run.sh; SKIP_EXTRACT=1 -> --skip-extract)
uv run amp seg run pump01 [--impl rigid|rigid2|kabsch|multicut|mbs] [--skip-extract] [--dry-run]

# Individual stages on an existing run (extract is GPU; the rest are CPU)
uv run amp seg extract <run_id> [--n-times 60]
uv run amp seg rigid <run_id> [--k 12 --min-size 15 --threshold-mult 1.0 --opacity-thresh 0.1]
uv run amp seg rigid2|kabsch|multicut|mbs <run_id> [impl flags]
uv run amp seg eval <run_id> [--drop-floaters] [--bg-label 0|auto]
```

Flags map to `segment.<impl>.*` / `seg_extract.*` / `seg_eval.*` config overrides (unset flags never clobber preset values). The reference modules (`motion_seg.extract_trajectories`, `segment_rigid`, `evaluate_segmentation`, `mbs_infer`) remain for their `--selftest`s and direct scripting, but routine work goes through `amp seg`.

### 5.5 Omniverse pipeline

```bash
uv run amp capture split --in "CONJUNTO BOMBAS.usd" --out CONJUNTO_BOMBAS_segmented.usd --group CONJUNTO_BOMBAS [--min-faces N --preview PNG --no-usd]
uv run amp capture motion --in CONJUNTO_BOMBAS_segmented.usd --out CONJUNTO_BOMBAS_animated.usd [--num-frames --fps --trans-amp-mm A B --groups K --exclude ...]
uv run amp capture gen-grid [--base-amp-mm ... --multipliers ... --only-cell 16x8 ...]   # scene-gen gen_scenes.py

# Capture: native Windows Isaac Sim subprocess only (Vulkan unsupported under WSL2/Docker).
# Wraps the same Layer-1 native mechanism (run_native_isaac_script); override the launcher
# with PIPELINE_ISAAC_NATIVE_PYTHON.
uv run amp capture run --config omniverse-pipeline/omniverse_pipeline/capture_config_pump.yaml [--usd X --out DIR --n-cameras N --frames N --headless 0|1]

uv run amp capture convert --capture <capture_dir> --out . --name pump01 [--near --far --target-radius]
uv run amp capture preview --capture <capture_dir> [--fps 24 --tile-cols 5]
uv run amp capture verify <capture_dir> [--cams 10 --frames 60 | --config <capture.yaml>]   # frame-count QA
```

### 5.6 Orchestrator

```bash
# Python API (orchestrator/planning/ARCHITECTURE.md):
#   from pipeline.api import run_pipeline
#   run_pipeline("pump01", overrides={...})

# Start the MCP server (never defaults the token)
$env:PIPELINE_MCP_TOKEN = "<generate-with-secrets.token_urlsafe(32)>"
uv run amp orch mcp [--host 127.0.0.1] [--port 8765]
# Default: http://127.0.0.1:8765/mcp

# UIs
uv run amp orch ui                  # orchestrator UI, http://localhost:8501
uv run amp orch ui --app amp        # standalone motion-amp UI (ampUI.py)

# Render benchmark (methods x VRAM modes x models x repeats; results.csv + run-registry record)
# Linux/CUDA-only: run_renders_auto imports core/render_amp.py natively (torch + simple_knn),
# so on the Windows host it exits 1 with a one-line explanation; run it inside the cuda
# container/Linux instead (containerization is future work, see the T26 task log).
uv run amp bench renders [--models ds/name ...] [--methods "base,abs segmented"] [--repeats 5] [--vram-modes full,low]
```

### 5.7 Experiments (T27)

A grid/sweep experiment is one declarative YAML under `experiments/` + one command — this
replaced the deleted `scene-gen/run_grid_4dgs.py` / `run_grid_seg.py` batch drivers:

```bash
uv run amp exp list                                            # specs under experiments/
uv run amp exp run experiments/pump_grid_v2.yaml --dry-run     # full plan, touches nothing
uv run amp exp run pump_grid_v2 [--only-cells a,b] [--only-impls rigid,kabsch]
uv run amp exp status pump_grid_v2                             # per-variant run status
uv run amp exp compare pump_grid_v2 [--metric ari]             # impl x variant pivot
uv run amp exp export [--out runs/experiment_results.csv]      # rebuild the results store
```

`experiments/pump_grid_v2.yaml` is the reference spec (pump amplitude×multiplier grid + Gaussian-count sweep + the 8-impl segmentation matrix). Runs are created via `run_pipeline` with run IDs `<experiment>-<variant>-<8hex>`; idempotency is by run-manifest metadata (`experiment`/`cell`/`variant`), never `force=True`. Results land in the unified store `runs/experiment_results.csv` keyed `(run_id, impl)`, re-derivable from manifests at any time via `amp exp export`. Per-impl seg eval results are keyed `seg_eval_result_<impl>.json` inside each run (Phase A).

### 5.8 Unified Exploration (T30)

Single-command experiment, run, configuration, and artifact navigator:

```bash
uv run amp explore                                       # workspace overview (experiments, runs, captures)
uv run amp explore <run_id>                             # inspect specific run (status, timeline, artifacts, metrics)
uv run amp explore <experiment_name>                    # inspect experiment (variant progress, cell matrix)
uv run amp explore <target> --view full                 # full configuration and detailed metrics
uv run amp explore <target> --human                     # format as human-readable markdown tables
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
- **The repo-root `.dockerignore` is load-bearing** (added 2026-10-09): the `cuda` Dockerfile's build context must stay at tens of MB (Dockerfile, `pyproject.toml`, `uv.lock`, `core/`, `submodules/` minus MBS `ckpt/`, and member `pyproject.toml` manifests). Without it the context is the whole repo (100+ GB — `runs/`, `data/`, `output/`, `.venv/`), and `ensure_image`'s automatic rebuild blocks for hours just *sending* the context. Keep new bulk-data dirs excluded; if the Dockerfile gains a `COPY`, re-include that path with a `!` negation.
- **The `cuda` Dockerfile builds the venv in `/opt/build`, not `/workspace`**, because `/workspace` is bind-mounted from the live repo at runtime and would shadow anything built there. Do not move the build back into `/workspace`.
- **The pre-uv `requirements.txt` files were deleted in T24** (root, torch 1.13.1 era; `submodules/multibody-sync-4dgs`, `open3d==0.11.2` era). The authoritative dependency set is the workspace `pyproject.toml` files + `uv.lock`. The mmcv pin remains a cu121/torch2.4 wheel URL (no cu126/torch≥2.5 cp312 wheel exists on download.openmmlab.com as of 2026-10-06 — known risk, Linux/Docker only).
- **`motion-seg/motion_seg/checkpoint-best.pth.tar` is not used.** The MultiBodySync checkpoint lives at `submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar` (downloaded from the Google Drive link in `orchestrator/planning/WINDOWS_SETUP.md` §7; gitignored, not vendored).
- **Option-A segmentation (`mbs_infer.py`) now runs on real data** (checkpoint downloaded to the expected path; evaluated across the pump grid/sweep models via `amp exp run pump_grid_v2 --only-impls mbs`), but segments poorly on these scenes (ARI ≈ 0, 2–8 clusters vs 107 GT parts — MotNet is out-of-distribution for mm-scale 4DGS trajectories, see `orchestrator/planning/WINDOWS_SETUP.md` §7). 2026-10-05 caveat: the published ARI ≈ 0 numbers substantially measure an evaluation artifact — the preset labels only 4 000 points, so ≥98% of points are labeled −1 and scored as one giant segment (`drop_floaters=False`); the out-of-distribution conclusion may hold but is not isolated by those numbers. Re-score the existing `segmentation_mbs.npz` artifacts with `--drop-floaters` before citing them. Option B (`segment_rigid.py`) remains the default.
- **The pump01 scene is the primary real-hardware benchmark:** 107 rigid parts, 10 cameras, 60 frames, mm-scale periodic motion.
- **Canonical output root is `runs/<run_id>/`**: all orchestrator pipeline stages, CLI commands (`amp explore`, `amp render`, `amp seg`), and local MLflow tracking expect run directories containing a valid `manifest.json`. Legacy directories under `output/multipleview/<scene>` (from direct-script runs) can be wrapped into modern `runs/` using `scripts/migrate_legacy_output.py` (see `docs/migration_legacy_output.md`).
- **Run ID convention**: Run IDs follow `<experiment>-<cell>-<8hex>` (for declarative experiments run via `amp exp run`) or `<preset>-<8hex>` (for standalone runs); characters are restricted to `[A-Za-z0-9_.-]+` and checked by `validate_run_id`.

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
