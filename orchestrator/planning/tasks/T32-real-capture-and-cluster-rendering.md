# T32 — Real-Capture Ingestion & Cluster Video Rendering Stages

- Status: done
- Phase: 9 (streamlining / pipeline extensibility)
- Depends on: T26, T09
- Environment: sandbox-testable (mocked container exec + CLI parsing); real CUDA / container verification on Bartosz's machine
- Completed: 2026-10-24

## Goal

Provide standard orchestrator stages and CLI subcommands for real multi-camera capture ingestion (`amp capture prep-real`) and motion-cluster colored video rendering (`render.cluster` / `amp render --cluster`). Eliminate the need for raw `docker exec` commands into `pipeline-cuda` or ad-hoc runner scripts like `scripts/run_lego_turbine_pipeline.py`.

## Problem & Background

During real-capture evaluations (e.g. Lego turbine series and `turbine_r1_missing_6`):
1. **Raw Container Execution**: When the user requested visual inspection of motion clusters (Gaussians dynamically colored by label across time), agents had to issue raw `docker exec pipeline-cuda python /workspace/core/render_cluster_video.py ...` commands. This bypassed orchestrator monitoring, VRAM tracking, and artifact registration in `manifest.json`.
2. **Ad-Hoc Capture Glue**: Synchronized frames and COLMAP poses were assembled via throwaway Python scripts (`scripts/run_lego_turbine_pipeline.py`) rather than a repeatable pipeline stage or CLI command.
3. **Dual Config Authoring**: Real captures required manually authoring both `core/arguments/multipleview/<name>.py` and `orchestrator/pipeline/config/presets/<name>.yaml`.

## In scope

- **`render.cluster` Stage (`orchestrator/pipeline/stages/render_cluster.py`)**:
  - Registered stage for the `render` role (`impl: cluster`).
  - Inputs: `("model", "segmentation")`, optional `("scene",)`.
  - Outputs: `("cluster_video",)` (`kind: video`).
  - Vendored container logic: `pipeline/vendored/cuda/render_cluster.py` (clean port of `core/render_cluster_video.py`).
  - Automatically registered in `manifest.artifacts`, uploaded to MLflow (under 50 MB), and tracked for peak VRAM.
- **CLI Command `amp render <run_id> --cluster`**:
  - Arguments: `--camera` (default `cam01`), `--side-by-side / --no-side-by-side`, `--add-labels`, `--resolution-scale 0.5`, `--fps 25`.
- **CLI Command `amp capture prep-real`**:
  - Standardizes `scripts/prep_real_multipleview.py` into the `amp capture` CLI group.
  - Subcommands: `--capture <synced_dir> --out <name> [--colmap-source <dir>] [--camera-model SIMPLE_RADIAL]`.
  - Idempotently generates `poses_bounds_multipleview.npy` and `points3D_multipleview.ply`.
- **Config Bridge for Real Captures**:
  - Enable YAML presets under `orchestrator/pipeline/config/presets/` to specify all parameters for real captures so manual creation of `core/arguments/multipleview/<name>.py` is no longer necessary.

## Out of scope

- Real-time OpenGL viewers or GUI Streamlit panels (owned by T15/T30).

## Deliverables
 
- `orchestrator/pipeline/stages/render_cluster.py` + `pipeline/vendored/cuda/render_cluster.py`.
- `amp render <run_id> --cluster` integration in `core/core_cli.py`.
- `amp capture prep-real` command in `omniverse-pipeline/omniverse_pipeline/cli.py` + `prep_real.py`.
- Config bridge for real captures: `RenderClusterConfig` model, `real_capture.yaml` preset, and updated real scene presets.
- Sandbox test suite verifying CLI argument construction and stage artifact declaration.

## Implementation Details

1. **`render.cluster` Stage (`orchestrator/pipeline/stages/render_cluster.py`)**:
   - Registered under `@register("render.cluster")` with `inputs=("model", "segmentation")`, optional `("scene",)`, and `outputs=("cluster_video",)` (`kind: "video"`).
   - Generates bridge argument file via `write_stage_bridge(ctx)` and translates paths into container space.
   - Executes vendored container script `orchestrator/pipeline/vendored/cuda/render_cluster.py` via `run_cuda_script`.
   - Verified in sandbox tests: CLI flag generation, `--source_path` inclusion when `scene` is present, missing output checks, and video artifact declaration.

2. **CLI Command `amp render <run_id> --cluster`**:
   - Added `--cluster`, `--camera`, `--side-by-side / --no-side-by-side`, `--add-labels / --no-add-labels`, `--resolution-scale`, and `--fps` flags to `core/core_cli.py`.
   - Strictly validates mutual exclusivity between `--cluster` and `--amp-factors`.
   - Configures overrides under `render.cluster` and launches `run_stage(run_id, "render.cluster")`.
   - Supports `--dry-run` and `--force`.

3. **CLI Command `amp capture prep-real` (`omniverse-pipeline/omniverse_pipeline/cli.py` & `prep_real.py`)**:
   - Ported and standardized from `scripts/prep_real_multipleview.py` into `omniverse_pipeline.prep_real`.
   - Automatically restructures multi-camera synced frames into `cam{01..10}/frame_{00001..N}.jpg`.
   - Locates COLMAP binary on PATH or via `colmap/COLMAP.bat`.
   - Runs feature extraction, matching, and mapper calibration, or borrows calibration rig from `--colmap-source`.
   - Exports `poses_bounds_multipleview.npy` and `points3D_multipleview.ply`.
   - Idempotently skips COLMAP execution when existing calibration files are present unless `--force` is given.

4. **Config Bridge & Presets**:
   - Added `RenderClusterConfig` and `render.impl = Literal["default", "cluster"]` to `orchestrator/pipeline/config/models.py`.
   - Added `real_capture.yaml` preset establishing standard real-camera defaults:
     - `opacity_reset_interval: 60000`
     - `kplanes_config.resolution: [64, 64, 64, 200]`
     - `seg_extract.n_times: 160`
     - default `render.cluster` settings.
   - Updated real-scene presets (`lego_turbine_base`, `lego_turbine_2_missing`, `lego_turbine_5_missing`, `controler_test_01`, `hand_test_002`, `turbine_r1_missing_6`) to inherit from `real_capture`.

