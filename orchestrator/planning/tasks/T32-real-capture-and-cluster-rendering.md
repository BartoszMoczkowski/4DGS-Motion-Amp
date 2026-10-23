# T32 — Real-Capture Ingestion & Cluster Video Rendering Stages

- Status: todo
- Phase: 9 (streamlining / pipeline extensibility)
- Depends on: T26, T09
- Environment: sandbox-testable (mocked container exec + CLI parsing); real CUDA / container verification on Bartosz's machine

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
- `amp render <run_id> --cluster` integration in `orchestrator/pipeline/render_cli.py`.
- `amp capture prep-real` command in `orchestrator/pipeline/capture_cli.py`.
- Sandbox test suite verifying CLI argument construction and stage artifact declaration.
