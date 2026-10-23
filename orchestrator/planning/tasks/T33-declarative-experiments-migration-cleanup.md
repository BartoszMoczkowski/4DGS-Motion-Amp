# T33 — Declarative Experiments Migration & Script Retirement

- Status: todo
- Phase: 11 (streamlining / cleanup)
- Depends on: T28b, T32, T30
- Environment: sandbox-testable (YAML validation + manifest parsing); filesystem verification on Bartosz's machine

## Goal

Migrate the Capture-Limitation suite and Lego-Turbine recordings to declarative experiment YAML specs under `experiments/` run via `amp exp run`. Clean up one-off batch scripts from `scripts/` and scratch artifacts from the workspace root, solidifying `runs/experiment_results.csv` as the unified evaluation store.

## Problem & Background

In recent runs, ad-hoc Python batch scripts were created to drive sweeps and captures, generating redundant state files in `runs/` (`batch_train_capture_limitation_status.json`, `capture_limitation_trained_runs.json`, `lego_turbine_pipeline_status.json`, `lego_turbine_evaluation_summary.json`). This broke the T27 declarative experiment contract, fractured results into separate CSVs, and left untracked scratch images in the repository root.

## In scope

- **Declarative Experiment Specs**:
  - `experiments/capture_limitation.yaml`: Declares the 18 cells (Series A, R, T, F), capture settings, presets, overrides, and segmentation impls (`rigid`, `rigid2`).
  - `experiments/lego_turbine.yaml`: Declares the 3 physical turbine variants (`base`, `2_missing`, `5_missing`).
- **Unified Store Aggregation**:
  - Ensure all variants and impl evaluations are exported into `runs/experiment_results.csv` keyed by `(run_id, impl)` with accurate statuses.
- **Script Retirement & Archival**:
  - Move/archive redundant batch drivers from `scripts/`:
    `batch_capture_series_a.py`, `batch_capture_series_r.py`, `batch_capture_series_tf.py`, `batch_train_capture_limitation.py`, `batch_seg_capture_limitation.py`, `run_lego_turbine_pipeline.py`.
  - Add a clear deprecation/archival header pointing to `amp exp run`.
- **Workspace Root Hygiene**:
  - Move untracked scratch images (`scratch_test_cc_sA0.png`, `scratch_test_proj.png`, `test_tmp/`, etc.) into `scratch/` or delete temporary test outputs.
  - Consolidate legacy status JSON files in `runs/`.

## Out of scope

- Re-running 4DGS training for already completed models (re-uses existing checkpoints).

## Deliverables

- `experiments/capture_limitation.yaml` and `experiments/lego_turbine.yaml`.
- Fully populated and validated `runs/experiment_results.csv`.
- Cleaned workspace root and retired `scripts/` batch drivers.
