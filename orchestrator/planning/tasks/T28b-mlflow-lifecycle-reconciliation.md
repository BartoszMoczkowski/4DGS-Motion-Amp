# T28b — Partial-Run Manifest Status Rollup, MLflow Lifecycle Termination & Run Reconciliation

- Status: done
- Phase: 9 (streamlining / bugfix)
- Depends on: T28
- Environment: sandbox-testable (synthetic manifests + temp MLflow file store); real MLflow SQLite update verified on Bartosz's machine

## Goal

Resolve the lifecycle disconnect where runs executing a subset of preset stages (`only=[...]` or `stage_names`) remain permanently stuck in `"running"` in both `manifest.json` and local MLflow (`mlflow.db`). Provide an idempotent reconciliation tool (`amp mlflow reconcile`) to backfill and terminate all historical runs with full metrics, artifact plots, and correct statuses. Add regression tests for the DAG engine's `gt_segmentation` auto-seeding.

## Problem & Root Cause

1. **Manifest Rollup Bug** (`pipeline/artifacts/manifest.py`):
   When `run_pipeline` is called with a preset (e.g. `pump01`) but filtered by `only=["train.default"]` or stepwise `amp seg`, `new_manifest` registers all 10 stages from the preset as `"pending"`. Because `record_stage_result` checks `all(s.status in ("success", "skipped") for s in m.stages.values())`, `manifest.status` never flips to `"success"` since the unselected stages remain `"pending"`.
2. **MLflow Run Stuck `RUNNING`** (`pipeline/tracking.py`):
   `finish_run()` only sets MLflow status to `FINISHED` when `manifest.status == "success"`. Because `manifest.status` is `"running"`, MLflow runs are left active indefinitely. In our audit, all 18 capture-limitation runs (`pump01-ff77f982` ... `pump01-d5c056eb`) remained in `RUNNING` status in `mlflow.db`.
3. **Missing Runs & Metrics in MLflow**:
   The 4 physical Lego turbine runs had no MLflow tracking entries. Furthermore, rich quantitative metrics computed during evaluation (Cat A/B split rate, pairs split rate, deviant target IoU/F1, displacement percentiles) were not ingested into MLflow.
4. **Untested Engine Diff**:
   An uncommitted diff in `orchestrator/pipeline/dag/scheduler.py` auto-seeds `gt_segmentation` from `scene` artifacts without test coverage.

## In scope

- **Manifest Status Rollup Fix**:
  Update `record_stage_result` and `run_dag` so terminal status is determined based on whether all *scheduled / executed* stages succeeded, rather than requiring unselected preset stages to be complete. If all executed stages are terminal (`success` or `skipped`), the run status is `"success"`.
- **MLflow Termination Hardening**:
  Ensure `track_run_completion` calls `finish_run` with terminal status (`FINISHED`) when the executed stage plan finishes without error.
- **`amp mlflow reconcile` CLI & Tool**:
  Idempotent reconciliation command:
  - Scans all runs in `runs/` with `manifest.json`.
  - Fixes `manifest.status = "success"` for runs whose completed stages succeeded.
  - Updates `runs/mlflow/mlflow.db` records from `RUNNING` to `FINISHED`.
  - Ingests `runs/capture_limitation_backbone_results.csv` scalar metrics onto the corresponding 18 MLflow runs.
  - Uploads summary plots (`docs/figures/*.png` under 50 MB) to MLflow stage artifacts.
  - Registers the 4 physical Lego turbine runs under MLflow experiment `lego_turbine`.
  - Re-runs `amp exp export` to update `runs/experiment_results.csv` with accurate terminal statuses.
- **Unit Test Coverage**:
  - Add test in `orchestrator/tests/test_tracking.py` verifying that partial runs (`only=["train.default"]`) terminate as `FINISHED` in MLflow.
  - Add test in `orchestrator/tests/test_scheduler_hardening.py` verifying `gt_segmentation` auto-seeding.

## Out of scope

- Redesigning the full metric catalog or house-style plotting functions (owned by T29).
- Unified navigator tool (owned by T30).

## Deliverables

- Patched `orchestrator/pipeline/dag/scheduler.py` and `orchestrator/pipeline/tracking.py`.
- New `orchestrator/pipeline/reconcile.py` wired to `amp mlflow reconcile` in `pipeline/tracking_cli.py`.
- Unit tests in `orchestrator/tests/test_tracking.py` and `orchestrator/tests/test_scheduler_hardening.py`.
- Reconciled MLflow database (`runs/mlflow/mlflow.db`) and refreshed `runs/experiment_results.csv`.

## Verification & Execution Results

- `test_partial_stage_run_terminates_as_finished` passing in `test_tracking.py`.
- `test_auto_seed_gt_segmentation_from_scene` passing in `test_scheduler_hardening.py`.
- `amp mlflow reconcile` executed live against `runs/` and `runs/mlflow/mlflow.db`:
  - 121 manifests scanned.
  - 21 fixed to `"success"`, 4 to `"failed"`.
  - 18 stuck MLflow runs terminated to `FINISHED`.
  - 4 physical turbine runs registered and finished in MLflow.
  - 2,544 evaluation scalar metrics ingested into MLflow runs.
  - 70 diagnostic renders and publication figures uploaded to MLflow artifacts.
  - `runs/experiment_results.csv` cleanly rebuilt with all completed runs.
  - Direct SQLite check (`SELECT status, count(*) FROM runs GROUP BY status`) confirmed 0 stuck runs: `[('FAILED', 4), ('FINISHED', 19)]`.
