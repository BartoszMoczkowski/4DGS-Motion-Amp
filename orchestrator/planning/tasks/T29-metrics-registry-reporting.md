# T29 — Standardized metrics registry & report layer (MLflow-native)

- Status: done
- Phase: 9 (streamlining)
- Depends on: T28, T28b
- Environment: pure CPU, fully sandbox-testable (synthetic manifests + fabricated metrics JSONs +
  MLflow test database); real-run regeneration check on Bartosz's machine

## Goal

Metrics and charts are defined once, in one house style, and every surface (CLI, MLflow,
Streamlit, MCP, thesis figures) draws from the same source. **Retire TensorBoard entirely** in
favor of MLflow's native step-metric tracking and multi-run comparison capabilities: no binary
`.tfevents` parsing, no fragile protobuf dependencies, and no clunky multi-directory TensorBoard
sessions.

## Why MLflow Over TensorBoard for This PyTorch Project

1. **Native Multi-Run Comparisons**: TensorBoard is notoriously awkward for comparing multiple
   runs across a grid or sweep — directory paths blend together without metadata joins, facet
   filtering by preset/experiment is impossible, and parameter diffs do not exist. MLflow UI provides
   built-in multi-run comparisons (`#/compare-runs`): select runs across an experiment to view
   overlaid step curves (loss, PSNR, iteration speed), parallel coordinate plots, scatter plots
   (e.g. ARI vs. displacement amplitude), and side-by-side config parameter diffs.
2. **Programmatic Time-Series Querying**: With `MlflowClient.get_metric_history(run_id, key)`,
   the full time-series curve `[Metric(key, value, timestamp, step), ...]` is retrieved directly as
   clean Python objects or DataFrames in milliseconds. `pipeline/reporting/charts.py` can directly
   query the local SQLite database (`runs/mlflow/mlflow.db`) to generate comparison charts without
   touching raw log files.
3. **Zero Binary Event Dependencies**: Replaces opaque binary `.tfevents.*` files with a clean
   contract: `core/train.py` emits structured step metrics (`train_out/training_curves.json`), which
   `pipeline.tracking` batches into MLflow via `client.log_batch(..., metrics=[Metric(..., step=...)])`.
   Layer 1 stays 100% pure-CPU, fast, and free of tensorboard dependencies.
4. **Single Source of Truth**: All metrics (training loss over iterations, validation PSNR/SSIM,
   motion segmentation ARI/IoU, interface split rates, and VRAM/wall-time benchmarks) live in
   **one** unified MLflow store (`runs/mlflow/mlflow.db`).

## In scope

- `pipeline/reporting/metrics.py` — the canonical metric catalog. Each entry declares
  name / phase / computation / source / schema. Initial entries:
  - train: step metrics (total loss, L1 loss, iteration time, coarse + fine validation PSNR/SSIM/LPIPS
    over global iterations) ingested from `training_curves.json` into MLflow step metrics;
  - render: per-frame PSNR/SSIM/LPIPS;
  - segment: `ari`, `mean_iou`, `n_pred`, `n_gt`, ROI variants — from
    `seg_eval_result.json` as written by `evaluate_segmentation`;
  - research / domain evaluation (Capture Limitation & Turbine suites):
    - boundary & interface: `cat_a_split_rate`, `cat_b_split_rate`, `pairs_split_rate`, `interface_auroc`;
    - deviant target part: `target_iou`, `target_f1`, `target_prec`, `target_rec`;
    - deformation & motion: `mean_displacement`, `p95_displacement`, `max_displacement`, `fft_dominant_freq_hz`, `estimated_rpm`;
    - seed survival: `n_seeds`, `n_labeled`, `n_discarded`, `k_pred`;
  - bench: wall time / peak VRAM (manifest `StageRecord`);
  - amp: runtime/VRAM + video path.
  This catalog is the contract whose ingest side T28/T28b declared.
- **Structured Step-Metric Contract for Training**:
  - `core/train.py` (and the `train.default` container stage) writes step-wise progress to
    `train_out/training_curves.json` (iteration, loss, L1, elapsed, test/train PSNR).
  - `pipeline.tracking.log_stage_metrics` ingests `training_curves.json` into MLflow via `client.log_batch`
    using native `step=iteration`.
- `pipeline/reporting/charts.py` — prebuilt figure functions behind a committed matplotlib
  style sheet with a **fixed impl→color map**. Initial catalog:
  - training curves (loss / PSNR over global iterations) fetched directly via MLflow metric history;
  - multi-run training curve overlays comparing different cells/presets on one plot;
  - ARI/mIoU vs `base_amp_mm`×`multiplier` scatter + heatmap, faceted by impl;
  - deformation time-series & FFT displacement spectral density (folding in `Conv1` diagnostics);
  - Gaussian-count sweep PSNR/VRAM curves;
  - benchmark runtime/VRAM bars;
  - segmentation preview / vs-GT grids in house style (folding in `evaluate_segmentation`'s
    ad-hoc plotting);
  - amp frame-strip contact sheets.
  Each function takes typed inputs and returns a `Figure`; a writer puts outputs under
  `runs/<id>/report/` under canonical filenames and auto-logs to MLflow (artifacts + scalars).
- `amp report <run_id>` — builds the single-run bundle: charts + generated `index.md` with config
  summary, git SHA, metric table, links (manifest, MLflow run URL).
- `amp report compare --experiment X --impls ... --metric ari` — cross-run comparison bundle:
  generates comparative summary markdown, overlaid step curves, pivot tables, and deep-links
  to MLflow compare UI (`http://127.0.0.1:5000/#/compare-runs?...`).
- **Stage contract extension:** stages declare `metrics_file`/`training_curves_file`/`plot` outputs
  in their stage result so the registry can ingest them generically (implementing T28's declared contract).

## Out of scope

PDF thesis assembly (optional later). Streamlit/MCP surfacing of the bundles (already scaffolded in T30).
Defining new metrics beyond the catalog above (add entries as owning stages/tasks need them).

## Deliverables

- `pipeline/reporting/{metrics,charts}.py`.
- Step metrics emitter in `core/train.py` (`training_curves.json`) and ingestion in `pipeline/tracking.py`.
- Committed matplotlib style sheet (`pipeline/reporting/house_style.mplstyle`) + fixed `impl→color` map.
- CLI subcommands under `amp report`: `amp report <run_id>` and `amp report compare`.
- Stage contract extension + registry wiring.
- Sandbox test suite verifying step-metric ingestion, MLflow history queries, and chart generation.

## Acceptance criteria

- Fully sandbox-tested: synthetic manifests + fabricated metrics JSONs + synthetic MLflow step-metric
  history run through every chart function; tests assert expected output files, schema-valid outputs,
  and numerical correctness of ingestion without importing torch or tensorboard.
- Multi-run comparison chart is regenerable from MLflow / store data alone (`amp report compare`),
  producing overlaid curves and deep-links to MLflow compare UI.
- House style enforced: every figure in the catalog renders through the committed style sheet;
  the `impl→color` map is the single source for facet colors across all surfaces.
- Stage contract: a stage declaring `metrics_file` or `training_curves.json` is ingested into both
  the registry and its MLflow run with step history without stage-specific hacks.
- Zero tensorboard dependencies added to `pipeline` (no protobuf/event-file parsing).

## Relevant existing files

`core/train.py` (add structured `training_curves.json` writer), `orchestrator/pipeline/tracking.py`
(`client.log_batch` with `step`), `motion-seg/motion_seg/evaluate_segmentation.py` (`seg_eval_result.json` +
ad-hoc PNGs), `orchestrator/pipeline/artifacts/` (`StageRecord` `wall_time_s`/`peak_vram_mb`),
`runs/experiment_results.csv` (unified store for comparison data).

## Notes / gotchas

- Reporting must stay pure CPU — no `torch` or `tensorboard` imports anywhere in `pipeline/reporting/`
  (enforced by AST scan in tests, matching Layer 1 discipline).
- MLflow client queries must be degradation-safe: if the MLflow store is unreachable, fall back
  to the on-disk `training_curves.json` and manifest files.
- Canonical filenames under `runs/<id>/report/` (`summary.png`, `training_curves.png`, `index.md`)
  are part of the contract — T30's `get_run_report` and the Streamlit Reports tab reference
  them literally.
- Chart functions take typed dataclasses/DataFrames, never raw manifest dictionaries directly, ensuring
  they remain reusable across CLI, UI, and test fixtures.

## Implementation notes (2026-10-24)

- **`pipeline/reporting/metrics.py`**: Canonical metric catalog (`METRIC_CATALOG`, `MetricDef`) declaring name, phase, source, schema, units, and formatting across train, render, segment, research, bench, and amp. Added `extract_manifest_metrics` to categorize scalar metrics from manifests, stage records, and disk files.
- **`pipeline/reporting/house_style.mplstyle`**: Committed publication-quality matplotlib style sheet ensuring uniform typography, crisp lines, subtle borders, and color palettes across all surfaces.
- **`pipeline/reporting/charts.py`**: Committed house-style figure plotting functions using locked `IMPL_COLORS` (`rigid`, `rigid2`, `kabsch`, `multicut`, `mbs`, `rigid2_roi`, `mask_lift`, `mask_lift_oracle`). Implemented `plot_training_curves`, `plot_multi_run_training_curves`, `plot_segmentation_grid_scatter_heatmap`, `plot_deformation_spectral`, `plot_gaussian_sweep`, `plot_benchmark_bars`, `plot_segmentation_preview`, `plot_amp_frame_strip`, and `save_report_figure`. Adheres to strict Layer 1 light-import discipline (no module-scope matplotlib/numpy).
- **`pipeline/reporting/report.py`**: Standardized single-run report bundle generator (`build_run_report`) creating `runs/<id>/report/` with canonical `summary.png`, `training_curves.png`, and `index.md` (and auto-logging to MLflow). Added cross-run comparison report generator (`build_comparison_report`) with overlaid curves, pivot tables, and deep-links to MLflow compare UI (`#/compare-runs?...`).
- **`pipeline/reporting/cli.py` & `src/amp/cli.py`**: Built `amp report <run_id>` and `amp report compare` commands using `ReportDefaultGroup`. Passes `test_amp_cli.py` lazy-import probe.
- **Structured step metrics in training**: `core/train.py` and `pipeline/vendored/cuda/train.py` updated in exact sync to write `train_out/training_curves.json` during training and evaluation. `TrainStage` registers `training_curves` artifact. `pipeline/tracking.py` ingests step metrics into MLflow via `client.log_batch` with native `step=iteration`, and provides `get_metric_history`.
- **Test verification**: Added 21 new unit tests across 4 test modules (`test_reporting_metrics.py`, `test_reporting_charts.py`, `test_reporting_report.py`, `test_tracking_step_metrics.py`). Full suite passes cleanly: 459 passed, 9 skipped. Docs consistency check passes (24/24 files).
