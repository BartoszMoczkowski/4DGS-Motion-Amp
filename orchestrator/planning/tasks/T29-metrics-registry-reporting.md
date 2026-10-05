# T29 — Standardized metrics registry & report layer

- Status: todo
- Phase: 9 (streamlining)
- Depends on: T28
- Environment: pure CPU, fully sandbox-testable (synthetic manifests + fabricated metrics +
  synthetic tensorboard events); real-run regeneration check on Bartosz's machine

## Goal

Metrics and charts are defined once, in one house style, and every surface (CLI, MLflow,
Streamlit, MCP, thesis figures) draws from the same source. No more per-script ad-hoc plotting
or hand-transcribed numbers.

## In scope

- `pipeline/reporting/metrics.py` — the canonical metric catalog. Each entry declares
  name / phase / computation / source / schema. Initial entries:
  - train: total loss, L1, PSNR, SSIM, LPIPS, coarse + fine — parsed from tensorboard event
    files in `train_out` (add a tensorboard-parsing dependency);
  - render: per-frame PSNR/SSIM/LPIPS;
  - segment: `ari`, `mean_iou`, `n_pred`, `n_gt`, ROI variants — from
    `seg_eval_result.json` as written by `evaluate_segmentation`;
  - bench: wall time / peak VRAM (manifest `StageRecord`);
  - amp: runtime/VRAM + video path.
  This catalog is the contract whose ingest side T28 declared.
- `pipeline/reporting/charts.py` — prebuilt figure functions behind a committed matplotlib
  style sheet with a **fixed impl→color map**. Initial catalog:
  - training curves;
  - ARI/mIoU vs `base_amp_mm`×`multiplier` scatter + heatmap, faceted by impl;
  - Gaussian-count sweep PSNR/VRAM curves;
  - benchmark runtime/VRAM bars;
  - segmentation preview / vs-GT grids in house style (folding in `evaluate_segmentation`'s
    ad-hoc plotting);
  - amp frame-strip contact sheets.
  Each function takes typed inputs and returns a `Figure`; a writer puts outputs under
  `runs/<id>/report/` under canonical filenames and auto-logs to MLflow (artifact + scalars).
- `amp report <run_id>` — builds the bundle: charts + generated `index.md` with config
  summary, git SHA, metric table, links (manifest, MLflow run).
- `amp report compare --experiment X --impls ... --metric ari` — cross-run comparison bundle.
- **Stage contract extension:** stages declare `metrics_file`/`plot` outputs in their stage
  result so the registry can ingest them generically (implementing T28's declared contract).

## Out of scope

PDF thesis assembly (optional later). Streamlit/MCP surfacing of the bundles (T30). Defining
new metrics beyond the catalog above (add entries as their owning stages/tasks need them).

## Deliverables

`pipeline/reporting/{metrics,charts}.py`, committed matplotlib style sheet + impl→color map,
`pipeline` pyproject additions (tensorboard parsing dep; matplotlib per light-import rules —
heavy imports inside functions), `amp report` command bodies, stage contract extension +
registry wiring, sandbox test suite (below).

## Acceptance criteria

- Fully sandbox-tested: synthetic manifests + fabricated metrics JSONs + synthetic tensorboard
  events run through every chart function; tests assert expected output files, schema-valid
  outputs, and numerical correctness of ingestion (parsed values match what was written into
  the fixtures).
- The pump-grid comparison chart is regenerable from store data alone (`amp report compare`),
  no live run needed (dry-run/regeneration check in sandbox; real-run confirmation on Bartosz's
  machine).
- House style enforced: every figure in the catalog renders through the committed style sheet;
  the impl→color map is the single source for facet colors.
- Stage contract: a stage declaring a `metrics_file` output is ingested into both the registry
  and its MLflow run without code specific to that stage.

## Relevant existing files

`core/train.py` (tensorboard `SummaryWriter` logging into train_out), `core/utils/image_utils.py`
(`psnr`), `motion-seg/motion_seg/evaluate_segmentation.py` (`seg_eval_result.json` +
ad-hoc PNGs), `amp-ui/amp_ui/run_renders_auto.py` (benchmark harness), `orchestrator/pipeline/
artifacts/` (StageRecord `wall_time_s`/`peak_vram_mb`), `orchestrator/pipeline/tracking.py`
(T28, ingest contract consumer), T27's unified results store (comparison data source).

## Notes / gotchas

Reporting must stay pure CPU — no torch imports anywhere in `pipeline/reporting/` (sandbox
import test, same discipline as the rest of Layer 1). Tensorboard event parsing is the
fragile ingestion point: pin the parsing approach with fixture event files written by the real
`SummaryWriter` so format drift is caught by tests. Canonical filenames under `runs/<id>/report/`
are part of the contract — T30's `get_run_report` and the Streamlit Reports tab will reference
them literally. Do not let chart functions read the manifest store directly; they take typed
inputs so they stay testable and reusable from any surface.
