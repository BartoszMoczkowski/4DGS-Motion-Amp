# T28 — Local MLflow tracking integration

- Status: todo
- Phase: 9 (streamlining)
- Depends on: T27
- Environment: sandbox-testable with a temp MLflow file-store backend; real server + container
  env injection verified on Bartosz's machine

## Goal

Every run logs to a **local** MLflow server; dashboards replace hand-transcribed CSV tables.
The manifest store (`runs/<id>/`) remains the source of truth — MLflow is a view over it, keyed
by the manifest run_id.

## In scope

- `orchestrator/pipeline/tracking.py` — owns **all** MLflow interaction, one place only.
  Per the light-import rule, `mlflow` is imported inside functions so Layer 1 stays
  sandbox-importable. New `mlflow` dependency on the `pipeline` package.
- **Locked storage layout** (owner decision): `mlflow server --backend-store-uri
  sqlite:///runs/mlflow/mlflow.db --default-artifact-root runs/mlflow/artifacts
  --no-serve-artifacts --host 127.0.0.1 --port 5000`; UI at http://127.0.0.1:5000;
  `runs/mlflow/` gitignored.
- `amp mlflow serve` — wrapped command for the above (T26's `mlflow` group stub gets its body).
- **Run lifecycle logging** at `run_pipeline`/`create_run`: experiment named after the
  spec/preset family; tags: manifest run_id (join key), git SHA, preset, scene-cell params;
  full resolved config logged as params.
- **Stage-transition metrics** via the existing runner hooks: `wall_time_s`, `peak_vram_mb`,
  status. Declares the per-stage metrics-file ingest contract (path + schema declaration in
  the stage result) that T29 implements a registry and producers for.
- **Artifact policy:** only small outputs logged as MLflow artifacts (amp/render mp4s,
  segmentation previews, eval plots). Checkpoints/PLY stay path-referenced with a
  `manifest_path` tag — never uploaded.
- **Graceful degradation:** server down → warning logged, run proceeds, nothing crashes.
- **Env injection:** `MLFLOW_TRACKING_URI` into container stage envs
  (`host.docker.internal:5000` via the existing per-env env maps in
  `pipeline/vendored/cuda/cuda_common.py` or its current equivalent) and into the native Isaac
  subprocess env.
- `amp mlflow prune --older-than Nd` cleanup command.
- Documented backup discipline for the SQLite db (it's a single file; say where backups go).

## Out of scope

Remote/cloud tracking, model registry, W&B (all rejected). Charts and metric definitions
(T29). MCP/UI surfacing of MLflow URLs (T30).

## Deliverables

`pipeline/tracking.py`, `pipeline` pyproject dependency, `amp mlflow` group bodies,
container/subprocess env injection, gitignore entry for `runs/mlflow/`, sandbox tests (temp
file-store backend: logging, tag join-key, offline degradation), container-env injection unit
test, task-log entry on Bartosz's machine.

## Acceptance criteria

- Running the pump spec from T27 produces one MLflow run per cell with params/metrics/artifacts
  visible at http://127.0.0.1:5000 (real verification on Bartosz's machine).
- MLflow run rows link back to `runs/<id>/` via the manifest run_id tag, and back again from
  the manifest to the MLflow URL.
- Sandbox tests with a temp MLflow file-store backend cover: logging lifecycle, run_id join
  key, offline degradation (server absent → run completes with warning).
- Container-env injection unit-tested: a stage env map built for a container contains
  `MLFLOW_TRACKING_URI=http://host.docker.internal:5000`; the native Isaac subprocess env
  contains the host-side URI.
- No import of `mlflow` at module scope anywhere in `pipeline/` (sandbox import test).

## Relevant existing files

`orchestrator/pipeline/api.py` (`run_pipeline`, `create_run`, `new_run_id`),
`orchestrator/pipeline/dag/` (runner hooks where stage transitions are visible),
`orchestrator/pipeline/artifacts/` (manifest, StageRecord with `wall_time_s`/`peak_vram_mb`),
`orchestrator/pipeline/stages/isaac_common.py` (native subprocess env),
`orchestrator/pipeline/vendored/cuda/cuda_common.py` (per-env env maps), T27's spec runner
(run metadata producer), `.gitignore`.

## Notes / gotchas

Local-only by decision — do not re-introduce W&B or a self-hosted tracking server. The SQLite
backend is fine for a single-user local workflow but is one file: losing it loses the tracking
view (the manifest store underneath survives). `--no-serve-artifacts` keeps artifact serving
off the HTTP API; artifacts live on disk under `runs/mlflow/artifacts/`. `host.docker.internal`
works from Docker Desktop containers on Windows — that's why the container URI differs from
the host URI (`127.0.0.1:5000`). Degradation must be airtight: a missing server should never
fail a run, only annotate it.
