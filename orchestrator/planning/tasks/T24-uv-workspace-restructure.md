# T24 — uv workspace restructure & environment cleanup

- Status: todo
- Phase: 8 (streamlining)
- Depends on: —
- Environment: sandbox-testable (fresh venv, `uv lock`, import smoke tests); the Dockerfile change is
  verified by `uv sync --frozen --package 4dgs-core` succeeding, which any machine with uv can check
  (the actual image build needs Bartosz's machine but adds nothing conceptually new vs T08/T09)

## Goal

One `uv sync` at the repo root installs every command and console script in the project.
Per-package sync (`uv sync --package X`, which mutates the single shared `.venv`) becomes the
exception for special environments (Docker builds), not the daily habit. Removes the stale
pre-uv-era files and the orphaned config keys that currently mislead a fresh checkout.

## In scope

- **Root becomes a real thin package** (`4dgs-motion-amp`): root `pyproject.toml` gains
  `[project]` metadata and `dependencies` listing every workspace member via
  `{ workspace = true }` sources. Root sync then provides all member packages and (after T25)
  every console script. Add a minimal root package dir for the package body (needed anyway by
  T25's CLI entry point).
- **Canonical default environment = GPU.** Define the GPU/CPU install sets with
  `[tool.uv] conflicts` so one `uv.lock` expresses both without a second lockfile.
- **`scene-gen/` becomes a workspace member** (package `scene-gen`) with a pyproject.toml
  depending on `omniverse-pipeline` and `pipeline` as workspace editables — this kills the
  `sys.path.insert` hack at `scene-gen/run_grid_4dgs.py:38-45`. Its 25+ loose scripts become the
  package body (full command migration is T26; this task only makes the package importable).
- **`submodules/multibody-sync-4dgs` under dependency management** so
  `motion-seg/motion_seg/mbs_infer.py:88-105` stops self-bootstrapping `sys.path` twice. Decide
  in-task: workspace member (if its packaging permits) or a properly declared git/editable
  source with pinned deps reconciled against the shared lockfile.
- **Delete stale files:** root `requirements.txt` (torch 1.13.1 era),
  `submodules/multibody-sync-4dgs/requirements.txt` (`open3d==0.11.2`), and pre-uv
  `__pycache__` trees containing cpython-37 `.pyc` files.
- **Remove the orphaned `depth-diff-gaussian-rasterization` key** in root
  `[tool.uv.sources]` — the real member is `diff-gaussian-rasterization`; the stale key is
  dead config that confuses lockfile readers.
- **Unify `requires-python` to `==3.12.12`** across members (orchestrator/`pipeline` and the
  `camera_sync` packages currently say `>=3.10`).
- **Reconcile opencv floors:** `core`/`amp-ui` pin `opencv-python>=5.0.0.93` while
  `camera_sync` pins `>=4.9.0` — resolve to one coherent constraint set in the lockfile.
- **Note the mmcv mismatch** (`mmcv` URL pin is a cu121/torch2.4 wheel vs `torch>=2.6` in core)
  in the task log; fix only if a clean wheel exists for the current torch/cu126 index —
  otherwise document as a known risk with a follow-up decision for Bartosz.
- **Move dev/test tooling** (pytest etc.) into PEP 735 `[dependency-groups]` (`dev`, `test`)
  so runtime members stay lean and `uv sync --group test` is the test-setup path.
- **Dockerfile:** copy ALL 11 member pyproject.toml manifests into the build context (today
  only 5 are copied, which is a footgun under `uv sync --frozen` lockfile validation when the
  lock references members whose manifests are missing from the image). Keep the `/opt/build`
  venv layout (do NOT move the build into `/workspace` — the bind mount shadows it) and keep the
  `TORCH_CUDA_ARCH_LIST="8.6+PTX"` note for the RTX 3090.

## Out of scope

Changing `core/`'s flat py-modules layout (upstream 4DGS heritage; Docker stage execs depend on
`train.py`/`render.py` being top-level modules on `sys.path` — this stays, documented by a
comment in `core/pyproject.toml`). Touching `camera_sync` packages beyond requires-python
alignment. WSL2 bundling (T16, deferred). CLI entry points themselves (T25/T26).

## Deliverables

New/changed `pyproject.toml` files (root, `scene-gen/`, `submodules/multibody-sync-4dgs` if it
becomes a member), updated root `[tool.uv.sources]`/`[tool.uv] conflicts`/`[dependency-groups]`,
regenerated `uv.lock`, updated `Dockerfile` manifest-copy step, minimal root package dir,
deleted stale requirements.txt/pycache files, task-log entry in `.claude_notes/` covering the
mmcv and mbs-packaging decisions. Sandbox test: import smoke test of every member package in a
clean venv created from the root sync.

## Acceptance criteria

- `uv sync` at the root once → all member packages installed and importable; after T25 the same
  sync puts every `amp` console script on PATH.
- `uv sync --frozen --package 4dgs-core` (the Docker path) still works with the new lockfile.
- `uv lock --check` passes (no lockfile drift).
- `scene-gen` scripts import `pipeline`/`omniverse_pipeline` with no `sys.path` manipulation;
  `mbs_infer.py` no longer self-bootstraps `sys.path`.
- Sandbox-verifiable: fresh root `uv sync` + import smoke test of all members in a clean venv,
  run as a script/test, not just manually.
- Docs note: the one remaining multi-env case — native Isaac Sim's own `python.bat`, which can
  never share the project venv — is documented as accepted, with `PIPELINE_ISAAC_NATIVE_PYTHON`
  as the override.

## Relevant existing files

Root `pyproject.toml` (aggregator, 11 members, `[tool.uv.sources]`), `uv.lock`,
`Dockerfile` (`/opt/build` venv, 5-of-11 manifests copied, `TORCH_CUDA_ARCH_LIST`),
`scene-gen/run_grid_4dgs.py:38-45` (sys.path hack), `motion-seg/motion_seg/mbs_infer.py:88-105`
(double sys.path bootstrap), root `requirements.txt` (stale),
`submodules/multibody-sync-4dgs/requirements.txt` (stale), `core/pyproject.toml` (flat
py-modules comment), `orchestrator/pyproject.toml` + `camera_sync/*` (requires-python `>=3.10`).

## Notes / gotchas

The shared mutable `.venv` problem disappears by making the root env the default: daily work
never needs `--package`, so nothing else mutates `.venv` mid-session. The lockfile is shared and
must stay coherent — after any member pyproject change, `uv lock` must be re-run and checked.
`uv tool` is for external utilities, NOT repo commands — do not use it for `amp` (that's T25's
`[project.scripts]`). Keep `requirements.txt` deletion honest: grep the repo for references
(Dockerfile, docs) before removing.
