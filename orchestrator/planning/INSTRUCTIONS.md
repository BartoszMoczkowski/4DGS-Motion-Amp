# Working agreement & conventions

Shared reference for how Claude and Bartosz work on this subproject. Keep this short and current.

## Locked decisions (2026-07-11)

- **Orchestration foundation:** custom lightweight DAG in pure Python (no Prefect/Dagster/Snakemake).
- **Runtime host:** WSL2 (Linux) on Bartosz's machine, driving Docker Desktop + NVIDIA Container
  Toolkit. All paths, container spin-up, and GPU access assume this.
- **Extensibility model:** plugin registry (swappable stage implementations behind a role
  interface) **plus** config presets (experiments declared as config, not code).
- **MCP transport:** **HTTP** (streamable HTTP / SSE), for flexibility (local or remote Claude),
  with auth. Not stdio.

## Ground rules for the work

- **Wrap, don't rewrite.** Stages initially invoke the existing, already-verified scripts
  (`train.py`, `render.py`, `omni_capture.py`, `motion_seg/*`, `render_amp.py`, ...) in the right
  environment. Refactor a script into importable functions only when it clearly pays off, as its
  own task.
- **One task at a time.** Each `tasks/Txx-*.md` is contained: don't start a task whose
  dependencies are unfinished. Update the task's `Status` header as you go
  (`todo` → `in-progress` → `done`).
- **Every task ends with verification.** No task is `done` without its acceptance criteria met
  (self-test, dry-run, or a real run where a GPU is available). CPU-only pieces must be verifiable
  in the sandbox; GPU pieces get a documented manual-run checklist for Bartosz's machine.
- **Config is the single source of truth.** Once T02 lands, no new `.sh` files and no new
  scattered YAML/`arguments/*.py` — new settings go into the config schema/presets.
- **Path translation lives in exactly one module** (T06). Never hardcode `Q:\` / `/mnt/q` /
  `/omniverse` / `/workspace` anywhere else.
- **Keep the old `.sh` scripts working** until a stage reaches parity, as a fallback.
- **Notes/decisions** during implementation still go to `../.claude_notes/` per project
  convention; architectural changes are reflected back into `ARCHITECTURE.md`.

## Environments (three)

- `host` — WSL2 Python venv (`.venv`, uv). CPU-only stages.
- `cuda` — the repo `Dockerfile` image (`nvidia/cuda:12.4.1-devel`, PyTorch cu124). train /
  render / seg_extract / amp / Option-A seg.
- `isaac` — `nvcr.io/nvidia/isaac-sim:6.0.1`, entry `/isaac-sim/python.sh`. prep_split /
  prep_motion / capture.

## Definition of "contained task"

A task is contained if it has: a single clear goal, explicit in/out scope, concrete deliverables,
acceptance criteria that can be checked, and named dependencies. If a task can't be verified on
its own, it's too big — split it.
