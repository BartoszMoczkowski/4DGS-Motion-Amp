# Working agreement & conventions

Shared reference for how Claude and Bartosz work on this subproject. Keep this short and current.

## Locked decisions (2026-07-11)

- **Orchestration foundation:** custom lightweight DAG in pure Python (no Prefect/Dagster/Snakemake).
- **Runtime host:** **revised 2026-07-14** — native Windows on Bartosz's machine, driving Docker
  Desktop directly (Docker Desktop is reachable from Windows the same way it's reachable from a
  WSL2 shell; nothing about *talking to* it required WSL2, only our own assumption about where our
  code ran did). All paths, container spin-up, and GPU access assume this. Running from inside a
  WSL2 distro (or bundling a packaged WSL2 + Docker setup) is explicitly **deferred future work**,
  not a requirement today — see `ARCHITECTURE.md`'s phasing notes and
  `.claude_notes/NOTES_pipeline_orchestration.md`'s "Runtime host moved off WSL2" entry.
- **Extensibility model:** plugin registry (swappable stage implementations behind a role
  interface) **plus** config presets (experiments declared as config, not code).
- **MCP transport:** **HTTP** (streamable HTTP / SSE), for flexibility (local or remote Claude),
  with auth. Not stdio.

## Ground rules for the work

- **Copy the logic in, don't call the original script.** (Superseded "wrap, don't rewrite" —
  2026-07-14, see `.claude_notes/NOTES_pipeline_orchestration.md`.) `omniverse_pipeline/`,
  `motion_seg/`, and the repo-root scripts (`train.py`, `render.py`, `render_amp.py`, ...) are
  throwaway/testing scripts — useful as a *reference* for already-verified logic, never as a live
  dependency. A stage must not `sys.path`-hack its way into importing them, and must not shell out
  to them as a subprocess. Instead, port the verified function(s) into the orchestrator's own tree
  (`pipeline/vendored/<env>/...`, see `ARCHITECTURE.md`) — copy, don't reimplement or redesign the
  logic while porting it. The **only** thing this project depends on externally is the *runtime
  environment* — the `isaac` and `cuda` container images (Isaac Sim, PyTorch/CUDA) — never the
  script files that happen to live outside `orchestrator/`. Refactor copied-in logic further only
  when it clearly pays off, as its own task.
- **One task at a time.** Each `tasks/Txx-*.md` is contained: don't start a task whose
  dependencies are unfinished. Update the task's `Status` header as you go
  (`todo` → `in-progress` → `done`).
- **Every task ends with verification.** No task is `done` without its acceptance criteria met
  (self-test, dry-run, or a real run where a GPU is available). CPU-only pieces must be verifiable
  in the sandbox; GPU pieces get a documented manual-run checklist for Bartosz's machine.
- **Config is the single source of truth.** Once T02 lands, no new `.sh` files and no new
  scattered YAML/`arguments/*.py` — new settings go into the config schema/presets.
- **Path translation lives in exactly one module** (T06). Never hardcode `Q:\` / `/omniverse` /
  `/workspace` anywhere else.
- **Keep the old `.sh` scripts working** until a stage reaches parity, as a fallback.
- **Notes/decisions** during implementation still go to `../.claude_notes/` per project
  convention; architectural changes are reflected back into `ARCHITECTURE.md`.

## Environments (three)

- `host` — native Windows Python venv (`.venv`, uv). CPU-only stages. (Revised 2026-07-14 — was a
  WSL2 Python venv; see the locked decision above.)
- `cuda` — the repo `Dockerfile` image (`nvidia/cuda:12.4.1-devel`, PyTorch cu124). train /
  render / seg_extract / amp / Option-A seg.
- `isaac` — `nvcr.io/nvidia/isaac-sim:6.0.1`, entry `/isaac-sim/python.sh`. prep_split /
  prep_motion / capture.

## Definition of "contained task"

A task is contained if it has: a single clear goal, explicit in/out scope, concrete deliverables,
acceptance criteria that can be checked, and named dependencies. If a task can't be verified on
its own, it's too big — split it.
