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
- **`capture.isaac` runs natively, not in the `isaac` Docker container — locked 2026-07-16.**
  NVIDIA has confirmed Vulkan (what Isaac Sim's Hydra/RTX renderer needs) isn't supported under
  WSL2, which is what backs Docker Desktop's Linux containers on Windows; real-hardware runs
  confirmed `omni_capture.py` completes with no exception but the RTX render products never
  produce a frame. `capture.isaac` now execs `omni_capture.py` as a native Windows subprocess
  against Bartosz's own real Isaac Sim install (`PIPELINE_ISAAC_NATIVE_PYTHON`, default
  `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`, **corrected 2026-07-18** —
  the original path didn't exist on his machine) instead — see
  `pipeline.stages.isaac_common.run_native_isaac_script`'s docstring and
  `.claude_notes/NOTES_pipeline_orchestration.md`'s "adjust the project plan" entry. `prep_split`/
  `prep_motion` are unaffected (CPU-only, no rendering) and keep running inside the `isaac`
  container as before — see the revised "Environments" section below. This was a deliberate
  trade against Bartosz's original preference for full container independence/portability, made
  because the Docker/WSL2 path is a hard platform limitation, not a config gap to keep chasing.

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

## Environments (three, one now split by execution mechanism)

- `host` — native Windows Python venv (`.venv`, uv). CPU-only stages. (Revised 2026-07-14 — was a
  WSL2 Python venv; see the locked decision above.)
- `cuda` — the repo `Dockerfile` image (`nvidia/cuda:12.4.1-devel`, PyTorch cu124). train /
  render / seg_extract / amp / Option-A seg.
- `isaac` — logically still one environment (stages keep `environment = "isaac"` for
  resource-exclusivity purposes — the two GPU images never run concurrently), but **revised
  2026-07-16**, split by how each stage actually executes:
  - `prep_split.default` / `prep_motion.default` — still exec inside
    `nvcr.io/nvidia/isaac-sim:6.0.1` via `/isaac-sim/python.sh` (`ISAAC_PYTHON`), same as before.
    Only need `pxr`/USD bindings, no rendering, unaffected by the Vulkan/WSL2 limitation.
  - `capture.isaac` — now a **native Windows subprocess** against Bartosz's real Isaac Sim
    install (`PIPELINE_ISAAC_NATIVE_PYTHON`, default
    `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`), not the Docker container
    — see the locked decision above.

## Definition of "contained task"

A task is contained if it has: a single clear goal, explicit in/out scope, concrete deliverables,
acceptance criteria that can be checked, and named dependencies. If a task can't be verified on
its own, it's too big — split it.
