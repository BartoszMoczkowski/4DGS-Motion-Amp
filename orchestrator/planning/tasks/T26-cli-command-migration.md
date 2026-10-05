# T26 — CLI command migration: kill raw-python invocations

- Status: todo
- Phase: 8 (streamlining)
- Depends on: T25
- Environment: sandbox-testable for arg parsing + dry-run/validation paths; GPU/Isaac commands
  verified on Bartosz's machine

## Goal

Every recurring operation becomes an `amp` subcommand; raw `python path/to/script.py` stops
being the documented path for routine work. This task migrates the command catalog; the
experiment spec runner and MLflow/report groups land in T27–T29.

## In scope

Commands to add, with the script each replaces:

- `amp capture split|motion|gen-grid|run|convert|preview` — replaces
  `omniverse-pipeline/omniverse_pipeline/split_mesh.py` / `add_motion.py`,
  `scene-gen/gen_scenes.py`, native-Isaac `omni_capture.py` (via
  `PIPELINE_ISAAC_NATIVE_PYTHON`), `omni_to_4dgs.py`, `scene-gen/frames_to_mp4.py`. Plus
  `amp capture verify <dir>` — the post-capture frame-count QA that today is ad-hoc
  `python -c` one-liners (documented pain in `docs/post-mortem`).
- `amp train <scene> [--preset ...]` — replaces `uv run --package 4dgs-core python
  core/train.py -s ... -m ... --configs ...`.
- `amp render <run> [--amp-factors ...] [--method ...] [--video ...]` — replaces
  `core/render.py` / `core/render_amp.py` invocations.
- `amp seg extract|rigid|mbs|kabsch|multicut|eval` plus `amp seg run <scene> [--skip-extract]` —
  replaces `motion-seg/motion_seg/run.sh` (3 chained steps + `SKIP_EXTRACT` env var) and the
  per-impl raw invocations.
- `amp bench renders [--models ...] [--methods ...] [--repeats N]` — replaces
  `amp-ui/amp_ui/run_renders_auto.py`'s hardcoded source lists at the top of the file; writes
  to the run registry instead of cwd-relative `results.csv`.
- `amp orch mcp|ui` — starts the MCP server and the orchestrator/standalone UIs with the right
  env (e.g. `PIPELINE_MCP_TOKEN` handling documented, not silently defaulted).

Also in scope: delete `motion-seg/motion_seg/run.sh` once `amp seg run` covers it (per
INSTRUCTIONS.md, keep the old path working until parity, then remove); move scene-gen
utilities into the `scene-gen` package (T24) as command bodies; anything left loose under
`scene-gen/` either becomes a command or moves to a clearly-marked `scratch/` with a README
stating it is not supported.

## Out of scope

Experiment grid/sweep specs (`amp exp ...` — T27). `amp mlflow`/`amp report` (T28/T29).
MCP-side changes (T30). The vendored stage logic itself — commands call Layer-1 API /
registered stages, they don't re-port logic.

## Deliverables

Per-member `cli.py` command bodies (T25 stubs filled), deleted `run.sh`, relocated/absorbed
scene-gen scripts, sandbox tests for arg parsing + dry-run/validation paths of every command,
updated docs pointing at `amp` (full AGENTS.md §5 rewrite may land here or in T31 — decide
in-task and record it).

## Acceptance criteria

- Every operation in the §5 command catalog of `AGENTS.md` has an `amp` equivalent (map each
  old invocation to its new command in the task log).
- No `sys.path` manipulation remains in any migrated script path.
- `amp capture verify` reports frame counts per camera against the capture config — replaces
  the post-mortem's ad-hoc one-liners.
- Sandbox tests cover arg parsing and dry-run/validation for every command; GPU/Isaac commands
  additionally verified on Bartosz's machine.
- `amp capture run` still executes as a native Windows Isaac subprocess only (Vulkan is
  unsupported under Docker — platform gap, not a regression).

## Relevant existing files

`motion-seg/motion_seg/run.sh`, `amp-ui/amp_ui/run_renders_auto.py` (hardcoded lists,
cwd-relative `results.csv`), `scene-gen/*.py` (25+ loose scripts; docstrings currently instruct
`.venv/Scripts/python.exe` raw invocation — those instructions must go),
`omniverse-pipeline/omniverse_pipeline/{split_mesh,add_motion,omni_capture,omni_to_4dgs}.py`,
`core/{train,render,render_amp}.py`, `orchestrator/pipeline/stages/isaac_common.py`
(`run_native_isaac_script`, `PIPELINE_ISAAC_NATIVE_PYTHON`), `AGENTS.md` §5.

## Notes / gotchas

`capture.isaac` remains native-Windows-subprocess-only (NVIDIA-confirmed Vulkan/WSL2 gap) —
the CLI wraps `run_native_isaac_script`, it does not change the execution mechanism. Keep raw
`python core/train.py` working throughout: Docker stage execs invoke the scripts directly
inside the container. Watch for module-scope heavy imports leaking into `amp --help` via new
command bodies — T25's lazy-import rule applies to every command added here. `run.sh`'s
`SKIP_EXTRACT` env var becomes the `--skip-extract` flag.
