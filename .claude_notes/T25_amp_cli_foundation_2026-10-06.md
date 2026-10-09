# T25 — `amp` CLI foundation (2026-10-06)

Task: `orchestrator/planning/tasks/T25-amp-cli-foundation.md` → status done.

## What landed

- Root package `amp` gained `cli.py` (Typer root app) and the `[project.scripts] amp = "amp.cli:app"`
  entry point in root `pyproject.toml`; `typer>=0.12` added to root dependencies (lock: typer 0.27.3).
- Nine groups mounted explicitly (no plugin discovery, per spec):

  | Group | Owner module | Notes |
  |-------|--------------|-------|
  | capture | `omniverse_pipeline/cli.py` | stub |
  | train / render | `core/core_cli.py` (two sub-apps) | stub; registered as `core_cli` py-module in `core/pyproject.toml` |
  | seg | `motion_seg/cli.py` | stub |
  | exp | `scene-gen/scene_gen_cli.py` | stub; registered as `scene_gen_cli` py-module |
  | orch | `pipeline/cli.py` | stub |
  | bench | `amp_ui/cli.py` | stub |
  | mlflow / report | root `amp/cli.py` placeholders | reserved for T28/T29 |

  core/scene-gen have flat py-module layouts, so their CLI modules use the unique top-level names
  `core_cli` / `scene_gen_cli` (a bare `cli` py-module would collide across members).

- Every stub imports **typer only** at module scope; docstrings state the lazy-import rule
  (heavy imports inside command functions, T26+).
- Sandbox test `orchestrator/tests/test_amp_cli.py`: runs a clean-subprocess probe asserting all 9
  groups appear in `amp --help`, each `amp <group> --help` exits 0, and none of
  torch/docker/mcp/streamlit/cv2/numpy land in `sys.modules`. Subprocess (not in-process) because
  the pytest process is pre-polluted by conftest and other test modules.

## Gotchas hit

- Windows cp1252 console: a `→` in the root help string crashed Rich's win32 writer with
  `UnicodeEncodeError`. Help text is now ASCII-arrow-free (em dash is fine in cp1252).

## Verification

- `uv lock --check` passes (169 packages); `uv sync` reinstalls root/core/scene-gen editables and
  puts `amp` on PATH.
- `uv run amp --help` lists all 9 groups; every `amp <group> --help` exits 0.
- New test green; full orchestrator suite run with `--group test` (plain `uv run pytest` misses the
  mcp extra needed by test_mcp_server.py / test_mcp_tools.py collection).
- Raw module invocations (`python core/train.py` etc.) untouched — CLI is purely additive.

## Follow-ups

- T26 fills command bodies (thin layers over `pipeline.api`, never subprocess wrappers).
- T28/T29 replace the mlflow/report placeholder `status` commands.
