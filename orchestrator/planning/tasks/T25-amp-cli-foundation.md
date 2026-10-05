# T25 — `amp` CLI foundation: root package & Typer skeleton

- Status: todo
- Phase: 8 (streamlining)
- Depends on: T24
- Environment: sandbox-testable (pure Python; import-time assertions need no GPU)

## Goal

One `amp` command on PATH after a single root `uv sync`. Each workspace member contributes a
Typer sub-app mounted explicitly by the root app. This task lands the skeleton and the group
structure only — command bodies come in T26.

## In scope

- Root package (created in T24) gains the CLI entry point: `[project.scripts] amp = "...cli:app"`.
- **Typer** as the CLI framework (type-hint native; composes with the existing Pydantic config
  stack — CLI flags parse into a validated Pydantic model, then into a `PipelineConfig`).
- Pattern: root Typer app in the root package, each member's `cli.py` exposes a Typer sub-app,
  root mounts them via `app.add_typer(member_app, name="...")` — **explicit, no pluggy or
  entry-point plugin discovery** (rejected as over-engineering for a fixed, known member set).
- Command groups reserved now, filled in T26/T27: `capture`, `train`, `render`, `seg`, `exp`,
  `orch`, `bench`, `mlflow` (T28), `report` (T29).
- **Lazy imports everywhere:** `amp --help` must never import torch, docker, mcp, streamlit, or
  any member module that pulls them. Heavy imports happen inside command functions only.

## Out of scope

Migrating any real script (T26). Any MCP change (T30). Experiment specs (T27). MLflow/report
groups get only their name reserved here.

## Deliverables

Root package `cli.py` (root app + group mounts), per-member `cli.py` stubs with group help text
and no-op or `--help`-only bodies where the command isn't due until a later task, root
`pyproject.toml` `[project.scripts]` entry, sandbox test asserting heavy modules are absent
from `sys.modules` after `--help`.

## Acceptance criteria

- `uv sync && amp --help` lists all command groups without importing torch/docker/mcp —
  proven by a test asserting none of the heavy modules appear in `sys.modules` (or an import-time
  budget assertion), not by eyeballing.
- `amp <group> --help` works for every registered group.
- `python core/train.py` and the other raw module invocations still work unchanged (Docker
  stage execs run them; the CLI is additive at this point).
- Fully sandbox-testable.

## Relevant existing files

Root `pyproject.toml` (T24 makes it a real package), `orchestrator/pipeline/api.py` (Layer-1
API the commands will wrap), `orchestrator/pipeline/config/` (Pydantic config stack the CLI
feeds), `INSTRUCTIONS.md` ground rule "copy the logic in, don't call the original script"
(commands are thin layers over Layer 1, not subprocess wrappers over scripts).

## Notes / gotchas

Do NOT shell out to `python core/train.py` from commands — per the locked rule, commands call
the Layer-1 API in-process (with the same in-container exec mechanism stages already use).
Lazy import discipline is the whole point of the framework choice; a single module-scope
`import torch` anywhere in the CLI import graph makes `--help` pay seconds of startup and
breaks the acceptance test. Typer groups map 1:1 to members, not to pipeline stages — one
member may contribute commands that touch several stages.
