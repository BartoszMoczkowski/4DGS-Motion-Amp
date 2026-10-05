# T30 — MCP consolidation & report surfaces

- Status: todo
- Phase: 10 (streamlining)
- Depends on: T28, T29, T17
- Environment: sandbox-testable over real HTTP + real MCP client (existing
  `tests/test_mcp_tools.py` pattern); Streamlit UI verification on Bartosz's machine

## Goal

Intent-level MCP tools — about 10, down from today's 15 thin 1:1 wrappers over Layer 1 —
following the locked decision that APIs must not map 1:1 to tools. UI and MCP stop drifting by
sharing the same Layer-1 shapes and report bundles.

## In scope

- **Consolidate to domain-grouped tools** (target ≤12 total):
  - `run_experiment(spec_path | preset, stages?)` — handles external-artifact seeding
    internally (the #1 documented `run_pipeline` failure mode: missing `raw_mesh` etc.
    surfacing as opaque DAG errors);
  - `run_stage`, `get_run_status`, `list_runs`, `get_run_artifact`, `get_run_preview`;
  - `list_presets`, `validate_preset`, `save_preset_variant` (promoted from UI-only,
    `orchestrator/ui/` — make it available to MCP clients too);
  - `gpu_status`, `container_start`, `container_stop`;
  - `get_run_report` — returns the T29 bundle path + MLflow URL.
- **Rich server `instructions`** encoding the conventions: check status before asking for
  artifacts, run-ID conventions (T27's `<experiment>-<cell>-<8hex>`), where experiment specs
  live.
- **Structured errors** without stack traces — clients get actionable, typed failure reasons.
- **Streamlit "Reports" tab** rendering T29 bundles (thin viewer; embedding the MLflow UI at
  :5000 via iframe is acceptable).
- `get_run_status` / `get_run_artifact` gain MLflow URLs alongside manifest paths.
- Keep the 3 `run://` resource templates unchanged.

Depends on T17 for real cancellation: this task **assumes T17 is complete** (reference its
spec; do not re-implement cancel). If T17 is still open when Phase 10 planning starts,
schedule T17 first.

## Out of scope

New stage functionality. Distributed/multi-process job coordination (the single-process
`_jobs` registry model from T14/T17 stays). Changes to the Layer-1 API itself.

## Deliverables

Rewritten `orchestrator/mcp_server/server.py` tool surface, updated `jobs.py` /
`artifact_view.py` as needed, rewritten `mcp_server/TOOLS.md`, Streamlit Reports tab in
`orchestrator/ui/app.py`, sandbox tests (real HTTP + real MCP client, seeded synthetic data,
the `tests/test_mcp_tools.py` pattern), task-log entry for UI verification on Bartosz's machine.

## Acceptance criteria

- ≤12 tools total; every tool has a rich description + use-case examples (per-tool descriptions
  are "prompt engineering for tools" — invest there).
- The consolidated `run_experiment` succeeds against a spec whose external artifacts exist and
  fails with a structured, actionable error (not a stack trace) when they don't — sandbox
  test, both directions.
- Sandbox suite runs against the real server over real HTTP with the real MCP client, seeded
  synthetic run/report data (no GPU).
- `get_run_report` returns the T29 bundle path and the MLflow URL for the same run_id.
- `TOOLS.md` rewritten to describe the new tool surface; the 3 resource templates still work
  (existing tests green or updated).
- Streamlit Reports tab verified on Bartosz's machine.

## Relevant existing files

`orchestrator/mcp_server/{server.py, jobs.py, artifact_view.py, TOOLS.md}`,
`orchestrator/ui/app.py` (UI-only `save_preset_variant` to promote), `orchestrator/ui/
layer1_client.py` (shared shapes — reuse, don't duplicate), `orchestrator/tests/
test_mcp_tools.py` (test pattern), `pipeline/api.py` (Layer 1), T27 run-ID conventions, T29
report bundles, `planning/tasks/T17-mcp-job-and-cancel-hardening.md` (cancel assumed done).

## Notes / gotchas

Do not let the tool count creep back up — if a new need appears, extend an existing domain
tool's parameters before adding a tool. Keep the UI and MCP on identical Layer-1/jobs/
artifact_view shapes (the T15 pattern that already prevents drift). The `run_experiment`
seeding logic must reuse `run_pipeline`'s `external_artifacts` mechanism (T11) rather than
reinventing artifact staging. Cancel behavior surfaced through `get_run_status` comes from
T17 — reference, don't re-implement.
