# UI (T15) — a thin Streamlit panel over Layer 1

Least important layer, kept thin per `planning/tasks/T15-ui.md`'s own notes: if the T14 MCP tool
surface already covers day-to-day use via Claude, this is just a convenience panel for Bartosz —
pick/edit a preset, launch a run, watch it, browse results, compare two runs.

## Talks to Layer 1 directly, not the T14 HTTP/MCP server

T15's spec asks to "pick one and document." This UI imports `pipeline`/`mcp_server` directly, in
the same process as Streamlit — it does **not** go through the T13/T14 HTTP MCP server.

Why: the UI's whole reason to exist is as a panel for Bartosz on the *same* Windows machine that
already runs Docker Desktop and the GPU (`ARCHITECTURE.md`'s Layer 3 note). Unlike an MCP client,
which might be remote, there's no reason for this UI to hop through a network + bearer-token
boundary to reach code running on the same box. It also means the UI works with no MCP server
process running at all.

`ui/layer1_client.py` is the one adapter module every view calls through — see its own docstring
for exactly which Layer 1 (`pipeline.api`) and Layer 2 (`mcp_server.jobs`, `mcp_server.artifact_view`)
functions it reuses, and why reusing `mcp_server`'s job/artifact-summary code here isn't a layering
violation (neither of those two `mcp_server` modules imports the `mcp` package itself). No pipeline
logic lives in `app.py`/`layer1_client.py` — every button calls straight into code Layer 1 (or
Layer 2, read-only) already owns.

## Running it

Needs Layer 1's own dependencies (same environment you'd run/test the orchestrator's `pipeline`
package in) plus `streamlit`. From the repo root, using the `uv` workspace:

```
uv sync --extra orchestrator-ui
uv run streamlit run orchestrator/ui/app.py
```

Or, from an existing orchestrator dev venv (`orchestrator/`) with `pipeline` already installed
editable:

```
cd orchestrator
pip install streamlit   # or: pip install -e .[ui]
streamlit run ui/app.py
```

Streamlit opens `http://localhost:8501` by default. No auth, no bearer token — same trust model as
running any other local dev tool directly against your own machine (unlike T13's MCP server, which
explicitly needs one because it's reachable over the network).

## Views

- **Presets** — pick a preset, resolve/validate it, see the full resolved config. Folds in
  `ampUI.py`'s amplification-parameter panel (per-channel factor/freq-cutoff editor, method
  picker) pre-filled from the preset's resolved `amp:` section. "Save as new preset" writes a new
  `pipeline/config/presets/<name>.yaml` (`extends: <this preset>` + the amp overrides) — this is
  config, not pipeline logic, so it doesn't violate the "no logic in the UI" rule.
- **Launch & Monitor** — launch a preset's auto-planned DAG (`from_stage`/`to_stage`/`only`/
  `force`, plus an advanced `raw_mesh` external-artifact field for a `prep_split.default`-starting
  run); the launch itself runs on a background thread (`mcp_server.jobs`, reused as-is) so the UI
  never blocks for a run's own duration. Monitor any `run_id` — per-stage status table, log
  tailing, best-effort cancel, optional 5s auto-refresh.
- **Runs & Artifacts** — every known run; pick one to see its status, then browse its artifacts
  (inline preview for `png`/`video`, a JSON summary — same shape Claude sees over MCP — for
  everything else).
- **Compare Runs** — two runs side by side: status tables, any shared `png` previews, and a flat
  key-by-key diff of their resolved configs (only differing keys shown).
- **GPU / Containers** — current VRAM/RAM (`pipeline.api.gpu_status`), managed container list,
  start/stop controls.

## What's out of scope (per the task spec)

No capability beyond what Layer 1 exposes — this UI cannot do anything `pipeline.api` (or Claude,
via T14) can't already do. Real cancellation is still Layer 1's own open gap (T12/T17's scope, not
this task's) — the Cancel button is honest about that, same as the MCP `cancel_run` tool.

## Verification status

Import-clean and syntax-checked in the sandbox (no GPU/Docker/Streamlit runtime here). Actually
launching `streamlit run` and clicking through a real run, previews, and a two-run comparison needs
Bartosz's own Windows + Docker Desktop + GPU machine — same honest "not yet run for real" status
every other real-hardware-dependent task in this project has carried until he's tried it.
