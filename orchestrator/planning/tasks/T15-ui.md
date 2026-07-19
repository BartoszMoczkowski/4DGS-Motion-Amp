# T15 — UI (Streamlit over Layer 1 API)

- Status: done
- Phase: 5
- Depends on: T09
- Environment: host

## Goal
A thin UI for Bartosz to run, tune, and review — over the *same* Layer 1 API (no logic
duplication). Deprioritized. (Milestone M5.)

## In scope
- Streamlit app (reuse `ampUI.py`'s amplification-param panel as one view).
- Views: pick/edit a preset; launch a run; watch per-stage progress + live logs + GPU meter;
  browse artifacts/previews (renders, segmentation PNGs, amp videos); compare runs.
- Talks to Layer 1 directly (or the T14 HTTP server) — pick one and document.

## Out of scope
New capabilities beyond what Layer 1 exposes.

## Deliverables
`orchestrator/ui/` Streamlit app + run instructions.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- Launch and monitor a run from the UI; see previews and compare two runs.
- No pipeline logic lives in the UI — it only calls the API.

## Relevant existing files
`ampUI.py` (existing Streamlit amp UI to fold in), Layer 1 API, manifest/artifacts.

## Notes / gotchas
Explicitly the least important layer — keep it thin. If the T14 tool surface already covers
Bartosz's day-to-day via Claude, this can stay minimal.

## Implementation notes (done, 2026-07-19)

Built `orchestrator/ui/{app.py,layer1_client.py,README.md,__init__.py}` — a single-file Streamlit
app (`app.py`) with five tabs (Presets, Launch & Monitor, Runs & Artifacts, Compare Runs, GPU/
Containers) plus one thin adapter module (`layer1_client.py`) every view calls through, so no
pipeline logic lives in `app.py` itself.

**Transport decision (the task's own "pick one and document"):** direct in-process import of
`pipeline`/`mcp_server`, not the T14 HTTP/MCP server — see `ui/README.md`'s "Talks to Layer 1
directly" section for the full rationale (this UI only ever runs on the same machine as Layer 1
itself, unlike an MCP client which might be remote). A genuinely useful discovery while wiring
this up: `mcp_server.jobs` (background-thread run wrapper) and `mcp_server.artifact_view`
(per-kind artifact summarization) neither one imports the `mcp` package at module scope — both are
pure-Python/read-only helpers Layer 2 happens to own. `layer1_client.py` imports both directly
rather than reimplementing async-run-threading or artifact-summary logic a second time, so the UI
and Claude (via MCP) see byte-identical run-status/artifact-summary shapes. This required adding a
new `ui` extra to `orchestrator/pyproject.toml` (and `orchestrator-ui` on the root one) — `streamlit`
only, no new dependency on `mcp` itself.

**Amp-parameter panel** (the task's explicit "reuse `ampUI.py`" ask): ported the per-channel
factor/freq-low/freq-high number-input layout and method dropdown verbatim in spirit (same
`AMP_CHANNELS` order, same `AMP_METHOD_ALIASES` label mapping from T02's schema) — pre-filled from
whatever preset is selected via `validate_config`, rather than `ampUI.py`'s own hardcoded
folder-scanning (`./output`/`./arguments`), which the new config-preset model (T02) already
replaces. "Save as new preset" writes a new `pipeline/config/presets/<name>.yaml`
(`extends: <selected preset>` + the edited `amp:` section) — config, not code, so it doesn't
violate "no pipeline logic in the UI." One real Streamlit gotcha found while building this: widget
keys must be namespaced by the selected preset name (`f"amp_factor_{preset}_{ch}"`, not just
`f"amp_factor_{ch}"`) — Streamlit ignores a widget's `value=` on every rerun once its key already
exists in `session_state`, so without the preset in the key, switching the Preset dropdown would
keep showing the *previous* preset's amp values instead of the newly-resolved one.

**Launching a run never blocks the UI thread:** `layer1_client.start_pipeline_run`/`start_stage_run`
delegate straight to `mcp_server.jobs`'s existing background-thread wrapper (see above) — training
alone can take hours, and Streamlit's own script-execution model has no separate request thread the
way an HTTP server does, so blocking here would freeze the whole app, not just one request.
Monitoring reads the manifest directly (`pipeline.api.get_status`, same as Claude's own polling
loop) plus `job_error` for a background-thread failure that happened before any stage record
existed — same two-tier failure-visibility story T14 already built for MCP.

**Verification:** `ast.parse` on both new files; a fresh isolated venv (`pip install -e
'.[ui,mcp]'`) exercising `layer1_client.py`'s full surface against real preset data and a
hand-seeded run (`create_run`/`record_stage_result`, mirroring `tests/test_stages_cpu.py`'s own
`_seed_run` pattern) — preset validation, save/round-trip a new preset variant, get_status/
list_runs/list_artifacts/tail_logs/read_artifact_summary/artifact_preview_info/cancel_run/
gpu_status all confirmed correct; full existing suite (`pytest -q`) still 232 passed/9 skipped, no
regression from the new `ui/` package; `streamlit run ui/app.py --server.headless true` boots
cleanly (real uvicorn, HTTP 200, no traceback in the server log). No automated test suite added
under `orchestrator/tests/` for the UI itself — Streamlit apps aren't meaningfully unit-testable
without a browser session, so `layer1_client.py`'s own functions (the only place with any real
logic) got the direct smoke-test coverage instead, same reasoning `ampUI.py` itself never had tests
either. **Real interactive verification** (clicking through tabs, launching an actual GPU run,
viewing a real preview, comparing two real runs) needs Bartosz's Windows + Docker Desktop + GPU
machine — not yet done, same honest "pending real-hardware confirmation" status every other
task in this project has carried until he's tried it himself.

Only the deferred T16 (WSL2/Linux-distro bundling, not scheduled) and T17 (MCP job/cancel
hardening, `todo`, unrelated to this task) remain outstanding in the whole orchestrator project.
