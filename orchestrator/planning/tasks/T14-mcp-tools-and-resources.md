# T14 — MCP tools & resources

- Status: done (2026-07-19)
- Phase: 4
- Depends on: T13, T09
- Environment: host (Windows)

## Goal
Expose the full control + read surface so Claude can run the pipeline and actually *see* results.
(Completes Milestone M4.)

## In scope
- Tools: `list_presets`, `validate_config`, `run_pipeline`, `run_stage`, `get_run_status`,
  `tail_logs`, `list_runs`, `list_artifacts`, `read_artifact` (text/JSON, or npz *summary* —
  keys/shapes/stats, not raw dumps), `get_preview` (returns PNG/video paths so Claude sees
  segmentation previews, renders, amp clips), `gpu_status`, `list_containers`, `start_container`,
  `stop_container`, `cancel_run`.
- Resources: run manifests, log files, preview images exposed as MCP resources.
- Result shaping so tool outputs stay small (summaries + paths, not big blobs).

## Out of scope
UI (T15).

## Deliverables
Full tool/resource set on the T13 server; a usage doc for Claude.

## Acceptance criteria
- Claude drives a real run: launch via `run_pipeline`, poll `get_run_status`, `tail_logs`, then
  `get_preview` a segmentation PNG and `read_artifact` a manifest — all over HTTP.
- Large artifacts never dumped wholesale; npz returns a summary.

## Relevant existing files
Layer 1 API + manifest/artifacts (T03); preview outputs (`*_preview.png`, renders, amp video).

## Notes / gotchas
`get_preview` is what lets Claude judge results visually — prioritize it. Keep every tool async or
fast; nothing that blocks for the length of a training run.

## Implementation notes (2026-07-19)

All 15 tools + 3 resource templates registered on `mcp_server/server.py`'s `build_mcp()`:
`list_presets`/`validate_config`/`list_runs`/`list_artifacts` (thin `pipeline.api` delegation),
`run_pipeline`/`run_stage`/`get_run_status`/`tail_logs`/`cancel_run` (run lifecycle),
`read_artifact`/`get_preview` (result shaping, new `mcp_server/artifact_view.py`), `gpu_status`
(T13, unchanged)/`list_containers`/`start_container`/`stop_container` (machine control). Resources:
`run://{run_id}/manifest`, `run://{run_id}/log/{stage}`, `run://{run_id}/artifact/{artifact_name}`.

**Async jobs (new `mcp_server/jobs.py`):** `pipeline.api.run_pipeline`/`run_stage` are synchronous,
blocking calls — this module is what turns them into fire-and-forget background threads so the
MCP tool call itself returns a `run_id` immediately. Needed a small, real Layer 1 change to make
this work cleanly: `pipeline.api.run_pipeline` gained an optional `run_id=` parameter (defaulting
to the same `new_run_id(preset)` scheme, now factored into its own function) so the id can be
generated *before* the background thread starts, rather than only being available from the
(blocking) call's own return value once the entire run finishes. `mcp_server.jobs` validates the
preset/run_id synchronously first (fails the MCP call immediately on a bad preset/unknown run_id)
before spawning the thread, and catches any exception the thread itself raises (e.g.
`MissingDependencyError` for a DAG's external inputs never supplied, raised by `run_dag` before any
per-stage manifest record exists) into a small `job_error` registry — otherwise a failure before
the first stage even starts would be invisible (a run stuck `"pending"` with zero explanation).
`get_run_status` surfaces this as a `job_error` key alongside the normal manifest content.

**Result shaping (new `mcp_server/artifact_view.py`):** `read_artifact_summary` — the exact
per-kind logic the acceptance criteria's "npz returns a summary" line asks for: json → parsed
content (dropped above 64KB, use the resource instead), npz → per-key shape/dtype/min/max/mean/
nan_count (never raw arrays), dataset/model → shallow one-level directory listing (capped at 200
entries), ply → vertex/face counts parsed from the header only. `get_preview`/`preview_kind` —
png → an inline `mcp.server.fastmcp.Image` (reads the file itself, base64s it into the tool
result); video → **not** inlined (no MCP video content type, clips are far too large) — path/size
+ the matching `run://.../artifact/...` resource URI instead, for a client that can fetch it
separately.

**`cancel_run`** is honestly best-effort: `pipeline.api.cancel` is still `NotImplementedError`
(T12 explicitly scoped real cancellation out of its own work, nothing since has picked it up) — the
tool confirms the run exists (a real error if not) and reports the gap in its own return value
(`{"cancelled": false, "reason": "..."}`) rather than letting a generic `NotImplementedError`
surface as an opaque tool error.

**Tests** (`tests/test_mcp_tools.py`, 20 new — 200 passed/9 skipped total, same pre-existing
unrelated `test_containers.py` sandbox-permission errors as every prior task): same "there's
nothing to fake" reasoning as T13's own suite for every read/discovery tool + both resources (real
server, real HTTP, real MCP client, seeded synthetic run data — real json/npz/png/video files on
disk plus a directory artifact, written directly via `pipeline.artifacts`, mirroring
`tests/test_stages_cpu.py`'s `_seed_run` pattern). For `run_pipeline` specifically: calling it
against the real `base` preset with no `external_artifacts` genuinely exercises the async-return-
immediately + `job_error`-capture path without touching Docker/GPU/native Isaac at all, since the
auto-planned DAG's `MissingDependencyError` (missing `raw_mesh`) raises before any stage attempts
to run. `list_containers` (no reachable Docker daemon here) is asserted to fail as a clean tool
error, not a crash/hang — real container-manager logic itself is T08's own suite's job, unchanged
here.

**Acceptance criteria status:** the literal "Claude drives a real run... over HTTP" sequence
(`run_pipeline` → `get_run_status`/`tail_logs` → `get_preview` a segmentation PNG → `read_artifact`
a manifest) is verified end-to-end against seeded data in the sandbox; running it against a real
GPU/container run still needs Bartosz's own machine, same honest status every other real-hardware-
touching task in this project has carried until someone actually ran it there. "Large artifacts
never dumped wholesale; npz returns a summary" — verified directly (`arrays` dict has no raw data,
just shape/dtype/stats).

**Milestone M4 reached:** the full tool/resource set exists and is verified against real (seeded)
data over real HTTP + auth; only real hardware/network confirmation remains, same gap T13 already
flagged for reachability.
