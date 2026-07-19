# T17 — MCP job/cancel hardening

- Status: todo
- Phase: 4
- Depends on: T14
- Environment: host (Windows) for real container-cancellation verification; everything else is
  sandbox-testable

## Goal
Close three known, deliberately-left-open gaps T14's own implementation notes flagged rather than
silently shipped: `cancel_run` can't actually stop anything yet, two jobs can race against the same
`run_id`, and `get_preview`'s video branch has no structured output schema. None of these blocked
T14's own acceptance criteria (a real run/status/preview/read flow works end to end), but all three
are real, named limitations worth resolving before this becomes the everyday way Bartosz drives
runs.

## In scope

- **Real cancellation.** `pipeline.api.cancel(run_id)` is still `NotImplementedError` — T12
  explicitly scoped it out, T14's `cancel_run` MCP tool just reports that honestly. Implement a
  best-effort version: mark the run's manifest `status="cancelled"` and stop scheduling further
  stages; if the currently-running stage is a `cuda`/`isaac` **container** exec, **stop the whole
  container** (`ContainerManager.stop`) — **locked decision (2026-07-19): Bartosz has tested this
  directly and confirmed a container stop is fast enough to use as the cancellation mechanism**,
  so this is not an open question anymore, just the implementation. Dropping a warm container's
  in-memory state (not its persisted volumes) on cancel is an accepted trade-off, not a blocker.
  For a native Isaac subprocess (`pipeline.stages.isaac_common.run_native_isaac_script`), terminate
  the subprocess handle if one is reachable from the calling thread. A stage with genuinely no safe
  way to interrupt (none identified yet, but don't assume) should fail cleanly with a clear "can't
  cancel this one" error, not a silent no-op.
  **Later improvement, not in this task's scope:** a finer-grained, per-`exec`-level cancellation
  (leaving the container itself warm) would be nicer than stopping the whole container, but Docker
  has no clean primitive for it — revisit only if the container-stop approach turns out too coarse
  in practice (e.g. it's disruptive to a container mid-warm-reuse in some future multi-stage-in-
  flight scenario). Not something to build speculatively now.
- **Concurrent-job guard.** `mcp_server/jobs.py`'s own docstring currently says starting two
  jobs (`run_pipeline`/`run_stage`) against the same `run_id` at once is "the caller's own
  responsibility to avoid" — undefined behavior if Claude (or a UI, later) ever does it by
  accident. Add an explicit guard: reject a second `start_pipeline_run`/`start_stage_run` call for
  a `run_id` that already has a live (non-terminated) background thread, with a clear MCP tool
  error rather than letting two threads race `pipeline.dag.scheduler.run_dag` against the same
  manifest.
- **Typed `get_preview` output.** Its video branch returns a plain, unannotated dict — FastMCP
  treats this as unstructured output (serialized `TextContent`, no `structuredContent`), verified
  directly in `tests/test_mcp_tools.py::test_get_preview_video_returns_pointer_not_inline_bytes`'s
  own docstring/workaround. Give the video branch a real return type (a small pydantic model, or
  split `get_preview` into kind-specific tools) so a client gets a proper schema/structured result
  instead of having to parse JSON out of a text block.

## Out of scope
UI (T15). Multi-process/distributed job coordination — this project runs one server process, one
`_jobs` registry; the guard above only needs to work within that.

## Deliverables
Updated `pipeline/api.py` (`cancel`), `pipeline/containers/manager.py` if stage-exec-level stopping
needs new plumbing, `mcp_server/jobs.py` (concurrency guard), `mcp_server/server.py` +
`mcp_server/artifact_view.py` (typed preview return). New/updated tests for all three. Updated
`mcp_server/TOOLS.md` if `cancel_run`'s/`get_preview`'s documented behavior changes.

## Acceptance criteria
- Cancelling a run whose current stage is a real container exec actually stops that container
  (the locked, accepted mechanism — see above) — verified fast enough in practice (Bartosz's own
  2026-07-19 test) to use without a separate performance check; a stage with no safe interrupt path
  is honestly reported as not possible for a documented, specific reason, never a blanket
  `NotImplementedError` passthrough like today.
- Starting a second `run_pipeline`/`run_stage` call against a `run_id` already mid-flight is
  rejected with a clear, distinct error — proven by a test that starts one job, then asserts the
  second call's rejection, not just documented.
- `get_preview`'s video branch produces real `structuredContent` in a test, not just parseable
  `TextContent`.

## Relevant existing files
`pipeline/api.py::cancel`, `mcp_server/jobs.py`, `mcp_server/server.py::get_preview`,
`pipeline/containers/manager.py` (`ContainerManager.stop`/`stop_by_id` — no per-`exec` stop today),
`planning/tasks/T12-resource-manager.md`'s original cancellation scope note,
`planning/tasks/T14-mcp-tools-and-resources.md`'s implementation notes (where all three gaps were
first named).

## Notes / gotchas
Docker's `exec` API has no clean "kill just this exec" primitive — real cancellation of a
container-running stage means stopping (or killing) the *container* (`ContainerManager.stop`),
which also drops a warm container's in-memory state (not its persisted volumes). **This is the
locked, accepted mechanism (2026-07-19) — Bartosz tested a container stop directly and confirmed
it's fast**, so implement against it directly rather than re-litigating the trade-off. A future,
finer-grained per-`exec` cancellation is a nice-to-have, explicitly out of this task's scope —
revisit only if the container-stop approach proves too coarse in practice.
