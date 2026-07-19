# Tools & resources (T14) — a usage guide for Claude

The T13 server proved the transport; this is the actual control + read surface. Read
`CONNECTING.md` first for how to reach the server at all (host/port/token). Everything below
assumes an authenticated connection is already established.

## Typical flow

1. `list_presets` → pick one, or `validate_config(preset)` to see its fully-resolved settings
   without running anything.
2. `run_pipeline(preset, ...)` → returns `{"run_id": ...}` **immediately**. This does not wait for
   the run to finish — training alone can take hours. If the DAG needs external inputs nothing in
   the run itself produces (most commonly `prep_split.default`'s `raw_mesh`), pass them via
   `external_artifacts` or the call will fail (see "Async jobs & failures" below).
3. Poll `get_run_status(run_id)` and `tail_logs(run_id, stage)` to watch progress. `list_runs` /
   `list_artifacts(run_id)` for a broader view across many runs.
4. Once a stage finishes, `read_artifact(run_id, artifact_name)` for a text/JSON/npz/directory
   summary, or `get_preview(run_id, artifact_name)` for a `png`/`video` artifact — this is what
   actually lets you *see* a segmentation preview, a render, or an amp clip, not just read a path.

## Tools

| Tool | What it does |
|---|---|
| `list_presets()` | Available config presets. |
| `validate_config(preset)` | Fully-resolved config for `preset`, or a validation error. |
| `run_pipeline(preset, external_artifacts=None, from_stage=None, to_stage=None, only=None, force=False)` | Launch/resume a preset's auto-planned DAG. Returns `run_id` immediately (async — see below). |
| `run_stage(run_id, stage, force=False)` | Run one stage of an existing run. Returns immediately, same async pattern. |
| `get_run_status(run_id)` | Per-stage status/timing/artifacts/peak-mem, plus `job_error` (see below). |
| `tail_logs(run_id, stage, max_lines=200)` | Last N lines of a stage's log — what it's doing *right now*. |
| `list_runs()` | Summaries of every known run, most recent first. |
| `list_artifacts(run_id)` | Every artifact a run has produced so far. |
| `read_artifact(run_id, artifact_name)` | Small JSON-safe summary of one artifact — never the raw file. json → parsed content; npz → per-key shape/dtype/min/max/mean (never raw arrays); dataset/model → shallow directory listing; ply → vertex/face counts. |
| `get_preview(run_id, artifact_name)` | A `png` artifact comes back as an inline image. A `video` artifact is **not** inlined (no MCP video content type, clips are too large) — returns its path + size + a `run://<run_id>/artifact/<name>` resource URI instead. Errors for any other kind. |
| `gpu_status()` | Current VRAM/RAM + free headroom. `None` sub-dicts mean unmeasurable here, not an error. |
| `list_containers()` / `start_container(env)` / `stop_container(container_id)` | Managed `cuda`/`isaac` container control. Rarely needed directly — a stage starts what it needs on its own. |
| `cancel_run(run_id)` | Best-effort. Layer 1 has no real cancellation implemented yet (T12's scope note) — this confirms the run exists and honestly reports that it can't actually be stopped from here; the run continues. |

## Resources

For a client that prefers reading over calling a tool — same underlying data:

- `run://<run_id>/manifest` — the full manifest (same content as `get_run_status`, minus the
  transient `job_error` field).
- `run://<run_id>/log/<stage>` — the *complete* (untruncated) log text for one stage.
- `run://<run_id>/artifact/<artifact_name>` — raw bytes of one file-kind artifact. What
  `get_preview`'s video pointer directs you to fetch; also usable for a png instead of the inline
  `get_preview` form. Errors for a directory-kind (`dataset`/`model`) artifact — use
  `read_artifact` for those.

## Async jobs & failures

`run_pipeline`/`run_stage` never block for the run's own duration — they validate what can be
checked synchronously (preset resolves, `run_id` exists) and then hand execution to a background
thread, returning right away. Two consequences:

- **Normal stage failures** (a real error inside a stage) show up exactly where you'd expect: that
  stage's own `"failed"` status + `error` message in `get_run_status`'s `stages` dict. Downstream
  stages stay `"pending"`.
- **A failure before any stage even started** (e.g. a missing external input, an unregistered
  stage name in `run_stage`) has nowhere in the manifest to land — `get_run_status`'s `job_error`
  field surfaces it instead. `job_error` is `None` in the overwhelming common case; check it if a
  run looks stuck `"pending"` with no explanation.

## Result shaping

Every tool here is designed to stay small: artifacts are summarized, not dumped (a multi-GB point
cloud or checkpoint directory never comes back as raw bytes in a tool response); logs are tailed,
not streamed whole (fetch the resource for the full text); a video preview is a pointer, not
inlined bytes. If something you need isn't covered, that's worth flagging rather than working
around — see `planning/tasks/T14-mcp-tools-and-resources.md`.
