# T14 — MCP tools & resources

- Status: todo
- Phase: 4
- Depends on: T13, T09
- Environment: host (WSL2)

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
