# T07 — Wrap CPU stages (vertical slice)

- Status: done (redone 2026-07-14 under the copy-in policy; see "Reopened" section below)
- Phase: 0
- Depends on: T02, T05, T06
- Environment: host

## Goal
First end-to-end proof of the framework on the stages that need no GPU/containers — validates
config → registry → DAG → cache → artifacts before adding container complexity. (Milestone M1.)

## Reopened 2026-07-14
Previously marked `done` (2026-07-13). That implementation had `convert.py`/`segment_rigid.py`/
`seg_eval.py` `sys.path.insert` the repo root and `import` straight from `omniverse_pipeline.
omni_to_4dgs` / `motion_seg.segment_rigid` / `motion_seg.evaluate_segmentation` — a live runtime
dependency on scripts that are only a testing reference, not something this project should call.
Superseded by the "copy the logic in, don't call the original script" rule
(`INSTRUCTIONS.md`, `ARCHITECTURE.md`'s "Vendored stage logic"). Reopening this task to redo the
three stage bodies against the new rule; nothing about the scheduler/config/artifact plumbing
around them needs to change.

## In scope
- Port (copy, not import) the already-verified logic for `convert` (`omni_to_4dgs.py`'s
  `convert()`), `segment.rigid` (`segment_rigid.py`'s `segment_trajectories()`), and `seg_eval`
  (`evaluate_segmentation.py`'s `evaluate()`/`_write_colored_ply()`) into
  `pipeline/vendored/host/`. All pure numpy/CPU, already verified on synthetic data.
- Each stage: declare inputs/outputs, translate paths, call the vendored-in function (no
  `sys.path` reach into `omniverse_pipeline`/`motion_seg`), register produced artifacts, stream
  logs to the run dir.
- A `segment.rigid`-centred mini-pipeline runnable via `run_pipeline` from a preset.
- Remove the `sys.path.insert(str(_REPO_ROOT), ...)` blocks and the `omniverse_pipeline.*` /
  `motion_seg.*` imports from `pipeline/stages/convert.py`, `segment_rigid.py`, `seg_eval.py`.

## Out of scope
GPU stages (T09); containers (T08).

## Deliverables
`pipeline/vendored/host/` with the three ported functions + three registered stages calling them
+ a working `run_pipeline(preset="...", only=[...])` over them.

## Acceptance criteria
- Runs the convert→segment→eval slice from one call, using existing synthetic self-test fixtures
  (`segment_rigid.py --selftest` data) where real inputs aren't present.
- No stage module imports `omniverse_pipeline.*`/`motion_seg.*` or does a `sys.path` insert to
  reach outside `orchestrator/` — `grep -r "sys.path" pipeline/stages/` comes back empty.
- Cache: rerun skips unchanged stages; changing `--threshold-mult` in the preset reruns only
  segment + eval.
- Manifest + artifacts populated and queryable.

## Relevant existing files (reference only — do not import/call)
`omniverse_pipeline/omni_to_4dgs.py` (has `--selftest`), `motion_seg/segment_rigid.py`
(`--selftest`), `motion_seg/evaluate_segmentation.py`, `motion_seg/run.sh` (ordering reference).

## Notes / gotchas
Fully verifiable in the CPU sandbox — this is the task that de-risks the whole framework. Lean on
the existing `--selftest` paths for acceptance without needing a trained model. When copying a
function over, port it verbatim (don't redesign it in transit) — `pipeline/vendored/host/` is a
copy, not a rewrite.

## Redone 2026-07-14 (copy-in rework)
Ported `convert()` (+ its geometry/COLMAP-writer helpers), `segment_trajectories()` (+
`rigidity_graph.py`'s `segment_by_rigidity`/`build_knn_edges`/`edge_rigidity_score`/
`otsu_threshold`/`otsu_threshold_log`/`merge_small_components`), and `evaluate()`/
`propagate_labels()`/`_write_colored_ply()` (+ `metrics.py`'s `adjusted_rand_index`/
`best_iou_matching`) verbatim into `pipeline/vendored/host/{convert,segment_rigid,rigidity_graph,
seg_eval,metrics}.py`. The three stage modules (`pipeline/stages/{convert,segment_rigid,
seg_eval}.py`) now import from `..vendored.host.*` instead of `sys.path`-hacking into
`omniverse_pipeline`/`motion_seg`; only the import line changed, stage bodies are untouched.
Removed the stale "wrap, don't rewrite" framing from `pipeline/stages/base.py` and
`pipeline/stages/__init__.py`'s module docstrings. Verified: `grep -r "sys.path"
pipeline/stages/` and `grep -r "omniverse_pipeline\|motion_seg" pipeline/stages/` (source imports)
both empty; full suite `pytest -q` from `orchestrator/` → 106 passed (same count as before the
rework — no behavior change, only where the code physically lives). See
`.claude_notes/NOTES_pipeline_orchestration.md`'s T07-redone log entry for detail.
