# T07 — Wrap CPU stages (vertical slice)

- Status: todo
- Phase: 0
- Depends on: T02, T05, T06
- Environment: host

## Goal
First end-to-end proof of the framework on the stages that need no GPU/containers — validates
config → registry → DAG → cache → artifacts before adding container complexity. (Milestone M1.)

## In scope
- Wrap as registered stages: `convert` (`omni_to_4dgs.py`), `segment.rigid`
  (`segment_rigid.py`), `seg_eval` (`evaluate_segmentation.py`). All pure numpy/CPU, already
  verified on synthetic data.
- Each stage: declare inputs/outputs, translate paths, invoke the existing script in `host` env,
  register produced artifacts, stream logs to the run dir.
- A `segment.rigid`-centred mini-pipeline runnable via `run_pipeline` from a preset.

## Out of scope
GPU stages (T09); containers (T08).

## Deliverables
Three registered stages + a working `run_pipeline(preset="...", only=[...])` over them.

## Acceptance criteria
- Runs the convert→segment→eval slice from one call, using existing synthetic self-test fixtures
  (`segment_rigid.py --selftest` data) where real inputs aren't present.
- Cache: rerun skips unchanged stages; changing `--threshold-mult` in the preset reruns only
  segment + eval.
- Manifest + artifacts populated and queryable.

## Relevant existing files
`omniverse_pipeline/omni_to_4dgs.py` (has `--selftest`), `motion_seg/segment_rigid.py`
(`--selftest`), `motion_seg/evaluate_segmentation.py`, `motion_seg/run.sh` (ordering reference).

## Notes / gotchas
Fully verifiable in the CPU sandbox — this is the task that de-risks the whole framework. Lean on
the existing `--selftest` paths for acceptance without needing a trained model.
