# T11 — Wrap Isaac stages (split / motion / capture)

- Status: todo
- Phase: 2
- Depends on: T08, T09
- Environment: isaac

## Goal
Bring the synthetic-capture front end under the orchestrator so the pipeline runs truly
end-to-end from a USD asset. (Milestone M3.)

## In scope
- Register `capture.isaac` wrapping `omni_capture.py` (run via `/isaac-sim/python.sh` in the
  `isaac` container), producing the multi-cam capture + GT poses/segmentation as artifacts.
- Register `prep.split` (`split_mesh.py`) and `prep.motion` (`add_motion.py`) — USD/trimesh CPU
  work; decide env (small CPU image vs. reuse `isaac` which already has USD) and document it.
- Chain `prep.split → prep.motion → capture.isaac → convert` so the full DAG connects to T09's
  half.

## Out of scope
Authoring new scenes/motions (that's asset work, done via these stages).

## Deliverables
Three registered stages + a preset running the entire pipeline from `CONJUNTO_BOMBAS.usd` to amp.

## Acceptance criteria (Bartosz's WSL2 machine)
- `run_capture.sh`'s smoke test (`--n-cameras 2 --frames 2`) reproduced via `run_stage`.
- Full `run_pipeline(preset="pump01")` from prep through amp completes; artifacts + manifest
  populated; warm Isaac container avoids repeated cold-start.

## Relevant existing files
`omniverse_pipeline/{split_mesh,add_motion,omni_capture}.py`,
`omniverse_pipeline/.devcontainer/{devcontainer.json,run_capture.sh}`,
`capture_config_pump.yaml`, NOTES_omniverse_pipeline.md.

## Notes / gotchas
Q: mount + `.pyc` staleness caused stale reads before (memory note) — prefer generate-to-tmp then
copy. Isaac cold-start/shader-cache is slow; rely on T08's warm container + persisted volumes.
