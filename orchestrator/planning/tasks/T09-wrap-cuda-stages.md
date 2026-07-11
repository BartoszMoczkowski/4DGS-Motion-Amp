# T09 — Wrap CUDA stages (train / render / seg_extract / amp)

- Status: todo
- Phase: 1
- Depends on: T07, T08
- Environment: cuda

## Goal
Run the reconstruction→segmentation→amplification half from one call inside the `cuda` container.
(Milestone M2 — most of problem #1's value.)

## In scope
- Register stages wrapping: `train` (`train.py`), `render` (`render.py`), `seg_extract`
  (`motion_seg/extract_trajectories.py`), `amp` (`render_amp.py`).
- Each: build the command from the unified config (emit the temp `arguments/multipleview/<name>.py`
  bridge from T02 for train/render), run via the container manager in `cuda`, register outputs
  (trained model, renders, `trajectories.npz`, amp video) as artifacts.
- Chain with the T07 CPU stages into the full `train → render → seg_extract → segment.rigid →
  seg_eval → amp` graph.

## Out of scope
Isaac capture (T11); Option-A seg (T10); adaptive memory (T12).

## Deliverables
Four registered CUDA stages + a full-half pipeline preset runnable end-to-end.

## Acceptance criteria (Bartosz's WSL2 machine)
- `run_pipeline(preset="pump01", from_stage="train", to_stage="amp")` reproduces what the current
  `train_pump.sh` + `render` + `motion_seg/run.sh` + amp produce, into the same output locations.
- Caching: rerunning after training skips `train` and reuses the model.
- Parity check vs. a known-good manual run (segmentation ARI/IoU + amp video sanity).

## Relevant existing files
`train.py`, `render.py`, `motion_seg/extract_trajectories.py`, `render_amp.py`,
`omniverse_pipeline/train_pump.sh`, `motion_seg/run.sh`.

## Notes / gotchas
`extract_trajectories.py` needs the same env as `train.py` (loads a trained model). `train` is the
long pole (hours) — make sure logs stream live and the manifest updates during, not just after.
