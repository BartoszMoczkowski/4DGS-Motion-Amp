# T10 — Wrap Option-A segmentation (mbs_infer)

- Status: todo
- Phase: 1
- Depends on: T09
- Environment: cuda (+ MBS `ext/` CUDA ops)

## Goal
Add the MBS-based segmentation backend as a second impl under the `segment` role, so A vs B is a
config switch — the concrete demonstration of the plugin/extensibility model (problem #3).

## In scope
- Register `segment.mbs` wrapping `motion_seg/mbs_infer.py`, consuming the same `trajectories.npz`
  as `segment.rigid` and producing the same segmentation artifact shape (so `seg_eval` is
  backend-agnostic).
- Config: MBS params (working-set size, checkpoint path), plus a documented setup step for
  building MBS `ext/` ops and obtaining the checkpoint (currently untested per NOTES §6d).
- Preset that selects `segment.mbs` instead of `segment.rigid`.

## Out of scope
Fine-tuning/retraining MBS; fixing any out-of-distribution quality issues (research, not infra).

## Deliverables
`segment.mbs` stage + preset variant; setup notes for `ext/` + checkpoint.

## Acceptance criteria (Bartosz's WSL2 machine)
- Swapping the preset's `segment` impl from `rigid` to `mbs` runs the alternate backend with no
  other change; `seg_eval` scores both identically-formatted outputs.
- First real run documented (expect shape/behaviour debugging per NOTES §6d).

## Relevant existing files
`motion_seg/mbs_infer.py` (untested), `submodules/multibody-sync-4dgs/` (`ext/`, `test.py`,
`models/`), NOTES_4dgs_motion_segmentation.md §6d.

## Notes / gotchas
This task doubles as the proof that "add a new idea = register an impl + a preset". Keep the
artifact contract identical to `segment.rigid` so downstream stages don't care which ran.
