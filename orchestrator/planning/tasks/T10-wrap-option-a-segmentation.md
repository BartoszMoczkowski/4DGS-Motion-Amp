# T10 — Wrap Option-A segmentation (mbs_infer)

- Status: done (2026-07-15) — implemented + sandbox-verified; real GPU run pending Bartosz's machine
- Phase: 1
- Depends on: T09
- Environment: cuda (+ MBS `ext/` CUDA ops)

## Goal
Add the MBS-based segmentation backend as a second impl under the `segment` role, so A vs B is a
config switch — the concrete demonstration of the plugin/extensibility model (problem #3).

## In scope
- Register `segment.mbs`, porting (copy, not import/subprocess) `motion_seg/mbs_infer.py`'s logic
  into `pipeline/vendored/cuda/`, consuming the same `trajectories.npz` as `segment.rigid` and
  producing the same segmentation artifact shape (so `seg_eval` is backend-agnostic). Per
  `INSTRUCTIONS.md`'s copy-in rule, only the `cuda` container (+ MBS `ext/` CUDA ops built inside
  it) is external — `mbs_infer.py` itself is a reference, not a dependency.
- Config: MBS params (working-set size, checkpoint path), plus a documented setup step for
  building MBS `ext/` ops and obtaining the checkpoint (currently untested per NOTES §6d).
- Preset that selects `segment.mbs` instead of `segment.rigid`.

## Out of scope
Fine-tuning/retraining MBS; fixing any out-of-distribution quality issues (research, not infra).

## Deliverables
`segment.mbs` stage + preset variant; setup notes for `ext/` + checkpoint.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- Swapping the preset's `segment` impl from `rigid` to `mbs` runs the alternate backend with no
  other change; `seg_eval` scores both identically-formatted outputs.
- First real run documented (expect shape/behaviour debugging per NOTES §6d).

## Relevant existing files (reference only — do not import/call; port the logic into `pipeline/vendored/cuda/`)
`motion_seg/mbs_infer.py` (untested), `submodules/multibody-sync-4dgs/` (`ext/`, `test.py`,
`models/`), NOTES_4dgs_motion_segmentation.md §6d.

## Notes / gotchas
This task doubles as the proof that "add a new idea = register an impl + a preset". Keep the
artifact contract identical to `segment.rigid` so downstream stages don't care which ran.

## Log (2026-07-15)

Implemented following T09's `cuda`-stage shape (MotNet needs the GPU, unlike `segment.rigid`'s
CPU-only `host` impl), not T07's in-process-import shape:

- `pipeline/vendored/cuda/mbs_infer.py` — verbatim port of `motion_seg/mbs_infer.py`'s
  `_load_mot_net`/`_select_working_set`/`run_mbs_segmentation`/`main` (own argparse CLI), per the
  copy-in rule. Two purely mechanical relocation fixes, no logic changes: `MBS_ROOT`'s relative
  path walk fixed for the file's new location (4 hops up instead of 1, since it moved from
  `motion_seg/` to `pipeline/vendored/cuda/`), and the reference script's `_REPO_ROOT` sys.path
  hack + its `main()`-tail lazy `from motion_seg.visualize import render_segmentation_png`
  preview-PNG block were dropped — both existed only to reach into `motion_seg`, the exact
  throwaway-script reference the copy-in rule forbids depending on. Preview-PNG generation stays
  unwired here, same as `segment_rigid.py`'s `preview_png` and `seg_eval.py`'s `comparison_png`
  (neither of those is wired up either — this isn't a new gap).
- `pipeline/stages/segment_mbs.py` — new `SegmentMbsStage`, `@register("segment.mbs")`,
  `environment="cuda"`, `inputs=("trajectories",)`, `outputs=("segmentation",)` — the identical
  contract `segment.rigid` (T07) declares. Never calls `write_stage_bridge` (T09's bridge-file
  plumbing): `mbs_infer.py`'s CLI needs none of the 4DGS `ModelParams`/etc. groups, and
  `pipeline.api._stage_config_for`'s `"_bridge"` merge is (correctly) scoped to
  `train`/`render`/`seg_extract`/`amp` only, not `segment`. Added one small piece of new logic
  beyond a plain CLI-arg builder: `SegmentMbsConfig.checkpoint` (T02, already in `models.py`) has
  no sensible default, so a relative value is resolved against `ctx.paths.get_roots()
  .repo_root_host` before the real host<->container translation (`ctx.paths.to_container`) —
  documented in `_resolve_checkpoint_host`'s own docstring as a stage-local convenience (filling
  in a missing base), not a T06 path-translation change.
- `pipeline/stages/cuda_common.py`'s `VENDORED_CUDA_SCRIPTS` gained a `"mbs_infer"` entry;
  `pipeline/stages/__init__.py` now imports `segment_mbs` for its registration side-effect.
- `pipeline/config/presets/pump01_segA.yaml` — new experiment preset, `extends: pump01`,
  `segment.impl: mbs`, checkpoint pointing at `submodules/multibody-sync-4dgs/ckpt/
  mbs_full.pth.tar` (repo-relative, resolved as above).
- `planning/WINDOWS_SETUP.md` gained a new "7. Option-A segmentation (MBS) setup" step: the
  `ext/` CUDA ops need no manual build step (they JIT-compile on first `segment.mbs` run, inside
  the already-`-devel` `cuda` image); the checkpoint download+placement *is* manual (no vendored
  weights, Google-Drive-hosted, per the reference script's own docstring and
  `.claude_notes/NOTES_4dgs_motion_segmentation.md` §6d).

**Verified in the sandbox** (no GPU/Docker/torch/compiled `ext/` ops here, same limitation T09
had) — 10 new tests in `tests/test_stages_mbs.py`, same strategy as T09's own
`test_stages_cuda.py`: fake `exec_in_container`, so what's actually exercised is CLI-argument
construction, the checkpoint relative/absolute-path resolution, the `"_bridge"`-exclusion, a
nonzero-exit failure path, and — the acceptance criteria's actual point —
`get_stage("segment.rigid").{inputs,outputs} == get_stage("segment.mbs").{inputs,outputs}` and
`pipeline.api._auto_stage_plan` picks `segment.rigid` for the `base` preset (default
`segment.impl`) vs. `segment.mbs` for `pump01_segA`, with nothing else about the DAG plan
changing. Full suite: 132 passed, 6 skipped (`test_containers_gpu.py`, same as before), plus 5
pre-existing failures in `tests/test_containers.py` unrelated to this task (a `_FakeExecAPI`/
`docker`-package interaction in this sandbox specifically — not something T10 touches, and not
present when T08/T09 originally ran their own suites here).

**Not yet done, needs Bartosz's machine** (documented, not swept under the rug, same as T09's own
status note): the actual first real run — MBS `ext/` ops JIT-compiling for real, a downloaded
checkpoint loading, MotNet inference producing a real segmentation, and `seg_eval.default` scoring
it — per this task's own acceptance criteria ("First real run documented (expect shape/behaviour
debugging per NOTES §6d)"). `planning/WINDOWS_SETUP.md`'s new step 7 has the checkpoint-download
instructions; run `pump01_segA` once `seg_extract.default` has already produced a real
`trajectories.npz` for `pump01`.
