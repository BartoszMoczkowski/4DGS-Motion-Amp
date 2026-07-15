# T09 — Wrap CUDA stages (train / render / seg_extract / amp)

- Status: done (2026-07-15)
- Phase: 1
- Depends on: T07, T08
- Environment: cuda

## Goal
Run the reconstruction→segmentation→amplification half from one call inside the `cuda` container.
(Milestone M2 — most of problem #1's value.)

## In scope
- Port (copy, not import/subprocess) the already-verified logic of `train` (`train.py`), `render`
  (`render.py`), `seg_extract` (`motion_seg/extract_trajectories.py`), `amp` (`render_amp.py`)
  into `pipeline/vendored/cuda/`. Per the "copy the logic in, don't call the original script" rule
  (`INSTRUCTIONS.md`), the *only* externally-depended-on thing here is the `cuda` container image
  itself (PyTorch/CUDA runtime) — never the script files in the repo root.
- Each stage: build its config from the unified config (emit the temp
  `arguments/multipleview/<name>.py` bridge from T02 for train/render), run the vendored-in code
  via the container manager inside `cuda` (T08 makes `pipeline/vendored/cuda/` available inside the
  container — bind-mount or image layer), register outputs (trained model, renders,
  `trajectories.npz`, amp video) as artifacts.
- Chain with the T07 CPU stages into the full `train → render → seg_extract → segment.rigid →
  seg_eval → amp` graph.

## Out of scope
Isaac capture (T11); Option-A seg (T10); adaptive memory (T12).

## Deliverables
Four registered CUDA stages + a full-half pipeline preset runnable end-to-end.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- `run_pipeline(preset="pump01", from_stage="train", to_stage="amp")` reproduces what the current
  `train_pump.sh` + `render` + `motion_seg/run.sh` + amp produce, into the same output locations.
- Caching: rerunning after training skips `train` and reuses the model.
- Parity check vs. a known-good manual run (segmentation ARI/IoU + amp video sanity).

## Relevant existing files (reference only — do not import/call; port the logic into `pipeline/vendored/cuda/`)
`train.py`, `render.py`, `motion_seg/extract_trajectories.py`, `render_amp.py`,
`omniverse_pipeline/train_pump.sh`, `motion_seg/run.sh`.

## Notes / gotchas
`extract_trajectories.py` needs the same env as `train.py` (loads a trained model). `train` is the
long pole (hours) — make sure logs stream live and the manifest updates during, not just after.

## Done (2026-07-15)

Design departed from the "import the vendored function in-process" pattern T07 used for `host`
stages: `train.py`/`render.py`/`extract_trajectories.py`/`render_amp.py`'s real dependencies
(`torch`, `arguments`, `scene`, `gaussian_renderer`, `diff_gaussian_rasterization`, `motion_amp`)
only exist inside the `cuda` container, so the four vendored modules
(`pipeline/vendored/cuda/{train,render,seg_extract,amp}.py`) are verbatim ports that keep their
own `argparse` CLI/`if __name__ == "__main__":` entry point, executed as a separate process via
`ctx.containers.exec_in_container` (T08) — never imported into the orchestrator's own host
process. `pipeline/stages/cuda_common.py` holds the shared CLI-flag builders + the
`PYTHONPATH=/workspace` exec env (a container-side script's own directory, not the exec `workdir`,
is what Python puts on `sys.path[0]` — needed for `from arguments import ...` etc. to resolve) +
the container-exec-and-raise-on-nonzero wrapper.

New `pipeline/config/bridge.py` writes a temp `arguments/multipleview/<name>.py`-style file (the
`ModelParams`/`PipelineParams`/`ModelHiddenParams`/`OptimizationParams` dict literals
`utils/params_utils.merge_hparams` reads) from the resolved config each stage call — this is how
config stays the single source of truth even for the field types (`kplanes_config`, `multires`,
...) `arguments/__init__.py`'s naive `ParamGroup` argparse wiring can't round-trip from a CLI
string. `source_path`/`model_path` are excluded from the bridge (derived per-run from the DAG's
own artifact wiring instead — passed as explicit CLI flags — since `merge_hparams` would
otherwise silently override them). `pipeline.api._stage_config_for` gained a `"_bridge"` merge for
exactly the `train`/`render`/`seg_extract`/`amp` roles (the 4 param groups above, alongside the
stage's own section) so each cuda stage's `ctx.config` has what it needs without handing it the
*whole* resolved config — which would have defeated T05's cache-key scoping (an unrelated section
changing must not invalidate `train`'s cache; the same bug class T07 already fixed once).

Found and fixed two more real pre-existing gaps while wiring real stages in (same pattern as
T07's `ctx.inputs` fix): `pipeline.dag.scheduler.run_dag` never actually set `ctx.paths`/
`ctx.containers` (both stayed `None` since T04, even after T06/T08 landed) — now set to the
`pipeline.paths`/`pipeline.containers` modules themselves on every stage call (host stages simply
never read them). `ContainerManager.exec`/`exec_in_container` gained an `environment=` kwarg
(docker-py's `exec_create` supports this directly) to carry the `PYTHONPATH` fix through to the
real Docker call; `tests/test_containers.py`'s fake exec API updated + two new tests cover it.

`train.default` writes its model under `ctx.run_dir/train_out` (not the legacy global
`output/multipleview/<name>/`) so every run's model is artifact/cache-tracked like everything else
T03 manages. `render.default` re-registers the same model directory as its own `renders` artifact
(no iteration-number parsing needed — `render.py`'s `iteration=-1` "load latest" is only resolved
*inside* the container process). `seg_extract.default`/`amp.default` compute their output paths
(`trajectories.npz`, `<model_path>/video/<video_path>`) deterministically and pass them explicitly
via `--out`/`--video_path` rather than relying on the reference script's own path defaults.
`amp.default` fails fast (`AmpFactorNotIntegerError`, before ever touching the container) if a
channel's `factor` isn't a whole number — `render_amp.py`'s `--amp_factors` is `type=int`, a
pre-existing quirk kept as-is per "copy the logic in, don't rewrite."

Verified in an isolated venv (no GPU/Docker — same story as T08's own unit tests): CLI-argument
construction, bridge-file content (round-tripped via `exec()`, mirroring the trust model
`pipeline.config.loader`/`mmengine.Config.fromfile` already use), `_stage_config_for`'s `_bridge`
merge and its cache-key scoping, and a `train.default -> render.default` run through `run_dag`
against a fake `pipeline.containers.exec_in_container` (not a fake Docker SDK — T08 already covers
that layer) showing cross-run caching (`skipped` on an unchanged rerun, model reused not
retrained). 33 new tests (146 total across the suite), all green. Real GPU/container execution
(the acceptance criteria's actual `run_pipeline(preset="pump01", from_stage="train", to_stage=
"amp")` end-to-end run, parity vs. a manual `train_pump.sh`/`motion_seg/run.sh` run, and the
caching behavior against the *real* Docker daemon) needs Bartosz's Windows + Docker Desktop + GPU
machine — not yet run for real, same as T08 before its 2026-07-15 GPU verification pass. Full log
in `.claude_notes/NOTES_pipeline_orchestration.md`.
