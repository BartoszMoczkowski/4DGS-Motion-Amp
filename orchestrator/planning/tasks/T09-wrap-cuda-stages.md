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

## Real-hardware finding (2026-07-16/18) — `cuda` image build was broken, twice, before `train.default` ever ran

`train.default` was the first stage in this task to actually execute on real hardware, and it
surfaced two build-time bugs in the repo-root `Dockerfile` (not this task's own code) back to
back: (1) the venv-creation step was commented out entirely (fixed by uncommenting it — see
`.claude_notes/NOTES_pipeline_orchestration.md`'s "cuda image never had a real Python" entry); (2)
once that was fixed and rebuilt, `diff-gaussian-rasterization`/`simple-knn`'s CUDA-extension builds
crashed with `IndexError: list index out of range` inside `torch.utils.cpp_extension`, because
`docker build` has no GPU passthrough and `TORCH_CUDA_ARCH_LIST` wasn't set — fixed by adding
`ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"` (Bartosz's RTX 3090's compute capability) to the Dockerfile.
See that same notes file's "Root cause found (2026-07-18)" entry for the full traceback and
mechanism. Neither bug is in `pipeline/vendored/cuda/`, `cuda_common.py`, or this task's stage
wiring — both are one-line Dockerfile/build-environment gaps that simply had never been exercised
before this task's first real run.

**Third bug, same day:** after both fixes above, the *exact same* "python not found" error came
back — this time surfacing after a genuinely successful build, not a build failure. Root cause:
the Dockerfile built the venv (and, by the same mechanism, `diff-gaussian-rasterization`/
`simple-knn`'s editable-installed compiled extensions) inside `/workspace`, which
`pipeline/containers/config.py` bind-mounts over with the *live* host repo at container runtime —
a bind mount replaces the underlying image layer entirely rather than merging with it, so anything
the build wrote to `/workspace` was invisible the instant the container started, no matter how
correctly the image built. Fixed by moving the entire build to `/opt/build` in the Dockerfile
(`WORKDIR`, the dependency `COPY`s, `uv venv`/`uv sync`, the resulting `PATH`), then switching back
to `WORKDIR /workspace` for everything downstream — full detail in
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Root cause found and fixed (2026-07-18, later
same day)" entry. Three real, distinct build/runtime-environment bugs in total from this task's
very first real execution, each only surfacing once the previous one was fixed. Still not yet
confirmed past this point — Bartosz needs to re-run to see whether `train.default` (and everything
downstream) actually completes now that the image builds cleanly *and* what it builds is reachable
at runtime.

**Fourth bug, same run, this time in `pipeline.config.bridge` (T09's own code, not the Dockerfile):**
with the mount-shadowing fix in place, `train.py` finally ran for real and immediately crashed with
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97` reading its `--configs` bridge file.
`write_bridge` wrote that file without an explicit encoding, so on Bartosz's native Windows process
it defaulted to cp1252; an em dash in the file's own generated header became an invalid UTF-8 byte
once read inside the Linux `cuda` container. Fixed by writing (and, in an audit of the rest of
`pipeline/`, several other read/write call sites) with explicit `encoding="utf-8"` — full detail in
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Fifth bug, same real-hardware attempt" entry.
Four real bugs total from this task's first real execution now (Dockerfile venv, `TORCH_CUDA_ARCH_
LIST`, mount shadowing, this encoding bug) — still waiting on a real run that gets past all of them
at once.

**Fifth bug, next re-run, deepest into the chain yet:** `train.default` completed for real this
time and `render.default` started rendering before `amp.default` crashed with `AttributeError:
module 'mmcv' has no attribute 'Config'`. `pipeline/vendored/cuda/amp.py` (the `render_amp.py`
port) still called `mmcv.Config.fromfile(...)` where `train.py`/`render.py`/`seg_extract.py` all
already call `mmengine.Config.fromfile(...)` — the module's own docstring had wrongly filed this
under "kept as-is, pre-existing reference-script inconsistency," but the pinned `mmcv==2.2.0`
genuinely removed `Config` (moved to `mmengine`), so this was a guaranteed crash, not a
preservable quirk. Fixed both occurrences in `amp.py`, corrected the docstring. Full detail:
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Seventh bug, further re-run" entry (numbered
across the whole real-hardware saga, not just this task). Five real bugs now from this task's
first real execution, each surfacing only once the run got one stage further than the last.

**Sixth bug, same re-run, arguably the most important one:** past the `mmcv` fix, `amp.default`
crashed differently — `FileNotFoundError` on a missing `train_out/point_cloud` directory.
`train.default` had reported success (exit 0) but never actually saved a checkpoint. Root cause,
traced via the run's own bridge file and `arguments/__init__.py`: the vendored `train.py`'s
`args.save_iterations.append(args.iterations)` ran *before* the `--configs` merge applied this
project's `iterations` override (e.g. `100` for a smoke test), so it appended the stale pre-merge
argparse default (`30_000`) instead — meaning the actual final training iteration was never in
`save_iterations`, and `scene.save()` never fired. This is a **silent-success failure mode that
would hit every normal run** overriding `iterations` via config (which is the whole point of the
bridge mechanism) — not specific to any one preset. Fixed by moving the append to after the
`merge_hparams` call. Full detail: `.claude_notes/NOTES_pipeline_orchestration.md`'s "Eighth bug,
same re-run" entry. Six real bugs total from this task's first real execution now.

Also hardened `TrainStage.run` itself against this exact failure mode recurring, same principle as
`capture.isaac`'s `cameras_gt.json`/`camNN` check: it now verifies `model_host / "point_cloud"`
actually exists and is non-empty before reporting success, raising `CudaStageError` otherwise
rather than letting a bare exit-0 get cross-run cached. `tests/test_stages_cuda.py`'s two
train-exec fakes updated to stub a checkpoint on simulated success; new regression test
`test_train_stage_raises_if_exit_zero_but_no_checkpoint_written`. 163 tests total, all green.

## Verified end-to-end on real hardware (2026-07-19)

After closing some memory-heavy programs (the exit-137 crash a run earlier was consistent with a
host-RAM OOM, not GPU VRAM — see `.claude_notes/NOTES_pipeline_orchestration.md`'s SIGKILL
discussion), Bartosz re-ran the full chain and it completed genuinely end to end:
`train.default`/`render.default`/`seg_extract.default`/`amp.default` all reported real `"success"`
(not skipped), with a real `amp_video` (`render.mp4`) on disk. This is the first time any of these
four stages has actually run to completion on real hardware — T09 had been `done` since
2026-07-15 on sandbox verification alone (fake `exec_in_container`, no real Docker/GPU), and it
took six additional real bugs found across 2026-07-18/19 (see this file's own entries above, plus
`T08-container-manager.md`'s Dockerfile-side fixes) before a real run actually got through all
four stages in one pass. See `T11-wrap-isaac-stages.md`'s "Milestone M3 actually reached" entry for
the full-chain framing.
