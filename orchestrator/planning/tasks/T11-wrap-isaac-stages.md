# T11 — Wrap Isaac stages (split / motion / capture)

- Status: done (2026-07-16)
- Phase: 2
- Depends on: T08, T09
- Environment: isaac

## Goal
Bring the synthetic-capture front end under the orchestrator so the pipeline runs truly
end-to-end from a USD asset. (Milestone M3.)

## In scope
- Register `capture.isaac`, porting (copy, not import/subprocess) `omni_capture.py`'s logic into
  `pipeline/vendored/isaac/` (run via `/isaac-sim/python.sh` in the `isaac` container), producing
  the multi-cam capture + GT poses/segmentation as artifacts. Per `INSTRUCTIONS.md`'s copy-in
  rule, only the `isaac` container (Isaac Sim runtime) is external — `omni_capture.py` itself is a
  reference, not a dependency.
- Register `prep.split` and `prep.motion`, porting `split_mesh.py`/`add_motion.py`'s logic into
  `pipeline/vendored/isaac/` (or a small CPU image's vendored dir, if that's the env decided
  below) — USD/trimesh CPU work; decide env (small CPU image vs. reuse `isaac` which already has
  USD) and document it.
- Chain `prep.split → prep.motion → capture.isaac → convert` so the full DAG connects to T09's
  half.

## Out of scope
Authoring new scenes/motions (that's asset work, done via these stages).

## Deliverables
Three registered stages + a preset running the entire pipeline from `CONJUNTO_BOMBAS.usd` to amp.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- `run_capture.sh`'s smoke test (`--n-cameras 2 --frames 2`) reproduced via `run_stage`.
- Full `run_pipeline(preset="pump01")` from prep through amp completes; artifacts + manifest
  populated; warm Isaac container avoids repeated cold-start.

## Relevant existing files (reference only — do not import/call; port the logic into `pipeline/vendored/isaac/`)
`omniverse_pipeline/{split_mesh,add_motion,omni_capture}.py`,
`omniverse_pipeline/.devcontainer/{devcontainer.json,run_capture.sh}`,
`capture_config_pump.yaml`, NOTES_omniverse_pipeline.md.

## Notes / gotchas
Q: mount + `.pyc` staleness caused stale reads before (memory note) — prefer generate-to-tmp then
copy. Isaac cold-start/shader-cache is slow; rely on T08's warm container + persisted volumes.

## Implementation summary (2026-07-16)

Registered as `prep_split.default` / `prep_motion.default` / `capture.isaac` — **not**
`prep.split`/`prep.motion` as sketched above: `pipeline.stages.registry.register`'s `"role.impl"`
split takes everything before the first dot as the role, so those two names would have collided
into one ambiguous `"prep"` role (two impls that aren't actually alternatives of the same thing,
unlike `segment.rigid`/`segment.mbs`). Each is now its own single-impl role, matching its own
top-level `PipelineConfig` section (`prep_split`/`prep_motion`) 1:1.

Env decision (the "isaac/host*" open question above): both CPU-only stages run in the `isaac`
container, not a separate small-CPU image — no such image exists in this project, and building one
was judged out of this task's contained scope. `trimesh` (needed by `split_mesh.py`, not
preinstalled in the Isaac Sim image, unlike `pxr`/usd-core) needs a one-time manual `pip install`
in the container — see `planning/WINDOWS_SETUP.md`'s new Isaac-stages setup step.

`capture.isaac` writes a full `--config` YAML from the resolved `CaptureConfig` per call (not a
single shared bridge file like T09's `write_stage_bridge`), overriding `scene.usd_path`/
`output.capture_dir` via `omni_capture.py`'s own `--usd`/`--out` CLI flags with the DAG's real
`animated_mesh` input / run directory.

`pipeline.api.run_pipeline` gained an `external_artifacts` parameter (a real, separate gap found
while wiring `prep_split.default` in — see `.claude_notes/NOTES_pipeline_orchestration.md`'s "T11
done" section for the full writeup).

Full log: `.claude_notes/NOTES_pipeline_orchestration.md`'s "T11 done" section. Tests:
`tests/test_stages_isaac.py` (12 new, 151 total).

## Real-hardware fixup (2026-07-16)

First real run on Bartosz's machine (`tests/test_stages_isaac_gpu.py`) found two genuine bugs,
both fixed, both still needing a real re-run to confirm:

1. **`prep_split.default`/`prep_motion.default`: `ModuleNotFoundError: No module named 'pxr'`.**
   `split_mesh.py`/`add_motion.py` do a bare `from pxr import ...` with no Kit runtime bootstrap —
   this module's own docstring assumed `pxr` is importable straight from `/isaac-sim/python.sh`,
   true on some earlier Isaac Sim releases, false on the actual
   `nvcr.io/nvidia/isaac-sim:6.0.1` image (`pxr` is supplied by Kit's extension loader at
   `SimulationApp` init, not a static `sys.path` entry). Fixed by adding
   `pipeline/stages/_isaac_kit_bootstrap.py` — new orchestrator glue (not a vendored/ported copy)
   that launches a headless do-nothing `SimulationApp` before handing off to the real script via
   `runpy`. `pipeline.stages.isaac_common.run_isaac_script` routes `split_mesh`/`add_motion`
   through it automatically; `omni_capture` is untouched (it already launches its own
   `SimulationApp`). The vendored scripts themselves are still byte-for-byte copies.

2. **`capture.isaac`: reported manifest "success" but never wrote `cameras_gt.json`.** The log
   showed `PermissionError: [Errno 13] Permission denied: '/isaac-sim/.cache/warp'` at Kit startup,
   cascading into `omni.replicator.core` failing to load — so `rep.writers.get("BasicWriter")`
   later raised `WriterRegistryError`, but Kit still exited 0, masking the failure as a manifest
   success. Root cause: the persisted `isaac-cache` Docker volume (`/isaac-sim/.cache`) wasn't
   writable by whatever UID `exec_create` runs commands as. Fixed with a best-effort
   `chmod -R 0777` on the three cache-volume mount points
   (`ContainerManager._fixup_isaac_cache_permissions`), run once right after a fresh `isaac`
   container is created.

Both fixes are covered by new/updated sandbox tests (`tests/test_containers.py`,
`tests/test_stages_isaac.py`); full suite still green (151 passed, 9 skipped). Neither fix could
be verified against real Isaac Sim/GPU hardware from this session — re-run
`tests/test_stages_isaac_gpu.py` (`planning/WINDOWS_SETUP.md` step 8.4) to confirm.

## Second real-hardware fixup + architecture revision (2026-07-16, same day)

Bartosz re-ran for real after both fixes above. Progress confirmed: the permission fix worked
(Warp cache initialized cleanly, `BasicWriter` registered, `omni_capture.py`'s `main()` ran to
completion with no exception) — but a *third*, different failure appeared: `IHydraTexture ...
no GPU foundation` + repeated "Timed out while waiting for pending Replicator writer schedules to
drain" (once per frame). The RTX render products never produced any actual frame data, so no
`camNN/` output directories ever got written — `cameras_gt.json` and the point-cloud files still
did (they come from stage geometry, not rendering), so the stage's own success check needed
strengthening to also verify the camera-directory *count*, not just `cameras_gt.json`'s existence
(now in `CaptureIsaacStage.run`).

Root cause, confirmed via NVIDIA's own developer forum: **Vulkan (what Isaac Sim's Hydra/RTX
renderer needs) is not supported under WSL2**, which is what backs Docker Desktop's Linux
containers on Windows — a hard, NVIDIA-stated platform limitation, not a Docker config gap. CUDA
compute (what the `cuda` container/PyTorch training needs) works fine under WSL2's GPU
paravirtualization; Vulkan rendering doesn't.

**Decision (locked, see `INSTRUCTIONS.md`): `capture.isaac` now runs `omni_capture.py` as a native
Windows subprocess against Bartosz's own real Isaac Sim install, not inside the `isaac` Docker
container.** `prep_split.default`/`prep_motion.default` are unaffected (CPU-only, no rendering) and
keep using the `isaac` container exactly as before — no code change needed there.

Implementation:
- `pipeline/stages/isaac_common.py`: new `run_native_isaac_script` (subprocess-based, no
  `ctx.containers`/path-translation involved) alongside the unchanged `run_isaac_script`; new
  `native_isaac_python()`/`PIPELINE_ISAAC_NATIVE_PYTHON` env var (default
  `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat` — **corrected 2026-07-18**;
  originally `Q:\Omniverse\ISAAC_SIM\IsaacSim\tools\packman\python.bat`, the convention
  `omni_capture.py`'s own original pre-orchestrator docstring documented, but that path doesn't
  exist on Bartosz's real machine). Reuses
  `pipeline.containers.CONTAINER_ENV["isaac"]`'s EULA env vars (pure data, no `docker` import
  pulled in).
- `pipeline/stages/capture_isaac.py`: drops all `ctx.paths.to_container(...)` calls — a native
  subprocess shares the orchestrator's own filesystem, so every path stays a plain host path.
  Calls `run_native_isaac_script` instead of `run_isaac_script`. The `cameras_gt.json` +
  camera-directory-count success checks are unchanged in spirit, strengthened as described above.
- `tests/test_stages_isaac.py`: capture-specific tests now fake `run_native_isaac_script` directly
  (`_FakeNativeIsaac`/`_fake_run_native_isaac`) instead of going through `ctx.containers`; the
  `run_dag` end-to-end chain test's call-count assertions split into container-exec calls
  (`split_mesh`/`add_motion`, now 2 not 3 per run) and native calls (`omni_capture`, 1 separate).
  Added `test_capture_isaac_stage_raises_if_cameras_gt_json_never_written` as a regression test for
  the false-success gap. Full suite green (155 passed, 9 skipped).
- `runs/.cache/index.json` deleted (poisoned by the earlier false "success" runs — cross-run
  caching has no way to know a recorded success wasn't real; see the scheduler's own cache-key
  design, `pipeline/dag/scheduler.py`).

Full write-up (including the NVIDIA forum citations and the diagnosis conversation) in
`.claude_notes/NOTES_pipeline_orchestration.md`'s "adjust the project plan" entry. Docs updated:
`ARCHITECTURE.md`'s DAG table + Layer 1 bullets, `INSTRUCTIONS.md`'s locked decisions +
"Environments" section, `WINDOWS_SETUP.md`'s step 8.5.

## Milestone M3 actually reached on real hardware (2026-07-19)

After the native-Isaac fix above, real-hardware re-runs surfaced eight further real bugs one at a
time — each only visible once the previous one was cleared — across the `cuda`-container stages
this task's chain feeds into: the repo `Dockerfile`'s venv-creation step being commented out,
`TORCH_CUDA_ARCH_LIST` missing (breaking CUDA-extension builds under `docker build`'s no-GPU
context), `/workspace` mount-shadowing swallowing anything the Dockerfile built there, a
cp1252-vs-UTF-8 encoding bug in `pipeline.config.bridge`'s generated config file, a wrong default
Isaac `python.bat` path, `pipeline/vendored/cuda/amp.py` still calling the removed `mmcv.Config`
API, and a `save_iterations` ordering bug in `train.py` that silently skipped writing any
checkpoint. Full detail for each: `.claude_notes/NOTES_pipeline_orchestration.md`'s
2026-07-18/2026-07-19 entries; `T08-container-manager.md` and `T09-wrap-cuda-stages.md`'s matching
sections.

**Confirmed 2026-07-19: the full `prep_split → prep_motion → capture.isaac → convert → train →
render → seg_extract → segment.rigid → amp` chain completed for real**, producing a genuine
amplified video (`train_out/video/render.mp4`, confirmed on disk) — the actual criterion-2
acceptance line this task always specified, now met for real rather than "wiring verified, not yet
run." (`tests/test_stages_isaac_gpu.py::test_pump01_prep_through_amp_completes`'s own per-stage
assertion had a bug that made this run *look* like a failure — it required every stage's status
`== "success"` literally, rejecting `"skipped"` for stages correctly served from the cross-run
cache; fixed to accept either, since a cache hit on unchanged upstream stages is the intended
outcome, not a failure.)
