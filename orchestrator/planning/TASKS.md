# Tasks — index & board

Each task is a contained unit of work with its own spec in `tasks/`. Work top-down within a phase;
respect the dependency graph. Update the `Status` line in each task file **and** this board.

## Status board

| ID | Title | Phase | Depends on | Status |
|----|-------|-------|-----------|--------|
| T01 | Subproject scaffold & tooling | 0 | — | done |
| T02 | Unified config schema & presets | 0 | T01 | done |
| T03 | Artifact store & run manifest | 0 | T01 | done |
| T04 | Stage base class & registry | 0 | T01 | done |
| T05 | DAG scheduler & caching | 0 | T03, T04 | done |
| T06 | Path-translation module | 0 | T01 | done |
| T07 | Wrap CPU stages (vertical slice) | 0 | T02, T05, T06 | done |
| T08 | Container manager | 1 | T05, T06 | done |
| T09 | Wrap CUDA stages (train/render/seg_extract/amp) | 1 | T07, T08 | done |
| T10 | Wrap Option-A segmentation (mbs_infer) | 1 | T09 | done |
| T11 | Wrap Isaac stages (split/motion/capture) | 2 | T08, T09 | done |
| T12 | Resource manager (VRAM/RAM + adaptive retry) | 3 | T09 | done |
| T13 | MCP server over HTTP (transport + auth) | 4 | T05 | todo |
| T14 | MCP tools & resources | 4 | T13, T09 | todo |
| T15 | UI (Streamlit over Layer 1 API) | 5 | T09 | todo |
| T16 | WSL2/Linux-distro bundling (deferred) | 6 | T08 | deferred |

## Dependency graph

```
T01 ─┬─ T02 ─┐
     ├─ T03 ─┼─ T05 ─┬─ T07 ─┬─ T09 ─┬─ T10
     ├─ T04 ─┘        │        │       ├─ T11
     └─ T06 ──────────┴────────┘       ├─ T12
                                        ├─ T14 (also needs T13)
                                        └─ T15
T05 ── T13 ── T14
T05, T06 ── T08 ── T09
T08 ── T16 (deferred, no downstream dependents)
```

## Critical path

T01 → T04/T03 → T05 → T07 → T08 → T09 → (then T11/T12/T14/T15 fan out).
T02 (config) and T06 (paths) are prerequisites for T07 and can be done in parallel with T03–T05.
T05 done (2026-07-13) — T13 (MCP server) is now unblocked too (only needs T05). T06 done
(2026-07-13) — both Phase-0 prerequisites for T07 are now met (T02, T05, T06 all done). T07 was
marked done (2026-07-13), **reopened 2026-07-14** (its stage bodies imported
`omniverse_pipeline`/`motion_seg` in-process via a `sys.path` hack instead of copying the verified
logic in, which the new "copy the logic in, don't call the original script" rule
(`INSTRUCTIONS.md`) disallows), and **redone / marked done again 2026-07-14**: `convert()`/
`segment_trajectories()`/`evaluate()`+helpers ported verbatim into `pipeline/vendored/host/`, the
three stage modules now import from there, no `sys.path`/`omniverse_pipeline`/`motion_seg`
reference remains in `pipeline/stages/` (`grep -r "sys.path" pipeline/stages/` empty) — see
`T07-wrap-cpu-stages.md` and `.claude_notes/NOTES_pipeline_orchestration.md`. Phase 0 / **M1
reached**. **T08 done (2026-07-14):** `pipeline/containers/` (`ContainerManager` + config, Docker
SDK only imported inside methods) — `ensure_image`/`start`/`exec`/`stop`/`list_containers`, warm
container reuse by deterministic name, T06's mounts + GPU passthrough + Isaac's persisted cache
volumes, all mirrored from the existing devcontainer defs; `pipeline.api`'s container-control
stubs now wired to it. 20 new tests (126 total) against a fake Docker client; real GPU/Isaac
behavior needs Bartosz's machine, with Docker Desktop + GPU support set up — see
`pipeline/containers/MANUAL_CHECKLIST.md`, `planning/WINDOWS_SETUP.md`, and
`T08-container-manager.md`. T09 (wrap CUDA stages, needs T07+T08, both done) is the critical
path's next stop.

**T08 GPU/Isaac verification passed for real (2026-07-15):** Bartosz ran
`tests/test_containers_gpu.py` (with `PIPELINE_TEST_ISAAC=1`) on his Windows + Docker Desktop + GPU
machine — all 6 passed in 1088s. Every `MANUAL_CHECKLIST.md` box is now checked; see
`T08-container-manager.md`'s "GPU/Isaac verified for real" note. T08 is fully closed, not just
unit-tested against a fake client.

**T09 done (2026-07-15):** `pipeline/vendored/cuda/{train,render,seg_extract,amp}.py` — verbatim
argparse-CLI ports of `train.py`/`render.py`/`motion_seg/extract_trajectories.py`/`render_amp.py`,
executed as a separate process *inside* the `cuda` container via `ctx.containers.exec_in_container`
(T08) rather than imported in-process (unlike T07's `host` stages — these scripts' real deps,
`torch`/`arguments`/`scene`/..., only exist in the container). New `pipeline/config/bridge.py`
writes a temp `arguments/multipleview/<name>.py`-style file from the resolved config each stage
call (the `ModelParams`/`PipelineParams`/`ModelHiddenParams`/`OptimizationParams` dict literals
`merge_hparams` reads); `pipeline.api._stage_config_for` gained a `"_bridge"` merge scoped to just
`train`/`render`/`seg_extract`/`amp` so cache-key scoping (T07's own fix) isn't defeated. Found and
fixed two more real pre-existing gaps, same pattern as T07's `ctx.inputs` fix: `run_dag` never
actually set `ctx.paths`/`ctx.containers` (T04 reserved the slot, T06/T08 never wired it) — now
always set to the `pipeline.paths`/`pipeline.containers` modules; `ContainerManager.exec` gained an
`environment=` kwarg to carry a `PYTHONPATH=/workspace` fix through to the container (a script's
own directory, not the exec `workdir`, is what Python puts on `sys.path[0]`). 33 new tests (146
total) verified in an isolated venv against a fake `exec_in_container` (no real GPU/Docker needed,
same story as T08's own unit tests) — CLI construction, bridge-file content, cache-key scoping, and
`train.default -> render.default` caching through a real `run_dag` call. Real GPU/container
execution (the acceptance criteria's actual end-to-end `run_pipeline` call + parity check) needs
Bartosz's machine, not yet run for real. Full log in `T09-wrap-cuda-stages.md` and
`.claude_notes/NOTES_pipeline_orchestration.md`. Next unblocked: T10 (Option-A segmentation,
`mbs_infer.py`) and T12 (resource manager) both need T09 (done); T11 (Isaac stages) needs T08+T09
(both done).

**T10 done (2026-07-15):** `segment.mbs` — a second impl behind the `segment` role
`segment.rigid` (T07) already occupies, the concrete "add a new idea = register an impl + a
preset" demonstration. `pipeline/vendored/cuda/mbs_infer.py` (verbatim port of
`motion_seg/mbs_infer.py`, T09's `cuda`-stage shape since MotNet needs the GPU, unlike
`segment.rigid`'s CPU-only `host` impl) + `pipeline/stages/segment_mbs.py` (`SegmentMbsStage`,
identical `inputs=("trajectories",)`/`outputs=("segmentation",)` contract as `segment.rigid`) +
new `pump01_segA.yaml` preset (`segment.impl: mbs`). 10 new tests (142 total, sandbox-verifiable
subset) confirm the config-switch contract itself: `_auto_stage_plan` resolves `segment.rigid` for
`base` and `segment.mbs` for `pump01_segA`, and both stage classes share the exact same
inputs/outputs tuple. Real GPU run (MBS `ext/` ops JIT-compiling, a downloaded checkpoint, actual
MotNet inference) needs Bartosz's machine — `planning/WINDOWS_SETUP.md`'s new "7. Option-A
segmentation (MBS) setup" step has the checkpoint-download instructions. Full log in
`T10-wrap-option-a-segmentation.md` and `.claude_notes/NOTES_pipeline_orchestration.md`. Next
unblocked: T11 (Isaac stages) and T12 (resource manager) both still only need T08/T09 (done); T10
had no downstream dependents to unblock.

**T11 done (2026-07-16):** `pipeline/vendored/isaac/{rig,split_mesh,add_motion,omni_capture}.py`
(verbatim ports) + `pipeline/stages/isaac_common.py` (shared CLI-exec plumbing targeting
`/isaac-sim/python.sh`) + three new stages — `prep_split.default`, `prep_motion.default`,
`capture.isaac` (all `environment="isaac"`; only `capture.isaac` needs the GPU). `capture.isaac`
produces `capture`, the artifact `convert.default` (T07) has declared as its external input since
Phase 0 — a preset's auto-planned DAG now runs `prep_split.default -> prep_motion.default ->
capture.isaac -> convert.default -> ... -> amp.default` end to end from a raw USD asset
(**Milestone M3 reached**). Renamed from `ARCHITECTURE.md`'s originally-sketched `prep.split`/
`prep.motion` to `prep_split.default`/`prep_motion.default`: the registry's `role.impl` split
would otherwise have collided both into one ambiguous `"prep"` role (two impls that aren't
actually alternatives of the same thing) — found while wiring this into a full auto-planned run
for the first time. `ArtifactKind` gained `"usd"` (single-file mesh outputs). `pipeline.api.
run_pipeline` gained an `external_artifacts` parameter — a second real gap, since a fresh
auto-planned run previously had no way to satisfy *any* external input (not just `prep_split`'s
new `raw_mesh`) before this task. 12 new tests (151 total), including a real
`prep_split.default -> prep_motion.default -> capture.isaac -> convert.default` chain through
`run_dag` (fake `exec_in_container`, real unmocked `convert.default` afterward) proving the
artifact hand-off, not just declaring it; found (documented, not fixed) a pre-existing T03/T05
cache-granularity gap this chain exposes for the first time: `capture`'s `dataset`-kind (directory)
artifact never gets a content hash, so `convert.default` stays cross-run-cached even when
`capture`'s actual content changes upstream. Real Isaac Sim/GPU run (the acceptance criteria's
`run_capture.sh` smoke test + a full `pump01` run from raw asset through amp) needs Bartosz's
machine, not yet run for real — `planning/WINDOWS_SETUP.md` gained an "8. Isaac prep/capture
stages setup" step (manual `trimesh` install in the `isaac` container; where the raw asset comes
from). Full log in `.claude_notes/NOTES_pipeline_orchestration.md`'s "T11 done" section; task
detail in `orchestrator/planning/tasks/T11-wrap-isaac-stages.md`. T11 had no downstream dependents
in the graph; T12 (resource manager) and T13 (MCP server) remain the two fully-unblocked `todo`
tasks.

**Runtime host revised 2026-07-14 (T06 + T08 both touched):** Bartosz asked to run and test the
whole thing directly from Windows for now, deferring WSL2/Docker "bundling" as its own later
feature (new `T16`, `deferred`, not scheduled). `pipeline/paths.py` (T06) dropped its three-space
(host/wsl/container) model down to two (host/container) — the runtime host is native Windows, so
there's no separate WSL2 execution environment with its own filesystem view anymore.
`pipeline/containers/manager.py` (T08)'s `ensure_image` build-context path source changed from
`repo_root_wsl` to `repo_root_host` accordingly; env vars renamed `PIPELINE_REPO_ROOT_WSL` →
`PIPELINE_REPO_ROOT`, `PIPELINE_ASSETS_ROOT_WSL` → `PIPELINE_ASSETS_ROOT`. Both tasks' own specs
and test suites were updated and re-verified green (see `T06-path-translation.md`'s and
`T08-container-manager.md`'s revision notes, and
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Runtime host moved off WSL2" entry). No task
that was already `done` got reopened — the acceptance bar each met (centralized path translation;
a working container manager with GPU passthrough) still holds under the new model.

**`capture.isaac` moved off the `isaac` Docker container onto native Windows execution, 2026-07-16
(same day as T11 done, real-hardware fixups still fresh).** Two real-hardware fixups earlier the
same day (Kit-bootstrap for `pxr`, cache-permission chmod) got `capture.isaac` past its first two
bugs, but a third real run still failed: Isaac Sim's RTX render products never produced a frame
(`IHydraTexture ... no GPU foundation`). Root cause, confirmed via NVIDIA's own developer forum:
**Vulkan isn't supported under WSL2**, which is what backs Docker Desktop's Linux containers on
Windows — a hard platform limitation, not a config gap. `capture.isaac` now execs `omni_capture.py`
as a native Windows subprocess against Bartosz's own real Isaac Sim install instead
(`pipeline.stages.isaac_common.run_native_isaac_script`, `PIPELINE_ISAAC_NATIVE_PYTHON` env var,
default `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`, corrected 2026-07-18
after the original default path turned out not to exist on Bartosz's machine) — a deliberate trade against
Bartosz's original preference for full container portability, made because there's no
Docker-side fix for an NVIDIA-confirmed platform gap. `prep_split.default`/`prep_motion.default`
are unaffected (CPU-only, no rendering) and keep using the `isaac` container exactly as before.
Also caught and fixed along the way: the stage's own "did this really succeed" check only verified
`cameras_gt.json` existed, which isn't written by rendering at all (pure USD geometry) — strengthened
to also check the camera-directory count matches `rig.n_cameras`. Full write-up:
`.claude_notes/NOTES_pipeline_orchestration.md`'s "adjust the project plan" entry;
`T11-wrap-isaac-stages.md`'s "Second real-hardware fixup" section. Docs updated: `ARCHITECTURE.md`,
`INSTRUCTIONS.md`, `WINDOWS_SETUP.md`. Sandbox suite green (155 passed, 9 skipped) — still needs a
real re-run on Bartosz's machine against his actual Isaac Sim install to confirm.

**T12 done (2026-07-19):** `pipeline/resources/{query,gating,adaptive,monitor,oom_retry}.py` —
VRAM (`pynvml`/`nvidia-smi`) + RAM (`psutil`) query, a pre-flight gate wired into
`pipeline.dag.scheduler.run_dag`'s per-stage loop (fails a too-large stage cleanly rather than
crashing), adaptive knobs (`low_vram_mode`/segmentation working-set/`rt_subframes`/
`opacity_thresh`) as pure headroom-to-value calculations, a `ResourceMonitor` filling T03's
nullable `peak_vram_mb`/`peak_ram_mb` manifest fields, and `run_with_oom_retry` (one reduced-memory
retry on an apparent CUDA OOM, recorded in a new `StageRecord.oom_fallback` field). `pipeline.api.
gpu_status()` now delegates to it; new `psutil>=5.9` dependency. 47 new tests (210 total
collected), 178 passed/skipped clean in an isolated venv — a pre-existing, unrelated sandbox
permission issue blocks `test_containers.py`'s 32 tests (a real leftover file from Bartosz's own
real-hardware runs, not a T12 regression — see `.claude_notes/NOTES_pipeline_orchestration.md`'s
"T12 done" entry for the full write-up, including a new `tests/conftest.py` autouse fixture needed
to stop this sandbox's own incidental RAM from wrongly gating T09/T10/T11's fake-exec tests). Real
VRAM/RAM gating, peak-mem accuracy, and OOM-retry's actual recovery all still need verification on
Bartosz's machine. Only T13 (MCP server) and T15 (UI) remain un-started, plus the deferred T16.

## Milestones

- **M1 (end of Phase 0): reached 2026-07-14.** CPU stages (`convert.default`/`segment.rigid`/
  `seg_eval.default`) run from `run_pipeline(preset=...)` with caching, no containers, calling
  only orchestrator-owned (`pipeline/vendored/host/`) code — T07 redone under the copy-in rule.
- **M2 (end of Phase 1):** reconstruction → segmentation → amplification runs end-to-end from one
  call using containers. Solves most of problem #1.
- **M3 (end of Phase 2): reached 2026-07-16.** True end-to-end from a raw USD asset through
  amplified render — `prep_split.default -> prep_motion.default -> capture.isaac ->
  convert.default -> train.default -> render.default -> seg_extract.default -> segment.* ->
  seg_eval.default -> amp.default`, all auto-planned from one preset. Real Isaac Sim/GPU execution
  of the newly-added prep/capture stages still needs verification on Bartosz's machine (T11's own
  status note), same as every other GPU-touching stage before its own real-hardware check.
- **M4 (end of Phase 4):** Claude can drive everything over HTTP MCP. Solves problem #2.
- **M5 (end of Phase 5):** UI for Bartosz. Solves the remainder of problem #1.
