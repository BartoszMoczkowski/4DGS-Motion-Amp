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
| T11 | Wrap Isaac stages (split/motion/capture) | 2 | T08, T09 | todo |
| T12 | Resource manager (VRAM/RAM + adaptive retry) | 3 | T09 | todo |
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

## Milestones

- **M1 (end of Phase 0): reached 2026-07-14.** CPU stages (`convert.default`/`segment.rigid`/
  `seg_eval.default`) run from `run_pipeline(preset=...)` with caching, no containers, calling
  only orchestrator-owned (`pipeline/vendored/host/`) code — T07 redone under the copy-in rule.
- **M2 (end of Phase 1):** reconstruction → segmentation → amplification runs end-to-end from one
  call using containers. Solves most of problem #1.
- **M3 (end of Phase 2):** true end-to-end from USD asset through amplified render.
- **M4 (end of Phase 4):** Claude can drive everything over HTTP MCP. Solves problem #2.
- **M5 (end of Phase 5):** UI for Bartosz. Solves the remainder of problem #1.
