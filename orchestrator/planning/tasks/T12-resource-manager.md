# T12 — Resource manager (VRAM/RAM + adaptive retry)

- Status: done (2026-07-19) — real-hardware verification of gating/OOM-retry/peak-mem still
  pending on Bartosz's machine, see the "Done (2026-07-19)" section below.
- Phase: 3
- Depends on: T09
- Environment: host

## Goal
Make the scheduler resource-aware so runs don't OOM and adapt workload to available memory.

## In scope
- Query total+free VRAM (pynvml/`nvidia-smi`) and system RAM.
- Gate the serial scheduler: never start a stage whose `vram_gb`/`ram_gb` estimate exceeds free.
- Adaptive knobs from measured headroom: `low_vram_mode` (amp/render_amp), segmentation
  working-set / subsample (`mbs_infer.py`), `rt_subframes` (capture), opacity thresholds.
- OOM-retry: on CUDA OOM, retry the stage once with reduced-memory settings before failing;
  record the fallback in the manifest.
- Fill the peak VRAM/RAM fields left nullable in T03.

## Out of scope
Multi-GPU / parallel scheduling (single-GPU serial by design).

## Deliverables
`pipeline/resources/` module + scheduler integration + estimates per stage in config.

## Acceptance criteria (Bartosz's Windows + Docker Desktop machine)
- Forcing a too-large stage triggers the reduced-memory fallback and completes (or fails cleanly
  with a clear message) rather than crashing the run.
- Manifest shows measured peak memory and any fallback applied.

## Relevant existing files
`ampUI.py` / `render_amp.py` (`low_vram_mode`, `torch.cuda.memory._record_memory_history`),
`motion_seg/mbs_infer.py` (working-set default 4000), `capture_config_pump.yaml` (`rt_subframes`).

## Notes / gotchas
Estimates start rough; refine from observed peak-mem logged in the manifest over real runs. Keep
the gating hook the one T05 already exposed.

## Done (2026-07-19)

`pipeline/resources/{query,gating,adaptive,monitor,oom_retry}.py` — see
`.claude_notes/NOTES_pipeline_orchestration.md`'s "T12 done" entry for the full write-up. Summary:

- `query.py`: VRAM via `pynvml` then `nvidia-smi` CLI fallback; RAM via `psutil`'s `available`
  figure. All three lazily imported inside functions (never at module scope).
- `gating.py`: `check_headroom`/`InsufficientResourcesError`, wired into
  `pipeline.dag.scheduler.run_dag`'s per-stage loop right before a real (non-cached) stage runs.
  Fails open whenever a dimension can't be measured.
- `adaptive.py`: `should_use_low_vram_mode`/`scaled_working_set`/`scaled_rt_subframes`/
  `scaled_opacity_thresh` — pure linear-ramp calculations from measured headroom, covering every
  knob this task's own "In scope" bullet named. Not yet auto-applied per-stage outside the
  OOM-retry path (see `oom_retry.reduced_memory_config`) — available for a future caller (T13/T15)
  to resolve into a stage's config ahead of a run, once there's real headroom data to tune the
  ramps against.
- `monitor.py`: `ResourceMonitor` — background-thread peak-VRAM/RAM sampler across one stage's
  execution, filling `StageRecord.peak_vram_mb`/`peak_ram_mb` (T03's nullable placeholders).
- `oom_retry.py`: `run_with_oom_retry` — detects an apparent CUDA OOM by scanning the failing
  stage's own captured log (`CudaStageError`/`IsaacStageError` gained a real `log_path` attribute
  for this), retries once with a stage-specific reduced-memory fallback (`amp.*` forces
  `low_vram_mode`, `segment.mbs` halves its working-set/subsample, `capture.isaac` halves
  `rt_subframes`) if one exists; `train`/`render`/`seg_extract` have no known safe knob yet and
  re-raise immediately on a real OOM.

`StageRecord` gained `oom_fallback: Optional[dict]` (backward-compatible); `record_stage_result`
gained matching `peak_vram_mb`/`peak_ram_mb`/`oom_fallback` kwargs. `pipeline.api.gpu_status()` now
delegates to `pipeline.resources.gpu_status()`. New dependency: `psutil>=5.9`.

47 new tests (210 total collected), 178 passed/skipped clean (169 passed, 9 pre-existing
real-GPU-gated skips) in an isolated venv — `tests/test_containers.py`'s 32 tests are blocked by a
pre-existing, unrelated sandbox permission issue on a real leftover file from Bartosz's own
real-hardware runs (see `.claude_notes/NOTES_pipeline_orchestration.md`, not a T12 regression). New
`tests/conftest.py` autouse fixture forces `pipeline.resources.query`'s functions to return `None`
by default in every test — this sandbox's own incidental ~4GB RAM has no bearing on what a stage
needs on Bartosz's real machine, and would otherwise wrongly gate T09/T10/T11's fake-exec
integration tests.

**Acceptance criteria (both) verified only against fakes so far** — a too-large stage cleanly
fails via `run_dag` (`tests/test_dag.py::test_gating_fails_a_too_large_stage_cleanly_via_run_dag`),
and peak-mem + a successful OOM-retry fallback both land in the manifest
(`test_peak_mem_and_oom_fallback_recorded_via_run_dag`) — but real VRAM gating, real peak-mem
accuracy, and a real CUDA OOM's actual reduced-memory recovery all still need verification on
Bartosz's Windows + Docker Desktop + GPU machine, same as every other GPU-touching task before its
own real-hardware check.
