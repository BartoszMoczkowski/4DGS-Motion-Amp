# T12 — Resource manager (VRAM/RAM + adaptive retry)

- Status: todo
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

## Acceptance criteria (Bartosz's WSL2 machine)
- Forcing a too-large stage triggers the reduced-memory fallback and completes (or fails cleanly
  with a clear message) rather than crashing the run.
- Manifest shows measured peak memory and any fallback applied.

## Relevant existing files
`ampUI.py` / `render_amp.py` (`low_vram_mode`, `torch.cuda.memory._record_memory_history`),
`motion_seg/mbs_infer.py` (working-set default 4000), `capture_config_pump.yaml` (`rt_subframes`).

## Notes / gotchas
Estimates start rough; refine from observed peak-mem logged in the manifest over real runs. Keep
the gating hook the one T05 already exposed.
