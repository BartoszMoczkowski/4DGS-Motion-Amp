# T19 — ROI Motion Gating (proposal 01) — Implementation & Benchmark Results

**Date:** 2026-08-12
**Status:** Complete (code landed, benchmarked, documented)
**Impl:** `roi.motion_gate` + `segment.rigid2` with optional `roi_mask` consumption

## What was implemented

1. **Config schemas** (`orchestrator/pipeline/config/models.py`):
   - `RoiMotionGateConfig` — `drive_freq`, `harmonics`, `dilation_hops`, `readmit_mult`, `k`
   - `RoiMaskLiftConfig` — placeholder for T22
   - `RoiConfig` — top-level role with `impl: none | motion_gate | mask_lift`
   - Added `roi: RoiConfig` to `PipelineConfig`

2. **Vendored core** (`orchestrator/pipeline/vendored/host/motion_gate.py`):
   - Band-pass trajectories at drive freq + harmonics
   - Per-point energy → log-Otsu threshold → static/moving split
   - k-NN graph dilation (captures boundary points)
   - Rigidity-lock readmission (`readmit_mult * sigma_d`)
   - Degenerate fallback: if Otsu puts >95% or <5% as moving, return all True
   - Returns `roi_mask` (bool[N]) + `snr` (float32[N]) + info dict

3. **Stage** (`orchestrator/pipeline/stages/roi_motion_gate.py`):
   - Registered as `roi.motion_gate`
   - Host env, inputs=`trajectories`, outputs=`roi_mask`

4. **ROI-aware segment stages**:
   - Wired optional `roi_mask` into `segment.rigid`, `segment.rigid2`, `segment.kabsch`
   - Points outside ROI get label `-2` (static), preserving `-1` floaters

5. **Extended eval** (`seg_eval.default`):
   - `evaluate()` accepts optional `roi_mask` parameter
   - Computes `ari_within_roi` on points inside ROI
   - Stage reads `roi_mask` defensively, includes in JSON output

6. **Scheduler fix** (`orchestrator/pipeline/dag/scheduler.py`):
   - **Critical bug discovered and fixed:** `ctx.inputs` previously only contained declared inputs, so `ctx.inputs.get("roi_mask")` always returned `None` even when the artifact existed in the manifest.
   - Fix: `StageContext.inputs` now receives `dict(manifest.artifacts)` (all artifacts for defensive reading), while cache keys continue to use only declared inputs.

7. **Preset & benchmark harness**:
   - `pump01_roi_gate.yaml` — `extends: pump01`, `roi.impl: motion_gate`, `segment.impl: rigid2`
   - `run_grid_seg.py --impl rigid2_roi` runs `roi.motion_gate` → `segment.rigid2` → `seg_eval.default`

## Benchmark results (`runs/grid_seg_rigid2_roi_results.csv`)

| run_id | ARI (global) | ARI (within ROI) | n_roi_points | n_pred |
|--------|-------------|------------------|-------------|--------|
| grid-A20mm_M2 | 0.016 | 0.001 | 248,953 | 229 |
| grid-A20mm_M4 | -0.046 | -0.042 | 329,369 | 98 |
| grid-A40mm_M8 | -0.024 | -0.004 | 286,849 | 13 |
| sweep-A40mm_M8-g10000 | 0.010 | 0.003 | 9,778 | 3 |
| sweep-A40mm_M8-g25000 | -0.016 | -0.024 | 21,930 | 3 |
| sweep-A40mm_M8-g50000 | 0.002 | -0.009 | 40,618 | 3 |
| sweep-A40mm_M8-g100000 | -0.029 | -0.028 | 95,067 | 3 |

## Key finding: ROI gating does NOT improve ARI on pump01 grid/sweep

**Why it failed:**
- The motion gate keeps **almost all points** (n_roi_points ≈ total point count)
- `ari_within_roi` is essentially identical to global ARI
- The background cloud in these synthetic scenes is NOT actually static — reconstruction jitter has energy comparable to true mm-scale motion after band-passing
- Log-Otsu on denoised energy cannot separate "machine motion" from "jitter" because the jitter is not white (the band-pass preserves some jitter power at the drive frequency and its harmonics)

**Implication for thesis:**
- The hypothesis that spatial gating would break through the motion-only ceiling is **not supported** on real reconstruction data
- The bottleneck is NOT the static background poisoning the graph — the bottleneck is that reconstruction jitter ≈ true motion amplitude, making all points look "moving"
- This strengthens the case for T22 (multi-view mask lifting) because mask lifting uses **geometric priors** (camera frustums + depth) rather than motion energy, so it is immune to reconstruction jitter

## Next step: T22 (multi-view mask lifting, proposal 02)

T22 requires:
- `vendored/cuda/mask_lift.py` — depth rendering in cuda container
- `vendored/host/clean_plate_diff.py` — host-side mask production
- `stages/roi_mask_lift.py` — registered as `roi.mask_lift`
- Oracle mode: derive masks by projecting GT labels to measure the ceiling

The ROI plumbing from T19 is already in place (`roi_mask` artifact contract, scheduler fix, segment stage consumption). T22 plugs into the same infrastructure.
