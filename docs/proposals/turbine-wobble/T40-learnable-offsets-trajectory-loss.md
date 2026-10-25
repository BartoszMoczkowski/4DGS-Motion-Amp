# T40 — Learnable offsets + trajectory reprojection loss (R7+R8)

**Status:** todo
**Phase:** 12 / Phase 2 (the timestamp fix)
**Depends on:** T39, T36
**Remedies:** R7, R8

## Goal

Refine the Phase-0 offset estimates jointly with the model, disambiguated by
the already-measured 2D wobble tracks.

## R7 — Learnable offset refinement (guarded)

- δₖ as optimizable parameters, initialized from `offsets.json` (or zero),
  registered in a dedicated optimizer group in `core/train.py` at 0.1–0.5× the
  backbone LR; frozen for the first ~10 k iterations (Sync-NeRF schedule).
- **Guardrails are mandatory:** cam01 anchored at δ = 0; δₖ clamped to
  ±40 ms; R8 loss active. Unguarded offset optimization on a weak-signal scene
  is the one configuration the literature consistently shows failing (ASTRA).

## R8 — Trajectory reprojection loss

- The diagnostic pipeline already tracked 55 points on the turbine at
  sub-pixel coherence.
- Project deformed ROI Gaussian positions into each camera at its corrected
  timestamp; L2-align projected 2D **displacement** against the measured
  tracks. Supervises amplitude, phase, axis, and the per-camera offsets
  simultaneously.
- Supervise displacement of identified points, **never dense optical flow**
  (rendered-Gaussian-flow supervision has a known domain gap — DispFlow-GS).

## Verification

- Offsets stay within clamp; final δ̂ within ~2 ms of Phase-0 values
  (validation metric 4).
- Enabling R8 improves metric 3 (cross-view phase coherence).
- Second retrain (R6+R7+R8): full metric set; PSNR/SSIM guardrail must not
  regress vs the static-collapse baseline.

## Deliverable

Optimizer-group change + trajectory-loss module + the R6+R7+R8 retrain with
metrics logged (MLflow + manifest).
