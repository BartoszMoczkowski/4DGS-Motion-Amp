# T37 — Coarse-stage gating (R10)

**Status:** todo
**Phase:** 12 / Phase 1 (config & loss quick wins)
**Depends on:** —
**Remedy:** R10

## Goal

Stop the 3 000-iteration static coarse stage from baking the motion-blurred
time average into canonical space — including the densified Gaussian
population (densification runs during coarse, `core/train.py:392-394`).

## Change points

- Coarse renders skip deformation entirely:
  `core/gaussian_renderer/__init__.py:80-81`.
- Stage switch: `core/train.py:505-507`; `coarse_iterations = 3000` in
  `core/arguments/multipleview/lego_turbine_base.py:26`.

## Variants

- **Option A (ship in Phase 1):** shorten/skip coarse
  (`coarse_iterations` 3 000 → 0–500) via preset. With T39's timestamps fixed
  there is no averaging artifact to warm-start away from.
- **Option B (with Phase 3):** gate the ROI out of coarse — ROI Gaussians
  query the deformation field from iteration 0 so the canonical turbine pose
  anchors at a reference instant, not the temporal mean.
- Option C (most principled, deferred): initialize canonical space from one
  time-consistent multi-view instant (frame whose corrected timestamps cluster
  tightest across cameras).

## Verification

- Coarse-output PLY is no longer the blurred average (visual diff on the
  turbine ROI).
- Fine-stage convergence unchanged on static background (PSNR guardrail).

## Deliverable

Preset change (A) + gated-stage implementation (B) behind a config flag.
