# T36 — Motion-mask weighted loss (R2)

**Status:** todo
**Phase:** 12 / Phase 1 (config & loss quick wins)
**Depends on:** —
**Remedy:** R2

## Goal

Weight the photometric loss toward the moving region so the ≤3 px turbine
signal is not drowned by static background (DyNeRF temporal-variance
importance weighting, ×20–50).

## Change points

- Loss site: `core/train.py:272-288` (L1 + DSSIM, currently uniform).
- Plumbing half-exists: the dataloader `mask` field
  (`core/scene/dataset.py:28,38-40`) is unused on the multipleview path.

## Design

1. Offline precompute per camera: per-pixel temporal weight map from
   **band-passed frame differences around 6–9 Hz** (rejects slow auto-exposure
   drift and lighting flicker that raw variance maps would amplify), dilated,
   clamped to [1, 50].
2. Multiply into the L1 term.
3. Fallback (equally acceptable for this scene): hard ROI rectangle around the
   turbine with soft falloff — trivially debuggable; the fancy map earns its
   complexity only if other scenes have motion outside the housing.

## Verification

- Weight-map visualizations inspected per scene (artifact per scene).
- Unit test: on a static synthetic input the masked loss reduces to stock.

## Deliverable

Mask precompute script + loss change + preset flag; exercised in the Phase-1
baseline retrain.
