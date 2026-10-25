# T37 — Coarse-stage gating (R10)

**Status:** done
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
- Stage switch: `core/train.py:550-575` / `orchestrator/pipeline/vendored/cuda/train.py:570-595`;
  `coarse_iterations = 3000` in `core/arguments/multipleview/lego_turbine_*.py`.

## Implementation Summary

1. **Option A (Phase 1 Baseline & Presets):**
   - Shortened coarse static stage to `coarse_iterations: 500` across all turbine argument configs:
     - `core/arguments/multipleview/lego_turbine_base.py`
     - `core/arguments/multipleview/lego_turbine_2_missing.py`
     - `core/arguments/multipleview/lego_turbine_5_missing.py`
     - `core/arguments/multipleview/turbine_r1_missing_6.py`
   - Added `coarse_iterations: 500` to orchestrator YAML presets (`lego_turbine_2_missing.yaml`, `lego_turbine_base.yaml`, `lego_turbine_5_missing.yaml`, `turbine_r1_missing_6.yaml`).
   - Created single-factor ladder step preset `lego_turbine_2_missing_r10.yaml` (`coarse_iterations: 500`, combining R1 + R2 + R10).
   - Preserved `coarse_iterations: 3000` on earlier ladder steps (`lego_turbine_2_missing_stock.yaml`, `lego_turbine_2_missing_r1.yaml`, `lego_turbine_2_missing_r2.yaml`) and `base.yaml`.
   - Updated `train.py` (both reference and vendored CUDA copies) to cleanly skip coarse stage when `coarse_iterations <= 0`.

2. **Option B (Render-Level Gated Coarse):**
   - Added `gated_coarse: bool = False` and `roi_bbox: Optional[list[float]] = None` to `OptimizationParams` and `PipelineParams` in both `core/arguments/__init__.py` and `orchestrator/pipeline/config/models.py`.
   - Propagated `gated_coarse` and `roi_bbox` from `opt` to `pipe` in `train.py`.
   - Updated `core/gaussian_renderer/__init__.py:80-135`:
     - When `stage == "coarse"` and `pipe.gated_coarse` is enabled:
       - If `pc._roi_mask` or `pipe.roi_bbox` is provided, only ROI Gaussians query `pc._deformation`; background points stay static.
       - If no ROI is specified, all Gaussians query `pc._deformation` from iteration 0.
     - When `stage == "coarse"` and `pipe.gated_coarse` is disabled (default): static Gaussians are used (upstream behavior).
     - When `stage == "fine"`: full deformation evaluated normally.

## Verification

- Automated test suite in `orchestrator/tests/test_coarse_gating.py` (10 passed):
  - `test_preset_ladder_coarse_iterations`: verifies ladder steps `stock` (3000), `r1` (3000), `r2` (3000), `r10` (500).
  - `test_real_turbine_presets_coarse_iterations`: verifies all 4 physical turbine presets resolve with `coarse_iterations == 500`.
  - `test_base_preset_defaults`: verifies `base` preset retains upstream 3000 coarse iterations.
  - `test_models_coarse_gating_fields`: verifies Pydantic validation for `coarse_iterations`, `gated_coarse`, `roi_bbox`.
  - `test_bridge_source_rendering`: verifies bridge config serialization.
  - `test_render_standard_coarse_stage`: verifies standard coarse skips deformation.
  - `test_render_gated_coarse_all_points`: verifies gated coarse queries deformation for all points when unmasked.
  - `test_render_gated_coarse_with_roi_bbox`: verifies gated coarse only deforms points within 3D bounding box.
  - `test_render_gated_coarse_with_roi_mask`: verifies gated coarse only deforms points matching `_roi_mask`.
  - `test_render_fine_stage_always_deforms`: verifies fine stage always evaluates deformation.
- Full regression suite: 501 passed, 9 skipped across `orchestrator/tests/`.
