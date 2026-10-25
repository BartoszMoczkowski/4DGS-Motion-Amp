# T36 — Motion-mask weighted loss (R2)

**Status:** done
**Phase:** 12 / Phase 1 (config & loss quick wins)
**Depends on:** —
**Remedy:** R2

## Goal

Weight the photometric loss toward the moving region so the ≤3 px turbine
signal is not drowned by static background (DyNeRF temporal-variance
importance weighting, ×20–50).

## Change points

- Loss site: `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py` (L1 loss weighted by mask).
- Loss utility: `core/utils/loss_utils.py` (`l1_loss(network_output, gt, mask=None)`).
- Dataset plumbing:
  - `core/scene/multipleview_dataset.py`: loads per-camera masks from `<scene>/motion_masks/` (`camXX.npy` or `.png`), provides `get_mask(index)`.
  - `core/scene/dataset.py`: wires mask retrieval in `FourDGSdataset.__getitem__` to `Camera(..., mask=mask)`.
- Configuration:
  - `core/arguments/__init__.py`: added `use_motion_mask`, `motion_mask_dir`, `motion_mask_max` to `OptimizationParams`.
  - `orchestrator/pipeline/config/models.py`: added fields to `OptimizationParams(StrictModel)`.
  - Presets: added `lego_turbine_2_missing_r2.yaml`, updated `lego_turbine_2_missing.yaml`, `lego_turbine_2_missing_r1.yaml` (R1-only), `lego_turbine_2_missing_stock.yaml`.
- Precompute script:
  - `scripts/precompute_motion_masks.py`: offline precomputation using bandpassed frame differences in 6.0–9.0 Hz (Parseval energy relation via `torch.fft.rfft`), rig-edge gating, noise-floor thresholding, morphological dilation (kernel 31 px, 2 iters), and Gaussian boundary smoothing. Also implements a soft ROI rectangle fallback mode.

## Deliverables & Artifacts

1. `scripts/precompute_motion_masks.py` CLI script supporting single scene, custom dir, or `--all-turbines`.
2. Precomputed motion masks for all 4 turbine scenes under `data/multipleview/<scene>/motion_masks/`:
   - `camXX.npy` (float32 array normalized to [0.0, 1.0])
   - `camXX_vis.png` (8-bit grayscale visualization)
   - `camXX_heatmap.png` (Turbo color-mapped visualization for visual inspection)
   - `manifest.json` (metadata, frequencies, statistics per camera view)
3. Unit test suite: `orchestrator/tests/test_motion_mask_loss.py` (10 passing tests verifying static loss reduction, weight scaling, gradient propagation, broadcasting, dataloader wiring, and preset configs).

## Verification

- **Static synthetic input:** `l1_loss(pred, gt, mask=None)` and `l1_loss(pred, gt, mask=torch.ones(...))` produce bit-identical loss to stock `l1_loss(pred, gt)` (tolerance 1e-7).
- **Weight scaling:** 50x weight in moving regions yields exact proportional loss and 50x gradient magnitude on target pixels.
- **Visual inspection:** inspected heatmaps across all 10 cameras; rotor and missing-blade wobble structure are brightly isolated with zero background clutter.
