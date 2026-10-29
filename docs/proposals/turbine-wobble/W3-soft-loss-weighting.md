# W3 (T51) — Continuous normalized soft loss weighting

**Status:** todo  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W1 (T49), W2 (T50)  
**Experiment:** E2 (Best of E1a/E1b + soft image-loss weights)  
**Gate before W4:** Static sequence yields weights $\approx 1.0$ everywhere; moving region yields smooth elevated weights with floor $\ge 1.0$ and unit mean normalization; per-camera overlays are clean of enclosure/flicker artifacts; no hard mask or ROI touches the deformation model.

---

## 1. Objective

Increase the photometric gradient budget around temporal motion evidence using **continuous, smooth, mean-normalized scalar loss weights** without altering any Gaussian's deformation representation or introducing binary masks.

## 2. Codebase Findings & Gaps to Address in W3

1. **Existing T36 motion-mask pipeline ([`scripts/precompute_motion_masks.py`](../../scripts/precompute_motion_masks.py), [`core/train.py`](../../core/train.py#L288-L327)):**
   - Previously, `precompute_motion_masks.py` defaulted to a narrow 6–9 Hz bandpass (missing the 5.691 Hz wobble and firing on the safety net enclosure), normalized by single-pixel max (`[0, 1]`), and [`core/train.py`](../../core/train.py#L303-L305) mapped it via `1.0 + (mask_max - 1.0) * cam_mask` with `motion_mask_max = 50.0` **without mean normalization**, changing the effective global learning rate across cameras and scenes.
2. **Design for W3 soft weight maps:**
   - Compute robust per-pixel temporal evidence independently per camera from the training frames (e.g. robust median-subtracted temporal deviation + broad 2–9 Hz in-band energy, suppressing slow global exposure drift via per-frame median luminance normalization and suppressing isolated JPEG/sensor noise via spatial Gaussian smoothing).
   - Map temporal evidence $e_k(x, y) \ge 0$ to unnormalized weights $\tilde{w}_k(x, y) \in [1, w_{\max}]$ with a conservative configurable cap $w_{\max} \in [10, 20]$ and floor $1.0$, then normalize per camera so $\frac{1}{HW}\sum_{x,y} w_k(x, y) = 1.0$ (or documented floor-preserving normalization with explicit mean target) so total L1 loss scale is preserved.
   - Keep weights strictly detached (`requires_grad=False`) and strictly confined to the image loss (`l1_loss` and optionally local `ssim_map` before spatial reduction).

## 3. Implementation Steps

1. Update/extend [`scripts/precompute_motion_masks.py`](../../scripts/precompute_motion_masks.py) (or add dedicated soft-weight generator mode) to produce continuous, spatially smoothed, drift-suppressed weight maps + diagnostic overlay PNGs and a provenance manifest (`weight_maps_meta.json` with SHA-256 hash, per-camera min/mean/p95/max stats).
2. Update weight application in [`core/train.py`](../../core/train.py), [`orchestrator/pipeline/vendored/cuda/train.py`](../../orchestrator/pipeline/vendored/cuda/train.py), and [`core/utils/loss_utils.py`](../../core/utils/loss_utils.py):
   - Enforce mean-weight normalization, configurable cap (`motion_mask_max`, default e.g. `15.0`), floor $\ge 1.0$, and verify `cam_name` attribution.
   - Record weight-map directory hash into the run manifest.
3. Create preset `E2` extending the winning `E1a`/`E1b` preset by enabling only soft loss weighting.

## 4. Verification & Tests

- **Static sequence identity test:** Constant frames (even with small global brightness offset) produce $w(x, y) = 1.0$ everywhere and weighted L1 equals unweighted L1 identically.
- **Synthetic localized mover test:** A small oscillating patch produces smooth elevated weights centered on the patch, zero NaN/Inf, strictly positive floor everywhere, cap respected, and mean $\approx 1.0$.
- **Disabled-mode no-op test:** `use_motion_mask=False` is bit-identical to W1.
- **Visual overlay inspection gate:** Generate and inspect per-camera overlays (`cam01`–`cam10`) before launching `E2`.

## 5. Deliverables & Stop Conditions

- **Deliverables:** Soft weight generator + provenance hash, normalized loss integration, unit tests (`orchestrator/tests/test_w3_soft_loss_weights.py`), per-camera visual overlays, and `E2` preset.
- **Stop conditions:** Reject weight maps dominated by exposure flicker, net/enclosure moiré, or compression noise. Do not tune the cap on the held-out test metrics of the target scene.

