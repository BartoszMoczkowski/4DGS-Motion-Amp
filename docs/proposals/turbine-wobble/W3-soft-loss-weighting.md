# W3 (T51) — Continuous normalized soft loss weighting

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W1 (T49), W2 (T50)  
**Experiment:** E2 (Best of E1a/E1b + soft image-loss weights)  
**Gate before W4:** Static sequence yields weights $\approx 1.0$ everywhere; moving region yields smooth elevated weights with floor $\ge 1.0$ and unit mean normalization; per-camera overlays are clean of enclosure/flicker artifacts; no hard mask or ROI touches the deformation model.

---

## 1. Objective

Increase the photometric gradient budget around temporal motion evidence using **continuous, smooth, mean-normalized scalar loss weights** without altering any Gaussian's deformation representation or introducing binary masks.

## 2. Codebase Findings & Gaps Addressed in W3

1. **Existing T36 motion-mask pipeline ([`scripts/precompute_motion_masks.py`](../../scripts/precompute_motion_masks.py), [`core/train.py`](../../core/train.py#L288-L327)):**
   - Previously, `precompute_motion_masks.py` defaulted to a narrow 6–9 Hz bandpass (missing the 5.691 Hz wobble and firing on the safety net enclosure), normalized by single-pixel max (`[0, 1]`), and [`core/train.py`](../../core/train.py#L303-L305) mapped it via `1.0 + (mask_max - 1.0) * cam_mask` with `motion_mask_max = 50.0` **without mean normalization**, changing the effective global learning rate across cameras and scenes.
2. **Completed W3 Design for Soft Weight Maps:**
   - **Exposure drift & flicker suppression:** Pre-normalizes per-frame median luminance $\mu_t = \operatorname{median}(I_t)$ prior to computing temporal bandpass differences, completely eliminating global auto-exposure fluctuations.
   - **Fundamental-centered bandpass:** Re-centered to 3.5–7.5 Hz (encompassing the 5.691 Hz fundamental wobble).
   - **Robust percentile normalization:** Normalizes active signal by the 99th percentile $p_{99}$ rather than single-pixel maximum.
   - **Continuous spatial smoothing:** Applies Gaussian spatial smoothing ($\sigma = 15.0$) over the $\pm 3\text{ px}$ wobble displacement envelope.
   - **Unit mean normalization:** Enforces $\mathbb{E}[w(x, y)] = 1.0$ per-camera via centralized helper [`normalize_loss_weights`](../../core/utils/loss_utils.py#L20-L46) with configurable cap $w_{\max} = 15.0$ and floor $\ge 1.0$, preserving the unweighted L1 loss scale identically for uniform errors.
   - **Detached autograd:** Weights are strictly detached (`requires_grad=False`).

## 3. Implementation Summary

1. **Centralized Loss Normalizer ([`core/utils/loss_utils.py`](../../core/utils/loss_utils.py)):**
   - Added `normalize_loss_weights(weights, mask_max=15.0, floor=1.0, normalize_mean=True)` ensuring floor $\ge 1.0$, cap $w_{\max}$, and spatial mean normalization across 2D/3D/4D tensors.
2. **Synchronized Training Integration ([`core/train.py`](../../core/train.py), [`orchestrator/pipeline/vendored/cuda/train.py`](../../orchestrator/pipeline/vendored/cuda/train.py)):**
   - Replaced unnormalized multiplication with `normalize_loss_weights` before assembling `mask_tensor`.
   - Updated `motion_mask_max` default from 50.0 to 15.0 and added `normalize_motion_mask = True` across both entry points and config models.
3. **Run Manifest & Provenance Tracking ([`orchestrator/pipeline/stages/train.py`](../../orchestrator/pipeline/stages/train.py)):**
   - `train.default` extracts `weight_maps_hash` and `weight_maps_meta_sha256` from `motion_masks/weight_maps_meta.json` into model artifact metadata.
4. **Updated Mask Generator ([`scripts/precompute_motion_masks.py`](../../scripts/precompute_motion_masks.py)):**
   - Generates `<cam_name>_overlay.png` (blended color heatmap over first grayscale frame for visual inspection).
   - Writes `weight_maps_meta.json` manifest with directory-wide SHA-256 hash and per-camera statistics (`min`, `mean`, `p50`, `p95`, `p99`, `max`, `fraction_active`).
5. **Declarative E2 Presets:**
   - Created [`orchestrator/pipeline/config/presets/turbine_r1_missing_6_w3_e2.yaml`](../../orchestrator/pipeline/config/presets/turbine_r1_missing_6_w3_e2.yaml) and [`lego_turbine_2_missing_w3_e2.yaml`](../../orchestrator/pipeline/config/presets/lego_turbine_2_missing_w3_e2.yaml) extending E1a solely by enabling `use_motion_mask: true`, `normalize_motion_mask: true`, `motion_mask_max: 15.0`.

## 4. Verification & Acceptance Tests

Unit test suite in [`orchestrator/tests/test_w3_soft_loss_weights.py`](../../orchestrator/tests/test_w3_soft_loss_weights.py) verifies:
- **Static sequence identity test:** Constant frames produce $w(x, y) = 1.0$ everywhere, and weighted L1 loss equals unweighted L1 loss identically.
- **Scale preservation:** Weighted L1 with mean-normalized weights preserves unweighted loss scale for uniform error ($0.15 \to 0.15$).
- **Exposure flicker suppression:** Injected global frame-to-frame luminance offsets do not trigger spurious motion weights.
- **Synthetic localized mover test:** Oscillating blob produces smooth elevated weights centered on the mover, strictly positive floor, and $\mathbb{E}[w] = 1.000000$.
- **Single-factor E2 preset validation:** E2 presets validate cleanly via `validate_config` and differ from E1a only in soft loss weighting parameters, preserving all timing settings and leaving all hard ROI/Fourier gates disabled.
- **Provenance tracking:** `precompute_scene_motion_masks` writes `weight_maps_meta.json` with per-camera and directory hashes.

