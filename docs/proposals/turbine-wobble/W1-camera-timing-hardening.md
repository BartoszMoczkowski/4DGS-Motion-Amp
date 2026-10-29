# W1 (T49) — Harden fixed and guarded learnable per-camera timing

**Status:** todo  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W0 (T48)  
**Experiments:** E1a (fixed sidecar timestamps) vs E1b (guarded learnable offsets)  
**Gate before W2/W3:** Loader/optimizer unit tests pass; training and evaluation timestamps match identically; fixed-vs-learnable preset pair differs *only* in offset optimization settings.

---

## 1. Objective

Make the fixed-versus-learnable timing comparison (`E1a` vs `E1b`) mathematically clean and self-consistent under native HexPlane 4DGS, with zero motion masks, zero rigid heads, and zero trajectory losses.

## 2. Codebase Findings & Gaps to Address in W1

1. **Silent fallback in [`core/scene/multipleview_dataset.py`](../../core/scene/multipleview_dataset.py#L58-L110):**
   - Currently, `_load_timing_and_offsets` silently returns with `has_offsets = False` if `offsets.json` is missing or malformed, or falls back silently to `optical_offsets.json` (which has known 21–40 ms errors on real scenes).
   - **Fix required:** When `use_camera_offsets=True`, require a valid timing file (`offsets.json` or explicitly configured `offsets_file`), validate camera coverage against `cam_extrinsics`, finite offset/drift values, positive `fps` and `exposure_s`, frame indexing, and record SHA-256 hash. Fail loudly with an actionable error on missing/malformed data; never fall back silently to `optical_offsets.json`.
2. **Single time convention across train, eval, render, and export:**
   - Document explicitly whether physical timestamps $t_{\text{phys}}(k, i)$ represent start-of-exposure or mid-exposure (if mid-exposure, apply `exposure_s / 2` in one central helper only) and centralize conversion between $t_{\text{phys}}$ and normalized model time $t_{\text{norm}} \in [0, 1)$.
   - Currently, during `learnable_camera_offsets=True` in [`core/train.py`](../../core/train.py#L254-L259), `custom_time` is passed to `render()` in the training step, **but** `training_report()` (held-out evaluation), `render.py`, and `seg_extract.py` still use static `viewpoint_camera.time`!
   - **Fix required:** Attach `camera_offsets_module` (or load `learned_offsets.json`) so evaluation (`training_report`) and offline rendering evaluate each camera at the exact learned timestamps.
3. **Structural anchoring and quadratic prior in [`core/scene/camera_offsets.py`](../../core/scene/camera_offsets.py):**
   - Currently, `self.delta_offsets` is an `nn.Parameter` of size $N_{\text{cams}}$ that relies on post-step `clamp_and_anchor()` to zero `anchor_idx`, and has no prior toward the sidecar initialization (`base_offsets`).
   - **Fix required:**
     - Anchor `cam01` (`reference_camera`) **by construction** in the forward graph (e.g. multiplying `delta_offsets` by a fixed buffer mask with `0.0` at `anchor_idx`, or parameterizing only non-anchor cameras) so `cam01` receives identically zero gradient and never moves.
     - Add a quadratic prior toward the sidecar initialization:
       $$\mathcal{L}_{\text{offset\_prior}} = \lambda_{\text{offset\_prior}} \sum_{k \neq \text{anchor}} \left(\frac{\delta_k}{\sigma_{\text{offset}}}\right)^2$$
       with explicit configurable `offset_prior_sigma_s` (in seconds) and weight, making `max_offset_clamp_s` a secondary safety bound rather than the primary regularizer.
     - Keep offsets frozen for `learnable_offsets_freeze_iter` iterations while coarse/initial fine geometry stabilizes.
     - Log initial, current, and $\Delta$ offsets plus prior loss at evaluation intervals and into `learned_offsets.json` / run manifest.

## 3. Implementation Steps

1. Centralize physical-to-normalized timestamp conversion in `core/scene/camera_offsets.py` / `core/scene/multipleview_dataset.py` and enforce strict validation when `use_camera_offsets=True`.
2. Upgrade `CameraOffsets` in [`core/scene/camera_offsets.py`](../../core/scene/camera_offsets.py) with structural anchor masking, `compute_prior_loss(sigma_s)`, and full diagnostic dictionary export (`init_offsets_s`, `current_offsets_s`, `delta_offsets_s`, `prior_loss`, `offsets_file_sha256`).
3. Wire prior loss and evaluation-time `custom_time` into both [`core/train.py`](../../core/train.py) and [`orchestrator/pipeline/vendored/cuda/train.py`](../../orchestrator/pipeline/vendored/cuda/train.py), and load `learned_offsets.json` in `Scene.__init__` when present so offline render/evaluation uses the refined camera timestamps.
4. Create the matched preset pair (`E1a` fixed timing vs `E1b` learnable timing) differing **only** in learnable-offset parameters.

## 4. Verification & Tests

- **Synthetic loader test:** Two cameras with known offsets/drift produce exact expected $t_{\text{phys}}$ and $t_{\text{norm}}$; malformed/missing `offsets.json` raises `ValueError`/`FileNotFoundError` when `use_camera_offsets=True`.
- **Round-trip timestamp parity test:** Training, `training_report` evaluation, and offline `Scene` reload produce identical per-camera timestamps.
- **Anchor-by-construction test:** `cam01` gradient is `0.0` before and after backward pass, and `get_offset("cam01") == 0.0` after arbitrary optimizer steps without calling `clamp_and_anchor()`.
- **Quadratic prior test:** Prior loss and gradient are zero at initialization ($\delta_k = 0$) and have exact analytical value/sign for non-zero $\delta_k$.
- **Freeze/unfreeze & W0 no-op test:** No offset parameter updates occur before `learnable_offsets_freeze_iter`; disabled mode is bit-identical to W0.

## 5. Deliverables & Stop Conditions

- **Deliverables:** Hardened loader and `CameraOffsets` module, preset pair for `E1a`/`E1b`, unit test suite (`orchestrator/tests/test_w1_camera_timing.py`), and exact CLI commands for running `E1a` and `E1b`.
- **Stop conditions:** Do not proceed if training and evaluation timestamps differ, if `cam01` can move, or if `E1a` and `E1b` presets differ in any non-timing setting.

