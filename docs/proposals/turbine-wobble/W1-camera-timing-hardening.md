# W1 (T49) — Harden fixed and guarded learnable per-camera timing

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W0 (T48)  
**Experiments:** E1a (fixed sidecar timestamps) vs E1b (guarded learnable offsets)  
**Gate before W2/W3:** Loader/optimizer unit tests pass; training and evaluation timestamps match identically; fixed-vs-learnable preset pair differs *only* in offset optimization settings. (**PASSED**)

---

## 1. Objective

Make the fixed-versus-learnable timing comparison (`E1a` vs `E1b`) mathematically clean and self-consistent under native HexPlane 4DGS, with zero motion masks, zero rigid heads, and zero trajectory losses.

---

## 2. Chosen Time Convention & Quadratic Prior Parameterization

### 2.1 Single Time Convention (`core/scene/camera_offsets.py`)
1. **Physical Reference Instant ($t_{\text{phys}}$, seconds):**
   Each image $(k, i)$ (camera $k$, frame index $i \in \{0, \dots, N_{\text{frames}} - 1\}$) has a physical exposure-reference instant $t_{\text{phys}}(k, i)$ measured in seconds relative to `cam01` frame $0$ ($t_{\text{phys}}(\text{cam01}, 0) = 0.0\text{ s}$):
   $$t_{\text{phys}}(k, i) = (1 + d_k)\frac{i}{\text{fps}} + \Delta t_k$$
   where $d_k$ is the per-camera clock drift (`drift_s_per_s`) and $\Delta t_k = \Delta t_k^{\text{init}} + \delta_k$ is the per-camera offset anchored at `cam01` ($\Delta t_{\text{cam01}} \equiv 0$).
   - **Start-of-exposure vs. mid-exposure note:** In the recovered RTSP/RTCP sidecar tables (`offsets.json`), all cameras share an identical $1/100\text{ s}$ ($10\text{ ms}$) shutter exposure (`exposure_s = 0.01`, `mid_exposure_corrected = True`). Because the $+5\text{ ms}$ shift (`exposure_s / 2`) from sensor frame-start to mid-exposure is identical across all cameras, pairwise relative offsets $\Delta t_k$ relative to `cam01` are identical under both start-of-exposure and mid-exposure conventions, and `exposure_s / 2` is never double-applied.
2. **Centralized Conversion to Normalized 4DGS Time ($t_{\text{norm}}$):**
   Conversion from $t_{\text{phys}}$ to normalized model time $t_{\text{norm}}$ happens exclusively via [`compute_physical_time`](../../../core/scene/camera_offsets.py#L43-L53), [`physical_to_normalized_time`](../../../core/scene/camera_offsets.py#L56-L66), [`compute_normalized_time`](../../../core/scene/camera_offsets.py#L69-L81), and [`resolve_camera_time`](../../../core/scene/camera_offsets.py#L84-L96):
   $$t_{\text{norm}}(k, i) = \frac{t_{\text{phys}}(k, i) \cdot \text{fps}}{N_{\text{frames}}} = \frac{(1 + d_k)\,i + \Delta t_k \cdot \text{fps}}{N_{\text{frames}}}$$
   Training (`scene_reconstruction`), held-out evaluation (`training_report`), and offline `Scene` reload (`learned_offsets.json` $\to$ `apply_offsets_dict`) all evaluate cameras at this exact timestamp.

### 2.2 Structural Anchor & Quadratic Sidecar Prior (`CameraOffsets`)
1. **Structural Anchor (`cam01`):**
   `CameraOffsets` registers a binary buffer `anchor_mask` with `0.0` at `anchor_idx` (`cam01`) and `1.0` elsewhere, multiplies `delta_offsets * anchor_mask` inside the forward graph (`get_masked_delta()`), and attaches a backward hook (`_zero_anchor_grad`) on `delta_offsets`. Consequently, `cam01` has identically zero gradient and remains fixed at `0.0` by construction before, during, and after any optimizer step without relying on post-step projection.
2. **Quadratic Sidecar Prior:**
   $$\mathcal{L}_{\text{offset\_prior}} = \lambda_{\text{offset\_prior}} \sum_{k \neq \text{anchor}} \left(\frac{\delta_k}{\sigma_{\text{offset}}}\right)^2$$
   where $\delta_k$ is the learned residual offset deviation (in seconds) from the sidecar initialization `base_offsets[k]`, $\sigma_{\text{offset}} = \text{offset\_prior\_sigma\_s} = 0.005\text{ s}$ ($5\text{ ms}$), and $\lambda_{\text{offset\_prior}} = 0.01$.
3. **Freeze Schedule & Secondary Safety Clamp:**
   - For coarse stage and fine iterations $< \text{learnable\_offsets\_freeze\_iter}$ (`3000`), `custom_time` is detached in `train.py` so `delta_offsets` receives zero gradient while initial geometry and HexPlane features stabilize.
   - `max_offset_clamp_s = 0.040` ($40\text{ ms}$) acts only as a secondary safety bound on total offset `base_offsets[k] + delta_k`.

---

## 3. Files Changed in W1

- [`core/scene/camera_offsets.py`](../../../core/scene/camera_offsets.py) — Centralized `compute_physical_time`, `physical_to_normalized_time`, `compute_normalized_time`, and `resolve_camera_time`; upgraded `CameraOffsets` with structural `anchor_mask` + backward hook, `compute_prior_loss()`, and full auditable telemetry export (`get_telemetry_dict()`).
- [`core/scene/multipleview_dataset.py`](../../../core/scene/multipleview_dataset.py) — Enforced strict validation of `offsets.json` (JSON validity, finite positive `fps` and `exposure_s`, finite offsets/drifts, full camera coverage against `cam_extrinsics`, SHA-256 digest `offsets_file_sha256`, and refusal of `optical_offsets.json` fallback when `strict_camera_offsets=True`); added `apply_offsets_dict()` for reload parity.
- [`core/scene/dataset_readers.py`](../../../core/scene/dataset_readers.py), [`core/scene/dataset.py`](../../../core/scene/dataset.py), [`core/scene/__init__.py`](../../../core/scene/__init__.py) — Threaded `strict_camera_offsets`, `offsets_file`, `offsets_file_sha256`, and `learned_offsets.json` reload into `Scene`.
- [`core/train.py`](../../../core/train.py) & [`orchestrator/pipeline/vendored/cuda/train.py`](../../../orchestrator/pipeline/vendored/cuda/train.py) — Wired `resolve_camera_time` into both training and `training_report` evaluation, detached `custom_time` during frozen iterations, added `compute_prior_loss()` to fine-stage loss after `learnable_offsets_freeze_iter`, and saved full telemetry to `learned_offsets.json` and `training_curves.json`.
- [`orchestrator/pipeline/stages/train.py`](../../../orchestrator/pipeline/stages/train.py) — Recorded `offsets_file_sha256` and `learned_offsets` telemetry into the `train.default` run manifest.
- [`core/arguments/__init__.py`](../../../core/arguments/__init__.py), [`core/utils/params_utils.py`](../../../core/utils/params_utils.py), [`orchestrator/pipeline/config/models.py`](../../../orchestrator/pipeline/config/models.py) — Added `strict_camera_offsets`, `offsets_file`, `offset_prior_sigma_s`, and `lambda_offset_prior` across schemas and `merge_hparams`.
- **Single-Factor Presets (`E1a` vs `E1b`):**
  - [`orchestrator/pipeline/config/presets/lego_turbine_2_missing_w1_e1a.yaml`](../../../orchestrator/pipeline/config/presets/lego_turbine_2_missing_w1_e1a.yaml)
  - [`orchestrator/pipeline/config/presets/lego_turbine_2_missing_w1_e1b.yaml`](../../../orchestrator/pipeline/config/presets/lego_turbine_2_missing_w1_e1b.yaml)
  - [`orchestrator/pipeline/config/presets/turbine_r1_missing_6_w1_e1a.yaml`](../../../orchestrator/pipeline/config/presets/turbine_r1_missing_6_w1_e1a.yaml)
  - [`orchestrator/pipeline/config/presets/turbine_r1_missing_6_w1_e1b.yaml`](../../../orchestrator/pipeline/config/presets/turbine_r1_missing_6_w1_e1b.yaml)
- [`orchestrator/tests/test_w1_camera_timing.py`](../../../orchestrator/tests/test_w1_camera_timing.py) — 7 unit and integration tests for W1.

---

## 4. Verification & Exact Commands for `E1a` / `E1b`

### Executed Verification
- `uv run --group test pytest orchestrator/tests/test_w1_camera_timing.py orchestrator/tests/test_w0_native_baseline.py orchestrator/tests/test_per_camera_timestamps.py orchestrator/tests/test_learnable_offsets_trajectory_loss.py -q` $\to$ **34 passed**
- Dry-run verification on all 4 W1 presets $\to$ **OK**

### Exact CLI Commands for Running the `E1a` / `E1b` Experiment Pair (after W2 Evaluation Tool Lands)

```bash
# Development scene 1: lego_turbine_2_missing
uv run amp train lego_turbine_2_missing_w1_e1a --scene data/multipleview/lego_turbine_2_missing
uv run amp train lego_turbine_2_missing_w1_e1b --scene data/multipleview/lego_turbine_2_missing

# Development scene 2: turbine_r1_missing_6
uv run amp train turbine_r1_missing_6_w1_e1a --scene data/multipleview/turbine_r1_missing_6
uv run amp train turbine_r1_missing_6_w1_e1b --scene data/multipleview/turbine_r1_missing_6
```
