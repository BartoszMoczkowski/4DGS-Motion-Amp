# T39 — Per-camera timestamps in the loader (R6) — PIVOTAL

**Status:** done
**Phase:** 12 / Phase 2 (the timestamp fix)
**Depends on:** T32 or T33 (an offset table)
**Remedy:** R6

## Goal

Fix the root defect: frame *i* of every camera was asserted to be simultaneous.
Give each camera its own continuous timestamp $t_k = t + \delta_k$.

## Implementation Summary

1. **`core/scene/multipleview_dataset.py`:**
   - Evaluates continuous normalized timestamp:
     $$t_{\text{norm}}(k, i) = \frac{(1 + d_k) \cdot i + \delta_k \cdot \text{fps}}{\text{image\_length}}$$
   - Evaluates continuous physical time in seconds:
     $$t_{\text{phys}}(k, i) = (1 + d_k) \cdot \frac{i}{\text{fps}} + \delta_k$$
   - Anchors offsets to `reference_camera` (`cam01`, $\delta = 0.0, d = 0.0$), ensuring `cam01` preserves exact canonical normalized time $i / \text{image\_length}$.
   - Loads `offsets.json` (or `optical_offsets.json` fallback).
   - Exposes accessors:
     - `get_physical_time(index)`: true capture instant in seconds
     - `get_camera_name(index)`: camera identifier (e.g. `cam02`)
     - `get_camera_offset(cam_name)`: relative offset $\delta_k$ in seconds
     - `get_camera_drift(cam_name)`: drift rate $d_k$ in s/s
   - Graceful fallback: when no offset table is found, defaults to legacy simultaneous frame times ($i / \text{image\_length}$ and physical time $i / \text{fps}$ with $\delta = 0$).
   - Full ablation support via `use_camera_offsets=False` and `PIPELINE_USE_CAMERA_OFFSETS=0`.

2. **`core/scene/dataset.py` (`FourDGSdataset`):**
   - Forwards `fps`, `offsets_s`, `has_offsets`, `drift_s_per_s`, and `get_physical_time` to wrapped dataset.
   - Forwards `physical_time` to instantiated `Camera` instances in `__getitem__`.

3. **`core/scene/cameras.py` (`Camera`):**
   - Stores `self.physical_time = physical_time` as optional per-camera observation timestamp in seconds.

4. **`core/arguments/__init__.py` & `orchestrator/pipeline/config/models.py`:**
   - Added `use_camera_offsets: bool = True` to `OptimizationParams` and `ModelParams`.
   - Verified serialization into training bridge files via `render_bridge_source`.

## Verification

- **Automated unit and integration test suite:**
  [`orchestrator/tests/test_per_camera_timestamps.py`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/orchestrator/tests/test_per_camera_timestamps.py)
  - `test_offsets_loading_synthetic_preflight`: verifies `offsets.json` loading from preflight scene.
  - `test_offsets_loading_real_turbine_scene`: verifies `offsets.json` and drift loading on `lego_turbine_2_missing`.
  - `test_optical_offsets_fallback`: verifies seamless fallback to `optical_offsets.json`.
  - `test_per_camera_timestamps_distinct_normalized`: verifies distinct per-camera $t_{\text{norm}}$ and $t_{\text{phys}}$.
  - `test_backward_compatibility_no_offsets`: verifies identical shared times when offsets table is absent.
  - `test_ablation_disabled_offsets`: verifies parameter and env var overrides disable offsets.
  - `test_fourdgsdataset_and_camera_wiring`: verifies `Camera.time` and `Camera.physical_time` propagation.
  - `test_config_models_and_bridge_generation`: verifies Pydantic model and bridge rendering.
  - Result: 8 passed in 4.25s.
- **Global Regression Check:** 516 passed, 9 skipped in 93.2s.
- **Pre-flight Gates G1–G5:** Verified on `synthetic_turbine_preflight` with max offset discrepancy 0.47 ms vs 2.0 ms bar.

