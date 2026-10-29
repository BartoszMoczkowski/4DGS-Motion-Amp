# W0 (T48) — Audit and freeze the native-4DGS baseline

**Status:** todo  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** —  
**Gate before W1:** Disabled configuration is numerically equivalent to native HexPlane deformation; dry-run resolves identically in both training entry points; baseline checkpoint load/save works without reading ROI or rigid-head files.

---

## 1. Objective

Establish a trustworthy, verified native-4DGS baseline before introducing timing refinement, soft loss weighting, or soft motion components. All hard spatial motion gates (`roi_bbox`, `use_roi_voxels`, `gated_coarse`, `pure_rigid`, `static_background`, `bypass_hexplane`), trajectory reprojection loss (`lambda_trajectory`), and rigid/Fourier heads (`use_fourier_rigid_head`) must be conclusively disabled and proven to be exact numerical no-ops.

## 2. Codebase Findings & Blockers to Fix in W0

1. **Unconditional artifact auto-load in [`core/scene/__init__.py`](../../core/scene/__init__.py#L95-L126):**
   - Currently, when `self.loaded_iter` is truthy (e.g. during evaluation, rendering, or trajectory extraction), `Scene.__init__` unconditionally checks `os.path.exists(os.path.join(args.source_path, "roi_voxels.npz"))` and `os.path.exists(os.path.join(self.model_path, "fourier_head.pth"))` and attaches `self.gaussians._roi_grid` and `self.gaussians.fourier_head` regardless of config flags (`use_roi_voxels`, `use_fourier_rigid_head`).
   - **Fix required:** Gate loading of `roi_voxels.npz` on `getattr(args, "use_roi_voxels", False)` and loading of `fourier_head.pth` on `getattr(args, "use_fourier_rigid_head", False)` (or explicit CLI flag). A present or missing `roi_voxels.npz` on disk must have zero effect when disabled.
2. **Dual training entry-point parity:**
   - [`core/train.py`](../../core/train.py) and [`orchestrator/pipeline/vendored/cuda/train.py`](../../orchestrator/pipeline/vendored/cuda/train.py) must resolve identical parameter values from [`orchestrator/pipeline/config/models.py`](../../orchestrator/pipeline/config/models.py) and [`orchestrator/pipeline/config/bridge.py`](../../orchestrator/pipeline/config/bridge.py).
3. **Inherited preset flags:**
   - Existing scene presets (e.g. `lego_turbine_2_missing.yaml`, `turbine_r1_missing_6.yaml`) inherit from `real_capture.yaml`, which enables `gated_coarse: true`, `use_motion_mask: true`, etc. The W0 baseline preset must explicitly override all motion-specific gates to `false` / `0.0` / `null` while keeping agreed coarse iterations (`500`) and temporal regularizer schedule fixed.

## 3. Implementation Steps

1. **Trace & document feature matrix** across [`orchestrator/pipeline/config/models.py`](../../orchestrator/pipeline/config/models.py), [`orchestrator/pipeline/config/bridge.py`](../../orchestrator/pipeline/config/bridge.py), [`core/arguments/__init__.py`](../../core/arguments/__init__.py), [`core/train.py`](../../core/train.py), [`orchestrator/pipeline/vendored/cuda/train.py`](../../orchestrator/pipeline/vendored/cuda/train.py), and [`core/gaussian_renderer/__init__.py`](../../core/gaussian_renderer/__init__.py) for:
   - `use_camera_offsets` (fixed per-camera offsets)
   - `learnable_camera_offsets`, `learnable_offsets_lr`, `learnable_offsets_freeze_iter`, `max_offset_clamp_s`
   - `lambda_trajectory`, `trajectory_track_file`, `trajectory_roi_bbox`
   - `use_fourier_rigid_head`, `fourier_head_*` (`pure_rigid`, `static_background`, `roi_bbox`)
   - `bypass_hexplane` derived condition in renderer & training regularizer
   - `roi_bbox`, `use_roi_voxels`, `_resolve_roi_mask`
   - `gated_coarse`
   - `use_motion_mask`, `motion_mask_dir`, `motion_mask_max`
   - `coarse_iterations` and HexPlane temporal regularizer weights/annealing
2. **Fix unconditional ROI/head loading** in [`core/scene/__init__.py`](../../core/scene/__init__.py) and ensure `_resolve_roi_mask` is never invoked when `gated_coarse=False` and `active_fourier_head is None`.
3. **Create native-4DGS baseline preset(s)** under `orchestrator/pipeline/config/presets/` (e.g., `turbine_w0_native_baseline.yaml` / scene-specific baseline) with native HexPlane deformation everywhere and all hard gates / extra heads / masks / trajectory losses explicitly off.
4. **Add dry-run parity check** verifying resolved config values match across `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`.

## 4. Verification & Tests

- **Disabled-feature no-op test:** Verify on a small synthetic/random Gaussian tensor that `render()` and deformation dispatch with baseline settings call `pc._deformation(...)` for 100% of Gaussians and never read `roi_voxels.npz`, even if `roi_voxels.npz` exists in `source_path`.
- **Config resolution & bridge parity test:** Verify both training entry points parse identical `ModelParams`, `PipelineParams`, `ModelHiddenParams`, and `OptimizationParams` from the baseline preset.
- **Checkpoint save/load smoke test:** Confirm saving and reloading a baseline checkpoint does not create or load `fourier_head.pth` or `_roi_grid`.

## 5. Deliverables & Stop Conditions

- **Deliverables:** Feature matrix table, baseline preset YAML, regression test file (`orchestrator/tests/test_w0_native_baseline.py`), and short audit note.
- **Stop conditions:** Do not proceed to W1 if any hard gate can still alter deformation, if the two training entry points disagree on any resolved parameter, or if baseline checkpoint/resume fails.

