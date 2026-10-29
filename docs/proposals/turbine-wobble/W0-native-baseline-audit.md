# W0 (T48) — Audit and freeze the native-4DGS baseline

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** —  
**Gate before W1:** Disabled configuration is numerically equivalent to native HexPlane deformation; dry-run resolves identically in both training entry points; baseline checkpoint load/save works without reading ROI or rigid-head files. (**PASSED**)

---

## 1. Objective

Establish a trustworthy, verified native-4DGS baseline before introducing timing refinement, soft loss weighting, or soft motion components. All hard spatial motion gates (`roi_bbox`, `use_roi_voxels`, `gated_coarse`, `pure_rigid`, `static_background`, `bypass_hexplane`), trajectory reprojection loss (`lambda_trajectory`), and rigid/Fourier heads (`use_fourier_rigid_head`) must be conclusively disabled and proven to be exact numerical no-ops.

---

## 2. Complete Configuration & Feature Matrix

Tracing from preset YAML (`orchestrator/pipeline/config/presets/*.yaml`) $\to$ `orchestrator/pipeline/config/models.py` $\to$ `orchestrator/pipeline/config/bridge.py` $\to$ `core/utils/params_utils.py:merge_hparams` $\to$ `core/arguments/__init__.py` $\to$ both `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`, plus `core/scene/__init__.py`, `core/gaussian_renderer/__init__.py`, and `core/utils/render_utils.py`:

| Feature / Mechanism | YAML Preset Keys | Pydantic Schema (`models.py`) | Bridge Dict(s) (`bridge.py`) | Core `arguments/__init__.py` Group(s) | Runtime Consumption Sites | W0 Native Baseline Value | Verified Disabled Behavior |
|---|---|---|---|---|---|---|---|
| **Fixed per-camera offsets** | `model.use_camera_offsets`, `optim.use_camera_offsets` | `ModelParams.use_camera_offsets`, `OptimizationParams.use_camera_offsets` (synced via `_sync_shared_param_group_flags`) | `ModelParams`, `OptimizationParams` | `ModelParams`, `OptimizationParams` | `Scene.__init__` $\to$ `MultipleView(..., use_camera_offsets=...)` (`core/scene/multipleview_dataset.py`) | `true` | Enabled: loads `offsets.json` (`20.004 fps` RTP/RTCP sidecar timestamps) into `CameraInfo.time`. |
| **Learnable per-camera offsets (R7)** | `optim.learnable_camera_offsets`, `learnable_offsets_lr`, `learnable_offsets_freeze_iter`, `max_offset_clamp_s` | `OptimizationParams` | `OptimizationParams` | `OptimizationParams` | `core/train.py`, `vendored/cuda/train.py` (`if getattr(opt, "learnable_camera_offsets", False):`) | `false` | Exact no-op: `camera_offsets_module = None`, `offsets_optimizer = None`, `custom_time = None`, `learned_offsets.json` never written. |
| **Trajectory reprojection loss (R8)** | `optim.lambda_trajectory`, `trajectory_track_file`, `trajectory_roi_bbox` | `OptimizationParams` | `OptimizationParams` | `OptimizationParams` | `core/train.py`, `vendored/cuda/train.py` (`if getattr(opt, "lambda_trajectory", 0.0) > 0.0:`) | `0.0`, `""`, `null` | Exact no-op: `TrajectoryReprojectionLoss` is never instantiated, optical track JSONs are never read, no trajectory term added to `loss`. |
| **Fourier / rigid head (R9)** | `optim.use_fourier_rigid_head`, `model.use_fourier_rigid_head`, `optim.fourier_head_*` | `OptimizationParams`, `ModelParams` (synced) | `OptimizationParams`, `ModelParams` | `OptimizationParams`, `ModelParams` | `core/train.py`, `vendored/cuda/train.py`, `core/scene/__init__.py`, `core/gaussian_renderer/__init__.py`, `core/utils/render_utils.py` | `false` (`pure_rigid: false`, `static_background: false`, `roi_bbox: null`) | Exact no-op: `fourier_head = None`, `gaussians.fourier_head = None`, `fourier_head.pth` never loaded or saved, `active_fourier_head is None`. |
| **HexPlane bypass (`bypass_hexplane`)** | Derived from `fourier_head_pure_rigid` $\land$ `fourier_head_static_background` | Derived at runtime | Derived at runtime | Derived at runtime | `core/gaussian_renderer/__init__.py:128`, `core/utils/render_utils.py:15`, `core/train.py:334`, `vendored/cuda/train.py:355` | `false` (derived) | Exact no-op: `active_fourier_head is None` forces `bypass_hexplane = False`; 100% of Gaussians execute `pc._deformation(...)` in fine stage and HexPlane regularizers remain active. |
| **Coarse-stage gating (R10 Option B)** | `optim.gated_coarse`, `pipeline_params.gated_coarse` | `OptimizationParams`, `PipelineParams` (synced) | `OptimizationParams`, `PipelineParams` | `OptimizationParams`, `PipelineParams` | `core/train.py`, `vendored/cuda/train.py`, `core/gaussian_renderer/__init__.py:143` | `false` | Exact no-op: coarse stage returns canonical static parameters `(means3D, scales, rotations, opacity, shs)` without calling `pc._deformation` or `_resolve_roi_mask`. |
| **3D ROI bounding box / voxel grid (T41/T43)** | `optim.roi_bbox`, `pipeline_params.roi_bbox`, `optim.use_roi_voxels`, `model.use_roi_voxels` | `OptimizationParams`, `PipelineParams`, `ModelParams` (synced) | `OptimizationParams`, `PipelineParams`, `ModelParams` | `OptimizationParams`, `PipelineParams`, `ModelParams` | `core/train.py`, `vendored/cuda/train.py`, `core/scene/__init__.py`, `core/gaussian_renderer/__init__.py:_resolve_roi_mask` | `roi_bbox: null`, `use_roi_voxels: false` | Exact no-op: `roi_voxels.npz` is never read even when present on disk; `gaussians._roi_grid = None`; `_resolve_roi_mask` is never invoked in coarse or fine stages. |
| **Motion-mask image loss (R2)** | `optim.use_motion_mask`, `motion_mask_dir`, `motion_mask_max` | `OptimizationParams` | `OptimizationParams` | `OptimizationParams` | `core/train.py:289`, `vendored/cuda/train.py:310` | `use_motion_mask: false`, `motion_mask_dir: ""` | Exact no-op: per-iteration `cam_mask` GPU transfer and bilinear interpolation are skipped; `mask_tensor = None`; `l1_loss` computes standard unweighted L1. |
| **Coarse iterations & HexPlane temporal regularizers (R1/R10 Option A)** | `optim.coarse_iterations`, `hidden.regularizer_anneal_iter`, `hidden.time_smoothness_weight_final`, `hidden.l1_time_planes_final` | `OptimizationParams`, `ModelHiddenParams` | `OptimizationParams`, `ModelHiddenParams` | `OptimizationParams`, `ModelHiddenParams` | `core/train.py:55,737`, `vendored/cuda/train.py:55,758` | `coarse_iterations: 500`, `regularizer_anneal_iter: 3000`, `time_smoothness_weight_final: 0.0001`, `l1_time_planes_final: 0.0` | Active baseline schedule: 500 coarse iterations followed by fine HexPlane stage with step-annealed temporal regularization at iter 3000. |

---

## 3. Audit Findings & Blockers Fixed in W0

1. **Blocker A — `merge_hparams` flat-namespace clobbering across parameter groups ([`core/utils/params_utils.py`](../../../core/utils/params_utils.py) & [`orchestrator/pipeline/config/models.py`](../../../orchestrator/pipeline/config/models.py)):**
   - *Root Cause:* `bridge.py` emits all four parameter dicts (`ModelParams`, `PipelineParams`, `ModelHiddenParams`, `OptimizationParams`) with all fields. Previously, `merge_hparams` iterated `["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]` onto a single flat `argparse.Namespace`. Because `ModelParams` and `PipelineParams` were merged *after* `OptimizationParams`, setting `optim.gated_coarse`, `optim.roi_bbox`, or `optim.use_camera_offsets` in a YAML preset without duplicating them under `pipeline_params:` / `model:` caused `PipelineParams` / `ModelParams` defaults to overwrite the `OptimizationParams` values on `args`.
   - *Fix:*
     1. Reordered `merge_hparams` in [`core/utils/params_utils.py`](../../../core/utils/params_utils.py) to `["ModelParams", "PipelineParams", "ModelHiddenParams", "OptimizationParams"]` with explicit reconciliation for shared keys (`gated_coarse`, `roi_bbox`, `use_camera_offsets`, `render_process`).
     2. Added `@model_validator(mode="after") def _sync_shared_param_group_flags` on `PipelineConfig` in [`orchestrator/pipeline/config/models.py`](../../../orchestrator/pipeline/config/models.py#L654-L681) using `.model_fields_set` so shared flags stay synchronized in the serialized bridge config as well.

2. **Blocker B — Unconditional auto-loading of `roi_voxels.npz` and `fourier_head.pth` on model reload ([`core/scene/__init__.py`](../../../core/scene/__init__.py#L95-L129) & [`core/arguments/__init__.py`](../../../core/arguments/__init__.py#L72-L74)):**
   - *Root Cause:* When `self.loaded_iter` was truthy (during evaluation, rendering, or trajectory extraction via `seg_extract`), `Scene.__init__` unconditionally checked `os.path.exists(os.path.join(args.source_path, "roi_voxels.npz"))` and `os.path.exists(os.path.join(self.model_path, "fourier_head.pth"))` and attached `self.gaussians._roi_grid` and `self.gaussians.fourier_head` regardless of config flags. Because `data/multipleview/lego_turbine_2_missing/roi_voxels.npz` and `data/multipleview/turbine_r1_missing_6/roi_voxels.npz` exist on disk from Phase 13, reloading any baseline model on those scenes silently attached `_roi_grid`.
   - *Fix:* Added `use_roi_voxels = False` and `use_fourier_rigid_head = False` to `ModelParams` in [`core/arguments/__init__.py`](../../../core/arguments/__init__.py#L72-L74) and [`orchestrator/pipeline/config/models.py`](../../../orchestrator/pipeline/config/models.py#L168-L173), and gated `roi_voxels.npz` loading on `if getattr(args, "use_roi_voxels", False):` and `fourier_head.pth` loading on `if getattr(args, "use_fourier_rigid_head", False):` in [`core/scene/__init__.py`](../../../core/scene/__init__.py#L95-L129).

3. **Blocker C — Unnecessary per-iteration GPU mask transfer & 6 MP interpolation when `use_motion_mask=False` ([`core/train.py`](../../../core/train.py#L288-L290) & [`orchestrator/pipeline/vendored/cuda/train.py`](../../../orchestrator/pipeline/vendored/cuda/train.py#L309-L311)):**
   - *Root Cause:* `multipleview_dataset` loads `<source_path>/motion_masks/*.npy` onto `CameraInfo.mask` if the folder exists on disk. Both `core/train.py` and `vendored/cuda/train.py` previously copied `cam_mask` to CUDA and ran `F.interpolate` to full 6 MP resolution on every training step whenever `cam_mask is not None`, only checking `opt.use_motion_mask` afterwards when assembling `mask_tensor`.
   - *Fix:* Gated the per-camera mask processing block on `if getattr(opt, "use_motion_mask", False) and cam_mask is not None:` in both [`core/train.py`](../../../core/train.py#L289) and [`orchestrator/pipeline/vendored/cuda/train.py`](../../../orchestrator/pipeline/vendored/cuda/train.py#L310).

---

## 4. Files Changed & Remaining Duplication Note

### Files Changed in W0
- [`core/scene/__init__.py`](../../../core/scene/__init__.py) — Gated `roi_voxels.npz` and `fourier_head.pth` auto-loading on `args.use_roi_voxels` and `args.use_fourier_rigid_head`.
- [`core/arguments/__init__.py`](../../../core/arguments/__init__.py) — Added `use_roi_voxels` and `use_fourier_rigid_head` to `ModelParams`.
- [`core/utils/params_utils.py`](../../../core/utils/params_utils.py) — Fixed `merge_hparams` merge order and shared-flag reconciliation.
- [`core/train.py`](../../../core/train.py) — Gated per-step `cam_mask` GPU transfer/interpolation on `opt.use_motion_mask`.
- [`orchestrator/pipeline/vendored/cuda/train.py`](../../../orchestrator/pipeline/vendored/cuda/train.py) — Synchronized `cam_mask` guard with `core/train.py`.
- [`orchestrator/pipeline/config/models.py`](../../../orchestrator/pipeline/config/models.py) — Added `use_roi_voxels` and `use_fourier_rigid_head` to `ModelParams` and added `_sync_shared_param_group_flags` validator on `PipelineConfig`.
- [`orchestrator/pipeline/config/presets/lego_turbine_2_missing_w0_baseline.yaml`](../../../orchestrator/pipeline/config/presets/lego_turbine_2_missing_w0_baseline.yaml) — W0 native-4DGS baseline preset for `lego_turbine_2_missing`.
- [`orchestrator/pipeline/config/presets/turbine_r1_missing_6_w0_baseline.yaml`](../../../orchestrator/pipeline/config/presets/turbine_r1_missing_6_w0_baseline.yaml) — W0 native-4DGS baseline preset for `turbine_r1_missing_6`.
- [`orchestrator/tests/test_w0_native_baseline.py`](../../../orchestrator/tests/test_w0_native_baseline.py) — W0 regression and parity test suite (7 tests).

### Remaining Code Duplication Between `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`
- Per the repository's vendoring convention (`AGENTS.md` §7), `orchestrator/pipeline/vendored/cuda/train.py` is a standalone in-container copy of `core/train.py` with one intentional addition: `_write_Per_Camera_Eval()` (which writes `eval_per_camera.json` during `training_report`).
- All W0–W7 training loop changes must either call shared modules under `core/scene/` or `core/utils/` (which are mounted at `/workspace/core` inside the `cuda` container) or update both `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py` in lockstep, guarded by parity tests in `orchestrator/tests/`.

---

## 5. Verification Summary & Exact Baseline Configuration for W1

### Executed Verification
- `uv run --group test pytest orchestrator/tests/test_w0_native_baseline.py -q` $\to$ **7 passed**
- `uv run --group test pytest orchestrator/tests/test_coarse_gating.py orchestrator/tests/test_motion_mask_loss.py orchestrator/tests/test_per_camera_timestamps.py orchestrator/tests/test_learnable_offsets_trajectory_loss.py orchestrator/tests/test_fourier_rigid_head.py -q` $\to$ **51 passed**
- `uv run amp train lego_turbine_2_missing_w0_baseline --scene data/multipleview/lego_turbine_2_missing --dry-run` $\to$ **Resolved cleanly** (`offsets.json` SHA-256: `f5f3ddfc823df37172b47ab42cb16058243e422fde7b74cb2079c4c66e0b8d25`)
- `uv run amp train turbine_r1_missing_6_w0_baseline --scene data/multipleview/turbine_r1_missing_6 --dry-run` $\to$ **Resolved cleanly** (`offsets.json` SHA-256: `f179372ad49eb80f7f76edde6f9ced96e915ff19403ca68d6afcd394ac0bcbef`)

### Exact Baseline Presets for W1 (`E1a` / `E1b`)
Work Package **W1 (T49)** must extend directly from:
- [`lego_turbine_2_missing_w0_baseline.yaml`](../../../orchestrator/pipeline/config/presets/lego_turbine_2_missing_w0_baseline.yaml)
- [`turbine_r1_missing_6_w0_baseline.yaml`](../../../orchestrator/pipeline/config/presets/turbine_r1_missing_6_w0_baseline.yaml)
