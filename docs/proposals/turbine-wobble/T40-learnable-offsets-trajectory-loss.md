# T40 — Learnable offsets + trajectory reprojection loss (R7+R8)

**Status:** done
**Phase:** 12 / Phase 2 (the timestamp fix)
**Depends on:** T39, T36
**Remedies:** R7, R8

## Goal

Refine the Phase-0 offset estimates jointly with the model, disambiguated by
the already-measured 2D wobble tracks.

## R7 — Learnable offset refinement (guarded)

- $\delta_k$ as optimizable parameters, initialized from `offsets.json` (or zero),
  registered in a dedicated optimizer group in `core/train.py` (`offsets_optimizer`)
  with learning rate `learnable_offsets_lr` (default 0.0001); frozen for the first ~10k iterations
  (`learnable_offsets_freeze_iter` schedule).
- **Guardrails are mandatory:** `cam01` anchored at $\delta_1 = 0$; $\delta_k$ clamped to
  $\pm 40$ ms (`max_offset_clamp_s`); continuous normalized time $t_{\text{norm}}$ remains differentiable
  with respect to $\delta_k$ throughout training.
- Checkpoints save effective learned offsets to `learned_offsets.json`.

## R8 — Trajectory reprojection loss

- The diagnostic pipeline tracked 55 points on the turbine at sub-pixel coherence.
- Project deformed ROI Gaussian positions into each camera at its corrected
  timestamp; Smooth-L1 align projected 2D **displacement** against the measured
  tracks (`trajectory_tracks.json`). Supervises amplitude, phase, axis, and the per-camera offsets
  simultaneously.
- Supervise displacement of identified points, **never dense optical flow**
  (rendered-Gaussian-flow supervision has a known domain gap — DispFlow-GS).
- Supported across measured and analytical tracks via `scripts/export_trajectory_tracks.py`.

## Deliverables & Implementation Summary

1. `core/scene/camera_offsets.py`: `CameraOffsets(nn.Module)` with anchor guardrail, clamp guardrail,
   differentiable normalized timestamp generation, and `clamp_and_anchor()`.
2. `core/utils/trajectory_loss.py`: `TrajectoryReprojectionLoss(nn.Module)` with 3D to 2D camera projection,
   multi-format track loading, and displacement Smooth-L1 loss.
3. `scripts/export_trajectory_tracks.py`: Track exporter for all 5 synthetic and real turbine scenes.
4. `core/gaussian_renderer/__init__.py`: Accepts `custom_time: Optional[torch.Tensor]` without breaking
   computation graph and exposes `means3D_final` in `render_pkg`.
5. `core/train.py` & `orchestrator/pipeline/vendored/cuda/train.py`: Dedicated optimizer group, freeze schedule,
   clamp guardrail, trajectory loss evaluation, and `learned_offsets.json` persistence.
6. `core/arguments/__init__.py` & `orchestrator/pipeline/config/models.py`: Schema additions for R7 & R8.
7. `orchestrator/pipeline/config/presets/lego_turbine_2_missing_r7.yaml` and `lego_turbine_2_missing_r8.yaml`.
8. `orchestrator/tests/test_learnable_offsets_trajectory_loss.py`: 11 automated unit and integration tests (all passing).

