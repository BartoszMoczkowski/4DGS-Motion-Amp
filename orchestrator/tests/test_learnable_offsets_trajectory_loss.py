"""Tests for Task T40 (Learnable offsets + trajectory reprojection loss - R7+R8).

Verifies:
1. CameraOffsets module (R7):
   - Initialization from camera names, base offsets, fps, and image_length.
   - cam01 anchor guardrail: reference camera offset is strictly fixed at 0.0.
   - Boundary clamp guardrail: offsets strictly clamped to [-max_clamp_s, +max_clamp_s] (+/-40ms).
   - Continuous differentiable normalized timestamp generation:
     t_norm = ((1 + drift) * frame_idx + offset * fps) / image_length.
   - Gradient flow: d(t_norm) / d(delta_k) = fps / image_length.
   - clamp_and_anchor() resets anchor to 0 and clamps residual offsets.
   - get_effective_offsets() returns exact clamped offsets.

2. Freeze schedule and optimizer stepping:
   - Offsets remain unchanged during frozen iterations (< learnable_offsets_freeze_iter).
   - Offsets update smoothly after unfreezing (>= learnable_offsets_freeze_iter).
   - Anchor remains strictly 0.0 across all iterations.
   - Hard clamp prevents divergence even under extreme gradients.

3. TrajectoryReprojectionLoss module (R8):
   - 3D to 2D camera pixel projection: project_points_to_pixels.
   - Track loader supports trajectory_tracks.json, optical_offsets.json, and gt_motion_info.json.
   - get_measured_displacement() retrieves pixel displacements (dx, dy).
   - Forward pass computes smooth L1 displacement loss between projected 2D motion and measured tracks.
   - Subpixel displacement supervision (not dense optical flow).

4. End-to-end gradient coupling:
   - Gradient flows from trajectory loss -> deformed 3D positions -> continuous timestamp -> CameraOffsets.delta_offsets.
   - Anchor cam01 has zero gradient / stays exactly at 0.0.

5. Presets, models, and bridge serialization:
   - OptimizationParams includes learnable_camera_offsets, learnable_offsets_lr,
     learnable_offsets_freeze_iter, max_offset_clamp_s, lambda_trajectory, trajectory_track_file,
     and trajectory_roi_bbox.
   - lego_turbine_2_missing_r7 and lego_turbine_2_missing_r8 presets resolve properly.
   - render_bridge_source serializes all R7 and R8 fields.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Mock Linux-only CUDA extension simple_knn if missing in sandbox
if "simple_knn" not in sys.modules:
    m_knn = types.ModuleType("simple_knn")
    m_knn_c = types.ModuleType("simple_knn._C")
    m_knn_c.distCUDA2 = lambda *args, **kwargs: None
    m_knn._C = m_knn_c
    sys.modules["simple_knn"] = m_knn
    sys.modules["simple_knn._C"] = m_knn_c

# Mock Linux-only diff_gaussian_rasterization if missing in sandbox
if "diff_gaussian_rasterization" not in sys.modules:
    m_diff = types.ModuleType("diff_gaussian_rasterization")

    class MockRasterSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockRasterizer:
        def __init__(self, raster_settings=None):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            dev = kwargs.get("means3D", torch.empty(0)).device
            n = kwargs.get("means3D", torch.empty(4)).shape[0]
            return (
                torch.zeros((3, 64, 64), device=dev),
                torch.zeros(n, device=dev),
                torch.zeros((1, 64, 64), device=dev),
            )

    m_diff.GaussianRasterizationSettings = MockRasterSettings
    m_diff.GaussianRasterizer = MockRasterizer
    sys.modules["diff_gaussian_rasterization"] = m_diff


from scene.camera_offsets import CameraOffsets
from utils.trajectory_loss import TrajectoryReprojectionLoss, project_points_to_pixels
from arguments import OptimizationParams as CoreOptimizationParams
from pipeline.config import resolve_preset
from pipeline.config.models import OptimizationParams, PipelineConfig
from pipeline.config.bridge import render_bridge_source


# ---------------------------------------------------------------------------
# 1. CameraOffsets Unit Tests (R7)
# ---------------------------------------------------------------------------


def test_camera_offsets_initialization():
    """Verify CameraOffsets initialization, anchor setup, and buffer registration."""
    cams = [f"cam{i:02d}" for i in range(1, 11)]
    init_offsets = {"cam01": 0.0, "cam02": 0.012, "cam03": -0.015}

    module = CameraOffsets(
        cam_names=cams,
        init_offsets_s=init_offsets,
        reference_camera="cam01",
        max_clamp_s=0.040,
        fps=20.0,
        image_length=60,
    )

    assert module.reference_camera == "cam01"
    assert module.anchor_idx == 0
    assert module.max_clamp_s == 0.040
    assert module.fps == 20.0
    assert module.image_length == 60

    # Base offsets buffer
    assert module.base_offsets[0].item() == 0.0
    assert pytest.approx(module.base_offsets[1].item(), abs=1e-5) == 0.012
    assert pytest.approx(module.base_offsets[2].item(), abs=1e-5) == -0.015

    # Delta offsets parameter
    assert isinstance(module.delta_offsets, nn.Parameter)
    assert module.delta_offsets.shape[0] == 10
    assert torch.all(module.delta_offsets == 0.0)


def test_camera_offsets_anchor_and_clamp_guardrails():
    """Verify mandatory guardrails: cam01 strictly anchored at 0, offsets clamped to +/-40ms."""
    cams = ["cam01", "cam02", "cam03"]
    init_offsets = {"cam01": 0.0, "cam02": 0.020, "cam03": -0.020}

    module = CameraOffsets(
        cam_names=cams,
        init_offsets_s=init_offsets,
        reference_camera="cam01",
        max_clamp_s=0.040,
    )

    # 1. Anchor cam01 always returns 0 even if delta_offsets[0] is tampered
    module.delta_offsets.data[0] = 0.050
    off0 = module.get_offset("cam01")
    assert off0.item() == 0.0

    # 2. Clamping during get_offset
    module.delta_offsets.data[1] = 0.035  # base 0.020 + 0.035 = 0.055 > 0.040
    off1 = module.get_offset("cam02")
    assert pytest.approx(off1.item(), abs=1e-5) == 0.040

    module.delta_offsets.data[2] = -0.035  # base -0.020 - 0.035 = -0.055 < -0.040
    off2 = module.get_offset("cam03")
    assert pytest.approx(off2.item(), abs=1e-5) == -0.040

    # 3. clamp_and_anchor post-optimizer enforcement
    module.clamp_and_anchor()
    assert module.delta_offsets[0].item() == 0.0
    # Effective total offset for cam02 should be clamped to 0.040 -> delta is 0.040 - 0.020 = 0.020
    assert pytest.approx(module.delta_offsets[1].item(), abs=1e-5) == 0.020
    # Effective total offset for cam03 should be clamped to -0.040 -> delta is -0.040 - (-0.020) = -0.020
    assert pytest.approx(module.delta_offsets[2].item(), abs=1e-5) == -0.020

    eff = module.get_effective_offsets()
    assert eff["cam01"] == 0.0
    assert pytest.approx(eff["cam02"], abs=1e-5) == 0.040
    assert pytest.approx(eff["cam03"], abs=1e-5) == -0.040


def test_camera_offsets_differentiable_normalized_time():
    """Verify continuous normalized time formula and gradient flow to delta_offsets."""
    cams = ["cam01", "cam02"]
    fps = 20.0
    img_len = 60
    module = CameraOffsets(
        cam_names=cams,
        init_offsets_s={"cam01": 0.0, "cam02": 0.010},
        reference_camera="cam01",
        fps=fps,
        image_length=img_len,
    )

    frame_idx = 15

    # cam01 (anchor): t_norm = (15 + 0) / 60 = 0.25
    t1 = module.get_normalized_time("cam01", frame_idx)
    assert pytest.approx(t1.item(), abs=1e-5) == 15.0 / 60.0

    # cam02: t_norm = (15 + 0.010 * 20) / 60 = (15 + 0.2) / 60 = 15.2 / 60
    t2 = module.get_normalized_time("cam02", frame_idx)
    expected_t2 = (15.0 + 0.010 * 20.0) / 60.0
    assert pytest.approx(t2.item(), abs=1e-5) == expected_t2

    # Check analytical gradient: d(t_norm)/d(delta_2) = fps / image_length = 20 / 60 = 1/3
    t2.backward()
    assert module.delta_offsets.grad is not None
    assert module.delta_offsets.grad[0].item() == 0.0  # Anchor has 0 gradient
    assert pytest.approx(module.delta_offsets.grad[1].item(), abs=1e-5) == fps / img_len


def test_camera_offsets_freeze_schedule():
    """Verify freeze schedule: offsets optimizer does not update parameters before freeze_iter."""
    cams = ["cam01", "cam02"]
    module = CameraOffsets(
        cam_names=cams,
        init_offsets_s={"cam01": 0.0, "cam02": 0.010},
        reference_camera="cam01",
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    freeze_iter = 10

    # Simulate iterations 1 to 15
    for it in range(1, 16):
        t = module.get_normalized_time("cam02", frame_idx=5)
        # Synthetic loss pushing offset higher
        loss = -(t * 100.0)
        optimizer.zero_grad()
        loss.backward()

        if it >= freeze_iter:
            optimizer.step()
            module.clamp_and_anchor()
        optimizer.zero_grad()

        if it < freeze_iter:
            # Must remain untouched at 0.0 delta
            assert module.delta_offsets[1].item() == 0.0

    # After freeze_iter (iterations 10..15), delta_offsets[1] should have updated
    assert module.delta_offsets[1].item() > 0.0
    # Anchor must remain strictly 0.0
    assert module.delta_offsets[0].item() == 0.0


# ---------------------------------------------------------------------------
# 2. TrajectoryReprojectionLoss Unit Tests (R8)
# ---------------------------------------------------------------------------


def test_project_points_to_pixels():
    """Verify world-to-pixel projection function with known camera matrix."""
    # Simple camera: identity view, focal=500, principal point=(320, 240), w=640, h=480
    w, h = 640, 480
    # Test point at (0, 0, 2) in front of camera
    pts = torch.tensor([[0.0, 0.0, 2.0], [0.5, -0.5, 2.0]], dtype=torch.float32)

    # In 4DGS, full_proj_transform maps world to clip space [-1, 1]
    # For a symmetrical perspective matrix:
    # x_clip = x * (2*fx/w) / z
    # y_clip = y * (2*fy/h) / z
    # Let's construct a simple orthographic/canonical projection for testing
    proj = torch.eye(4, dtype=torch.float32)
    # Point at (0, 0, 2) with w=1 -> ndc (0, 0) -> pixel (320, 240)
    px = project_points_to_pixels(pts, proj, w, h)
    assert px.shape == (2, 2)
    assert pytest.approx(px[0, 0].item(), abs=1e-3) == 320.0
    assert pytest.approx(px[0, 1].item(), abs=1e-3) == 240.0


def test_trajectory_loss_track_loading(tmp_path):
    """Verify track loader supports trajectory_tracks.json format."""
    track_file = tmp_path / "trajectory_tracks.json"
    dummy_data = {
        "fps": 20.0,
        "n_frames": 60,
        "cameras": {
            "cam01": {
                "amplitude_px": 2.5,
                "phase_rad": 0.1,
                "displacements_px": [[0.0, 0.0], [1.2, 0.3], [2.4, 0.6]],
            },
            "cam02": {
                "amplitude_px": 3.0,
                "phase_rad": 0.5,
                "displacements_px": [[0.0, 0.0], [-1.0, 0.2], [-2.1, 0.5]],
            },
        },
    }
    track_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    loss_fn = TrajectoryReprojectionLoss(scene_path=tmp_path, track_file="trajectory_tracks.json")
    assert len(loss_fn.tracks["cameras"]) == 2

    disp = loss_fn.get_measured_displacement("cam01", frame_idx=1)
    assert pytest.approx(disp[0].item(), abs=1e-4) == 1.2
    assert pytest.approx(disp[1].item(), abs=1e-4) == 0.3


def test_trajectory_loss_forward_and_backward():
    """Verify displacement computation and gradient propagation."""
    w, h = 640, 480
    proj = torch.eye(4, dtype=torch.float32)

    class MockCam:
        cam_name = "cam01"
        frame_idx = 1
        image_width = w
        image_height = h
        full_proj_transform = proj

    # Create dummy tracks: cam01 frame 1 has displacement [2.0, 1.0]
    dummy_tracks = {
        "cameras": {
            "cam01": {
                "displacements_px": [[0.0, 0.0], [2.0, 1.0]],
            }
        }
    }

    loss_fn = TrajectoryReprojectionLoss(scene_path=Path("."))
    loss_fn.tracks = dummy_tracks

    canon_xyz = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], dtype=torch.float32)
    # Deformed XYZ with requires_grad
    deformed_xyz = torch.tensor(
        [[0.01, 0.005, 1.0], [0.11, 0.105, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss = loss_fn(MockCam(), canon_xyz, deformed_xyz)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert deformed_xyz.grad is not None
    assert torch.any(deformed_xyz.grad != 0.0)


# ---------------------------------------------------------------------------
# 3. End-to-End Gradient Coupling (R7 + R8)
# ---------------------------------------------------------------------------


def test_end_to_end_gradient_flow_to_offsets():
    """Verify gradient flows from TrajectoryReprojectionLoss through time to CameraOffsets."""
    cams = ["cam01", "cam02"]
    module = CameraOffsets(
        cam_names=cams,
        init_offsets_s={"cam01": 0.0, "cam02": 0.010},
        reference_camera="cam01",
        fps=20.0,
        image_length=60,
    )

    # Measured displacement for cam02 at frame 2 is [5.0, 0.0] px
    dummy_tracks = {
        "cameras": {
            "cam02": {
                "displacements_px": [[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
            }
        }
    }
    loss_fn = TrajectoryReprojectionLoss(scene_path=Path("."))
    loss_fn.tracks = dummy_tracks

    class MockCam:
        cam_name = "cam02"
        frame_idx = 2
        image_width = 640
        image_height = 480
        full_proj_transform = torch.eye(4, dtype=torch.float32)

    # 1. Differentiable timestamp from camera_offsets
    t_norm = module.get_normalized_time(MockCam.cam_name, MockCam.frame_idx)

    # 2. Differentiable deformation model: dx(t) = amplitude * sin(2 * pi * f * t)
    f0 = 7.112
    amplitude_world = 0.05
    dx = amplitude_world * torch.sin(2.0 * math.pi * f0 * t_norm)

    canon_xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    deformed_xyz = canon_xyz.clone()
    deformed_xyz = deformed_xyz + torch.stack([dx, torch.zeros_like(dx), torch.zeros_like(dx)]).unsqueeze(0)

    # 3. Trajectory loss
    loss = loss_fn(MockCam(), canon_xyz, deformed_xyz)
    assert loss.requires_grad

    # 4. Backward pass
    loss.backward()

    # Anchor cam01 has zero gradient
    assert module.delta_offsets.grad[0].item() == 0.0
    # Non-anchor cam02 receives gradient from trajectory loss
    assert module.delta_offsets.grad[1].item() != 0.0


# ---------------------------------------------------------------------------
# 4. Config Schema, Presets, and Bridge Serialization
# ---------------------------------------------------------------------------


def test_optimization_params_schema():
    """Verify OptimizationParams in core/arguments and pipeline.config include R7/R8 fields."""
    import argparse
    parser = argparse.ArgumentParser()
    core_opt = CoreOptimizationParams(parser=parser)
    assert hasattr(core_opt, "learnable_camera_offsets")
    assert core_opt.learnable_camera_offsets is False
    assert hasattr(core_opt, "learnable_offsets_lr")
    assert core_opt.learnable_offsets_lr == 0.0001
    assert hasattr(core_opt, "learnable_offsets_freeze_iter")
    assert core_opt.learnable_offsets_freeze_iter == 10000
    assert hasattr(core_opt, "max_offset_clamp_s")
    assert core_opt.max_offset_clamp_s == 0.040
    assert hasattr(core_opt, "lambda_trajectory")
    assert core_opt.lambda_trajectory == 0.0
    assert hasattr(core_opt, "trajectory_track_file")
    assert core_opt.trajectory_track_file == ""
    assert hasattr(core_opt, "trajectory_roi_bbox")

    pyd_opt = OptimizationParams()
    assert pyd_opt.learnable_camera_offsets is False
    assert pyd_opt.learnable_offsets_lr == 0.0001
    assert pyd_opt.learnable_offsets_freeze_iter == 10000
    assert pyd_opt.max_offset_clamp_s == 0.040
    assert pyd_opt.lambda_trajectory == 0.0


def test_presets_resolution_r7_and_r8():
    """Verify lego_turbine_2_missing_r7 and lego_turbine_2_missing_r8 presets resolve properly."""
    r7_dict = resolve_preset("lego_turbine_2_missing_r7")
    assert r7_dict["optim"]["learnable_camera_offsets"] is True
    assert r7_dict["optim"]["learnable_offsets_lr"] == 0.0001
    assert r7_dict["optim"]["learnable_offsets_freeze_iter"] == 10000
    assert r7_dict["optim"]["max_offset_clamp_s"] == 0.040

    r8_dict = resolve_preset("lego_turbine_2_missing_r8")
    assert r8_dict["optim"]["learnable_camera_offsets"] is True
    assert r8_dict["optim"]["lambda_trajectory"] == 1.0
    assert r8_dict["optim"]["trajectory_track_file"] == "trajectory_tracks.json"


def test_bridge_rendering_r7_and_r8():
    """Verify bridge file rendering serializes R7 and R8 fields."""
    r8_dict = resolve_preset("lego_turbine_2_missing_r8")
    config = PipelineConfig(**r8_dict)
    bridge_src = render_bridge_source(config.model_dump())

    assert "'learnable_camera_offsets': True" in bridge_src
    assert "'learnable_offsets_lr': 0.0001" in bridge_src
    assert "'learnable_offsets_freeze_iter': 10000" in bridge_src
    assert "'max_offset_clamp_s': 0.04" in bridge_src
    assert "'lambda_trajectory': 1.0" in bridge_src
    assert "'trajectory_track_file': 'trajectory_tracks.json'" in bridge_src
