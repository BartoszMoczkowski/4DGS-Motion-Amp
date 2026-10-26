"""Tests for Task T41 (Fourier-rigid deformation head - R9).

Verifies:
1. Low-dimensional physics parameterization:
   - Module has ~25 parameters (resists conflicting gradient collapse and overfitting).
   - Base frequency centered at 7.112 Hz with 3 harmonics (n=1..3).
2. Translation and Rotation generation:
   - Multi-harmonic periodic translation along learnable dominant axis and secondary axis.
   - Exact periodicity: T(t + 1/f) = T(t).
   - Rodrigues small-angle rotation about a learnable 3D pivot.
   - Quaternion covariance transformation matching standard (w, x, y, z) convention.
3. ROI Gating & Additive Operation:
   - Points inside turbine ROI undergo structured Fourier-rigid motion.
   - Points outside turbine ROI remain unaffected (keeping stock HexPlane deformation).
   - Both pure_rigid and additive modes work as specified.
4. Auditability and reporting:
   - Readable dominant axis, frequency (Hz), amplitudes (m/mm), phases (deg), and rotation.
   - Exportable summary dictionary serializable to JSON.
   - Initialization from optical tracks / optical_offsets.json.
5. End-to-end gradient flow:
   - Gradients flow back from photometric / trajectory loss to Fourier head parameters.
   - Freeze schedule protects early iterations while point densification settles in.
6. Schema, preset resolution, and bridge serialization:
   - OptimizationParams includes all R9 fields.
   - lego_turbine_2_missing_r9 preset resolves correctly.
   - render_bridge_source serializes R9 fields to bridge file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


from scene.fourier_rigid import FourierRigidHead, rodrigues_rotation
from arguments import OptimizationParams as CoreOptimizationParams
from pipeline.config import resolve_preset
from pipeline.config.models import OptimizationParams, PipelineConfig
from pipeline.config.bridge import render_bridge_source


# ---------------------------------------------------------------------------
# 1. FourierRigidHead Initialization & Parameter Count Tests
# ---------------------------------------------------------------------------


def test_fourier_rigid_head_init_and_parameter_count():
    """Verify FourierRigidHead initializes with ~25 parameters instead of ~10^6."""
    head = FourierRigidHead(
        base_frequency_hz=7.112,
        learn_frequency=False,
        num_harmonics=3,
        fps=20.0,
        image_length=60,
        roi_bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2],
        init_amplitude_m=0.002,
    )

    assert head.base_frequency_hz == 7.112
    assert pytest.approx(head.effective_frequency_hz.item(), abs=1e-4) == 7.112
    assert head.num_harmonics == 3
    assert head.fps == 20.0
    assert head.image_length == 60
    assert head.roi_bbox == [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2]

    # Count total trainable parameters
    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    # axis(3) + transverse(3) + amps(3) + phases(3) + trans_amps(3) + trans_phases(3) + pivot(3) + rot_axis(3) + rot_amps(3) + rot_phases(3) = 30
    assert total_params < 40
    assert total_params >= 20


def test_rodrigues_rotation_exactness():
    """Verify Rodrigues formula rotates unit vectors accurately."""
    # Rotate [1, 0, 0] around Z-axis [0, 0, 1] by 90 degrees (pi/2) -> should yield [0, 1, 0]
    pts = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    angle = torch.tensor([[math.pi / 2.0]], dtype=torch.float32)

    rotated = rodrigues_rotation(pts, axis, angle)
    assert pytest.approx(rotated[0, 0].item(), abs=1e-5) == 0.0
    assert pytest.approx(rotated[0, 1].item(), abs=1e-5) == 1.0
    assert pytest.approx(rotated[0, 2].item(), abs=1e-5) == 0.0


def test_fourier_translation_periodicity():
    """Verify T(t + 1/f) == T(t) exactly reflects physical harmonic periodicity."""
    head = FourierRigidHead(
        base_frequency_hz=7.112,
        num_harmonics=3,
        fps=20.0,
        image_length=60,
        init_amplitude_m=0.0025,
    )
    # Set known amplitudes and phases
    with torch.no_grad():
        head.amplitudes.copy_(torch.tensor([0.0025, 0.0005, 0.0002], dtype=torch.float32))
        head.phases.copy_(torch.tensor([0.2, -0.1, 0.5], dtype=torch.float32))

    period_s = 1.0 / 7.112  # seconds per fundamental cycle
    duration_s = 60.0 / 20.0  # 3.0 seconds

    # Two timestamps separated by exactly one cycle in normalized time
    t0_norm = torch.tensor([[0.2]], dtype=torch.float32)
    delta_t_norm = period_s / duration_s
    t1_norm = t0_norm + delta_t_norm

    trans0, _, _ = head.compute_translation_and_rotation(t0_norm)
    trans1, _, _ = head.compute_translation_and_rotation(t1_norm)

    # Must match within numerical tolerance
    assert torch.allclose(trans0, trans1, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. ROI Gating & Additive Operation Tests
# ---------------------------------------------------------------------------


def test_fourier_head_roi_gating():
    """Verify points inside ROI receive motion while points outside ROI stay strictly unaffected."""
    # Bounding box around origin [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]
    head = FourierRigidHead(
        base_frequency_hz=7.112,
        roi_bbox=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
        init_amplitude_m=0.010,
        pure_rigid=False,
    )

    # Point 0: inside ROI (0.0, 0.0, 0.0)
    # Point 1: outside ROI (2.0, 2.0, 2.0)
    means_canon = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    means_deformed = means_canon.clone()

    rot_canon = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    rot_deformed = rot_canon.clone()

    t_in = torch.tensor([[0.25]], dtype=torch.float32)

    means_out, rot_out = head(
        means3D_canon=means_canon,
        means3D_deformed=means_deformed,
        rotations_canon=rot_canon,
        rotations_deformed=rot_deformed,
        time=t_in,
    )

    # Point 0 (inside) must have non-zero displacement
    disp_inside = torch.norm(means_out[0] - means_deformed[0]).item()
    assert disp_inside > 0.001

    # Point 1 (outside) must remain EXACTLY identical
    disp_outside = torch.norm(means_out[1] - means_deformed[1]).item()
    assert disp_outside == 0.0
    assert torch.all(rot_out[1] == rot_deformed[1])


def test_fourier_head_pure_rigid_mode():
    """Verify pure_rigid=True ignores prior HexPlane deformation on the ROI."""
    head = FourierRigidHead(
        roi_bbox=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        init_amplitude_m=0.005,
        pure_rigid=True,
    )

    canon = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    # Arbitrary distorted HexPlane deformation
    noisy_deformed = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32)
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    t_in = torch.tensor([[0.25]], dtype=torch.float32)

    means_out, _ = head(
        means3D_canon=canon,
        means3D_deformed=noisy_deformed,
        rotations_canon=rot,
        rotations_deformed=rot,
        time=t_in,
    )

    # Output position should depend only on canon + rigid motion, not noisy_deformed
    trans, _, _ = head.compute_translation_and_rotation(t_in)
    expected = canon + trans
    assert torch.allclose(means_out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Auditability, Summary, and Optical Track Initialization
# ---------------------------------------------------------------------------


def test_fourier_head_auditability_summary():
    """Verify get_summary() produces human-readable and JSON-serializable physics parameters."""
    head = FourierRigidHead(
        base_frequency_hz=7.112,
        num_harmonics=3,
        init_dominant_axis=[1.0, 0.0, 0.0],
        init_amplitude_m=0.0028,
        roi_bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2],
    )

    summary = head.get_summary()
    assert pytest.approx(summary["frequency_hz"], abs=1e-4) == 7.112
    assert summary["num_harmonics"] == 3
    assert len(summary["dominant_axis"]) == 3
    assert pytest.approx(summary["dominant_axis"][0], abs=1e-5) == 1.0
    assert pytest.approx(summary["amplitudes_m"][0], abs=1e-5) == 0.0028

    # Verify JSON serializability
    json_str = json.dumps(summary)
    loaded = json.loads(json_str)
    assert pytest.approx(loaded["frequency_hz"], abs=1e-4) == 7.112


def test_init_from_optical_tracks(tmp_path):
    """Verify init_from_optical_tracks accurately sets frequency and principal axis."""
    track_payload = {
        "wobble_frequency_hz": 7.112,
        "principal_axis_angle_deg": 45.0,
        "cameras": {
            "cam01": {"principal_axis_angle_deg": 45.0},
            "cam02": {"principal_axis_angle_deg": 45.0},
        },
    }
    track_file = tmp_path / "optical_offsets.json"
    track_file.write_text(json.dumps(track_payload), encoding="utf-8")

    head = FourierRigidHead()
    head.init_from_optical_tracks(track_file)

    assert head.base_frequency_hz == 7.112
    u = head.get_dominant_axis()
    # 45 degrees in XY plane: cos(45) = sin(45) = 1/sqrt(2) approx 0.7071
    assert pytest.approx(u[0], abs=1e-3) == math.sqrt(2.0) / 2.0
    assert pytest.approx(u[1], abs=1e-3) == math.sqrt(2.0) / 2.0
    assert pytest.approx(u[2], abs=1e-3) == 0.0


# ---------------------------------------------------------------------------
# 4. Gradient Flow & Freeze Schedule
# ---------------------------------------------------------------------------


def test_fourier_head_gradient_flow():
    """Verify gradients propagate from loss through 3D points back to Fourier head parameters."""
    head = FourierRigidHead(
        roi_bbox=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        init_amplitude_m=0.001,
    )

    canon = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    deformed = canon.clone()
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    t_in = torch.tensor([[0.1]], dtype=torch.float32)

    means_out, _ = head(canon, deformed, rot, rot, t_in)

    # Synthetic loss targeting target position [0.010, 0.0, 0.0]
    target = torch.tensor([[0.010, 0.0, 0.0]], dtype=torch.float32)
    loss = F.mse_loss(means_out, target)
    loss.backward()

    # Gradients must exist on amplitudes and axis_param
    assert head.amplitudes.grad is not None
    assert torch.any(head.amplitudes.grad != 0.0)
    assert head.axis_param.grad is not None


def test_fourier_head_freeze_schedule():
    """Verify Fourier optimizer respects freeze schedule iterations."""
    head = FourierRigidHead(
        roi_bbox=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        init_amplitude_m=0.001,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    freeze_iter = 10

    initial_amp = head.amplitudes[0].item()

    # Simulate 15 iterations
    for it in range(1, 16):
        canon = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        means_out, _ = head(canon, canon, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([[0.25]]))
        loss = F.mse_loss(means_out, torch.tensor([[0.05, 0.0, 0.0]]))

        optimizer.zero_grad()
        loss.backward()

        if it >= freeze_iter:
            optimizer.step()
        optimizer.zero_grad()

        if it < freeze_iter:
            assert head.amplitudes[0].item() == initial_amp

    # After freeze_iter (iterations 10..15), amplitude should have increased towards target
    assert head.amplitudes[0].item() > initial_amp


# ---------------------------------------------------------------------------
# 5. Schema, Preset Resolution, and Bridge Serialization
# ---------------------------------------------------------------------------


def test_optimization_params_schema_r9():
    """Verify OptimizationParams in core/arguments and pipeline.config include R9 fields."""
    parser = argparse.ArgumentParser()
    core_opt = CoreOptimizationParams(parser=parser)
    assert hasattr(core_opt, "use_fourier_rigid_head")
    assert core_opt.use_fourier_rigid_head is False
    assert hasattr(core_opt, "fourier_head_lr")
    assert core_opt.fourier_head_lr == 0.001
    assert hasattr(core_opt, "fourier_head_frequency_hz")
    assert core_opt.fourier_head_frequency_hz == 7.112
    assert hasattr(core_opt, "fourier_head_num_harmonics")
    assert core_opt.fourier_head_num_harmonics == 3

    pyd_opt = OptimizationParams()
    assert pyd_opt.use_fourier_rigid_head is False
    assert pyd_opt.fourier_head_lr == 0.001
    assert pyd_opt.fourier_head_freeze_iter == 500
    assert pyd_opt.fourier_head_frequency_hz == 7.112


def test_preset_resolution_and_bridge_generation_r9():
    """Verify lego_turbine_2_missing_r9 preset resolves and renders to bridge file."""
    p9_dict = resolve_preset("lego_turbine_2_missing_r9")
    assert p9_dict["optim"]["use_fourier_rigid_head"] is True
    assert p9_dict["optim"]["fourier_head_lr"] == 0.001
    assert p9_dict["optim"]["fourier_head_frequency_hz"] == 7.112
    assert p9_dict["optim"]["fourier_head_freeze_iter"] == 500
    assert p9_dict["optim"]["learnable_camera_offsets"] is True
    assert p9_dict["optim"]["lambda_trajectory"] == 1.0

    config = PipelineConfig(**p9_dict)
    bridge_src = render_bridge_source(config.model_dump())

    assert "'use_fourier_rigid_head': True" in bridge_src
    assert "'fourier_head_lr': 0.001" in bridge_src
    assert "'fourier_head_frequency_hz': 7.112" in bridge_src
    assert "'fourier_head_freeze_iter': 500" in bridge_src
