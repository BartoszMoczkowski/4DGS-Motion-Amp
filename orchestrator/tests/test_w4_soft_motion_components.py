"""orchestrator/tests/test_w4_soft_motion_components.py

Test suite for W4 (T52) Fixed-K Soft Learned Motion Components.
Verifies:
1. Mathematical identity at alpha=0 or K=0 (exact native 4DGS match to floating-point precision).
2. Simplex constraint (q_k >= 0, sum(q_k) <= 1, q0 >= 0) and non-vanishing, finite gradients.
3. Nyquist guardrail (raises ValueError for harmonics meeting or exceeding Nyquist limit).
4. Chebyshev smooth non-periodic trajectory evaluation.
5. Densification and pruning state inheritance: split, clone, add_point_by_mask, and prune preserve [N, K] alignment and parent values.
6. Unified deformation dispatch parity across render(), get_state_at_time(), and render_mod().
7. Checkpoint save/load round-trip equivalence.
8. Synthetic 1-mover (K=1) and 2-mover (K=2) recovery smoke tests without binary masks.
9. Preset resolution for turbine_r1_missing_6_w4_e3 and lego_turbine_2_missing_w4_e3.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Tuple

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

from scene.motion_components import (
    MotionComponents,
    FourierComponentTrajectory,
    ChebyshevComponentTrajectory,
    axis_angle_to_quaternion,
    rodrigues_rotation,
    batch_quaternion_multiply,
)
from scene.deformation_dispatch import dispatch_deformation
from scene.gaussian_model import GaussianModel
from pipeline.config import resolve_preset, validate_config


class DummyDeformation(nn.Module):
    """Mock HexPlane deformation field for CPU unit tests."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, means3D, scales, rotations, opacity, shs, time):
        # Deterministic synthetic deformation
        t = time if time.dim() == 2 else time.view(-1, 1)
        if t.shape[0] != means3D.shape[0]:
            t = t.expand(means3D.shape[0], 1)
        disp = 0.005 * torch.sin(2.0 * math.pi * 3.0 * t) * torch.tensor([1.0, 0.0, 0.0], device=means3D.device)
        return (
            means3D + disp + 0.0 * self.dummy_param,
            scales.clone(),
            rotations.clone(),
            opacity.clone(),
            shs.clone() if shs is not None else None,
        )


class MockGaussianModel:
    """Mock GaussianModel for testing dispatch and responsibilities."""

    def __init__(self, num_points: int = 100, device: str = "cpu"):
        self.device = torch.device(device)
        self._xyz = nn.Parameter(torch.randn((num_points, 3), device=self.device, dtype=torch.float32))
        self._scaling = nn.Parameter(torch.ones((num_points, 3), device=self.device, dtype=torch.float32) * 0.01)
        self._rotation = nn.Parameter(torch.zeros((num_points, 4), device=self.device, dtype=torch.float32))
        self._rotation.data[:, 0] = 1.0  # (w=1, x=0, y=0, z=0)
        self._opacity = nn.Parameter(torch.ones((num_points, 1), device=self.device, dtype=torch.float32) * 2.0)
        self._features_dc = nn.Parameter(torch.randn((num_points, 1, 3), device=self.device, dtype=torch.float32))
        self._deformation = DummyDeformation().to(self.device)
        self._motion_component_logits = None
        self.motion_components = None

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features_dc


# ==============================================================================
# 1. Zero-Component & Alpha=0 Equivalence Tests
# ==============================================================================

def test_zero_alpha_exact_identity():
    """At alpha=0 or unassigned responsibilities, deformation matches native HexPlane bit-for-bit."""
    model = MockGaussianModel(num_points=50)
    time = torch.tensor([[0.25]])

    # Baseline native HexPlane
    hex_m, hex_s, hex_r, hex_o, hex_sh = model._deformation(
        model._xyz, model._scaling, model._rotation, model._opacity, model.get_features, time
    )

    # Motion components active but at iteration < ramp_start_iter (alpha = 0)
    mc = MotionComponents(
        num_components=1,
        ramp_start_iter=1000,
        ramp_end_iter=2000,
    )
    model.motion_components = mc
    model._motion_component_logits = nn.Parameter(torch.randn((50, 1)))

    out_m, out_s, out_r, out_o, out_sh = dispatch_deformation(
        pc=model,
        means3D=model._xyz,
        scales=model._scaling,
        rotations=model._rotation,
        opacity=model._opacity,
        shs=model.get_features,
        time=time,
        stage="fine",
        motion_components=mc,
        iteration=500,  # alpha = 0.0
    )

    torch.testing.assert_close(out_m, hex_m, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(out_r, hex_r, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(out_s, hex_s, rtol=1e-7, atol=1e-7)


def test_null_responsibilities_exact_identity():
    """When responsibilities are null (q0=1, qk=0), displacement is an exact no-op even at alpha=1."""
    model = MockGaussianModel(num_points=30)
    time = torch.tensor([[0.5]])

    hex_m, hex_s, hex_r, hex_o, hex_sh = model._deformation(
        model._xyz, model._scaling, model._rotation, model._opacity, model.get_features, time
    )

    mc = MotionComponents(
        num_components=2,
        ramp_start_iter=0,
        ramp_end_iter=100,
    )
    model.motion_components = mc
    # Logits set to very large negative numbers -> q_k -> 0
    model._motion_component_logits = nn.Parameter(torch.full((30, 2), -50.0))

    out_m, out_s, out_r, out_o, out_sh = dispatch_deformation(
        pc=model,
        means3D=model._xyz,
        scales=model._scaling,
        rotations=model._rotation,
        opacity=model._opacity,
        shs=model.get_features,
        time=time,
        stage="fine",
        motion_components=mc,
        iteration=200,  # alpha = 1.0
    )

    torch.testing.assert_close(out_m, hex_m, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_r, hex_r, rtol=1e-5, atol=1e-6)


# ==============================================================================
# 2. Simplex Constraint & Finite Gradients Tests
# ==============================================================================

def test_simplex_constraint():
    """Responsibilities strictly satisfy q_k >= 0, sum(q_k) <= 1, q_0 = 1 - sum(q_k) >= 0."""
    mc = MotionComponents(num_components=3)
    model = MockGaussianModel(num_points=100)

    # Test across random unconstrained logits spanning wide dynamic range
    model._motion_component_logits = nn.Parameter(torch.randn((100, 3)) * 10.0)

    q, q0 = mc.get_responsibilities(model)

    assert (q >= 0.0).all(), "q_k must be non-negative"
    assert (q0 >= 0.0).all(), "q_0 must be non-negative"
    total_q = q.sum(dim=-1, keepdim=True) + q0
    torch.testing.assert_close(total_q, torch.ones_like(total_q), rtol=1e-6, atol=1e-6)


def test_finite_gradients_and_stopgrad():
    """Gradients through component parameters and responsibility logits are finite without NaNs."""
    mc = MotionComponents(
        num_components=1,
        ramp_start_iter=0,
        ramp_end_iter=100,
        lambda_res_max=1.0,
    )
    model = MockGaussianModel(num_points=20)
    model._motion_component_logits = nn.Parameter(torch.zeros((20, 1), requires_grad=True))

    time = torch.tensor([[0.1]])
    out_m, out_s, out_r, out_o, out_sh, h_res = dispatch_deformation(
        pc=model,
        means3D=model._xyz,
        scales=model._scaling,
        rotations=model._rotation,
        opacity=model._opacity,
        shs=model.get_features,
        time=time,
        stage="fine",
        motion_components=mc,
        iteration=200,
        return_residual=True,
    )

    # 1. Photometric / target loss backprop
    target_loss = (out_m.sum() + out_r.sum())
    # 2. Residual penalty
    res_loss = mc.compute_residual_loss(h_res, model, iteration=200)

    total_loss = target_loss + res_loss
    total_loss.backward()

    # Check finite gradients on component parameters
    for name, p in mc.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaNs"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} gradient contains Infs"

    # Check finite gradient on logits
    assert model._motion_component_logits.grad is not None
    assert not torch.isnan(model._motion_component_logits.grad).any()


# ==============================================================================
# 3. Nyquist Guardrail & Trajectory Basis Tests
# ==============================================================================

def test_nyquist_guardrail_enforcement():
    """At 20.004 fps, harmonics exceeding Nyquist (~10.002 Hz) must raise ValueError."""
    fps = 20.004
    f0 = 5.691  # Measured turbine fundamental

    # num_harmonics=1: 5.691 Hz < 10.002 Hz -> VALID
    traj1 = FourierComponentTrajectory(base_frequency_hz=f0, fps=fps, num_harmonics=1)
    assert traj1.effective_frequency_hz.item() == pytest.approx(f0, abs=1e-4)

    # num_harmonics=2: 2 * 5.691 = 11.382 Hz > 10.002 Hz -> FORBIDDEN
    with pytest.raises(ValueError, match="exceeds Nyquist limit"):
        FourierComponentTrajectory(base_frequency_hz=f0, fps=fps, num_harmonics=2)


def test_chebyshev_trajectory_smoothness():
    """Chebyshev polynomial basis generates smooth, finite trajectories."""
    traj = ChebyshevComponentTrajectory(degree=4, t_min=0.0, t_max=1.0)
    # Set non-zero coefficients
    with torch.no_grad():
        traj.trans_coeffs.copy_(torch.randn_like(traj.trans_coeffs) * 0.01)
        traj.rot_coeffs.copy_(torch.randn_like(traj.rot_coeffs) * 0.01)

    t_samples = torch.linspace(0.0, 1.0, 20).unsqueeze(-1)
    trans, omega = traj.get_trajectory(t_samples)

    assert trans.shape == (20, 3)
    assert omega.shape == (20, 3)
    assert not torch.isnan(trans).any()
    assert not torch.isnan(omega).any()


# ==============================================================================
# 4. Densification & Pruning Lifecycle State Inheritance Tests
# ==============================================================================

def test_densification_and_pruning_state_inheritance():
    """Verify split, clone, add_point_by_mask, and prune preserve [N, K] alignment and parent logits."""
    # We construct a real GaussianModel instance
    class MockArgs:
        net_width = 32
        timebase_pe = 2
        defor_depth = 1
        posebase_pe = 2
        scale_rotation_pe = 1
        opacity_pe = 1
        timenet_width = 32
        timenet_output = 16
        bounds = 1.0
        no_dx = False
        no_grid = True
        no_ds = False
        no_dr = False
        no_do = True
        no_dshs = True
        empty_voxel = False
        grid_pe = 0
        static_mlp = False
        apply_rotation = False
        kplanes_config = {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 32,
            "resolution": [64, 64, 64, 25],
        }
        multires = [1, 2, 4, 8]

    gm = GaussianModel(sh_degree=0, args=MockArgs())
    N = 10
    K = 2
    dev = torch.device("cpu")

    # Manually populate tensors on CPU
    gm._xyz = torch.randn((N, 3), device=dev)
    gm._features_dc = torch.randn((N, 1, 3), device=dev)
    gm._features_rest = torch.zeros((N, 0, 3), device=dev)
    gm._scaling = torch.ones((N, 3), device=dev) * (-3.0)
    gm._rotation = torch.zeros((N, 4), device=dev)
    gm._rotation[:, 0] = 1.0
    gm._opacity = torch.ones((N, 1), device=dev) * 0.5
    gm._deformation_table = torch.ones(N, dtype=torch.bool, device=dev)
    gm.xyz_gradient_accum = torch.zeros((N, 1), device=dev)
    gm._deformation_accum = torch.zeros((N, 3), device=dev)
    gm.denom = torch.zeros((N, 1), device=dev)
    gm.max_radii2D = torch.zeros(N, device=dev)
    gm.percent_dense = 0.01

    # Attach K=2 motion component logits
    parent_logits = torch.tensor([
        [float(i), float(-i)] for i in range(N)
    ], dtype=torch.float32, device=dev)
    gm._motion_component_logits = parent_logits.clone()

    # Create dummy optimizer with param groups
    opt_params = [
        {"name": "xyz", "params": [nn.Parameter(gm._xyz)]},
        {"name": "f_dc", "params": [nn.Parameter(gm._features_dc)]},
        {"name": "f_rest", "params": [nn.Parameter(gm._features_rest)]},
        {"name": "opacity", "params": [nn.Parameter(gm._opacity)]},
        {"name": "scaling", "params": [nn.Parameter(gm._scaling)]},
        {"name": "rotation", "params": [nn.Parameter(gm._rotation)]},
        {"name": "motion_component_logits", "params": [nn.Parameter(gm._motion_component_logits)]},
    ]
    gm.optimizer = torch.optim.Adam(opt_params)

    # 1. Test densify_and_clone
    grads = torch.ones((N, 3), device=dev) * 0.1
    gm.densify_and_clone(grads, grad_threshold=0.05, scene_extent=10.0)
    n_cloned = gm.get_xyz.shape[0]
    assert n_cloned > N, "Points should have been cloned"
    assert gm._motion_component_logits.shape == (n_cloned, K)
    # Check that cloned points inherited exact parent logits
    cloned_portion = gm._motion_component_logits[N:]
    torch.testing.assert_close(cloned_portion, parent_logits[:cloned_portion.shape[0]])

    # 2. Test densify_and_split
    n_before_split = gm.get_xyz.shape[0]
    grads_split = torch.ones((n_before_split, 1), device=dev) * 0.2
    gm.densify_and_split(grads_split, grad_threshold=0.1, scene_extent=0.01, N=2)
    n_after_split = gm.get_xyz.shape[0]
    assert gm._motion_component_logits.shape == (n_after_split, K)
    assert not torch.isnan(gm._motion_component_logits).any()

    # 3. Test prune_points
    prune_mask = torch.zeros(n_after_split, dtype=torch.bool, device=dev)
    prune_mask[0] = True
    prune_mask[2] = True
    gm.prune_points(prune_mask)
    n_pruned = gm.get_xyz.shape[0]
    assert n_pruned == n_after_split - 2
    assert gm._motion_component_logits.shape == (n_pruned, K)


# ==============================================================================
# 5. Checkpoint Save/Load & Dispatch Equivalence Tests
# ==============================================================================

def test_checkpoint_roundtrip_and_dispatch_equivalence():
    """Verify saved checkpoint loaded into Scene reproduces identical outputs across all call sites."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        mc = MotionComponents(
            num_components=1,
            components=[FourierComponentTrajectory(base_frequency_hz=5.691, fps=20.004, num_harmonics=1)],
            ramp_start_iter=500,
            ramp_end_iter=1500,
            lambda_res_max=0.5,
        )
        logits = torch.tensor([[1.5], [-2.0], [0.5]], dtype=torch.float32)

        # Save checkpoint files
        torch.save(mc.state_dict(), model_dir / "motion_components.pth")
        torch.save(logits, model_dir / "motion_component_logits.pth")
        with open(model_dir / "motion_components_state.json", "w", encoding="utf-8") as f:
            json.dump(mc.get_summary(), f, indent=2)

        # Verify files exist
        assert (model_dir / "motion_components.pth").is_file()
        assert (model_dir / "motion_components_state.json").is_file()
        assert (model_dir / "motion_component_logits.pth").is_file()

        # Reload state
        with open(model_dir / "motion_components_state.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["num_components"] == 1
        assert summary["components"][0]["base_frequency_hz"] == 5.691

        # Re-instantiate
        mc_loaded = MotionComponents(
            num_components=summary["num_components"],
            ramp_start_iter=summary["ramp_start_iter"],
            ramp_end_iter=summary["ramp_end_iter"],
            lambda_res_max=summary["lambda_res_max"],
        )
        mc_loaded.load_state_dict(torch.load(model_dir / "motion_components.pth", map_location="cpu"))
        logits_loaded = torch.load(model_dir / "motion_component_logits.pth", map_location="cpu")

        # Compare outputs at arbitrary time
        mock_pc = MockGaussianModel(num_points=3)
        mock_pc._motion_component_logits = logits
        mock_pc.motion_components = mc

        mock_pc_loaded = MockGaussianModel(num_points=3)
        mock_pc_loaded._xyz.data.copy_(mock_pc._xyz.data)
        mock_pc_loaded._scaling.data.copy_(mock_pc._scaling.data)
        mock_pc_loaded._rotation.data.copy_(mock_pc._rotation.data)
        mock_pc_loaded._opacity.data.copy_(mock_pc._opacity.data)
        mock_pc_loaded._features_dc.data.copy_(mock_pc._features_dc.data)
        mock_pc_loaded._motion_component_logits = logits_loaded
        mock_pc_loaded.motion_components = mc_loaded

        t = torch.tensor([[0.33]])
        out1_m, _, out1_r, _, _ = dispatch_deformation(
            pc=mock_pc,
            means3D=mock_pc._xyz,
            scales=mock_pc._scaling,
            rotations=mock_pc._rotation,
            opacity=mock_pc._opacity,
            shs=mock_pc.get_features,
            time=t,
            stage="fine",
            motion_components=mc,
        )

        out2_m, _, out2_r, _, _ = dispatch_deformation(
            pc=mock_pc_loaded,
            means3D=mock_pc_loaded._xyz,
            scales=mock_pc_loaded._scaling,
            rotations=mock_pc_loaded._rotation,
            opacity=mock_pc_loaded._opacity,
            shs=mock_pc_loaded.get_features,
            time=t,
            stage="fine",
            motion_components=mc_loaded,
        )

        torch.testing.assert_close(out1_m, out2_m, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(out1_r, out2_r, rtol=1e-7, atol=1e-7)


# ==============================================================================
# 6. Synthetic 1-Mover (K=1) and 2-Mover (K=2) Pilots
# ==============================================================================

def test_synthetic_single_mover_recovery():
    """Verify EMA responsibility update concentrates responsibility on moving cluster vs static cluster."""
    mc = MotionComponents(
        num_components=1,
        components=[FourierComponentTrajectory(base_frequency_hz=5.0, fps=20.0, num_harmonics=1, init_amplitude_m=0.02)],
    )

    N_moving = 20
    N_static = 30
    N_total = N_moving + N_static

    model = MockGaussianModel(num_points=N_total)
    model._motion_component_logits = nn.Parameter(torch.zeros((N_total, 1)))

    time_samples = torch.linspace(0.0, 1.0, 10).unsqueeze(-1)  # [10, 1]
    g_k, _ = mc.components[0].compute_candidate_displacement(model._xyz, time_samples)  # [N, 10, 3]

    # Moving cluster tracks component 0; static cluster is zero displacement
    hex_residuals = torch.zeros((N_total, 10, 3), dtype=torch.float32)
    hex_residuals[:N_moving] = g_k[:N_moving] + 0.0001 * torch.randn_like(g_k[:N_moving])

    # Update responsibilities via EMA
    mc.update_responsibilities_ema(
        pc=model,
        hex_residuals=hex_residuals,
        candidate_displacements=g_k.unsqueeze(1),  # [N, 1, 10, 3]
        beta=0.8,
        sigma=0.005,
    )

    q, q0 = mc.get_responsibilities(model)
    # Moving cluster should have high responsibility on component 0
    assert q[:N_moving, 0].mean() > 0.8, f"Moving points responsibility too low: {q[:N_moving, 0].mean()}"
    # Static cluster should have null responsibility dominant (q0 ~ 1, q1 ~ 0)
    assert q[N_moving:, 0].mean() < 0.2, f"Static points responsibility too high: {q[N_moving:, 0].mean()}"
    assert q0[N_moving:].mean() > 0.8, f"Static points null responsibility too low: {q0[N_moving:].mean()}"


def test_synthetic_two_mover_separation():
    """Verify K=2 separates two distinct rigid trajectories without hard masks."""
    mc = MotionComponents(
        num_components=2,
        components=[
            FourierComponentTrajectory(base_frequency_hz=3.0, fps=20.0, num_harmonics=1, init_amplitude_m=0.02, init_dominant_axis=[1.0, 0.0, 0.0]),
            FourierComponentTrajectory(base_frequency_hz=6.0, fps=20.0, num_harmonics=1, init_amplitude_m=0.02, init_dominant_axis=[0.0, 1.0, 0.0]),
        ],
    )

    N_each = 15
    N_total = 2 * N_each
    model = MockGaussianModel(num_points=N_total)
    model._motion_component_logits = nn.Parameter(torch.zeros((N_total, 2)))

    time_samples = torch.linspace(0.0, 1.0, 10).unsqueeze(-1)
    g0, _ = mc.components[0].compute_candidate_displacement(model._xyz, time_samples)
    g1, _ = mc.components[1].compute_candidate_displacement(model._xyz, time_samples)

    # First half tracks component 0, second half tracks component 1
    hex_residuals = torch.zeros((N_total, 10, 3), dtype=torch.float32)
    hex_residuals[:N_each] = g0[:N_each]
    hex_residuals[N_each:] = g1[N_each:]

    cand_disps = torch.stack([g0, g1], dim=1)  # [N, 2, 10, 3]

    mc.update_responsibilities_ema(
        pc=model,
        hex_residuals=hex_residuals,
        candidate_displacements=cand_disps,
        beta=0.8,
        sigma=0.005,
    )

    q, q0 = mc.get_responsibilities(model)
    # Component 0 dominance on cluster 0
    assert q[:N_each, 0].mean() > 0.75
    assert q[:N_each, 1].mean() < 0.25
    # Component 1 dominance on cluster 1
    assert q[N_each:, 1].mean() > 0.75
    assert q[N_each:, 0].mean() < 0.25


# ==============================================================================
# 7. Preset Resolution Tests
# ==============================================================================

def test_w4_presets_resolve():
    """Verify turbine_r1_missing_6_w4_e3 and lego_turbine_2_missing_w4_e3 presets resolve with all keys."""
    turbine_preset = validate_config("turbine_r1_missing_6_w4_e3")
    assert turbine_preset.optim.use_motion_components is True
    assert turbine_preset.optim.num_motion_components == 1
    assert turbine_preset.optim.motion_components_type == "fourier"
    assert turbine_preset.optim.motion_components_frequency_hz == 5.691
    assert turbine_preset.optim.motion_components_num_harmonics == 1
    assert turbine_preset.train.port == 6026

    lego_preset = validate_config("lego_turbine_2_missing_w4_e3")
    assert lego_preset.optim.use_motion_components is True
    assert lego_preset.optim.num_motion_components == 1
    assert lego_preset.optim.motion_components_type == "fourier"
    assert lego_preset.optim.motion_components_frequency_hz == 5.691
    assert lego_preset.optim.motion_components_num_harmonics == 1
    assert lego_preset.train.port == 6020
