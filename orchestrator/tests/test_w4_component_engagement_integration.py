"""orchestrator/tests/test_w4_component_engagement_integration.py

Integration test requiring enabled W4 motion components to actively engage,
receive non-zero finite gradients, and measurably update from their initialization
during an end-to-end training loop with densification and pruning.
"""

import math
from pathlib import Path
import sys
import types
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Mock Linux-only CUDA extension simple_knn if missing on Windows host
if "simple_knn" not in sys.modules:
    m_knn = types.ModuleType("simple_knn")
    m_knn_c = types.ModuleType("simple_knn._C")
    m_knn_c.distCUDA2 = lambda *args, **kwargs: None
    m_knn._C = m_knn_c
    sys.modules["simple_knn"] = m_knn
    sys.modules["simple_knn._C"] = m_knn_c

from scene.gaussian_model import GaussianModel
from scene.motion_components import (
    MotionComponents,
    FourierComponentTrajectory,
)


class MockHiddenArgs:
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


class MockTrainArgs:
    percent_dense = 0.01
    position_lr_init = 0.00016
    position_lr_final = 0.000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 1000
    deformation_lr_init = 0.00016
    deformation_lr_final = 0.000016
    deformation_lr_delay_mult = 0.01
    grid_lr_init = 0.0016
    grid_lr_final = 0.00016
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    motion_components_responsibilities_lr = 0.05
    motion_components_lr = 0.01


def test_w4_component_engagement_end_to_end():
    """Verify component parameters and responsibilities measurably change from initialization."""
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N = 40
    K = 1
    f0 = 5.690745
    fps = 20.004
    image_length = 261

    # 1. Initialize Gaussian Model
    gm = GaussianModel(sh_degree=0, args=MockHiddenArgs())
    gm.spatial_lr_scale = 1.0
    gm._xyz = nn.Parameter(torch.randn((N, 3), device=dev) * 0.1)
    gm._features_dc = nn.Parameter(torch.randn((N, 1, 3), device=dev))
    gm._features_rest = nn.Parameter(torch.zeros((N, 0, 3), device=dev))
    gm._scaling = nn.Parameter(torch.ones((N, 3), device=dev) * (-3.0))
    gm._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(N, 1))
    gm._opacity = nn.Parameter(torch.ones((N, 1), device=dev) * 0.5)
    gm._deformation_table = torch.ones(N, dtype=torch.bool, device=dev)
    gm.xyz_gradient_accum = torch.zeros((N, 1), device=dev)
    gm._deformation_accum = torch.zeros((N, 3), device=dev)
    gm.denom = torch.zeros((N, 1), device=dev)
    gm.max_radii2D = torch.zeros(N, device=dev)
    gm.percent_dense = 0.01

    # 2. Attach W4 Motion Components (ramp 5 -> 15 so it's fully active quickly)
    comp0 = FourierComponentTrajectory(
        base_frequency_hz=f0,
        fps=fps,
        image_length=image_length,
        num_harmonics=1,
        init_amplitude_m=0.001,
        init_dominant_axis=[0.0, 0.0, 1.0],
    )
    mc = MotionComponents(
        num_components=1,
        components=[comp0],
        ramp_start_iter=5,
        ramp_end_iter=15,
        lambda_res_max=1.0,
    )
    if dev.type == "cuda":
        mc = mc.cuda()

    init_logits = torch.full((N, K), -5.0, device=dev)
    # First half represents movers, second half static
    init_logits[:20, 0] = 1.0  # q1 ~ 0.73
    gm._motion_component_logits = nn.Parameter(init_logits.clone().requires_grad_(True))
    gm.motion_components = mc

    # 3. Setup optimizers
    train_args = MockTrainArgs()
    gm.training_setup(train_args)

    mc_optimizer = torch.optim.Adam(
        list(mc.parameters()),
        lr=train_args.motion_components_lr,
    )

    # Initial parameter snapshots
    initial_a_trans = comp0.a_trans[0].item()
    initial_logits = gm._motion_component_logits.clone().detach()

    # 4. Synthesize ground-truth moving signal on mover points: 15 mm vibration along z
    # Synthetic target displacement: 0.015 * cos(2 pi * f0 * t_sec) on movers
    T_total = 261
    duration_s = float(image_length) / fps

    # Run short training loop across iterations 1 to 25
    for iteration in range(1, 26):
        t_norm = ((iteration * 7) % T_total) / float(T_total)
        t_sec = t_norm * duration_s
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=dev).view(1, 1).repeat(gm.get_xyz.shape[0], 1)

        # Ground-truth target position
        target_disp = torch.zeros_like(gm._xyz)
        mover_count = 20
        # Mover displacement
        target_disp[:mover_count, 2] = 0.015 * math.cos(2.0 * math.pi * f0 * t_sec)
        p_target = gm._xyz.detach() + target_disp

        # Forward pass through motion components
        p_final, _ = mc.apply_deformation(
            pc=gm,
            means3D_canon=gm.get_xyz,
            means3D_hex=gm.get_xyz,  # static hex for integration test
            rotations_hex=gm._rotation,
            time=t_tensor,
            iteration=iteration,
        )

        # Reconstruction loss (mimicking photometric loss)
        loss_recon = torch.mean((p_final - p_target) ** 2)

        # Residual loss
        h_res = torch.zeros_like(gm._xyz)  # hex displacement is zero
        loss_res = mc.compute_residual_loss(h_res, gm, iteration)
        loss = loss_recon + loss_res

        loss.backward()

        # Instrumentation assertions during active ramp (iter > 15)
        if iteration >= 16:
            alpha = mc.get_alpha(iteration)
            assert alpha == 1.0, f"Alpha should be 1.0, got {alpha}"
            # Check component gradients
            assert comp0.a_trans.grad is not None, "a_trans gradient is None"
            assert torch.isfinite(comp0.a_trans.grad).all(), "a_trans gradient is non-finite"
            assert comp0.a_trans.grad.abs().item() > 1e-8, "a_trans gradient is zero"

            # Check logits gradients
            assert gm._motion_component_logits.grad is not None, "Logits gradient is None"
            assert torch.isfinite(gm._motion_component_logits.grad).all(), "Logits gradient is non-finite"
            assert gm._motion_component_logits.grad.abs().sum().item() > 1e-8, "Logits gradients are all zero"

        mc_optimizer.step()
        mc_optimizer.zero_grad()
        gm.optimizer.step()
        gm.optimizer.zero_grad()

        # At iteration 10, simulate densification/prune step
        if iteration == 10:
            grads = torch.ones_like(gm.get_xyz) * 0.1
            gm.densify_and_clone(grads, grad_threshold=0.05, scene_extent=1.0)
            assert gm._motion_component_logits.shape[0] == gm.get_xyz.shape[0]
            # Ensure optimizer still contains motion_component_logits
            has_mc_group = any(g.get("name") == "motion_component_logits" for g in gm.optimizer.param_groups)
            assert has_mc_group, "motion_component_logits param group lost during densification"

    # 5. Final Assertions: Components MUST have changed measurably from initialization
    final_a_trans = comp0.a_trans[0].item()
    a_delta = abs(final_a_trans - initial_a_trans)
    print(f"Component a_trans: initial={initial_a_trans*1000:.3f} mm, final={final_a_trans*1000:.3f} mm, delta={a_delta*1000:.3f} mm")
    assert a_delta > 1e-4, f"Component a_trans failed to update from initialization: delta={a_delta}"

    final_logits = gm._motion_component_logits[:N].clone().detach()
    logits_delta = torch.norm(final_logits - initial_logits).item()
    print(f"Logits L2 delta: {logits_delta:.5f}")
    assert logits_delta > 1e-3, f"Logits failed to update from initialization: delta={logits_delta}"

    print("ALL INTEGRATION ASSERTIONS PASSED: Components successfully engage and update.")


if __name__ == "__main__":
    test_w4_component_engagement_end_to_end()
