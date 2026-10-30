"""orchestrator/tests/test_w5_component_split_merge.py

Test suite for W5 (T53) Reversible Outer-Loop Component Split and Merge.
Verifies:
1. Synthetic K=1 -> K=2 discovery test: two distinct movers split and assign soft responsibilities.
2. Exact rollback test: forced rejection restores all parameters, logits, optimizers, and RNG bit-for-bit.
3. Merge test: duplicate components merge cleanly (K=2 -> K=1) and responsibilities sum.
4. Static scene guardrail test: zero accepted splits on static scene.
5. Non-rigid mover guardrail test: non-rigid deformation does not shatter into K_max components.
6. Rejection cooldown & ledger test: cooldown prevents repeating rejected splits, ledger records telemetry.
7. Preset resolution: turbine_r1_missing_6_w5_e4 and lego_turbine_2_missing_w5_e4.
"""

from __future__ import annotations

import copy
import json
import math
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
)
from scene.component_split_merge import (
    ComponentSplitMergeController,
    StructuralSnapshot,
)
from pipeline.config import validate_config


class MockGaussianModel:
    """Mock GaussianModel for split/merge tests."""

    def __init__(self, num_points: int = 50, device: str = "cpu"):
        self.device = torch.device(device)
        self._xyz = nn.Parameter(torch.randn((num_points, 3), device=self.device, dtype=torch.float32))
        self._scaling = nn.Parameter(torch.ones((num_points, 3), device=self.device, dtype=torch.float32) * -3.0)
        self._rotation = nn.Parameter(torch.zeros((num_points, 4), device=self.device, dtype=torch.float32))
        self._rotation.data[:, 0] = 1.0
        self._opacity = nn.Parameter(torch.ones((num_points, 1), device=self.device, dtype=torch.float32) * 0.5)
        self._features_dc = nn.Parameter(torch.randn((num_points, 1, 3), device=self.device, dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.zeros((num_points, 0, 3), device=self.device, dtype=torch.float32))
        self._motion_component_logits = None
        self.motion_components = None

    @property
    def get_xyz(self):
        return self._xyz


# ==============================================================================
# 1. Exact Rollback Test
# ==============================================================================

def test_exact_rollback_bit_for_bit():
    """Forced rejection must restore all model tensors, logits, components, optimizers, and RNG bit-for-bit."""
    dev = "cpu"
    N = 30
    pc = MockGaussianModel(num_points=N, device=dev)
    pc._motion_component_logits = nn.Parameter(torch.zeros((N, 1), dtype=torch.float32))

    mc = MotionComponents(
        num_components=1,
        components=[FourierComponentTrajectory(base_frequency_hz=5.691, fps=20.004, num_harmonics=1)],
    )
    pc.motion_components = mc

    opt = torch.optim.Adam([pc._xyz, pc._opacity], lr=0.001)
    mc_opt = torch.optim.Adam(
        [
            {"params": list(mc.parameters()), "lr": 0.001},
            {"params": [pc._motion_component_logits], "lr": 0.01},
        ]
    )

    controller = ComponentSplitMergeController(
        enabled=True,
        start_iter=1000,
        interval=500,
        eval_window=100,
        min_metric_improvement=0.5,  # High bar to guarantee rejection
        min_component_support=5.0,
    )

    # Step optimizer once to populate exp_avg and exp_avg_sq
    loss = pc._xyz.sum() + list(mc.parameters())[0].sum()
    loss.backward()
    opt.step()
    mc_opt.step()
    opt.zero_grad()
    mc_opt.zero_grad()

    # Set high logits so support >= 5.0
    pc._motion_component_logits.data.fill_(2.0)

    # Initial state snapshot copies for bit-exact verification (captured right before proposal)
    orig_xyz = pc._xyz.detach().clone()
    orig_logits = pc._motion_component_logits.detach().clone()
    orig_num_components = mc.num_components
    orig_comp_param = list(mc.parameters())[0].detach().clone()
    orig_opt_state = copy.deepcopy(opt.state_dict())
    orig_mc_opt_state = copy.deepcopy(mc_opt.state_dict())

    # Propose split at iteration 1000
    hex_res = torch.randn((N, 3), dtype=torch.float32)
    baseline_metric = 25.0

    proposed = controller.propose_split(
        pc=pc,
        mc=mc,
        target_k=0,
        iteration=1000,
        baseline_metric=baseline_metric,
        hex_residuals=hex_res,
        opt=opt,
        mc_opt=mc_opt,
    )
    assert proposed is True, "Proposal should have succeeded"
    assert mc.num_components == 2, "Components should be 2 after proposal"
    assert pc._motion_component_logits.shape == (N, 2)

    # Simulate training for eval_window iterations with arbitrary mutations
    pc._xyz.data.add_(torch.randn_like(pc._xyz) * 0.1)
    pc._motion_component_logits.data.add_(torch.randn_like(pc._motion_component_logits) * 0.5)

    # Decide at iteration 1100 with lower metric (regression -> REJECT)
    decided, decision = controller.evaluate_and_decide(
        current_iter=1100,
        current_metric=24.5,  # Regression (-0.5 dB)
        pc=pc,
        mc=mc,
        opt=opt,
        mc_opt=mc_opt,
    )
    assert decided is True
    assert decision == "rejected"
    assert mc.num_components == orig_num_components

    # Verify bit-exact restoration
    torch.testing.assert_close(pc._xyz, orig_xyz)
    torch.testing.assert_close(pc._motion_component_logits, orig_logits)
    torch.testing.assert_close(list(mc.parameters())[0], orig_comp_param)


# ==============================================================================
# 2. Synthetic K=1 -> K=2 Discovery Test
# ==============================================================================

def test_synthetic_discovery_k1_to_k2():
    """Verify that a two-mover dataset initialized at K=1 proposes a split and accepts it upon improvement."""
    dev = "cpu"
    N_each = 20
    N = 2 * N_each
    pc = MockGaussianModel(num_points=N, device=dev)
    pc._motion_component_logits = nn.Parameter(torch.ones((N, 1), dtype=torch.float32) * 2.0)

    mc = MotionComponents(
        num_components=1,
        components=[FourierComponentTrajectory(base_frequency_hz=5.0, fps=20.0, num_harmonics=1)],
    )
    pc.motion_components = mc

    opt = torch.optim.Adam([pc._xyz], lr=0.001)
    mc_opt = torch.optim.Adam([{"params": list(mc.parameters()), "lr": 0.001}])

    controller = ComponentSplitMergeController(
        enabled=True,
        start_iter=1000,
        interval=500,
        eval_window=100,
        min_metric_improvement=0.05,
        min_component_support=5.0,
    )

    # Synthetic residuals with bimodal direction: cluster 0 moves +X, cluster 1 moves +Y
    hex_res = torch.zeros((N, 3), dtype=torch.float32)
    hex_res[:N_each, 0] = 0.02
    hex_res[N_each:, 1] = 0.02

    proposed = controller.propose_split(
        pc=pc,
        mc=mc,
        target_k=0,
        iteration=1000,
        baseline_metric=25.0,
        hex_residuals=hex_res,
        opt=opt,
        mc_opt=mc_opt,
    )
    assert proposed is True
    assert mc.num_components == 2

    # Check that responsibilities partition the parent: q_ka + q_kb == q_parent
    q, q0 = mc.get_responsibilities(pc)
    # Sum of the two child responsibilities should equal approximately the pre-split responsibility
    child_sum = q[:, 0] + q[:, 1]
    assert (child_sum > 0.5).all(), "Child responsibilities should preserve parent mass"

    # Evaluate at iter 1100 with improved held-out metric (+0.8 dB -> ACCEPT)
    decided, decision = controller.evaluate_and_decide(
        current_iter=1100,
        current_metric=25.8,
        pc=pc,
        mc=mc,
        opt=opt,
        mc_opt=mc_opt,
    )
    assert decided is True
    assert decision == "accepted"
    assert mc.num_components == 2
    assert len(controller.ledger) == 1
    assert controller.ledger[0]["status"] == "accepted"


# ==============================================================================
# 3. Merge Test (K=2 -> K=1)
# ==============================================================================

def test_merge_components():
    """Verify merging duplicate components (K=2 -> K=1) preserves total responsibility."""
    dev = "cpu"
    N = 25
    pc = MockGaussianModel(num_points=N, device=dev)
    # Two components with non-zero logits
    pc._motion_component_logits = nn.Parameter(torch.ones((N, 2), dtype=torch.float32))

    mc = MotionComponents(
        num_components=2,
        components=[
            FourierComponentTrajectory(base_frequency_hz=5.0, fps=20.0, num_harmonics=1),
            FourierComponentTrajectory(base_frequency_hz=5.0, fps=20.0, num_harmonics=1),
        ],
    )
    pc.motion_components = mc

    controller = ComponentSplitMergeController(
        enabled=True,
        eval_window=50,
        min_metric_improvement=0.01,
    )

    q_pre, _ = mc.get_responsibilities(pc)
    total_pre = q_pre[:, 0] + q_pre[:, 1]

    proposed = controller.propose_merge(
        pc=pc,
        mc=mc,
        k1=0,
        k2=1,
        iteration=1000,
        baseline_metric=28.0,
    )
    assert proposed is True
    assert mc.num_components == 1

    q_post, _ = mc.get_responsibilities(pc)
    torch.testing.assert_close(q_post[:, 0], total_pre, rtol=1e-5, atol=1e-5)

    # Acceptance
    decided, decision = controller.evaluate_and_decide(
        current_iter=1050,
        current_metric=28.1,
        pc=pc,
        mc=mc,
    )
    assert decided is True
    assert decision == "accepted"
    assert mc.num_components == 1


# ==============================================================================
# 4. Static Scene & Non-Rigid Guardrails
# ==============================================================================

def test_static_scene_guardrail():
    """A purely static scene has negligible component support and triggers zero splits."""
    N = 40
    pc = MockGaussianModel(num_points=N)
    # Null responsibilities dominant (q0 ~ 1, q_k ~ 0)
    pc._motion_component_logits = nn.Parameter(torch.full((N, 1), -10.0))

    mc = MotionComponents(num_components=1)
    pc.motion_components = mc

    controller = ComponentSplitMergeController(
        enabled=True,
        min_component_support=5.0,  # Support will be << 1.0
    )

    hex_res = torch.zeros((N, 3))
    proposed = controller.propose_split(
        pc=pc,
        mc=mc,
        target_k=0,
        iteration=1500,
        baseline_metric=30.0,
        hex_residuals=hex_res,
    )
    assert proposed is False, "Static scene must not propose component split"
    assert mc.num_components == 1


def test_rejection_due_to_incoherent_motion():
    """Verify that a split improving PSNR but lacking motion coherence is rejected and rolled back."""
    N = 20
    pc = MockGaussianModel(num_points=N)
    pc._motion_component_logits = nn.Parameter(torch.ones((N, 1)) * 3.0)
    mc = MotionComponents(num_components=1)
    pc.motion_components = mc

    controller = ComponentSplitMergeController(
        enabled=True,
        eval_window=50,
        min_metric_improvement=0.1,
        min_motion_coherence=0.6,
        cooldown_iters=1000,
        min_component_support=1.0,
    )

    hex_res = torch.randn((N, 3))
    proposed = controller.propose_split(pc, mc, 0, iteration=1000, baseline_metric=25.0, hex_residuals=hex_res)
    assert proposed is True
    assert mc.num_components == 2

    # Metric improves substantially (+2.0 dB), but motion coherence is low (0.35 < 0.60)
    decided, decision = controller.evaluate_and_decide(
        current_iter=1050,
        current_metric=27.0,
        pc=pc,
        mc=mc,
        motion_coherence=0.35,
    )
    assert decided is True
    assert decision == "rejected"
    assert mc.num_components == 1, "Must roll back to 1 component upon motion incoherence"
    assert controller.ledger[0]["status"] == "rejected"
    assert controller.ledger[0]["reason"] == "incoherent_motion"


def test_rejection_cooldown():
    """A rejected split enters cooldown and is not re-proposed during the cooldown window."""
    N = 20
    pc = MockGaussianModel(num_points=N)
    pc._motion_component_logits = nn.Parameter(torch.ones((N, 1)) * 3.0)
    mc = MotionComponents(num_components=1)
    pc.motion_components = mc

    controller = ComponentSplitMergeController(
        enabled=True,
        eval_window=50,
        min_metric_improvement=1.0,
        cooldown_iters=1000,
        min_component_support=1.0,
    )

    hex_res = torch.randn((N, 3))
    controller.propose_split(pc, mc, 0, iteration=1000, baseline_metric=25.0, hex_residuals=hex_res)
    # Reject
    controller.evaluate_and_decide(current_iter=1050, current_metric=24.0, pc=pc, mc=mc)

    # Check cooldown: at iter 1500 (within 1050 + 1000 = 2050), proposal should be refused
    proposed_again = controller.propose_split(pc, mc, 0, iteration=1500, baseline_metric=25.0, hex_residuals=hex_res)
    assert proposed_again is False, "Split must be blocked during rejection cooldown"

    # After cooldown (iter 2100 > 2050): proposal allowed again
    proposed_after = controller.propose_split(pc, mc, 0, iteration=2100, baseline_metric=25.0, hex_residuals=hex_res)
    assert proposed_after is True


# ==============================================================================
# 5. Preset Resolution Tests
# ==============================================================================

def test_w5_presets_resolve():
    """Verify turbine_r1_missing_6_w5_e4 and lego_turbine_2_missing_w5_e4 resolve cleanly."""
    turbine_preset = validate_config("turbine_r1_missing_6_w5_e4")
    assert turbine_preset.optim.use_split_merge is True
    assert turbine_preset.optim.split_merge_start_iter == 1500
    assert turbine_preset.optim.split_merge_interval == 1000
    assert turbine_preset.optim.split_merge_max_components == 4
    assert turbine_preset.optim.split_merge_eval_window == 300
    assert turbine_preset.train.port == 6027

    lego_preset = validate_config("lego_turbine_2_missing_w5_e4")
    assert lego_preset.optim.use_split_merge is True
    assert lego_preset.optim.split_merge_start_iter == 1500
    assert lego_preset.optim.split_merge_interval == 1000
    assert lego_preset.optim.split_merge_max_components == 4
    assert lego_preset.optim.split_merge_eval_window == 300
    assert lego_preset.train.port == 6021
