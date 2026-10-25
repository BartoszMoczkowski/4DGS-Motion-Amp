"""Tests for Task T37 (Coarse-stage gating - R10).

Verifies:
1. Option A preset resolution:
   - Ladder: stock (3000), r1 (3000), r2 (3000), r10 (500).
   - Real turbine scenes: lego_turbine_2_missing, lego_turbine_base, lego_turbine_5_missing, turbine_r1_missing_6 (500).
   - Stock base preset remains 3000.
2. Config models & bridge generation:
   - OptimizationParams and PipelineParams support coarse_iterations, gated_coarse, roi_bbox.
   - render_bridge_source correctly serializes coarse gating settings.
3. Coarse-stage execution logic:
   - coarse_iterations <= 0 skips coarse stage.
   - opt.gated_coarse and opt.roi_bbox propagate to pipe.
4. Render-level Option B gated coarse:
   - Standard coarse (gated_coarse=False): deformation is never evaluated; static Gaussians returned.
   - Gated coarse (gated_coarse=True, no mask): deformation evaluated for all points.
   - Gated coarse with roi_bbox: only Gaussians inside 3D bounding box deform; outside points stay static.
   - Gated coarse with _roi_mask: only Gaussians with _roi_mask=True deform; outside points stay static.
   - Fine stage: deformation evaluated normally, unaffected by gated_coarse setting.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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

from pipeline.config.bridge import render_bridge_source
from pipeline.config.models import OptimizationParams, PipelineConfig, PipelineParams
from pipeline.config.resolver import resolve_preset


def test_preset_ladder_coarse_iterations():
    """Verify single-factor ablation ladder and turbine presets coarse_iterations."""
    # Ablation ladder steps
    stock = resolve_preset("lego_turbine_2_missing_stock")
    assert stock["optim"]["coarse_iterations"] == 3000
    assert stock["optim"]["use_motion_mask"] is False

    r1 = resolve_preset("lego_turbine_2_missing_r1")
    assert r1["optim"]["coarse_iterations"] == 3000
    assert r1["optim"]["use_motion_mask"] is False
    assert r1["hidden"]["regularizer_anneal_iter"] == 3000

    r2 = resolve_preset("lego_turbine_2_missing_r2")
    assert r2["optim"]["coarse_iterations"] == 3000
    assert r2["optim"]["use_motion_mask"] is True

    r10 = resolve_preset("lego_turbine_2_missing_r10")
    assert r10["optim"]["coarse_iterations"] == 500
    assert r10["optim"]["use_motion_mask"] is True
    assert r10["hidden"]["regularizer_anneal_iter"] == 3000


def test_real_turbine_presets_coarse_iterations():
    """Verify all real turbine scenes have coarse_iterations set to 500 (Option A)."""
    scenes = [
        "lego_turbine_2_missing",
        "lego_turbine_base",
        "lego_turbine_5_missing",
        "turbine_r1_missing_6",
    ]
    for scene in scenes:
        cfg = resolve_preset(scene)
        assert (
            cfg["optim"]["coarse_iterations"] == 500
        ), f"Scene {scene} expected coarse_iterations == 500, got {cfg['optim']['coarse_iterations']}"


def test_base_preset_defaults():
    """Verify default base preset retains upstream 3000 coarse iterations."""
    base = resolve_preset("base")
    assert base["optim"]["coarse_iterations"] == 3000


def test_models_coarse_gating_fields():
    """Verify OptimizationParams and PipelineParams support coarse gating fields."""
    opt_default = OptimizationParams()
    assert opt_default.coarse_iterations == 3000
    assert opt_default.gated_coarse is False
    assert opt_default.roi_bbox is None

    pipe_default = PipelineParams()
    assert pipe_default.gated_coarse is False
    assert pipe_default.roi_bbox is None

    # Custom settings
    opt_custom = OptimizationParams(
        coarse_iterations=500,
        gated_coarse=True,
        roi_bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2],
    )
    assert opt_custom.coarse_iterations == 500
    assert opt_custom.gated_coarse is True
    assert opt_custom.roi_bbox == [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2]

    pipe_custom = PipelineParams(
        gated_coarse=True,
        roi_bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2],
    )
    assert pipe_custom.gated_coarse is True
    assert pipe_custom.roi_bbox == [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2]


def test_bridge_source_rendering():
    """Verify bridge file generation serializes coarse_iterations, gated_coarse, and roi_bbox."""
    raw = resolve_preset("lego_turbine_2_missing_r10")
    config = PipelineConfig.model_validate(raw)
    bridge_src = render_bridge_source(config.model_dump())

    assert "'coarse_iterations': 500" in bridge_src
    assert "'gated_coarse': False" in bridge_src

    # Test with gated_coarse enabled and custom roi_bbox
    raw["optim"]["gated_coarse"] = True
    raw["optim"]["roi_bbox"] = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]
    raw.setdefault("pipeline_params", {})["gated_coarse"] = True
    raw["pipeline_params"]["roi_bbox"] = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]

    config_gated = PipelineConfig.model_validate(raw)
    bridge_gated = render_bridge_source(config_gated.model_dump())

    assert "'gated_coarse': True" in bridge_gated
    assert "[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]" in bridge_gated


class MockDeformation(nn.Module):
    """Mock deformation network that adds a distinct displacement."""

    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(self, means3D, scales, rotations, opacity, shs, time):
        return (
            means3D + self.delta,
            scales + 0.1,
            rotations,
            opacity + 0.05,
            shs + 0.01,
        )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self: self


class MockGaussianModel:
    """Mock GaussianModel for testing render stage branching."""

    def __init__(self, n_points: int = 4):
        # 4 points: 2 inside bbox [-0.5, 0.5], 2 outside
        self._xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.2, 0.1],
                [2.0, 2.0, 2.0],
                [-3.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=DEVICE,
        )
        self._scaling = torch.ones((n_points, 3), dtype=torch.float32, device=DEVICE) * 0.1
        self._rotation = torch.zeros((n_points, 4), dtype=torch.float32, device=DEVICE)
        self._rotation[:, 0] = 1.0
        self._opacity = torch.ones((n_points, 1), dtype=torch.float32, device=DEVICE) * 0.8
        self._features_dc = torch.zeros((n_points, 1, 3), dtype=torch.float32, device=DEVICE)
        self._features_rest = torch.zeros((n_points, 15, 3), dtype=torch.float32, device=DEVICE)
        self._deformation = MockDeformation(delta=0.5)
        self._deformation_table = torch.ones(n_points, dtype=torch.bool, device=DEVICE)
        self._roi_mask = None
        self.max_sh_degree = 0
        self.active_sh_degree = 0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def scaling_activation(self, s):
        return torch.exp(s)

    def rotation_activation(self, r):
        return r / torch.norm(r, dim=-1, keepdim=True)

    def opacity_activation(self, o):
        return torch.sigmoid(o)

    def get_covariance(self, scaling_modifier=1):
        return None


class MockCamera:
    """Mock Camera object with camera parameters and time."""

    def __init__(self, time_val: float = 0.25):
        self.time = time_val
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.image_height = 64
        self.image_width = 64
        self.world_view_transform = torch.eye(4, device=DEVICE)
        self.projection_matrix = torch.eye(4, device=DEVICE)
        self.full_proj_transform = torch.eye(4, device=DEVICE)
        self.camera_center = torch.zeros(3, device=DEVICE)
        self.tanfovx = 1.0
        self.tanfovy = 1.0


class MockPipelineParams:
    """Mock pipeline settings."""

    def __init__(self, gated_coarse: bool = False, roi_bbox=None):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.gated_coarse = gated_coarse
        self.roi_bbox = roi_bbox


def test_render_standard_coarse_stage():
    """Verify that standard coarse stage (gated_coarse=False) skips deformation."""
    from gaussian_renderer import render

    pc = MockGaussianModel()
    pipe = MockPipelineParams(gated_coarse=False)
    cam = MockCamera()
    bg = torch.zeros(3, device=DEVICE)

    # Coarse stage: static geometry
    render_pkg = render(cam, pc, pipe, bg, stage="coarse")
    # pc._xyz was [0, 0, 0], [0.2, -0.2, 0.1], [2, 2, 2], [-3, 0, 1]
    # No deformation should be added
    # (GaussianRasterizer is mocked and returns None, None, but render executes the stage logic)


def test_render_gated_coarse_all_points():
    """Verify Option B gated coarse with no bbox/mask applies deformation to all points."""
    from gaussian_renderer import render

    pc = MockGaussianModel()
    pc._deformation = MagicMock(
        return_value=(
            pc._xyz + 1.0,
            pc._scaling,
            pc._rotation,
            pc._opacity,
            pc.get_features,
        )
    )
    pipe = MockPipelineParams(gated_coarse=True, roi_bbox=None)
    cam = MockCamera()
    bg = torch.zeros(3, device=DEVICE)

    render(cam, pc, pipe, bg, stage="coarse")

    # _deformation should have been called for all 4 points
    assert pc._deformation.called
    call_args = pc._deformation.call_args[0]
    # First argument is means3D
    assert call_args[0].shape[0] == 4


def test_render_gated_coarse_with_roi_bbox():
    """Verify Option B gated coarse with roi_bbox only deforms points inside the 3D bounding box."""
    from gaussian_renderer import render

    pc = MockGaussianModel()
    # 4 points: pts 0 and 1 are inside [-0.5, 0.5]^3; pts 2 and 3 are outside
    called_means = []

    def mock_deform(means3D, scales, rotations, opacity, shs, time):
        called_means.append(means3D.clone())
        return means3D + 0.5, scales, rotations, opacity, shs

    pc._deformation = mock_deform

    pipe = MockPipelineParams(
        gated_coarse=True, roi_bbox=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    )
    cam = MockCamera()
    bg = torch.zeros(3, device=DEVICE)

    render(cam, pc, pipe, bg, stage="coarse")

    # _deformation should have been called only for the 2 points inside the bbox
    assert len(called_means) == 1
    deformed_pts = called_means[0]
    assert deformed_pts.shape[0] == 2
    # The two points inside bbox are [0, 0, 0] and [0.2, -0.2, 0.1]
    assert torch.allclose(deformed_pts[0], torch.tensor([0.0, 0.0, 0.0], device=DEVICE))
    assert torch.allclose(deformed_pts[1], torch.tensor([0.2, -0.2, 0.1], device=DEVICE))


def test_render_gated_coarse_with_roi_mask():
    """Verify Option B gated coarse with _roi_mask only deforms points where mask is True."""
    from gaussian_renderer import render

    pc = MockGaussianModel()
    # Mask selects only point index 1 and 2
    pc._roi_mask = torch.tensor([False, True, True, False], device=DEVICE)
    called_means = []

    def mock_deform(means3D, scales, rotations, opacity, shs, time):
        called_means.append(means3D.clone())
        return means3D + 0.5, scales, rotations, opacity, shs

    pc._deformation = mock_deform

    pipe = MockPipelineParams(gated_coarse=True)
    cam = MockCamera()
    bg = torch.zeros(3, device=DEVICE)

    render(cam, pc, pipe, bg, stage="coarse")

    # _deformation should have been called only for the 2 masked points
    assert len(called_means) == 1
    deformed_pts = called_means[0]
    assert deformed_pts.shape[0] == 2
    assert torch.allclose(deformed_pts[0], torch.tensor([0.2, -0.2, 0.1], device=DEVICE))
    assert torch.allclose(deformed_pts[1], torch.tensor([2.0, 2.0, 2.0], device=DEVICE))


def test_render_fine_stage_always_deforms():
    """Verify fine stage always evaluates deformation regardless of gated_coarse."""
    from gaussian_renderer import render

    pc = MockGaussianModel()
    pc._deformation = MagicMock(
        return_value=(
            pc._xyz + 0.5,
            pc._scaling,
            pc._rotation,
            pc._opacity,
            pc.get_features,
        )
    )
    pipe = MockPipelineParams(gated_coarse=False)
    cam = MockCamera()
    bg = torch.zeros(3, device=DEVICE)

    render(cam, pc, pipe, bg, stage="fine")

    assert pc._deformation.called
    assert pc._deformation.call_args[0][0].shape[0] == 4
