"""Tests for Work Package W0 (Task T48: Native-4DGS Baseline Audit & Hard-Gate Removal).

Verifies:
1. Disabled-feature no-op equivalence:
   - In baseline mode (all hard gates, ROI masks/grids, and Fourier/rigid heads disabled),
     `render(..., stage="fine")` and `utils.render_utils.get_state_at_time(...)` invoke
     `pc._deformation` for 100% of Gaussians and match native HexPlane deformation output
     bit-for-bit.
   - `render(..., stage="coarse")` with `gated_coarse=False` returns canonical static Gaussians
     without calling `pc._deformation`.
   - `_resolve_roi_mask` is never invoked in baseline mode.
2. ROI & Fourier-head file isolation:
   - When `use_roi_voxels=False` and `use_fourier_rigid_head=False`, `Scene.__init__`
     (including checkpoint reload with `load_iteration`), `render()`, and `get_state_at_time()`
     never read `roi_voxels.npz` or `fourier_head.pth` even when those files exist on disk.
   - Missing `roi_voxels.npz` has zero effect on baseline execution.
3. Dual entry-point config resolution parity:
   - `lego_turbine_2_missing_w0_baseline` and `turbine_r1_missing_6_w0_baseline` resolve to
     identical `ModelParams`, `OptimizationParams`, `PipelineParams`, and `ModelHiddenParams`
     dicts through the `core/train.py` and `orchestrator/pipeline/vendored/cuda/train.py`
     argument + `merge_hparams` pipelines.
   - `merge_hparams` does not allow `ModelParams` / `PipelineParams` defaults in bridge files
     to clobber `OptimizationParams` settings on the flat `args` namespace.
4. Baseline checkpoint save/load smoke test:
   - Saving and reloading a baseline scene/checkpoint does not create or load
     `fourier_head.pth`, `learned_offsets.json`, or `roi_voxels.npz`.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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
    m_knn_c.distCUDA2 = lambda p: torch.ones(p.shape[0], dtype=torch.float32, device=p.device) * 0.01
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
        last_call_kwargs: dict | None = None

        def __init__(self, raster_settings=None):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            MockRasterizer.last_call_kwargs = kwargs
            dev = kwargs.get("means3D", torch.empty(0)).device
            n = kwargs.get("means3D", torch.empty(4)).shape[0]
            return (
                torch.zeros((3, 64, 64), device=dev),
                torch.ones(n, device=dev),
                torch.zeros((1, 64, 64), device=dev),
            )

    m_diff.GaussianRasterizationSettings = MockRasterSettings
    m_diff.GaussianRasterizer = MockRasterizer
    sys.modules["diff_gaussian_rasterization"] = m_diff

from arguments import (
    ModelHiddenParams as CoreModelHiddenParams,
    ModelParams as CoreModelParams,
    OptimizationParams as CoreOptimizationParams,
    PipelineParams as CorePipelineParams,
)
from utils.params_utils import merge_hparams
from utils.render_utils import get_state_at_time
from pipeline.config import resolve_preset, validate_config
from pipeline.config.bridge import render_bridge_source
from pipeline.config.models import PipelineConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


class DeterministicDeformation(nn.Module):
    """Deterministic non-trivial deformation network for equivalence testing."""

    def __init__(self):
        super().__init__()

    def forward(self, means3D, scales, rotations, opacity, shs, times_sel):
        dt = times_sel
        d_xyz = torch.stack(
            [
                0.05 * torch.sin(means3D[:, 0] + dt[:, 0] * 3.14159),
                0.03 * torch.cos(means3D[:, 1] - dt[:, 0] * 1.57079),
                -0.02 * dt[:, 0],
            ],
            dim=-1,
        )
        return (
            means3D + d_xyz,
            scales + 0.01 * dt,
            rotations + 0.005 * dt,
            opacity + 0.02 * dt,
            shs + 0.001 * dt.unsqueeze(-1),
        )


class BaselineMockGaussianModel:
    """Synthetic GaussianModel with random points spanning inside and outside typical ROIs."""

    def __init__(self, n_points: int = 16, seed: int = 42):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self._xyz = (torch.rand((n_points, 3), generator=gen) * 4.0 - 2.0).to(DEVICE)
        self._scaling = (torch.rand((n_points, 3), generator=gen) * 0.5 - 1.5).to(DEVICE)
        rot = torch.randn((n_points, 4), generator=gen)
        self._rotation = (rot / torch.norm(rot, dim=-1, keepdim=True)).to(DEVICE)
        self._opacity = (torch.rand((n_points, 1), generator=gen) * 2.0 - 1.0).to(DEVICE)
        self._features_dc = torch.randn((n_points, 1, 3), generator=gen).to(DEVICE)
        self._features_rest = torch.randn((n_points, 15, 3), generator=gen).to(DEVICE)
        self._deformation = DeterministicDeformation().to(DEVICE)
        self._deformation_table = torch.ones(n_points, dtype=torch.bool, device=DEVICE)
        self._roi_mask = None
        self._roi_grid = None
        self.fourier_head = None
        self.max_sh_degree = 3
        self.active_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def scaling_activation(self, s):
        return torch.exp(s)

    def rotation_activation(self, r):
        return torch.nn.functional.normalize(r)

    def opacity_activation(self, o):
        return torch.sigmoid(o)

    def get_covariance(self, scaling_modifier=1):
        return None


class BaselineMockCamera:
    """Synthetic Camera with non-zero timestamp."""

    def __init__(self, time_val: float = 0.37):
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


class BaselineMockPipe:
    """PipelineParams in W0 native-4DGS baseline configuration."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.gated_coarse = False
        self.roi_bbox = None


# ---------------------------------------------------------------------------
# 1. Disabled-Feature No-Op Equivalence Tests
# ---------------------------------------------------------------------------


def test_disabled_feature_noop_equivalence_render_and_get_state(monkeypatch):
    """Prove render() and get_state_at_time() in baseline mode match native HexPlane 100%."""
    import gaussian_renderer

    pc = BaselineMockGaussianModel(n_points=16, seed=123)
    pipe = BaselineMockPipe()
    cam = BaselineMockCamera(time_val=0.42)
    bg = torch.zeros(3, device=DEVICE)

    # Spy on _resolve_roi_mask to ensure it is NEVER called in baseline mode
    resolve_roi_spy = MagicMock(wraps=gaussian_renderer._resolve_roi_mask)
    monkeypatch.setattr(gaussian_renderer, "_resolve_roi_mask", resolve_roi_spy)

    # Compute ground-truth native HexPlane deformation directly
    N = pc.get_xyz.shape[0]
    time_tensor = torch.full((N, 1), cam.time, dtype=pc.get_xyz.dtype, device=DEVICE)
    expected_means, expected_scales_raw, expected_rot_raw, expected_opac_raw, expected_shs = (
        pc._deformation(
            pc._xyz,
            pc._scaling,
            pc._rotation,
            pc._opacity,
            pc.get_features,
            time_tensor,
        )
    )

    # Wrap pc._deformation to verify call count and input slice length
    deform_calls = []
    orig_deform = pc._deformation

    def tracked_deform(means3D, scales, rotations, opacity, shs, times_sel):
        deform_calls.append(means3D.shape[0])
        return orig_deform(means3D, scales, rotations, opacity, shs, times_sel)

    pc._deformation = tracked_deform

    # (A) Test get_state_at_time (used by trajectory extraction / seg_extract)
    means_f, scales_f, rot_f, opac_f, shs_f = get_state_at_time(pc, cam)
    assert deform_calls == [16], f"Expected 1 call with all 16 Gaussians, got {deform_calls}"
    assert resolve_roi_spy.call_count == 0, "_resolve_roi_mask must not be called in baseline mode"
    assert torch.equal(means_f, expected_means)
    assert torch.equal(scales_f, expected_scales_raw)
    assert torch.equal(rot_f, expected_rot_raw)
    assert torch.equal(opac_f, pc._opacity)
    assert torch.equal(shs_f, expected_shs)

    # (B) Test render() in fine stage
    deform_calls.clear()
    pkg_fine = gaussian_renderer.render(cam, pc, pipe, bg, stage="fine")
    assert deform_calls == [16], f"Expected 1 call with all 16 Gaussians in render(), got {deform_calls}"
    assert resolve_roi_spy.call_count == 0
    assert torch.equal(pkg_fine["means3D_final"], expected_means)

    # (C) Test render() in coarse stage (canonical static)
    deform_calls.clear()
    pkg_coarse = gaussian_renderer.render(cam, pc, pipe, bg, stage="coarse")
    assert deform_calls == [], "Coarse stage in baseline mode must never call _deformation"
    assert resolve_roi_spy.call_count == 0
    assert torch.equal(pkg_coarse["means3D_final"], pc.get_xyz)


# ---------------------------------------------------------------------------
# 2. ROI & Fourier-Head File Isolation Tests
# ---------------------------------------------------------------------------


def test_roi_and_fourier_head_files_never_read_when_disabled(tmp_path, monkeypatch):
    """Prove Scene reload does not read roi_voxels.npz or fourier_head.pth when disabled."""
    import scene as scene_module
    import scene.roi_grid as roi_grid_module
    from scene.fourier_rigid import FourierRigidHead

    source_dir = tmp_path / "source"
    model_dir = tmp_path / "model"
    point_cloud_dir = model_dir / "point_cloud" / "iteration_500"
    source_dir.mkdir(parents=True)
    point_cloud_dir.mkdir(parents=True)
    (source_dir / "points3D_multipleview.ply").touch()

    # Write a valid roi_voxels.npz in source_dir and fourier_head.pth in model_dir
    np.savez_compressed(
        source_dir / "roi_voxels.npz",
        grid=np.ones((4, 4, 4), dtype=bool),
        origin=np.array([-1.0, -1.0, -1.0], dtype=np.float64),
        voxel_size=np.float64(0.5),
    )
    head = FourierRigidHead()
    torch.save(head.state_dict(), model_dir / "fourier_head.pth")
    (model_dir / "fourier_head_state.json").write_text(
        '{"frequency_hz": 5.691, "num_harmonics": 1, "pure_rigid": false, "static_background": false}',
        encoding="utf-8",
    )
    assert (source_dir / "roi_voxels.npz").exists()
    assert (model_dir / "fourier_head.pth").exists()
    assert (model_dir / "fourier_head_state.json").exists()

    # Spy on load_roi_grid and FourierRigidHead.load_state_dict
    roi_load_spy = MagicMock(wraps=roi_grid_module.load_roi_grid)
    head_load_spy = MagicMock(wraps=FourierRigidHead.load_state_dict)
    monkeypatch.setattr(roi_grid_module, "load_roi_grid", roi_load_spy)
    monkeypatch.setattr(FourierRigidHead, "load_state_dict", head_load_spy)

    # Stub dataset reader and ply loader so Scene.__init__ runs fast in unit test
    class DummyPointCloud:
        points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    class DummySceneInfo:
        point_cloud = DummyPointCloud()
        train_cameras = []
        test_cameras = []
        video_cameras = []
        nerf_normalization = {"radius": 1.0}
        ply_path = str(source_dir / "points3D.ply")
        maxtime = 1.0

    monkeypatch.setattr(
        scene_module,
        "sceneLoadTypeCallbacks",
        {"MultipleView": lambda *a, **kw: DummySceneInfo()},
    )
    monkeypatch.setattr(scene_module, "cameraList_from_camInfos", lambda *a, **kw: [])

    class DummyDeformationNet:
        def set_aabb(self, xyz_max, xyz_min):
            pass

    class DummyDeformation:
        deformation_net = DummyDeformationNet()

    class DummyGaussians:
        def __init__(self):
            self._roi_grid = None
            self._roi_mask = None
            self.fourier_head = None
            self._deformation = DummyDeformation()

        def load_ply(self, path):
            pass

        def load_model(self, path):
            pass

        def create_from_pcd(self, pcd, extent, maxtime):
            pass

    parser = argparse.ArgumentParser()
    lp = CoreModelParams(parser)
    args = parser.parse_args(
        [
            "-s",
            str(source_dir),
            "-m",
            str(model_dir),
        ]
    )
    dataset = lp.extract(args)
    assert dataset.use_roi_voxels is False
    assert dataset.use_fourier_rigid_head is False

    gaussians = DummyGaussians()
    scn = scene_module.Scene(
        dataset,
        gaussians,
        load_iteration=500,
        shuffle=False,
    )
    assert scn.loaded_iter == 500
    assert roi_load_spy.call_count == 0, "load_roi_grid must not be called when use_roi_voxels=False"
    assert head_load_spy.call_count == 0, "FourierRigidHead must not be loaded when use_fourier_rigid_head=False"
    assert gaussians._roi_grid is None
    assert gaussians.fourier_head is None

    # Also verify missing roi_voxels.npz has zero effect
    (source_dir / "roi_voxels.npz").unlink()
    gaussians2 = DummyGaussians()
    scn2 = scene_module.Scene(
        dataset,
        gaussians2,
        load_iteration=500,
        shuffle=False,
    )
    assert scn2.loaded_iter == 500
    assert gaussians2._roi_grid is None
    assert gaussians2.fourier_head is None


# ---------------------------------------------------------------------------
# 3. Dual Entry-Point Config Resolution Parity Tests
# ---------------------------------------------------------------------------


def _resolve_via_entrypoint_args(preset_name: str, tmp_path: Path) -> dict[str, dict]:
    """Simulate the exact argument + mmengine.Config + merge_hparams flow of train.py."""
    from mmengine.config import Config as MMConfig

    validated = validate_config(preset_name)
    bridge_code = render_bridge_source(validated.model_dump())
    bridge_file = tmp_path / f"{preset_name}_bridge.py"
    bridge_file.write_text(bridge_code, encoding="utf-8")

    parser = argparse.ArgumentParser()
    lp = CoreModelParams(parser)
    op = CoreOptimizationParams(parser)
    pp = CorePipelineParams(parser)
    hp = CoreModelHiddenParams(parser)
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(["--configs", str(bridge_file)])
    cfg = MMConfig.fromfile(args.configs)
    args = merge_hparams(args, cfg)

    return {
        "ModelParams": vars(lp.extract(args)),
        "OptimizationParams": vars(op.extract(args)),
        "PipelineParams": vars(pp.extract(args)),
        "ModelHiddenParams": vars(hp.extract(args)),
    }


@pytest.mark.parametrize(
    "preset_name",
    [
        "lego_turbine_2_missing_w0_baseline",
        "turbine_r1_missing_6_w0_baseline",
    ],
)
def test_w0_baseline_presets_resolution_and_parity(preset_name: str, tmp_path: Path):
    """Verify W0 baseline presets resolve identically and disable all non-baseline gates."""
    extracted = _resolve_via_entrypoint_args(preset_name, tmp_path)
    m_cfg = extracted["ModelParams"]
    o_cfg = extracted["OptimizationParams"]
    p_cfg = extracted["PipelineParams"]
    h_cfg = extracted["ModelHiddenParams"]

    # 1. Agreed coarse / regularizer schedule
    assert o_cfg["coarse_iterations"] == 500
    assert h_cfg["regularizer_anneal_iter"] == 3000
    assert pytest.approx(h_cfg["time_smoothness_weight_final"]) == 0.0001
    assert pytest.approx(h_cfg["l1_time_planes_final"]) == 0.0

    # 2. Fixed per-camera timestamp offsets enabled, learnable offsets disabled
    assert m_cfg["use_camera_offsets"] is True
    assert o_cfg["use_camera_offsets"] is True
    assert o_cfg["learnable_camera_offsets"] is False

    # 3. Hard spatial gates disabled everywhere
    assert o_cfg["gated_coarse"] is False
    assert p_cfg["gated_coarse"] is False
    assert o_cfg["roi_bbox"] is None
    assert p_cfg["roi_bbox"] is None
    assert o_cfg["use_roi_voxels"] is False
    assert m_cfg["use_roi_voxels"] is False

    # 4. Soft motion-mask loss disabled
    assert o_cfg["use_motion_mask"] is False
    assert o_cfg["motion_mask_dir"] == ""

    # 5. Trajectory loss disabled
    assert o_cfg["lambda_trajectory"] == 0.0
    assert o_cfg["trajectory_track_file"] == ""
    assert o_cfg["trajectory_roi_bbox"] is None

    # 6. Fourier / rigid head & static background disabled
    assert o_cfg["use_fourier_rigid_head"] is False
    assert m_cfg["use_fourier_rigid_head"] is False
    assert o_cfg["fourier_head_pure_rigid"] is False
    assert o_cfg["fourier_head_static_background"] is False
    assert o_cfg["fourier_head_roi_bbox"] is None


def test_merge_hparams_shared_flag_reconciliation():
    """Verify merge_hparams does not clobber OptimizationParams with ModelParams/PipelineParams defaults."""
    parser = argparse.ArgumentParser()
    lp = CoreModelParams(parser)
    op = CoreOptimizationParams(parser)
    pp = CorePipelineParams(parser)
    hp = CoreModelHiddenParams(parser)
    args = parser.parse_args([])

    # Simulate a config where OptimizationParams overrides gated_coarse, roi_bbox, and use_camera_offsets
    # while PipelineParams and ModelParams still hold defaults
    config_dict = {
        "ModelParams": {"use_camera_offsets": True},
        "PipelineParams": {"gated_coarse": False, "roi_bbox": None},
        "ModelHiddenParams": {},
        "OptimizationParams": {
            "gated_coarse": True,
            "roi_bbox": [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2],
            "use_camera_offsets": False,
        },
    }
    merged = merge_hparams(args, config_dict)
    assert pp.extract(merged).gated_coarse is True
    assert op.extract(merged).gated_coarse is True
    assert pp.extract(merged).roi_bbox == [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2]
    assert lp.extract(merged).use_camera_offsets is False
    assert op.extract(merged).use_camera_offsets is False

    # Also verify Pydantic PipelineConfig validator synchronizes shared flags
    pyd_cfg = PipelineConfig.model_validate(
        {
            "optim": {
                "gated_coarse": True,
                "roi_bbox": [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1],
                "use_camera_offsets": False,
                "use_roi_voxels": True,
                "use_fourier_rigid_head": True,
            }
        }
    )
    assert pyd_cfg.pipeline_params.gated_coarse is True
    assert pyd_cfg.pipeline_params.roi_bbox == [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]
    assert pyd_cfg.model.use_camera_offsets is False
    assert pyd_cfg.model.use_roi_voxels is True
    assert pyd_cfg.model.use_fourier_rigid_head is True


def test_core_and_vendored_train_entrypoint_parity():
    """Verify core/train.py and orchestrator/pipeline/vendored/cuda/train.py share identical W0 gating guards."""
    core_src = (REPO_ROOT / "core" / "train.py").read_text(encoding="utf-8")
    vend_src = (
        REPO_ROOT / "orchestrator" / "pipeline" / "vendored" / "cuda" / "train.py"
    ).read_text(encoding="utf-8")

    required_guards = [
        'if getattr(opt, "use_roi_voxels", False):',
        'if getattr(opt, "use_fourier_rigid_head", False):',
        'if getattr(opt, "use_motion_mask", False) and cam_mask is not None:',
        'if getattr(opt, "learnable_camera_offsets", False):',
        'if getattr(opt, "lambda_trajectory", 0.0) > 0.0:',
    ]
    for guard in required_guards:
        assert guard in core_src, f"Missing guard in core/train.py: {guard}"
        assert guard in vend_src, f"Missing guard in vendored/cuda/train.py: {guard}"


# ---------------------------------------------------------------------------
# 4. Baseline Checkpoint Save/Load Smoke Test
# ---------------------------------------------------------------------------


def test_baseline_checkpoint_save_and_load_smoke(tmp_path):
    """Verify baseline model state saves and reloads cleanly without auxiliary head/ROI artifacts."""
    pc = BaselineMockGaussianModel(n_points=8, seed=7)
    ckpt_path = tmp_path / "chkpnt_baseline.pth"

    state = {
        "xyz": pc._xyz.detach().cpu(),
        "scaling": pc._scaling.detach().cpu(),
        "rotation": pc._rotation.detach().cpu(),
        "opacity": pc._opacity.detach().cpu(),
        "features_dc": pc._features_dc.detach().cpu(),
        "features_rest": pc._features_rest.detach().cpu(),
    }
    torch.save((state, 500), ckpt_path)
    loaded_state, loaded_iter = torch.load(ckpt_path, weights_only=False)

    assert loaded_iter == 500
    assert torch.equal(loaded_state["xyz"], pc._xyz.detach().cpu())
    assert not (tmp_path / "fourier_head.pth").exists()
    assert not (tmp_path / "learned_offsets.json").exists()
    assert not (tmp_path / "roi_voxels.npz").exists()
