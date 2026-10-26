"""Tests for Task T39 (Per-camera timestamps in the loader - R6).

Verifies:
1. Offset table loading:
   - Successfully loads offsets.json from synthetic and real scenes.
   - Fallback to optical_offsets.json when offsets.json is absent.
   - Anchors offsets to reference_camera (default cam01 with delta = 0.0).
   - Accurately parses fps, offsets_s, and drift_s_per_s.
2. Distinct per-camera continuous timestamps:
   - For frame i, different cameras produce distinct continuous normalized timestamps:
     t_norm(k, i) = ((1 + d_k) * i + delta_k * fps) / image_length.
   - True physical seconds t_phys = (1 + d_k) * (i / fps) + delta_k.
   - Reference camera cam01 preserves exact canonical time i / image_length.
3. Backward compatibility:
   - When offsets file is absent, has_offsets is False, and all cameras receive
     identical legacy timestamps i / image_length.
4. Single-factor ablation controls:
   - use_camera_offsets=False forces legacy shared timestamps even if offsets.json exists.
   - PIPELINE_USE_CAMERA_OFFSETS="0" environment variable forces legacy timestamps.
5. FourDGSdataset and Camera integration:
   - FourDGSdataset forwards fps, offsets_s, has_offsets, and get_physical_time.
   - Camera receives time=t_norm and physical_time=t_phys.
6. Pydantic config & bridge file generation:
   - OptimizationParams and ModelParams include use_camera_offsets: bool = True.
   - render_bridge_source serializes use_camera_offsets.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

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


from scene.multipleview_dataset import multipleview_dataset
from scene.dataset import FourDGSdataset
from scene.cameras import Camera
from pipeline.config.models import ModelParams, OptimizationParams, PipelineConfig
from pipeline.config.bridge import render_bridge_source


class MockExtrinsic:
    def __init__(self, cam_num: int):
        self.qvec = [1, 0, 0, 0]
        self.tvec = [0, 0, 0]
        self.name = f"frame{str(cam_num).zfill(2)}.jpg"


class MockIntrinsic:
    def __init__(self, width: int = 50, height: int = 40):
        self.params = [500.0]
        self.height = height
        self.width = width


def _build_mock_multicam_scene(tmp_path: Path, num_cams: int = 10, num_frames: int = 20):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    img = np.ones((40, 50, 3), dtype=np.uint8) * 128

    extrinsics = {}
    intrinsics = {1: MockIntrinsic(50, 40)}

    for c in range(1, num_cams + 1):
        c_name = f"cam{str(c).zfill(2)}"
        cam_dir = scene_dir / c_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f in range(1, num_frames + 1):
            cv2.imwrite(str(cam_dir / f"frame_{str(f).zfill(5)}.jpg"), img)
        extrinsics[c] = MockExtrinsic(c)

    return scene_dir, extrinsics, intrinsics


def test_offsets_loading_synthetic_preflight():
    """Verify loading offsets.json from the preflight synthetic scene."""
    scene_path = REPO_ROOT / "data" / "multipleview" / "synthetic_turbine_preflight"
    assert (scene_path / "offsets.json").exists()

    extrinsics = {1: MockExtrinsic(1), 2: MockExtrinsic(2)}
    intrinsics = {1: MockIntrinsic()}

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_path),
        split="train",
    )

    assert mv_ds.has_offsets is True
    assert mv_ds.fps == 20.0
    assert mv_ds.reference_camera == "cam01"
    assert mv_ds.offsets_s["cam01"] == 0.0
    assert pytest.approx(mv_ds.offsets_s["cam02"], rel=1e-5) == 0.018
    assert pytest.approx(mv_ds.offsets_s["cam05"], rel=1e-5) == -0.0226
    assert len(mv_ds.offsets_s) == 10


def test_offsets_loading_real_turbine_scene():
    """Verify loading offsets.json and affine drift from lego_turbine_2_missing."""
    scene_path = REPO_ROOT / "data" / "multipleview" / "lego_turbine_2_missing"
    assert (scene_path / "offsets.json").exists()

    extrinsics = {1: MockExtrinsic(1), 2: MockExtrinsic(2)}
    intrinsics = {1: MockIntrinsic()}

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_path),
        split="train",
    )

    assert mv_ds.has_offsets is True
    assert pytest.approx(mv_ds.fps, rel=1e-4) == 20.0039
    assert mv_ds.reference_camera == "cam01"
    assert mv_ds.offsets_s["cam01"] == 0.0
    assert pytest.approx(mv_ds.offsets_s["cam02"], rel=1e-4) == 0.018
    # Drift rate for cam03 is ~0.156 ms/s
    assert "cam03" in mv_ds.drift_s_per_s
    assert pytest.approx(mv_ds.drift_s_per_s["cam03"], rel=1e-3) == 0.00015606


def test_optical_offsets_fallback(tmp_path):
    """Verify fallback to optical_offsets.json when offsets.json is absent."""
    scene_dir, extrinsics, intrinsics = _build_mock_multicam_scene(tmp_path, num_cams=2, num_frames=5)

    optical_payload = {
        "fps": 20.0,
        "ref_camera": "cam01",
        "optical_offsets_s": {
            "cam01": 0.0,
            "cam02": 0.015,
        },
    }
    (scene_dir / "optical_offsets.json").write_text(json.dumps(optical_payload), encoding="utf-8")

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
    )

    assert mv_ds.has_offsets is True
    assert mv_ds.fps == 20.0
    assert mv_ds.reference_camera == "cam01"
    assert mv_ds.offsets_s["cam01"] == 0.0
    assert pytest.approx(mv_ds.offsets_s["cam02"], rel=1e-5) == 0.015


def test_per_camera_timestamps_distinct_normalized(tmp_path):
    """Verify that frame i across cameras receives distinct continuous timestamps."""
    scene_dir, extrinsics, intrinsics = _build_mock_multicam_scene(tmp_path, num_cams=3, num_frames=20)

    # Injected offsets: cam01=0.0s, cam02=+0.020s, cam03=-0.015s
    # Drift: cam03=+0.0001 s/s
    offsets_payload = {
        "fps": 20.0,
        "reference_camera": "cam01",
        "offsets_s": {
            "cam01": 0.0,
            "cam02": 0.020,
            "cam03": -0.015,
        },
        "drift_s_per_s": {
            "cam01": 0.0,
            "cam02": 0.0,
            "cam03": 0.0001,
        },
    }
    (scene_dir / "offsets.json").write_text(json.dumps(offsets_payload), encoding="utf-8")

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
    )

    assert mv_ds.has_offsets is True
    assert mv_ds.image_length == 20
    fps = 20.0
    denom = 20.0

    # Test frame 0 (first frame of each camera)
    # indices: cam01 frame 0 is index 0
    #          cam02 frame 0 is index 20
    #          cam03 frame 0 is index 40
    t_c1_0 = mv_ds.image_times[0]
    t_c2_0 = mv_ds.image_times[20]
    t_c3_0 = mv_ds.image_times[40]

    # cam01 anchored at 0.0
    assert t_c1_0 == 0.0
    # cam02: (0 + 0.020 * 20) / 20 = 0.40 / 20 = 0.020
    assert pytest.approx(t_c2_0, rel=1e-5) == 0.020
    # cam03: (0 - 0.015 * 20) / 20 = -0.30 / 20 = -0.015
    assert pytest.approx(t_c3_0, rel=1e-5) == -0.015

    # Times across cameras for frame 0 must be strictly distinct
    assert t_c1_0 != t_c2_0
    assert t_c2_0 != t_c3_0
    assert t_c1_0 != t_c3_0

    # Test physical times
    phys_c1_0 = mv_ds.get_physical_time(0)
    phys_c2_0 = mv_ds.get_physical_time(20)
    phys_c3_0 = mv_ds.get_physical_time(40)

    assert phys_c1_0 == 0.0
    assert pytest.approx(phys_c2_0, rel=1e-5) == 0.020
    assert pytest.approx(phys_c3_0, rel=1e-5) == -0.015

    # Test frame 10 (i=10)
    # cam01 frame 10 is index 10
    # cam02 frame 10 is index 30
    # cam03 frame 10 is index 50
    t_c1_10 = mv_ds.image_times[10]
    t_c2_10 = mv_ds.image_times[30]
    t_c3_10 = mv_ds.image_times[50]

    expected_c1_10 = 10.0 / denom  # 0.5
    expected_c2_10 = (10.0 + 0.020 * fps) / denom  # (10 + 0.4) / 20 = 0.52
    expected_c3_10 = ((1.0 + 0.0001) * 10.0 + (-0.015) * fps) / denom  # (10.001 - 0.3) / 20 = 0.48505

    assert pytest.approx(t_c1_10, rel=1e-5) == expected_c1_10
    assert pytest.approx(t_c2_10, rel=1e-5) == expected_c2_10
    assert pytest.approx(t_c3_10, rel=1e-5) == expected_c3_10


def test_backward_compatibility_no_offsets(tmp_path):
    """Verify that absent offsets file results in exact legacy shared timestamps."""
    scene_dir, extrinsics, intrinsics = _build_mock_multicam_scene(tmp_path, num_cams=2, num_frames=10)

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
    )

    assert mv_ds.has_offsets is False
    assert mv_ds.fps == 20.0
    assert mv_ds.offsets_s == {}

    # For frame i, all cameras have identical time i / image_length
    for i in range(10):
        t_c1 = mv_ds.image_times[i]
        t_c2 = mv_ds.image_times[10 + i]
        assert t_c1 == t_c2
        assert pytest.approx(t_c1, rel=1e-5) == float(i / 10.0)


def test_ablation_disabled_offsets(tmp_path, monkeypatch):
    """Verify that use_camera_offsets=False and PIPELINE_USE_CAMERA_OFFSETS="0" disable offsets."""
    scene_dir, extrinsics, intrinsics = _build_mock_multicam_scene(tmp_path, num_cams=2, num_frames=10)

    offsets_payload = {
        "fps": 20.0,
        "offsets_s": {"cam01": 0.0, "cam02": 0.020},
    }
    (scene_dir / "offsets.json").write_text(json.dumps(offsets_payload), encoding="utf-8")

    # 1. Direct parameter toggle
    mv_ds_param = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
        use_camera_offsets=False,
    )
    assert mv_ds_param.has_offsets is False
    assert mv_ds_param.offsets_enabled is False
    assert mv_ds_param.image_times[0] == mv_ds_param.image_times[10]  # Frame 0 simultaneous

    # 2. Environment variable toggle
    monkeypatch.setenv("PIPELINE_USE_CAMERA_OFFSETS", "0")
    mv_ds_env = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
        use_camera_offsets=True,
    )
    assert mv_ds_env.has_offsets is False
    assert mv_ds_env.offsets_enabled is False
    assert mv_ds_env.image_times[0] == mv_ds_env.image_times[10]


def test_fourdgsdataset_and_camera_wiring(tmp_path):
    """Verify FourDGSdataset wires normalized and physical times to Camera."""
    scene_dir, extrinsics, intrinsics = _build_mock_multicam_scene(tmp_path, num_cams=2, num_frames=10)

    offsets_payload = {
        "fps": 20.0,
        "offsets_s": {"cam01": 0.0, "cam02": 0.015},
    }
    (scene_dir / "offsets.json").write_text(json.dumps(offsets_payload), encoding="utf-8")

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
    )

    four_ds = FourDGSdataset(mv_ds, args=None, dataset_type="MultipleView")

    assert four_ds.fps == 20.0
    assert four_ds.has_offsets is True
    assert four_ds.offsets_s["cam02"] == 0.015

    # Retrieve Camera 1 frame 0
    cam1 = four_ds[0]
    assert isinstance(cam1, Camera)
    assert cam1.time == 0.0
    assert cam1.physical_time == 0.0

    # Retrieve Camera 2 frame 0 (index 10)
    cam2 = four_ds[10]
    assert isinstance(cam2, Camera)
    expected_norm = 0.015 * 20.0 / 10.0  # 0.03
    assert pytest.approx(cam2.time, rel=1e-5) == expected_norm
    assert pytest.approx(cam2.physical_time, rel=1e-5) == 0.015

    # Retrieve Camera 2 frame 5 (index 15)
    cam2_f5 = four_ds[15]
    expected_norm_f5 = (5.0 + 0.015 * 20.0) / 10.0  # 0.53
    expected_phys_f5 = 5.0 / 20.0 + 0.015  # 0.265
    assert pytest.approx(cam2_f5.time, rel=1e-5) == expected_norm_f5
    assert pytest.approx(cam2_f5.physical_time, rel=1e-5) == expected_phys_f5


def test_config_models_and_bridge_generation():
    """Verify OptimizationParams/ModelParams schema and bridge generation for use_camera_offsets."""
    opt = OptimizationParams()
    assert opt.use_camera_offsets is True

    model = ModelParams()
    assert model.use_camera_offsets is True

    opt_custom = OptimizationParams(use_camera_offsets=False)
    assert opt_custom.use_camera_offsets is False

    cfg = PipelineConfig(
        model=ModelParams(use_camera_offsets=True),
        optim=OptimizationParams(use_camera_offsets=False),
    )
    bridge_src = render_bridge_source(cfg.model_dump())
    assert "'use_camera_offsets': True" in bridge_src
    assert "'use_camera_offsets': False" in bridge_src
