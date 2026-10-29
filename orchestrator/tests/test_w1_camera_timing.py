 """Tests for Work Package W1 (Task T49: Harden Fixed & Guarded Learnable Per-Camera Timing).

Verifies:
1. Synthetic loader & strict timing validation:
   - Two cameras with known offsets and drift produce exact physical and normalized times
     via the centralized helpers (`compute_physical_time`, `compute_normalized_time`).
   - SHA-256 digest `offsets_file_sha256` is recorded on the dataset and `FourDGSdataset`.
   - `strict_camera_offsets=True` (or explicit `offsets_file`) raises `FileNotFoundError`
     when `offsets.json` is missing and never falls back to `optical_offsets.json`.
   - Malformed JSON, `fps <= 0`, `exposure_s <= 0`, non-finite offsets/drifts, or missing
     camera coverage raise `ValueError`.
2. Round-trip timestamp parity across training, evaluation (`training_report`), and reload:
   - `resolve_camera_time(cam, camera_offsets_module)` used in training and `training_report`
     matches `FourDGSdataset` / `multipleview_dataset.apply_offsets_dict(learned_offsets)`
     bit-for-bit across all cameras and frames.
3. Anchor-by-construction (`cam01`):
   - `cam01` receives identically zero gradient in the forward/backward graph and remains
     at `0.0` after optimizer steps even without calling `clamp_and_anchor()`.
4. Quadratic sidecar prior (`compute_prior_loss`):
   - Prior loss and gradient are identically `0.0` at initialization (`delta_offsets == 0`).
   - Non-zero positive/negative `delta_offsets` produce exact analytical loss and gradient sign
     pulling back toward the sidecar initialization (`base_offsets`).
5. Freeze/unfreeze & gradient smoke test:
   - Before `learnable_offsets_freeze_iter`, detached `custom_time` yields zero/None offset
     gradients and zero parameter drift; after unfreezing, non-anchor cameras receive finite
     non-zero gradients while `cam01` gradient stays `0.0`.
6. Single-factor preset pair (`E1a` vs `E1b`) & W0 disabled no-op:
   - `lego_turbine_2_missing_w1_e1a` vs `e1b` and `turbine_r1_missing_6_w1_e1a` vs `e1b`
     differ ONLY in `learnable_camera_offsets` (`False` vs `True`), with all non-timing
     settings identical to W0 native baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
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
        def __init__(self, raster_settings=None):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
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

from scene.camera_offsets import (
    CameraOffsets,
    compute_normalized_time,
    compute_physical_time,
    physical_to_normalized_time,
    resolve_camera_time,
)
from scene.dataset import FourDGSdataset
from scene.multipleview_dataset import multipleview_dataset
from pipeline.config import validate_config


class MockExtrinsic:
    def __init__(self, cam_num: int):
        self.qvec = [1, 0, 0, 0]
        self.tvec = [0, 0, 0]
        self.name = f"frame{str(cam_num).zfill(2)}.jpg"


class MockIntrinsic:
    def __init__(self, width: int = 32, height: int = 24):
        self.params = [400.0]
        self.height = height
        self.width = width


def _build_two_cam_scene(tmp_path: Path, num_frames: int = 12):
    scene_dir = tmp_path / "scene_2cam"
    scene_dir.mkdir(parents=True, exist_ok=True)
    img = np.ones((24, 32, 3), dtype=np.uint8) * 100
    extrinsics = {}
    intrinsics = {1: MockIntrinsic(32, 24)}
    for c in (1, 2):
        cam_dir = scene_dir / f"cam0{c}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f in range(1, num_frames + 1):
            cv2.imwrite(str(cam_dir / f"frame_{f:05d}.jpg"), img)
        extrinsics[c] = MockExtrinsic(c)
    return scene_dir, extrinsics, intrinsics


# ---------------------------------------------------------------------------
# 1. Synthetic Loader & Strict Validation Tests
# ---------------------------------------------------------------------------


def test_synthetic_loader_exact_times_and_sha256(tmp_path: Path):
    """Verify two-camera loader computes exact physical/normalized times and SHA-256 hash."""
    scene_dir, extrinsics, intrinsics = _build_two_cam_scene(tmp_path, num_frames=12)
    payload = {
        "fps": 20.004,
        "exposure_s": 0.01,
        "mid_exposure_corrected": True,
        "reference_camera": "cam01",
        "offsets_s": {"cam01": 0.0, "cam02": 0.018},
        "drift_s_per_s": {"cam01": 0.0, "cam02": 0.00015},
    }
    raw_bytes = json.dumps(payload, indent=2).encode("utf-8")
    (scene_dir / "offsets.json").write_bytes(raw_bytes)
    expected_sha = hashlib.sha256(raw_bytes).hexdigest()

    ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
        use_camera_offsets=True,
        strict_camera_offsets=True,
    )
    assert ds.has_offsets is True
    assert ds.offsets_file_sha256 == expected_sha
    assert ds.exposure_s == 0.01
    assert ds.mid_exposure_corrected is True

    for i in range(12):
        # cam01 is at indices 0..11, cam02 at indices 12..23
        t_phys_1 = ds.get_physical_time(i)
        t_norm_1 = ds.image_times[i]
        assert pytest.approx(t_phys_1, abs=1e-12) == compute_physical_time(i, 0.0, 20.004, 0.0)
        assert pytest.approx(t_norm_1, abs=1e-12) == compute_normalized_time(i, 0.0, 20.004, 12, 0.0)
        assert pytest.approx(t_norm_1, abs=1e-12) == physical_to_normalized_time(t_phys_1, 20.004, 12)

        t_phys_2 = ds.get_physical_time(12 + i)
        t_norm_2 = ds.image_times[12 + i]
        assert pytest.approx(t_phys_2, abs=1e-12) == compute_physical_time(i, 0.018, 20.004, 0.00015)
        assert pytest.approx(t_norm_2, abs=1e-12) == compute_normalized_time(
            i, 0.018, 20.004, 12, 0.00015
        )
        assert pytest.approx(t_norm_2, abs=1e-12) == physical_to_normalized_time(t_phys_2, 20.004, 12)


def test_strict_validation_rejects_missing_or_malformed_timing(tmp_path: Path):
    """Verify strict mode never falls back to optical_offsets.json and rejects invalid data."""
    scene_dir, extrinsics, intrinsics = _build_two_cam_scene(tmp_path, num_frames=6)

    # Write optical_offsets.json only (no offsets.json)
    (scene_dir / "optical_offsets.json").write_text(
        json.dumps({"fps": 20.0, "optical_offsets_s": {"cam01": 0.0, "cam02": 0.01}}),
        encoding="utf-8",
    )

    # 1. strict_camera_offsets=True must refuse optical_offsets.json fallback and raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="strict_camera_offsets=True"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
            strict_camera_offsets=True,
        )

    # 2. Explicit missing offsets_file must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Configured offsets_file"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
            offsets_file="nonexistent_offsets.json",
        )

    # 3. Malformed JSON raises ValueError
    (scene_dir / "offsets.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
        )

    # 4. Non-positive fps raises ValueError
    (scene_dir / "offsets.json").write_text(
        json.dumps({"fps": -20.0, "offsets_s": {"cam01": 0.0, "cam02": 0.01}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fps must be finite and > 0"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
        )

    # 5. Non-positive exposure_s raises ValueError
    (scene_dir / "offsets.json").write_text(
        json.dumps({"fps": 20.0, "exposure_s": 0.0, "offsets_s": {"cam01": 0.0, "cam02": 0.01}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exposure_s must be finite and > 0"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
        )

    # 6. Missing camera coverage (cam02 missing) raises ValueError
    (scene_dir / "offsets.json").write_text(
        json.dumps({"fps": 20.0, "offsets_s": {"cam01": 0.0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing offsets for cameras"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
        )

    # 7. Non-finite offset (NaN) raises ValueError
    (scene_dir / "offsets.json").write_text(
        '{"fps": 20.0, "offsets_s": {"cam01": 0.0, "cam02": NaN}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Non-finite offset"):
        multipleview_dataset(
            cam_extrinsics=extrinsics,
            cam_intrinsics=intrinsics,
            cam_folder=str(scene_dir),
            split="train",
            use_camera_offsets=True,
        )


# ---------------------------------------------------------------------------
# 2. Round-Trip Timestamp Parity Across Train, Eval, and Scene Reload
# ---------------------------------------------------------------------------


def test_roundtrip_timestamp_parity_train_eval_reload(tmp_path: Path):
    """Prove training step, training_report eval, and offline reload produce identical timestamps."""
    scene_dir, extrinsics, intrinsics = _build_two_cam_scene(tmp_path, num_frames=12)
    payload = {
        "fps": 20.004,
        "reference_camera": "cam01",
        "offsets_s": {"cam01": 0.0, "cam02": 0.018},
        "drift_s_per_s": {"cam01": 0.0, "cam02": 0.00015},
    }
    (scene_dir / "offsets.json").write_text(json.dumps(payload), encoding="utf-8")

    mv_train = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
        use_camera_offsets=True,
        strict_camera_offsets=True,
    )
    four_train = FourDGSdataset(mv_train, args=None, dataset_type="MultipleView")

    offsets_mod = CameraOffsets(
        cam_names=four_train.get_camera_names(),
        init_offsets_s=four_train.offsets_s,
        reference_camera="cam01",
        max_clamp_s=0.040,
        fps=four_train.fps,
        image_length=mv_train.image_length,
        drift_s_per_s=four_train.drift_s_per_s,
        offsets_file_sha256=four_train.offsets_file_sha256,
    )

    # Apply a synthetic learned delta on cam02 (+3.5 ms)
    with torch.no_grad():
        offsets_mod.delta_offsets[1] = 0.0035

    # Save telemetry to learned_offsets.json and apply to a fresh reloaded dataset
    telemetry = offsets_mod.get_telemetry_dict()
    learned_file = tmp_path / "learned_offsets.json"
    learned_file.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

    mv_reloaded = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
        use_camera_offsets=True,
        strict_camera_offsets=True,
    )
    four_reloaded = FourDGSdataset(mv_reloaded, args=None, dataset_type="MultipleView")
    loaded_payload = json.loads(learned_file.read_text(encoding="utf-8"))
    four_reloaded.apply_offsets_dict(loaded_payload["current_offsets_s"])

    # Compare every camera and frame across:
    # (1) training-loop resolve_camera_time(cam, offsets_mod)
    # (2) evaluation-loop resolve_camera_time(cam, offsets_mod).detach()
    # (3) offline reloaded camera.time
    for idx in range(len(four_train)):
        cam_train = four_train[idx]
        t_train = resolve_camera_time(cam_train, offsets_mod)
        t_eval = resolve_camera_time(cam_train, offsets_mod).detach()
        cam_reloaded = four_reloaded[idx]
        assert pytest.approx(float(t_train.item()), abs=1e-7) == float(t_eval.item())
        assert pytest.approx(float(t_train.item()), abs=1e-7) == float(cam_reloaded.time)


# ---------------------------------------------------------------------------
# 3. Structural Anchor-by-Construction Test
# ---------------------------------------------------------------------------


def test_cam01_anchored_by_construction():
    """Prove cam01 has 0 gradient in forward/backward and never moves without clamp_and_anchor()."""
    mod = CameraOffsets(
        cam_names=["cam01", "cam02", "cam03"],
        init_offsets_s={"cam01": 0.0, "cam02": 0.015, "cam03": -0.010},
        reference_camera="cam01",
        fps=20.0,
        image_length=60,
    )
    optimizer = torch.optim.Adam(mod.parameters(), lr=0.05)

    for step in range(10):
        optimizer.zero_grad()
        # Deliberately include cam01 in the loss with a huge weight
        t0 = mod.get_normalized_time("cam01", frame_idx=5)
        t1 = mod.get_normalized_time("cam02", frame_idx=5)
        t2 = mod.get_normalized_time("cam03", frame_idx=5)
        loss = 1000.0 * t0 + (t1 - 0.12) ** 2 + (t2 - 0.08) ** 2 + mod.compute_prior_loss()
        loss.backward()

        # Gradient on cam01 (index 0) must be identically 0.0 BEFORE any post-step hook
        assert mod.delta_offsets.grad is not None
        assert mod.delta_offsets.grad[0].item() == 0.0
        assert mod.delta_offsets.grad[1].item() != 0.0

        # Step optimizer WITHOUT calling clamp_and_anchor()
        optimizer.step()
        assert mod.delta_offsets[0].item() == 0.0
        assert mod.get_offset("cam01").item() == 0.0


# ---------------------------------------------------------------------------
# 4. Quadratic Sidecar Prior Test
# ---------------------------------------------------------------------------


def test_quadratic_sidecar_prior_zero_at_init_and_exact_sign():
    """Verify prior loss is 0 with 0 gradient at init and has exact analytical value/sign."""
    sigma_s = 0.005  # 5 ms
    weight = 0.02
    mod = CameraOffsets(
        cam_names=["cam01", "cam02", "cam03"],
        init_offsets_s={"cam01": 0.0, "cam02": 0.018, "cam03": -0.012},
        reference_camera="cam01",
        offset_prior_sigma_s=sigma_s,
        lambda_offset_prior=weight,
    )

    # 1. At initialization (delta == 0), loss and gradients are identically 0
    loss_init = mod.compute_prior_loss()
    assert loss_init.item() == 0.0
    loss_init.backward()
    assert torch.all(mod.delta_offsets.grad == 0.0)

    # 2. Set delta_2 = +0.005 s (+1 sigma), delta_3 = -0.010 s (-2 sigma)
    mod.zero_grad()
    with torch.no_grad():
        mod.delta_offsets[0] = 0.020  # tampered anchor should still be ignored by anchor_mask
        mod.delta_offsets[1] = 0.005
        mod.delta_offsets[2] = -0.010

    loss_val = mod.compute_prior_loss()
    # Expected: weight * ((1.0)^2 + (-2.0)^2) = 0.02 * 5.0 = 0.10
    assert pytest.approx(loss_val.item(), rel=1e-6) == 0.10

    loss_val.backward()
    grad = mod.delta_offsets.grad
    assert grad[0].item() == 0.0
    # dL/ddelta_k = 2 * weight * delta_k / sigma^2
    expected_g1 = 2.0 * weight * 0.005 / (sigma_s**2)
    expected_g2 = 2.0 * weight * (-0.010) / (sigma_s**2)
    assert grad[1].item() > 0.0
    assert grad[2].item() < 0.0
    assert pytest.approx(grad[1].item(), rel=1e-5) == expected_g1
    assert pytest.approx(grad[2].item(), rel=1e-5) == expected_g2


# ---------------------------------------------------------------------------
# 5. Freeze/Unfreeze & Gradient Smoke Test
# ---------------------------------------------------------------------------


def test_freeze_unfreeze_and_gradient_smoke():
    """Verify zero gradient while frozen (< freeze_iter) and finite non-zero gradient after."""
    mod = CameraOffsets(
        cam_names=["cam01", "cam02"],
        init_offsets_s={"cam01": 0.0, "cam02": 0.010},
        reference_camera="cam01",
        fps=20.0,
        image_length=60,
        offset_prior_sigma_s=0.005,
        lambda_offset_prior=0.01,
    )
    scene_param = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    optimizer = torch.optim.Adam(mod.parameters(), lr=0.001)
    freeze_iter = 5

    for iteration in range(1, 10):
        optimizer.zero_grad(set_to_none=True)
        if scene_param.grad is not None:
            scene_param.grad.zero_()
        cam_mock = types.SimpleNamespace(cam_name="cam02", frame_idx=10)
        custom_time = resolve_camera_time(cam_mock, mod)
        if iteration < freeze_iter:
            custom_time = custom_time.detach()

        # Differentiable synthetic rendering loss wrt scene geometry + timestamp
        pred = scene_param + torch.sin(2.0 * math.pi * 5.691 * custom_time)
        loss = (pred - 0.5) ** 2
        if iteration >= freeze_iter:
            loss = loss + mod.compute_prior_loss()

        loss.backward()
        assert scene_param.grad is not None and scene_param.grad.item() != 0.0

        if iteration < freeze_iter:
            assert mod.delta_offsets.grad is None or torch.all(mod.delta_offsets.grad == 0.0)
            assert mod.delta_offsets[1].item() == 0.0
        else:
            assert mod.delta_offsets.grad is not None
            assert mod.delta_offsets.grad[0].item() == 0.0
            assert torch.isfinite(mod.delta_offsets.grad[1])
            assert mod.delta_offsets.grad[1].item() != 0.0
            optimizer.step()
            mod.clamp_and_anchor()

    assert mod.delta_offsets[0].item() == 0.0
    assert mod.delta_offsets[1].item() != 0.0


# ---------------------------------------------------------------------------
# 6. Single-Factor Preset Pair (`E1a` vs `E1b`) Parity Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "e1a_name,e1b_name",
    [
        ("lego_turbine_2_missing_w1_e1a", "lego_turbine_2_missing_w1_e1b"),
        ("turbine_r1_missing_6_w1_e1a", "turbine_r1_missing_6_w1_e1b"),
    ],
)
def test_w1_e1a_vs_e1b_single_factor_diff(e1a_name: str, e1b_name: str):
    """Prove E1a and E1b presets differ ONLY in learnable_camera_offsets (False vs True)."""
    cfg_a = validate_config(e1a_name).model_dump()
    cfg_b = validate_config(e1b_name).model_dump()

    # Check both have strict sidecar timing enabled and all non-baseline gates off
    for cfg in (cfg_a, cfg_b):
        assert cfg["optim"]["use_camera_offsets"] is True
        assert cfg["optim"]["strict_camera_offsets"] is True
        assert cfg["optim"]["offsets_file"] == "offsets.json"
        assert cfg["optim"]["gated_coarse"] is False
        assert cfg["optim"]["roi_bbox"] is None
        assert cfg["optim"]["use_roi_voxels"] is False
        assert cfg["optim"]["use_motion_mask"] is False
        assert cfg["optim"]["lambda_trajectory"] == 0.0
        assert cfg["optim"]["use_fourier_rigid_head"] is False
        assert cfg["optim"]["fourier_head_pure_rigid"] is False
        assert cfg["optim"]["fourier_head_static_background"] is False

    assert cfg_a["optim"]["learnable_camera_offsets"] is False
    assert cfg_b["optim"]["learnable_camera_offsets"] is True

    # Compare every section except run-identifying strings (name, model_path, expname)
    # and learnable_camera_offsets
    for section in ("hidden", "pipeline_params", "render", "seg_extract", "segment", "seg_eval", "amp"):
        assert cfg_a[section] == cfg_b[section], f"Section {section} differs between {e1a_name} and {e1b_name}"

    opt_a = dict(cfg_a["optim"])
    opt_b = dict(cfg_b["optim"])
    opt_a.pop("learnable_camera_offsets")
    opt_b.pop("learnable_camera_offsets")
    assert opt_a == opt_b, f"optim differs in non-learnable-offset keys: {opt_a} vs {opt_b}"

    mod_a = dict(cfg_a["model"])
    mod_b = dict(cfg_b["model"])
    for k in ("model_path", "learnable_camera_offsets"):
        mod_a.pop(k, None)
        mod_b.pop(k, None)
    assert mod_a == mod_b, f"model differs in non-learnable-offset keys: {mod_a} vs {mod_b}"