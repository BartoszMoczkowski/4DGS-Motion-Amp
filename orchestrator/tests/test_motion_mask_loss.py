"""Tests for Task T36 (Motion-mask weighted loss - R2).

Verifies:
1. Static synthetic input: masked L1 loss reduces identically to stock L1 loss.
2. Weight scaling: moving region weights scale proportionally up to motion_mask_max (e.g. 50x).
3. Gradient propagation: gradients in moving regions scale proportionally vs background.
4. Mask shape broadcasting: 2D [H, W], 3D [1, H, W], and 4D [B, 1, H, W] broadcast seamlessly.
5. Soft ROI fallback: produces smooth, normalized falloff rectangle around turbine bounds.
6. Bandpass variance on static signal: zero motion produces zero variance / unit weight.
7. Dataloader integration: multipleview_dataset and FourDGSdataset retrieve and wire masks to Camera.
8. Pydantic config and bridge file generation: OptimizationParams properly serialize R2 fields.
9. YAML presets: lego_turbine_2_missing_r2 and related presets resolve with expected R2 configs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
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
    import types
    m_knn = types.ModuleType("simple_knn")
    m_knn_c = types.ModuleType("simple_knn._C")
    m_knn_c.distCUDA2 = lambda *args, **kwargs: None
    m_knn._C = m_knn_c
    sys.modules["simple_knn"] = m_knn
    sys.modules["simple_knn._C"] = m_knn_c

from pipeline.config import validate_config, OptimizationParams
from pipeline.config.bridge import render_bridge_source
from utils.loss_utils import l1_loss
from scripts.precompute_motion_masks import (
    compute_camera_bandpass_variance,
    postprocess_motion_mask,
    generate_soft_roi_mask,
    get_camera_rig_roi,
)


def test_static_synthetic_loss_reduces_to_stock():
    """Verify that on static inputs (mask=None or uniform 1.0), masked loss is identical to stock."""
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    pred = torch.randn(B, C, H, W, requires_grad=True)
    gt = torch.randn(B, C, H, W)

    stock_loss = l1_loss(pred, gt)
    masked_loss_none = l1_loss(pred, gt, mask=None)
    masked_loss_ones = l1_loss(pred, gt, mask=torch.ones((B, 1, H, W)))
    masked_loss_2d_ones = l1_loss(pred, gt, mask=torch.ones((H, W)))

    assert torch.isclose(stock_loss, masked_loss_none, atol=1e-7)
    assert torch.isclose(stock_loss, masked_loss_ones, atol=1e-7)
    assert torch.isclose(stock_loss, masked_loss_2d_ones, atol=1e-7)


def test_masked_loss_weight_scaling():
    """Verify that normalized masked loss preserves overall scale while weighting moving regions by max_weight."""
    B, C, H, W = 1, 3, 32, 32
    pred = torch.zeros(B, C, H, W)
    gt = torch.ones(B, C, H, W)  # uniform error of 1.0 everywhere

    # Create mask: left half static (1.0), right half moving (50.0)
    weight_map = torch.ones(B, 1, H, W)
    weight_map[:, :, :, 16:] = 50.0

    # 1. Scale preservation on uniform error: sum(w * 1) / sum(w) = 1.0 (no artificial 25.5x inflation)
    loss_uniform = l1_loss(pred, gt, mask=weight_map)
    assert torch.isclose(loss_uniform, torch.tensor(1.0), atol=1e-5)

    # 2. Moving vs static region loss weighting: moving error yields 50x higher loss than static error
    pred_static_err = gt.clone()
    pred_static_err[:, :, :, :16] = 0.0  # error 1.0 on static half (weight 1.0)
    loss_static_err = l1_loss(pred_static_err, gt, mask=weight_map)

    pred_moving_err = gt.clone()
    pred_moving_err[:, :, :, 16:] = 0.0  # error 1.0 on moving half (weight 50.0)
    loss_moving_err = l1_loss(pred_moving_err, gt, mask=weight_map)

    assert torch.isclose(loss_moving_err / loss_static_err, torch.tensor(50.0), atol=1e-4)


def test_masked_loss_gradient_scaling():
    """Verify that gradient magnitudes scale proportionally in the moving region."""
    H, W = 16, 16
    pred = torch.zeros(1, 1, H, W, requires_grad=True)
    gt = torch.ones(1, 1, H, W)

    # Top half weight 1.0, bottom half weight 50.0
    weight_map = torch.ones(1, 1, H, W)
    weight_map[:, :, 8:, :] = 50.0

    loss = l1_loss(pred, gt, mask=weight_map)
    loss.backward()

    grad = pred.grad
    top_grad = grad[:, :, :8, :].abs().mean()
    bottom_grad = grad[:, :, 8:, :].abs().mean()

    assert torch.isclose(bottom_grad / top_grad, torch.tensor(50.0), atol=1e-4)


def test_mask_shapes_and_broadcasting():
    """Verify support for 2D [H, W], 3D [1, H, W], and 4D [B, 1, H, W] masks."""
    B, C, H, W = 2, 3, 20, 20
    pred = torch.randn(B, C, H, W)
    gt = torch.randn(B, C, H, W)

    mask_2d = torch.rand(H, W) + 1.0
    mask_3d = mask_2d.unsqueeze(0)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

    loss_2d = l1_loss(pred, gt, mask=mask_2d)
    loss_3d = l1_loss(pred, gt, mask=mask_3d)
    loss_4d = l1_loss(pred, gt, mask=mask_4d)

    assert torch.isclose(loss_2d, loss_3d, atol=1e-6)
    assert torch.isclose(loss_2d, loss_4d, atol=1e-6)


def test_soft_roi_fallback():
    """Verify that generate_soft_roi_mask produces a smooth bounded mask in [0, 1]."""
    h, w = 180, 320
    roi_lower = generate_soft_roi_mask(h, w, cam_idx=1, blur_sigma=10.0)
    roi_upper = generate_soft_roi_mask(h, w, cam_idx=6, blur_sigma=10.0)

    assert roi_lower.shape == (h, w)
    assert roi_upper.shape == (h, w)
    assert roi_lower.min() >= 0.0
    assert roi_lower.max() == pytest.approx(1.0)
    assert roi_upper.min() >= 0.0
    assert roi_upper.max() == pytest.approx(1.0)

    # Upper and lower ring cameras have different vertical centers
    y_lower_center = np.argmax(roi_lower.sum(axis=1))
    y_upper_center = np.argmax(roi_upper.sum(axis=1))
    assert y_upper_center < y_lower_center


def test_bandpass_variance_on_static_frames(tmp_path):
    """Verify that pure static frames produce zero variance and zero mask."""
    import cv2
    cam_dir = tmp_path / "cam01"
    cam_dir.mkdir()

    # Create 10 identical static frames
    static_frame = (np.ones((60, 80), dtype=np.uint8) * 128)
    frame_paths = []
    for i in range(10):
        p = cam_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(p), static_frame)
        frame_paths.append(p)

    vmap = compute_camera_bandpass_variance(frame_paths, fps=20.0, chunk_rows=30)
    assert vmap.shape == (60, 80)
    assert vmap.max() == pytest.approx(0.0, abs=1e-7)

    mask = postprocess_motion_mask(vmap, cam_idx=1)
    assert mask.shape == (60, 80)
    assert mask.max() == 0.0  # Zero motion yields zero mask (stock loss weight 1.0)


def test_optimization_params_pydantic_model():
    """Verify that OptimizationParams validates R2 parameters and defaults."""
    opt = OptimizationParams()
    assert opt.use_motion_mask is False
    assert opt.motion_mask_dir == ""
    assert opt.motion_mask_max == pytest.approx(50.0)

    opt_custom = OptimizationParams(use_motion_mask=True, motion_mask_max=20.0, motion_mask_dir="/path/to/masks")
    assert opt_custom.use_motion_mask is True
    assert opt_custom.motion_mask_max == pytest.approx(20.0)
    assert opt_custom.motion_mask_dir == "/path/to/masks"


def test_bridge_source_renders_motion_mask_params():
    """Verify that bridge file renders OptimizationParams with motion mask settings."""
    cfg = validate_config("lego_turbine_2_missing_r2")
    source = render_bridge_source(cfg.model_dump())

    assert "OptimizationParams = " in source
    assert "'use_motion_mask': True" in source
    assert "'motion_mask_max': 50.0" in source


def test_preset_ladder_resolution():
    """Verify that presets resolve the correct ladder steps for stock, R1, and R2."""
    stock = validate_config("lego_turbine_2_missing_stock")
    assert stock.hidden.regularizer_anneal_iter == 0
    assert stock.optim.use_motion_mask is False

    r1 = validate_config("lego_turbine_2_missing_r1")
    assert r1.hidden.regularizer_anneal_iter == 3000
    assert r1.hidden.time_smoothness_weight_final == pytest.approx(0.0001)
    assert r1.optim.use_motion_mask is False

    r2 = validate_config("lego_turbine_2_missing_r2")
    assert r2.hidden.regularizer_anneal_iter == 3000
    assert r2.optim.use_motion_mask is True
    assert r2.optim.motion_mask_max == pytest.approx(50.0)


def test_dataset_mask_plumbing(tmp_path):
    """Verify that multipleview_dataset and FourDGSdataset wire mask to Camera."""
    import cv2
    from PIL import Image
    from scene.multipleview_dataset import multipleview_dataset
    from scene.dataset import FourDGSdataset
    from scene.dataset_readers import CameraInfo

    scene_dir = tmp_path / "mock_scene"
    scene_dir.mkdir()
    cam_dir = scene_dir / "cam01"
    cam_dir.mkdir()

    # 2 mock frames
    img = np.ones((40, 50, 3), dtype=np.uint8) * 100
    cv2.imwrite(str(cam_dir / "frame_00001.jpg"), img)
    cv2.imwrite(str(cam_dir / "frame_00002.jpg"), img)

    # Create motion_masks folder
    mask_dir = scene_dir / "motion_masks"
    mask_dir.mkdir()
    mock_mask = np.ones((40, 50), dtype=np.float32) * 0.8
    np.save(str(mask_dir / "cam01.npy"), mock_mask)

    class MockExtrinsic:
        qvec = [1, 0, 0, 0]
        tvec = [0, 0, 0]
        name = "frame01.jpg"

    class MockIntrinsic:
        params = [500.0]
        height = 40
        width = 50

    extrinsics = {1: MockExtrinsic()}
    intrinsics = {1: MockIntrinsic()}

    mv_ds = multipleview_dataset(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        cam_folder=str(scene_dir),
        split="train",
    )

    assert "cam01" in mv_ds.camera_masks
    mask_tensor = mv_ds.get_mask(0)
    assert mask_tensor is not None
    assert mask_tensor.shape == (40, 50)
    assert torch.isclose(mask_tensor[0, 0], torch.tensor(0.8))

    four_ds = FourDGSdataset(mv_ds, args=None, dataset_type="MultipleView")
    cam = four_ds[0]
    assert hasattr(cam, "mask")
    assert cam.mask is not None
    assert torch.isclose(cam.mask[0, 0], torch.tensor(0.8))
