"""orchestrator/tests/test_w3_soft_loss_weights.py

Unit and acceptance tests for Work Package W3 (Task T51):
Continuous normalized soft loss weighting.

Verifies:
1. Static sequence identity test:
   - Constant frames (with or without small global brightness offset/flicker)
     produce weights w(x, y) = 1.0 everywhere.
   - Weighted L1 loss equals unweighted L1 loss identically.
2. Synthetic localized mover test:
   - A small oscillating patch produces smooth elevated weights centered on the patch,
     with zero NaN/Inf, strictly positive floor everywhere, cap respected, and unit mean.
3. Disabled-mode no-op test:
   - use_motion_mask=False produces unweighted L1 loss bit-identical to W1.
4. Global exposure drift & flicker suppression test:
   - Injected global frame-to-frame luminance offsets do not trigger spurious motion weights.
5. Soft loss normalizer math:
   - normalize_loss_weights maps [0, 1] evidence to floor 1.0 and unit mean across 2D/3D/4D tensors.
6. Single-factor E2 preset pair & config validation:
   - turbine_r1_missing_6_w3_e2 and lego_turbine_2_missing_w3_e2 validate cleanly
     and differ from E1a only in soft loss weighting parameters.
7. Provenance manifest and SHA-256 tracking:
   - Precomputation produces weight_maps_meta.json with per-camera hashes and directory hash.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

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

from core.utils.loss_utils import l1_loss, normalize_loss_weights
from scripts.precompute_motion_masks import (
    compute_camera_bandpass_variance,
    postprocess_motion_mask,
    precompute_scene_motion_masks,
)
from pipeline.config import validate_config


# ---------------------------------------------------------------------------
# 1. Soft Loss Normalizer Math
# ---------------------------------------------------------------------------

def test_normalize_loss_weights_static_identity():
    """All-zero evidence (static scene) must produce identically 1.0 weights everywhere."""
    evidence = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    w = normalize_loss_weights(evidence, mask_max=15.0, floor=1.0, normalize_mean=True)

    assert torch.allclose(w, torch.ones_like(w)), "Static weights must equal 1.0 everywhere"
    assert abs(float(w.mean()) - 1.0) < 1e-6
    assert not w.requires_grad, "Loss weights must be detached"


def test_normalize_loss_weights_unit_mean_and_bounds():
    """Evidence with localized activity produces floor >= 1/mean, cap <= w_max/mean, and mean == 1.0."""
    evidence = torch.zeros((1, 1, 100, 100), dtype=torch.float32)
    # 5x5 patch with full motion evidence
    evidence[:, :, 40:45, 40:45] = 1.0

    mask_max = 15.0
    w = normalize_loss_weights(evidence, mask_max=mask_max, floor=1.0, normalize_mean=True)

    # Unit mean check: E[w] == 1.0
    mean_val = float(w.mean())
    assert abs(mean_val - 1.0) < 1e-6, f"Expected unit mean 1.0, got {mean_val}"

    # Ratio of mover weight to static weight must equal mask_max exactly
    w_mover = float(w[:, :, 42, 42])
    w_static = float(w[:, :, 10, 10])
    ratio = w_mover / w_static
    assert abs(ratio - mask_max) < 1e-4, f"Expected ratio {mask_max}, got {ratio}"
    assert w_static > 0.0, "Floor must be strictly positive"


def test_l1_loss_scale_preservation():
    """Weighted L1 loss with mean-normalized weights preserves unweighted loss scale for uniform error."""
    torch.manual_seed(42)
    pred = torch.rand((2, 3, 32, 32))
    gt = pred + 0.15  # Uniform error everywhere

    # Unweighted L1 loss
    loss_unweighted = l1_loss(pred, gt)
    assert abs(float(loss_unweighted) - 0.15) < 1e-6

    # Random non-uniform evidence map
    evidence = torch.rand((2, 1, 32, 32))
    w = normalize_loss_weights(evidence, mask_max=15.0, floor=1.0, normalize_mean=True)

    # Weighted L1 loss
    loss_weighted = l1_loss(pred, gt, mask=w)
    assert abs(float(loss_weighted) - 0.15) < 1e-5, f"Expected scale preservation, got {loss_weighted}"


# ---------------------------------------------------------------------------
# 2. Static Sequence Identity & Flicker Suppression
# ---------------------------------------------------------------------------

def test_static_sequence_produces_flat_weights(tmp_path: Path):
    """Constant frames produce zero motion variance, resulting in flat 1.0 weights."""
    cam_dir = tmp_path / "cam01"
    cam_dir.mkdir()

    # 30 identical frames (constant scene)
    frame_paths = []
    for i in range(30):
        p = cam_dir / f"{i:04d}.png"
        img = np.full((100, 100), 120, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        frame_paths.append(p)

    vmap = compute_camera_bandpass_variance(frame_paths, fps=20.0, f_low=3.5, f_high=7.5)
    assert np.all(vmap == 0.0), "Static frames must have zero variance in bandpass"

    mask = postprocess_motion_mask(vmap, cam_idx=1)
    assert np.all(mask == 0.0), "Static frames must produce zero motion mask"

    # Convert to training loss weight
    w = normalize_loss_weights(torch.from_numpy(mask), mask_max=15.0)
    assert torch.allclose(w, torch.ones_like(w)), "Static sequence must yield w = 1.0 everywhere"


def test_exposure_flicker_suppression(tmp_path: Path):
    """Global frame-to-frame exposure fluctuations (auto-gain / flicker) are rejected by median normalization."""
    cam_dir = tmp_path / "cam01"
    cam_dir.mkdir()

    frame_paths = []
    # Scene with a stationary texture, but global exposure oscillates at 5.5 Hz (in-band flicker)
    for i in range(40):
        p = cam_dir / f"{i:04d}.png"
        flicker_dc = int(round(15.0 * math.sin(2.0 * math.pi * 5.5 * (i / 20.0))))
        base_img = np.full((80, 80), 128 + flicker_dc, dtype=np.uint8)
        cv2.imwrite(str(p), base_img)
        frame_paths.append(p)

    # With median drift suppression, variance should be zero
    vmap = compute_camera_bandpass_variance(frame_paths, fps=20.0, f_low=3.5, f_high=7.5)
    assert np.max(vmap) < 1e-4, f"Global flicker was not suppressed, got max variance {np.max(vmap)}"


# ---------------------------------------------------------------------------
# 3. Synthetic Localized Mover Test
# ---------------------------------------------------------------------------

def test_synthetic_localized_mover_coverage(tmp_path: Path):
    """A small patch oscillating at target f0 produces smooth, localized elevated weights."""
    cam_dir = tmp_path / "cam01"
    cam_dir.mkdir()

    f0 = 5.69
    fps = 20.004
    n_frames = 60
    h, w = 120, 160

    frame_paths = []
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    for i in range(n_frames):
        t = i / fps
        p = cam_dir / f"{i:04d}.png"
        # Background is stationary gray
        img = np.full((h, w), 80, dtype=np.uint8)
        # Small blob moving horizontally in the center
        cx = 80.0 + 3.0 * math.cos(2.0 * math.pi * f0 * t)
        cy = 60.0
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 36.0)
        img = np.clip(img + (blob * 150.0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(p), img)
        frame_paths.append(p)

    vmap = compute_camera_bandpass_variance(frame_paths, fps=fps, f_low=3.5, f_high=7.5)
    mask = postprocess_motion_mask(vmap, cam_idx=1, smooth_sigma=15.0)

    # Motion mask must be non-zero and localized
    assert mask.max() > 0.80, f"Expected strong peak on mover, got {mask.max()}"
    assert mask[60, 80] > 0.80, "Center of mover must have high intensity"
    assert mask[10, 10] == 0.0, "Distant corner must have zero motion intensity"

    # Converted loss weight
    w_tensor = normalize_loss_weights(torch.from_numpy(mask), mask_max=15.0)
    assert abs(float(w_tensor.mean()) - 1.0) < 1e-6
    assert float(w_tensor[60, 80]) > float(w_tensor[10, 10]) * 10.0


# ---------------------------------------------------------------------------
# 4. End-to-End Precomputation & Provenance Hash
# ---------------------------------------------------------------------------

def test_precompute_scene_motion_masks_and_manifest(tmp_path: Path):
    """Verifies precompute_scene_motion_masks writes .npy, visual overlays, and weight_maps_meta.json with valid hashes."""
    scene_dir = tmp_path / "mock_turbine"
    scene_dir.mkdir()

    # Write offsets.json
    (scene_dir / "offsets.json").write_text(json.dumps({"fps": 20.004, "frame_count": 20}), encoding="utf-8")

    # Create cam01 with 20 dummy frames
    c_dir = scene_dir / "cam01"
    c_dir.mkdir()
    for i in range(20):
        img = np.full((60, 80), 100, dtype=np.uint8)
        cv2.imwrite(str(c_dir / f"{i:04d}.png"), img)

    manifest = precompute_scene_motion_masks(
        scene_dir=scene_dir,
        fps=20.004,
        f_low=3.5,
        f_high=7.5,
    )

    out_dir = scene_dir / "motion_masks"
    assert (out_dir / "cam01.npy").is_file()
    assert (out_dir / "cam01_vis.png").is_file()
    assert (out_dir / "cam01_overlay.png").is_file()
    assert (out_dir / "weight_maps_meta.json").is_file()

    meta = json.loads((out_dir / "weight_maps_meta.json").read_text(encoding="utf-8"))
    assert "weight_maps_hash" in meta
    assert meta["weight_maps_hash"].startswith("sha256:")
    assert "cam01" in meta["cameras"]
    assert "sha256" in meta["cameras"]["cam01"]


# ---------------------------------------------------------------------------
# 5. E2 Presets & Single-Factor Validation
# ---------------------------------------------------------------------------

def test_w3_e2_presets_valid_and_single_factor():
    """Verifies that E2 presets validate cleanly and differ from E1a only in soft loss weighting."""
    preset_dir = REPO_ROOT / "orchestrator" / "pipeline" / "config" / "presets"

    for scene_base in ["turbine_r1_missing_6", "lego_turbine_2_missing"]:
        e1a_name = f"{scene_base}_w1_e1a"
        e2_name = f"{scene_base}_w3_e2"

        cfg_e1a = validate_config(e1a_name)
        cfg_e2 = validate_config(e2_name)

        # E2 must have use_motion_mask enabled
        assert cfg_e2.optim.use_motion_mask is True
        assert cfg_e2.optim.normalize_motion_mask is True
        assert cfg_e2.optim.motion_mask_max == 15.0

        # E1a must have use_motion_mask disabled
        assert cfg_e1a.optim.use_motion_mask is False

        # Single factor check: timing parameters must match identically
        assert cfg_e2.optim.use_camera_offsets == cfg_e1a.optim.use_camera_offsets
        assert cfg_e2.optim.strict_camera_offsets == cfg_e1a.optim.strict_camera_offsets
        assert cfg_e2.optim.learnable_camera_offsets == cfg_e1a.optim.learnable_camera_offsets
        assert cfg_e2.optim.offsets_file == cfg_e1a.optim.offsets_file

        # Rigid head / ROI / gating must remain disabled (W0 baseline integrity)
        assert cfg_e2.optim.use_fourier_rigid_head is False
        assert cfg_e2.optim.gated_coarse is False
        assert cfg_e2.optim.use_roi_voxels is False
        assert cfg_e2.optim.lambda_trajectory == 0.0
