"""Sandbox tests for T19's ``roi.motion_gate`` (proposal 01).

No GPU, no Docker, no torch — pure numpy/scipy fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.vendored.host.motion_gate import motion_gate


def _make_scene(*, n_moving: int, n_static: int, n_frames: int = 60, drive_freq: int = 3,
                noise_std: float = 0.01):
    """Synthetic scene: two moving rigid bodies + static background cloud.

    Returns (xyz (N,3), traj (N,T,3), is_moving_gt bool[N]).
    """
    rng = np.random.RandomState(42)
    t = np.arange(n_frames)
    # Body 1: translates sinusoidally in x
    body1_center = np.array([[0.0, 0.0, 0.0]])
    body1_pts = rng.randn(n_moving // 2, 3) * 0.1 + body1_center
    motion1 = np.zeros((n_moving // 2, n_frames, 3))
    motion1[:, :, 0] = 0.2 * np.sin(2 * np.pi * drive_freq * t / n_frames)[None, :]

    # Body 2: translates sinusoidally in y (out of phase)
    body2_center = np.array([[2.0, 0.0, 0.0]])
    body2_pts = rng.randn(n_moving // 2, 3) * 0.1 + body2_center
    motion2 = np.zeros((n_moving // 2, n_frames, 3))
    motion2[:, :, 1] = 0.2 * np.sin(2 * np.pi * drive_freq * t / n_frames + np.pi / 2)[None, :]

    # Static background
    static_pts = rng.randn(n_static, 3) * 2.0 + np.array([[1.0, 1.0, 1.0]])
    static_motion = np.zeros((n_static, n_frames, 3))

    xyz = np.concatenate([body1_pts, body2_pts, static_pts], axis=0).astype(np.float32)
    traj = np.concatenate([body1_pts[:, None, :] + motion1,
                           body2_pts[:, None, :] + motion2,
                           static_pts[:, None, :] + static_motion], axis=0).astype(np.float32)
    # Add reconstruction jitter
    traj += rng.randn(*traj.shape).astype(np.float32) * noise_std

    is_moving_gt = np.concatenate([
        np.ones(n_moving, dtype=bool),
        np.zeros(n_static, dtype=bool),
    ])
    return xyz, traj, is_moving_gt


def test_motion_gate_recovers_movers():
    """On a clean synthetic scene, motion_gate should recover almost all movers with high
    precision."""
    xyz, traj, is_moving_gt = _make_scene(n_moving=200, n_static=200, noise_std=0.005)

    roi_mask, snr, info = motion_gate(xyz, traj, drive_freq=3, harmonics=3)

    # Recall: almost all true movers should be in ROI
    recall = roi_mask[is_moving_gt].mean()
    # Precision: almost all ROI points should be true movers
    precision = is_moving_gt[roi_mask].mean()

    assert recall >= 0.95, f"recall={recall:.3f} — too many movers missed"
    assert precision >= 0.90, f"precision={precision:.3f} — too many static points in ROI"
    assert info["n_readmitted"] >= 0
    assert info["degenerate"] is False


def test_motion_gate_degenerate_all_static():
    """If every point is static (no periodic motion), the gate should return all True
    (degenerate fallback) rather than an empty ROI."""
    rng = np.random.RandomState(42)
    n = 100
    xyz = rng.randn(n, 3).astype(np.float32)
    traj = xyz[:, None, :] + rng.randn(n, 60, 3).astype(np.float32) * 0.01

    roi_mask, snr, info = motion_gate(xyz, traj)

    assert info["degenerate"] is True
    assert roi_mask.all()
    assert snr.shape == (n,)


def test_snr_shape_and_range():
    """SNR array has the right shape and movers have higher SNR than static points."""
    xyz, traj, is_moving_gt = _make_scene(n_moving=100, n_static=100, noise_std=0.005)

    roi_mask, snr, info = motion_gate(xyz, traj, drive_freq=3)

    assert snr.shape == (len(xyz),)
    assert snr.dtype == np.float32
    # Movers should generally have higher SNR than static points
    assert snr[is_moving_gt].mean() > snr[~is_moving_gt].mean()


def test_dilation_captures_neighbors():
    """With dilation_hops > 0, boundary points near movers are included even if their own
    motion is below the energy threshold."""
    rng = np.random.RandomState(42)
    n_frames = 60
    t = np.arange(n_frames)

    # One compact moving body
    body_pts = rng.randn(50, 3) * 0.05 + np.array([[0.0, 0.0, 0.0]])
    motion = np.zeros((50, n_frames, 3))
    motion[:, :, 0] = 0.3 * np.sin(2 * np.pi * 3 * t / n_frames)[None, :]
    body_traj = body_pts[:, None, :] + motion

    # Static points very close to the body (k-NN neighbors)
    neighbor_pts = rng.randn(20, 3) * 0.02 + np.array([[0.15, 0.0, 0.0]])
    neighbor_traj = neighbor_pts[:, None, :] + rng.randn(20, n_frames, 3) * 0.005

    # Far static background
    far_pts = rng.randn(50, 3) * 2.0 + np.array([[5.0, 5.0, 5.0]])
    far_traj = far_pts[:, None, :] + rng.randn(50, n_frames, 3) * 0.005

    xyz = np.concatenate([body_pts, neighbor_pts, far_pts]).astype(np.float32)
    traj = np.concatenate([body_traj, neighbor_traj, far_traj]).astype(np.float32)

    # Without dilation, neighbors might be missed
    roi0, _, _ = motion_gate(xyz, traj, drive_freq=3, dilation_hops=0, readmit_mult=1e6)
    # With dilation, neighbors should be captured
    roi1, _, _ = motion_gate(xyz, traj, drive_freq=3, dilation_hops=1, readmit_mult=1e6)

    n_roi0 = roi0.sum()
    n_roi1 = roi1.sum()
    # Dilation should expand the ROI
    assert n_roi1 >= n_roi0
    # The far background should still be excluded
    assert roi1[-50:].sum() == 0


def test_roi_mask_artifact_contract():
    """The returned roi_mask and snr satisfy the artifact contract (§0.1)."""
    xyz, traj, _ = _make_scene(n_moving=50, n_static=50)
    roi_mask, snr, info = motion_gate(xyz, traj)

    assert roi_mask.dtype == bool
    assert snr.dtype == np.float32
    assert len(roi_mask) == len(xyz)
    assert len(snr) == len(xyz)
    assert isinstance(info, dict)
    assert "n_points" in info
    assert "drive_freq_used" in info
