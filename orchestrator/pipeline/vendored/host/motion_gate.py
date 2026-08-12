"""ROI motion gating for T19 (``roi.motion_gate``, proposal 01 in
``docs/proposals/01-motion-gated-roi.md``).

Extracts a machine-ROI mask from 4DGS trajectories by:
1. Band-passing trajectories at the drive frequency + harmonics (the true motion is
   narrowband; reconstruction jitter is approximately white).
2. Computing per-point energy on the denoised trajectories.
3. Log-Otsu threshold on energy → initial static/moving split.
4. k-NN graph dilation of the moving region (captures boundary points whose own motion
   is too small to pass the energy gate).
5. Rigidity-lock readmission: static points that are rigidly connected (low edge score)
   to the dilated ROI are readmitted (handles near-stationary points on moving bodies).

Pure numpy/scipy — no torch, no GPU, safe to import at module scope in the orchestrator.
"""

from __future__ import annotations

import numpy as np

from .rigidity_graph import build_knn_edges, edge_rigidity_score, otsu_threshold_log
from .rigidity_graph2 import calibrate_sigma_d
from .trajectory_denoise import bandpass, trajectory_energy


def motion_gate(
    xyz: np.ndarray,
    traj: np.ndarray,
    *,
    drive_freq: float | None = None,
    harmonics: int = 3,
    dilation_hops: int = 1,
    readmit_mult: float = 3.0,
    k: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Band-limited energy gate + k-NN dilation + rigidity-lock readmission.

    Args:
        xyz: (N, 3) canonical positions.
        traj: (N, T, 3) positions over time.
        drive_freq: cycles per clip window (rfft bin units); None => auto-detect.
        harmonics: number of drive-frequency harmonics to keep in the band-pass.
        dilation_hops: number of k-NN hops to dilate the initial moving region.
        readmit_mult: rigidity-lock readmission threshold, in units of the calibrated
            noise-floor sigma (sigma_d from static-static edges).
        k: k for the k-NN graph used in dilation + readmission.

    Returns:
        roi_mask: bool[N] — True = inside machine ROI (moving or readmitted).
        snr: float32[N] — per-point SNR = energy / Otsu_threshold.
        info: dict with diagnostics (n_points, n_moving_init, n_dilated, n_readmitted,
              drive_freq_used, otsu_threshold, sigma_d).
    """
    n = len(xyz)

    # 1. Band-pass at drive frequency + harmonics
    traj_denoised, f0 = bandpass(traj, drive_freq=drive_freq, harmonics=harmonics)

    # 2. Energy on denoised trajectories
    energy = trajectory_energy(traj_denoised)

    # 3. Log-Otsu threshold → initial static/moving split
    thr = otsu_threshold_log(energy)

    moving = energy > thr
    n_moving = int(moving.sum())

    # Guard degenerate cases: Otsu threshold >= max energy, or no points pass the gate.
    # An empty ROI would break downstream segmentation, so fall back to all True.
    if thr <= 0 or thr >= energy.max() or n_moving == 0:
        roi_mask = np.ones(n, dtype=bool)
        snr = np.ones(n, dtype=np.float32)
        return roi_mask, snr, {
            "n_points": n,
            "n_moving_init": n_moving,
            "n_dilated": n_moving,
            "n_readmitted": 0,
            "drive_freq_used": int(f0),
            "otsu_threshold": float(thr),
            "sigma_d": None,
            "degenerate": True,
        }

    # 4. Build k-NN graph
    edges = build_knn_edges(xyz, k=k)

    # 5. Dilation: expand moving region by dilation_hops on the k-NN graph
    roi = moving.copy()
    for _ in range(dilation_hops):
        # Find all edges where at least one endpoint is currently in ROI
        border = roi[edges[:, 0]] | roi[edges[:, 1]]
        # Add both endpoints of those edges
        roi[edges[border].reshape(-1)] = True

    n_dilated = int(roi.sum())

    # 6. Rigidity-lock readmission
    # Calibrate noise floor from the ORIGINAL static points (not yet readmitted)
    static = ~moving
    sigma_d = calibrate_sigma_d(traj, edges, static=static)
    readmit_thr = readmit_mult * sigma_d

    n_readmitted = 0
    if readmit_thr > 0 and n_dilated < n:
        # Candidate edges: one endpoint in ROI, the other static (not in ROI)
        in_roi_not_other = roi[edges[:, 0]] & ~roi[edges[:, 1]]
        other_not_roi = ~roi[edges[:, 0]] & roi[edges[:, 1]]
        cand_mask = in_roi_not_other | other_not_roi
        cand_edges = edges[cand_mask]

        if len(cand_edges) > 0:
            scores = edge_rigidity_score(traj, cand_edges)
            # The static endpoint of each candidate edge
            static_end = np.where(~roi[cand_edges[:, 0]], cand_edges[:, 0], cand_edges[:, 1])
            # A static point is readmitted if ANY of its edges to ROI is below threshold
            readmit = np.zeros(n, dtype=bool)
            readmit[static_end[scores < readmit_thr]] = True
            n_readmitted = int(readmit.sum())
            roi = roi | readmit

    snr = (energy / thr).astype(np.float32)

    info = {
        "n_points": n,
        "n_moving_init": n_moving,
        "n_dilated": n_dilated,
        "n_readmitted": n_readmitted,
        "drive_freq_used": int(f0),
        "otsu_threshold": float(thr),
        "sigma_d": float(sigma_d),
        "readmit_threshold": float(readmit_thr),
        "degenerate": False,
    }
    return roi, snr, info
