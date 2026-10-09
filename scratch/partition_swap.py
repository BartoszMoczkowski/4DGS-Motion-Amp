#!/usr/bin/env python3
"""partition_swap.py — Implementation of global rigid-body prior partitions inside dynamic ROIs."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

from pipeline.vendored.host.trajectory_denoise import (
    detect_drive_freq,
    motion_fingerprint,
    trajectory_energy,
)
from pipeline.vendored.host.kabsch_em import (
    _kmeans_plus_plus,
    _lloyd_kmeans,
    _compute_residuals,
    _m_step,
    _e_step,
    _bic,
)
from pipeline.vendored.host.metrics import adjusted_rand_index as compute_ari, best_iou_matching


def fit_fft_kmeans_bic(
    traj: np.ndarray,
    xyz: np.ndarray,
    k_min: int = 2,
    k_max: int = 35,
    drive_freq: float | None = None,
    harmonics: int = 3,
    rng_seed: int = 42,
) -> tuple[np.ndarray, int, float]:
    """Fit K-Means on FFT motion fingerprints with BIC model selection.

    Args:
        traj: (N_roi, T, 3) trajectories of ROI points.
        xyz: (N_roi, 3) canonical coordinates of ROI points.
        k_min: minimum number of clusters.
        k_max: maximum number of clusters.
        drive_freq: fundamental frequency.
        harmonics: number of harmonics.
        rng_seed: random seed.

    Returns:
        (best_labels (N_roi,), best_k, best_bic)
    """
    N, T, _ = traj.shape
    if N == 0:
        return np.zeros(0, dtype=np.int64), 0, 0.0
    rng = np.random.default_rng(rng_seed)
    f0 = int(drive_freq) if drive_freq is not None else detect_drive_freq(traj)
    fp = motion_fingerprint(traj, f0, harmonics)  # (N, H*3) complex
    mag = np.abs(fp)
    phase = np.angle(fp)
    features = np.concatenate([mag, np.cos(phase), np.sin(phase)], axis=1)

    k_max_clamped = min(k_max, max(k_min, N // 10))
    best_bic = float("inf")
    best_labels = np.zeros(N, dtype=np.int64)
    best_k = k_min

    for k in range(k_min, k_max_clamped + 1):
        try:
            centers = _kmeans_plus_plus(features, k, rng)
            labels, _ = _lloyd_kmeans(features, centers)
            
            # One-hot gamma & M-step to evaluate rigid fit residuals
            gamma = np.zeros((N, k), dtype=np.float64)
            for c in range(k):
                gamma[labels == c, c] = 1.0
            # Guard against empty clusters
            gamma += 1e-6
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            R, tau = _m_step(traj, xyz, gamma)
            residuals = _compute_residuals(traj, R, tau, xyz)
            score = _bic(residuals, gamma, k, T)
            if score < best_bic:
                best_bic = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    return best_labels, best_k, best_bic


def fit_spatial_connected_components(
    xyz: np.ndarray,
    radius: float = 0.25,
    min_cluster_size: int = 15,
) -> tuple[np.ndarray, int]:
    """Cluster points within ROI into spatial connected components.

    Args:
        xyz: (N_roi, 3) canonical coordinates.
        radius: neighbor search radius.
        min_cluster_size: minimum points per component.

    Returns:
        (labels (N_roi,), n_components)
    """
    N = len(xyz)
    if N == 0:
        return np.zeros(0, dtype=np.int64), 0

    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) > 0:
        data = np.ones(len(pairs), dtype=bool)
        adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(N, N))
        adj = adj + adj.T
    else:
        adj = csr_matrix((N, N), dtype=bool)

    n_comp, raw_labels = csgraph.connected_components(
        adj, directed=False, return_labels=True
    )

    # Filter small clusters into nearest large component
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_components = unique[counts >= min_cluster_size]

    if len(valid_components) == 0:
        return np.zeros(N, dtype=np.int64), 1

    comp_map = {comp: idx for idx, comp in enumerate(valid_components)}
    labels = np.full(N, -1, dtype=np.int64)
    for comp, new_idx in comp_map.items():
        labels[raw_labels == comp] = new_idx

    # Assign remaining noise points to nearest valid point
    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        assigned = np.where(labels >= 0)[0]
        sub_tree = cKDTree(xyz[assigned])
        _, nearest_idx = sub_tree.query(xyz[unassigned], k=1)
        labels[unassigned] = labels[assigned[nearest_idx]]

    return labels, len(valid_components)
