"""Kabsch EM rigid-body clustering for T20 (proposal 05,
``docs/proposals/05-iterative-kabsch-em.md``).

E-step: soft-assign each Gaussian to the body whose rigid motion best explains its full
T-frame trajectory.
M-step: weighted Kabsch per body per frame to re-fit rigid motions.

This replaces per-edge rigidity thresholding (Option B, T07/T18) with a principled
maximum-likelihood fit that pools evidence across all T frames and across all points in a
putative body.  The residual r_ik² = Σ_t ||p_i(t) − R_k(t)μ_i − τ_k(t)||² averages 3T
squared errors → std(r_ik²)/E[r_ik²] ≈ √(2/(3T)), ~7× tighter than a per-frame test for
T = 60.  This is exactly the analytic replacement for MotNet's learned affinity.

Pure numpy — no torch, no GPU, safe to import at module scope in the orchestrator.
"""

from __future__ import annotations

import numpy as np

from .trajectory_denoise import detect_drive_freq, motion_fingerprint, trajectory_energy


# ---------------------------------------------------------------------------
# Kabsch — weighted absolute orientation
# ---------------------------------------------------------------------------


def weighted_kabsch(
    pts_src: np.ndarray,
    pts_tgt: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Kabsch: find R, τ minimizing Σ w_i ||(R p_i + τ) − q_i||².

    Args:
        pts_src: (N, 3) source points μ_i (canonical).
        pts_tgt: (N, 3) target points q_i (one frame).
        weights: (N,) non-negative weights; ``None`` => uniform.

    Returns:
        R (3, 3) in SO(3), τ (3,).
    """
    if weights is None:
        weights = np.ones(len(pts_src))
    W = weights.sum()
    if W < 1e-12:
        return np.eye(3), np.zeros(3)

    w = weights / W
    c_src = (w[:, None] * pts_src).sum(axis=0)
    c_tgt = (w[:, None] * pts_tgt).sum(axis=0)

    X = pts_src - c_src
    Y = pts_tgt - c_tgt
    H = (w[:, None, None] * X[:, :, None] * Y[:, None, :]).sum(axis=0)  # (3, 3)

    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    tau = c_tgt - R @ c_src
    return R, tau


# ---------------------------------------------------------------------------
# EM — E-step / M-step
# ---------------------------------------------------------------------------


def _compute_residuals(
    traj: np.ndarray,  # (N, T, 3)
    R: np.ndarray,     # (K, T, 3, 3)
    tau: np.ndarray,   # (K, T, 3)
    xyz: np.ndarray,   # (N, 3)
) -> np.ndarray:
    """Per-point-per-body squared residuals: r_ik² = Σ_t ||p_i(t) − R_k(t)μ_i − τ_k(t)||².
    Returns (N, K).  Memory-efficient: loops over k, vectorised over N×T."""
    N, T, _ = traj.shape
    K = R.shape[0]
    residuals = np.empty((N, K), dtype=np.float64)
    for k in range(K):
        # predicted positions for body k at all frames: (N, T, 3) matching traj
        pred = np.einsum("tab,nb->nta", R[k], xyz) + tau[k][None, :, :]
        residuals[:, k] = ((traj - pred) ** 2).sum(axis=(1, 2))
    return residuals


def _e_step(
    residuals: np.ndarray,  # (N, K)
    sigma: float,
    pi: np.ndarray | None = None,
    min_resp: float = 1e-6,
) -> np.ndarray:
    """Soft responsibilities γ_ik ∝ π_k exp(−r_ik² / 2σ²).  Returns (N, K)."""
    N, K = residuals.shape
    if pi is None:
        pi = np.ones(K) / K
    # log-responsibilities for numerical stability
    log_gamma = -0.5 * residuals / max(sigma ** 2, 1e-24) + np.log(np.maximum(pi, 1e-24))
    log_gamma -= log_gamma.max(axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    # Prevent hard zeros (small regularisation)
    gamma = np.clip(gamma, min_resp, 1.0)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def _m_step(
    traj: np.ndarray,   # (N, T, 3)
    xyz: np.ndarray,    # (N, 3)
    gamma: np.ndarray,  # (N, K)
    min_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-fit each body's motion by weighted Kabsch per frame.
    Returns (R (K, T, 3, 3), tau (K, T, 3))."""
    N, T, _ = traj.shape
    K = gamma.shape[1]
    R = np.empty((K, T, 3, 3), dtype=np.float64)
    tau = np.empty((K, T, 3), dtype=np.float64)
    for k in range(K):
        w = gamma[:, k]
        W = w.sum()
        if W < min_weight * N:
            # Degenerate body — keep previous motion (identity + zero)
            R[k] = np.eye(3)
            tau[k] = 0.0
            continue
        # Vectorised over T: compute weighted centroids per frame
        w_norm = w / W
        mu_bar = (w_norm[:, None] * xyz).sum(axis=0)  # (3,)
        p_bar = np.einsum("i,itk->tk", w_norm, traj)  # (T, 3)
        # H[t] = Σ_i w_i (μ_i − μ̄)(p_i(t) − p̄(t))ᵀ  → (T, 3, 3)
        d_mu = xyz - mu_bar
        d_p = traj - p_bar[None, :, :]  # (N, T, 3)
        H = np.einsum("i,ia,itb->tab", w_norm, d_mu, d_p)
        # Batch SVD over T: H[t] = U[t] @ diag(S[t]) @ Vt[t]
        U, _, Vt = np.linalg.svd(H)
        # R[t] = Vt[t].T @ U[t].T  (V @ Uᵀ in standard Kabsch notation)
        R_batch = np.einsum("tij,tjk->tik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
        # Fix reflections
        dets = np.sign(np.linalg.det(R_batch))
        Vt_copy = Vt.copy()
        Vt_copy[:, 2, :] *= dets[:, None]
        R[k] = np.einsum(
            "tij,tjk->tik", Vt_copy.transpose(0, 2, 1), U.transpose(0, 2, 1)
        )
        tau[k] = p_bar - np.einsum("tij,j->ti", R[k], mu_bar)
    return R, tau


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ centre selection.  Returns (k, d)."""
    n, d = X.shape
    if k >= n:
        return X.copy()
    centers = np.empty((k, d), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    dists = ((X - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        probs = dists / max(dists.sum(), 1e-300)
        centers[c] = X[rng.choice(n, p=probs)]
        dists = np.minimum(dists, ((X - centers[c]) ** 2).sum(axis=1))
    return centers


def _lloyd_kmeans(X: np.ndarray, centers: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd iterations.  Returns (labels (n,), final centres (k, d))."""
    for _ in range(max_iter):
        labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1).argmin(axis=1)
        new_centers = np.array([
            X[labels == c].mean(axis=0) if (labels == c).sum() > 0 else centers[c]
            for c in range(len(centers))
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def _init_fft(
    xyz: np.ndarray,
    traj: np.ndarray,
    n_clusters: int,
    drive_freq: float | None,
    harmonics: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """FFT-fingerprint initialisation: cluster complex FFT coefficients at drive freq + harmonics.
    Points on one rigid part share the same {R(t), τ(t)} so their fingerprints are linearly
    related — amplitude/phase features cluster naturally."""
    f0 = int(drive_freq) if drive_freq is not None else detect_drive_freq(traj)
    fp = motion_fingerprint(traj, f0, harmonics)  # (N, H*3) complex
    # Real feature vector: magnitude + sin/cos of phase → (N, 3*H*3)
    mag = np.abs(fp)
    phase = np.angle(fp)
    features = np.concatenate([mag, np.cos(phase), np.sin(phase)], axis=1)
    centers = _kmeans_plus_plus(features, n_clusters, rng)
    labels, _ = _lloyd_kmeans(features, centers)
    return labels


def _init_kmeans_spatial(xyz: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Spatial k-means++ initialisation (fallback when no drive freq is known)."""
    centers = _kmeans_plus_plus(xyz, n_clusters, rng)
    labels, _ = _lloyd_kmeans(xyz, centers)
    return labels


def _init_from_labels(
    traj: np.ndarray,
    xyz: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From hard labels, build one-hot γ and run one M-step to get initial R, τ."""
    K = n_clusters
    gamma = np.zeros((len(xyz), K), dtype=np.float64)
    valid = (labels >= 0) & (labels < K)
    gamma[np.arange(len(xyz))[valid], labels[valid]] = 1.0
    # Renormalise in case some labels were out of range
    gamma /= gamma.sum(axis=1, keepdims=True)
    R, tau = _m_step(traj, xyz, gamma)
    return gamma, R, tau


# ---------------------------------------------------------------------------
# Model selection (BIC)
# ---------------------------------------------------------------------------


def _bic(residuals: np.ndarray, gamma: np.ndarray, n_clusters: int, t_frames: int) -> float:
    """BIC(K) = Σ_i,k γ_ik r_ik² + (6·T·K + K)·log(N).  Lower is better."""
    N = len(residuals)
    weighted_r = float((gamma * residuals).sum())
    n_params = 6 * t_frames * n_clusters + n_clusters
    return weighted_r + n_params * np.log(max(N, 1))


# ---------------------------------------------------------------------------
# Single K EM run
# ---------------------------------------------------------------------------


def _em_single(
    xyz: np.ndarray,
    traj: np.ndarray,
    n_clusters: int,
    sigma: float,
    *,
    init: str = "fft",
    drive_freq: float | None = None,
    harmonics: int = 3,
    max_iter: int = 30,
    tol: float = 1e-4,
    min_weight: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Run EM for a fixed number of clusters.

    Returns (gamma (N,K), R (K,T,3,3), tau (K,T,3), residuals (N,K), n_iter, info).
    """
    rng = rng or np.random.default_rng()
    N, T, _ = traj.shape

    # --- initialise --------------------------------------------------------
    if init == "fft":
        labels = _init_fft(xyz, traj, n_clusters, drive_freq, harmonics, rng)
    else:
        labels = _init_kmeans_spatial(xyz, n_clusters, rng)

    gamma, R, tau = _init_from_labels(traj, xyz, labels, n_clusters)

    # Annealing: start with a softer temperature (larger sigma) and tighten
    sigma_current = max(sigma, 1.0)

    # --- EM iterations -----------------------------------------------------
    for it in range(max_iter):
        residuals = _compute_residuals(traj, R, tau, xyz)
        gamma_new = _e_step(residuals, sigma_current)
        R_new, tau_new = _m_step(traj, xyz, gamma_new, min_weight=min_weight)

        # Adaptive sigma: ML estimate from weighted residuals
        weighted_r = float((gamma_new * residuals).sum())
        sigma_est = np.sqrt(weighted_r / max(3 * T * N, 1))
        # Anneal toward the data-driven estimate
        sigma_current = 0.7 * sigma_current + 0.3 * max(sigma_est, sigma * 0.5)

        # Convergence: maximum responsibility shift
        max_shift = float(np.abs(gamma_new - gamma).max())
        gamma, R, tau = gamma_new, R_new, tau_new
        if max_shift < tol:
            break

    residuals = _compute_residuals(traj, R, tau, xyz)
    return gamma, R, tau, residuals, it + 1, {"max_shift": max_shift, "converged": max_shift < tol}


# ---------------------------------------------------------------------------
# Greedy split
# ---------------------------------------------------------------------------


def _greedy_split(
    xyz: np.ndarray,
    traj: np.ndarray,
    gamma: np.ndarray,
    R: np.ndarray,
    tau: np.ndarray,
    sigma: float,
    residuals: np.ndarray,
    *,
    max_iter: int = 20,
    split_tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """After EM convergence, split the body with the largest within-residual if the split
    improves BIC.  Each split runs a short EM on the two new bodies.

    Returns updated (gamma, R, tau, labels, residuals, info).
    """
    rng = rng or np.random.default_rng()
    N, T, _ = traj.shape
    K = gamma.shape[1]
    labels = gamma.argmax(axis=1)

    # Per-body mean within-residual
    body_r = np.array([
        residuals[labels == k, k].mean() if (labels == k).sum() > 0 else 0.0
        for k in range(K)
    ])
    split_k = int(np.argmax(body_r))
    if body_r[split_k] <= 0:
        return gamma, R, tau, labels, residuals, {"n_splits": 0}

    mask = labels == split_k
    n_split = int(mask.sum())
    if n_split < 30:
        return gamma, R, tau, labels, residuals, {"n_splits": 0}

    # Try k-means on the spatial coordinates of the split body (2-way split)
    sub_xyz = xyz[mask]
    sub_traj = traj[mask]
    sub_labels = _init_kmeans_spatial(sub_xyz, 2, rng)
    sub_gamma, sub_R, sub_tau = _init_from_labels(sub_traj, sub_xyz, sub_labels, 2)

    # Short EM on the split body only
    for _ in range(max_iter):
        sub_res = _compute_residuals(sub_traj, sub_R, sub_tau, sub_xyz)
        sub_gamma_new = _e_step(sub_res, sigma)
        sub_R_new, sub_tau_new = _m_step(sub_traj, sub_xyz, sub_gamma_new)
        if np.abs(sub_gamma_new - sub_gamma).max() < split_tol:
            break
        sub_gamma, sub_R, sub_tau = sub_gamma_new, sub_R_new, sub_tau_new

    # Rebuild full gamma/R/tau with the new split body
    new_K = K + 1
    new_gamma = np.zeros((N, new_K), dtype=np.float64)
    new_R = np.empty((new_K, T, 3, 3), dtype=np.float64)
    new_tau = np.empty((new_K, T, 3), dtype=np.float64)

    # Copy old bodies, shifting indices after split_k
    for k in range(K):
        dst = k if k < split_k else k + 1
        new_gamma[:, dst] = gamma[:, k]
        new_R[dst] = R[k]
        new_tau[dst] = tau[k]

    # Fill in the two new split bodies
    split_idx = np.where(mask)[0]
    new_gamma[split_idx, split_k] = sub_gamma[:, 0]
    new_gamma[split_idx, split_k + 1] = sub_gamma[:, 1]
    new_R[split_k] = sub_R[0]
    new_R[split_k + 1] = sub_R[1]
    new_tau[split_k] = sub_tau[0]
    new_tau[split_k + 1] = sub_tau[1]

    # Renormalise
    new_gamma /= new_gamma.sum(axis=1, keepdims=True)

    new_residuals = _compute_residuals(traj, new_R, new_tau, xyz)
    new_labels = new_gamma.argmax(axis=1)

    # Check BIC improvement
    old_bic = _bic(residuals, gamma, K, T)
    new_bic = _bic(new_residuals, new_gamma, new_K, T)

    info = {"n_splits": 1, "split_k": int(split_k), "old_bic": old_bic, "new_bic": new_bic}
    if new_bic < old_bic:
        return new_gamma, new_R, new_tau, new_labels, new_residuals, info
    else:
        return gamma, R, tau, labels, residuals, {**info, "accepted": False}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def segment_by_kabsch(
    xyz: np.ndarray,
    traj: np.ndarray,
    *,
    n_clusters: int = 0,
    k_range: list[int] | None = None,
    init: str = "fft",
    max_iter: int = 30,
    sigma: float | None = None,
    spatial_prior: bool = False,
    greedy_split: bool = False,
    fps_subsample: int = 0,
    propagate_q: int = 3,
    drive_freq: float | None = None,
    harmonics: int = 3,
    tolerance: float = 1e-4,
    min_size: int = 15,
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Kabsch EM segmentation with BIC model selection.

    Args:
        xyz: (N, 3) canonical positions.
        traj: (N, T, 3) deformed trajectories.
        n_clusters: fixed K (overrides BIC search if > 0).
        k_range: candidate K values for BIC search; default [2, 200].
        init: ``"fft"`` (fingerprint + k-means++), ``"kmeans"`` (spatial k-means++).
        max_iter: EM iterations per K.
        sigma: per-coordinate noise std; ``None`` => auto from trajectory energy.
        spatial_prior: (not yet implemented) Potts smoothness on k-NN graph.
        greedy_split: try one BIC-improving split after EM convergence.
        fps_subsample: > 0 => FPS subsample to this many points, run EM, propagate back.
        propagate_q: q-NN majority vote for propagation.
        drive_freq: cycles/clip for FFT init; ``None`` => auto-detect.
        harmonics: number of FFT harmonics for fingerprint init.
        tolerance: EM convergence threshold (max responsibility shift).
        min_size: tiny clusters are merged into nearest neighbour.
        rng_seed: random seed.

    Returns:
        (labels (N,) int64, info dict).
    """
    rng = np.random.default_rng(rng_seed)
    N, T, _ = traj.shape
    info: dict = {"n_points": N, "t_frames": T, "init": init}

    # --- auto-estimate sigma ------------------------------------------------
    if sigma is None:
        # σ² ≈ median trajectory energy / (3T)  →  per-coordinate, per-frame variance
        energy = trajectory_energy(traj)
        sigma = float(np.sqrt(np.median(energy) / max(3 * T, 1)))
    info["sigma"] = sigma

    # --- FPS subsample ------------------------------------------------------
    if fps_subsample and fps_subsample < N:
        from .rigidity_graph2 import fps_subsample as _fps_subsample, propagate_labels

        sel = _fps_subsample(xyz, fps_subsample, rng_seed=rng_seed)
        sub_labels, sub_info = segment_by_kabsch(
            xyz[sel], traj[sel],
            n_clusters=n_clusters, k_range=k_range, init=init, max_iter=max_iter,
            sigma=sigma, spatial_prior=spatial_prior, greedy_split=greedy_split,
            fps_subsample=0, propagate_q=propagate_q,
            drive_freq=drive_freq, harmonics=harmonics, tolerance=tolerance,
            min_size=min_size, rng_seed=rng_seed,
        )
        labels = propagate_labels(xyz[sel], sub_labels, xyz, q=propagate_q)
        labels = np.unique(labels, return_inverse=True)[1]
        info.update(sub_info, n_points=N, n_subsample=len(sel), propagated=True)
        return labels, info

    # --- BIC search over k_range --------------------------------------------
    if n_clusters > 0:
        k_candidates = [n_clusters]
    else:
        k_range = k_range or [2, 200]
        lo, hi = k_range
        # Coarse search: geometric progression, then fine around the minimum
        k_candidates = []
        k = lo
        while k <= hi:
            k_candidates.append(k)
            k = max(k + 1, int(k * 1.5))
        info["k_candidates"] = k_candidates

    best_bic = np.inf
    best_result = None
    bics: dict[int, float] = {}

    for k in k_candidates:
        gamma, R, tau, residuals, it, run_info = _em_single(
            xyz, traj, k, sigma,
            init=init, drive_freq=drive_freq, harmonics=harmonics,
            max_iter=max_iter, tol=tolerance, rng=rng,
        )

        if greedy_split:
            gamma, R, tau, labels_k, residuals, split_info = _greedy_split(
                xyz, traj, gamma, R, tau, sigma, residuals, rng=rng,
            )
            run_info.update(split_info)
            k = gamma.shape[1]  # may have increased by 1

        bic = _bic(residuals, gamma, k, T)
        bics[k] = bic
        run_info.update(bic=bic, n_iter=it, n_clusters=k)

        if bic < best_bic:
            best_bic = bic
            best_result = (gamma, R, tau, residuals, run_info)

    assert best_result is not None
    gamma, R, tau, residuals, run_info = best_result
    labels = gamma.argmax(axis=1)

    # --- merge tiny clusters ------------------------------------------------
    from .rigidity_graph import merge_small_components

    labels = merge_small_components(xyz, labels, min_size)
    labels = np.unique(labels, return_inverse=True)[1]

    info.update(
        n_clusters_final=int(labels.max() + 1) if N else 0,
        bics={int(k): float(v) for k, v in bics.items()},
        best_k=int(gamma.shape[1]),
    )
    info.update(run_info)

    # Per-body residual floor (diagnostic: bodies with high floor are non-rigid)
    body_residuals = []
    for k in range(gamma.shape[1]):
        mask = labels == k
        if mask.sum() > 0:
            body_residuals.append(float(residuals[mask, k].mean()))
    if body_residuals:
        info["body_residual_mean"] = float(np.mean(body_residuals))
        info["body_residual_max"] = float(np.max(body_residuals))

    return labels, info
