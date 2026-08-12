"""Upgraded rigidity-graph segmentation core for T18 (``segment.rigid2``, proposal 06 in
``docs/proposals/06-multiscale-snr-multiscale.md``).

Extends the Option-B baseline (:mod:`pipeline.vendored.host.rigidity_graph`, kept untouched for
regression) with the three noise-regime fixes measured necessary on the real grid models
(``runs/grid_seg_results.csv``: baseline ARI ~ 0.002-0.009 vs 107 GT parts):

1. **Denoise before scoring** — band-pass trajectories at the drive frequency + harmonics
   (:mod:`pipeline.vendored.host.trajectory_denoise`); the reconstruction jitter is ~white,
   the true motion is narrowband.
2. **Calibrated z-scores instead of a magic threshold** — the per-edge rigidity std is divided
   by a noise floor ``sigma_d`` estimated *per scene from static points*, so the cut becomes a
   significance level that adapts to each model's reconstruction quality (the sweep models,
   where noise dominates, are exactly where the baseline goes negative-ARI).
3. **Graph statistics instead of edge cutting** — a spectral partition over the
   ``exp(-z^2/2)``-weighted k-NN graph with eigengap-chosen K; a part boundary then needs many
   independent high-z edges, so single-edge noise flips no longer split/merge parts. The old
   "cut + connected components" path is kept as ``partition="components"``.

Plus the separability diagnostic (:func:`separability_auroc`): given GT labels, the AUROC of
the z-score as a same-part/different-part edge classifier — the go/no-go signal for per-edge
methods per model (``docs/proposals/IMPLEMENTATION_PLAN.md`` §3).

Pure numpy/scipy — no torch, no GPU, safe to import at module scope in the orchestrator.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from .rigidity_graph import (
    build_knn_edges,
    edge_rigidity_score,
    merge_small_components,
    otsu_threshold_log,
)
from .trajectory_denoise import bandpass, trajectory_energy


# --- noise calibration ------------------------------------------------------------------------


def static_mask(traj: np.ndarray) -> np.ndarray:
    """Per-point static/moving split via log-Otsu on raw trajectory energy. Static points are
    the calibrated noise reference — their trajectories contain *only* reconstruction jitter."""
    energy = trajectory_energy(traj)
    thr = otsu_threshold_log(energy)
    return energy <= thr


def calibrate_sigma_d(traj: np.ndarray, edges: np.ndarray, static: np.ndarray | None = None) -> float:
    """Per-scene noise floor for the edge rigidity score: the median std of the pairwise
    distance series over edges whose both endpoints are static (proposal 06 §2). Falls back to
    the median over *all* edges when the scene has no usable static-static edges."""
    if static is None:
        static = static_mask(traj)
    both_static = static[edges[:, 0]] & static[edges[:, 1]]
    ref_edges = edges[both_static] if both_static.any() else edges
    scores = edge_rigidity_score(traj, ref_edges)
    sigma = float(np.median(scores))
    # Guard against exactly-noiseless data (synthetic self-tests): a zero floor would make
    # every z-score infinite. 1e-12 is far below any real jitter (~1e-3 on pump01).
    return max(sigma, 1e-12)


def edge_zscores(traj_denoised: np.ndarray, edges: np.ndarray, sigma_d: float) -> np.ndarray:
    """Rigidity significance per edge: std of the denoised pairwise-distance series, in units
    of the calibrated noise floor. z ~ O(1) for a truly rigid edge at the noise floor; large
    for cross-part edges. Capped to keep exp(-z^2/2) arithmetic finite on noiseless data."""
    scores = edge_rigidity_score(traj_denoised, edges)
    return np.clip(scores / sigma_d, 0.0, 1e6)


# --- partitions -------------------------------------------------------------------------------


def _kmeans(x: np.ndarray, k: int, rng: np.random.RandomState, n_init: int = 3,
            max_iter: int = 25) -> np.ndarray:
    """Small dependency-free k-means++ (best of n_init restarts by inertia). Deliberately not
    sklearn — the orchestrator's vendored host code must stay numpy/scipy-only (T18 plan §2.3)."""
    n = len(x)
    if k >= n:
        return np.arange(n)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = np.empty((k, x.shape[1]), dtype=x.dtype)
        centers[0] = x[rng.randint(n)]
        d2 = ((x - centers[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            probs = d2 / max(d2.sum(), 1e-300)
            centers[c] = x[rng.choice(n, p=probs)]
            d2 = np.minimum(d2, ((x - centers[c]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=np.int64)
        for _it in range(max_iter):
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members):
                    centers[c] = members.mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def adaptive_z_cut(z: np.ndarray, z_thresh: float, mad_mult: float = 6.0) -> float:
    """Robust adaptive significance cut: ``min(z_thresh, median + mad_mult * MAD(z))``.

    Why both: the calibrated ``z_thresh`` (a significance level, default 3) is the right gate
    when the same-part z distribution is broad (real noisy models); but on *clean* scenes the
    same-part z's sit far below it (e.g. median 0.3) and genuine boundary edges at z ~ 1 would
    pass — the median+MAD rule catches those, since MAD scales with the scene's own same-part
    spread. The min keeps whichever gate is stricter for the scene at hand. (Found necessary by
    the T18 sandbox fixture: with z_thresh=3 alone, no edge was ever cut on a near-clean scene
    and the un-gated soft affinity left the whole graph connected, fragmenting the partition.)
    """
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    return float(min(z_thresh, med + mad_mult * mad))


def _spectral_split(W_sub, *, min_clusters: int, max_clusters: int, n_clusters: int,
                    rng_seed: int) -> tuple[np.ndarray, int, list[float]]:
    """Spectral sub-clustering of one *connected* component's affinity matrix. On a connected
    graph the top eigenvalue of the normalized adjacency is simple, so ``eigsh(which="LA")``
    is numerically reliable here (unlike the global problem, where degenerate eigenvalue
    multiplicities from many near-disconnected parts make Lanczos miss them — confirmed on the
    T18 fixture against a dense solve: 7x1.0 dense, only 2 found sparse). Returns
    (labels (m,), k_selected, normalized-Laplacian eigenvalues for diagnostics)."""
    m = W_sub.shape[0]
    deg = np.asarray(W_sub.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    S = W_sub.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :]).tocsr()

    k_cand = int(min(max_clusters, max(2, m - 2)))
    n_eig = min(k_cand + 1, m - 1)
    if n_eig < 2:
        return np.zeros(m, dtype=np.int64), 1, []
    evals_s, evecs = eigsh(S.astype(np.float64), k=n_eig, which="LA", tol=1e-6, maxiter=8000)
    order = np.argsort(evals_s)[::-1]  # lambda_S near 1 = near-disconnected sub-structure
    evals_s, evecs = evals_s[order], evecs[:, order]
    evals = np.clip(1.0 - evals_s, 0.0, None)  # normalized-Laplacian eigenvalues, ascending

    if n_clusters > 0:
        k_sel = max(1, int(min(n_clusters, n_eig)))
    else:
        # gaps[k-1] = lambda_{k+1} - lambda_k; k_sel in [1, k_cand] (1 = "do not split").
        gaps = np.diff(evals[: k_cand + 1])
        lo = max(min_clusters, 1)
        k_sel = lo + int(np.argmax(gaps[lo - 1:]))
    if k_sel <= 1:
        return np.zeros(m, dtype=np.int64), 1, [float(v) for v in evals[: k_cand + 1]]
    U = evecs[:, :k_sel]
    row_norm = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.where(row_norm > 0, row_norm, 1.0)
    labels = _kmeans(U, k_sel, np.random.RandomState(rng_seed))
    return labels, k_sel, [float(v) for v in evals[: k_cand + 1]]


def spectral_partition(n: int, edges: np.ndarray, z: np.ndarray, *, min_clusters: int = 2,
                       max_clusters: int = 50, n_clusters: int = 0, z_cut: float | None = None,
                       min_split_size: int = 30, split_z_ratio: float = 1.5,
                       rng_seed: int = 0) -> tuple[np.ndarray, dict]:
    """Hybrid graph partition (proposal 06 §3, hardened by the T18 fixture):

    1. Affinity W_ij = exp(-z_ij^2/2); edges with z above ``z_cut`` (see
       :func:`adaptive_z_cut`) are hard-gated to W = 0. Without the gate, a scene whose z's
       are all small in absolute terms keeps a fully-connected graph and spectral clustering
       fragments uniform regions (e.g. a static background cloud) instead of respecting the
       true bottlenecks.
    2. Connected components of the gated graph are taken as coarse clusters for free.
    3. Each component of >= ``min_split_size`` points is sub-split by spectral clustering
       (:func:`_spectral_split`) with eigengap model selection, so parts the gate couldn't
       fully separate (weak but extended boundaries — the expected real-data case) are still
       split on aggregated multi-path evidence, not on any single edge.

    Returns (labels (n,), info with per-component eigenvalue spectra for diagnostics).
    """
    w = np.exp(-0.5 * z ** 2)
    if z_cut is not None:
        w = np.where(z <= z_cut, w, 0.0)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([w, w])
    W = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W.eliminate_zeros()  # gated edges must be gone structurally, not just stored as 0.0

    n_comp, comp = connected_components(W, directed=False)
    labels = np.zeros(n, dtype=np.int64)
    offset = 0
    spectra: dict[int, list[float]] = {}
    # Per-edge component membership, for the sub-split acceptance guard below.
    edge_comp = comp[edges[:, 0]]
    edge_internal = comp[edges[:, 0]] == comp[edges[:, 1]]
    for c in range(n_comp):
        idx = np.where(comp == c)[0]
        if len(idx) < max(min_split_size, min_clusters + 2):
            labels[idx] = offset
            offset += 1
            continue
        sub_labels, k_sel, evals = _spectral_split(
            W[idx][:, idx], min_clusters=min_clusters, max_clusters=max_clusters,
            n_clusters=n_clusters, rng_seed=rng_seed,
        )
        if k_sel > 1:
            # Rigidity-relative acceptance guard: a sub-split is only kept if the edges it
            # cuts are *clearly less rigid* than the component's typical edge. Without this,
            # the spectral step happily "partitions" uniform one-part clouds (a static base
            # plate has plenty of weak geometric bottlenecks with perfectly rigid z's —
            # splitting there is what tanked ARI to ~0.17 on the T18 fixture). A true part
            # boundary the hard gate missed has elevated z by construction.
            in_c = edge_internal & (edge_comp == c)
            zc = z[in_c]
            ec = edges[in_c]
            pos = np.empty(n, dtype=np.int64)
            pos[idx] = np.arange(len(idx))
            cut = sub_labels[pos[ec[:, 0]]] != sub_labels[pos[ec[:, 1]]]
            if not cut.any() or zc[cut].mean() < split_z_ratio * zc.mean():
                sub_labels = np.zeros(len(idx), dtype=np.int64)
                k_sel = 1
        labels[idx] = offset + sub_labels
        offset += k_sel
        if evals:
            spectra[int(c)] = evals
    labels = np.unique(labels, return_inverse=True)[1]
    info = {
        "n_gated_components": int(n_comp),
        "eigengap_k": int(labels.max() + 1) if n else 0,
        "eigenvalues": spectra,
    }
    return labels, info


def components_partition(n: int, edges: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, dict]:
    """The baseline path: cut non-kept edges, connected components (ported from
    :func:`rigidity_graph.segment_by_rigidity`'s graph construction, unchanged)."""
    rows = np.concatenate([edges[keep, 0], edges[keep, 1]])
    cols = np.concatenate([edges[keep, 1], edges[keep, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    return labels, {"n_components_raw": int(n_components)}


# --- subsample + propagate (MBS's scalability trick, proposal 06 §4) ---------------------------


def fps_subsample(xyz: np.ndarray, n_sub: int, rng_seed: int = 0) -> np.ndarray:
    """Farthest-point sampling; returns indices (n_sub,). Same idea as the FPS working-set in
    the vendored ``mbs_infer.py`` (reference for the convention; independent implementation —
    that one is torch/GPU, this is CPU numpy)."""
    n = len(xyz)
    n_sub = min(n_sub, n)
    rng = np.random.RandomState(rng_seed)
    sel = np.empty(n_sub, dtype=np.int64)
    sel[0] = rng.randint(n)
    d2 = ((xyz - xyz[sel[0]]) ** 2).sum(axis=1)
    for c in range(1, n_sub):
        sel[c] = int(np.argmax(d2))
        d2 = np.minimum(d2, ((xyz - xyz[sel[c]]) ** 2).sum(axis=1))
    return sel


def propagate_labels(xyz_sub: np.ndarray, labels_sub: np.ndarray, xyz_full: np.ndarray,
                     q: int = 3) -> np.ndarray:
    """q-NN majority-vote label propagation from the subsample to the full point set, in
    canonical space (the CPU analogue of ``mbs_infer.py``'s 3-NN propagation)."""
    q = max(1, min(q, len(xyz_sub)))
    _, idx = cKDTree(xyz_sub).query(xyz_full, k=q)
    if idx.ndim == 1:
        idx = idx[:, None]
    votes = labels_sub[idx]  # (N, q)
    out = np.empty(len(xyz_full), dtype=np.int64)
    for i in range(len(xyz_full)):
        vals, counts = np.unique(votes[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


# --- separability diagnostic -------------------------------------------------------------------


def separability_auroc(z: np.ndarray, edges: np.ndarray, gt_labels: np.ndarray) -> dict:
    """AUROC of the per-edge z-score as a *different-part* classifier over k-NN edges, vs GT.
    AUROC < 0.8 on a model means per-edge methods cap out there (IMPLEMENTATION_PLAN §3).
    Computed by the Mann-Whitney rank statistic — O(E log E), no sklearn."""
    same = gt_labels[edges[:, 0]] == gt_labels[edges[:, 1]]
    pos, neg = z[~same], z[same]  # positive class = different-part
    if len(pos) == 0 or len(neg) == 0:
        return {"auroc": None, "n_same_edges": int(same.sum()), "n_diff_edges": int((~same).sum())}
    vals = np.concatenate([pos, neg])
    ranks = _tie_averaged_ranks(vals)
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return {
        "auroc": float(auc),
        "n_same_edges": int(same.sum()),
        "n_diff_edges": int((~same).sum()),
        "z_median_same": float(np.median(neg)),
        "z_median_diff": float(np.median(pos)),
    }


def _tie_averaged_ranks(vals: np.ndarray) -> np.ndarray:
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=np.float64)
    sorted_vals = vals[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


# --- full pipeline ------------------------------------------------------------------------------


def segment_by_rigidity2(
    xyz: np.ndarray,
    traj: np.ndarray,
    *,
    k: int = 12,
    min_size: int = 15,
    denoise: bool = True,
    drive_freq: float | None = None,
    harmonics: int = 3,
    calibrate_sigma: bool = True,
    z_thresh: float = 3.0,
    threshold_mult: float = 1.0,
    partition: str = "components",
    min_clusters: int = 2,
    max_clusters: int = 50,
    n_clusters: int = 0,
    n_subsample: int = 0,
    propagate_q: int = 3,
    gt_labels: np.ndarray | None = None,
    rng_seed: int = 0,
):
    """Full T18 pipeline. xyz: (N,3) canonical; traj: (N,T,3). Returns (labels (N,) int, info).

    With ``n_subsample > 0`` the core runs on an FPS subsample and labels are propagated back
    by q-NN majority vote (proposal 06 §4). ``gt_labels`` (aligned with xyz) only feeds the
    separability diagnostic in ``info["separability"]`` — never the segmentation itself.
    """
    n = len(xyz)
    info: dict = dict(n_points=n)

    if n_subsample and n_subsample < n:
        sel = fps_subsample(xyz, n_subsample, rng_seed=rng_seed)
        sub_labels, sub_info = segment_by_rigidity2(
            xyz[sel], traj[sel], k=k, min_size=min_size, denoise=denoise,
            drive_freq=drive_freq, harmonics=harmonics, calibrate_sigma=calibrate_sigma,
            z_thresh=z_thresh, threshold_mult=threshold_mult, partition=partition,
            min_clusters=min_clusters, max_clusters=max_clusters, n_clusters=n_clusters,
            n_subsample=0, propagate_q=propagate_q,
            gt_labels=gt_labels[sel] if gt_labels is not None else None, rng_seed=rng_seed,
        )
        labels = propagate_labels(xyz[sel], sub_labels, xyz, q=propagate_q)
        labels = np.unique(labels, return_inverse=True)[1]
        sub_info.update(n_points=n, n_subsample=int(len(sel)), propagated=True)
        return labels, sub_info

    edges = build_knn_edges(xyz, k=k)
    traj_used, f0 = bandpass(traj, drive_freq=drive_freq, harmonics=harmonics) if denoise else (traj, 0)
    info.update(n_edges=len(edges), drive_freq_used=int(f0), denoise=bool(denoise))

    sigma_d = calibrate_sigma_d(traj, edges) if calibrate_sigma else None
    z = edge_zscores(traj_used, edges, sigma_d if calibrate_sigma else 1.0)
    info["sigma_d"] = sigma_d

    if partition == "spectral":
        z_cut = adaptive_z_cut(z, z_thresh) if calibrate_sigma else None
        labels, pinfo = spectral_partition(
            n, edges, z, min_clusters=min_clusters, max_clusters=max_clusters,
            n_clusters=n_clusters, z_cut=z_cut, rng_seed=rng_seed,
        )
        info.update(pinfo, z_cut=z_cut)
    else:  # "components" — the baseline shape, with the calibrated z or legacy Otsu threshold
        if calibrate_sigma:
            thr = adaptive_z_cut(z, z_thresh)
            keep = z <= thr
        else:
            scores = edge_rigidity_score(traj_used, edges)
            thr = otsu_threshold_log(scores) * threshold_mult
            keep = scores <= thr
        labels, pinfo = components_partition(n, edges, keep)
        info.update(pinfo, threshold=thr, n_kept_edges=int(keep.sum()))

    labels = merge_small_components(xyz, labels, min_size)
    labels = np.unique(labels, return_inverse=True)[1]
    info["n_components_final"] = int(labels.max() + 1) if n else 0

    if gt_labels is not None:
        info["separability"] = separability_auroc(z, edges, gt_labels)
        raw_scores = edge_rigidity_score(traj, edges)
        info["separability_raw"] = separability_auroc(raw_scores, edges, gt_labels)

    return labels, info
