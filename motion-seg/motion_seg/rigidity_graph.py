"""Core, GPU-free motion-segmentation logic ("Option B" in
.claude_notes/NOTES_4dgs_motion_segmentation.md): build a local k-NN graph in canonical
(rest-pose) space, score each edge by how *rigid* it is over the clip (does the pairwise
distance between the two points stay constant over time?), cut the non-rigid edges, and take
connected components as the motion segments.

Why this works for 4DGS: a rigid transform preserves pairwise distances exactly, so two
points on the *same* rigid part have ~constant ||p_i(t) - p_j(t)|| for all t, while two points
on *different*, independently-moving parts generally don't. Restricting to a k-NN graph in
canonical space (rather than all O(N^2) pairs) is valid because rigid parts are spatially
contiguous — the only edges that need to be "cut" are the ones straddling a part boundary.

Pure numpy/scipy — no torch, no GPU. Safe to unit-test with `--selftest` on synthetic data
without a trained 4DGS model.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def build_knn_edges(xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """Undirected, deduplicated k-NN graph on `xyz` (N,3). Returns edges (E,2) int."""
    n = len(xyz)
    k = min(k, n - 1)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)  # column 0 is the point itself
    if idx.ndim == 1:  # k==1 edge case: scipy squeezes to 1D
        idx = idx[:, None]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    edges = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return edges


def edge_rigidity_score(traj: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Std-dev of pairwise distance over time for each edge. traj: (N,T,3). Returns (E,).
    Edges touching a NaN/Inf trajectory score NaN by propagation — callers must handle
    them (segment_by_rigidity cuts and reports them); the FP warning is suppressed here
    because that propagation is intentional."""
    i, j = edges[:, 0], edges[:, 1]
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(traj[i] - traj[j], axis=-1)  # (E, T)
        return d.std(axis=1)


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """1D Otsu threshold: the cut that maximizes between-class variance of a bimodal
    distribution. Used here to auto-separate "rigid" (same-body) edges from "non-rigid"
    (cross-body) ones without a hand-picked absolute cutoff, since the right cutoff depends
    on the scene's physical scale and the trained model's own position noise floor."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.max() == v.min():
        return float(v.max()) if len(v) else 0.0
    hist, bin_edges = np.histogram(v, bins=n_bins)
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = mu / w0
        mu1 = (mu_t - mu) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    # Plateau tie-breaking: argmax returns the FIRST bin of a max plateau, which
    # hugs the noise cluster on cleanly bimodal distributions and lets far points
    # leak in.  Use the plateau midpoint instead.
    idx = np.flatnonzero(between == between.max())
    return float(0.5 * (centers[idx[0]] + centers[idx[-1]]))


def otsu_threshold_log(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu threshold computed in log-space, then mapped back to linear.

    Plain (linear-histogram) Otsu breaks on the real edge-score distribution: it's heavily
    right-skewed (most edges near 0, a long tail out to rare large outliers — e.g. a few
    under-trained/noisy Gaussians with erratic trajectories), so linear histogram bins put
    almost all of the real same-body/cross-body separation into one or two bins near zero and
    let a handful of extreme outliers stretch the range. Otsu then finds "the tail vs. the
    rest" instead of "rigid vs. non-rigid", picking a threshold near the max and keeping
    ~every edge (this is exactly the bug seen on the real pump01 run: 99.99% of edges kept,
    threshold 0.38 vs. a median score of 0.0013). Log-space compresses that tail so the
    genuine near-zero/non-zero separation dominates the histogram instead.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    if v.max() == v.min():
        # Single-valued input (e.g. all scores clamped to the numerical-noise floor):
        # there is nothing to separate, and the exp/log roundtrip would otherwise return
        # a threshold a hair *below* every score, cutting all edges.
        return float(v.max())
    positive = v[v > 0]
    eps = float(positive.min() * 1e-3) if len(positive) else 1e-12
    log_thr = otsu_threshold(np.log(v + eps), n_bins=n_bins)
    return float(np.exp(log_thr) - eps)


def merge_small_components(xyz: np.ndarray, labels: np.ndarray, min_size: int):
    """Fold components smaller than `min_size` into their nearest large-enough neighbor
    (by canonical-space centroid distance), to avoid a long tail of noise-sized fragments.

    Returns (labels, merge_info) where merge_info has:
      n_small_merged: number of small components folded into a big neighbor
      merge_skipped_no_big_component: True if merging was requested but NO component
        reached `min_size` (nothing to merge into) — previously a silent no-op.
    """
    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    small = uniq[counts < min_size]
    big = uniq[counts >= min_size]
    merge_info = {"n_small_merged": 0, "merge_skipped_no_big_component": False}
    if len(small) == 0:
        return labels, merge_info
    if len(big) == 0:
        # No component reaches min_size: there is nothing to merge the small fragments
        # into. Keep labels as-is but flag it loudly — the output will be a long tail of
        # tiny fragments and downstream consumers should know.
        print(f"[warn] merge_small_components: {len(small)} component(s) below "
              f"min_size={min_size}, but NO component reaches min_size — merging "
              f"skipped, labels left as-is. Consider lowering --min-size.",
              file=sys.stderr)
        merge_info["merge_skipped_no_big_component"] = True
        return labels, merge_info
    big_mask = np.isin(labels, big)
    tree = cKDTree(xyz[big_mask])
    big_labels = labels[big_mask]
    for lab in small:
        pts_idx = np.where(labels == lab)[0]
        centroid = xyz[pts_idx].mean(axis=0, keepdims=True)
        _, nn = tree.query(centroid, k=1)
        labels[pts_idx] = big_labels[int(np.asarray(nn).reshape(-1)[0])]
    merge_info["n_small_merged"] = len(small)
    return labels, merge_info


def segment_by_rigidity(
    xyz: np.ndarray,
    traj: np.ndarray,
    k: int = 12,
    threshold: float | None = None,
    threshold_mult: float = 1.0,
    min_size: int = 15,
):
    """Full pipeline. xyz: (N,3) canonical positions. traj: (N,T,3) positions over time.

    NaN/Inf handling (added 2026-10-05): points whose canonical position or trajectory
    contains any non-finite value would otherwise poison edge scores (NaN comparisons are
    False, so `scores <= thr` silently CUTS every edge touching them, orphaning the point
    into an arbitrary neighbor). Instead they are handled deterministically and reported:
    (a) all graph edges touching a non-finite point are cut up front, so it becomes an
    isolated node; (b) for graph-building/merging purposes its position is replaced by the
    median of the finite cloud (used only when the position itself is non-finite), so it
    is folded into the component nearest the cloud center by merge_small_components;
    (c) the count is reported in info["n_nonfinite_points"].

    Returns (labels (N,) int, info dict with edge/threshold/component diagnostics).
    """
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64), dict(
            n_points=0, n_edges=0, n_kept_edges=0, threshold=0.0, score_median=0.0,
            score_p90=0.0, n_components_raw=0, n_components_final=0,
            n_nonfinite_points=0, n_small_merged=0, merge_skipped_no_big_component=False,
        )

    # --- NaN/Inf guard -----------------------------------------------------
    finite_pts = np.isfinite(xyz).all(axis=1) & np.isfinite(
        traj.reshape(n, -1)).all(axis=1)
    n_nonfinite = int((~finite_pts).sum())
    xyz_graph = xyz
    if n_nonfinite:
        n_finite = int(finite_pts.sum())
        if n_finite == 0:
            raise ValueError(
                "segment_by_rigidity: every point has a non-finite position or "
                "trajectory — nothing to segment. Check extract_trajectories output."
            )
        print(f"[warn] {n_nonfinite}/{n} point(s) have NaN/Inf in canonical position or "
              f"trajectory — cutting all their graph edges and folding them into the "
              f"component nearest the finite-cloud median (see docstring).",
              file=sys.stderr)
        xyz_graph = np.array(xyz, dtype=np.float64, copy=True)
        median_pos = np.median(xyz[finite_pts], axis=0)
        bad_xyz = ~np.isfinite(xyz).all(axis=1)
        xyz_graph[bad_xyz] = median_pos

    edges = build_knn_edges(xyz_graph, k=k)
    scores = edge_rigidity_score(traj, edges)
    # Numerical-noise floor tied to the scene scale: rigid edges are only rigid up to
    # float64 rounding (~1e-16 relative), so on clean data the "rigid" class is a mix of
    # exact zeros and ~1e-17 noise. Without this floor, log-space Otsu treats that rounding
    # noise as a real class and splits *inside* it (threshold between exact 0 and 1e-17),
    # shattering genuinely rigid parts into singletons. Clamping everything below
    # ~1e-12 of the scene's bounding-box diagonal collapses the noise band into one
    # histogram spike, far below any real non-rigid motion.
    scale = float(np.linalg.norm(xyz_graph.max(axis=0) - xyz_graph.min(axis=0)))
    scores = np.maximum(scores, scale * 1e-12)
    thr = otsu_threshold_log(scores) if threshold is None else threshold
    thr = thr * threshold_mult
    keep = scores <= thr
    if n_nonfinite:
        # Cut every edge touching a non-finite point, regardless of score (its score is
        # NaN/Inf and meaningless). Deterministic: isolated node -> merged by centroid.
        keep &= finite_pts[edges[:, 0]] & finite_pts[edges[:, 1]]

    rows = np.concatenate([edges[keep, 0], edges[keep, 1]])
    cols = np.concatenate([edges[keep, 1], edges[keep, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    n_components_raw, labels = connected_components(graph, directed=False)

    labels, merge_info = merge_small_components(xyz_graph, labels, min_size)
    labels = np.unique(labels, return_inverse=True)[1]  # relabel to 0..K-1 contiguous

    info = dict(
        n_points=n,
        n_edges=len(edges),
        n_kept_edges=int(keep.sum()),
        threshold=float(thr),
        score_median=float(np.median(scores[np.isfinite(scores)])) if np.isfinite(scores).any() else float("nan"),
        score_p90=float(np.percentile(scores[np.isfinite(scores)], 90)) if np.isfinite(scores).any() else float("nan"),
        n_components_raw=int(n_components_raw),
        n_components_final=int(labels.max() + 1) if n else 0,
        n_nonfinite_points=n_nonfinite,
        n_small_merged=merge_info["n_small_merged"],
        merge_skipped_no_big_component=merge_info["merge_skipped_no_big_component"],
    )
    return labels, info
