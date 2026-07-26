"""PNG renders of a point-cloud segmentation — no external mesh viewer needed to sanity
check results (a raw PLY is hard to reason about at a glance; these are just matplotlib
scatter projections, but they open anywhere and are easy to eyeball / drop into a write-up).

Pure numpy/matplotlib — no GPU, no torch.
"""
from __future__ import annotations

import numpy as np


def _palette(labels: np.ndarray) -> dict:
    """One color per label (tab20, cycled), with -1 (floaters/unlabeled) fixed to grey."""
    import matplotlib

    # matplotlib.cm.get_cmap(name) was removed in 3.9+; matplotlib.colormaps[name] is the
    # replacement and also works on older (3.5+) versions, so use it unconditionally.
    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    cmap = matplotlib.colormaps["tab20"]
    colors = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors[-1] = (0.6, 0.6, 0.6, 1.0)
    return colors


def _subsample(*arrays, max_points: int, seed: int = 0):
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.random.RandomState(seed).choice(n, max_points, replace=False)
    return tuple(a[idx] for a in arrays)


def _robust_limits(xyz: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, pad_frac: float = 0.08):
    """Per-axis (x,y,z) plot limits from a percentile range instead of full min/max, so a
    handful of far-flung outlier points (stray/under-trained Gaussians, common in an
    early-stage reconstruction) don't zoom the whole plot out until the real point cloud is a
    tiny speck in the middle. Points outside the range still get drawn — matplotlib just won't
    auto-expand the axes to fit them; only the visible *window* is clipped, not the data."""
    lo = np.percentile(xyz, lo_pct, axis=0)
    hi = np.percentile(xyz, hi_pct, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    pad = span * pad_frac
    return lo - pad, hi + pad


def render_segmentation_png(xyz, labels, out_path, title=None, max_points=120_000, point_size=1.5):
    """Three orthographic projections (top/front/side) of `xyz`, colored by `labels`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz, labels = _subsample(np.asarray(xyz), np.asarray(labels), max_points=max_points)
    colors_by_label = _palette(labels)
    point_colors = [colors_by_label[int(l)] for l in labels]

    # Draw floaters/unlabeled first so real segments layer on top and stay visible.
    order = np.argsort(labels == -1)[::-1]
    xyz, labels = xyz[order], labels[order]
    point_colors = [point_colors[i] for i in order]

    lo, hi = _robust_limits(xyz)
    views = [("Top (X-Y)", 0, 1), ("Front (X-Z)", 0, 2), ("Side (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, a, b) in zip(axes, views):
        ax.scatter(xyz[:, a], xyz[:, b], c=point_colors, s=point_size, linewidths=0)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        ax.invert_yaxis() if b == 2 else None  # z "up" reads more naturally increasing upward...
        ax.set_xticks([]); ax.set_yticks([])
    n_segments = len({l for l in labels if l != -1})
    n_floaters = int((labels == -1).sum())
    suptitle = title or f"{n_segments} segments" + (f", {n_floaters} floaters (grey)" if n_floaters else "")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_comparison_png(xyz, labels_a, labels_b, name_a, name_b, out_path,
                           max_points=120_000, point_size=1.5):
    """Two rows (top/front/side x 2 labelings) for a direct GT-vs-predicted visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz, labels_a, labels_b = _subsample(
        np.asarray(xyz), np.asarray(labels_a), np.asarray(labels_b), max_points=max_points
    )
    lo, hi = _robust_limits(xyz)
    views = [("Top (X-Y)", 0, 1), ("Front (X-Z)", 0, 2), ("Side (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, (labels, name) in enumerate([(labels_a, name_a), (labels_b, name_b)]):
        colors_by_label = _palette(labels)
        point_colors = np.array([colors_by_label[int(l)] for l in labels])
        order = np.argsort(labels == -1)[::-1]
        xyz_r, colors_r = xyz[order], point_colors[order]
        for col, (view_name, a, b) in enumerate(views):
            ax = axes[row, col]
            ax.scatter(xyz_r[:, a], xyz_r[:, b], c=colors_r, s=point_size, linewidths=0)
            ax.set_title(f"{name} — {view_name}")
            ax.set_aspect("equal")
            ax.set_xlim(lo[a], hi[a])
            ax.set_ylim(lo[b], hi[b])
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
