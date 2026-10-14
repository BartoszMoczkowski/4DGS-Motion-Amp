"""
Renders 2D orthographic and 3D perspective visualizations of motion segmentation results.
Saves PNG files into runs/<run_id>/segmentation_renders/
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

def palette(labels: np.ndarray) -> dict:
    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    cmap = matplotlib.colormaps["tab20"]
    colors = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors[-1] = (0.7, 0.7, 0.7, 0.3) # subtle grey for floaters/unassigned
    return colors

def subsample(*arrays, max_points: int = 150_000, seed: int = 0):
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.random.RandomState(seed).choice(n, max_points, replace=False)
    return tuple(a[idx] for a in arrays)

def robust_limits(xyz: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0, pad_frac: float = 0.08):
    lo = np.percentile(xyz, lo_pct, axis=0)
    hi = np.percentile(xyz, hi_pct, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    pad = span * pad_frac
    return lo - pad, hi + pad

def render_segmentation(npz_path: str, out_prefix: str, scene_title: str):
    data = np.load(npz_path)
    xyz = data["points"]
    labels = data["labels"]

    xyz, labels = subsample(xyz, labels, max_points=120_000)
    colors_by_label = palette(labels)
    point_colors = np.array([colors_by_label[int(l)] for l in labels])

    # Put floaters (-1) in the background so real clusters appear prominently
    order = np.argsort(labels == -1)[::-1]
    xyz = xyz[order]
    labels = labels[order]
    point_colors = point_colors[order]

    lo, hi = robust_limits(xyz)
    n_segments = len({l for l in labels if l != -1})
    n_floaters = int((labels == -1).sum())

    # 1. Orthographic 3-View Figure
    views = [("Top View (X-Y)", 0, 1), ("Front View (X-Z)", 0, 2), ("Side View (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#121212")
    
    for ax, (name, a, b) in zip(axes, views):
        ax.set_facecolor("#1e1e1e")
        ax.scatter(xyz[:, a], xyz[:, b], c=point_colors, s=1.2, linewidths=0, alpha=0.85)
        ax.set_title(name, color="white", fontsize=14, pad=10)
        ax.set_aspect("equal")
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        if b == 2:
            ax.invert_yaxis()
        ax.tick_params(colors="gray")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

    fig.suptitle(f"{scene_title} — Orthographic Projections\n({n_segments} Motion Clusters, {n_floaters} Floaters Filtered)", 
                 color="white", fontsize=16, y=0.98)
    fig.tight_layout()
    ortho_png = f"{out_prefix}_orthographic.png"
    fig.savefig(ortho_png, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {ortho_png}")

    # 2. 3D Perspective View Figure
    fig3d = plt.figure(figsize=(10, 8), facecolor="#121212")
    ax3d = fig3d.add_subplot(111, projection="3d", facecolor="#1e1e1e")
    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=point_colors, s=1.2, depthshade=True, alpha=0.8)
    ax3d.set_title(f"{scene_title} — 3D Perspective Clustering", color="white", fontsize=14, pad=12)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.tick_params(colors="gray")
    ax3d.view_init(elev=25, azim=45)
    
    fig3d.tight_layout()
    p3d_png = f"{out_prefix}_3d.png"
    fig3d.savefig(p3d_png, dpi=160, facecolor=fig3d.get_facecolor(), bbox_inches="tight")
    plt.close(fig3d)
    print(f"[Saved] {p3d_png}")

    return ortho_png, p3d_png

def main():
    runs_dir = REPO_ROOT / "runs"
    
    if len(sys.argv) > 1:
        target_runs = [runs_dir / arg for arg in sys.argv[1:]]
    else:
        target_runs = [runs_dir / "hand_test_002-a324fdd2", runs_dir / "controler_test_01-0aaf60a0"]

    for rdir in target_runs:
        if not rdir.is_dir():
            print(f"Skipping non-existent directory {rdir}")
            continue
        
        run_name = rdir.name
        out_renders_dir = rdir / "segmentation_renders"
        os.makedirs(out_renders_dir, exist_ok=True)

        for npz_file in sorted(rdir.glob("segmentation_*.npz")):
            method_name = npz_file.stem.replace("segmentation_", "")
            print(f"\nProcessing {run_name} / {method_name}...")
            out_prefix = str(out_renders_dir / f"{method_name}")
            render_segmentation(str(npz_file), out_prefix, f"{run_name} ({method_name})")

if __name__ == "__main__":
    main()
