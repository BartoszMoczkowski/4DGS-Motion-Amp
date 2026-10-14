"""
Generates tuned segmentations and renders high-res visual comparison figures:
- Kabsch sequential: K=6, K=12, K=20
- Tuned Rigid: min_size=500
- Tuned Rigid2: z_thresh=5.0, min_size=500
- Tuned Multicut: min_size=1200
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.vendored.host.rigidity_graph import segment_by_rigidity
from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2
from pipeline.vendored.host.multicut import postprocess_min_cluster_size

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(r"C:\Users\barte\.gemini\antigravity-ide\brain\3952d530-f235-4242-a1a9-4121ab2d036c")

def palette(labels: np.ndarray) -> dict:
    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    cmap = matplotlib.colormaps["tab20"]
    colors = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors[-1] = (0.6, 0.6, 0.6, 0.25)
    return colors

def subsample(*arrays, max_points: int = 120_000, seed: int = 42):
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

def render_figure(xyz_sub: np.ndarray, labels_sub: np.ndarray, title: str, out_prefix: str):
    colors_by_label = palette(labels_sub)
    point_colors = np.array([colors_by_label[int(l)] for l in labels_sub])

    order = np.argsort(labels_sub == -1)[::-1]
    xyz_sub = xyz_sub[order]
    labels_sub = labels_sub[order]
    point_colors = point_colors[order]

    lo, hi = robust_limits(xyz_sub)
    n_segments = len({l for l in labels_sub if l != -1})
    n_floaters = int((labels_sub == -1).sum())

    # 1. Orthographic 3-View
    views = [("Top View (X-Y)", 0, 1), ("Front View (X-Z)", 0, 2), ("Side View (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#121212")
    for ax, (name, a, b) in zip(axes, views):
        ax.set_facecolor("#1e1e1e")
        ax.scatter(xyz_sub[:, a], xyz_sub[:, b], c=point_colors, s=1.0, edgecolors="none")
        ax.set_title(name, color="white", fontsize=12, pad=8)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("#444444")
    fig.suptitle(f"{title}\nActive Clusters: {n_segments} | Floaters: {n_floaters:,} ({n_floaters/len(labels_sub)*100:.1f}%)",
                 color="white", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ortho_path = f"{out_prefix}_ortho.png"
    fig.savefig(ortho_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 2. 3D Perspective
    fig3d = plt.figure(figsize=(10, 8), facecolor="#121212")
    ax3d = fig3d.add_subplot(111, projection="3d", facecolor="#1e1e1e")
    ax3d.scatter(xyz_sub[:, 0], xyz_sub[:, 1], xyz_sub[:, 2], c=point_colors, s=1.2, depthshade=True, alpha=0.8)
    ax3d.set_title(f"{title} (3D Perspective)", color="white", fontsize=14, pad=12)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.tick_params(colors="gray")
    ax3d.view_init(elev=25, azim=45)
    fig3d.tight_layout()
    p3d_path = f"{out_prefix}_3d.png"
    fig3d.savefig(p3d_path, dpi=160, facecolor=fig3d.get_facecolor(), bbox_inches="tight")
    plt.close(fig3d)
    print(f"[Saved Renders] {ortho_path} & {p3d_path}", flush=True)

    # Copy to artifacts
    try:
        base_name = Path(out_prefix).name
        import shutil
        shutil.copy2(ortho_path, ARTIFACTS_DIR / f"{base_name}_ortho.png")
        shutil.copy2(p3d_path, ARTIFACTS_DIR / f"{base_name}_3d.png")
    except Exception as e:
        print(f"Artifact copy error: {e}", flush=True)

def process_run(run_name: str, prefix_tag: str):
    run_dir = REPO_ROOT / "runs" / run_name
    traj_path = run_dir / "trajectories.npz"
    data = np.load(traj_path)
    xyz, traj, op = data["canonical_xyz"], data["traj"], data["opacity"]
    n_total = len(xyz)
    keep = op > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]

    out_renders = run_dir / "segmentation_renders"
    out_renders.mkdir(exist_ok=True)

    # 1. KABSCH SEQUENTIAL: K=6, 12, 20
    for K in [6, 12, 20]:
        print(f"\nComputing Kabsch K={K} for {run_name}...", flush=True)
        sub_labels, _ = segment_by_kabsch(xyz_k, traj_k, n_clusters=K, max_iter=15, rng_seed=42)
        full_labels = np.full(n_total, -1, dtype=np.int64)
        full_labels[keep] = sub_labels
        
        npz_out = run_dir / f"segmentation_kabsch_k{K}.npz"
        np.savez(npz_out, points=xyz.astype(np.float32), labels=full_labels)
        
        xyz_sub, lbl_sub = subsample(xyz, full_labels)
        render_figure(xyz_sub, lbl_sub, f"{run_name} — Kabsch (K={K})", str(out_renders / f"{prefix_tag}_kabsch_k{K}"))

    # 2. TUNED RIGID (min_size=500)
    print(f"\nComputing Tuned Rigid (min_size=500) for {run_name}...", flush=True)
    sub_labels, _ = segment_by_rigidity(xyz_k, traj_k, k=16, threshold_mult=1.0, min_size=500)
    full_labels = np.full(n_total, -1, dtype=np.int64)
    full_labels[keep] = sub_labels
    npz_out = run_dir / "segmentation_rigid_tuned.npz"
    np.savez(npz_out, points=xyz.astype(np.float32), labels=full_labels)
    xyz_sub, lbl_sub = subsample(xyz, full_labels)
    render_figure(xyz_sub, lbl_sub, f"{run_name} — Rigid Tuned (min_size=500)", str(out_renders / f"{prefix_tag}_rigid_tuned"))

    # 3. TUNED RIGID2 (z_thresh=5.0, min_size=500)
    print(f"\nComputing Tuned Rigid2 (z_thresh=5.0, min_size=500) for {run_name}...", flush=True)
    sub_labels, _ = segment_by_rigidity2(xyz_k, traj_k, k=16, z_thresh=5.0, min_size=500, partition="components")
    full_labels = np.full(n_total, -1, dtype=np.int64)
    full_labels[keep] = sub_labels
    npz_out = run_dir / "segmentation_rigid2_tuned.npz"
    np.savez(npz_out, points=xyz.astype(np.float32), labels=full_labels)
    xyz_sub, lbl_sub = subsample(xyz, full_labels)
    render_figure(xyz_sub, lbl_sub, f"{run_name} — Rigid2 Tuned (z_thresh=5.0, min_size=500)", str(out_renders / f"{prefix_tag}_rigid2_tuned"))

    # 4. TUNED MULTICUT (min_size=1200)
    orig_mc = run_dir / "segmentation_multicut.npz"
    if orig_mc.exists():
        print(f"\nComputing Tuned Multicut (min_size=1200) for {run_name}...", flush=True)
        raw_mc = np.load(orig_mc)["labels"]
        sub_mc = raw_mc[keep]
        merged_sub = postprocess_min_cluster_size(xyz_k, sub_mc, min_size=1200)
        full_labels = np.full(n_total, -1, dtype=np.int64)
        full_labels[keep] = merged_sub
        npz_out = run_dir / "segmentation_multicut_tuned.npz"
        np.savez(npz_out, points=xyz.astype(np.float32), labels=full_labels)
        xyz_sub, lbl_sub = subsample(xyz, full_labels)
        render_figure(xyz_sub, lbl_sub, f"{run_name} — Multicut Tuned (min_size=1200)", str(out_renders / f"{prefix_tag}_multicut_tuned"))

def main():
    print("Processing hand_test_002...", flush=True)
    process_run("hand_test_002-a324fdd2", "hand")
    print("\nProcessing controler_test_01...", flush=True)
    process_run("controler_test_01-0aaf60a0", "ctrl")

if __name__ == "__main__":
    main()
