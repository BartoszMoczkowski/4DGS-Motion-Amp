#!/usr/bin/env python3
"""Generate multi-view segmentation visualization figures and benchmark plots for the spinning cubes dataset.

Outputs:
  - Benchmark line charts: ARI vs k and Mean IoU vs k across all 6 algorithms
  - Multi-view 3D scatter projection panels for scenes (Top view, Front view, 3D perspective)
  - Saves images to the conversation artifact directory for rich presentation in markdown
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(r"C:\Users\barte\.gemini\antigravity-ide\brain\2a8f94c7-82eb-434d-9eac-ac1f638d6062")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def plot_benchmark_curves():
    summary_path = REPO_ROOT / "runs" / "cubes_seg_summary.csv"
    if not summary_path.is_file():
        print(f"Summary file not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_impl = {}
    for r in rows:
        impl = r["impl"]
        k = int(r["k"])
        ari = float(r["ari"]) if r["ari"] else 0.0
        iou = float(r["mean_iou"]) if r["mean_iou"] else 0.0
        by_impl.setdefault(impl, {})[k] = (ari, iou)

    colors = {
        "mask_lift_oracle": "#10b981",  # Emerald green (ceiling)
        "rigid2": "#3b82f6",            # Blue
        "kabsch": "#8b5cf6",            # Purple
        "rigid2_roi": "#f59e0b",        # Amber
        "rigid": "#ef4444",             # Red
        "mbs": "#64748b",               # Slate
    }

    labels = {
        "mask_lift_oracle": "Oracle Mask ROI (T22 Ceiling)",
        "rigid2": "Upgraded Rigidity Graph (T18 rigid2)",
        "kabsch": "Kabsch EM (T20 kabsch)",
        "rigid2_roi": "Motion-Gated ROI (T19 rigid2_roi)",
        "rigid": "Baseline Rigidity Graph (rigid)",
        "mbs": "MultiBodySync (Option A MBS)",
    }

    markers = {
        "mask_lift_oracle": "o",
        "rigid2": "s",
        "kabsch": "^",
        "rigid2_roi": "D",
        "rigid": "x",
        "mbs": "v",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    ks = list(range(1, 9))
    for impl, name in labels.items():
        if impl not in by_impl:
            continue
        aris = [by_impl[impl].get(k, (0.0, 0.0))[0] for k in ks]
        ious = [by_impl[impl].get(k, (0.0, 0.0))[1] for k in ks]
        c = colors.get(impl, "#ffffff")
        m = markers.get(impl, "o")

        ax1.plot(ks, aris, label=name, color=c, marker=m, linewidth=2.5, markersize=8)
        ax2.plot(ks, ious, label=name, color=c, marker=m, linewidth=2.5, markersize=8)

    ax1.set_title("Adjusted Rand Index (ARI) vs Cube Count (k)", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Dynamic Cubes (k)", color="#e2e8f0", fontsize=12)
    ax1.set_ylabel("ARI (Higher is Better)", color="#e2e8f0", fontsize=12)
    ax1.set_xticks(ks)
    ax1.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=10, loc="upper left")

    ax2.set_title("Mean IoU vs Cube Count (k)", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Dynamic Cubes (k)", color="#e2e8f0", fontsize=12)
    ax2.set_ylabel("Mean IoU (Higher is Better)", color="#e2e8f0", fontsize=12)
    ax2.set_xticks(ks)
    ax2.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "cubes_segmentation_benchmark_curves.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark curves to {out_path}")


def get_palette(labels):
    uniq = sorted(int(l) for l in np.unique(labels) if l not in (-1, -2))
    cmap = matplotlib.colormaps["tab20"]
    colors = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors[-1] = (0.35, 0.40, 0.45, 0.4)  # Neutral slate grey for static background / floaters
    colors[-2] = (0.35, 0.40, 0.45, 0.4)  # Gated static background
    return colors


def render_scene_comparison(k: int):
    run_id = f"cubes-k{k}"
    run_dir = REPO_ROOT / "runs" / run_id
    traj_file = run_dir / "trajectories.npz"
    gt_file = run_dir / "convert_out" / "data" / "multipleview" / f"spinning_cubes_k{k}" / "gt_segmentation.npz"

    if not traj_file.is_file() or not gt_file.is_file():
        print(f"Missing data for {run_id}")
        return

    traj_data = np.load(traj_file)
    xyz = traj_data["canonical_xyz"]
    opacity = traj_data["opacity"] if "opacity" in traj_data.files else None

    # Load GT
    gt_data = np.load(gt_file)
    _, nn = cKDTree(gt_data["points"]).query(xyz, k=1)
    gt_labels = gt_data["labels"][nn]

    # Subsample for crisp visualization
    max_pts = 40000
    n = len(xyz)
    if n > max_pts:
        idx = np.random.RandomState(42).choice(n, max_pts, replace=False)
        xyz_sub = xyz[idx]
        gt_sub = gt_labels[idx]
        opacity_sub = opacity[idx] if opacity is not None else None
    else:
        xyz_sub = xyz
        gt_sub = gt_labels
        opacity_sub = opacity

    methods = [
        ("Ground Truth", gt_sub, f"GT ({k} cubes + floor)"),
        ("Oracle Mask ROI (T22)", "mask_lift_oracle", "Oracle Ceiling"),
        ("Upgraded Rigidity (rigid2)", "rigid2", "rigid2 (FFT + Calibrated z)"),
        ("Kabsch EM (kabsch)", "kabsch", f"kabsch (K={k+1})"),
        ("Motion-Gated ROI (rigid2_roi)", "rigid2_roi", "rigid2_roi (Energy Gating)"),
        ("Baseline Rigidity (rigid)", "rigid", "rigid (Baseline)"),
    ]

    fig = plt.figure(figsize=(20, 13), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for col_idx, (name, impl_or_labels, subtitle) in enumerate(methods):
        if isinstance(impl_or_labels, str):
            # Load predicted segmentation from colored PLY or npz
            pred_ply = run_dir / f"segmentation_colored_{impl_or_labels}.ply"
            pred_npz = run_dir / "segmentation.npz"
            # Read label from seg_eval result
            if impl_or_labels == "mask_lift_oracle":
                # Run quick eval mapping
                from pipeline.vendored.host.seg_eval import propagate_labels
                # Load saved segmentation
                data = np.load(pred_npz) if pred_npz.exists() else None
            # Extract labels from colored PLY or recompute/load
            # Or load directly from segmentation.npz if matches
        
        # We can map colors directly from predicted labels
        # Let's load the predictions
        # For simplicity and accuracy, load the stage's output
        pass

    plt.close(fig)


def render_all_pointcloud_visualizations():
    """Render 3D and 2D orthographic projections of all algorithms for key scenes."""
    run_ids = ["cubes-k1", "cubes-k2", "cubes-cubes_k2_both", "cubes-cubes_k2_ring", "cubes-k4", "cubes-k6", "cubes-k8"]
    
    for run_id in run_ids:
        run_dir = REPO_ROOT / "runs" / run_id
        traj_file = run_dir / "trajectories.npz"
        if not traj_file.is_file():
            continue

        gt_file = None
        for p in run_dir.rglob("gt_segmentation.npz"):
            gt_file = p
            break
        if not gt_file or not gt_file.is_file():
            continue

        traj_data = np.load(traj_file)
        xyz = traj_data["canonical_xyz"]
        gt_data = np.load(gt_file)
        _, nn = cKDTree(gt_data["points"]).query(xyz, k=1)
        gt_labels = gt_data["labels"][nn]

        # Subsample for rendering speed & clarity
        n = len(xyz)
        max_pts = 35000
        idx = np.random.RandomState(42).choice(n, min(n, max_pts), replace=False)
        xyz_s = xyz[idx]
        gt_s = gt_labels[idx]

        # Load segmentations
        impls = [
            ("Ground Truth", gt_s),
            ("Oracle Mask (T22)", run_dir / "segmentation_colored_mask_lift_oracle.ply"),
            ("rigid2 (T18)", run_dir / "segmentation_colored_rigid2.ply"),
            ("kabsch (T20)", run_dir / "segmentation_colored_kabsch.ply"),
            ("rigid2_roi (T19)", run_dir / "segmentation_colored_rigid2_roi.ply"),
            ("rigid (Baseline)", run_dir / "segmentation_colored_rigid.ply"),
        ]

        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor("#0f172a")

        # 2 rows: Row 1 = Top View (X-Y), Row 2 = 3D Perspective View
        for i, (title, data_source) in enumerate(impls):
            if isinstance(data_source, np.ndarray):
                labels_s = data_source
                palette = get_palette(labels_s)
                colors_s = [palette.get(int(l), (0.4, 0.4, 0.4, 0.4)) for l in labels_s]
            else:
                # Read rgb colors from PLY
                ply_path = data_source
                if not ply_path.exists():
                    continue
                # Read vertex colors from PLY
                with open(ply_path, "rb") as pf:
                    # Quick binary parse
                    header = ""
                    while not header.endswith("end_header\n"):
                        header += pf.readline().decode("ascii", errors="ignore")
                    dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
                    raw_data = np.fromfile(pf, dtype=dtype)
                    # Use subsampled indices
                    colors_raw = raw_data["rgb"][idx] / 255.0
                    # Set alpha
                    colors_s = np.column_stack([colors_raw, np.full(len(colors_raw), 0.7)])

            # Row 1: Top (X-Y)
            ax_top = fig.add_subplot(2, 6, i + 1)
            ax_top.set_facecolor("#1e293b")
            ax_top.scatter(xyz_s[:, 0], xyz_s[:, 1], c=colors_s, s=1.2, linewidths=0)
            ax_top.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
            ax_top.set_aspect("equal")
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            for sp in ax_top.spines.values():
                sp.set_color("#475569")

            # Row 2: 3D Perspective View
            ax_3d = fig.add_subplot(2, 6, 6 + i + 1, projection="3d")
            ax_3d.set_facecolor("#1e293b")
            ax_3d.scatter(xyz_s[:, 0], xyz_s[:, 1], xyz_s[:, 2], c=colors_s, s=0.8, depthshade=True)
            ax_3d.view_init(elev=28, azim=-45)
            ax_3d.set_xticks([])
            ax_3d.set_yticks([])
            ax_3d.set_zticks([])
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            ax_3d.xaxis.pane.set_edgecolor("#334155")
            ax_3d.yaxis.pane.set_edgecolor("#334155")
            ax_3d.zaxis.pane.set_edgecolor("#334155")

        fig.suptitle(f"Scene {run_id} — Segmentation Visual Comparison",
                     color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        out_file = ARTIFACT_DIR / f"{run_id}_seg_visual.png"
        plt.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"Generated visual comparison -> {out_file}")


def main():
    print("Generating segmentation benchmark visualization figures...")
    plot_benchmark_curves()
    render_all_pointcloud_visualizations()
    print("All visualizations rendered successfully!")


if __name__ == "__main__":
    main()
