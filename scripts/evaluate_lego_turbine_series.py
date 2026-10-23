"""
scripts/evaluate_lego_turbine_series.py — Motion segmentation, cluster visualization,
and comparative vibration/defect analysis across:
1. lego_turbine_base-0a0a74b8
2. lego_turbine_2_missing-b9c0d39f
3. lego_turbine_5_missing-76ed4b89
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.vendored.host.rigidity_graph import segment_by_rigidity
from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

MODELS = [
    {
        "scene": "lego_turbine_base",
        "run_id": "lego_turbine_base-0a0a74b8",
        "condition": "Baseline intact (all blades present)",
        "missing_blades": 0,
    },
    {
        "scene": "lego_turbine_2_missing",
        "run_id": "lego_turbine_2_missing-b9c0d39f",
        "condition": "2 missing blades (minor imbalance)",
        "missing_blades": 2,
    },
    {
        "scene": "lego_turbine_5_missing",
        "run_id": "lego_turbine_5_missing-76ed4b89",
        "condition": "5 missing blades (severe structural damage)",
        "missing_blades": 5,
    },
]

SUMMARY_JSON = REPO_ROOT / "runs" / "lego_turbine_evaluation_summary.json"
SUMMARY_CSV = REPO_ROOT / "runs" / "lego_turbine_evaluation_summary.csv"


def render_figures(xyz: np.ndarray, labels: np.ndarray, out_prefix: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    cmap = matplotlib.colormaps["tab20"]
    colors_table = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors_table[-1] = (0.55, 0.55, 0.55, 0.2)

    n_pts = len(xyz)
    if n_pts > 100_000:
        idx = np.random.RandomState(42).choice(n_pts, 100_000, replace=False)
        xyz_sub, labels_sub = xyz[idx], labels[idx]
    else:
        xyz_sub, labels_sub = xyz, labels

    point_colors = np.array([colors_table[int(l)] for l in labels_sub])

    order = np.argsort(labels_sub == -1)[::-1]
    xyz_sub = xyz_sub[order]
    labels_sub = labels_sub[order]
    point_colors = point_colors[order]

    lo = np.percentile(xyz_sub, 2.0, axis=0)
    hi = np.percentile(xyz_sub, 98.0, axis=0)
    pad = np.maximum(hi - lo, 1e-6) * 0.08
    lo -= pad
    hi += pad

    # Orthographic Projections
    views = [("Top View (X-Y)", 0, 1), ("Front View (X-Z)", 0, 2), ("Side View (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#181818")
    for ax, (vname, a, b) in zip(axes, views):
        ax.set_facecolor("#222222")
        ax.scatter(xyz_sub[:, a], xyz_sub[:, b], c=point_colors, s=1.2, edgecolors="none")
        ax.set_title(vname, color="white", fontsize=13, pad=8)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        if b == 2:
            ax.invert_yaxis()
        ax.tick_params(colors="gray")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.4)

    fig.suptitle(f"{title}\n({len(uniq)} Motion Clusters)", color="white", fontsize=15, y=0.98)
    fig.tight_layout()
    ortho_png = f"{out_prefix}_orthographic.png"
    fig.savefig(ortho_png, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 3D View
    fig3d = plt.figure(figsize=(10, 8), facecolor="#181818")
    ax3d = fig3d.add_subplot(111, projection="3d", facecolor="#222222")
    ax3d.scatter(xyz_sub[:, 0], xyz_sub[:, 1], xyz_sub[:, 2], c=point_colors, s=1.2, depthshade=True)
    ax3d.set_title(f"{title} — 3D Cluster View", color="white", fontsize=13, pad=10)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.tick_params(colors="gray")
    ax3d.view_init(elev=25, azim=45)
    fig3d.tight_layout()
    p3d_png = f"{out_prefix}_3d.png"
    fig3d.savefig(p3d_png, dpi=160, facecolor=fig3d.get_facecolor(), bbox_inches="tight")
    plt.close(fig3d)


def evaluate_model(m_info: dict) -> dict:
    scene = m_info["scene"]
    run_id = m_info["run_id"]
    condition = m_info["condition"]
    run_dir = REPO_ROOT / "runs" / run_id

    print(f"\n=========================================================================")
    print(f"Evaluating: {scene} ({run_id})")
    print(f"Condition: {condition}")
    print(f"=========================================================================")

    traj_path = run_dir / "trajectories.npz"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")

    data = np.load(traj_path)
    xyz = data["canonical_xyz"]
    traj = data["traj"]
    opacity = data["opacity"]

    keep = opacity > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]

    # Trajectory displacement stats
    disp = np.linalg.norm(traj_k - xyz_k[:, np.newaxis, :], axis=-1)  # (N_kept, T)
    max_disp = np.max(disp, axis=-1)
    mean_disp = float(np.mean(max_disp))
    median_disp = float(np.median(max_disp))
    p95_disp = float(np.percentile(max_disp, 95))
    peak_disp = float(np.max(max_disp))

    print(f"Gaussians total: {len(xyz)}, Kept (>0.1 op): {len(xyz_k)} ({len(xyz_k)/len(xyz)*100:.1f}%)")
    print(f"Trajectory Excursion: mean={mean_disp:.4f}, median={median_disp:.4f}, p95={p95_disp:.4f}, max={peak_disp:.4f}")

    renders_out_dir = run_dir / "segmentation_renders"
    renders_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kabsch
    print("Running Kabsch segmentation (K=8)...", flush=True)
    k_labels, _ = segment_by_kabsch(xyz_k, traj_k, n_clusters=8, max_iter=10, rng_seed=42)
    full_k = np.full(len(xyz), -1, dtype=np.int64)
    full_k[keep] = k_labels
    np.savez_compressed(run_dir / "segmentation_kabsch.npz", points=xyz, labels=full_k)
    render_figures(xyz, full_k, str(renders_out_dir / "kabsch"), f"{scene} — Kabsch (K=8)")
    k_clusters = len(set(k_labels[k_labels != -1]))

    # 2. Rigid Graph
    print("Running Rigid Graph clustering...", flush=True)
    r_labels, _ = segment_by_rigidity(xyz_k, traj_k, k=16, threshold_mult=2.0, min_size=200)
    full_r = np.full(len(xyz), -1, dtype=np.int64)
    full_r[keep] = r_labels
    np.savez_compressed(run_dir / "segmentation_rigid.npz", points=xyz, labels=full_r)
    render_figures(xyz, full_r, str(renders_out_dir / "rigid"), f"{scene} — Rigid Graph")
    r_clusters = len(set(r_labels[r_labels != -1]))

    # 3. Rigid2 Spectral
    print("Running Rigid2 Spectral clustering...", flush=True)
    r2_labels, _ = segment_by_rigidity2(xyz_k, traj_k, z_thresh=5.0, min_size=200, partition="components")
    full_r2 = np.full(len(xyz), -1, dtype=np.int64)
    full_r2[keep] = r2_labels
    np.savez_compressed(run_dir / "segmentation_rigid2.npz", points=xyz, labels=full_r2)
    render_figures(xyz, full_r2, str(renders_out_dir / "rigid2"), f"{scene} — Rigid2 Spectral")
    r2_clusters = len(set(r2_labels[r2_labels != -1]))

    print(f"Clusters: Kabsch={k_clusters}, Rigid={r_clusters}, Rigid2={r2_clusters}", flush=True)

    return {
        "scene": scene,
        "run_id": run_id,
        "condition": condition,
        "missing_blades": m_info["missing_blades"],
        "total_gaussians": len(xyz),
        "kept_gaussians": len(xyz_k),
        "mean_displacement": round(mean_disp, 4),
        "p95_displacement": round(p95_disp, 4),
        "max_displacement": round(peak_disp, 4),
        "kabsch_clusters": k_clusters,
        "rigid_clusters": r_clusters,
        "rigid2_clusters": r2_clusters,
    }


def main():
    results = []
    for m in MODELS:
        res = evaluate_model(m)
        results.append(res)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    keys = list(results[0].keys())
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n=========================================================================")
    print("EVALUATION COMPLETE ACROSS ALL 3 MODELS!")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Summary CSV:  {SUMMARY_CSV}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
