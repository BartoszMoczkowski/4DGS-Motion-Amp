"""scratch/calibrate_multicut_weights.py — Calibrate Multicut logit weights on grid-A20mm_M2."""

from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scene-gen"))

import numpy as np
from scipy.spatial import cKDTree
from pipeline.vendored.host.multicut import (
    build_multichannel_graph,
    compute_edge_logits,
    solve_multicut,
    postprocess_min_cluster_size,
)
from pipeline.vendored.host.metrics import adjusted_rand_index
from run_pump01_spatial_cc import read_point_cloud_ply, fast_best_iou_matching

def main():
    t0 = time.time()
    run_dir = Path("runs/grid-A20mm_M2")
    traj_data = np.load(run_dir / "trajectories.npz")
    xyz = traj_data["canonical_xyz"]
    traj = traj_data["traj"]
    ply = read_point_cloud_ply(run_dir / "train_out/point_cloud/iteration_15000/point_cloud.ply")
    rgb = ply["rgb"]

    # GT labels
    gt_files = list((run_dir / "convert_out/data/multipleview").glob("*/gt_segmentation.npz"))
    gt = np.load(gt_files[0])
    tree = cKDTree(gt["points"])
    _, nn = tree.query(xyz, k=1)
    gt_labels = gt["labels"][nn]

    roi_mask = gt_labels > 0
    print(f"Loaded {len(xyz)} points, {len(np.unique(gt_labels))} GT parts, {roi_mask.sum()} in dynamic ROI (in {time.time()-t0:.2f}s)", flush=True)

    # Build k-NN graph
    t1 = time.time()
    edges, feats, info = build_multichannel_graph(
        xyz, traj=traj, rgb=rgb, k=12, radius_max=0.03, f0=10.0, include_lifted=False
    )
    print(f"Graph built in {time.time()-t1:.2f}s: {info}", flush=True)

    u = edges[:, 0]
    v = edges[:, 1]
    same = (gt_labels[u] == gt_labels[v])
    print(f"Total edges: {len(edges)}, Same-part: {same.sum()} ({same.mean():.2%}), Cross-part: {(~same).sum()} ({(~same).mean():.2%})", flush=True)

    # Systematic parameter search
    best_ari_roi = -1.0
    best_weights = None
    best_summary = None

    print("\nStarting parameter sweep...", flush=True)
    # Testing radius scale r_0, color threshold c_0, phasor weight p_w
    for r_0 in [0.003, 0.004, 0.005, 0.006, 0.008]:
        for c_0 in [0.10, 0.15, 0.20, 0.25]:
            for p_w in [0.0, 0.5, 1.0, 2.0]:
                # Logit formula:
                # w_e = 4.0 - 4.0 * (d_xyz / r_0) - 4.0 * (d_rgb / c_0) - p_w * disc_phasor
                w_xyz = -4.0 / r_0
                w_rgb = -4.0 / c_0
                intercept = 4.0

                w_dict = {
                    "intercept": intercept,
                    "w_xyz": w_xyz,
                    "w_rgb": w_rgb,
                    "w_phasor": 0.0,
                    "w_disc_phasor": -p_w,
                    "w_disc_traj": 0.0,
                    "w_lifted": -2.0,
                }
                weights = compute_edge_logits(feats, w_dict)
                labels, meta = solve_multicut(len(xyz), edges, weights, refine_kl=False)
                labels = postprocess_min_cluster_size(xyz, labels, min_size=15)

                ari_roi = float(adjusted_rand_index(gt_labels[roi_mask], labels[roi_mask]))
                ari_global = float(adjusted_rand_index(gt_labels, labels))
                k_pred = len(np.unique(labels))

                if ari_roi > best_ari_roi:
                    best_ari_roi = ari_roi
                    best_weights = w_dict
                    iou, _ = fast_best_iou_matching(gt_labels, labels)
                    best_summary = {
                        "r_0_mm": r_0 * 1000,
                        "c_0": c_0,
                        "p_w": p_w,
                        "ari_roi": ari_roi,
                        "ari_global": ari_global,
                        "mean_iou": iou,
                        "k_pred": k_pred,
                    }
                    print(f"--> NEW BEST: r_0={r_0*1000:.1f}mm c_0={c_0:.2f} p_w={p_w:.1f} | ARI_roi={ari_roi:.4f} ARI_glob={ari_global:.4f} IoU={iou:.4f} K_pred={k_pred}", flush=True)

    print(f"\n=======================================================", flush=True)
    print(f"Optimal Calibrated Weights: {best_weights}", flush=True)
    print(f"Optimal Summary: {best_summary}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == "__main__":
    main()
