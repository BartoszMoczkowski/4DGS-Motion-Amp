"""scene-gen/run_grid_multicut_calibrated.py — Calibrated Multi-Channel Multicut Ablation on pump01.

Implements:
1. Logistic regression edge weight calibration on grid-A20mm_M2 GT.
2. Per-channel feature separability AUROC measurement (spatial, appearance, motion, full fused).
3. 4-way ablation across all 3 grid models (A20mm_M2, A20mm_M4, A40mm_M8) with frozen weights:
   (a) spatial only
   (b) spatial + appearance
   (c) spatial + motion
   (d) spatial + appearance + motion (full fusion)
4. Benchmarks against the S6 spatial-CC baseline.
5. Emits runs/grid_seg_multicut_calibrated_results.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.vendored.host.metrics import adjusted_rand_index
from pipeline.vendored.host.multicut import (
    build_multichannel_graph,
    extract_motion_features,
    solve_multicut,
    postprocess_min_cluster_size,
)
from run_pump01_spatial_cc import read_point_cloud_ply, fast_best_iou_matching, spatial_cc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_feature_auroc(feat_values: np.ndarray, same_labels: np.ndarray, higher_is_same: bool = False, max_eval: int = 10000) -> float:
    """Compute AUROC of a 1D edge feature separating same-part (pos) from cross-part (neg) edges."""
    pos_vals = feat_values[same_labels]
    neg_vals = feat_values[~same_labels]
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return 0.5

    rng = np.random.default_rng(42)
    n_p = min(max_eval, len(pos_vals))
    n_n = min(max_eval, len(neg_vals))
    p_sub = rng.choice(pos_vals, size=n_p, replace=False)
    n_sub = rng.choice(neg_vals, size=n_n, replace=False)

    if higher_is_same:
        gt_mat = p_sub[:, None] > n_sub[None, :]
        eq_mat = p_sub[:, None] == n_sub[None, :]
    else:
        gt_mat = p_sub[:, None] < n_sub[None, :]
        eq_mat = p_sub[:, None] == n_sub[None, :]

    auroc = float(np.mean(gt_mat) + 0.5 * np.mean(eq_mat))
    return auroc


def fit_logistic_weights(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100000,
    rng_seed: int = 42,
) -> Tuple[Dict[str, float], float]:
    """Fit balanced L2-regularized logistic regression for edge probability: P(same | X)."""
    N_edges = len(y)
    if N_edges > max_samples:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(N_edges, size=max_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    mu = X_sub.mean(axis=0)
    sigma = X_sub.std(axis=0) + 1e-6
    X_norm = (X_sub - mu) / sigma
    X_ext = np.column_stack([np.ones(len(X_sub)), X_norm])

    # Balanced class weights
    w_pos = 1.0 / (y_sub.mean() + 1e-6)
    w_neg = 1.0 / ((1.0 - y_sub.mean()) + 1e-6)
    sample_weights = np.where(y_sub == 1, w_pos, w_neg)

    def obj(w: np.ndarray) -> float:
        logits = np.clip(X_ext @ w, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        bce = -np.mean(sample_weights * (y_sub * np.log(p + 1e-9) + (1.0 - y_sub) * np.log(1.0 - p + 1e-9)))
        reg = 1e-4 * np.sum(w[1:] ** 2)
        return bce + reg

    res = minimize(obj, np.zeros(X_ext.shape[1]), method="L-BFGS-B")
    w_opt = res.x

    # Un-normalize weights back to raw feature space:
    # logit = w0 + sum_i w_i * (x_i - mu_i) / sigma_i = (w0 - sum_i w_i * mu_i / sigma_i) + sum_i (w_i / sigma_i) * x_i
    raw_intercept = float(w_opt[0] - np.sum(w_opt[1:] * mu / sigma))
    raw_betas = (w_opt[1:] / sigma).tolist()

    weights_dict = {"intercept": raw_intercept}
    for name, b in zip(feature_names, raw_betas):
        weights_dict[name] = float(b)

    # Compute fused model AUROC on the full edge set
    full_X_ext = np.column_stack([np.ones(len(X)), X])
    full_w = np.array([raw_intercept] + raw_betas)
    fused_logits = full_X_ext @ full_w
    fused_auroc = compute_feature_auroc(fused_logits, y.astype(bool), higher_is_same=True)

    return weights_dict, fused_auroc


def evaluate_segmentation_dict(
    full_pred_labels: np.ndarray,
    gt_on_pred: np.ndarray,
    oracle_roi: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate ARI (within ROI and global), Mean IoU, and cluster counts."""
    ari_global = float(adjusted_rand_index(gt_on_pred, full_pred_labels))
    mean_iou_global, _ = fast_best_iou_matching(gt_on_pred, full_pred_labels)

    roi_pts = oracle_roi
    ari_roi = float(adjusted_rand_index(gt_on_pred[roi_pts], full_pred_labels[roi_pts]))
    mean_iou_roi, _ = fast_best_iou_matching(gt_on_pred[roi_pts], full_pred_labels[roi_pts])

    k_pred_roi = int(len(np.unique(full_pred_labels[roi_pts])))
    k_pred_total = int(len(np.unique(full_pred_labels)))

    return {
        "ari_within_roi": ari_roi,
        "mean_iou_within_roi": mean_iou_roi,
        "k_pred_roi": k_pred_roi,
        "ari_global": ari_global,
        "mean_iou_global": mean_iou_global,
        "k_pred_total": k_pred_total,
    }


def main():
    print("================================================================================", flush=True)
    print("      E6R: CALIBRATED MULTI-CHANNEL AFFINITY GRAPH FUSION & ABLATION            ", flush=True)
    print("================================================================================", flush=True)

    checkpoints = [
        ("grid-A20mm_M2", "A20mm_M2", True),    # Train set
        ("grid-A20mm_M4", "A20mm_M4", False),   # Held-out
        ("grid-A40mm_M8", "A40mm_M8", False),   # Held-out
    ]

    # Load Training Model: grid-A20mm_M2
    train_dir = REPO_ROOT / "runs" / "grid-A20mm_M2"
    train_traj_data = np.load(train_dir / "trajectories.npz")
    train_xyz = train_traj_data["canonical_xyz"]
    train_traj = train_traj_data["traj"]
    train_ply = read_point_cloud_ply(train_dir / "train_out/point_cloud/iteration_15000/point_cloud.ply")
    train_rgb = train_ply["rgb"]

    train_gt_file = list(train_dir.glob("**/gt_segmentation.npz"))[0]
    train_gt_data = np.load(train_gt_file)
    train_tree = cKDTree(train_gt_data["points"])
    _, train_nn = train_tree.query(train_xyz, k=1)
    train_gt_labels = train_gt_data["labels"][train_nn]
    train_roi = train_gt_labels > 0

    print(f"\n[1] Extracting multi-channel features on training cell grid-A20mm_M2...", flush=True)
    train_edges, train_feats, train_info = build_multichannel_graph(
        train_xyz,
        traj=train_traj,
        rgb=train_rgb,
        k=12,
        radius_max=0.03,
        f0=10.0,
        include_lifted=False,
    )
    u_tr = train_edges[:, 0]
    v_tr = train_edges[:, 1]
    y_train = (train_gt_labels[u_tr] == train_gt_labels[v_tr]).astype(np.float64)

    print(f"    Graph: {train_info['n_nodes']} nodes, {train_info['n_edges']} edges. Same-part: {y_train.sum():.0f} ({y_train.mean()*100:.2f}%)", flush=True)

    # Feature columns:
    # col 0: d_xyz
    # col 1: d_rgb
    # col 2: d_phasor
    # col 3: disc_phasor (1 - r_phasor)
    # col 4: disc_traj (1 - r_traj)
    # col 5: is_lifted

    # Measure Per-Channel AUROCs
    auroc_spatial = compute_feature_auroc(train_feats[:, 0], y_train.astype(bool), higher_is_same=False)
    auroc_appearance = compute_feature_auroc(train_feats[:, 1], y_train.astype(bool), higher_is_same=False)
    auroc_phasor = compute_feature_auroc(train_feats[:, 3], y_train.astype(bool), higher_is_same=False)
    auroc_traj = compute_feature_auroc(train_feats[:, 4], y_train.astype(bool), higher_is_same=False)

    print("\n--------------------------------------------------------------------------------", flush=True)
    print("                    PER-CHANNEL EDGE SEPARABILITY AUROC                         ", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)
    print(f"  Channel 1 (Spatial d_xyz)             : AUROC = {auroc_spatial:.4f}", flush=True)
    print(f"  Channel 2 (Appearance d_rgb)          : AUROC = {auroc_appearance:.4f}", flush=True)
    print(f"  Channel 3 (Motion Phasor disc_phasor) : AUROC = {auroc_phasor:.4f}", flush=True)
    print(f"  Channel 4 (Motion Full Trajectory)    : AUROC = {auroc_traj:.4f}", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    # Fit Logistic Regression Models for the 4 Ablation Configurations
    # Config (a): Spatial only [d_xyz]
    w_spatial, auroc_fit_a = fit_logistic_weights(
        train_feats[:, [0]], y_train, ["beta_xyz"]
    )
    # Config (b): Spatial + Appearance [d_xyz, d_rgb]
    w_spat_app, auroc_fit_b = fit_logistic_weights(
        train_feats[:, [0, 1]], y_train, ["beta_xyz", "beta_rgb"]
    )
    # Config (c): Spatial + Motion [d_xyz, disc_phasor]
    w_spat_mot, auroc_fit_c = fit_logistic_weights(
        train_feats[:, [0, 3]], y_train, ["beta_xyz", "beta_phasor"]
    )
    # Config (d): Spatial + Appearance + Motion (Full Fusion) [d_xyz, d_rgb, disc_phasor]
    w_full, auroc_fit_d = fit_logistic_weights(
        train_feats[:, [0, 1, 3]], y_train, ["beta_xyz", "beta_rgb", "beta_phasor"]
    )

    print("\n--------------------------------------------------------------------------------", flush=True)
    print("               LEARNED LOGISTIC REGRESSION WEIGHTS (FROZEN)                     ", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)
    print(f"  (a) Spatial Only     : intercept = {w_spatial['intercept']:+7.3f}, beta_xyz = {w_spatial['beta_xyz']:+8.3f} | Fused AUROC = {auroc_fit_a:.4f}", flush=True)
    print(f"  (b) Spatial + Appear : intercept = {w_spat_app['intercept']:+7.3f}, beta_xyz = {w_spat_app['beta_xyz']:+8.3f}, beta_rgb = {w_spat_app['beta_rgb']:+8.3f} | Fused AUROC = {auroc_fit_b:.4f}", flush=True)
    print(f"  (c) Spatial + Motion : intercept = {w_spat_mot['intercept']:+7.3f}, beta_xyz = {w_spat_mot['beta_xyz']:+8.3f}, beta_mot = {w_spat_mot['beta_phasor']:+8.3f} | Fused AUROC = {auroc_fit_c:.4f}", flush=True)
    print(f"  (d) Full Fusion      : intercept = {w_full['intercept']:+7.3f}, beta_xyz = {w_full['beta_xyz']:+8.3f}, beta_rgb = {w_full['beta_rgb']:+8.3f}, beta_mot = {w_full['beta_phasor']:+8.3f} | Fused AUROC = {auroc_fit_d:.4f}", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    configs = {
        "spatial_only": (w_spatial, [0], "Spatial Only"),
        "spatial_appearance": (w_spat_app, [0, 1], "Spatial + Appearance"),
        "spatial_motion": (w_spat_mot, [0, 3], "Spatial + Motion"),
        "full_fusion": (w_full, [0, 1, 3], "Spatial + Appear + Motion"),
    }

    out_csv = REPO_ROOT / "runs" / "grid_seg_multicut_calibrated_results.csv"
    csv_rows = []

    print("\n[2] Running 4-Way Ablation Benchmark across all 3 Grid Models...", flush=True)

    for dir_name, exp_name, is_train in checkpoints:
        run_dir = REPO_ROOT / "runs" / dir_name
        print(f"\n>>> Evaluating on {'[TRAIN]' if is_train else '[HELD-OUT]'} {exp_name} ({run_dir.name})...", flush=True)

        traj_data = np.load(run_dir / "trajectories.npz")
        xyz = traj_data["canonical_xyz"]
        traj = traj_data["traj"]
        ply_file = sorted((run_dir / "train_out" / "point_cloud").glob("*/*.ply"))[-1]
        ply = read_point_cloud_ply(ply_file)
        rgb = ply["rgb"]

        gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
        gt_data = np.load(gt_file)
        tree = cKDTree(gt_data["points"])
        _, nn = tree.query(xyz, k=1)
        gt_labels = gt_data["labels"][nn]
        oracle_roi = (gt_labels > 0)
        n_roi = int(oracle_roi.sum())

        # Baseline S6 Spatial CC (r=3mm / r=5mm)
        # On A20mm_M2 best r=3mm, on A40mm_M8 best r=5mm
        r_opt = 0.003 if "A20mm" in exp_name else 0.005
        cc_labels, k_cc = spatial_cc(xyz[oracle_roi], radius=r_opt, min_cluster_size=15)
        full_cc_labels = np.full(len(xyz), -1, dtype=np.int64)
        full_cc_labels[oracle_roi] = cc_labels
        full_cc_labels[~oracle_roi] = 0
        eval_s6 = evaluate_segmentation_dict(full_cc_labels, gt_labels, oracle_roi)

        row_s6 = {
            "checkpoint": exp_name,
            "is_train": is_train,
            "config_id": "baseline_s6_spatial_cc",
            "config_name": f"S6 Spatial CC (r={r_opt*100:.1f}cm)",
            "ari_within_roi": eval_s6["ari_within_roi"],
            "mean_iou_within_roi": eval_s6["mean_iou_within_roi"],
            "k_pred_roi": eval_s6["k_pred_roi"],
            "ari_global": eval_s6["ari_global"],
            "mean_iou_global": eval_s6["mean_iou_global"],
            "k_pred_total": eval_s6["k_pred_total"],
            "delta_vs_s6": 0.0,
        }
        csv_rows.append(row_s6)
        print(f"    Baseline S6 Spatial CC : ARI_roi = {eval_s6['ari_within_roi']:.4f}, IoU_roi = {eval_s6['mean_iou_within_roi']:.4f}, K_roi = {eval_s6['k_pred_roi']:3d}", flush=True)

        # Build candidate graph for multicut
        edges, feats, info = build_multichannel_graph(
            xyz,
            traj=traj,
            rgb=rgb,
            k=12,
            radius_max=0.03,
            f0=10.0,
            include_lifted=False,
        )

        for cfg_id, (w_dict, feat_cols, cfg_name) in configs.items():
            # Compute logits using the frozen calibrated linear model
            intercept = w_dict["intercept"]
            betas = [w_dict[k] for k in w_dict if k != "intercept"]
            logits = np.full(len(edges), intercept, dtype=np.float64)
            for col, b in zip(feat_cols, betas):
                logits += b * feats[:, col]

            # Multicut partition
            raw_labels, meta = solve_multicut(len(xyz), edges, logits.astype(np.float32), refine_kl=True)
            labels = postprocess_min_cluster_size(xyz, raw_labels, min_size=15)

            # Mask outside dynamic ROI to static casing (0)
            full_mc_labels = np.full(len(xyz), 0, dtype=np.int64)
            full_mc_labels[oracle_roi] = labels[oracle_roi]

            eval_mc = evaluate_segmentation_dict(full_mc_labels, gt_labels, oracle_roi)
            delta = eval_mc["ari_within_roi"] - eval_s6["ari_within_roi"]

            row_mc = {
                "checkpoint": exp_name,
                "is_train": is_train,
                "config_id": cfg_id,
                "config_name": cfg_name,
                "ari_within_roi": eval_mc["ari_within_roi"],
                "mean_iou_within_roi": eval_mc["mean_iou_within_roi"],
                "k_pred_roi": eval_mc["k_pred_roi"],
                "ari_global": eval_mc["ari_global"],
                "mean_iou_global": eval_mc["mean_iou_global"],
                "k_pred_total": eval_mc["k_pred_total"],
                "delta_vs_s6": delta,
            }
            csv_rows.append(row_mc)
            print(f"    ({cfg_id[:1].upper()}) {cfg_name:24s} : ARI_roi = {eval_mc['ari_within_roi']:.4f} (diff={delta:+.4f}), IoU_roi = {eval_mc['mean_iou_within_roi']:.4f}, K_roi = {eval_mc['k_pred_roi']:3d}", flush=True)

    # Write results CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[3] Results successfully saved to {out_csv}", flush=True)

    # Gate Evaluation & Summary
    print("\n================================================================================", flush=True)
    print("                             ABLATION SUMMARY & GATE VERDICT                    ", flush=True)
    print("================================================================================", flush=True)
    header = f"{'Checkpoint':<14} | {'Method':<28} | {'ARI_roi':<8} | {'IoU_roi':<8} | {'K_roi':<5} | {'diff_S6':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in csv_rows:
        print(f"{r['checkpoint']:<14} | {r['config_name']:<28} | {r['ari_within_roi']:<8.4f} | {r['mean_iou_within_roi']:<8.4f} | {r['k_pred_roi']:<5d} | {r['delta_vs_s6']:<+8.4f}", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
