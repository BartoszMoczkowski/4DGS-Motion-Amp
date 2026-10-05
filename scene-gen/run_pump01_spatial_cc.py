#!/usr/bin/env python3
"""run_pump01_spatial_cc.py — Spatial Connected Components and Appearance Clustering on pump01.

Evaluates spatial CC (canonical xyz) and appearance-augmented CC on the 3 pump01 grid checkpoints
(A20mm_M2, A20mm_M4, A40mm_M8) with no motion/trajectory information, comparing against the
entire pump01 motion campaign.

Performs:
1. Calibrated radius sweep on canonical xyz with and without opacity filtering (alpha > 0.1).
2. Evaluation of ARI within dynamic ROI, global ARI, Mean IoU, K_pred vs K_gt=107.
3. Diagnostic decomposition of merged GT part pairs:
   (a) shared Gaussians / joint boundary blending (physical contact in CAD)
   (b) distinct but adjacent clusters (clearance gap bridged by radius/Gaussians)
4. Appearance augmentation:
   - Color-gated spatial CC (spatial distance <= r AND color distance <= theta_c)
   - Weighted [xyz, w_c * RGB] spatial-color CC
   - Appearance pre-clustering (E0d-style) + Spatial CC
5. Output results to runs/pump01_spatial_cc_results.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scene-gen"))

from pipeline.vendored.host.metrics import adjusted_rand_index


def fast_best_iou_matching(labels_true: np.ndarray, labels_pred: np.ndarray):
    """Hungarian-match predicted clusters to GT classes maximizing IoU via contingency table."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = len(classes)
    n_clusters = len(clusters)

    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    np.add.at(contingency, (class_idx, cluster_idx), 1)

    class_sizes = contingency.sum(axis=1, keepdims=True)
    cluster_sizes = contingency.sum(axis=0, keepdims=True)

    unions = class_sizes + cluster_sizes - contingency
    ious = np.zeros_like(contingency, dtype=np.float64)
    valid = unions > 0
    ious[valid] = contingency[valid] / unions[valid]

    row, col = linear_sum_assignment(-ious)
    matches = [
        (classes[a], clusters[b], float(ious[a, b]), int(class_sizes[a, 0]), int(cluster_sizes[0, b]))
        for a, b in zip(row, col)
    ]
    matches.sort(key=lambda m: -m[3])
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return mean_iou, matches


def read_point_cloud_ply(ply_path: Path) -> dict[str, np.ndarray]:
    """Read vertex properties from 4DGS point_cloud.ply binary file."""
    with open(ply_path, "rb") as f:
        header = b""
        while b"end_header\n" not in header:
            header += f.readline()
        header_len = f.tell()

    props = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    for i in range(45):
        props.append((f"f_rest_{i}", "<f4"))
    props.extend([
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    dt = np.dtype(props)
    with open(ply_path, "rb") as f:
        f.seek(header_len)
        data = np.fromfile(f, dtype=dt)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
    # SH DC to RGB: RGB = 0.5 + C0 * f_dc
    rgb = np.clip(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0)
    
    raw_opacity = data["opacity"]
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -20.0, 20.0)))

    return {
        "xyz": xyz,
        "f_dc": f_dc,
        "rgb": rgb,
        "opacity": sigmoid_opacity,
    }


def spatial_cc(
    xyz: np.ndarray,
    radius: float,
    min_cluster_size: int = 15,
) -> tuple[np.ndarray, int]:
    """Run spatial connected components with noise point reassignment."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0

    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) > 0:
        data = np.ones(len(pairs), dtype=bool)
        adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n, n), dtype=bool)

    n_comp, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_components = unique[counts >= min_cluster_size]

    if len(valid_components) == 0:
        return np.zeros(n, dtype=np.int64), 1

    comp_map = {comp: idx for idx, comp in enumerate(valid_components)}
    labels = np.full(n, -1, dtype=np.int64)
    for comp, new_idx in comp_map.items():
        labels[raw_labels == comp] = new_idx

    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        assigned = np.where(labels >= 0)[0]
        sub_tree = cKDTree(xyz[assigned])
        _, nearest_idx = sub_tree.query(xyz[unassigned], k=1)
        labels[unassigned] = labels[assigned[nearest_idx]]

    return labels, len(valid_components)


def color_gated_spatial_cc(
    xyz: np.ndarray,
    rgb: np.ndarray,
    radius: float,
    color_thresh: float,
    min_cluster_size: int = 15,
) -> tuple[np.ndarray, int]:
    """Run spatial CC where edges are only connected if spatial dist <= radius AND color dist <= color_thresh."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0

    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) > 0:
        c_diff = rgb[pairs[:, 0]] - rgb[pairs[:, 1]]
        c_dist = np.linalg.norm(c_diff, axis=1)
        valid_edge = c_dist <= color_thresh
        pairs = pairs[valid_edge]

    if len(pairs) > 0:
        data = np.ones(len(pairs), dtype=bool)
        adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n, n), dtype=bool)

    n_comp, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_components = unique[counts >= min_cluster_size]

    if len(valid_components) == 0:
        return np.zeros(n, dtype=np.int64), 1

    comp_map = {comp: idx for idx, comp in enumerate(valid_components)}
    labels = np.full(n, -1, dtype=np.int64)
    for comp, new_idx in comp_map.items():
        labels[raw_labels == comp] = new_idx

    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        assigned = np.where(labels >= 0)[0]
        sub_tree = cKDTree(xyz[assigned])
        _, nearest_idx = sub_tree.query(xyz[unassigned], k=1)
        labels[unassigned] = labels[assigned[nearest_idx]]

    return labels, len(valid_components)


def appearance_precluster_spatial_cc(
    xyz: np.ndarray,
    rgb: np.ndarray,
    n_color_clusters: int,
    radius: float,
    min_cluster_size: int = 15,
    rng_seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Cluster by appearance first (K-means on RGB), then run spatial CC within each color cluster."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0

    from pipeline.vendored.host.kabsch_em import _kmeans_plus_plus, _lloyd_kmeans
    rng = np.random.default_rng(rng_seed)
    k_color = min(n_color_clusters, n)
    centers = _kmeans_plus_plus(rgb, k_color, rng)
    color_labels, _ = _lloyd_kmeans(rgb, centers)

    final_labels = np.full(n, -1, dtype=np.int64)
    cluster_offset = 0

    for c in range(k_color):
        mask = (color_labels == c)
        if mask.sum() == 0:
            continue
        c_xyz = xyz[mask]
        c_labels, n_c = spatial_cc(c_xyz, radius=radius, min_cluster_size=min_cluster_size)
        final_labels[mask] = c_labels + cluster_offset
        cluster_offset += n_c

    # Re-index labels contiguously
    uniq = np.unique(final_labels)
    remap = {old: new for new, old in enumerate(uniq)}
    remapped = np.array([remap[l] for l in final_labels], dtype=np.int64)
    return remapped, len(uniq)


def evaluate_segmentation(
    pred_labels: np.ndarray,
    gt_labels_on_points: np.ndarray,
    roi_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute ARI (global + within ROI) and Mean IoU."""
    ari_global = float(adjusted_rand_index(gt_labels_on_points, pred_labels))
    mean_iou, _ = fast_best_iou_matching(gt_labels_on_points, pred_labels)

    result = {
        "ari_global": ari_global,
        "mean_iou": float(mean_iou),
        "k_pred": int(len(np.unique(pred_labels))),
    }

    if roi_mask is not None:
        in_roi = np.asarray(roi_mask, dtype=bool)
        if in_roi.any():
            result["ari_within_roi"] = float(
                adjusted_rand_index(gt_labels_on_points[in_roi], pred_labels[in_roi])
            )
            roi_iou, _ = fast_best_iou_matching(gt_labels_on_points[in_roi], pred_labels[in_roi])
            result["mean_iou_within_roi"] = float(roi_iou)
            result["k_pred_within_roi"] = int(len(np.unique(pred_labels[in_roi])))
        else:
            result["ari_within_roi"] = 0.0
            result["mean_iou_within_roi"] = 0.0
            result["k_pred_within_roi"] = 0

    return result


def analyze_merged_pairs(
    xyz: np.ndarray,
    gt_on_pred: np.ndarray,
    pred_labels: np.ndarray,
    gt_pts: np.ndarray,
    gt_labs: np.ndarray,
    radius: float,
) -> dict[str, Any]:
    """Perform diagnostic decomposition of merged GT parts.
    
    Categorizes merged dynamic GT part pairs (p_i, p_j) into:
    (a) Irreducible Joint Boundary Blending / Contact: touch at physical joint in CAD (d_CAD < 2mm) & d_4DGS <= radius
    (b) Spatial Clearance Bridging: clearance gap in CAD (d_CAD >= 2mm) bridged by radius or floater Gaussians.
    """
    dynamic_gt_ids = np.unique(gt_on_pred[gt_on_pred > 0])
    n_dynamic = len(dynamic_gt_ids)

    # Build per-part GT point clouds and trees for fast CAD distance query
    gt_part_pts = {pid: gt_pts[gt_labs == pid] for pid in dynamic_gt_ids}
    gt_part_trees = {pid: cKDTree(pts) for pid, pts in gt_part_pts.items() if len(pts) > 0}

    # Build per-part 4DGS Gaussian trees
    pred_part_pts = {pid: xyz[gt_on_pred == pid] for pid in dynamic_gt_ids}
    pred_part_trees = {pid: cKDTree(pts) for pid, pts in pred_part_pts.items() if len(pts) > 0}

    # Which cluster contains each part (majority vote)
    part_to_cluster = {}
    for pid in dynamic_gt_ids:
        p_mask = (gt_on_pred == pid)
        if p_mask.sum() > 0:
            clusters, counts = np.unique(pred_labels[p_mask], return_counts=True)
            part_to_cluster[pid] = clusters[np.argmax(counts)]

    total_possible_pairs = n_dynamic * (n_dynamic - 1) // 2
    merged_pairs = []
    separated_pairs = []

    for i in range(len(dynamic_gt_ids)):
        p_i = dynamic_gt_ids[i]
        c_i = part_to_cluster.get(p_i)
        for j in range(i + 1, len(dynamic_gt_ids)):
            p_j = dynamic_gt_ids[j]
            c_j = part_to_cluster.get(p_j)
            if c_i is not None and c_j is not None and c_i == c_j:
                merged_pairs.append((p_i, p_j, c_i))
            else:
                separated_pairs.append((p_i, p_j))

    # For each merged pair, compute min CAD distance and min 4DGS distance
    cat_a_joint_blending = []  # d_CAD < 2mm
    cat_b_clearance_bridged = []  # d_CAD >= 2mm

    cad_distances = []
    dgs_distances = []
    direct_touch_edges = 0

    for p_i, p_j, cluster_id in merged_pairs:
        # Distance in CAD
        if p_i in gt_part_trees and p_j in gt_part_pts and len(gt_part_pts[p_j]) > 0:
            dists, _ = gt_part_trees[p_i].query(gt_part_pts[p_j], k=1)
            min_cad_d = float(dists.min())
        else:
            min_cad_d = float("inf")

        # Distance in 4DGS
        if p_i in pred_part_trees and p_j in pred_part_pts and len(pred_part_pts[p_j]) > 0:
            dists_dgs, _ = pred_part_trees[p_i].query(pred_part_pts[p_j], k=1)
            min_dgs_d = float(dists_dgs.min())
        else:
            min_dgs_d = float("inf")

        cad_distances.append(min_cad_d)
        dgs_distances.append(min_dgs_d)
        if min_dgs_d <= radius:
            direct_touch_edges += 1

        pair_info = {
            "part_i": int(p_i),
            "part_j": int(p_j),
            "cluster": int(cluster_id),
            "cad_dist_mm": min_cad_d * 1000.0,
            "dgs_dist_mm": min_dgs_d * 1000.0,
            "direct_touch": bool(min_dgs_d <= radius),
        }

        if min_cad_d < 0.002:  # < 2 mm = mechanical joint contact
            cat_a_joint_blending.append(pair_info)
        else:
            cat_b_clearance_bridged.append(pair_info)

    n_merged = len(merged_pairs)
    frac_cat_a = len(cat_a_joint_blending) / max(1, n_merged)
    frac_cat_b = len(cat_b_clearance_bridged) / max(1, n_merged)

    return {
        "n_dynamic_parts": n_dynamic,
        "total_possible_pairs": total_possible_pairs,
        "n_merged_pairs": n_merged,
        "n_separated_pairs": len(separated_pairs),
        "n_cat_a_joint_contact": len(cat_a_joint_blending),
        "n_cat_b_clearance_bridged": len(cat_b_clearance_bridged),
        "frac_cat_a": frac_cat_a,
        "frac_cat_b": frac_cat_b,
        "direct_touch_edges": direct_touch_edges,
        "mean_cad_dist_merged_mm": float(np.mean(cad_distances)) * 1000.0 if cad_distances else 0.0,
        "median_cad_dist_merged_mm": float(np.median(cad_distances)) * 1000.0 if cad_distances else 0.0,
        "mean_dgs_dist_merged_mm": float(np.mean(dgs_distances)) * 1000.0 if dgs_distances else 0.0,
        "cat_a_sample": cat_a_joint_blending[:10],
        "cat_b_sample": cat_b_clearance_bridged[:10],
    }


def main():
    checkpoints = [
        ("grid-A20mm_M2", "A20mm_M2"),
        ("grid-A20mm_M4", "A20mm_M4"),
        ("grid-A40mm_M8", "A40mm_M8"),
    ]

    out_csv = REPO_ROOT / "runs" / "pump01_spatial_cc_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    diagnostic_reports = {}

    radii_m = [0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]

    for dir_name, exp_name in checkpoints:
        run_dir = REPO_ROOT / "runs" / dir_name
        print(f"\n==================================================", flush=True)
        print(f"Processing Checkpoint: {exp_name} ({run_dir})", flush=True)
        print(f"==================================================", flush=True)

        traj_file = run_dir / "trajectories.npz"
        traj_data = np.load(traj_file)
        canonical_xyz = traj_data["canonical_xyz"]
        traj_opacity = traj_data["opacity"]

        # Find GT file
        gt_files = list(run_dir.glob("**/gt_segmentation.npz"))
        if not gt_files:
            print(f"ERROR: No gt_segmentation.npz found for {run_dir}", flush=True)
            continue
        gt_file = gt_files[0]
        gt_data = np.load(gt_file)
        gt_pts = gt_data["points"]
        gt_labs = gt_data["labels"]

        # Load PLY colors
        ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
        ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
        ply_file = ply_files[-1]
        print(f"Loading PLY: {ply_file.relative_to(run_dir)}", flush=True)
        ply_dict = read_point_cloud_ply(ply_file)
        rgb = ply_dict["rgb"]
        ply_opacity = ply_dict["opacity"]

        # NN label transfer
        print(f"Building cKDTree on GT points ({len(gt_pts)} pts)...", flush=True)
        tree = cKDTree(gt_pts)
        dists, nn = tree.query(canonical_xyz, k=1)
        gt_on_pred = gt_labs[nn]

        # Oracle dynamic ROI: gt_on_pred > 0 (label 0 is casing/background)
        oracle_roi = (gt_on_pred > 0)
        n_roi = int(oracle_roi.sum())
        print(f"Total Gaussians: {len(canonical_xyz)}, Oracle Dynamic ROI points: {n_roi} ({n_roi/len(canonical_xyz)*100:.1f}%)", flush=True)

        # Sweep 1: Pure Spatial Connected Components (Radius sweep x Opacity filter)
        for op_filter in [False, True]:
            op_str = "opacity_gt_0.1" if op_filter else "raw_all"
            if op_filter:
                active_mask = (traj_opacity > 0.1) & oracle_roi
            else:
                active_mask = oracle_roi

            sub_xyz = canonical_xyz[active_mask]
            sub_gt = gt_on_pred[active_mask]
            sub_rgb = rgb[active_mask]
            sub_n = len(sub_xyz)

            for r in radii_m:
                r_cm = r * 100.0
                method_name = f"spatial_cc_r{r_cm:04.1f}cm_{op_str}"
                
                labels_in_sub, k_pred_sub = spatial_cc(sub_xyz, radius=r, min_cluster_size=15)
                
                # Build full-cloud labels for global ARI evaluation
                full_pred_labels = np.full(len(canonical_xyz), -1, dtype=np.int64)
                full_pred_labels[active_mask] = labels_in_sub
                full_pred_labels[~oracle_roi] = 0

                eval_res = evaluate_segmentation(
                    full_pred_labels,
                    gt_on_pred,
                    roi_mask=oracle_roi,
                )

                row = {
                    "checkpoint": exp_name,
                    "method": method_name,
                    "modality": "spatial_xyz",
                    "radius_cm": r_cm,
                    "opacity_filter": op_filter,
                    "color_weight": 0.0,
                    "color_thresh": 0.0,
                    "k_pred_roi": eval_res["k_pred_within_roi"],
                    "k_pred_total": eval_res["k_pred"],
                    "k_gt_roi": len(np.unique(gt_on_pred[oracle_roi])),
                    "k_gt_total": len(np.unique(gt_on_pred)),
                    "n_roi_points": n_roi,
                    "n_active_points": sub_n,
                    "ari_within_roi": eval_res["ari_within_roi"],
                    "mean_iou_within_roi": eval_res["mean_iou_within_roi"],
                    "ari_global": eval_res["ari_global"],
                    "mean_iou_global": eval_res["mean_iou"],
                }
                csv_rows.append(row)
                print(f"[{exp_name}] {method_name:32s} -> ARI_within_ROI: {eval_res['ari_within_roi']:7.4f}, IoU_ROI: {eval_res['mean_iou_within_roi']:7.4f}, K_ROI: {eval_res['k_pred_within_roi']:3d}, Global_ARI: {eval_res['ari_global']:7.4f}", flush=True)

        # Sweep 2: Appearance Augmentation A — Color-gated Spatial CC
        # (Connect edge if spatial dist <= r AND color dist <= theta_c)
        for r in [0.005, 0.010, 0.020]:
            r_cm = r * 100.0
            for c_th in [0.05, 0.10, 0.15, 0.20, 0.30]:
                method_name = f"color_gated_cc_r{r_cm:04.1f}cm_cth{c_th:.2f}"
                active_mask = oracle_roi
                sub_xyz = canonical_xyz[active_mask]
                sub_rgb = rgb[active_mask]

                labels_in_sub, k_pred_sub = color_gated_spatial_cc(
                    sub_xyz, sub_rgb, radius=r, color_thresh=c_th, min_cluster_size=15
                )

                full_pred_labels = np.full(len(canonical_xyz), -1, dtype=np.int64)
                full_pred_labels[active_mask] = labels_in_sub
                full_pred_labels[~oracle_roi] = 0

                eval_res = evaluate_segmentation(
                    full_pred_labels,
                    gt_on_pred,
                    roi_mask=oracle_roi,
                )

                row = {
                    "checkpoint": exp_name,
                    "method": method_name,
                    "modality": "color_gated_spatial",
                    "radius_cm": r_cm,
                    "opacity_filter": False,
                    "color_weight": 0.0,
                    "color_thresh": c_th,
                    "k_pred_roi": eval_res["k_pred_within_roi"],
                    "k_pred_total": eval_res["k_pred"],
                    "k_gt_roi": len(np.unique(gt_on_pred[oracle_roi])),
                    "k_gt_total": len(np.unique(gt_on_pred)),
                    "n_roi_points": n_roi,
                    "n_active_points": len(sub_xyz),
                    "ari_within_roi": eval_res["ari_within_roi"],
                    "mean_iou_within_roi": eval_res["mean_iou_within_roi"],
                    "ari_global": eval_res["ari_global"],
                    "mean_iou_global": eval_res["mean_iou"],
                }
                csv_rows.append(row)
                print(f"[{exp_name}] {method_name:32s} -> ARI_within_ROI: {eval_res['ari_within_roi']:7.4f}, IoU_ROI: {eval_res['mean_iou_within_roi']:7.4f}, K_ROI: {eval_res['k_pred_within_roi']:3d}", flush=True)

        # Sweep 3: Appearance Augmentation B — E0d-style Appearance Pre-cluster + Spatial CC
        for n_colors in [5, 10, 20, 40]:
            for r in [0.005, 0.010, 0.020]:
                r_cm = r * 100.0
                method_name = f"e0d_precluster_k{n_colors}_r{r_cm:04.1f}cm"
                active_mask = oracle_roi
                sub_xyz = canonical_xyz[active_mask]
                sub_rgb = rgb[active_mask]

                labels_in_sub, k_pred_sub = appearance_precluster_spatial_cc(
                    sub_xyz, sub_rgb, n_color_clusters=n_colors, radius=r, min_cluster_size=15
                )

                full_pred_labels = np.full(len(canonical_xyz), -1, dtype=np.int64)
                full_pred_labels[active_mask] = labels_in_sub
                full_pred_labels[~oracle_roi] = 0

                eval_res = evaluate_segmentation(
                    full_pred_labels,
                    gt_on_pred,
                    roi_mask=oracle_roi,
                )

                row = {
                    "checkpoint": exp_name,
                    "method": method_name,
                    "modality": "appearance_precluster_spatial",
                    "radius_cm": r_cm,
                    "opacity_filter": False,
                    "color_weight": 0.0,
                    "color_thresh": 0.0,
                    "k_pred_roi": eval_res["k_pred_within_roi"],
                    "k_pred_total": eval_res["k_pred"],
                    "k_gt_roi": len(np.unique(gt_on_pred[oracle_roi])),
                    "k_gt_total": len(np.unique(gt_on_pred)),
                    "n_roi_points": n_roi,
                    "n_active_points": len(sub_xyz),
                    "ari_within_roi": eval_res["ari_within_roi"],
                    "mean_iou_within_roi": eval_res["mean_iou_within_roi"],
                    "ari_global": eval_res["ari_global"],
                    "mean_iou_global": eval_res["mean_iou"],
                }
                csv_rows.append(row)
                print(f"[{exp_name}] {method_name:32s} -> ARI_within_ROI: {eval_res['ari_within_roi']:7.4f}, IoU_ROI: {eval_res['mean_iou_within_roi']:7.4f}, K_ROI: {eval_res['k_pred_within_roi']:3d}", flush=True)

        # Step 3: Diagnostic Decomposition of Merged Pairs on best pure spatial CC (r = 0.5 cm)
        print(f"\nRunning Diagnostic Decomposition for {exp_name} at r=0.5cm...", flush=True)
        active_mask = oracle_roi
        sub_xyz = canonical_xyz[active_mask]
        sub_gt = gt_on_pred[active_mask]
        pred_labels_05cm, _ = spatial_cc(sub_xyz, radius=0.005, min_cluster_size=15)
        decomp_05cm = analyze_merged_pairs(
            xyz=sub_xyz,
            gt_on_pred=sub_gt,
            pred_labels=pred_labels_05cm,
            gt_pts=gt_pts,
            gt_labs=gt_labs,
            radius=0.005,
        )
        diagnostic_reports[f"{exp_name}_r0.5cm"] = decomp_05cm

        print(f"Running Diagnostic Decomposition for {exp_name} at r=1.0cm...", flush=True)
        pred_labels_10cm, _ = spatial_cc(sub_xyz, radius=0.010, min_cluster_size=15)
        decomp_10cm = analyze_merged_pairs(
            xyz=sub_xyz,
            gt_on_pred=sub_gt,
            pred_labels=pred_labels_10cm,
            gt_pts=gt_pts,
            gt_labs=gt_labs,
            radius=0.010,
        )
        diagnostic_reports[f"{exp_name}_r1.0cm"] = decomp_10cm

    # Write CSV
    print(f"\nWriting results to {out_csv} ({len(csv_rows)} rows)...", flush=True)
    fieldnames = list(csv_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write JSON diagnostic decomposition
    out_decomp_json = REPO_ROOT / "runs" / "pump01_spatial_cc_diagnostic_decomposition.json"
    print(f"Writing diagnostic decomposition to {out_decomp_json}...", flush=True)
    with open(out_decomp_json, "w", encoding="utf-8") as f:
        json.dump(diagnostic_reports, f, indent=2)

    print("\nAll experiments and diagnostics completed successfully!", flush=True)


if __name__ == "__main__":
    main()
