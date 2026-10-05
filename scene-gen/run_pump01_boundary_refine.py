#!/usr/bin/env python3
"""run_pump01_boundary_refine.py — Local Motion Boundary Refinement on pump01.

Takes the S6 spatial CC / color-gated CC backbone and applies motion trajectories
locally at candidate merged part-pair interfaces (the Category A/B merges) to split them.

Evaluates:
1. Local pairwise separability AUROC at boundary interfaces (boundary Gaussians within r).
2. Local trajectory discrepancy (band-passed FFT phasor features at drive freq f0=10 cycles/clip vs raw).
3. Supervised / Diagnostic boundary split (Category A vs Category B splits, metric lift).
4. Unsupervised intra-cluster motion bisection (spectral / k-means split on composite spatial clusters).
5. Emits runs/pump01_boundary_refine_results.csv and runs/pump01_boundary_refine_report.json.
"""

from __future__ import annotations

import csv
import json
import os
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


def extract_motion_features(traj: np.ndarray, f0: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Extract complex phasor features at drive frequency f0 and normalized trajectory curves.
    
    traj: (N, 60, 3)
    Returns:
      phasors_6d: (N, 6) real & imag parts of 3D phasor at f0
      norm_traj: (N, 180) zero-mean unit-variance normalized trajectory
    """
    N, T, _ = traj.shape
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * f0 * t / T) # (60,)
    
    # Zero-mean per Gaussian across time
    traj_centered = traj - traj.mean(axis=1, keepdims=True)
    
    # FFT coefficient at f0: sum_t traj_c(t) * e^(-j 2pi f0 t / T)
    phasors = np.tensordot(traj_centered, basis, axes=([1], [0])) # (N, 3) complex
    phasors_6d = np.concatenate([phasors.real, phasors.imag], axis=1).astype(np.float32) # (N, 6)
    
    # Normalized full trajectory vectors (N, 180)
    traj_flat = traj_centered.reshape(N, -1)
    norms = np.linalg.norm(traj_flat, axis=1, keepdims=True)
    norm_traj = np.zeros_like(traj_flat)
    valid = norms[:, 0] > 1e-9
    norm_traj[valid] = traj_flat[valid] / norms[valid]
    
    return phasors_6d, norm_traj


def compute_pairwise_auroc(feat_a: np.ndarray, feat_b: np.ndarray, max_pairs: int = 5000) -> float:
    """Compute local separability AUROC between two groups of features.
    
    Positive class = cross-group edge (feat_a vs feat_b distance).
    Negative class = same-group edge (feat_a vs feat_a or feat_b vs feat_b distance).
    """
    na = len(feat_a)
    nb = len(feat_b)
    if na < 2 or nb < 2:
        return 0.5

    rng = np.random.default_rng(42)
    
    # Sample within-group distances
    n_same = min(max_pairs, (na * (na - 1) // 2) + (nb * (nb - 1) // 2))
    same_dists = []
    
    # A-A pairs
    if na > 1:
        idx_a1 = rng.integers(0, na, size=n_same // 2)
        idx_a2 = rng.integers(0, na, size=n_same // 2)
        diff = feat_a[idx_a1] - feat_a[idx_a2]
        d = np.linalg.norm(diff, axis=1)
        same_dists.extend(d[idx_a1 != idx_a2])
        
    # B-B pairs
    if nb > 1:
        idx_b1 = rng.integers(0, nb, size=n_same // 2)
        idx_b2 = rng.integers(0, nb, size=n_same // 2)
        diff = feat_b[idx_b1] - feat_b[idx_b2]
        d = np.linalg.norm(diff, axis=1)
        same_dists.extend(d[idx_b1 != idx_b2])
        
    same_dists = np.array(same_dists)
    if len(same_dists) == 0:
        return 0.5
        
    # Sample cross-group distances
    n_cross = min(max_pairs, na * nb)
    idx_a = rng.integers(0, na, size=n_cross)
    idx_b = rng.integers(0, nb, size=n_cross)
    cross_dists = np.linalg.norm(feat_a[idx_a] - feat_b[idx_b], axis=1)
    
    # Exact vector comparison for sampled pairs:
    n_eval = min(1000, len(same_dists))
    same_sub = same_dists[:n_eval]
    cross_sub = cross_dists[:min(1000, len(cross_dists))]
    
    gt_matrix = cross_sub[:, None] > same_sub[None, :]
    eq_matrix = cross_sub[:, None] == same_sub[None, :]
    auroc = float(np.mean(gt_matrix + 0.5 * eq_matrix))
    return auroc


def two_sample_motion_test(feat_a: np.ndarray, feat_b: np.ndarray) -> tuple[float, float, float]:
    """Perform two-sample multivariate discrepancy test.
    
    Returns:
      discrepancy_snr (d-prime): ||mu_a - mu_b|| / sqrt(0.5*(var_a + var_b))
      t_stat: Welch-like t-statistic on feature magnitudes
      p_val: approximate p-value
    """
    na = len(feat_a)
    nb = len(feat_b)
    if na < 2 or nb < 2:
        return 0.0, 0.0, 1.0

    mu_a = np.mean(feat_a, axis=0)
    mu_b = np.mean(feat_b, axis=0)
    
    var_a = np.mean(np.sum((feat_a - mu_a) ** 2, axis=1))
    var_b = np.mean(np.sum((feat_b - mu_b) ** 2, axis=1))
    
    delta_mu = float(np.linalg.norm(mu_a - mu_b))
    pooled_sd = np.sqrt(0.5 * (var_a + var_b) + 1e-12)
    snr = delta_mu / pooled_sd
    
    # Welch t-test on projected delta vector
    proj_dir = (mu_a - mu_b) / (delta_mu + 1e-12)
    proj_a = feat_a @ proj_dir
    proj_b = feat_b @ proj_dir
    
    from scipy.stats import ttest_ind
    t_res = ttest_ind(proj_a, proj_b, equal_var=False)
    
    return snr, float(t_res.statistic), float(t_res.pvalue)


def process_checkpoint(
    dir_name: str,
    exp_name: str,
    radii_m: list[float] = [0.005, 0.010],
    f0: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute local motion boundary refinement on a given checkpoint."""
    run_dir = REPO_ROOT / "runs" / dir_name
    print(f"\n========================================================", flush=True)
    print(f"Executing Boundary Refinement: {exp_name} ({run_dir})", flush=True)
    print(f"========================================================", flush=True)

    traj_file = run_dir / "trajectories.npz"
    traj_data = np.load(traj_file)
    canonical_xyz = traj_data["canonical_xyz"]
    traj = traj_data["traj"] # (N, 60, 3)
    traj_opacity = traj_data["opacity"]

    # GT
    gt_files = list(run_dir.glob("**/gt_segmentation.npz"))
    if not gt_files:
        gt_files = [REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_segmentation.npz"]
    gt_file = gt_files[0]
    gt_data = np.load(gt_file)
    gt_pts = gt_data["points"]
    gt_labs = gt_data["labels"]

    print(f"Building cKDTree on GT points ({len(gt_pts)} pts)...", flush=True)
    gt_tree = cKDTree(gt_pts)
    _, nn = gt_tree.query(canonical_xyz, k=1)
    gt_on_pred = gt_labs[nn]

    oracle_roi = (gt_on_pred > 0)
    n_roi = int(oracle_roi.sum())
    print(f"Total points: {len(canonical_xyz)}, Oracle ROI points: {n_roi}", flush=True)

    # Extract motion features
    print(f"Extracting phasor features at drive frequency f0={f0} cycles/clip...", flush=True)
    phasors_6d, norm_traj = extract_motion_features(traj, f0=f0)

    # Sub-select ROI points
    roi_xyz = canonical_xyz[oracle_roi]
    roi_gt = gt_on_pred[oracle_roi]
    roi_phasors = phasors_6d[oracle_roi]
    roi_norm_traj = norm_traj[oracle_roi]

    # Pre-build per-part GT point trees for CAD distance calculation
    dynamic_gt_ids = np.unique(roi_gt[roi_gt > 0])
    gt_part_pts = {pid: gt_pts[gt_labs == pid] for pid in dynamic_gt_ids}
    gt_part_trees = {pid: cKDTree(pts) for pid, pts in gt_part_pts.items() if len(pts) > 0}

    results_rows = []
    diagnostic_report = {
        "checkpoint": exp_name,
        "n_dynamic_parts": len(dynamic_gt_ids),
        "interfaces": {},
    }

    # Run for backbone radius r=0.005 (0.5cm) and r=0.010 (1.0cm)
    for r in radii_m:
        r_cm = r * 100.0
        print(f"\n--- Backbone: Pure Spatial CC (r = {r_cm:.1f} cm) ---", flush=True)
        bb_labels_roi, k_bb = spatial_cc(roi_xyz, radius=r, min_cluster_size=15)
        
        # Base evaluation
        full_pred = np.full(len(canonical_xyz), -1, dtype=np.int64)
        full_pred[oracle_roi] = bb_labels_roi
        full_pred[~oracle_roi] = 0
        
        base_eval = evaluate_segmentation(full_pred, gt_on_pred, roi_mask=oracle_roi)
        print(f"Backbone baseline -> ARI_within_ROI: {base_eval['ari_within_roi']:.4f}, IoU: {base_eval['mean_iou_within_roi']:.4f}, K_roi: {base_eval['k_pred_within_roi']}", flush=True)

        results_rows.append({
            "checkpoint": exp_name,
            "stage": "backbone_baseline",
            "backbone_radius_cm": r_cm,
            "refine_method": "none",
            "feature_type": "none",
            "auroc_threshold": 0.0,
            "k_pred_roi": base_eval["k_pred_within_roi"],
            "ari_within_roi": base_eval["ari_within_roi"],
            "mean_iou_within_roi": base_eval["mean_iou_within_roi"],
            "ari_global": base_eval["ari_global"],
            "mean_iou_global": base_eval["mean_iou"],
            "n_merged_pairs_initial": 0,
            "n_pairs_split_cat_a": 0,
            "n_pairs_split_cat_b": 0,
            "n_pairs_split_total": 0,
            "mean_local_auroc_cat_a": 0.0,
            "mean_local_auroc_cat_b": 0.0,
        })

        # Identify candidate merged interfaces in this backbone
        # Map each GT part to its dominant cluster
        part_to_cluster = {}
        for pid in dynamic_gt_ids:
            p_mask = (roi_gt == pid)
            if p_mask.sum() > 0:
                clusters, counts = np.unique(bb_labels_roi[p_mask], return_counts=True)
                part_to_cluster[pid] = clusters[np.argmax(counts)]

        merged_pairs = []
        for i in range(len(dynamic_gt_ids)):
            p_i = dynamic_gt_ids[i]
            c_i = part_to_cluster.get(p_i)
            for j in range(i + 1, len(dynamic_gt_ids)):
                p_j = dynamic_gt_ids[j]
                c_j = part_to_cluster.get(p_j)
                if c_i is not None and c_j is not None and c_i == c_j:
                    merged_pairs.append((p_i, p_j, c_i))

        print(f"Found {len(merged_pairs)} merged part pairs in {k_bb} clusters at r={r_cm:.1f}cm.", flush=True)

        # Analyze local motion discrepancy for all merged pairs
        interface_details = []
        cat_a_aurocs_phasor = []
        cat_b_aurocs_phasor = []
        cat_a_aurocs_raw = []
        cat_b_aurocs_raw = []

        # Trees for per-part Gaussians in ROI
        roi_part_indices = {pid: np.where(roi_gt == pid)[0] for pid in dynamic_gt_ids}
        roi_part_trees = {pid: cKDTree(roi_xyz[idxs]) for pid, idxs in roi_part_indices.items() if len(idxs) > 0}

        for p_i, p_j, cluster_id in merged_pairs:
            # CAD distance
            if p_i in gt_part_trees and p_j in gt_part_pts and len(gt_part_pts[p_j]) > 0:
                cad_d, _ = gt_part_trees[p_i].query(gt_part_pts[p_j], k=1)
                min_cad_d = float(cad_d.min())
            else:
                min_cad_d = float("inf")

            is_cat_a = (min_cad_d < 0.002) # < 2mm CAD joint contact
            cat_name = "Cat_A (Joint Contact)" if is_cat_a else "Cat_B (Clearance Gap)"

            # Boundary Gaussians in 4DGS: points in p_i within r_boundary of p_j
            idx_i = roi_part_indices[p_i]
            idx_j = roi_part_indices[p_j]
            
            tree_i = roi_part_trees[p_i]
            tree_j = roi_part_trees[p_j]
            
            d_ij, _ = tree_i.query(roi_xyz[idx_j], k=1)
            d_ji, _ = tree_j.query(roi_xyz[idx_i], k=1)
            
            min_dgs_d = float(min(d_ij.min(), d_ji.min())) if len(d_ij) > 0 and len(d_ji) > 0 else float("inf")
            
            # Boundary zone: within r of opposite part (or up to 1.5*r)
            r_bound = max(r * 1.5, 0.010)
            bound_i = idx_i[d_ji <= r_bound]
            bound_j = idx_j[d_ij <= r_bound]

            # If boundary subset is too small, use all part points
            pts_i_eval = bound_i if len(bound_i) >= 5 else idx_i
            pts_j_eval = bound_j if len(bound_j) >= 5 else idx_j

            # Local AUROC - Phasor (f0=10)
            auroc_phasor = compute_pairwise_auroc(roi_phasors[pts_i_eval], roi_phasors[pts_j_eval])
            # Local AUROC - Raw Trajectory
            auroc_raw = compute_pairwise_auroc(roi_norm_traj[pts_i_eval], roi_norm_traj[pts_j_eval])

            # Two-sample test on phasors
            snr_phasor, t_stat, p_val = two_sample_motion_test(roi_phasors[pts_i_eval], roi_phasors[pts_j_eval])
            # Two-sample test on raw
            snr_raw, t_stat_raw, p_val_raw = two_sample_motion_test(roi_norm_traj[pts_i_eval], roi_norm_traj[pts_j_eval])

            if is_cat_a:
                cat_a_aurocs_phasor.append(auroc_phasor)
                cat_a_aurocs_raw.append(auroc_raw)
            else:
                cat_b_aurocs_phasor.append(auroc_phasor)
                cat_b_aurocs_raw.append(auroc_raw)

            split_verdict_phasor = bool(auroc_phasor >= 0.70 or (snr_phasor > 1.0 and p_val < 0.01))
            split_verdict_raw = bool(auroc_raw >= 0.70 or (snr_raw > 1.0 and p_val_raw < 0.01))

            pair_record = {
                "part_i": int(p_i),
                "part_j": int(p_j),
                "cluster_id": int(cluster_id),
                "category": cat_name,
                "is_cat_a": bool(is_cat_a),
                "cad_dist_mm": min_cad_d * 1000.0,
                "dgs_dist_mm": min_dgs_d * 1000.0,
                "n_pts_i": int(len(idx_i)),
                "n_pts_j": int(len(idx_j)),
                "n_bound_i": int(len(bound_i)),
                "n_bound_j": int(len(bound_j)),
                "auroc_phasor_f0": float(auroc_phasor),
                "auroc_raw_traj": float(auroc_raw),
                "snr_phasor": float(snr_phasor),
                "snr_raw": float(snr_raw),
                "p_val_phasor": float(p_val),
                "p_val_raw": float(p_val_raw),
                "split_verdict_phasor": split_verdict_phasor,
                "split_verdict_raw": split_verdict_raw,
            }
            interface_details.append(pair_record)

        diagnostic_report["interfaces"][f"r{r_cm:.1f}cm"] = interface_details

        mean_auroc_a_phasor = float(np.mean(cat_a_aurocs_phasor)) if cat_a_aurocs_phasor else 0.5
        mean_auroc_b_phasor = float(np.mean(cat_b_aurocs_phasor)) if cat_b_aurocs_phasor else 0.5
        mean_auroc_a_raw = float(np.mean(cat_a_aurocs_raw)) if cat_a_aurocs_raw else 0.5
        mean_auroc_b_raw = float(np.mean(cat_b_aurocs_raw)) if cat_b_aurocs_raw else 0.5

        print(f"\n[Local Separability AUROC Summary at r={r_cm:.1f}cm]")
        print(f"  Cat A (Joint Contact <2mm) [N={len(cat_a_aurocs_phasor)}]: Phasor f0 AUROC = {mean_auroc_a_phasor:.4f}, Raw AUROC = {mean_auroc_a_raw:.4f}")
        print(f"  Cat B (Clearance Gap  >=2mm) [N={len(cat_b_aurocs_phasor)}]: Phasor f0 AUROC = {mean_auroc_b_phasor:.4f}, Raw AUROC = {mean_auroc_b_raw:.4f}")

        # Sweep Refinement Split Strategies:
        # Strategy 1: Diagnostic / Oracle-boundary split based on Phasor test (thresholds)
        for auroc_th in [0.60, 0.70, 0.80]:
            refined_roi_labels = np.copy(bb_labels_roi)
            next_cluster_id = int(refined_roi_labels.max()) + 1
            
            n_split_a = 0
            n_split_b = 0
            
            for item in interface_details:
                if item["auroc_phasor_f0"] >= auroc_th:
                    p_j = item["part_j"]
                    mask_j = (roi_gt == p_j)
                    refined_roi_labels[mask_j] = next_cluster_id
                    next_cluster_id += 1
                    if item["is_cat_a"]:
                        n_split_a += 1
                    else:
                        n_split_b += 1
                        
            full_pred_ref = np.full(len(canonical_xyz), -1, dtype=np.int64)
            full_pred_ref[oracle_roi] = refined_roi_labels
            full_pred_ref[~oracle_roi] = 0
            
            eval_ref = evaluate_segmentation(full_pred_ref, gt_on_pred, roi_mask=oracle_roi)
            n_split_tot = n_split_a + n_split_b
            print(f"  -> Phasor Refined (AUROC >= {auroc_th:.2f}) [Split {n_split_tot}/{len(merged_pairs)}: A={n_split_a}, B={n_split_b}]: ARI_within_ROI = {eval_ref['ari_within_roi']:.4f}, IoU = {eval_ref['mean_iou_within_roi']:.4f}, K_roi = {eval_ref['k_pred_within_roi']}", flush=True)

            results_rows.append({
                "checkpoint": exp_name,
                "stage": "local_boundary_refine_phasor",
                "backbone_radius_cm": r_cm,
                "refine_method": "phasor_auroc_cut",
                "feature_type": f"fft_phasor_f0_{f0}",
                "auroc_threshold": auroc_th,
                "k_pred_roi": eval_ref["k_pred_within_roi"],
                "ari_within_roi": eval_ref["ari_within_roi"],
                "mean_iou_within_roi": eval_ref["mean_iou_within_roi"],
                "ari_global": eval_ref["ari_global"],
                "mean_iou_global": eval_ref["mean_iou"],
                "n_merged_pairs_initial": len(merged_pairs),
                "n_pairs_split_cat_a": n_split_a,
                "n_pairs_split_cat_b": n_split_b,
                "n_pairs_split_total": n_split_tot,
                "mean_local_auroc_cat_a": mean_auroc_a_phasor,
                "mean_local_auroc_cat_b": mean_auroc_b_phasor,
            })

        # Strategy 2: Unsupervised Intra-Cluster Motion Bisection
        print(f"\nRunning Unsupervised Intra-Cluster Motion Bisection on spatial CC (r={r_cm:.1f}cm)...", flush=True)
        unsupervised_roi_labels = np.full(len(roi_xyz), -1, dtype=np.int64)
        next_unsupervised_id = 0
        
        unique_clusters = np.unique(bb_labels_roi)
        for c in unique_clusters:
            c_mask = (bb_labels_roi == c)
            c_pts = c_mask.sum()
            if c_pts < 30:
                unsupervised_roi_labels[c_mask] = next_unsupervised_id
                next_unsupervised_id += 1
                continue
                
            c_phasors = roi_phasors[c_mask]
            
            from pipeline.vendored.host.kabsch_em import _kmeans_plus_plus, _lloyd_kmeans
            rng = np.random.default_rng(42)
            centers = _kmeans_plus_plus(c_phasors, 2, rng)
            k_labs, _ = _lloyd_kmeans(c_phasors, centers)
            
            sub0 = (k_labs == 0)
            sub1 = (k_labs == 1)
            
            if sub0.sum() >= 15 and sub1.sum() >= 15:
                auroc_sub = compute_pairwise_auroc(c_phasors[sub0], c_phasors[sub1])
                snr_sub, _, p_sub = two_sample_motion_test(c_phasors[sub0], c_phasors[sub1])
                
                if auroc_sub >= 0.70 and p_sub < 0.05:
                    c_indices = np.where(c_mask)[0]
                    unsupervised_roi_labels[c_indices[sub0]] = next_unsupervised_id
                    next_unsupervised_id += 1
                    unsupervised_roi_labels[c_indices[sub1]] = next_unsupervised_id
                    next_unsupervised_id += 1
                    continue
                    
            unsupervised_roi_labels[c_mask] = next_unsupervised_id
            next_unsupervised_id += 1

        full_pred_unsup = np.full(len(canonical_xyz), -1, dtype=np.int64)
        full_pred_unsup[oracle_roi] = unsupervised_roi_labels
        full_pred_unsup[~oracle_roi] = 0
        
        eval_unsup = evaluate_segmentation(full_pred_unsup, gt_on_pred, roi_mask=oracle_roi)
        print(f"  -> Unsupervised Motion Bisection: ARI_within_ROI = {eval_unsup['ari_within_roi']:.4f}, IoU = {eval_unsup['mean_iou_within_roi']:.4f}, K_roi = {eval_unsup['k_pred_within_roi']}", flush=True)

        results_rows.append({
            "checkpoint": exp_name,
            "stage": "unsupervised_cluster_motion_bisection",
            "backbone_radius_cm": r_cm,
            "refine_method": "kmeans2_bisection",
            "feature_type": f"fft_phasor_f0_{f0}",
            "auroc_threshold": 0.70,
            "k_pred_roi": eval_unsup["k_pred_within_roi"],
            "ari_within_roi": eval_unsup["ari_within_roi"],
            "mean_iou_within_roi": eval_unsup["mean_iou_within_roi"],
            "ari_global": eval_unsup["ari_global"],
            "mean_iou_global": eval_unsup["mean_iou"],
            "n_merged_pairs_initial": len(merged_pairs),
            "n_pairs_split_cat_a": 0,
            "n_pairs_split_cat_b": 0,
            "n_pairs_split_total": eval_unsup["k_pred_within_roi"] - k_bb,
            "mean_local_auroc_cat_a": mean_auroc_a_phasor,
            "mean_local_auroc_cat_b": mean_auroc_b_phasor,
        })

    return results_rows, diagnostic_report


def main():
    checkpoints = [
        ("grid-A40mm_M8", "A40mm_M8"),
        ("grid-A20mm_M4", "A20mm_M4"),
        ("grid-A20mm_M2", "A20mm_M2"),
    ]

    out_csv = REPO_ROOT / "runs" / "pump01_boundary_refine_results.csv"
    out_json = REPO_ROOT / "runs" / "pump01_boundary_refine_report.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_reports = {}

    for dir_name, exp_name in checkpoints:
        rows, report = process_checkpoint(dir_name, exp_name, radii_m=[0.005, 0.010], f0=10)
        all_rows.extend(rows)
        all_reports[exp_name] = report

    # Write CSV
    print(f"\n========================================================", flush=True)
    print(f"Writing boundary refinement benchmark to {out_csv} ({len(all_rows)} rows)...", flush=True)
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write JSON report
    print(f"Writing per-interface diagnostic report to {out_json}...", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("\nAll boundary refinement experiments and evaluations completed successfully!", flush=True)


if __name__ == "__main__":
    main()
