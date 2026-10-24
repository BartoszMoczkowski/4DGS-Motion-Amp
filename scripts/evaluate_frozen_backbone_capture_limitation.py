"""
evaluate_frozen_backbone_capture_limitation.py — Scientific Audit & Frozen CC Backbone Evaluation
across all 18 cells of the Capture-Limitation suite.

Addresses audit items:
1. Evaluates the actual frozen CC backbone + color gating (r=0.3 cm, min_cluster_size=15).
2. Reports 3 dilation modes: dilated_1nn, no_dilation_cminus1, no_dilation_singletons.
3. Reports n_scored, n_labeled, n_discarded, n_seeds, K_pred per row.
4. Computes paired interface edge AUROC (Cat A contact <0.2mm, Cat B clearance >=0.2mm)
   for both motion (phasor discrepancy at f0=10Hz) and appearance (RGB distance).
5. Dual-scores Series A against both 107-class CAD GT and 2-class physical Motion-Class GT.
6. Analyzes pixel-space motion floors (px vs mm) across Series A and Series R.
7. Emits:
   - runs/capture_limitation_backbone_results.csv
   - runs/capture_limitation_edge_auroc.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csgraph, csr_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.vendored.host.metrics import (
    PUMP01_BASELINE_MERGED_PAIRS_141,
    adjusted_rand_index,
    compute_pairs_split_rate,
)
from pipeline.vendored.host.multicut import build_multichannel_graph, extract_motion_features

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
    rgb = np.clip(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0)
    raw_opacity = data["opacity"]
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -20.0, 20.0)))

    return {
        "xyz": xyz,
        "f_dc": f_dc,
        "rgb": rgb,
        "opacity": sigmoid_opacity,
    }

MAPPING_FILE = REPO_ROOT / "runs" / "capture_limitation_trained_runs.json"
if not MAPPING_FILE.is_file():
    MAPPING_FILE = REPO_ROOT / "runs" / "legacy_status" / "capture_limitation_trained_runs.json"
BACKBONE_CSV = REPO_ROOT / "runs" / "capture_limitation_backbone_results.csv"
AUROC_CSV = REPO_ROOT / "runs" / "capture_limitation_edge_auroc.csv"

TARGET_PART_LABEL = 8  # part_008 in pump01 CAD
CAT_A_PAIRS = set(PUMP01_BASELINE_MERGED_PAIRS_141[:39])
CAT_B_PAIRS = set(PUMP01_BASELINE_MERGED_PAIRS_141[39:])
ALL_141_PAIRS = set(PUMP01_BASELINE_MERGED_PAIRS_141)

CELLS = [
    # Series A (Motion Amplitude Sweep)
    {"id": "sA0", "dataset": "pump01_sA0", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 0.0, "shear_mm": 1.0, "px_per_mm": 1.0, "desc": "target static, common-mode 1mm non-target"},
    {"id": "sA1", "dataset": "pump01_sA1", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 2.0, "shear_mm": 1.0, "px_per_mm": 1.0, "desc": "target 2A, common-mode 1A (|B-A|=1A)"},
    {"id": "sA2", "dataset": "pump01_sA2", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 5.0, "shear_mm": 4.0, "px_per_mm": 1.0, "desc": "target 5A, common-mode 1A (|B-A|=4A)"},
    {"id": "sA3", "dataset": "pump01_sA3", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 10.0, "shear_mm": 9.0, "px_per_mm": 1.0, "desc": "target 10A, common-mode 1A (|B-A|=9A)"},
    {"id": "sA4", "dataset": "pump01_sA4", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 20.0, "shear_mm": 19.0, "px_per_mm": 1.0, "desc": "target 20A, common-mode 1A (|B-A|=19A)"},
    {"id": "sA5", "dataset": "pump01_sA5", "series": "A", "res": "1600x900", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 1.0, "px_per_mm": 1.0, "desc": "target 1A only, non-target static"},

    # Series R (Resolution & Distance Sweep)
    {"id": "sR0", "dataset": "pump01_sR0", "series": "R", "res": "1600x900", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.0, "desc": "baseline reference (1600x900, R=4.2m)"},
    {"id": "sR1", "dataset": "pump01_sR1", "series": "R", "res": "2560x1440", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.6, "desc": "res 2560x1440, R=4.2m"},
    {"id": "sR2", "dataset": "pump01_sR2", "series": "R", "res": "3000x2000", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.9, "desc": "res 3000x2000 (6 MP), R=4.2m"},
    {"id": "sR3", "dataset": "pump01_sR3", "series": "R", "res": "1600x900", "dist": 2.1, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 2.0, "desc": "dist half (1600x900, R=2.1m)"},
    {"id": "sR4", "dataset": "pump01_sR4", "series": "R", "res": "3000x2000", "dist": 2.1, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 3.8, "desc": "res 3k + dist half (3000x2000, R=2.1m)"},
    {"id": "sR5", "dataset": "pump01_sR5", "series": "R", "res": "3000x2000", "dist": 2.1, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 3.8, "desc": "res 3k + dist half + 120 frames"},

    # Series T (Material Realism Ablation)
    {"id": "sT0", "dataset": "pump01_sT0", "series": "T", "res": "1600x900", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.0, "desc": "pastel displayColors baseline"},
    {"id": "sT1", "dataset": "pump01_sT1", "series": "T", "res": "1600x900", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.0, "desc": "uniform brushed steel metallic"},
    {"id": "sT2", "dataset": "pump01_sT2", "series": "T", "res": "1600x900", "dist": 4.2, "target_amp_mm": 1.0, "shear_mm": 0.0, "px_per_mm": 1.0, "desc": "metallic + procedural surface texture"},

    # Series F (Compound Realistic Cells)
    {"id": "sF0", "dataset": "pump01_sF0", "series": "F", "res": "1600x900", "dist": 4.2, "target_amp_mm": 0.0, "shear_mm": 1.0, "px_per_mm": 1.0, "desc": "compound baseline control (A0 x T0 x R0)"},
    {"id": "sF1", "dataset": "pump01_sF1", "series": "F", "res": "3000x2000", "dist": 2.1, "target_amp_mm": 10.0, "shear_mm": 9.0, "px_per_mm": 3.8, "desc": "compound honest (A3 10x x T1 metallic x R4)"},
    {"id": "sF2", "dataset": "pump01_sF2", "series": "F", "res": "3000x2000", "dist": 2.1, "target_amp_mm": 10.0, "shear_mm": 9.0, "px_per_mm": 3.8, "desc": "compound honest + texture (A3 10x x T2 text x R4)"},
]


def compute_exact_auroc(score: np.ndarray, y: np.ndarray, higher_is_same: bool = False) -> float:
    """Exact Mann-Whitney U AUROC without subsampling.
    If higher_is_same is False, higher score indicates distinct/cross-part (distance/discrepancy).
    """
    if len(score) == 0:
        return 0.5
    # y: 1 for cross-part (positive class for discrimination), 0 for same-part
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    if higher_is_same:
        score = -score
    ranks = scipy.stats.rankdata(score)
    u = np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def fast_best_iou_matching(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Hungarian matching mean IoU over GT classes."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    u_true, inv_true = np.unique(labels_true, return_inverse=True)
    u_pred, inv_pred = np.unique(labels_pred, return_inverse=True)
    n_true = len(u_true)
    n_pred = len(u_pred)

    mat = coo_matrix((np.ones(len(labels_true)), (inv_true, inv_pred)), shape=(n_true, n_pred)).tocsr()
    if n_pred > 200:
        cand_cols = np.unique(mat.indices)
        mat_sub = mat[:, cand_cols].toarray()
        pred_sizes = np.bincount(inv_pred)[cand_cols]
    else:
        mat_sub = mat.toarray()
        pred_sizes = np.bincount(inv_pred)

    true_sizes = np.bincount(inv_true)
    union = true_sizes[:, None] + pred_sizes[None, :] - mat_sub
    iou = mat_sub / np.maximum(union, 1)

    row, col = linear_sum_assignment(-iou)
    matched_sum = float(iou[row, col].sum())
    return float(matched_sum / n_true if n_true else 0.0)


def evaluate_target_part(gt_on_pred: np.ndarray, pred_labels: np.ndarray, target_label: int = TARGET_PART_LABEL) -> Dict[str, Any]:
    """Target part (part_008) precision, recall, F1, and IoU."""
    gt_mask = (gt_on_pred == target_label)
    gt_count = int(gt_mask.sum())
    if gt_count == 0:
        return {"target_gt_points": 0, "target_iou": 0.0, "target_prec": 0.0, "target_rec": 0.0, "target_f1": 0.0}

    clusters = np.unique(pred_labels)
    clusters = clusters[clusters != -1]

    best_iou, best_prec, best_rec, best_f1 = 0.0, 0.0, 0.0, 0.0
    for c in clusters:
        pred_mask = (pred_labels == c)
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = float(inter / union) if union > 0 else 0.0
        prec = float(inter / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
        rec = float(inter / gt_count) if gt_count > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if iou > best_iou:
            best_iou, best_prec, best_rec, best_f1 = iou, prec, rec, f1

    return {
        "target_gt_points": gt_count,
        "target_iou": round(best_iou, 6),
        "target_prec": round(best_prec, 6),
        "target_rec": round(best_rec, 6),
        "target_f1": round(best_f1, 6),
    }


def get_series_a_motion_gt(cid: str, sub_gt_cad: np.ndarray) -> np.ndarray:
    """Construct ground truth motion classes for Series A within dynamic ROI."""
    # In dynamic ROI: non-target CAD parts (1..106 except 8) vs target part_008 (8)
    motion_gt = np.zeros_like(sub_gt_cad)
    if cid == "sA0":
        # Target static (same as casing, class 0), common-mode body moving (class 1)
        motion_gt[sub_gt_cad == TARGET_PART_LABEL] = 0
        motion_gt[sub_gt_cad != TARGET_PART_LABEL] = 1
    elif cid in ("sA1", "sA2", "sA3", "sA4"):
        # Common-mode non-target (class 1) vs Deviant target (class 2)
        motion_gt[sub_gt_cad != TARGET_PART_LABEL] = 1
        motion_gt[sub_gt_cad == TARGET_PART_LABEL] = 2
    elif cid == "sA5":
        # Non-target static (class 0), target moving (class 1)
        motion_gt[sub_gt_cad != TARGET_PART_LABEL] = 0
        motion_gt[sub_gt_cad == TARGET_PART_LABEL] = 1
    return motion_gt


def main():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        trained_runs = json.load(f)

    print("=" * 80, flush=True)
    print("FROZEN CC BACKBONE EVALUATION & PRE-REGISTERED AUROC AUDIT (18 CELLS)", flush=True)
    print("=" * 80, flush=True)

    backbone_rows = []
    auroc_rows = []

    cat_a_pairs_list = PUMP01_BASELINE_MERGED_PAIRS_141[:39]
    cat_b_pairs_list = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

    # Fast 2D lookup matrix for Cat A and Cat B
    lut_cat_a = np.zeros((110, 110), dtype=bool)
    for p1, p2 in cat_a_pairs_list:
        lut_cat_a[p1, p2] = True
        lut_cat_a[p2, p1] = True

    lut_cat_b = np.zeros((110, 110), dtype=bool)
    for p1, p2 in cat_b_pairs_list:
        lut_cat_b[p1, p2] = True
        lut_cat_b[p2, p1] = True

    for idx, c in enumerate(CELLS, 1):
        cid = c["id"]
        run_id = trained_runs[cid]
        run_dir = REPO_ROOT / "runs" / run_id
        print(f"\n[{idx}/18] Processing {cid} ({c['series']}) | {run_id} | {c['desc']}", flush=True)

        # 1. Load Trajectories & Coordinates
        traj_data = np.load(run_dir / "trajectories.npz")
        xyz = traj_data["canonical_xyz"]
        traj = traj_data["traj"]

        # 2. Load GT Labels
        dataset_dir = REPO_ROOT / "data" / "multipleview" / c["dataset"]
        gt_path = dataset_dir / "gt_segmentation.npz"
        if not gt_path.exists():
            from pipeline.artifacts import load_manifest
            m = load_manifest(run_id)
            gt_path = Path(m.artifacts["gt_segmentation"].path)
        gt_data = np.load(gt_path)
        tree_gt = cKDTree(gt_data["points"])
        _, nn = tree_gt.query(xyz, k=1)
        gt_labels = gt_data["labels"][nn]

        # 3. Load PLY Attributes (RGB & Opacity)
        ply_file = sorted((run_dir / "train_out/point_cloud").glob("*/*.ply"))[-1]
        ply_data = read_point_cloud_ply(ply_file)
        rgb = ply_data["rgb"]
        op = ply_data["opacity"]

        # Dynamic ROI (excluding casing/background label 0)
        oracle_roi = (gt_labels > 0)
        n_scored = int(oracle_roi.sum())
        sub_xyz = xyz[oracle_roi]
        sub_gt_cad = gt_labels[oracle_roi]
        sub_rgb = rgb[oracle_roi]
        sub_op = op[oracle_roi]
        sub_traj = traj[oracle_roi]

        # Motion classes for Series A
        sub_gt_motion = get_series_a_motion_gt(cid, sub_gt_cad) if c["series"] == "A" else None

        # -------------------------------------------------------------
        # PART A: Paired Interface Edge AUROC (A16 Protocol)
        # -------------------------------------------------------------
        # Extract k-NN graph within ROI (k=12, r=30mm)
        edges, feats, _ = build_multichannel_graph(
            sub_xyz, traj=sub_traj, rgb=sub_rgb, k=12, radius_max=0.03, f0=10.0, include_lifted=False
        )
        u, v = edges[:, 0], edges[:, 1]
        p_i, p_j = sub_gt_cad[u], sub_gt_cad[v]

        same_edges = (p_i == p_j)
        cross_edges = ~same_edges

        # Vectorized lookup for Cat A and Cat B cross edges
        is_cat_a = cross_edges & lut_cat_a[p_i, p_j]
        is_cat_b = cross_edges & lut_cat_b[p_i, p_j]

        disc_phasor = feats[:, 3]
        d_rgb = feats[:, 1]

        # AUROC for Cat A: positive class = Cat A cross-edges, negative class = same-part edges
        mask_cat_a = same_edges | is_cat_a
        y_cat_a = is_cat_a[mask_cat_a].astype(int)
        auroc_motion_cat_a = compute_exact_auroc(disc_phasor[mask_cat_a], y_cat_a, higher_is_same=False)
        auroc_app_cat_a = compute_exact_auroc(d_rgb[mask_cat_a], y_cat_a, higher_is_same=False)

        # AUROC for Cat B: positive class = Cat B cross-edges, negative class = same-part edges
        mask_cat_b = same_edges | is_cat_b
        y_cat_b = is_cat_b[mask_cat_b].astype(int)
        auroc_motion_cat_b = compute_exact_auroc(disc_phasor[mask_cat_b], y_cat_b, higher_is_same=False)
        auroc_app_cat_b = compute_exact_auroc(d_rgb[mask_cat_b], y_cat_b, higher_is_same=False)

        # Overall edge AUROC
        y_all = cross_edges.astype(int)
        auroc_motion_all = compute_exact_auroc(disc_phasor, y_all, higher_is_same=False)
        auroc_app_all = compute_exact_auroc(d_rgb, y_all, higher_is_same=False)

        # Deviant part boundary AUROC (edges connecting part_008 to other parts vs internal part_008 edges)
        target_internal = (p_i == TARGET_PART_LABEL) & (p_j == TARGET_PART_LABEL)
        target_cross = ((p_i == TARGET_PART_LABEL) & (p_j != TARGET_PART_LABEL)) | ((p_j == TARGET_PART_LABEL) & (p_i != TARGET_PART_LABEL))
        mask_target_edge = target_internal | target_cross
        y_target = target_cross[mask_target_edge].astype(int)
        auroc_motion_target = compute_exact_auroc(disc_phasor[mask_target_edge], y_target, higher_is_same=False)

        auroc_rows.append({
            "cell": cid,
            "series": c["series"],
            "res": c["res"],
            "dist": c["dist"],
            "shear_mm": c["shear_mm"],
            "shear_px": round(c["shear_mm"] * c["px_per_mm"], 2),
            "n_roi_pts": n_scored,
            "n_edges": len(edges),
            "n_same_edges": int(same_edges.sum()),
            "n_cat_a_edges": int(is_cat_a.sum()),
            "n_cat_b_edges": int(is_cat_b.sum()),
            "auroc_motion_cat_a": round(auroc_motion_cat_a, 4),
            "auroc_motion_cat_b": round(auroc_motion_cat_b, 4),
            "auroc_motion_all": round(auroc_motion_all, 4),
            "auroc_motion_target_part": round(auroc_motion_target, 4),
            "auroc_appearance_cat_a": round(auroc_app_cat_a, 4),
            "auroc_appearance_cat_b": round(auroc_app_cat_b, 4),
            "auroc_appearance_all": round(auroc_app_all, 4),
        })

        print(f"       AUROC -> Motion CatA: {auroc_motion_cat_a:.4f} | Motion CatB: {auroc_motion_cat_b:.4f} | "
              f"Target Boundary: {auroc_motion_target:.4f} | App All: {auroc_app_all:.4f}")

        # -------------------------------------------------------------
        # PART B: Frozen CC Backbone Evaluation (r=0.3 cm)
        # -------------------------------------------------------------
        tree_sub = cKDTree(sub_xyz)
        pairs_3mm = tree_sub.query_pairs(r=0.003, output_type="ndarray")

        # 1. Pure Spatial CC (r=0.3 cm)
        # Build adjacency graph
        if len(pairs_3mm) > 0:
            adj = csr_matrix((np.ones(len(pairs_3mm), dtype=bool), (pairs_3mm[:, 0], pairs_3mm[:, 1])), shape=(n_scored, n_scored))
            adj = adj + adj.T
        else:
            adj = csr_matrix((n_scored, n_scored), dtype=bool)

        n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
        unique, counts = np.unique(raw_labels, return_counts=True)
        valid_comp = unique[counts >= 15]

        n_seeds = len(valid_comp)
        n_labeled = int(counts[counts >= 15].sum()) if n_seeds > 0 else 0
        n_discarded = n_scored - n_labeled

        # Dilation Mode 1: dilated_1nn (Voronoi nearest seed assignment)
        labels_dilated = np.full(n_scored, -1, dtype=np.int64)
        comp_map = {comp: idx for idx, comp in enumerate(valid_comp)}
        for comp, idx in comp_map.items():
            labels_dilated[raw_labels == comp] = idx

        unassigned = np.where(labels_dilated == -1)[0]
        assigned = np.where(labels_dilated >= 0)[0]
        if len(assigned) > 0 and len(unassigned) > 0:
            sub_assigned_tree = cKDTree(sub_xyz[assigned])
            _, nearest = sub_assigned_tree.query(sub_xyz[unassigned], k=1)
            labels_dilated[unassigned] = labels_dilated[assigned[nearest]]
        elif len(assigned) == 0:
            labels_dilated[:] = 0

        # Dilation Mode 2: no_dilation_cminus1 (unassigned = -1)
        labels_cminus1 = np.full(n_scored, -1, dtype=np.int64)
        for comp, idx in comp_map.items():
            labels_cminus1[raw_labels == comp] = idx

        # Dilation Mode 3: no_dilation_singletons (unassigned = unique singleton)
        labels_singletons = labels_cminus1.copy()
        if len(unassigned) > 0:
            labels_singletons[unassigned] = np.arange(len(valid_comp), len(valid_comp) + len(unassigned))

        modes = [
            ("spatial_cc_r0.3cm", "dilated_1nn", labels_dilated),
            ("spatial_cc_r0.3cm", "no_dilation_cminus1", labels_cminus1),
            ("spatial_cc_r0.3cm", "no_dilation_singletons", labels_singletons),
        ]

        # 2. Color-Gated Spatial CC (r=0.3 cm, c_th=0.15)
        if len(pairs_3mm) > 0:
            c_diff = sub_rgb[pairs_3mm[:, 0]] - sub_rgb[pairs_3mm[:, 1]]
            c_dist = np.linalg.norm(c_diff, axis=1)
            valid_c_edge = (c_dist <= 0.15)
            pairs_cg = pairs_3mm[valid_c_edge]
            if len(pairs_cg) > 0:
                adj_cg = csr_matrix((np.ones(len(pairs_cg), dtype=bool), (pairs_cg[:, 0], pairs_cg[:, 1])), shape=(n_scored, n_scored))
                adj_cg = adj_cg + adj_cg.T
            else:
                adj_cg = csr_matrix((n_scored, n_scored), dtype=bool)
        else:
            adj_cg = csr_matrix((n_scored, n_scored), dtype=bool)

        _, raw_cg_labels = csgraph.connected_components(adj_cg, directed=False)
        u_cg, cnts_cg = np.unique(raw_cg_labels, return_counts=True)
        val_cg = u_cg[cnts_cg >= 15]
        n_seeds_cg = len(val_cg)
        n_lab_cg = int(cnts_cg[cnts_cg >= 15].sum()) if n_seeds_cg > 0 else 0
        n_disc_cg = n_scored - n_lab_cg

        labels_cg_dilated = np.full(n_scored, -1, dtype=np.int64)
        comp_map_cg = {comp: idx for idx, comp in enumerate(val_cg)}
        for comp, idx in comp_map_cg.items():
            labels_cg_dilated[raw_cg_labels == comp] = idx
        un_cg = np.where(labels_cg_dilated == -1)[0]
        as_cg = np.where(labels_cg_dilated >= 0)[0]
        if len(as_cg) > 0 and len(un_cg) > 0:
            tree_as_cg = cKDTree(sub_xyz[as_cg])
            _, near_cg = tree_as_cg.query(sub_xyz[un_cg], k=1)
            labels_cg_dilated[un_cg] = labels_cg_dilated[as_cg[near_cg]]
        elif len(as_cg) == 0:
            labels_cg_dilated[:] = 0

        modes.append(("color_gated_cc_r0.3cm", "dilated_1nn", labels_cg_dilated))

        # Evaluate each mode
        for method_name, d_mode, pred in modes:
            ari_cad = float(adjusted_rand_index(sub_gt_cad, pred))
            ari_motion = float(adjusted_rand_index(sub_gt_motion, pred)) if sub_gt_motion is not None else np.nan
            mean_iou = fast_best_iou_matching(sub_gt_cad, pred)
            k_pred = len(np.unique(pred))

            ps_cat_a = compute_pairs_split_rate(sub_gt_cad, pred, merged_pairs=cat_a_pairs_list)
            ps_cat_b = compute_pairs_split_rate(sub_gt_cad, pred, merged_pairs=cat_b_pairs_list)
            ps_141 = compute_pairs_split_rate(sub_gt_cad, pred, merged_pairs=PUMP01_BASELINE_MERGED_PAIRS_141)

            t_metrics = evaluate_target_part(sub_gt_cad, pred, target_label=TARGET_PART_LABEL)

            cur_seeds = n_seeds_cg if "color_gated" in method_name else n_seeds
            cur_lab = n_lab_cg if "color_gated" in method_name else n_labeled
            cur_disc = n_disc_cg if "color_gated" in method_name else n_discarded

            backbone_rows.append({
                "cell": cid,
                "series": c["series"],
                "res": c["res"],
                "dist": c["dist"],
                "shear_mm": c["shear_mm"],
                "shear_px": round(c["shear_mm"] * c["px_per_mm"], 2),
                "method": method_name,
                "dilation_mode": d_mode,
                "n_scored": n_scored,
                "n_seeds": cur_seeds,
                "n_labeled": cur_lab,
                "n_discarded": cur_disc,
                "k_pred": k_pred,
                "roi_ari_cad": round(ari_cad, 4),
                "roi_ari_motion": round(ari_motion, 4) if not np.isnan(ari_motion) else "",
                "mean_iou_cad": round(mean_iou, 4),
                "cat_a_split_rate": round(ps_cat_a["pairs_split_rate"], 4),
                "n_cat_a_split": ps_cat_a["n_pairs_split"],
                "cat_b_split_rate": round(ps_cat_b["pairs_split_rate"], 4),
                "n_cat_b_split": ps_cat_b["n_pairs_split"],
                "pairs_split_141": round(ps_141["pairs_split_rate"], 4),
                "n_pairs_split_141": ps_141["n_pairs_split"],
                **t_metrics,
            })

            if d_mode == "dilated_1nn":
                print(f"       [{method_name}] CAD ARI: {ari_cad:.4f} " +
                      (f"(Motion ARI: {ari_motion:.4f}) " if not np.isnan(ari_motion) else "") +
                      f"| Seeds: {cur_seeds} | K_pred: {k_pred} | CatA Split: {ps_cat_a['pairs_split_rate']:.3f} | CatB Split: {ps_cat_b['pairs_split_rate']:.3f}")

    # Write out Backbone CSV
    fieldnames_bb = list(backbone_rows[0].keys())
    with open(BACKBONE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_bb)
        writer.writeheader()
        writer.writerows(backbone_rows)
    print(f"\nSaved Frozen CC Backbone results to: {BACKBONE_CSV}")

    # Write out AUROC CSV
    fieldnames_auroc = list(auroc_rows[0].keys())
    with open(AUROC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_auroc)
        writer.writeheader()
        writer.writerows(auroc_rows)
    print(f"Saved Paired Interface Edge AUROC results to: {AUROC_CSV}")


if __name__ == "__main__":
    main()
