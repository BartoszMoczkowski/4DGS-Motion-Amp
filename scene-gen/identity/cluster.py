"""scene-gen/identity/cluster.py — Evaluation, post-processing, and PLY recoloring for E2.

Evaluates discrete identity clusterings against CAD GT (107 parts / 2 cubes) and
motion equivalence classes, generates colored PLYs, and saves segmentation.npz.
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score


def propagate_gt_to_pred(
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    pred_points: np.ndarray,
) -> np.ndarray:
    """Propagate GT labels from sparse init points to dense predicted Gaussians via nearest neighbor."""
    tree = cKDTree(gt_points)
    _, idx = tree.query(pred_points)
    return gt_labels[idx]


def compute_best_iou(gt_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, Dict[int, int]]:
    """Compute mean best-match IoU between GT instances and predicted clusters."""
    gt_unique = np.unique(gt_labels)
    pred_unique = np.unique(pred_labels)

    ious = []
    matches = {}

    for g in gt_unique:
        if g == 0:  # Skip background in IoU calculation
            continue
        g_mask = (gt_labels == g)
        best_iou = 0.0
        best_p = -1
        for p in pred_unique:
            p_mask = (pred_labels == p)
            intersection = np.sum(g_mask & p_mask)
            union = np.sum(g_mask | p_mask)
            if union > 0:
                iou = float(intersection) / float(union)
                if iou > best_iou:
                    best_iou = iou
                    best_p = p
        ious.append(best_iou)
        matches[int(g)] = int(best_p)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, matches


def evaluate_identity_segmentation(
    pred_points: np.ndarray,
    pred_labels: np.ndarray,
    gt_segmentation_path: str,
    gt_motion_path: Optional[str] = None,
    roi_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score predicted segmentation against GT."""
    gt_data = np.load(gt_segmentation_path)
    gt_pts = gt_data["points"]
    gt_labels_raw = gt_data["labels"]
    n_gt_parts = len(np.unique(gt_labels_raw))

    gt_on_pred = propagate_gt_to_pred(gt_pts, gt_labels_raw, pred_points)

    # 1. Global ARI
    ari_global = float(adjusted_rand_score(gt_on_pred, pred_labels))

    # 2. Mean IoU
    mean_iou, matches = compute_best_iou(gt_on_pred, pred_labels)

    # 3. ARI within dynamic ROI (gt > 0)
    tree = cKDTree(gt_pts)
    dists, _ = tree.query(pred_points)

    if n_gt_parts <= 4 and (gt_labels_raw == 0).any():
        in_roi = (gt_on_pred > 0) & (dists <= 0.25)
    else:
        in_roi = (gt_on_pred > 0) if (gt_on_pred == 0).any() else np.ones(len(pred_points), dtype=bool)

    if in_roi.sum() > 0:
        ari_within_roi = float(adjusted_rand_score(gt_on_pred[in_roi], pred_labels[in_roi]))
    else:
        ari_within_roi = ari_global

    # 4. ARI vs motion classes
    ari_motion = None
    mean_iou_motion = None
    if gt_motion_path and os.path.isfile(gt_motion_path):
        gt_motion_raw = np.load(gt_motion_path)
        gt_motion_on_pred = propagate_gt_to_pred(gt_pts, gt_motion_raw, pred_points)
        ari_motion = float(adjusted_rand_score(gt_motion_on_pred, pred_labels))
        mean_iou_motion, _ = compute_best_iou(gt_motion_on_pred, pred_labels)

    n_pred_clusters = len(np.unique(pred_labels))
    n_gt_parts = len(np.unique(gt_labels_raw))

    results = {
        "n_points": len(pred_points),
        "n_gt_parts": n_gt_parts,
        "n_pred_clusters": n_pred_clusters,
        "ari_global": ari_global,
        "ari_within_roi": ari_within_roi,
        "mean_iou": mean_iou,
        "ari_motion": ari_motion,
        "mean_iou_motion": mean_iou_motion,
    }

    return results


def write_colored_ply(points: np.ndarray, labels: np.ndarray, out_path: str):
    """Export points colored by discrete cluster label to PLY."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Generate distinct HSV colors
    rng = np.random.default_rng(42)
    hues = np.linspace(0, 1, n_clusters, endpoint=False)
    rng.shuffle(hues)

    label_to_color = {}
    for i, l in enumerate(unique_labels):
        if l == 0:
            # Background = dark gray
            label_to_color[l] = np.array([40, 40, 40], dtype=np.uint8)
        else:
            r, g, b = colorsys.hsv_to_rgb(hues[i], 0.85, 0.95)
            label_to_color[l] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)

    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for l in unique_labels:
        colors[labels == l] = label_to_color[l]

    # Create structured array for plyfile
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(len(points), dtype=dtype)
    elements["x"] = points[:, 0]
    elements["y"] = points[:, 1]
    elements["z"] = points[:, 2]
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(out_path)
    print(f"[cluster] Wrote colored PLY to {out_path}")
