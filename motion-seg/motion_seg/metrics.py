"""Segmentation evaluation metrics — no sklearn dependency (this repo only carries scipy),
just numpy + scipy.optimize.linear_sum_assignment for Hungarian matching.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """Standard Adjusted Rand Index from a contingency table. 1.0 = identical clusterings
    (up to relabeling), ~0.0 = agreement no better than chance, can go negative."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n == 0:
        return 1.0

    _, class_idx = np.unique(labels_true, return_inverse=True)
    _, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = class_idx.max() + 1
    n_clusters = cluster_idx.max() + 1

    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    np.add.at(contingency, (class_idx, cluster_idx), 1)

    def comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return x * (x - 1) / 2.0

    sum_comb_c = comb2(contingency).sum()
    sum_comb_a = comb2(contingency.sum(axis=1)).sum()
    sum_comb_b = comb2(contingency.sum(axis=0)).sum()
    total_comb = comb2(n)

    if total_comb == 0:
        return 1.0
    expected = sum_comb_a * sum_comb_b / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        # Perfect agreement or a degenerate (e.g. single-cluster) case.
        return 1.0 if sum_comb_c == max_index else 0.0
    return float((sum_comb_c - expected) / denom)


def best_iou_matching(labels_true, labels_pred):
    """Hungarian-match predicted clusters to GT classes maximizing IoU.

    Returns (mean_iou, matches) where matches is a list of
    (gt_label, pred_label, iou, gt_size, pred_size) sorted by GT size descending.
    Unmatched GT classes / predicted clusters (when counts differ) are not included in the
    mean but are reported separately for context.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    iou = np.zeros((len(classes), len(clusters)))
    class_masks = [labels_true == c for c in classes]
    cluster_masks = [labels_pred == k for k in clusters]
    for a, mask_c in enumerate(class_masks):
        for b, mask_k in enumerate(cluster_masks):
            inter = np.logical_and(mask_c, mask_k).sum()
            union = np.logical_or(mask_c, mask_k).sum()
            iou[a, b] = inter / union if union else 0.0

    row, col = linear_sum_assignment(-iou)
    matches = [
        (classes[a], clusters[b], float(iou[a, b]), int(class_masks[a].sum()), int(cluster_masks[b].sum()))
        for a, b in zip(row, col)
    ]
    matches.sort(key=lambda m: -m[3])
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return mean_iou, matches
