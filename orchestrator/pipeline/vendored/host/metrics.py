"""Vendored, verbatim copy of ``motion_seg/metrics.py`` (2026-07-14 copy-in rework of T07; see
``pipeline.vendored``'s module docstring; re-synced 2026-10-06 to pick up the reference's
``best_iou_matching`` mean-over-GT-classes convention fix of 2026-10-05, review item M2).
Only the module path changed — every function body below is byte-for-byte the reference
script's.

Segmentation evaluation metrics — no sklearn dependency (this repo only carries scipy),
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

    CONVENTION (changed 2026-10-05, fixes review item M2): `mean_iou` is the mean over
    **GT classes**, with every GT class that the Hungarian matching leaves unmatched
    (because there are fewer predicted clusters than GT classes) contributing IoU 0.
    Before this fix the mean covered only the min(n_gt, n_pred) matched pairs, inflating
    the score whenever the prediction had fewer clusters than the GT. Numbers produced by
    this version are NOT comparable to mean_iou values computed before 2026-10-05.
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
    matched_rows = set(row.tolist())
    n_unmatched_gt = len(classes) - len(matched_rows)
    # Mean over GT classes: unmatched GT classes contribute IoU 0 (see docstring).
    matched_sum = float(sum(m[2] for m in matches))
    mean_iou = matched_sum / len(classes) if len(classes) else 0.0
    if n_unmatched_gt > 0:
        print(f"[metrics] mean_iou convention: mean over {len(classes)} GT classes, "
              f"{n_unmatched_gt} unmatched GT class(es) counted as IoU 0 "
              f"({len(matches)} matched pairs)")
    return mean_iou, matches


def k_alignment(n_gt: int, n_pred: int) -> float:
    """Cluster count alignment ratio in [0.0, 1.0].

    Returns 1.0 when predicted cluster count matches ground truth part count exactly,
    and scales down smoothly towards 0.0 when severely under- or over-fragmented.
    """
    if n_gt <= 0 or n_pred <= 0:
        return 0.0
    return float(min(n_gt, n_pred) / max(n_gt, n_pred))


def hungarian_part_iou(
    labels_true,
    labels_pred,
    matches: list | None = None,
    bg_label: int | None = 0,
    terciles: tuple[float, float] = (33.33, 66.67),
) -> dict[str, float]:
    """Compute detailed Hungarian-matched per-part IoU statistics stratified by part size.

    Excludes background (by default `bg_label=0` if present) so per-part statistics
    genuinely evaluate mechanical parts.

    Returns:
        dict with:
        - "mean_part_iou": mean IoU over evaluated GT parts
        - "median_part_iou": median IoU over evaluated GT parts
        - "part_iou_small": mean IoU for small parts (lower tercile of size)
        - "part_iou_medium": mean IoU for medium parts (middle tercile)
        - "part_iou_large": mean IoU for large parts (upper tercile)
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if matches is None:
        _, matches = best_iou_matching(labels_true, labels_pred)

    matched_gt = {int(m[0]): float(m[2]) for m in matches}
    unique_gt = np.unique(labels_true)

    # Evaluate non-background classes if bg_label is specified and present
    eval_classes = [int(c) for c in unique_gt if not (bg_label is not None and c == bg_label)]
    if not eval_classes:
        eval_classes = [int(c) for c in unique_gt]

    if not eval_classes:
        return {
            "mean_part_iou": 0.0,
            "median_part_iou": 0.0,
            "part_iou_small": 0.0,
            "part_iou_medium": 0.0,
            "part_iou_large": 0.0,
        }

    class_sizes = [int((labels_true == c).sum()) for c in eval_classes]
    class_ious = [matched_gt.get(c, 0.0) for c in eval_classes]

    mean_part_iou = float(np.mean(class_ious))
    median_part_iou = float(np.median(class_ious))

    q_low = float(np.percentile(class_sizes, terciles[0]))
    q_high = float(np.percentile(class_sizes, terciles[1]))

    small_ious = [iou for sz, iou in zip(class_sizes, class_ious) if sz <= q_low]
    med_ious = [iou for sz, iou in zip(class_sizes, class_ious) if q_low < sz <= q_high]
    large_ious = [iou for sz, iou in zip(class_sizes, class_ious) if sz > q_high]

    return {
        "mean_part_iou": mean_part_iou,
        "median_part_iou": median_part_iou,
        "part_iou_small": float(np.mean(small_ious)) if small_ious else mean_part_iou,
        "part_iou_medium": float(np.mean(med_ious)) if med_ious else mean_part_iou,
        "part_iou_large": float(np.mean(large_ious)) if large_ious else mean_part_iou,
    }


def size_stratified_f1(
    labels_true,
    labels_pred,
    matches: list | None = None,
    bg_label: int | None = 0,
    terciles: tuple[float, float] = (33.33, 66.67),
) -> dict[str, float]:
    """Compute Hungarian-matched F1 / Dice scores stratified by GT part size.

    Excludes background (by default `bg_label=0` if present).

    Returns:
        dict with:
        - "f1_macro": mean F1 across all evaluated GT parts
        - "f1_small": mean F1 for GT parts in the lower tercile of size
        - "f1_medium": mean F1 for GT parts in the middle tercile
        - "f1_large": mean F1 for GT parts in the upper tercile
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if matches is None:
        _, matches = best_iou_matching(labels_true, labels_pred)

    matched_gt = {int(m[0]): float(m[2]) for m in matches}
    unique_gt = np.unique(labels_true)

    eval_classes = [int(c) for c in unique_gt if not (bg_label is not None and c == bg_label)]
    if not eval_classes:
        eval_classes = [int(c) for c in unique_gt]

    if not eval_classes:
        return {
            "f1_macro": 0.0,
            "f1_small": 0.0,
            "f1_medium": 0.0,
            "f1_large": 0.0,
        }

    class_sizes = [int((labels_true == c).sum()) for c in eval_classes]
    class_ious = [matched_gt.get(c, 0.0) for c in eval_classes]
    # F1 = 2 * IoU / (1 + IoU)
    class_f1s = [float(2.0 * iou / (1.0 + iou)) if iou > 0.0 else 0.0 for iou in class_ious]

    f1_macro = float(np.mean(class_f1s))

    q_low = float(np.percentile(class_sizes, terciles[0]))
    q_high = float(np.percentile(class_sizes, terciles[1]))

    small_f1s = [f1 for sz, f1 in zip(class_sizes, class_f1s) if sz <= q_low]
    med_f1s = [f1 for sz, f1 in zip(class_sizes, class_f1s) if q_low < sz <= q_high]
    large_f1s = [f1 for sz, f1 in zip(class_sizes, class_f1s) if sz > q_high]

    return {
        "f1_macro": f1_macro,
        "f1_small": float(np.mean(small_f1s)) if small_f1s else f1_macro,
        "f1_medium": float(np.mean(med_f1s)) if med_f1s else f1_macro,
        "f1_large": float(np.mean(large_f1s)) if large_f1s else f1_macro,
    }


#: The 141 merged dynamic part-pairs from the S6 spatial CC baseline (grid-A20mm_M2, r=0.5cm),
#: derived in runs/pump01_boundary_refine_report.json and runs/pump01_spatial_cc_diagnostic_decomposition.json.
PUMP01_BASELINE_MERGED_PAIRS_141: list[tuple[int, int]] = [
    # Cat A (Joint contact, CAD gap < 0.2 mm, 39 pairs)
    (1, 51), (1, 81), (2, 43), (2, 45), (2, 46), (2, 82), (2, 84), (2, 85),
    (3, 63), (3, 64), (3, 65), (3, 68), (3, 69), (4, 59), (4, 73), (12, 21),
    (12, 40), (38, 89), (39, 88), (40, 87), (41, 86), (43, 82), (44, 83),
    (45, 84), (46, 85), (48, 78), (49, 79), (50, 80), (51, 81), (53, 76),
    (54, 75), (55, 74), (56, 70), (57, 71), (59, 73), (61, 66), (62, 67),
    (63, 68), (64, 69),
    # Cat B (Clearance bridged / transitive, 102 pairs)
    (2, 24), (2, 27), (3, 13), (3, 26), (3, 90), (3, 97), (4, 23), (6, 19),
    (6, 33), (6, 35), (9, 12), (9, 21), (9, 40), (9, 87), (10, 37), (10, 52),
    (12, 87), (13, 26), (13, 63), (13, 64), (13, 65), (13, 68), (13, 69),
    (13, 90), (13, 97), (15, 28), (16, 29), (19, 33), (19, 35), (21, 40),
    (21, 87), (23, 59), (23, 73), (24, 27), (24, 43), (24, 45), (24, 46),
    (24, 82), (24, 84), (24, 85), (26, 63), (26, 64), (26, 65), (26, 68),
    (26, 69), (26, 90), (26, 97), (27, 43), (27, 45), (27, 46), (27, 82),
    (27, 84), (27, 85), (32, 102), (32, 104), (32, 105), (33, 35), (36, 61),
    (36, 66), (37, 52), (42, 92), (42, 95), (43, 45), (43, 46), (43, 84),
    (43, 85), (45, 46), (45, 82), (45, 85), (46, 82), (46, 84), (47, 93),
    (47, 94), (60, 91), (63, 64), (63, 65), (63, 69), (63, 90), (63, 97),
    (64, 65), (64, 68), (64, 90), (64, 97), (65, 68), (65, 69), (65, 90),
    (65, 97), (68, 69), (68, 90), (68, 97), (69, 90), (69, 97), (82, 84),
    (82, 85), (84, 85), (90, 97), (92, 95), (93, 94), (96, 106), (102, 104),
    (102, 105), (104, 105),
]


def compute_pairs_split_rate(
    labels_true,
    labels_pred,
    merged_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Calculate the split rate of baseline merged part-pairs (S6 decomposition).

    Determines whether pairs merged under the spatial CC baseline are successfully
    assigned to distinct clusters in `labels_pred`.

    Returns:
        dict with:
        - "pairs_split_rate": float in [0.0, 1.0]
        - "n_pairs_split": int
        - "n_pairs_merged_base": int
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if merged_pairs is None:
        merged_pairs = PUMP01_BASELINE_MERGED_PAIRS_141

    unique_true = set(np.unique(labels_true).tolist())
    eval_pairs = [p for p in merged_pairs if p[0] in unique_true and p[1] in unique_true]

    if not eval_pairs:
        return {
            "pairs_split_rate": 0.0,
            "n_pairs_split": 0,
            "n_pairs_merged_base": 0,
        }

    # Majority cluster assignment per GT part
    part_to_cluster: dict[int, int] = {}
    for pid in unique_true:
        mask = (labels_true == pid)
        if mask.any():
            clusters, counts = np.unique(labels_pred[mask], return_counts=True)
            # Exclude floater -1 if other clusters exist
            if len(clusters) > 1 and -1 in clusters:
                valid = clusters != -1
                clusters, counts = clusters[valid], counts[valid]
            part_to_cluster[int(pid)] = int(clusters[np.argmax(counts)])

    n_split = 0
    for p_i, p_j in eval_pairs:
        c_i = part_to_cluster.get(p_i)
        c_j = part_to_cluster.get(p_j)
        if c_i is not None and c_j is not None and c_i != c_j:
            n_split += 1

    rate = float(n_split / len(eval_pairs)) if eval_pairs else 0.0
    return {
        "pairs_split_rate": rate,
        "n_pairs_split": int(n_split),
        "n_pairs_merged_base": int(len(eval_pairs)),
    }

