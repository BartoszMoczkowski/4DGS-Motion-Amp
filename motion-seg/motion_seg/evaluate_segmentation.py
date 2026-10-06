#!/usr/bin/env python3
"""Compare a predicted segmentation (motion_seg/segment_rigid.py output) against the
Omniverse ground-truth per-part labels (data/multipleview/<name>/gt_segmentation.npz,
produced by omni_to_4dgs.py from omni_capture.py's points3D_labels.npy).

The two point sets differ (GT = the sampled init cloud; predicted = the trained Gaussians,
whose count changes with densification/pruning), but they live in the same coordinate frame
(both went through the same omni_to_4dgs.py scale normalization), so GT labels are propagated
onto the predicted points by nearest neighbor before scoring.

Usage:
    python -m motion_seg.evaluate_segmentation \
        --pred output/multipleview/pump01/segmentation.npz \
        --gt data/multipleview/pump01/gt_segmentation.npz \
        --recolored-ply output/multipleview/pump01/segmentation_preview.ply
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from motion_seg.metrics import adjusted_rand_index, best_iou_matching


def propagate_labels(src_points, src_labels, dst_points):
    """Nearest-neighbor label transfer from src (GT) onto dst (predicted) points."""
    tree = cKDTree(src_points)
    _, nn = tree.query(dst_points, k=1)
    return src_labels[nn]


def evaluate(pred_points, pred_labels, gt_points, gt_labels, *, drop_floaters: bool = False,
             roi_mask: np.ndarray | None = None, bg_label=None) -> dict:
    """Score a predicted segmentation against GT.

    `bg_label` selects a GT class to exclude from the additional `ari_within_roi` score:
    - None (default): no exclusion, no ROI score is computed.
    - an integer: exclude that GT label.
    - "auto": legacy heuristic — exclude GT label 0 whenever label 0 and any positive
      label coexist. WARNING: label 0 is just the first mesh in USD traversal order (see
      omni_capture.py); it has no background semantics. Only use "auto" when you have
      verified label 0 really is the background for this scene.

    Returns a dict with ari, mean_iou, matches, gt_on_pred, pred_points, pred_labels,
    n_gt, n_pred, n_pred_nonfloater, and optionally ari_within_roi / n_roi_points /
    bg_label_excluded.
    """
    pred_points = np.asarray(pred_points)
    pred_labels = np.asarray(pred_labels)
    gt_points = np.asarray(gt_points)
    gt_labels = np.asarray(gt_labels)

    if drop_floaters:
        mask = pred_labels != -1
        pred_points, pred_labels = pred_points[mask], pred_labels[mask]

    gt_on_pred = propagate_labels(gt_points, gt_labels, pred_points)

    ari = adjusted_rand_index(gt_on_pred, pred_labels)
    mean_iou, matches = best_iou_matching(gt_on_pred, pred_labels)

    uniq_pred = np.unique(pred_labels)
    n_floaters = int((pred_labels == -1).sum())
    result = {
        "ari": ari,
        "mean_iou": mean_iou,
        "matches": matches,
        "gt_on_pred": gt_on_pred,
        "pred_points": pred_points,
        "pred_labels": pred_labels,
        "n_gt": len(np.unique(gt_labels)),
        "n_pred": len(uniq_pred),
        # Predicted segments excluding the floater label -1 (honest segment count).
        "n_pred_nonfloater": int(len(uniq_pred[uniq_pred != -1])),
        "n_floater_points": n_floaters,
        "bg_label_excluded": None,
    }

    eval_roi_mask = roi_mask
    excluded = None
    if eval_roi_mask is None and bg_label is not None:
        if bg_label == "auto":
            # Legacy heuristic (pre-2026-10-05 default): exclude GT label 0 when it
            # coexists with positive labels. Label 0 has NO background semantics — it is
            # the first mesh in USD traversal order — so this is only correct by
            # coincidence. Kept for backward comparability; prefer an explicit integer.
            if (gt_on_pred > 0).any() and (gt_on_pred == 0).any():
                excluded = 0
        else:
            excluded = int(bg_label)
            if not (gt_on_pred == excluded).any():
                print(f"[warn] --bg-label {excluded}: no GT points carry that label; "
                      f"ROI = whole cloud", file=sys.stderr)
                excluded = None
        if excluded is not None:
            eval_roi_mask = gt_on_pred != excluded
            print(f"[eval] ari_within_roi: EXCLUDING GT label {excluded} "
                  f"({int((gt_on_pred == excluded).sum())} of {len(gt_on_pred)} points) "
                  f"from the ROI score (bg_label={bg_label!r})")

    result["bg_label_excluded"] = excluded

    if eval_roi_mask is not None:
        eval_roi_mask = np.asarray(eval_roi_mask)
        if len(eval_roi_mask) != len(pred_points):
            raise ValueError(
                f"roi_mask length {len(eval_roi_mask)} != pred_points length {len(pred_points)}"
            )
        in_roi = eval_roi_mask
        if in_roi.any():
            result["ari_within_roi"] = float(adjusted_rand_index(
                gt_on_pred[in_roi], pred_labels[in_roi]
            ))
        else:
            result["ari_within_roi"] = None
        result["n_roi_points"] = int(in_roi.sum())

    return result


def _write_colored_ply(path, xyz, labels):
    """Small self-contained PLY writer (pseudo-color per label) for a quick visual sanity
    check in any mesh viewer (MeshLab, CloudCompare, Blender...)."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    rng = np.random.RandomState(0)
    palette = {lab: rng.randint(40, 230, size=3) for lab in uniq}
    palette[-1] = np.array([80, 80, 80])  # floaters / unlabeled -> grey
    rgb = np.array([palette[l] for l in labels], dtype=np.uint8)

    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    data = np.empty(n, dtype=dtype)
    data["xyz"] = xyz
    data["rgb"] = rgb
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="segmentation.npz (points, labels) from segment_rigid.py")
    ap.add_argument("--gt", required=True, help="gt_segmentation.npz (points, labels) from omni_to_4dgs.py")
    ap.add_argument("--drop-floaters", action="store_true",
                     help="exclude predicted label == -1 (floaters) from scoring")
    ap.add_argument("--bg-label", default=None, metavar="LABEL|auto",
                     help="GT label to exclude from the additional ari_within_roi score. "
                          "Default: no exclusion. Pass an integer to exclude that GT class, "
                          "or 'auto' for the legacy heuristic (exclude GT label 0 when it "
                          "coexists with positive labels — only valid if you have verified "
                          "label 0 is really background; USD traversal order assigns labels "
                          "arbitrarily). The excluded label is always printed loudly.")
    ap.add_argument("--recolored-ply", default=None,
                     help="optional: write a PLY colored by predicted label for visual QA")
    ap.add_argument("--comparison-png", default=None,
                     help="write a GT-vs-predicted 3-view comparison PNG here "
                          "(default: <pred>_vs_gt.png; pass '' to skip)")
    ap.add_argument("--top-n", type=int, default=15, help="how many best/worst matches to print")
    args = ap.parse_args()

    bg_label = args.bg_label
    if bg_label is not None and bg_label != "auto":
        try:
            bg_label = int(bg_label)
        except ValueError:
            ap.error(f"--bg-label must be an integer or 'auto', got {bg_label!r}")

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    result = evaluate(
        pred["points"], pred["labels"], gt["points"], gt["labels"],
        drop_floaters=args.drop_floaters, bg_label=bg_label,
    )
    ari, mean_iou, matches = result["ari"], result["mean_iou"], result["matches"]
    pred_points, pred_labels = result["pred_points"], result["pred_labels"]
    n_gt = result["n_gt"]
    n_pred_seg = result["n_pred_nonfloater"]
    n_floater = result["n_floater_points"]

    print(f"GT instances: {n_gt}  |  predicted segments: {n_pred_seg}  |  "
          f"predicted points: {len(pred_labels)}")
    if n_floater > 0 and not args.drop_floaters:
        print(f"NOTE: {n_floater} floater point(s) carry predicted label -1 and are NOT "
              f"counted in 'predicted segments' above. In the ARI score below, -1 is "
              f"treated as one ordinary cluster (it absorbs all unmatched/unsampled "
              f"points); pass --drop-floaters to exclude them from scoring entirely.")
    print(f"Adjusted Rand Index: {ari:.4f}")
    n_matched = len(matches)
    print(f"Mean best-match IoU: {mean_iou:.4f}  [convention: mean over {n_gt} GT classes, "
          f"{n_matched} Hungarian-matched pair(s), unmatched GT classes count as IoU 0]")
    if result.get("ari_within_roi") is not None or result.get("bg_label_excluded") is not None:
        print(f"ARI within ROI (GT label {result['bg_label_excluded']} excluded, "
              f"{result.get('n_roi_points', 0)} points): "
              f"{result.get('ari_within_roi')}")

    print(f"\nTop {args.top_n} GT parts by size and their best-matching predicted segment:")
    print(f"{'gt_label':>10} {'gt_size':>8} {'pred_label':>11} {'pred_size':>10} {'iou':>7}")
    for gt_l, pred_l, iou, gt_sz, pred_sz in matches[: args.top_n]:
        print(f"{gt_l:>10} {gt_sz:>8} {pred_l:>11} {pred_sz:>10} {iou:>7.3f}")

    if args.recolored_ply:
        _write_colored_ply(args.recolored_ply, pred_points, pred_labels)
        print(f"\n[ok] wrote {args.recolored_ply} (color-per-predicted-segment, grey = floaters)")

    comparison_png = args.comparison_png
    if comparison_png is None:
        base, _ext = os.path.splitext(args.pred)
        comparison_png = (base or args.pred) + "_vs_gt.png"
    if comparison_png:
        from motion_seg.visualize import render_comparison_png

        render_comparison_png(
            pred_points, result["gt_on_pred"], pred_labels, "GT (propagated)", "Predicted", comparison_png
        )
        print(f"[ok] wrote {comparison_png}")


if __name__ == "__main__":
    main()
