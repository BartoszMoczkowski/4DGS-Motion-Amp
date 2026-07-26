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

import numpy as np
from scipy.spatial import cKDTree

from motion_seg.metrics import adjusted_rand_index, best_iou_matching


def propagate_labels(src_points, src_labels, dst_points):
    """Nearest-neighbor label transfer from src (GT) onto dst (predicted) points."""
    tree = cKDTree(src_points)
    _, nn = tree.query(dst_points, k=1)
    return src_labels[nn]


def evaluate(pred_points, pred_labels, gt_points, gt_labels, *, drop_floaters: bool = False) -> dict:
    """Score a predicted segmentation against GT — the logic ``main()`` below drives from the
    CLI, graduated into an importable function (``planning/INSTRUCTIONS.md``'s "refactor into
    importable functions only when it clearly pays off" — this is that case: the pipeline
    orchestrator's ``seg_eval.default`` stage, T07, needs to call this in-process rather than
    duplicating the propagate/score glue).

    Returns a dict with ``ari``, ``mean_iou``, ``matches`` (as :func:`best_iou_matching` returns
    them), ``gt_on_pred`` (GT labels propagated onto ``pred_points``), the (possibly
    floater-dropped) ``pred_points``/``pred_labels`` actually scored, and ``n_gt``/``n_pred``
    instance counts — everything ``main()`` prints or writes a preview from.
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

    return {
        "ari": ari,
        "mean_iou": mean_iou,
        "matches": matches,
        "gt_on_pred": gt_on_pred,
        "pred_points": pred_points,
        "pred_labels": pred_labels,
        "n_gt": len(np.unique(gt_labels)),
        "n_pred": len(np.unique(pred_labels)),
    }


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
    ap.add_argument("--recolored-ply", default=None,
                     help="optional: write a PLY colored by predicted label for visual QA")
    ap.add_argument("--comparison-png", default=None,
                     help="write a GT-vs-predicted 3-view comparison PNG here "
                          "(default: <pred>_vs_gt.png; pass '' to skip)")
    ap.add_argument("--top-n", type=int, default=15, help="how many best/worst matches to print")
    args = ap.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    result = evaluate(
        pred["points"], pred["labels"], gt["points"], gt["labels"],
        drop_floaters=args.drop_floaters,
    )
    ari, mean_iou, matches = result["ari"], result["mean_iou"], result["matches"]
    pred_points, pred_labels = result["pred_points"], result["pred_labels"]
    n_gt, n_pred = result["n_gt"], result["n_pred"]

    print(f"GT instances: {n_gt}  |  predicted segments: {n_pred}  |  predicted points: {len(pred_labels)}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Mean best-match IoU (Hungarian, {min(n_gt, n_pred)} matches): {mean_iou:.4f}")

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
