"""Vendored, verbatim copy of ``motion-seg/motion_seg/evaluate_segmentation.py``'s ``propagate_labels()``,
``evaluate()``, and ``_write_colored_ply()`` (2026-07-14 copy-in rework of T07; see
``pipeline.vendored``'s module docstring). Function bodies are byte-for-byte the reference
script's; the only change is importing ``adjusted_rand_index``/``best_iou_matching`` from the
sibling vendored module (:mod:`pipeline.vendored.host.metrics`) instead of
``motion_seg.metrics``. The reference script's CLI/argparse/``main`` are intentionally not
ported — only these three functions are a production dependency of
``pipeline/stages/seg_eval.py``.

T19 addition: ``evaluate()`` gains an optional ``roi_mask`` argument for ARI-within-ROI
scoring.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .metrics import adjusted_rand_index, best_iou_matching


def propagate_labels(src_points, src_labels, dst_points):
    """Nearest-neighbor label transfer from src (GT) onto dst (predicted) points."""
    tree = cKDTree(src_points)
    _, nn = tree.query(dst_points, k=1)
    return src_labels[nn]


def evaluate(pred_points, pred_labels, gt_points, gt_labels, *, drop_floaters: bool = False,
             roi_mask: np.ndarray | None = None) -> dict:
    """Score a predicted segmentation against GT.

    Returns a dict with ``ari``, ``mean_iou``, ``matches`` (as :func:`best_iou_matching` returns
    them), ``gt_on_pred`` (GT labels propagated onto ``pred_points``), the (possibly
    floater-dropped) ``pred_points``/``pred_labels`` actually scored, and ``n_gt``/``n_pred``
    instance counts.  When ``roi_mask`` is provided, also returns ``ari_within_roi`` computed
    on the subset inside the ROI.
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

    result = {
        "ari": ari,
        "mean_iou": mean_iou,
        "matches": matches,
        "gt_on_pred": gt_on_pred,
        "pred_points": pred_points,
        "pred_labels": pred_labels,
        "n_gt": len(np.unique(gt_labels)),
        "n_pred": len(np.unique(pred_labels)),
    }

    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask)
        if len(roi_mask) != len(pred_points):
            raise ValueError(
                f"roi_mask length {len(roi_mask)} != pred_points length {len(pred_points)}"
            )
        in_roi = roi_mask
        if in_roi.any():
            result["ari_within_roi"] = adjusted_rand_index(
                gt_on_pred[in_roi], pred_labels[in_roi]
            )
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
