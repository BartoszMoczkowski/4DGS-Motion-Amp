"""Vendored, verbatim copy of ``motion-seg/motion_seg/evaluate_segmentation.py``'s ``propagate_labels()``,
``evaluate()``, and ``_write_colored_ply()`` (2026-07-14 copy-in rework of T07; see
``pipeline.vendored``'s module docstring). Function bodies are byte-for-byte the reference
script's; the only change is importing ``adjusted_rand_index``/``best_iou_matching`` from the
sibling vendored module (:mod:`pipeline.vendored.host.metrics`) instead of
``motion_seg.metrics``. The reference script's CLI/argparse/``main`` are intentionally not
ported — only these three functions are a production dependency of
``pipeline/stages/seg_eval.py``.

Original docstring:

    Compare a predicted segmentation (motion-seg/motion_seg/segment_rigid.py output) against the
    Omniverse ground-truth per-part labels. The two point sets differ (GT = the sampled init
    cloud; predicted = the trained Gaussians, whose count changes with densification/pruning),
    but they live in the same coordinate frame (both went through the same omni_to_4dgs.py
    scale normalization), so GT labels are propagated onto the predicted points by nearest
    neighbor before scoring.
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


def evaluate(pred_points, pred_labels, gt_points, gt_labels, *, drop_floaters: bool = False) -> dict:
    """Score a predicted segmentation against GT.

    Returns a dict with ``ari``, ``mean_iou``, ``matches`` (as :func:`best_iou_matching` returns
    them), ``gt_on_pred`` (GT labels propagated onto ``pred_points``), the (possibly
    floater-dropped) ``pred_points``/``pred_labels`` actually scored, and ``n_gt``/``n_pred``
    instance counts.
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
