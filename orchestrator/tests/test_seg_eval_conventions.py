"""Tests for the 2026-10-06 seg_eval convention sync of the vendored scoring code
(`pipeline/vendored/host/metrics.py` + `pipeline/vendored/host/seg_eval.py`) against the
fixed reference (`motion-seg/motion_seg/metrics.py` / `evaluate_segmentation.py`, fixed
2026-10-05, review items M2 + background-exclusion + floater-reporting items in
`reviews/2026-10-05-motion-seg-review.md`).

Pins the three ported conventions:
1. `mean_iou` is the mean over **GT classes** — unmatched GT classes count as IoU 0 (the
   old vendored mean-over-matched-pairs behavior inflated scores when cluster counts
   differed).
2. Background-label exclusion is **explicit only** (`bg_label=None` default => no
   exclusion); `"auto"` remains available as an opt-in legacy heuristic; any exclusion is
   recorded in `bg_label_excluded`.
3. Floater reporting: `n_pred` still counts the -1 label (ARI scoring unchanged), but the
   honest counts `n_pred_nonfloater` / `n_floater_points` are reported alongside.
"""

from __future__ import annotations

import numpy as np

from pipeline.vendored.host.seg_eval import evaluate


def _four_class_scene(n_per_class: int = 25):
    """4 balanced GT classes; pred_points == gt_points so NN propagation is exact."""
    rng = np.random.RandomState(0)
    gt_labels = np.repeat(np.arange(4), n_per_class)
    # Well-separated class centroids so points are unambiguous.
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    gt_points = centers[gt_labels] + rng.randn(len(gt_labels), 3) * 0.01
    return gt_points.astype(np.float32), gt_labels


def test_mean_iou_counts_unmatched_gt_classes_as_zero():
    """4 balanced GT classes vs 1 predicted cluster: the single Hungarian match has
    IoU 0.25 (its class covers 1/4 of the union == the whole cloud); the 3 unmatched GT
    classes contribute IoU 0 -> mean_iou = 0.25/4 = 0.0625, NOT 0.25 (old behavior)."""
    gt_points, gt_labels = _four_class_scene()
    pred_points = gt_points
    pred_labels = np.zeros(len(gt_points), dtype=int)  # one cluster for everything

    result = evaluate(pred_points, pred_labels, gt_points, gt_labels)

    assert result["n_gt"] == 4
    assert len(result["matches"]) == 1
    assert result["matches"][0][2] == np.float64(0.25) or abs(result["matches"][0][2] - 0.25) < 1e-9
    assert abs(result["mean_iou"] - 0.0625) < 1e-9


def test_bg_label_default_is_no_exclusion():
    """GT labels 0 and 1 coexist — the OLD vendored heuristic would auto-exclude label 0
    and compute ari_within_roi here. The synced code must do neither by default."""
    rng = np.random.RandomState(1)
    gt_labels = np.repeat([0, 1], 30)
    centers = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    gt_points = centers[gt_labels] + rng.randn(len(gt_labels), 3) * 0.01
    pred_labels = gt_labels.copy()

    result = evaluate(gt_points, pred_labels, gt_points, gt_labels)

    assert result["bg_label_excluded"] is None
    assert "ari_within_roi" not in result
    assert "n_roi_points" not in result


def test_explicit_bg_label_excludes_and_records():
    """bg_label=0 excludes GT label 0 from the ROI score and records it loudly in the
    result dict; 'auto' opts back into the legacy heuristic."""
    rng = np.random.RandomState(1)
    gt_labels = np.repeat([0, 1], 30)
    centers = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    gt_points = centers[gt_labels] + rng.randn(len(gt_labels), 3) * 0.01
    pred_labels = gt_labels.copy()

    result = evaluate(gt_points, pred_labels, gt_points, gt_labels, bg_label=0)
    assert result["bg_label_excluded"] == 0
    assert result["n_roi_points"] == 30  # only the label-1 points remain in the ROI
    assert abs(result["ari_within_roi"] - 1.0) < 1e-9  # perfect within the ROI

    result_auto = evaluate(gt_points, pred_labels, gt_points, gt_labels, bg_label="auto")
    assert result_auto["bg_label_excluded"] == 0
    assert result_auto["n_roi_points"] == 30


def test_floater_reporting_fields():
    """-1 floaters stay in `n_pred` and in ARI scoring, but are honestly reported via
    `n_pred_nonfloater` / `n_floater_points`."""
    rng = np.random.RandomState(2)
    gt_labels = np.repeat([0, 1], 30)
    centers = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    gt_points = centers[gt_labels] + rng.randn(len(gt_labels), 3) * 0.01
    pred_labels = gt_labels.copy()
    pred_labels[:5] = -1  # five floater points

    result = evaluate(gt_points, pred_labels, gt_points, gt_labels, bg_label=1)

    assert result["n_pred"] == 3            # {-1, 0, 1} — floater still counted (ARI unchanged)
    assert result["n_pred_nonfloater"] == 2
    assert result["n_floater_points"] == 5
    assert result["bg_label_excluded"] == 1
