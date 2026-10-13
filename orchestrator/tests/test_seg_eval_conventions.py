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


# --- T27 Phase A: per-impl seg_eval / segmentation keying ----------------------------------------


def _two_part_fixture(tmp_path):
    """Static base + one sinusoidally rotating part, T=12 — tiny, just enough for both
    segment.rigid and segment.kabsch to run on."""
    rng = np.random.RandomState(0)
    T = 12
    base = rng.uniform(-1, 1, size=(80, 3))
    part = np.array([3.0, 0.0, 0.0]) + rng.uniform(-0.3, 0.3, size=(60, 3))
    xyz = np.concatenate([base, part])

    times = np.linspace(0.0, 1.0, T, endpoint=False)
    centroid = part.mean(axis=0)
    rel = part - centroid
    part_traj = np.empty((len(part), T, 3))
    for ti, t in enumerate(times):
        theta = np.deg2rad(20) * np.sin(2 * np.pi * 2 * t)
        cos, sin = np.cos(theta), np.sin(theta)
        rotated = np.stack(
            [rel[:, 0] * cos - rel[:, 1] * sin, rel[:, 0] * sin + rel[:, 1] * cos, rel[:, 2]],
            axis=1,
        )
        part_traj[:, ti, :] = centroid + rotated
    traj = np.concatenate([np.repeat(base[:, None, :], T, axis=1), part_traj])
    gt_labels = np.repeat([0, 1], [len(base), len(part)])

    traj_path = tmp_path / "trajectories.npz"
    np.savez(traj_path, canonical_xyz=xyz, traj=traj)
    gt_path = tmp_path / "gt_segmentation.npz"
    np.savez(gt_path, points=xyz.astype(np.float32), labels=gt_labels)
    return traj_path, gt_path


def test_two_impls_in_one_run_produce_distinct_results(tmp_path):
    """T27 acceptance: two different segment impls evaluated in ONE run must produce two
    distinct segmentation files, two distinct seg_eval result files, and two distinct manifest
    artifacts — no overwrite (the old fixed names forced run_grid_seg.py's _backup_eval dance)."""
    import json

    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.dag import run_dag

    traj_path, gt_path = _two_part_fixture(tmp_path)
    run_id = "t27-twoimpl"
    create_run(run_id, "t27", {}, stage_names=[], runs_root=tmp_path)

    def _seed(m):
        m.artifacts["trajectories"] = Artifact(
            name="trajectories", kind="npz", path=str(traj_path), producing_stage="external"
        )
        m.artifacts["gt_segmentation"] = Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        )

    update_manifest(run_id, _seed, runs_root=tmp_path)

    run_dir = tmp_path / run_id
    stage_configs = {
        # Near-noiseless synthetic scene: huge threshold_mult keeps all edges (same trick as
        # test_stages_cpu — the auto-threshold is fragile on noiseless fixtures).
        "segment.rigid": {"k": 6, "min_size": 5, "threshold_mult": 1e7},
        "segment.kabsch": {"n_clusters": 2, "max_iter": 5, "min_size": 5},
        "seg_eval.default": {},
    }

    m1 = run_dag(
        run_id, ["segment.rigid", "seg_eval.default"], {}, preset="t27",
        stage_configs=stage_configs, runs_root=tmp_path,
    )
    assert m1.status == "success"
    assert m1.artifacts["segmentation"].producing_stage == "segment.rigid"
    assert "seg_eval_result_rigid" in m1.artifacts
    assert (run_dir / "segmentation_rigid.npz").is_file()
    assert (run_dir / "seg_eval_result_rigid.json").is_file()

    m2 = run_dag(
        run_id, ["segment.kabsch", "seg_eval.default"], {}, preset="t27",
        stage_configs=stage_configs, runs_root=tmp_path,
    )
    assert m2.status == "success"

    # Two distinct files each for segmentation and eval results — nothing overwritten.
    for name in (
        "segmentation_rigid.npz", "segmentation_kabsch.npz",
        "seg_eval_result_rigid.json", "seg_eval_result_kabsch.json",
    ):
        assert (run_dir / name).is_file(), f"missing {name}"
    assert "seg_eval_result_rigid" in m2.artifacts
    assert "seg_eval_result_kabsch" in m2.artifacts
    assert (
        m2.artifacts["seg_eval_result_rigid"].path
        != m2.artifacts["seg_eval_result_kabsch"].path
    )
    # The rigid result survived the kabsch phase unchanged.
    rigid_summary = json.loads((run_dir / "seg_eval_result_rigid.json").read_text())
    assert rigid_summary["n_gt"] == 2
    # The shared "segmentation" DAG-contract key now points at the latest impl's file.
    assert m2.artifacts["segmentation"].producing_stage == "segment.kabsch"
    assert m2.artifacts["segmentation"].path.endswith("segmentation_kabsch.npz")


def test_seg_eval_falls_back_to_fixed_name_for_external_segmentation(tmp_path):
    """Back-compat: a `segmentation` artifact NOT produced by a segment.* stage (externally
    seeded) keeps the legacy fixed seg_eval_result.json / seg_eval_result names."""
    import json

    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.dag import run_dag

    rng = np.random.RandomState(0)
    gt_labels = np.repeat([0, 1], 30)
    centers = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    pts = (centers[gt_labels] + rng.randn(len(gt_labels), 3) * 0.01).astype(np.float32)
    seg_path = tmp_path / "some_segmentation.npz"
    np.savez(seg_path, points=pts, labels=gt_labels)
    gt_path = tmp_path / "gt.npz"
    np.savez(gt_path, points=pts, labels=gt_labels)

    run_id = "t27-extseg"
    create_run(run_id, "t27", {}, stage_names=[], runs_root=tmp_path)

    def _seed(m):
        m.artifacts["segmentation"] = Artifact(
            name="segmentation", kind="npz", path=str(seg_path), producing_stage="external"
        )
        m.artifacts["gt_segmentation"] = Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        )

    update_manifest(run_id, _seed, runs_root=tmp_path)
    m = run_dag(
        run_id, ["seg_eval.default"], {}, preset="t27",
        stage_configs={"seg_eval.default": {}}, runs_root=tmp_path,
    )
    assert m.status == "success"
    assert (tmp_path / run_id / "seg_eval_result.json").is_file()
    assert "seg_eval_result" in m.artifacts
    summary = json.loads((tmp_path / run_id / "seg_eval_result.json").read_text())
    assert abs(summary["ari"] - 1.0) < 1e-9


def test_m6_k_alignment_metric():
    from pipeline.vendored.host.metrics import k_alignment

    assert k_alignment(107, 107) == 1.0
    assert k_alignment(50, 100) == 0.5
    assert k_alignment(200, 100) == 0.5
    assert k_alignment(0, 50) == 0.0
    assert k_alignment(50, 0) == 0.0


def test_m6_hungarian_part_iou_and_stratified_f1():
    from pipeline.vendored.host.metrics import hungarian_part_iou, size_stratified_f1

    # 4 classes: class 0 (bg, 100 pts), class 1 (small, 10 pts), class 2 (med, 30 pts), class 3 (large, 60 pts)
    gt_labels = np.concatenate([
        np.zeros(100, dtype=int),
        np.ones(10, dtype=int),
        np.full(30, 2, dtype=int),
        np.full(60, 3, dtype=int),
    ])
    # Perfect prediction for class 3, partial for class 2, zero for class 1
    pred_labels = np.concatenate([
        np.zeros(100, dtype=int),
        np.zeros(10, dtype=int),  # class 1 merged with bg
        np.full(15, 2, dtype=int), np.zeros(15, dtype=int),  # class 2 split in half
        np.full(60, 3, dtype=int),  # class 3 perfect
    ])

    part_iou = hungarian_part_iou(gt_labels, pred_labels, bg_label=0)
    assert "mean_part_iou" in part_iou
    assert "median_part_iou" in part_iou
    assert "part_iou_small" in part_iou
    assert "part_iou_medium" in part_iou
    assert "part_iou_large" in part_iou
    assert part_iou["part_iou_small"] == 0.0
    assert part_iou["part_iou_large"] == 1.0

    f1_stats = size_stratified_f1(gt_labels, pred_labels, bg_label=0)
    assert "f1_macro" in f1_stats
    assert "f1_small" in f1_stats
    assert "f1_medium" in f1_stats
    assert "f1_large" in f1_stats
    assert f1_stats["f1_small"] == 0.0
    assert f1_stats["f1_large"] == 1.0
    assert f1_stats["f1_macro"] > 0.0


def test_m6_compute_pairs_split_rate():
    from pipeline.vendored.host.metrics import compute_pairs_split_rate

    gt_labels = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    # Pred: part 1 and 2 merged into cluster 10; part 3 and 4 in separate clusters 20 and 30
    pred_labels = np.array([10, 10, 10, 10, 20, 20, 30, 30])
    pairs = [(1, 2), (3, 4)]

    res = compute_pairs_split_rate(gt_labels, pred_labels, merged_pairs=pairs)
    assert res["n_pairs_merged_base"] == 2
    assert res["n_pairs_split"] == 1  # pair (3,4) is split; (1,2) is merged
    assert res["pairs_split_rate"] == 0.5


def test_m6_evaluate_emits_standard_metrics():
    from pipeline.vendored.host.seg_eval import evaluate

    gt_points = np.arange(40)[:, None].repeat(3, axis=1).astype(np.float32)
    gt_labels = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int), np.full(10, 2, dtype=int), np.full(10, 3, dtype=int)])
    pred_points = gt_points
    pred_labels = gt_labels.copy()

    res = evaluate(pred_points, pred_labels, gt_points, gt_labels, bg_label=0)
    expected_keys = [
        "k_alignment", "k_ratio", "mean_part_iou", "median_part_iou",
        "part_iou_small", "part_iou_medium", "part_iou_large",
        "f1_macro", "f1_small", "f1_medium", "f1_large",
        "pairs_split_rate", "n_pairs_split", "n_pairs_merged_base",
    ]
    for k in expected_keys:
        assert k in res, f"missing {k} in evaluate() result"
    assert res["k_alignment"] == 1.0
    assert res["f1_macro"] == 1.0
    assert res["mean_part_iou"] == 1.0

