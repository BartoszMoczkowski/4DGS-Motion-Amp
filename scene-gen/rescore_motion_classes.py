#!/usr/bin/env python3
"""
rescore_motion_classes.py — Execute each segmentation implementation DAG and re-score outputs
against both the 107-label CAD GT and the 94-label ground-truth motion classes.

Emits:
- runs/grid_seg_motionclass_rescore.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.api import _stage_config_for
from pipeline.artifacts import Artifact, load_manifest, update_manifest
from pipeline.config import validate_config
from pipeline.dag import run_dag
from pipeline.vendored.host.metrics import adjusted_rand_index, best_iou_matching

SEGMENT_STAGE: dict[str, str] = {
    "rigid": "segment.rigid",
    "mbs": "segment.mbs",
    "rigid2": "segment.rigid2",
    "kabsch": "segment.kabsch",
    "rigid2_roi": "segment.rigid2",
    "mask_lift_oracle": "segment.rigid2",
    "mask_lift": "segment.rigid2",
}

STAGES = {
    "rigid": ["seg_extract.default", "segment.rigid", "seg_eval.default"],
    "mbs": ["segment.mbs", "seg_eval.default"],
    "rigid2": ["segment.rigid2", "seg_eval.default"],
    "kabsch": ["segment.kabsch", "seg_eval.default"],
    "rigid2_roi": ["roi.motion_gate", "segment.rigid2", "seg_eval.default"],
    "mask_lift_oracle": ["roi.mask_oracle", "segment.rigid2", "seg_eval.default"],
    "mask_lift": ["roi.mask_lift", "segment.rigid2", "seg_eval.default"],
}

PRESET = {
    "rigid": "pump01",
    "mbs": "pump01_segA",
    "rigid2": "pump01_segB2",
    "kabsch": "pump01_kabsch",
    "rigid2_roi": "pump01_roi_gate",
    "mask_lift_oracle": "pump01_mask_oracle",
    "mask_lift": "pump01_mask_lift",
}

RUN_IDS = [
    "grid-A20mm_M2",
    "grid-A20mm_M4",
    "grid-A40mm_M8",
    "sweep-A40mm_M8-g10000",
    "sweep-A40mm_M8-g25000",
    "sweep-A40mm_M8-g50000",
    "sweep-A40mm_M8-g100000",
]


def seed_gt(run_id: str) -> Path:
    manifest = load_manifest(run_id)
    scene = Path(manifest.artifacts["scene"].path)
    gt = scene / "gt_segmentation.npz"
    if "gt_segmentation" not in manifest.artifacts:
        if not gt.is_file():
            raise FileNotFoundError(f"{run_id}: no gt_segmentation.npz under {scene}")
        update_manifest(
            run_id,
            lambda m: m.artifacts.update(
                {"gt_segmentation": Artifact(name="gt_segmentation", kind="npz",
                                             path=str(gt), producing_stage="external")}
            ),
        )
    return gt


def propagate_labels(src_points: np.ndarray, src_labels: np.ndarray, dst_points: np.ndarray, tree: cKDTree) -> np.ndarray:
    _, nn = tree.query(dst_points, k=1)
    return src_labels[nn]


def main():
    gt_seg_path = REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_segmentation.npz"
    gt_motion_path = REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_motion_classes.npy"

    if not gt_motion_path.exists():
        raise FileNotFoundError(f"Missing GT motion classes file at {gt_motion_path}. Run build_gt_motion_classes.py first!")

    gt_data = np.load(gt_seg_path)
    gt_points = gt_data["points"]
    gt_cad_labels = gt_data["labels"]
    gt_motion_labels = np.load(gt_motion_path)

    print("[rescore] Building KDTree on 100k GT points...", flush=True)
    tree = cKDTree(gt_points)

    impls = ["rigid", "mbs", "rigid2", "kabsch", "rigid2_roi", "mask_lift_oracle"]

    header = [
        "impl", "run_id", "status",
        "n_gt_cad", "n_gt_motion", "n_pred",
        "ari_107", "ari_motion",
        "mean_iou_107", "mean_iou_motion",
        "ari_within_roi_107", "ari_within_roi_motion",
        "n_roi_points"
    ]

    out_csv_path = REPO_ROOT / "runs" / "grid_seg_motionclass_rescore.csv"
    rescore_rows = []

    print(f"[rescore] Executing DAG runs and re-scoring across {len(impls)} implementations...\n", flush=True)

    for impl in impls:
        preset_name = PRESET[impl]
        stages = STAGES[impl]
        seg_stage = SEGMENT_STAGE[impl]

        print(f"\n==================================================", flush=True)
        print(f"   Implementation: {impl} (preset: {preset_name})", flush=True)
        print(f"==================================================", flush=True)

        for run_id in RUN_IDS:
            resolved = validate_config(preset_name).model_dump()
            gt_path = seed_gt(run_id)

            if impl in ("rigid2", "rigid2_roi", "mask_lift_oracle", "mask_lift"):
                resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)

            print(f"[run_dag] Executing {run_id} ({impl})...", flush=True)
            try:
                manifest = run_dag(
                    run_id, stages, resolved, preset=preset_name, force=True,
                    stage_configs={name: _stage_config_for(name, resolved) for name in stages}
                )
                status = manifest.status
            except Exception as exc:
                print(f"[FAIL] {run_id} ({impl}): {exc}", flush=True)
                status = "failed"

            if status != "success":
                print(f"[skip_eval] {run_id} ({impl}) status={status}", flush=True)
                continue

            run_dir = REPO_ROOT / "runs" / run_id
            seg_npz = run_dir / "segmentation.npz"
            if not seg_npz.exists():
                print(f"[WARN] missing segmentation.npz for {run_id}", flush=True)
                continue

            pred_data = np.load(seg_npz)
            pred_points = pred_data["points"]
            pred_labels = pred_data["labels"]

            gt_cad_on_pred = propagate_labels(gt_points, gt_cad_labels, pred_points, tree)
            gt_motion_on_pred = propagate_labels(gt_points, gt_motion_labels, pred_points, tree)

            ari_107 = adjusted_rand_index(gt_cad_on_pred, pred_labels)
            ari_motion = adjusted_rand_index(gt_motion_on_pred, pred_labels)

            mean_iou_107, _ = best_iou_matching(gt_cad_on_pred, pred_labels)
            mean_iou_motion, _ = best_iou_matching(gt_motion_on_pred, pred_labels)

            n_pred = int(len(np.unique(pred_labels)))
            n_gt_cad = int(len(np.unique(gt_cad_labels)))
            n_gt_motion = int(len(np.unique(gt_motion_labels)))

            roi_mask_npz = run_dir / "roi_mask.npz"
            ari_within_roi_107 = None
            ari_within_roi_motion = None
            n_roi_points = None

            if roi_mask_npz.exists():
                roi_data = np.load(roi_mask_npz)
                in_roi = roi_data.get("in_roi", roi_data.get("mask", None))
                if in_roi is not None and len(in_roi) == len(pred_points) and in_roi.any():
                    ari_within_roi_107 = adjusted_rand_index(gt_cad_on_pred[in_roi], pred_labels[in_roi])
                    ari_within_roi_motion = adjusted_rand_index(gt_motion_on_pred[in_roi], pred_labels[in_roi])
                    n_roi_points = int(in_roi.sum())

            row = {
                "impl": impl,
                "run_id": run_id,
                "status": status,
                "n_gt_cad": n_gt_cad,
                "n_gt_motion": n_gt_motion,
                "n_pred": n_pred,
                "ari_107": f"{ari_107:.6f}",
                "ari_motion": f"{ari_motion:.6f}",
                "mean_iou_107": f"{mean_iou_107:.6f}",
                "mean_iou_motion": f"{mean_iou_motion:.6f}",
                "ari_within_roi_107": f"{ari_within_roi_107:.6f}" if ari_within_roi_107 is not None else "",
                "ari_within_roi_motion": f"{ari_within_roi_motion:.6f}" if ari_within_roi_motion is not None else "",
                "n_roi_points": str(n_roi_points) if n_roi_points is not None else ""
            }
            rescore_rows.append(row)
            print(f"  --> {run_id:25s} | Pred: {n_pred:3d} clusters | ARI_107: {ari_107:+.4f} | ARI_motion: {ari_motion:+.4f}", flush=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rescore_rows)

    print(f"\n[rescore] Successfully generated {out_csv_path} with {len(rescore_rows)} rows!", flush=True)


if __name__ == "__main__":
    main()
