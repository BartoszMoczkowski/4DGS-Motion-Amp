"""
[DEPRECATED / ARCHIVED — T33]
This ad-hoc batch driver has been retired.
Its workflow is now executed declaratively via:
    amp exp run experiments/capture_limitation.yaml

Retained for historical audit and reference only.
"""

"""
batch_seg_capture_limitation.py — Batch motion segmentation and evaluation across all 18 cells
of the Capture-Limitation suite (Series A, R, T, F).

Pipeline stages per cell:
1. Verify / load run from runs/capture_limitation_trained_runs.json.
2. Auto-seed / verify gt_segmentation from data/multipleview/{dataset}/gt_segmentation.npz.
3. Extract trajectories (GPU): seg_extract.default (n_times=120 for sR5, 60 for all others).
4. Option B baseline clustering (CPU): segment.rigid.
5. Evaluation (CPU): seg_eval.default with bg_label=0 (computes ARI + Within-ROI ARI).
6. FFT-denoised clustering (CPU): segment.rigid2.
7. Evaluation of rigid2 (CPU): seg_eval.default with bg_label=0.
8. Extract standardized evaluation metrics:
   - Overall ARI & ROI ARI (excluding static frame_base)
   - Mean IoU & Macro F1
   - Cat A Split Rate (contact < 0.2 mm, 39 pairs)
   - Cat B Split Rate (clearance gap >= 0.2 mm, 102 pairs)
   - 141 Baseline Pairs Split Rate
   - Deviant target part (part_008, label 8): IoU, Precision, Recall, F1, matched cluster
   - Cluster counts (n_pred, non-floater clusters, floater count)
9. Output results to:
   - runs/capture_limitation_seg_results.json
   - runs/capture_limitation_seg_results.csv
   - runs/experiment_results.csv (via pipeline.experiments.results.export_csv)
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from pipeline import api
from pipeline.artifacts import Artifact, load_manifest, update_manifest
from pipeline.experiments.results import export_csv
from pipeline.vendored.host.metrics import (
    PUMP01_BASELINE_MERGED_PAIRS_141,
    adjusted_rand_index,
    compute_pairs_split_rate,
)
from pipeline.vendored.host.seg_eval import propagate_labels

MAPPING_FILE = Path("runs/capture_limitation_trained_runs.json")
RESULTS_JSON = Path("runs/capture_limitation_seg_results.json")
RESULTS_CSV = Path("runs/capture_limitation_seg_results.csv")

TARGET_PART_LABEL = 8  # part_008 in pump01
CAT_A_PAIRS = PUMP01_BASELINE_MERGED_PAIRS_141[:39]
CAT_B_PAIRS = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

CELLS = [
    # Series A (Motion Amplitude Sweep)
    {"id": "sA0", "dataset": "pump01_sA0", "series": "A", "n_times": 60, "desc": "baseline pastel, common-mode non-target, target static"},
    {"id": "sA1", "dataset": "pump01_sA1", "series": "A", "n_times": 60, "desc": "deviant target 2x base motion (|B-A|=1A shear)"},
    {"id": "sA2", "dataset": "pump01_sA2", "series": "A", "n_times": 60, "desc": "deviant target 5x base motion (|B-A|=4A shear)"},
    {"id": "sA3", "dataset": "pump01_sA3", "series": "A", "desc": "deviant target 10x base motion (|B-A|=9A shear)", "n_times": 60},
    {"id": "sA4", "dataset": "pump01_sA4", "series": "A", "desc": "deviant target 20x base motion (|B-A|=19A shear)", "n_times": 60},
    {"id": "sA5", "dataset": "pump01_sA5", "series": "A", "desc": "target only motion, non-target static", "n_times": 60},

    # Series R (Resolution & Distance Sweep)
    {"id": "sR0", "dataset": "pump01_sR0", "series": "R", "desc": "baseline reference (1600x900, R=420cm)", "n_times": 60},
    {"id": "sR1", "dataset": "pump01_sR1", "series": "R", "desc": "res 2560x1440, R=420cm", "n_times": 60},
    {"id": "sR2", "dataset": "pump01_sR2", "series": "R", "desc": "res 3000x2000 (6 MP), R=420cm", "n_times": 60},
    {"id": "sR3", "dataset": "pump01_sR3", "series": "R", "desc": "dist half (1600x900, R=210cm)", "n_times": 60},
    {"id": "sR4", "dataset": "pump01_sR4", "series": "R", "desc": "res 3k + dist half (3000x2000, R=210cm)", "n_times": 60},
    {"id": "sR5", "dataset": "pump01_sR5", "series": "R", "desc": "res 3k + dist half + 120 frames", "n_times": 120},

    # Series T (Material Realism Ablation)
    {"id": "sT0", "dataset": "pump01_sT0", "series": "T", "desc": "pastel displayColors baseline", "n_times": 60},
    {"id": "sT1", "dataset": "pump01_sT1", "series": "T", "desc": "uniform brushed steel metallic", "n_times": 60},
    {"id": "sT2", "dataset": "pump01_sT2", "series": "T", "desc": "metallic + procedural surface texture", "n_times": 60},

    # Series F (Compound Realistic Cells)
    {"id": "sF0", "dataset": "pump01_sF0", "series": "F", "desc": "compound baseline control (A0 x T0 x R0)", "n_times": 60},
    {"id": "sF1", "dataset": "pump01_sF1", "series": "F", "desc": "compound honest (A3 10x motion x T1 metallic x R4 capture)", "n_times": 60},
    {"id": "sF2", "dataset": "pump01_sF2", "series": "F", "desc": "compound honest + texture (A3 10x x T2 texture x R4 capture)", "n_times": 60},
]


def evaluate_target_part(gt_on_pred: np.ndarray, pred_labels: np.ndarray, target_label: int = TARGET_PART_LABEL) -> Dict[str, Any]:
    """Compute precision, recall, F1, and IoU for target part (part_008) against predicted clusters."""
    gt_mask = (gt_on_pred == target_label)
    gt_count = int(gt_mask.sum())
    if gt_count == 0:
        return {
            "target_gt_points": 0,
            "target_matched_cluster": -1,
            "target_iou": 0.0,
            "target_precision": 0.0,
            "target_recall": 0.0,
            "target_f1": 0.0,
            "target_detected": False,
        }

    clusters = np.unique(pred_labels)
    clusters = clusters[clusters != -1]  # Exclude floater cluster -1

    best_iou = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_f1 = 0.0
    best_cluster = -1

    for c in clusters:
        pred_mask = (pred_labels == c)
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = float(inter / union) if union > 0 else 0.0
        prec = float(inter / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
        rec = float(inter / gt_count) if gt_count > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best_prec = prec
            best_rec = rec
            best_f1 = f1
            best_cluster = int(c)

    return {
        "target_gt_points": gt_count,
        "target_matched_cluster": best_cluster,
        "target_iou": round(best_iou, 6),
        "target_precision": round(best_prec, 6),
        "target_recall": round(best_rec, 6),
        "target_f1": round(best_f1, 6),
        "target_detected": bool(best_iou >= 0.10),
    }


def compute_extended_metrics(run_id: str, impl: str) -> Dict[str, Any]:
    """Load the evaluated segmentation and compute Cat A/B split rates and target part metrics."""
    manifest = load_manifest(run_id)
    pred_path = Path(manifest.artifacts["segmentation"].path)
    gt_path = Path(manifest.artifacts["gt_segmentation"].path)
    eval_json_key = f"seg_eval_result_{impl}"
    eval_json_path = Path(manifest.artifacts[eval_json_key].path)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    pred = np.load(pred_path)
    gt = np.load(gt_path)

    pred_points = pred["points"]
    pred_labels = pred["labels"]
    gt_points = gt["points"]
    gt_labels = gt["labels"]

    gt_on_pred = propagate_labels(gt_points, gt_labels, pred_points)

    # Cat A split rate (39 pairs)
    cat_a_res = compute_pairs_split_rate(gt_on_pred, pred_labels, merged_pairs=CAT_A_PAIRS)
    # Cat B split rate (102 pairs)
    cat_b_res = compute_pairs_split_rate(gt_on_pred, pred_labels, merged_pairs=CAT_B_PAIRS)
    # Total baseline pairs split rate (141 pairs)
    cat_141_res = compute_pairs_split_rate(gt_on_pred, pred_labels, merged_pairs=PUMP01_BASELINE_MERGED_PAIRS_141)

    # Target part metrics
    target_res = evaluate_target_part(gt_on_pred, pred_labels, target_label=TARGET_PART_LABEL)

    return {
        "ari": eval_data.get("ari", 0.0),
        "ari_within_roi": eval_data.get("ari_within_roi", 0.0),
        "mean_iou": eval_data.get("mean_iou", 0.0),
        "mean_part_iou": eval_data.get("mean_part_iou", 0.0),
        "f1_macro": eval_data.get("f1_macro", 0.0),
        "n_gt": eval_data.get("n_gt", 0),
        "n_pred": eval_data.get("n_pred", 0),
        "n_pred_nonfloater": eval_data.get("n_pred_nonfloater", 0),
        "n_floater_points": eval_data.get("n_floater_points", 0),
        "k_alignment": eval_data.get("k_alignment", 0.0),
        "cat_a_split_rate": cat_a_res["pairs_split_rate"],
        "n_cat_a_split": cat_a_res["n_pairs_split"],
        "cat_b_split_rate": cat_b_res["pairs_split_rate"],
        "n_cat_b_split": cat_b_res["n_pairs_split"],
        "pairs_split_rate_141": cat_141_res["pairs_split_rate"],
        "n_pairs_split_141": cat_141_res["n_pairs_split"],
        **target_res,
    }


def main():
    if not MAPPING_FILE.exists():
        print(f"Error: {MAPPING_FILE} not found. Please train models first.", file=sys.stderr)
        sys.exit(1)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        trained_runs = json.load(f)

    print("=" * 80)
    print("CAPTURE-LIMITATION BATCH MOTION SEGMENTATION & EVALUATION")
    print(f"Total cells to evaluate: {len(CELLS)}")
    print("=" * 80)

    # Load existing results if resuming
    all_results = {}
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    IMPLS = ["rigid", "rigid2"]

    for idx, cell_info in enumerate(CELLS, 1):
        cid = cell_info["id"]
        run_id = trained_runs.get(cid)
        if not run_id:
            print(f"[{idx}/{len(CELLS)}] Warning: No trained run found for cell {cid}, skipping.")
            continue

        dataset = cell_info["dataset"]
        series = cell_info["series"]
        n_times = cell_info["n_times"]
        desc = cell_info["desc"]

        print(f"\n[{idx}/{len(CELLS)}] Cell {cid} (Series {series}) | run_id: {run_id}")
        print(f"       Desc: {desc}")

        # 1. Trajectory Extraction (seg_extract.default)
        manifest = load_manifest(run_id)
        if "trajectories" not in manifest.artifacts:
            print(f"       -> Running seg_extract.default (n_times={n_times})...")
            t0 = time.time()
            api.run_stage(run_id, "seg_extract.default", overrides={"seg_extract": {"n_times": n_times}})
            print(f"       -> Extracted trajectories in {time.time() - t0:.1f}s")
        else:
            print("       -> Trajectories already extracted (cached).")

        if cid not in all_results:
            all_results[cid] = {
                "cell": cid,
                "series": series,
                "dataset": dataset,
                "desc": desc,
                "run_id": run_id,
                "impls": {},
            }

        # 2. Run Segmentation & Evaluation for each implementation
        for impl in IMPLS:
            print(f"       -> Segmentation: {impl} ...")
            eval_key = f"seg_eval_result_{impl}"
            manifest = load_manifest(run_id)

            # Check if already evaluated
            if eval_key not in manifest.artifacts or impl not in all_results[cid]["impls"]:
                t_seg0 = time.time()
                api.run_stage(run_id, f"segment.{impl}")
                print(f"          {impl} clustering finished in {time.time() - t_seg0:.1f}s")

                t_eval0 = time.time()
                api.run_stage(run_id, "seg_eval.default", overrides={"seg_eval": {"bg_label": 0}})
                print(f"          seg_eval ({impl}) finished in {time.time() - t_eval0:.1f}s")

            # Compute extended metrics
            metrics = compute_extended_metrics(run_id, impl)
            all_results[cid]["impls"][impl] = metrics

            print(f"          [Result {impl}] ARI: {metrics['ari']:.4f} | ROI ARI: {metrics['ari_within_roi']:.4f} | "
                  f"CatA Split: {metrics['cat_a_split_rate']:.3f} | CatB Split: {metrics['cat_b_split_rate']:.3f} | "
                  f"Target IoU: {metrics['target_iou']:.4f} (Det: {metrics['target_detected']})")

        # Save progress after each cell
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # 3. Export flat CSV
    csv_rows = []
    for cid, data in all_results.items():
        base = {
            "cell": cid,
            "series": data["series"],
            "dataset": data["dataset"],
            "desc": data["desc"],
            "run_id": data["run_id"],
        }
        for impl, m in data["impls"].items():
            row = dict(base)
            row["impl"] = impl
            row.update(m)
            csv_rows.append(row)

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved CSV results to {RESULTS_CSV}")

    # Rebuild orchestrator experiment store
    try:
        export_csv()
        print("Updated runs/experiment_results.csv successfully.")
    except Exception as e:
        print(f"Warning: could not export to experiment_results.csv: {e}")

    print("\nBatch segmentation and evaluation complete across all cells!")


if __name__ == "__main__":
    main()
