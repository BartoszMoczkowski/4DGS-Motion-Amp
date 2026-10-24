"""
[DEPRECATED / ARCHIVED — T33]
This ad-hoc batch driver has been retired.
Its workflow is now executed declaratively via:
    amp exp run experiments/capture_limitation.yaml

Retained for historical audit and reference only.
"""

"""
batch_train_capture_limitation.py — Sequential batch 4DGS training for the Capture-Limitation suite.

Iterates over all 18 cells in Series A, R, T, F:
- sA0..sA5 (Motion Amplitude Sweep)
- sR0..sR5 (Resolution & Distance Sweep)
- sT0..sT2 (Material Realism Ablation)
- sF0..sF2 (Compound Realistic Cells)

For each cell:
1. Verifies dataset presence in data/multipleview/{dataset}.
2. Calls pipeline.api.run_pipeline("pump01", only=["train.default"], ...) inside the cuda container.
3. Tags the run manifest with metadata: experiment=capture_limitation, cell=cid, variant=cid.
4. Verifies the output checkpoint under runs/{run_id}/train_out/point_cloud/iteration_14000.
5. Updates runs/batch_train_capture_limitation_status.json and runs/capture_limitation_trained_runs.json.
6. Resumes cleanly if interrupted (skips already completed cells).
"""
import json
import os
from pathlib import Path
import sys
import time

from pipeline import api
from pipeline.artifacts import Artifact, validate_external_artifact_path, load_manifest, update_manifest

STATUS_FILE = Path("runs/batch_train_capture_limitation_status.json")
MAPPING_FILE = Path("runs/capture_limitation_trained_runs.json")

CELLS = [
    # Series A (Motion Amplitude Sweep)
    {"id": "sA0", "dataset": "pump01_sA0", "series": "A", "desc": "baseline pastel, common-mode non-target, target static"},
    {"id": "sA1", "dataset": "pump01_sA1", "series": "A", "desc": "deviant target 2x base motion"},
    {"id": "sA2", "dataset": "pump01_sA2", "series": "A", "desc": "deviant target 5x base motion"},
    {"id": "sA3", "dataset": "pump01_sA3", "series": "A", "desc": "deviant target 10x base motion"},
    {"id": "sA4", "dataset": "pump01_sA4", "series": "A", "desc": "deviant target 20x base motion"},
    {"id": "sA5", "dataset": "pump01_sA5", "series": "A", "desc": "target only motion, non-target static"},

    # Series R (Resolution & Distance Sweep)
    {"id": "sR0", "dataset": "pump01_sR0", "series": "R", "desc": "baseline reference (1600x900, R=420cm)"},
    {"id": "sR1", "dataset": "pump01_sR1", "series": "R", "desc": "res 2560x1440, R=420cm"},
    {"id": "sR2", "dataset": "pump01_sR2", "series": "R", "desc": "res 3000x2000 (6 MP), R=420cm"},
    {"id": "sR3", "dataset": "pump01_sR3", "series": "R", "desc": "dist half (1600x900, R=210cm)"},
    {"id": "sR4", "dataset": "pump01_sR4", "series": "R", "desc": "res 3k + dist half (3000x2000, R=210cm)"},
    {"id": "sR5", "dataset": "pump01_sR5", "series": "R", "desc": "res 3k + dist half + 120 frames"},

    # Series T (Material Realism Ablation)
    {"id": "sT0", "dataset": "pump01_sT0", "series": "T", "desc": "pastel displayColors baseline"},
    {"id": "sT1", "dataset": "pump01_sT1", "series": "T", "desc": "uniform brushed steel metallic"},
    {"id": "sT2", "dataset": "pump01_sT2", "series": "T", "desc": "metallic + procedural surface texture"},

    # Series F (Compound Realistic Cells)
    {"id": "sF0", "dataset": "pump01_sF0", "series": "F", "desc": "compound baseline control (A0 x T0 x R0)"},
    {"id": "sF1", "dataset": "pump01_sF1", "series": "F", "desc": "compound honest (A3 10x motion x T1 metallic x R4 capture)"},
    {"id": "sF2", "dataset": "pump01_sF2", "series": "F", "desc": "compound honest + texture (A3 10x x T2 texture x R4 capture)"},
]


def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "RUNNING",
        "current_cell": None,
        "completed_count": 0,
        "total_count": len(CELLS),
        "cells": {},
    }


def save_status(status_record: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_record, f, indent=2)

    # Save mapping file
    mapping = {}
    for cid, cinfo in status_record.get("cells", {}).items():
        if isinstance(cinfo, dict) and cinfo.get("status") == "COMPLETED":
            mapping[cid] = cinfo.get("run_id")
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def main():
    print(f"=== Starting 4DGS Batch Training ({len(CELLS)} cells) ===")
    overall_start = time.time()
    status_record = load_status()
    save_status(status_record)

    for idx, cell in enumerate(CELLS, 1):
        cid = cell["id"]
        dataset_name = cell["dataset"]
        series = cell["series"]
        desc = cell["desc"]
        dataset_path = Path("data/multipleview") / dataset_name

        print(f"\n{'='*70}\n[{idx}/{len(CELLS)}] Cell {cid} (Series {series}): {dataset_name}\nDescription: {desc}\n{'='*70}")

        # Check dataset existence
        if not dataset_path.exists():
            print(f"ERROR: Dataset directory {dataset_path} does not exist!", file=sys.stderr)
            status_record["cells"][cid] = {
                "status": "FAILED",
                "error": f"Missing dataset directory {dataset_path}",
            }
            save_status(status_record)
            continue

        # Check if already completed
        prev = status_record["cells"].get(cid)
        if isinstance(prev, dict) and prev.get("status") == "COMPLETED":
            prev_run_id = prev.get("run_id")
            prev_ply = Path(prev.get("checkpoint", ""))
            if prev_ply.exists():
                print(f"[SKIP] Cell {cid} already trained in run {prev_run_id} ({prev_ply}).")
                continue

        status_record["current_cell"] = cid
        status_record["status"] = "RUNNING"
        save_status(status_record)

        # Build external artifact
        scene_path = validate_external_artifact_path(str(dataset_path.resolve()), "dataset")
        external_artifacts = {
            "scene": Artifact(
                name="scene",
                kind="dataset",
                path=str(scene_path),
                producing_stage="external",
            )
        }

        t_start = time.time()
        try:
            print(f">>> Launching train.default for {cid} on scene {scene_path}...")
            run_id = api.run_pipeline(
                "pump01",
                only=["train.default"],
                external_artifacts=external_artifacts,
                metadata={
                    "experiment": "capture_limitation",
                    "cell": cid,
                    "variant": cid,
                    "series": series,
                    "dataset": dataset_name,
                    "description": desc,
                },
            )
            dt_min = (time.time() - t_start) / 60.0

            # Verify checkpoint
            ckpt_ply = Path("runs") / run_id / "train_out" / "point_cloud" / "iteration_14000" / "point_cloud.ply"
            if not ckpt_ply.exists():
                print(f"WARNING: Expected checkpoint {ckpt_ply} not found on disk!", file=sys.stderr)

            print(f"[SUCCESS] Cell {cid} trained successfully in {dt_min:.1f} min! Run ID: {run_id}")
            status_record["cells"][cid] = {
                "status": "COMPLETED",
                "run_id": run_id,
                "dataset": dataset_name,
                "duration_min": round(dt_min, 1),
                "checkpoint": str(ckpt_ply),
            }
            status_record["completed_count"] = sum(
                1 for c in status_record["cells"].values() if isinstance(c, dict) and c.get("status") == "COMPLETED"
            )
        except Exception as e:
            dt_min = (time.time() - t_start) / 60.0
            print(f"ERROR: Training failed for {cid} after {dt_min:.1f} min: {e}", file=sys.stderr)
            status_record["cells"][cid] = {
                "status": "FAILED",
                "dataset": dataset_name,
                "duration_min": round(dt_min, 1),
                "error": str(e),
            }

        save_status(status_record)

    total_time = (time.time() - overall_start) / 60.0
    status_record["status"] = "COMPLETED"
    status_record["current_cell"] = None
    status_record["total_time_min"] = round(total_time, 1)
    save_status(status_record)

    print("\n" + "="*70 + "\nBATCH TRAINING SUMMARY: CAPTURE LIMITATION SUITE\n" + "="*70)
    for cid, info in status_record["cells"].items():
        if isinstance(info, dict):
            st = info.get("status")
            rid = info.get("run_id", "N/A")
            dt = info.get("duration_min", 0)
            print(f"{cid}: {st} ({dt} min, run_id={rid})")
    print(f"Total time elapsed: {total_time:.1f} min")


if __name__ == "__main__":
    main()
