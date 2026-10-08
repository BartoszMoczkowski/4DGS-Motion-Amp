#!/usr/bin/env python3
"""Run all motion segmentation algorithms over the trained spinning cubes 4DGS models (k=1..8).

Supported backends:
  - rigid: Baseline Option B rigidity-graph clustering (segment.rigid)
  - rigid2: T18 upgraded Option B with FFT denoising + calibrated z-scores (segment.rigid2)
  - kabsch: T20 iterative Kabsch EM rigid-body clustering (segment.kabsch)
  - rigid2_roi: T19 motion-gated ROI + rigid2 (roi.motion_gate -> segment.rigid2)
  - mask_lift_oracle: T22 oracle GT mask ROI ceiling + rigid2 (roi.mask_oracle -> segment.rigid2)
  - mbs: Option A MultiBodySync MotNet (segment.mbs)
  - all: Runs all backends across k=1..8 and compiles a master comparison report

Usage:
  python scene-gen/run_cubes_seg.py --impl all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.api import _stage_config_for
from pipeline.artifacts import Artifact, load_manifest, update_manifest
from pipeline.config import validate_config
from pipeline.dag import run_dag

SEGMENT_STAGE: dict[str, str] = {
    "rigid": "segment.rigid",
    "rigid2": "segment.rigid2",
    "kabsch": "segment.kabsch",
    "rigid2_roi": "segment.rigid2",
    "mask_lift_oracle": "segment.rigid2",
    "mbs": "segment.mbs",
    "multicut": "segment.multicut",
}

RESULTS_CSV: dict[str, Path] = {
    "rigid": REPO_ROOT / "runs" / "cubes_seg_rigid_results.csv",
    "rigid2": REPO_ROOT / "runs" / "cubes_seg_rigid2_results.csv",
    "kabsch": REPO_ROOT / "runs" / "cubes_seg_kabsch_results.csv",
    "rigid2_roi": REPO_ROOT / "runs" / "cubes_seg_rigid2_roi_results.csv",
    "mask_lift_oracle": REPO_ROOT / "runs" / "cubes_seg_mask_lift_oracle_results.csv",
    "mbs": REPO_ROOT / "runs" / "cubes_seg_mbs_results.csv",
    "multicut": REPO_ROOT / "runs" / "cubes_seg_multicut_results.csv",
}

SUMMARY_CSV = REPO_ROOT / "runs" / "cubes_seg_summary.csv"

STAGES: dict[str, list[str]] = {
    "rigid": ["segment.rigid", "seg_eval.default"],
    "rigid2": ["segment.rigid2", "seg_eval.default"],
    "kabsch": ["segment.kabsch", "seg_eval.default"],
    "rigid2_roi": ["roi.motion_gate", "segment.rigid2", "seg_eval.default"],
    "mask_lift_oracle": ["roi.mask_oracle", "segment.rigid2", "seg_eval.default"],
    "mbs": ["segment.mbs", "seg_eval.default"],
    "multicut": ["segment.multicut", "seg_eval.default"],
}

PRESET: dict[str, str] = {
    "rigid": "cubes",
    "rigid2": "cubes_segB2",
    "kabsch": "cubes_kabsch",
    "rigid2_roi": "cubes_roi_gate",
    "mask_lift_oracle": "cubes_mask_oracle",
    "mbs": "cubes_segA",
    "multicut": "cubes_multicut",
}


def ensure_trajectories(run_id: str) -> None:
    """Extract deformation trajectories if not present and register artifact in manifest."""
    run_dir = REPO_ROOT / "runs" / run_id
    traj_path = run_dir / "trajectories.npz"
    if not traj_path.is_file():
        print(f"[seg_extract] Extracting dense trajectories for {run_id}...")
        resolved = validate_config("cubes").model_dump()
        stages = ["seg_extract.default"]
        run_dag(
            run_id,
            stages,
            resolved,
            preset="cubes",
            force=True,
            stage_configs={name: _stage_config_for(name, resolved) for name in stages},
        )
        print(f"[seg_extract] Trajectories extracted -> {traj_path}")

    manifest = load_manifest(run_id)
    if "trajectories" not in manifest.artifacts:
        update_manifest(
            run_id,
            lambda m: m.artifacts.update(
                {"trajectories": Artifact(name="trajectories", kind="npz", path=str(traj_path), producing_stage="seg_extract.default")}
            ),
        )


def seed_gt(run_id: str) -> Path:
    """Add gt_segmentation artifact to manifest if missing."""
    manifest = load_manifest(run_id)
    scene = Path(manifest.artifacts["scene"].path)
    gt = scene / "gt_segmentation.npz"
    if "gt_segmentation" not in manifest.artifacts:
        if not gt.is_file():
            raise FileNotFoundError(f"{run_id}: no gt_segmentation.npz under {scene}")
        update_manifest(
            run_id,
            lambda m: m.artifacts.update(
                {"gt_segmentation": Artifact(name="gt_segmentation", kind="npz", path=str(gt), producing_stage="external")}
            ),
        )
    return gt


def run_one(run_id: str, k: int, impl: str) -> dict:
    ensure_trajectories(run_id)
    gt_path = seed_gt(run_id)
    stages = STAGES[impl]
    run_dir = REPO_ROOT / "runs" / run_id
    eval_path = run_dir / "seg_eval_result.json"

    resolved = validate_config(PRESET[impl]).model_dump()
    resolved["seg_eval"]["recolored_ply"] = f"segmentation_colored_{impl}.ply"

    # For rigid2-based impls, provide GT for separability diagnostics
    if impl in ("rigid2", "rigid2_roi", "mask_lift_oracle"):
        resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)

    # For kabsch, set n_clusters to k + 1 (k rotating cubes + 1 static floor)
    if impl == "kabsch":
        resolved["segment"]["kabsch"]["n_clusters"] = k + 1

    print(f"\n--- Running {impl} segmentation on {run_id} (k={k} cubes) ---")
    error = ""
    manifest = None
    try:
        manifest = run_dag(
            run_id,
            stages,
            resolved,
            preset=PRESET[impl],
            force=True,
            stage_configs={name: _stage_config_for(name, resolved) for name in stages},
        )
        status = manifest.status
        if status != "success":
            error = "; ".join(
                f"{n}: {manifest.stages[n].error}" for n in stages if manifest.stages[n].status == "failed"
            )
    except Exception as exc:
        status = "exception"
        error = repr(exc)
        print(f"[FAIL] {run_id} ({impl}): {error}")

    stages_rec = manifest.stages if manifest else {}
    summary = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
    seg_stage = SEGMENT_STAGE[impl]

    sep_path = run_dir / "separability.json"
    sep_auroc = ""
    if sep_path.is_file():
        try:
            sep_data = json.loads(sep_path.read_text(encoding="utf-8"))
            sep_auroc = sep_data.get("denoised_z", {}).get("auroc", "")
        except Exception:
            pass

    row = {
        "run_id": run_id,
        "k": k,
        "impl": impl,
        "status": status,
        "ari": summary.get("ari", ""),
        "ari_within_roi": summary.get("ari_within_roi", ""),
        "mean_iou": summary.get("mean_iou", ""),
        "n_gt": summary.get("n_gt", ""),
        "n_pred": summary.get("n_pred", ""),
        "separability_auroc": sep_auroc,
        "n_roi_points": summary.get("n_roi_points", ""),
        "segment_s": getattr(stages_rec.get(seg_stage), "wall_time_s", "") or "",
        "error": error,
    }

    # Save to backend-specific CSV
    csv_file = RESULTS_CSV[impl]
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file.exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)

    # Save to master summary CSV
    write_summary_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_summary_header:
            w.writeheader()
        w.writerow(row)

    print(f"[done] {run_id} ({impl}): {status} | ARI={row['ari']} | Mean IoU={row['mean_iou']} | Clusters: {row['n_pred']}/{row['n_gt']}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="List of k cube counts (default: 1..8)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run_id in runs/ to segment",
    )
    parser.add_argument(
        "--impl",
        choices=["all", "rigid", "rigid2", "kabsch", "rigid2_roi", "mask_lift_oracle", "mbs", "multicut"],
        default="all",
        help="Segmentation backend to run (default: all)",
    )
    args = parser.parse_args()

    impls = (
        ["rigid", "rigid2", "kabsch", "rigid2_roi", "mask_lift_oracle", "mbs", "multicut"]
        if args.impl == "all"
        else [args.impl]
    )

    # Clean old summary CSV if running all from scratch
    if args.impl == "all" and args.k == [1, 2, 3, 4, 5, 6, 7, 8] and not args.run_id and SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    if args.run_id:
        target_runs = [(args.run_id, args.k[0])]
    else:
        target_runs = []
        for k in args.k:
            runs_dir = REPO_ROOT / "runs"
            candidates = sorted(runs_dir.glob(f"cubes-*k{k}*"))
            if not candidates:
                print(f"[warning] Run directory not found for k={k}, skipping.")
            else:
                for c in candidates:
                    target_runs.append((c.name, k))

    for run_id, k in target_runs:
        run_dir = REPO_ROOT / "runs" / run_id
        if not run_dir.is_dir():
            print(f"[warning] Run directory not found: {run_dir}, skipping.")
            continue
        for impl in impls:
            run_one(run_id, k, impl)


if __name__ == "__main__":
    main()
