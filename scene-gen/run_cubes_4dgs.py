#!/usr/bin/env python3
"""Run the captured spinning cubes scenes (k=1..8) through 4DGS reconstruction via the orchestrator.

Runs the full 4DGS DAG (convert.default -> train.default -> render.default)
inside the CUDA container environment on the GPU.

Usage (from the repo root):
    python scene-gen/run_cubes_4dgs.py                  # runs k=1..8 sequentially
    python scene-gen/run_cubes_4dgs.py --k 1 2 3        # runs only k=1,2,3
    python scene-gen/run_cubes_4dgs.py --smoke          # runs a fast 100-iter smoke run on k=1

Results are appended to `runs/cubes_4dgs_results.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.api import _stage_config_for
from pipeline.artifacts import Artifact, create_run, update_manifest
from pipeline.config import validate_config
from pipeline.dag import run_dag

RENDERS_ROOT = Path("Q:/Omniverse/renders")
RESULTS_CSV = REPO_ROOT / "runs" / "cubes_4dgs_results.csv"
STAGES = ["convert.default", "train.default", "render.default"]


def base_resolved(cell: str, smoke: bool = False) -> dict:
    resolved = validate_config("cubes").model_dump()
    resolved["convert"]["name"] = cell
    resolved["render"]["skip_train"] = True
    resolved["render"]["skip_test"] = True
    resolved["train"]["expname"] = f"multipleview/{cell}"

    if smoke:
        resolved["optim"]["coarse_iterations"] = 50
        resolved["optim"]["iterations"] = 100
        resolved["train"]["test_iterations"] = []
        resolved["train"]["save_iterations"] = []
    return resolved


def count_gaussians(run_dir: Path) -> int | None:
    plys = sorted(run_dir.glob("train_out/point_cloud/iteration_*/point_cloud.ply"))
    if not plys:
        return None
    with open(plys[-1], "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return None


def already_done(run_id: str) -> bool:
    manifest_path = REPO_ROOT / "runs" / run_id / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        return False
    return manifest.get("stages", {}).get("render.default", {}).get("status") in ("success", "skipped")


def run_one(run_id: str, cell: str, capture_dir: Path, resolved: dict, force: bool = False) -> None:
    if not force and already_done(run_id):
        print(f"[skip] {run_id} already finished")
        return
    print(f"\n================================================================================")
    print(f"[run] Starting 4DGS pipeline for {run_id} (capture={capture_dir})")
    print(f"================================================================================\n")
    import shutil
    run_dir = REPO_ROOT / "runs" / run_id
    if run_dir.is_dir():
        try:
            shutil.rmtree(run_dir)
        except Exception as e:
            print(f"[warn] could not clean old run dir: {e}")
    create_run(run_id, "cubes", resolved, stage_names=STAGES)
    update_manifest(
        run_id,
        lambda m: m.artifacts.update(
            {"capture": Artifact(name="capture", kind="dataset", path=str(capture_dir), producing_stage="external")}
        ),
    )
    error = ""
    try:
        manifest = run_dag(
            run_id,
            STAGES,
            resolved,
            preset="cubes",
            force=True,
            stage_configs={name: _stage_config_for(name, resolved) for name in STAGES},
        )
        status = manifest.status
        if status != "success":
            error = "; ".join(
                f"{n}: {manifest.stages[n].error}" for n in STAGES if manifest.stages[n].status == "failed"
            )
    except Exception as exc:
        manifest = None
        status = "exception"
        error = repr(exc)
        print(f"[FAIL] {run_id}: {error}")

    stages = manifest.stages if manifest else {}
    row = {
        "run_id": run_id,
        "cell": cell,
        "actual_gaussians": count_gaussians(REPO_ROOT / "runs" / run_id) or "",
        "status": status,
        "convert_s": getattr(stages.get("convert.default"), "wall_time_s", "") or "",
        "train_s": getattr(stages.get("train.default"), "wall_time_s", "") or "",
        "render_s": getattr(stages.get("render.default"), "wall_time_s", "") or "",
        "train_peak_vram_mb": getattr(stages.get("train.default"), "peak_vram_mb", "") or "",
        "error": error,
    }
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[done] {run_id}: {status} gaussians={row['actual_gaussians']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="List of k cube counts to train (default: 1 2 3 4 5 6 7 8)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 100-iter smoke run on k=1 to validate the pipeline",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running even if run_id was already finished",
    )
    args = parser.parse_args()

    if args.smoke:
        k = 1
        cell = f"spinning_cubes_k{k}"
        capture_dir = RENDERS_ROOT / f"capture_{cell}"
        run_id = f"cubes-k{k}-smoke"
        run_one(run_id, cell, capture_dir, base_resolved(cell, smoke=True), force=args.force)
        return

    for k in args.k:
        # Search for matching capture folders for k (e.g. capture_cubes_k2_both, capture_cubes_k2_rotation, capture_spinning_cubes_k2)
        candidates = sorted(RENDERS_ROOT.glob(f"capture_*k{k}*"))
        if not candidates:
            print(f"[warning] No capture directory found matching k={k} in {RENDERS_ROOT}, skipping")
            continue
        for capture_dir in candidates:
            cell = capture_dir.name.replace("capture_", "")
            run_id = f"cubes-k{k}" if cell == f"spinning_cubes_k{k}" else f"cubes-{cell}"
            run_one(run_id, cell, capture_dir, base_resolved(cell, smoke=False), force=args.force)


if __name__ == "__main__":
    main()
