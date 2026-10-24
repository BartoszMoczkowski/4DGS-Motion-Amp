"""
[DEPRECATED / ARCHIVED — T33]
This ad-hoc batch driver has been retired.
Its workflow is now executed declaratively via:
    amp exp run experiments/capture_limitation.yaml

Retained for historical audit and reference only.
"""

"""
batch_capture_series_tf.py — Sequential batch capture and conversion for Series T and Series F.

Handles:
- sT0: baseline pastel (linked from sR0, converted to pump01_sT0)
- sF0: baseline pastel common-mode (linked from sA0, converted to pump01_sF0)
- sT1: metallic material ablation (1600x900, R=420cm, 60f)
- sT2: metallic + procedural texture ablation (1600x900, R=420cm, 60f)
- sF1: compound honest cell (A3 10x deviant motion, T1 metallic, R4 3000x2000, R=210cm, 60f)
- sF2: compound honest textured cell (A3 10x deviant, T2 textured, R4 3000x2000, R=210cm, 60f)

Writes live progress to runs/batch_capture_series_tf_status.json.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STATUS_FILE = Path("runs/batch_capture_series_tf_status.json")

EQUIVALENCE_CELLS = [
    {
        "id": "sT0",
        "name": "pump01_sT0_mat_pastel",
        "source_capture_dir": "Q:/Omniverse/renders/capture_pump01_sR0_baseline_1600x900",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sT0_mat_pastel",
        "dataset_name": "pump01_sT0",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sF0",
        "name": "pump01_sF0_combined_baseline",
        "source_capture_dir": "Q:/Omniverse/renders/capture_pump01_sA0_common_target_static",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sF0_combined_baseline",
        "dataset_name": "pump01_sF0",
        "cams": 10,
        "frames": 60,
    },
]

RENDER_CELLS = [
    {
        "id": "sT1",
        "name": "pump01_sT1_mat_metallic",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sT1_mat_metallic.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sT1_mat_metallic",
        "dataset_name": "pump01_sT1",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sT2",
        "name": "pump01_sT2_mat_metallic_texture",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sT2_mat_metallic_texture.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sT2_mat_metallic_texture",
        "dataset_name": "pump01_sT2",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sF1",
        "name": "pump01_sF1_combined_honest",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sF1_combined_honest.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sF1_combined_honest",
        "dataset_name": "pump01_sF1",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sF2",
        "name": "pump01_sF2_combined_honest_textured",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sF2_combined_honest_textured.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sF2_combined_honest_textured",
        "dataset_name": "pump01_sF2",
        "cams": 10,
        "frames": 60,
    },
]


def update_status(status_dict: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> [{desc}] Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f">>> [{desc}] Finished in {dt:.1f}s (exit {res.returncode})")


def is_already_captured(capture_dir: str, n_cams: int, n_frames: int) -> bool:
    cap_path = Path(capture_dir)
    if not cap_path.exists():
        return False
    for c in range(n_cams):
        cam_dir = cap_path / f"cam{c:02d}"
        if not cam_dir.exists():
            return False
        pngs = list(cam_dir.glob("*.png"))
        if len(pngs) < n_frames:
            return False
    return True


def link_or_copy_dir(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        print(f"Destination {dst_path} already exists.")
        return
    print(f"Creating junction from {src_path} to {dst_path}...")
    try:
        import _winapi
        _winapi.CreateJunction(str(src_path), str(dst_path))
        print(f"Junction successfully created: {dst_path} -> {src_path}")
    except Exception as e:
        print(f"Junction creation failed ({e}), falling back to symlink / copy...")
        try:
            os.symlink(str(src_path), str(dst_path), target_is_directory=True)
        except Exception as e2:
            print(f"Symlink failed ({e2}), falling back to robocopy...")
            subprocess.run(["robocopy", str(src_path), str(dst_path), "/E", "/NFL", "/NDL", "/NJH", "/NJS"], check=False)


def main():
    print(f"=== Starting Batch Capture & Conversion for Series T & Series F ===")
    overall_start = time.time()
    status_record = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "RUNNING",
        "cells": {},
    }
    update_status(status_record)

    # Phase 1: Equivalence cells (sT0 and sF0)
    print("\n" + "="*70 + "\nPHASE 1: Processing Equivalence Cells (sT0, sF0)\n" + "="*70)
    for cell in EQUIVALENCE_CELLS:
        cid = cell["id"]
        cname = cell["name"]
        print(f"\n--- Setting up {cid} ({cname}) from {cell['source_capture_dir']} ---")
        t_start = time.time()
        try:
            link_or_copy_dir(cell["source_capture_dir"], cell["capture_dir"])

            # Verify
            run_cmd(
                ["uv", "run", "amp", "capture", "verify", cell["capture_dir"],
                 "--cams", str(cell["cams"]), "--frames", str(cell["frames"])],
                f"{cid} Frame Verification",
            )

            # Convert to multipleview dataset
            run_cmd(
                ["uv", "run", "amp", "capture", "convert", "--capture", cell["capture_dir"],
                 "--out", ".", "--name", cell["dataset_name"]],
                f"{cid} 4DGS Dataset Conversion",
            )

            # Video preview
            run_cmd(
                ["uv", "run", "amp", "capture", "preview", "--capture", cell["capture_dir"], "--tile-cols", "5"],
                f"{cid} Video Preview",
            )

            dt = time.time() - t_start
            status_record["cells"][cid] = f"OK ({dt:.1f}s, reused {Path(cell['source_capture_dir']).name})"
            print(f"[SUCCESS] {cid} ready in {dt:.1f}s!")
        except Exception as e:
            print(f"ERROR processing {cid}: {e}", file=sys.stderr)
            status_record["cells"][cid] = f"FAILED ({e})"
        update_status(status_record)

    # Phase 2: Active Render cells (sT1, sT2, sF1, sF2)
    print("\n" + "="*70 + "\nPHASE 2: Active Isaac Sim Captures (sT1, sT2, sF1, sF2)\n" + "="*70)
    for i, cell in enumerate(RENDER_CELLS, 1):
        cid = cell["id"]
        cname = cell["name"]
        print(f"\n{'='*70}\n[{i}/{len(RENDER_CELLS)}] Processing Cell {cid}: {cname}\n{'='*70}")
        t_start = time.time()

        # 1. Capture (skip if already completed)
        if is_already_captured(cell["capture_dir"], cell["cams"], cell["frames"]):
            print(f"Cell {cid} already has full capture in {cell['capture_dir']}. Skipping capture run.")
        else:
            try:
                run_cmd(
                    ["uv", "run", "amp", "capture", "run", "--config", cell["yaml"]],
                    f"{cid} Isaac Sim Capture",
                )
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Capture failed for {cid}: {e}", file=sys.stderr)
                status_record["cells"][cid] = f"FAILED capture ({e})"
                update_status(status_record)
                continue

        # 2. Verify
        try:
            run_cmd(
                ["uv", "run", "amp", "capture", "verify", cell["capture_dir"],
                 "--cams", str(cell["cams"]), "--frames", str(cell["frames"])],
                f"{cid} Frame Verification",
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Verification failed for {cid}: {e}", file=sys.stderr)

        # 3. Convert to multipleview dataset
        try:
            run_cmd(
                ["uv", "run", "amp", "capture", "convert", "--capture", cell["capture_dir"],
                 "--out", ".", "--name", cell["dataset_name"]],
                f"{cid} 4DGS Dataset Conversion",
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Convert failed for {cid}: {e}", file=sys.stderr)
            status_record["cells"][cid] = f"FAILED convert ({e})"
            update_status(status_record)
            continue

        # 4. Preview video
        try:
            run_cmd(
                ["uv", "run", "amp", "capture", "preview", "--capture", cell["capture_dir"], "--tile-cols", "5"],
                f"{cid} Video Preview",
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Preview generation failed for {cid}: {e}", file=sys.stderr)

        total_dt = time.time() - t_start
        status_record["cells"][cid] = f"OK ({total_dt/60:.1f} min)"
        update_status(status_record)
        print(f"[SUCCESS] Cell {cid} fully captured, verified, and converted in {total_dt/60:.1f} min!")

    total_pipeline_time = time.time() - overall_start
    status_record["status"] = "COMPLETED"
    status_record["total_time_min"] = round(total_pipeline_time / 60, 1)
    update_status(status_record)

    print("\n" + "="*70 + "\nBATCH CAPTURE SUMMARY: SERIES T & SERIES F\n" + "="*70)
    for cid, st in status_record["cells"].items():
        print(f"{cid}: {st}")
    print(f"Total time: {total_pipeline_time/60:.1f} min")


if __name__ == "__main__":
    main()
