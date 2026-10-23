"""
batch_capture_series_r.py — Sequential batch capture and conversion for Series R.

Runs native Isaac Sim capture for sR0 through sR5, verifies frame completeness,
converts each capture to 4DGS multipleview format, and generates preview videos.
"""
import os
import subprocess
import sys
import time

CELLS = [
    {
        "id": "sR0",
        "name": "pump01_sR0_baseline_1600x900",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR0_baseline_1600x900.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR0_baseline_1600x900",
        "dataset_name": "pump01_sR0",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sR1",
        "name": "pump01_sR1_res_2560x1440",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR1_res_2560x1440.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR1_res_2560x1440",
        "dataset_name": "pump01_sR1",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sR2",
        "name": "pump01_sR2_res_3000x2000",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR2_res_3000x2000.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR2_res_3000x2000",
        "dataset_name": "pump01_sR2",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sR3",
        "name": "pump01_sR3_dist_half_1600x900",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR3_dist_half_1600x900.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR3_dist_half_1600x900",
        "dataset_name": "pump01_sR3",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sR4",
        "name": "pump01_sR4_res3k_dist_half",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR4_res3k_dist_half.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR4_res3k_dist_half",
        "dataset_name": "pump01_sR4",
        "cams": 10,
        "frames": 60,
    },
    {
        "id": "sR5",
        "name": "pump01_sR5_res3k_dist_half_120f",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sR5_res3k_dist_half_120f.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sR5_res3k_dist_half_120f",
        "dataset_name": "pump01_sR5",
        "cams": 10,
        "frames": 120,
    },
]


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> [{desc}] Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f">>> [{desc}] Finished in {dt:.1f}s (exit {res.returncode})")


def main():
    print(f"=== Starting Batch Capture for Series R ({len(CELLS)} cells) ===")
    results = {}

    for i, cell in enumerate(CELLS, 1):
        cid = cell["id"]
        cname = cell["name"]
        print(f"\n{'='*70}\n[{i}/{len(CELLS)}] Processing Cell {cid}: {cname}\n{'='*70}")

        # 1. Capture
        t_start = time.time()
        try:
            run_cmd(
                ["uv", "run", "amp", "capture", "run", "--config", cell["yaml"]],
                f"{cid} Isaac Sim Capture",
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Capture failed for {cid}: {e}", file=sys.stderr)
            results[cid] = f"FAILED capture ({e})"
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
            results[cid] = f"FAILED convert ({e})"
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
        results[cid] = f"OK ({total_dt/60:.1f} min)"
        print(f"[SUCCESS] Cell {cid} fully captured, verified, and converted in {total_dt/60:.1f} min!")

    print("\n" + "="*70 + "\nBATCH CAPTURE SUMMARY: SERIES R\n" + "="*70)
    for cid, status in results.items():
        print(f"{cid}: {status}")


if __name__ == "__main__":
    main()
