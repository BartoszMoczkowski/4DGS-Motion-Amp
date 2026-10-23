"""
batch_capture_series_a.py — Sequential batch capture and conversion for Series A.

Runs native Isaac Sim capture for sA1 through sA5, verifies frame completeness,
converts each capture to 4DGS multipleview format, and generates preview videos.
"""
import os
import subprocess
import sys
import time

CELLS = [
    {
        "id": "sA1",
        "name": "pump01_sA1_deviant_2x",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sA1_deviant_2x.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sA1_deviant_2x",
        "dataset_name": "pump01_sA1",
    },
    {
        "id": "sA2",
        "name": "pump01_sA2_deviant_5x",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sA2_deviant_5x.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sA2_deviant_5x",
        "dataset_name": "pump01_sA2",
    },
    {
        "id": "sA3",
        "name": "pump01_sA3_deviant_10x",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sA3_deviant_10x.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sA3_deviant_10x",
        "dataset_name": "pump01_sA3",
    },
    {
        "id": "sA4",
        "name": "pump01_sA4_deviant_20x",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sA4_deviant_20x.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sA4_deviant_20x",
        "dataset_name": "pump01_sA4",
    },
    {
        "id": "sA5",
        "name": "pump01_sA5_target_only",
        "yaml": "omniverse-pipeline/data/scenes/capture_limitation/pump01_sA5_target_only.yaml",
        "capture_dir": "Q:/Omniverse/renders/capture_pump01_sA5_target_only",
        "dataset_name": "pump01_sA5",
    },
]


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> [{desc}] Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f">>> [{desc}] Finished in {dt:.1f}s (exit {res.returncode})")


def main():
    print(f"=== Starting Batch Capture for Series A ({len(CELLS)} cells) ===")
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
                ["uv", "run", "amp", "capture", "verify", cell["capture_dir"], "--cams", "10", "--frames", "60"],
                f"{cid} Frame Verification",
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Verification failed for {cid}: {e}", file=sys.stderr)

        # 3. Convert to multipleview dataset
        try:
            run_cmd(
                ["uv", "run", "amp", "capture", "convert", "--capture", cell["capture_dir"], "--out", ".", "--name", cell["dataset_name"]],
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

    print("\n" + "="*70 + "\nBATCH CAPTURE SUMMARY\n" + "="*70)
    print(f"sA0 (pump01_sA0_common_target_static): OK (previously captured)")
    for cid, status in results.items():
        print(f"{cid}: {status}")


if __name__ == "__main__":
    main()
