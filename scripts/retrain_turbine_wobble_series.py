#!/usr/bin/env python3
"""scripts/retrain_turbine_wobble_series.py

Task T42: Retraining driver for the turbine wobble series:
  - lego_turbine_base_r9 (0 missing blades: baseline intact)
  - lego_turbine_2_missing_r9 (2 missing blades: minor imbalance)
  - lego_turbine_5_missing_r9 (5 missing blades: severe imbalance)
  - turbine_r1_missing_6_r9 (6 missing blades: extreme imbalance)

Each run executes `train.default` using the full R9 structured motion stack,
anchoring the scene's `offsets.json` hash in the run manifest metadata and
seeding external artifacts (`scene`, `offsets`).

Usage:
  uv run python scripts/retrain_turbine_wobble_series.py --dry-run
  uv run python scripts/retrain_turbine_wobble_series.py --scene lego_turbine_2_missing_r9
  uv run python scripts/retrain_turbine_wobble_series.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import api
from pipeline.artifacts import Artifact, validate_external_artifact_path

TURBINE_SERIES = [
    {
        "preset": "lego_turbine_base_r9",
        "scene_name": "lego_turbine_base",
        "description": "Retrain baseline intact turbine under full R9 Fourier-rigid stack",
    },
    {
        "preset": "lego_turbine_2_missing_r9",
        "scene_name": "lego_turbine_2_missing",
        "description": "Retrain 2-missing blades turbine under full R9 Fourier-rigid stack",
    },
    {
        "preset": "lego_turbine_5_missing_r9",
        "scene_name": "lego_turbine_5_missing",
        "description": "Retrain 5-missing blades turbine under full R9 Fourier-rigid stack",
    },
    {
        "preset": "turbine_r1_missing_6_r9",
        "scene_name": "turbine_r1_missing_6",
        "description": "Retrain 6-missing blades turbine under full R9 Fourier-rigid stack",
    },
]

STATUS_FILE = REPO_ROOT / "runs" / "turbine_retrain_status.json"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrain_scene(
    preset_name: str,
    scene_name: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    scene_dir = REPO_ROOT / "data" / "multipleview" / scene_name
    offsets_file = scene_dir / "offsets.json"

    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not offsets_file.exists():
        raise FileNotFoundError(f"offsets.json not found in scene directory: {offsets_file}")

    offsets_hash = hash_file(offsets_file)
    scene_path_valid = validate_external_artifact_path(str(scene_dir.resolve()), "dataset")
    offsets_path_valid = validate_external_artifact_path(str(offsets_file.resolve()), "json")

    external_artifacts = {
        "scene": Artifact(
            name="scene",
            kind="dataset",
            path=str(scene_path_valid),
            producing_stage="external",
        ),
        "offsets": Artifact(
            name="offsets",
            kind="json",
            path=str(offsets_path_valid),
            producing_stage="external",
            content_hash=f"sha256:{offsets_hash}",
        ),
    }

    metadata = {
        "experiment": "turbine_wobble_recovery",
        "task": "T42",
        "preset": preset_name,
        "scene": scene_name,
        "offsets_json_hash": offsets_hash,
        "offsets_json_path": str(offsets_file.resolve()),
    }

    resolved = api.validate_config(preset_name)

    print(f"\n=========================================================================")
    print(f"  RETRAINING: {preset_name}")
    print(f"  Scene:      {scene_dir.relative_to(REPO_ROOT)}")
    print(f"  Offsets:    {offsets_file.name} (SHA-256: {offsets_hash[:16]}...)")
    print(f"  Fourier:    {resolved.get('optim', {}).get('use_fourier_rigid_head')} @ {resolved.get('optim', {}).get('fourier_head_frequency_hz')} Hz")
    print(f"=========================================================================")

    if dry_run:
        print(f"[dry-run] Would execute run_pipeline('{preset_name}', only=['train.default'])")
        print(f"[dry-run] Manifest metadata: {metadata}")
        return {
            "preset": preset_name,
            "scene": scene_name,
            "offsets_hash": offsets_hash,
            "status": "dry_run",
            "run_id": f"{preset_name}-dryrun",
        }

    t0 = time.time()
    try:
        run_id = api.run_pipeline(
            preset_name,
            only=["train.default"],
            external_artifacts=external_artifacts,
            metadata=metadata,
            force=force,
        )
        elapsed = time.time() - t0
        print(f"--> SUCCESS: {preset_name} finished in {elapsed/60.0:.1f} min; run_id: {run_id}")
        return {
            "preset": preset_name,
            "scene": scene_name,
            "offsets_hash": offsets_hash,
            "status": "success",
            "run_id": run_id,
            "wall_time_s": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"--> FAILED: {preset_name} after {elapsed/60.0:.1f} min: {e}")
        return {
            "preset": preset_name,
            "scene": scene_name,
            "offsets_hash": offsets_hash,
            "status": "failed",
            "error": str(e),
            "wall_time_s": round(elapsed, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Retrain turbine wobble series under R9 full stack.")
    parser.add_argument("--all", action="store_true", help="Retrain all 4 turbine scenes in sequence.")
    parser.add_argument("--scene", type=str, default="", help="Specific preset to retrain (e.g. lego_turbine_2_missing_r9).")
    parser.add_argument("--dry-run", action="store_true", help="Print resolution and manifests without training.")
    parser.add_argument("--force", action="store_true", help="Force retrain ignoring existing cached artifacts.")
    args = parser.parse_args()

    targets = []
    if args.scene:
        matches = [s for s in TURBINE_SERIES if s["preset"] == args.scene or s["scene_name"] == args.scene]
        if not matches:
            print(f"Error: Unknown scene/preset {args.scene}. Choose from: {[s['preset'] for s in TURBINE_SERIES]}")
            sys.exit(1)
        targets = matches
    elif args.all:
        targets = TURBINE_SERIES
    else:
        # Default to dry-run preview if no action specified
        targets = TURBINE_SERIES
        args.dry_run = True
        print("[INFO] No target specified; running dry-run across all 4 scenes.")

    results = []
    for item in targets:
        res = retrain_scene(item["preset"], item["scene_name"], dry_run=args.dry_run, force=args.force)
        results.append(res)

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "runs": results}, f, indent=2)

    print("\n=========================================================================")
    print("  RETRAINING BATCH SUMMARY")
    print("=========================================================================")
    for r in results:
        status = r.get("status", "unknown")
        run_id = r.get("run_id", "N/A")
        print(f"  {r['preset']:<30} -> {status.upper():<10} (run_id: {run_id})")
    print(f"Saved tracking status to {STATUS_FILE.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
