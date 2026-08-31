#!/usr/bin/env python3
"""
run_cubes_capture.py — batch runner for Isaac Sim synthetic multi-camera captures of spinning cubes scenes.

Executes Isaac Sim's native Python runtime against omni_capture.py for each generated
spinning cubes scene (k=1..8).

Usage:
    python scene-gen/run_cubes_capture.py                  # captures scenes from manifest or k=1..8
    python scene-gen/run_cubes_capture.py --k 1 2 3        # captures only k=1,2,3
    python scene-gen/run_cubes_capture.py --manifest omniverse-pipeline/data/scenes/cubes/dataset_manifest.json
    python scene-gen/run_cubes_capture.py --convert-to-4dgs # also runs omni_to_4dgs.py after each capture
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ISAAC_PYTHON_DEFAULT = r"Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat"
REPO_ROOT = Path(__file__).resolve().parent.parent
OMNI_CAPTURE_SCRIPT = REPO_ROOT / "omniverse-pipeline" / "omniverse_pipeline" / "omni_capture.py"
OMNI_TO_4DGS_SCRIPT = REPO_ROOT / "omniverse-pipeline" / "omniverse_pipeline" / "omni_to_4dgs.py"
SCENES_DIR = REPO_ROOT / "omniverse-pipeline" / "data" / "scenes" / "cubes"


def get_isaac_python() -> str:
    override = os.environ.get("PIPELINE_ISAAC_NATIVE_PYTHON")
    if override and os.path.isfile(override):
        return override
    if os.path.isfile(ISAAC_PYTHON_DEFAULT):
        return ISAAC_PYTHON_DEFAULT
    raise RuntimeError(
        f"Isaac Sim python.bat not found at '{ISAAC_PYTHON_DEFAULT}' and "
        f"PIPELINE_ISAAC_NATIVE_PYTHON is not set!"
    )


def validate_capture(capture_dir: Path, expected_cameras: int = 16, min_frames: int = 60) -> bool:
    """Validate that the capture completed and produced valid frames and metadata."""
    if not capture_dir.is_dir():
        print(f"[validation] ERROR: capture directory does not exist: {capture_dir}")
        return False
    gt_json = capture_dir / "cameras_gt.json"
    if not gt_json.is_file():
        print(f"[validation] ERROR: missing cameras_gt.json in {capture_dir}")
        return False

    cam_dirs = sorted([d for d in capture_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    if len(cam_dirs) < expected_cameras:
        print(f"[validation] ERROR: found only {len(cam_dirs)}/{expected_cameras} camera directories in {capture_dir}")
        return False

    for cam_dir in cam_dirs:
        frames = list(cam_dir.glob("rgb_*.png")) or list((cam_dir / "rgb").glob("*.png"))
        if len(frames) < min_frames:
            print(f"[validation] ERROR: {cam_dir.name} has only {len(frames)}/{min_frames} frames in {capture_dir}")
            return False

    print(f"[validation] SUCCESS: Verified {len(cam_dirs)} cameras and >= {min_frames} frames each in {capture_dir}")
    return True


def run_capture(config_path: Path, isaac_python: str) -> None:
    print(f"\n================================================================================")
    print(f"[capture] Starting Isaac Sim capture: {config_path.name}")
    print(f"[capture] Using Isaac Python: {isaac_python}")
    print(f"================================================================================\n")
    t0 = time.time()

    cmd = [isaac_python, str(OMNI_CAPTURE_SCRIPT), "--config", str(config_path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Isaac Sim capture process failed with exit code {result.returncode} for {config_path}")
    print(f"\n[capture] Completed {config_path.name} in {elapsed:.1f}s")


def run_convert(capture_dir: Path, scene_name: str, config_path: Path) -> None:
    usd_path = config_path.parent / f"{scene_name}.usd"
    if usd_path.is_file():
        print(f"\n[gt_export] Exporting GT point cloud for {scene_name} from {usd_path.name}...")
        gt_cmd = [
            sys.executable,
            str(REPO_ROOT / "scene-gen" / "export_cube_gt_pointcloud.py"),
            "--usd",
            str(usd_path),
            "--out",
            str(capture_dir),
        ]
        subprocess.run(gt_cmd, cwd=str(REPO_ROOT), check=True)

    print(f"\n[convert] Converting capture {capture_dir} -> 4DGS multipleview dataset ({scene_name})...")
    cmd = [sys.executable, str(OMNI_TO_4DGS_SCRIPT), "--capture", str(capture_dir), "--out", ".", "--name", scene_name]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="List of k values to capture (e.g. 1 2 3 4)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset_manifest.json (preferred over loose globs)",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=str(SCENES_DIR),
        help=f"Directory containing scene YAML configs (default: {SCENES_DIR})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit path to a YAML capture config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Filter by motion mode (e.g. both, rotation, translation)",
    )
    parser.add_argument(
        "--convert-to-4dgs",
        action="store_true",
        help="Also run omni_to_4dgs conversion after each capture completes",
    )
    parser.add_argument(
        "--isaac-python",
        type=str,
        default=None,
        help="Path to Isaac Sim python.bat executable",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-capture output verification",
    )
    args = parser.parse_args()

    isaac_py = args.isaac_python or get_isaac_python()
    scenes_dir = Path(args.scenes_dir)

    configs_to_run = []
    if args.config:
        cfg = Path(args.config).resolve()
        if cfg.is_file():
            configs_to_run.append(cfg)
        else:
            print(f"[error] Specified config does not exist: {cfg}")
            sys.exit(1)
    else:
        # Check manifest first
        manifest_path = Path(args.manifest) if args.manifest else (scenes_dir / "dataset_manifest.json")
        if manifest_path.is_file():
            print(f"[manifest] Reading scene list from manifest: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            # Manifest can be a list of scene dicts or dict of scenes
            scenes = manifest_data if isinstance(manifest_data, list) else manifest_data.get("scenes", [manifest_data])
            for s in scenes:
                name = s.get("scene_name") if isinstance(s, dict) else s
                k_val = s.get("k_cubes") if isinstance(s, dict) else None
                if args.k is not None and k_val is not None and k_val not in args.k:
                    continue
                if args.mode and args.mode not in name:
                    continue
                cfg = scenes_dir / f"capture_config_{name}.yaml"
                if cfg.is_file() and cfg not in configs_to_run:
                    configs_to_run.append(cfg)
        else:
            target_k = args.k if args.k is not None else [1, 2, 3, 4, 5, 6, 7, 8]
            for k_val in target_k:
                patterns = [
                    f"capture_config_cubes_k{k_val}_*.yaml",
                    f"capture_config_spinning_cubes_k{k_val}_*.yaml",
                ]
                found = []
                for pat in patterns:
                    for p in sorted(scenes_dir.glob(pat)):
                        if args.mode and args.mode not in p.name:
                            continue
                        if p not in found:
                            found.append(p)
                if not found:
                    print(f"[warning] Config not found for k={k_val} in {scenes_dir}, skipping")
                else:
                    configs_to_run.extend(found)

    if not configs_to_run:
        print("[error] No matching capture configs found to execute.")
        sys.exit(1)

    print(f"Starting batch renders for {len(configs_to_run)} config(s): {[c.name for c in configs_to_run]}")
    total_t0 = time.time()

    for config_path in configs_to_run:
        print(f"\n---> Capturing: {config_path.name}")
        name_part = config_path.stem.replace("capture_config_", "")
        capture_dir = Path(f"Q:/Omniverse/renders/capture_{name_part}")

        run_capture(config_path, isaac_py)

        if not args.skip_validation:
            if not validate_capture(capture_dir):
                raise RuntimeError(f"Post-capture validation failed for {config_path.name} in {capture_dir}!")

        if args.convert_to_4dgs:
            run_convert(capture_dir, name_part, config_path)

        # Allow GPU Vulkan memory to settle before launching next Isaac Sim process
        print("[capture] Cooling down GPU for 5s...")
        time.sleep(5.0)

    total_elapsed = time.time() - total_t0
    print(f"\n================================================================================")
    print(f"All requested renders finished in {total_elapsed / 60:.1f} minutes.")
    print(f"================================================================================")


if __name__ == "__main__":
    main()

