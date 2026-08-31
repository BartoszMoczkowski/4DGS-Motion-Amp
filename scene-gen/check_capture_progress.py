#!/usr/bin/env python3
"""
check_capture_progress.py — CLI monitoring tool for Isaac Sim multi-view synthetic captures.

Inspects live or completed captures, displays chunk rendering progress, frame counts
across cameras, and validation status of GT artifacts.

Usage:
    python scene-gen/check_capture_progress.py
    python scene-gen/check_capture_progress.py --watch 2
    python scene-gen/check_capture_progress.py --capture-dir Q:/Omniverse/renders/capture_cubes_k2_both
    python scene-gen/check_capture_progress.py --renders-dir Q:/Omniverse/renders
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_RENDERS_DIR = Path(r"Q:\Omniverse\renders")


def _format_bar(pct: float, width: int = 20) -> str:
    filled = int(round(width * (pct / 100.0)))
    filled = max(0, min(filled, width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def inspect_capture_dir(capture_path: Path) -> dict:
    info = {
        "path": capture_path,
        "name": capture_path.name.replace("capture_", ""),
        "status": "unknown",
        "progress_pct": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "chunk_info": "-",
        "cams_found": 0,
        "min_cam_frames": 0,
        "max_cam_frames": 0,
        "gt_json": False,
        "gt_ply": False,
        "error": None,
        "last_updated": None,
    }

    # Check status.json
    status_file = capture_path / "status.json"
    if status_file.is_file():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            info["status"] = sdata.get("status", "unknown")
            info["progress_pct"] = float(sdata.get("overall_progress_pct", 0.0))
            info["current_frame"] = int(sdata.get("current_frame", 0))
            info["total_frames"] = int(sdata.get("total_frames", 0))
            cur_c = sdata.get("current_chunk", 0)
            tot_c = sdata.get("total_chunks", 0)
            info["chunk_info"] = f"{cur_c}/{tot_c}" if tot_c > 0 else "-"
            info["error"] = sdata.get("error")
            up_at = sdata.get("updated_at")
            if up_at:
                info["last_updated"] = time.strftime("%H:%M:%S", time.localtime(up_at))
        except Exception:
            pass

    # Check cameras and frames on disk
    cam_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir() and d.name.startswith("cam")])
    info["cams_found"] = len(cam_dirs)
    if cam_dirs:
        counts = []
        for cd in cam_dirs:
            frames = list(cd.glob("rgb_*.png")) or list((cd / "rgb").glob("*.png"))
            counts.append(len(frames))
        info["min_cam_frames"] = min(counts)
        info["max_cam_frames"] = max(counts)
    else:
        info["min_cam_frames"] = 0
        info["max_cam_frames"] = 0

    info["gt_json"] = (capture_path / "cameras_gt.json").is_file()
    info["gt_ply"] = (capture_path / "points3D_gt.ply").is_file()

    # Infer status if status.json is absent or outdated
    if info["status"] == "unknown":
        if info["gt_json"] and info["cams_found"] >= 16 and info["min_cam_frames"] >= 60:
            info["status"] = "completed"
            info["progress_pct"] = 100.0
        elif info["cams_found"] > 0:
            info["status"] = "in_progress"

    return info


def print_status_table(captures: list[dict]) -> None:
    if not captures:
        print("[progress] No captures found in the specified directory.")
        return

    header = (
        f"{'SCENE NAME':<24} | {'STATUS':<11} | {'CHUNK':<7} | {'CAMS':<6} | "
        f"{'FRAMES/CAM':<12} | {'PROGRESS':<28} | {'GT ASSETS':<9} | {'UPDATED':<8}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for c in captures:
        status_str = c["status"].upper()
        if c["status"] == "completed":
            status_disp = f"\033[92m{status_str:<11}\033[0m" if sys.stdout.isatty() else f"{status_str:<11}"
        elif c["status"] == "failed":
            status_disp = f"\033[91m{status_str:<11}\033[0m" if sys.stdout.isatty() else f"{status_str:<11}"
        else:
            status_disp = f"\033[93m{status_str:<11}\033[0m" if sys.stdout.isatty() else f"{status_str:<11}"

        frames_str = (
            f"{c['min_cam_frames']}"
            if c["min_cam_frames"] == c["max_cam_frames"]
            else f"{c['min_cam_frames']}-{c['max_cam_frames']}"
        )
        if c["total_frames"] > 0:
            frames_str += f"/{c['total_frames']}"

        bar_str = _format_bar(c["progress_pct"], width=18)
        gt_str = f"JSON:{'OK' if c['gt_json'] else '--'} PLY:{'OK' if c['gt_ply'] else '--'}"
        time_str = c["last_updated"] or "-"

        print(
            f"{c['name']:<24} | {status_disp} | {c['chunk_info']:<7} | {c['cams_found']:<6} | "
            f"{frames_str:<12} | {bar_str:<28} | {gt_str:<9} | {time_str:<8}"
        )
        if c["error"]:
            print(f"   ↳ Error: {c['error']}")

    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renders-dir",
        type=str,
        default=str(DEFAULT_RENDERS_DIR),
        help=f"Directory containing capture folders (default: {DEFAULT_RENDERS_DIR})",
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        default=None,
        help="Path to a single capture directory to monitor",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="Refresh interval in seconds (e.g. --watch 2)",
    )
    args = parser.parse_args()

    try:
        while True:
            captures = []
            if args.capture_dir:
                cap_path = Path(args.capture_dir)
                if cap_path.is_dir():
                    captures.append(inspect_capture_dir(cap_path))
                else:
                    print(f"[error] Specified directory does not exist: {cap_path}")
                    sys.exit(1)
            else:
                renders_dir = Path(args.renders_dir)
                if renders_dir.is_dir():
                    dirs = sorted([d for d in renders_dir.iterdir() if d.is_dir() and d.name.startswith("capture_")])
                    for d in dirs:
                        captures.append(inspect_capture_dir(d))
                else:
                    print(f"[warning] Renders directory not found: {renders_dir}")

            if args.watch:
                # Clear terminal screen
                os.system("cls" if os.name == "nt" else "clear")
                print(f"[check_capture_progress] Watching captures in {args.capture_dir or args.renders_dir} (interval: {args.watch}s) — Press Ctrl+C to exit")

            print_status_table(captures)

            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
