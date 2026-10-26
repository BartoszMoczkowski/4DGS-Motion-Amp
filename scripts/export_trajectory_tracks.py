#!/usr/bin/env python3
"""scripts/export_trajectory_tracks.py

Task T40: Export measured 2D trajectory tracks for Trajectory Reprojection Loss (R8).
Extracts or synthesizes per-camera per-frame 2D displacement tracks (displacements_px: [dx, dy])
from optical tracking (optical_offsets.json) or ground truth parameters (gt_motion_info.json)
and writes `trajectory_tracks.json` in the scene directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"


def export_tracks_for_scene(scene_name: str) -> Path | None:
    scene_dir = MULTIPLEVIEW_DIR / scene_name
    if not scene_dir.is_dir():
        print(f"[ERROR] Scene directory not found: {scene_dir}")
        return None

    out_file = scene_dir / "trajectory_tracks.json"

    # 1. Check for optical_offsets.json
    optical_file = scene_dir / "optical_offsets.json"
    gt_file = scene_dir / "gt_motion_info.json"
    offsets_file = scene_dir / "offsets.json"

    fps = 20.0
    n_frames = 60

    if offsets_file.exists():
        try:
            offs_data = json.loads(offsets_file.read_text(encoding="utf-8"))
            fps = float(offs_data.get("fps", fps))
            n_frames = int(offs_data.get("frame_count", n_frames))
        except Exception:
            pass

    cam_tracks = {}

    if optical_file.exists():
        opt_data = json.loads(optical_file.read_text(encoding="utf-8"))
        f0 = float(opt_data.get("wobble_frequency_hz", 7.112))
        cams = opt_data.get("cameras", {})

        for cam_id, cinfo in cams.items():
            amp_px = float(cinfo.get("amplitude_px", 0.0))
            phase_rad = float(cinfo.get("phase_rad", 0.0))
            angle_deg = float(cinfo.get("principal_axis_angle_deg", 0.0))
            theta = math.radians(angle_deg)
            cos_th = math.cos(theta)
            sin_th = math.sin(theta)

            displacements = []
            for i in range(n_frames):
                t_i = float(i) / fps
                sig = amp_px * math.sin(2.0 * math.pi * f0 * t_i + phase_rad)
                dx = sig * cos_th
                dy = sig * sin_th
                displacements.append([float(dx), float(dy)])

            cam_tracks[cam_id] = {
                "amplitude_px": amp_px,
                "phase_rad": phase_rad,
                "principal_axis_angle_deg": angle_deg,
                "displacements_px": displacements,
            }

    elif gt_file.exists():
        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
        f0 = float(gt_data.get("wobble_frequency_hz", 7.112))
        per_cam_amps = gt_data.get("per_camera_projected_amplitudes_px", {})
        injected = gt_data.get("injected_offsets_s", {})

        for cam_id, amp_px in per_cam_amps.items():
            delta_s = float(injected.get(cam_id, 0.0))
            displacements = []
            for i in range(n_frames):
                t_phys = float(i) / fps + delta_s
                dx = float(amp_px) * math.sin(2.0 * math.pi * f0 * t_phys)
                dy = 0.0
                displacements.append([float(dx), float(dy)])

            cam_tracks[cam_id] = {
                "amplitude_px": float(amp_px),
                "phase_rad": 0.0,
                "principal_axis_angle_deg": 0.0,
                "displacements_px": displacements,
            }
    else:
        print(f"[WARN] Neither optical_offsets.json nor gt_motion_info.json found in {scene_dir}")
        return None

    payload = {
        "scene": scene_name,
        "fps": fps,
        "n_frames": n_frames,
        "n_cameras": len(cam_tracks),
        "cameras": cam_tracks,
    }

    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[WROTE] {out_file.relative_to(REPO_ROOT)} ({len(cam_tracks)} cameras, {n_frames} frames)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="T40: Export measured 2D trajectory tracks.")
    parser.add_argument("--scene", type=str, default="all",
                        help="Scene name or 'all'")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = [
            "synthetic_turbine_preflight",
            "lego_turbine_base",
            "lego_turbine_2_missing",
            "lego_turbine_5_missing",
            "turbine_r1_missing_6",
        ]
    else:
        scenes = [args.scene]

    for s in scenes:
        export_tracks_for_scene(s)


if __name__ == "__main__":
    main()
