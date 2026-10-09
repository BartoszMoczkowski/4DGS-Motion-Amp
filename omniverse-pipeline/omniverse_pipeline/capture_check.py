"""Capture-directory QA for Isaac Sim multi-view synthetic captures (T26).

Moved here from ``scene-gen/check_capture_progress.py`` (capture QA belongs to the capture
pipeline; scene-gen can't be imported from omniverse_pipeline, only vice versa). The
previously hardcoded completion heuristics (>=16 cameras, >=60 frames/camera) are now
parameters: pass ``expected_cams``/``expected_frames`` explicitly or derive them from a
capture config YAML via :func:`load_expectations_from_config` (``rig.n_cameras`` /
``capture.num_frames``). The legacy defaults (16/60) are kept as the function defaults so
existing callers behave identically.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional


def _format_bar(pct: float, width: int = 20) -> str:
    filled = int(round(width * (pct / 100.0)))
    filled = max(0, min(filled, width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def load_expectations_from_config(config_path) -> tuple[Optional[int], Optional[int]]:
    """Read ``rig.n_cameras`` / ``capture.num_frames`` from an omni_capture config YAML.

    Returns ``(expected_cams, expected_frames)``; either is ``None`` if the key is absent.
    """
    import yaml  # light, but only needed on this path

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    n_cameras = (data.get("rig") or {}).get("n_cameras")
    num_frames = (data.get("capture") or {}).get("num_frames")
    return (
        int(n_cameras) if n_cameras is not None else None,
        int(num_frames) if num_frames is not None else None,
    )


def inspect_capture_dir(
    capture_path: Path,
    *,
    expected_cams: Optional[int] = 16,
    expected_frames: Optional[int] = 60,
) -> dict:
    """Inspect one capture directory (live or completed).

    ``expected_cams``/``expected_frames`` drive the completion-inference fallback used when
    ``status.json`` is absent/outdated; pass ``None`` for "no expectation" (a capture with
    no status.json is then never inferred "completed"). The returned dict carries per-camera
    frame counts under ``cam_frames`` in addition to the aggregated keys the original
    scene-gen monitoring table uses.
    """
    capture_path = Path(capture_path)
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
        "cam_frames": {},
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
    counts = []
    for cd in cam_dirs:
        frames = list(cd.glob("rgb_*.png")) or list((cd / "rgb").glob("*.png"))
        info["cam_frames"][cd.name] = len(frames)
        counts.append(len(frames))
    info["min_cam_frames"] = min(counts) if counts else 0
    info["max_cam_frames"] = max(counts) if counts else 0

    info["gt_json"] = (capture_path / "cameras_gt.json").is_file()
    info["gt_ply"] = (capture_path / "points3D_gt.ply").is_file()

    # Infer status if status.json is absent or outdated
    if info["status"] == "unknown":
        if (
            info["gt_json"]
            and expected_cams is not None
            and expected_frames is not None
            and info["cams_found"] >= expected_cams
            and info["min_cam_frames"] >= expected_frames
        ):
            info["status"] = "completed"
            info["progress_pct"] = 100.0
        elif info["cams_found"] > 0:
            info["status"] = "in_progress"

    return info


def verify_capture_dir(
    capture_path,
    *,
    expected_cams: Optional[int] = None,
    expected_frames: Optional[int] = None,
) -> tuple[bool, list[str]]:
    """Check a capture directory against expectations; return ``(ok, report_lines)``.

    This is the `amp capture verify` backend (replacing the post-mortem's ad-hoc
    ``python -c`` frame counting): cameras_gt.json presence, camera count, and per-camera
    frame counts. Expectations left as ``None`` are reported but not checked.
    """
    path = Path(capture_path)
    if not path.is_dir():
        return False, [f"capture directory does not exist: {path}"]

    info = inspect_capture_dir(
        path, expected_cams=expected_cams, expected_frames=expected_frames
    )
    lines = [f"capture: {path}"]
    ok = True

    lines.append(f"cameras_gt.json: {'present' if info['gt_json'] else 'MISSING'}")
    if not info["gt_json"]:
        ok = False

    exp_c = f" (expected {expected_cams})" if expected_cams is not None else ""
    lines.append(f"cameras: {info['cams_found']}{exp_c}")
    if expected_cams is not None and info["cams_found"] != expected_cams:
        lines.append(f"  MISMATCH: found {info['cams_found']}, expected {expected_cams}")
        ok = False

    exp_f = f" (expected {expected_frames}/cam)" if expected_frames is not None else ""
    lines.append(f"frames per camera{exp_f}:")
    for cam, n in sorted(info["cam_frames"].items()):
        mark = ""
        if expected_frames is not None and n != expected_frames:
            mark = f"  MISMATCH (expected {expected_frames})"
            ok = False
        lines.append(f"  {cam}: {n}{mark}")

    lines.append(f"status: {info['status']} ({info['progress_pct']:.1f}%)")
    return ok, lines


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
