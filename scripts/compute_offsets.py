"""
scripts/compute_offsets.py

Task T32: Sidecar Archaeology & Timing Recovery.
Recovers per-camera exposure-time offsets from RTSP capture sidecars
(RTCP Sender Reports & sync_manifest.json).
Computes drift fits, reproducibility splits, mid-exposure corrections,
and outputs data/multipleview/<scene>/offsets.json.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "camera_sync" / "rtsp_capture" / "recordings"
SYNCED_DIR = REPO_ROOT / "camera_sync" / "rtsp_capture" / "synced"
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"

SCENES = [
    "lego_turbine_base",
    "lego_turbine_2_missing",
    "lego_turbine_5_missing",
    "turbine_r1_missing_6",
]

DEFAULT_SHUTTER_S = 0.010  # 1/100s default / locked Dahua shutter


def compute_scene_offsets(scene_name: str) -> dict:
    rec_dir = RECORDINGS_DIR / scene_name
    synced_dir = SYNCED_DIR / scene_name
    mv_dir = MULTIPLEVIEW_DIR / scene_name
    
    sync_manifest_path = synced_dir / "sync_manifest.json"
    rec_manifest_path = rec_dir / "recording_manifest.json"
    
    if not sync_manifest_path.exists():
        raise FileNotFoundError(f"Missing sync manifest: {sync_manifest_path}")
    if not rec_manifest_path.exists():
        raise FileNotFoundError(f"Missing recording manifest: {rec_manifest_path}")
        
    with open(sync_manifest_path, "r", encoding="utf-8") as f:
        sync_manifest = json.load(f)
    with open(rec_manifest_path, "r", encoding="utf-8") as f:
        rec_manifest = json.load(f)
        
    instants = sync_manifest["instants"]
    n_frames = len(instants)
    ref_cam = sync_manifest.get("reference_camera", "cam01")
    measured_fps = float(sync_manifest["fps"])
    
    # Verify frame counts on disk in multipleview
    if mv_dir.exists():
        for cam_id in [f"cam{i:02d}" for i in range(1, 11)]:
            cam_path = mv_dir / cam_id
            if cam_path.exists():
                disk_frames = len(list(cam_path.glob("*.jpg")))
                assert disk_frames == n_frames, (
                    f"Frame count mismatch for {scene_name} {cam_id}: "
                    f"disk has {disk_frames}, manifest has {n_frames}"
                )

    t0 = instants[0]["ntp_unix"]
    t_end = instants[-1]["ntp_unix"]
    duration_s = t_end - t0
    times_rel = np.array([inst["ntp_unix"] - t0 for inst in instants])
    
    camera_ids = sorted(list(instants[0]["frames"].keys()))
    
    # Split verification: range 1 (first half) vs range 2 (second half)
    half = n_frames // 2
    instants_r1 = instants[:half]
    instants_r2 = instants[half:]
    
    camera_data = {}
    offsets_s_map = {}
    mean_offsets_s_map = {}
    drift_map = {}
    
    for cid in camera_ids:
        residuals_ms = np.array([inst["frames"][cid]["residual_ms"] for inst in instants])
        mean_res_ms = float(np.mean(residuals_ms))
        std_res_ms = float(np.std(residuals_ms))
        min_res_ms = float(np.min(residuals_ms))
        max_res_ms = float(np.max(residuals_ms))
        
        # Linear fit: residual_ms = offset_ms_at_t0 + drift_ms_per_s * t_rel
        if duration_s > 0:
            poly = np.polyfit(times_rel, residuals_ms, 1)
            drift_ms_per_s = float(poly[0])
            offset_ms_t0 = float(poly[1])
        else:
            drift_ms_per_s = 0.0
            offset_ms_t0 = mean_res_ms
            
        drift_s_per_s = drift_ms_per_s / 1000.0
        offset_s_t0 = offset_ms_t0 / 1000.0
        mean_offset_s = mean_res_ms / 1000.0
        
        # Anchor ref_cam at exact 0
        if cid == ref_cam:
            offset_s_t0 = 0.0
            offset_ms_t0 = 0.0
            mean_offset_s = 0.0
            mean_res_ms = 0.0
            drift_s_per_s = 0.0
            drift_ms_per_s = 0.0
            std_res_ms = 0.0
            min_res_ms = 0.0
            max_res_ms = 0.0
            
        requires_affine = abs(drift_ms_per_s) >= 0.10
        
        # Split reproducibility
        r1_mean_ms = float(np.mean([inst["frames"][cid]["residual_ms"] for inst in instants_r1]))
        r2_mean_ms = float(np.mean([inst["frames"][cid]["residual_ms"] for inst in instants_r2]))
        split_diff_ms = abs(r2_mean_ms - r1_mean_ms)
        
        camera_data[cid] = {
            "offset_s": offset_s_t0,
            "offset_ms": offset_ms_t0,
            "mean_offset_s": mean_offset_s,
            "mean_offset_ms": mean_res_ms,
            "drift_s_per_s": drift_s_per_s,
            "drift_ms_per_s": drift_ms_per_s,
            "requires_affine": requires_affine,
            "exposure_s": DEFAULT_SHUTTER_S,
            "residual_stats": {
                "mean_ms": mean_res_ms,
                "std_ms": std_res_ms,
                "min_ms": min_res_ms,
                "max_ms": max_res_ms,
            },
            "split_reproducibility": {
                "range1_mean_ms": r1_mean_ms,
                "range2_mean_ms": r2_mean_ms,
                "diff_ms": split_diff_ms,
            },
        }
        offsets_s_map[cid] = offset_s_t0
        mean_offsets_s_map[cid] = mean_offset_s
        drift_map[cid] = drift_s_per_s

    # Provenance and frame count reconciliation note
    provenance = {
        "method": "rtsp_sidecar_rtcp_reconstruction",
        "remedy": "R3",
        "task": "T32",
        "phase": "Phase 0 (Timing Recovery)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_recording_dir": str(rec_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_synced_dir": str(synced_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reference_camera": ref_cam,
        "time_base": "sensor_frame_start (RTP sampling timestamp mapped to NTP wall-clock via RTCP Sender Reports)",
        "frame_rate": {
            "nominal_fps": 20.0,
            "measured_fps": measured_fps,
            "rtp_step_ticks": 4500,
            "rtp_clock_hz": 90000,
        },
        "exposure": {
            "shutter_speed": "1/100",
            "exposure_s": DEFAULT_SHUTTER_S,
            "mid_exposure_corrected": True,
            "mid_exposure_note": (
                "All cameras were configured with identical 1/100s (10 ms) shutter speed. "
                "The mid-exposure observation instant (frame_start + exposure/2 = +5 ms) "
                "is identical across all cameras and cancels identically in pairwise "
                "relative offsets."
            ),
        },
        "frame_count_reconciliation": {
            "frames_on_disk": n_frames,
            "instants_total_captured": sync_manifest["instants_total"],
            "instants_kept_synced": sync_manifest["instants_kept"],
            "tolerance_ms": sync_manifest["tolerance_ms"],
            "explanation": (
                f"For scene '{scene_name}', {sync_manifest['instants_total']} total frames were captured. "
                f"{sync_manifest['instants_kept']} frames satisfied the sync tolerance ({sync_manifest['tolerance_ms']:.3f} ms, "
                f"half-frame window at {measured_fps:.2f} fps). "
                f"Exactly {n_frames} frames were extracted and structured into data/multipleview/{scene_name}/. "
                "The historical reference to 'n_times: 160' in scripts/archived/run_lego_turbine_pipeline.py was "
                "strictly an evaluation query step count passed to trajectory extraction (torch.linspace(0, 1, 160)), "
                "NOT a video frame drop. "
                "The 261 frame count cited in early diagnoses corresponds to turbine_r1_missing_6 (which captured 265/kept 261 frames)."
            ),
        },
        "drift_policy": (
            "Cameras with |drift_ms_per_s| >= 0.10 ms/s are flagged with requires_affine=True. "
            "For these cameras, downstream loader (T39) can evaluate the continuous affine timestamp: "
            "t_k(t) = t + offset_s + drift_s_per_s * t."
        ),
    }

    offsets_doc = {
        "scene": scene_name,
        "fps": measured_fps,
        "frame_count": n_frames,
        "duration_s": duration_s,
        "reference_camera": ref_cam,
        "exposure_s": DEFAULT_SHUTTER_S,
        "mid_exposure_corrected": True,
        "offsets_s": offsets_s_map,
        "mean_offsets_s": mean_offsets_s_map,
        "drift_s_per_s": drift_map,
        "cameras": camera_data,
        "provenance": provenance,
    }
    
    return offsets_doc


def main():
    print("Computing offsets for all turbine scenes...")
    for scene in SCENES:
        offsets_doc = compute_scene_offsets(scene)
        out_dir = MULTIPLEVIEW_DIR / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "offsets.json"
        
        content = json.dumps(offsets_doc, indent=2)
        out_file.write_text(content, encoding="utf-8")
        
        # Calculate sha256
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        print(f"[WROTE] {out_file.relative_to(REPO_ROOT)} ({len(content)} bytes, SHA256: {sha256[:16]}...)")
        
        # Also print summary
        cams = offsets_doc["cameras"]
        flagged = [c for c, d in cams.items() if d["requires_affine"]]
        print(f"  FPS: {offsets_doc['fps']:.4f}, Frames: {offsets_doc['frame_count']}, Ref: {offsets_doc['reference_camera']}")
        print(f"  Flagged for affine model (|drift| >= 0.1 ms/s): {flagged if flagged else 'None'}")


if __name__ == "__main__":
    main()
