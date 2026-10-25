"""
orchestrator/tests/test_offsets_recovery.py

Tests for T32: Sidecar archaeology & offsets.json recovery.
Verifies structure, values, anchored reference, and reproducibility of recovered offsets.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"

TURBINE_SCENES = [
    "lego_turbine_base",
    "lego_turbine_2_missing",
    "lego_turbine_5_missing",
    "turbine_r1_missing_6",
]


@pytest.mark.parametrize("scene_name", TURBINE_SCENES)
def test_offsets_json_exists_and_valid(scene_name: str):
    scene_dir = MULTIPLEVIEW_DIR / scene_name
    offsets_file = scene_dir / "offsets.json"
    
    assert offsets_file.exists(), f"offsets.json not found in {scene_dir}"
    data = json.loads(offsets_file.read_text(encoding="utf-8"))
    
    # Required top-level fields
    for field in ["scene", "fps", "frame_count", "reference_camera", "exposure_s", 
                  "mid_exposure_corrected", "offsets_s", "mean_offsets_s", "drift_s_per_s", 
                  "cameras", "provenance"]:
        assert field in data, f"Missing required top-level field {field!r} in {offsets_file}"
        
    assert data["scene"] == scene_name
    assert data["reference_camera"] == "cam01"
    assert data["fps"] > 19.5 and data["fps"] < 20.5, f"Unexpected fps: {data['fps']}"
    assert data["exposure_s"] == 0.010
    assert data["mid_exposure_corrected"] is True
    
    # Exactly 10 cameras
    expected_cams = [f"cam{i:02d}" for i in range(1, 11)]
    assert sorted(list(data["cameras"].keys())) == expected_cams
    assert sorted(list(data["offsets_s"].keys())) == expected_cams
    assert sorted(list(data["drift_s_per_s"].keys())) == expected_cams
    
    # Cam01 anchored at 0.0
    assert data["offsets_s"]["cam01"] == 0.0
    assert data["cameras"]["cam01"]["offset_s"] == 0.0
    assert data["cameras"]["cam01"]["drift_s_per_s"] == 0.0
    
    # Per-camera validation
    for cid in expected_cams:
        cam_info = data["cameras"][cid]
        for cam_field in ["offset_s", "offset_ms", "drift_s_per_s", "drift_ms_per_s", 
                          "requires_affine", "exposure_s", "residual_stats", "split_reproducibility"]:
            assert cam_field in cam_info, f"Missing {cam_field!r} in camera {cid}"
            
        assert data["offsets_s"][cid] == cam_info["offset_s"]
        assert data["drift_s_per_s"][cid] == cam_info["drift_s_per_s"]
        
        # Residual bounds: Dahua tolerance was 25 ms, so all offsets must be within [-25ms, +25ms]
        assert -0.026 <= cam_info["offset_s"] <= 0.026, (
            f"Offset for {cid} ({cam_info['offset_s']} s) outside tolerance bounds"
        )
        
        # Split reproducibility: range split difference should be < 10 ms for all cameras
        diff_ms = cam_info["split_reproducibility"]["diff_ms"]
        assert diff_ms < 10.0, f"Split reproducibility difference too large for {cid}: {diff_ms} ms"

    # Frame count check against disk
    expected_frame_count = data["frame_count"]
    for cid in expected_cams:
        cam_dir = scene_dir / cid
        if cam_dir.exists():
            disk_frames = len(list(cam_dir.glob("*.jpg")))
            assert disk_frames == expected_frame_count, (
                f"Disk frames {disk_frames} != offsets frame_count {expected_frame_count} for {cid}"
            )
            
    # Provenance check
    prov = data["provenance"]
    assert prov["task"] == "T32"
    assert prov["remedy"] == "R3"
    assert "frame_count_reconciliation" in prov
    assert "time_base" in prov
    assert "sensor_frame_start" in prov["time_base"]
