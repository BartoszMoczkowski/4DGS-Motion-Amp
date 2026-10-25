"""
orchestrator/tests/test_track_wobble_phase.py

Tests for T33: track_wobble_phase.py optical offset fitter.
Verifies optical offset tables, harmonic content, and anchor gauge across turbine scenes.
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
def test_optical_offsets_json_exists_and_valid(scene_name: str):
    scene_dir = MULTIPLEVIEW_DIR / scene_name
    opt_file = scene_dir / "optical_offsets.json"

    assert opt_file.exists(), f"optical_offsets.json not found in {scene_dir}"
    data = json.loads(opt_file.read_text(encoding="utf-8"))

    # Top-level required fields
    for field in [
        "scene", "fps", "frame_count", "reference_camera",
        "wobble_frequency_hz", "wobble_period_s", "conversion_deg_per_ms",
        "optical_offsets_s", "validation_summary", "cameras",
    ]:
        assert field in data, f"Missing required top-level field {field!r} in {opt_file}"

    assert data["scene"] == scene_name
    assert data["reference_camera"] == "cam01"
    assert data["wobble_frequency_hz"] > 5.0 and data["wobble_frequency_hz"] < 7.5
    assert data["optical_offsets_s"]["cam01"] == 0.0

    # Exactly 10 cameras
    expected_cams = [f"cam{i:02d}" for i in range(1, 11)]
    assert sorted(list(data["cameras"].keys())) == expected_cams
    assert sorted(list(data["optical_offsets_s"].keys())) == expected_cams

    for cid in expected_cams:
        cam_data = data["cameras"][cid]
        for field in [
            "camera_id", "n_frames", "n_points_tracked", "amplitude_px",
            "phase_rad", "raw_offset_s", "resolved_offset_s", "harmonics",
        ]:
            assert field in cam_data, f"Missing field {field!r} in {cid}"

        assert cam_data["resolved_offset_s"] == data["optical_offsets_s"][cid]
        assert -math_pi_bound(data["wobble_period_s"]) <= cam_data["resolved_offset_s"] <= math_pi_bound(data["wobble_period_s"])

        # Harmonics check for T41 Fourier head
        harm = cam_data["harmonics"]
        for hf in ["amp_f1", "amp_f2", "amp_f3", "thd"]:
            assert hf in harm, f"Missing harmonic metric {hf!r} in {cid}"
        assert harm["thd"] >= 0.0


def math_pi_bound(period_s: float) -> float:
    # Bounded by period
    return period_s * 1.5
