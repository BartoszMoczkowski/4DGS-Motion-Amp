"""
orchestrator/tests/test_synthetic_preflight.py

Tests for Task T38: Synthetic sync pre-flight scene & diagnostics.
Verifies:
1. Synthetic scene directory structure and COLMAP binary assets.
2. Injected offset table conforms to authoritative T32 offsets.json schema.
3. Ground-truth motion metadata (7.112 Hz, ~2.8 px amplitude, 10 cameras).
4. Precomputed motion masks isolate the oscillating target without background leakage (T36).
5. Pre-registered gates pass:
   - G1: Spectral peak at 7.11 Hz.
   - G2: Per-camera offset accuracy <= 2.0 ms (max discrepancy <= 1.0 ms).
   - G3: Amplitude fidelity <= 30%.
   - G4: Motion mask isolation.
   - G5: Cross-view phase coherence under timestamp correction (< 3 deg).
6. Orchestrator preset resolution for synthetic_turbine_preflight.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"
SCENE_DIR = MULTIPLEVIEW_DIR / "synthetic_turbine_preflight"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_synthetic_preflight import run_preflight_checks


def load_colmap_loader():
    spec = importlib.util.spec_from_file_location("colmap_loader", REPO_ROOT / "core" / "scene" / "colmap_loader.py")
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    return cl


def test_synthetic_scene_filesystem_structure():
    """Verify that all files and folders required by the 4DGS multipleview loader exist."""
    assert SCENE_DIR.exists(), f"Synthetic pre-flight scene directory {SCENE_DIR} does not exist"

    # 10 camera directories
    for i in range(1, 11):
        cam_dir = SCENE_DIR / f"cam{i:02d}"
        assert cam_dir.is_dir(), f"Missing camera directory {cam_dir}"
        frames = list(cam_dir.glob("frame_*.jpg"))
        assert len(frames) == 60, f"Expected 60 frames in {cam_dir}, found {len(frames)}"

    # Sparse COLMAP folder
    sparse_dir = SCENE_DIR / "sparse_"
    assert sparse_dir.is_dir(), "Missing sparse_ directory"
    assert (sparse_dir / "cameras.bin").exists(), "Missing cameras.bin"
    assert (sparse_dir / "images.bin").exists(), "Missing images.bin"
    assert (sparse_dir / "points3D.bin").exists(), "Missing points3D.bin"

    # Point cloud and LLFF poses
    assert (SCENE_DIR / "points3D_multipleview.ply").exists(), "Missing points3D_multipleview.ply"
    assert (SCENE_DIR / "poses_bounds_multipleview.npy").exists(), "Missing poses_bounds_multipleview.npy"
    assert (SCENE_DIR / "offsets.json").exists(), "Missing offsets.json"
    assert (SCENE_DIR / "gt_motion_info.json").exists(), "Missing gt_motion_info.json"
    assert (SCENE_DIR / "motion_masks").is_dir(), "Missing motion_masks directory"


def test_offsets_json_schema_and_values():
    """Verify that offsets.json matches the authoritative T32 format."""
    offsets_file = SCENE_DIR / "offsets.json"
    assert offsets_file.exists()
    data = json.loads(offsets_file.read_text(encoding="utf-8"))

    # Required top-level fields
    for field in [
        "scene", "fps", "frame_count", "reference_camera", "exposure_s",
        "mid_exposure_corrected", "offsets_s", "mean_offsets_s", "drift_s_per_s",
        "cameras", "provenance"
    ]:
        assert field in data, f"Missing required top-level field {field!r} in offsets.json"

    assert data["scene"] == "synthetic_turbine_preflight"
    assert data["reference_camera"] == "cam01"
    assert data["fps"] == 20.0
    assert data["exposure_s"] == 0.010
    assert data["mid_exposure_corrected"] is True

    # 10 cameras present and cam01 anchored at 0.0
    expected_cams = [f"cam{i:02d}" for i in range(1, 11)]
    assert sorted(list(data["cameras"].keys())) == expected_cams
    assert sorted(list(data["offsets_s"].keys())) == expected_cams
    assert data["offsets_s"]["cam01"] == 0.0
    assert data["cameras"]["cam01"]["offset_s"] == 0.0

    # Per-camera validation
    for cid in expected_cams:
        cam_info = data["cameras"][cid]
        for cam_field in [
            "offset_s", "offset_ms", "drift_s_per_s", "drift_ms_per_s",
            "requires_affine", "exposure_s", "residual_stats", "split_reproducibility"
        ]:
            assert cam_field in cam_info, f"Missing field {cam_field!r} in {cid}"


def test_gt_motion_info_metadata():
    """Verify ground-truth parameters recorded in gt_motion_info.json."""
    gt_file = SCENE_DIR / "gt_motion_info.json"
    assert gt_file.exists()
    gt = json.loads(gt_file.read_text(encoding="utf-8"))

    assert gt["scene"] == "synthetic_turbine_preflight"
    assert abs(gt["wobble_frequency_hz"] - 7.112) < 1e-4
    assert gt["fps"] == 20.0
    assert gt["n_frames"] == 60
    assert gt["target_projected_amplitude_px"] == 2.8
    assert gt["n_total_points3d"] > 1000
    assert gt["n_dynamic_points3d"] > 0
    assert gt["n_static_points3d"] > 0
    assert len(gt["injected_offsets_s"]) == 10


def test_motion_masks_validity():
    """Verify precomputed motion masks (T36) for synthetic scene."""
    masks_dir = SCENE_DIR / "motion_masks"
    manifest_file = masks_dir / "manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    for i in range(1, 11):
        cid = f"cam{i:02d}"
        npy_file = masks_dir / f"{cid}.npy"
        vis_file = masks_dir / f"{cid}_vis.png"
        heat_file = masks_dir / f"{cid}_heatmap.png"

        assert npy_file.exists(), f"Missing {npy_file}"
        assert vis_file.exists(), f"Missing {vis_file}"
        assert heat_file.exists(), f"Missing {heat_file}"

        mask_arr = np.load(npy_file)
        assert mask_arr.shape == (900, 1600)
        assert 0.0 <= mask_arr.min() <= mask_arr.max() <= 1.0

        # Rotor motion occupies ~4% to 15% of frame area
        act_frac = float(np.mean(mask_arr > 0.05))
        assert 0.02 <= act_frac <= 0.20, f"Unexpected active mask fraction {act_frac} for {cid}"


def test_optical_tracking_gates_end_to_end():
    """Run full pre-flight verification battery and assert all 5 gates pass."""
    passed, record = run_preflight_checks(
        scene_dir=SCENE_DIR,
        target_frequency=7.112,
        max_offset_tolerance_ms=2.0,
        amplitude_tolerance_pct=30.0,
        regenerate=False,
    )

    assert passed is True, f"Pre-flight checks failed: {record['gates']}"
    assert record["overall_passed"] is True
    assert record["verdict"] == "PASS"

    # Gate G1
    assert record["gates"]["G1_spectral_peak"] is True

    # Gate G2: offset accuracy <= 2.0 ms across all cameras
    assert record["gates"]["G2_offset_accuracy"] is True
    assert record["metrics"]["max_discrepancy_ms"] <= 2.0
    assert record["metrics"]["mean_discrepancy_ms"] <= 1.0

    # Gate G3: amplitude fidelity <= 30%
    assert record["gates"]["G3_amplitude_fidelity"] is True
    assert record["metrics"]["max_amplitude_error_pct"] <= 30.0

    # Gate G4: motion mask isolation
    assert record["gates"]["G4_motion_mask_isolation"] is True

    # Gate G5: cross-view phase error after correction <= 3.0 deg
    assert record["gates"]["G5_timestamp_alignment"] is True
    assert record["metrics"]["max_residual_phase_deg"] <= 3.0


def test_colmap_model_and_pointcloud():
    """Verify that COLMAP binary files and PLY are readable by standard loaders."""
    cl = load_colmap_loader()
    cameras = cl.read_intrinsics_binary(SCENE_DIR / "sparse_" / "cameras.bin")
    assert 1 in cameras
    assert cameras[1].width == 1600
    assert cameras[1].height == 900

    images = cl.read_extrinsics_binary(SCENE_DIR / "sparse_" / "images.bin")
    assert len(images) == 10

    ply = PlyData.read(SCENE_DIR / "points3D_multipleview.ply")
    assert len(ply["vertex"]) > 1000
    for prop in ["x", "y", "z", "red", "green", "blue"]:
        assert prop in ply["vertex"].data.dtype.names

    poses_bounds = np.load(SCENE_DIR / "poses_bounds_multipleview.npy")
    assert poses_bounds.shape == (10, 17)


def test_synthetic_preset_resolution():
    """Verify that orchestrator preset synthetic_turbine_preflight resolves cleanly."""
    from pipeline.config.resolver import validate_config

    config = validate_config("synthetic_turbine_preflight")
    assert config.model.source_path == "data/multipleview/synthetic_turbine_preflight"
    assert config.optim.coarse_iterations == 500
    assert config.optim.use_motion_mask is True
    assert config.optim.motion_mask_max == 50.0
    assert config.hidden.regularizer_anneal_iter == 3000
    assert config.hidden.time_smoothness_weight_final == 0.0001
    assert config.hidden.l1_time_planes_final == 0.0
