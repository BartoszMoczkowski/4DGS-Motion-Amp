"""orchestrator/tests/test_cross_scene_validation.py

Test suite for Task T42: Cross-scene validation and amplitude-ordering report.
Verifies:
1. Named presets ladder resolution and schema validation for base / 2_missing / 5_missing / missing_6 under R9.
2. Cross-scene rig prior calculation, empirical Bayes pooling, and residual tolerances.
3. Evaluation logic for all 5 pre-registered validation metrics:
   - G1: 7.112 Hz spectral peak test
   - G2: Amplitude fidelity vs optical tracks (+-30%)
   - G3: Cross-view phase coherence under corrected timestamps (< 15 deg)
   - G4: Offset accuracy vs empirical rig prior (<= 2.0 ms for stable views)
   - G5: Rendering quality guardrail (PSNR >= 24.5 dB, SSIM >= 0.82)
4. Monotonic amplitude ordering physics sanity check (proving inversion of pathological baseline).
5. Summary JSON and CSV report artifact serialization and schemas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import validate_config
from scripts.compute_rig_prior import compute_rig_prior
from scripts.run_cross_scene_validation import (
    SCENE_SPECS,
    NOMINAL_FREQUENCY_HZ,
    evaluate_scene_motion,
    run_cross_scene_validation,
)


def test_preset_ladder_resolution_all_scenes():
    """Verify that all R7, R8, R9 presets resolve cleanly under Pydantic strict validation."""
    scenes = ["lego_turbine_base", "lego_turbine_2_missing", "lego_turbine_5_missing", "turbine_r1_missing_6"]
    
    for s in scenes:
        p_r7 = f"{s}_r7"
        p_r8 = f"{s}_r8"
        p_r9 = f"{s}_r9"
        
        cfg7 = validate_config(p_r7)
        assert cfg7.optim.use_camera_offsets is True
        assert cfg7.optim.learnable_camera_offsets is True
        assert cfg7.optim.learnable_offsets_lr == 0.0001
        assert cfg7.optim.max_offset_clamp_s == 0.040
        assert cfg7.optim.coarse_iterations == 500
        assert cfg7.optim.use_motion_mask is True
        
        cfg8 = validate_config(p_r8)
        assert cfg8.optim.lambda_trajectory == 1.0
        assert cfg8.optim.trajectory_track_file == "trajectory_tracks.json"
        
        cfg9 = validate_config(p_r9)
        assert cfg9.optim.use_fourier_rigid_head is True
        assert cfg9.optim.fourier_head_frequency_hz == 7.112
        assert cfg9.optim.fourier_head_num_harmonics == 3
        assert cfg9.optim.fourier_head_freeze_iter == 500
        assert cfg9.optim.fourier_head_lr == 0.001


def test_cross_scene_rig_prior_computation(tmp_path):
    """Verify empirical Bayes pooling across multi-cam rig captures."""
    out_prior_file = tmp_path / "test_rig_prior.json"
    prior_bundle = compute_rig_prior(out_path=out_prior_file)
    
    assert out_prior_file.exists()
    assert prior_bundle["reference_camera"] == "cam01"
    assert prior_bundle["rig_prior_offsets_s"]["cam01"] == 0.0
    
    # 10 cameras present
    for i in range(1, 11):
        cid = f"cam{i:02d}"
        assert cid in prior_bundle["rig_prior_offsets_s"]
        assert cid in prior_bundle["per_camera_analysis"]
        
    # cam10 has extremely high stability across all 4 sessions (std < 2.0 ms)
    cam10_std = prior_bundle["per_camera_analysis"]["cam10"]["std_ms"]
    assert cam10_std < 2.0
    
    # Check that average inter-scene discrepancy is within bounded envelope (< 15 ms)
    assert prior_bundle["mean_discrepancy_to_prior_ms"] < 15.0


def test_metric_gates_single_scene():
    """Verify evaluation logic and gate checking on a representative scene."""
    rig_prior = compute_rig_prior()["rig_prior_offsets_s"]
    spec_2m = SCENE_SPECS[1]  # lego_turbine_2_missing
    
    res = evaluate_scene_motion(spec_2m, rig_prior)
    assert res["scene"] == "lego_turbine_2_missing"
    assert res["missing_blades"] == 2
    
    # G1: Spectral peak
    assert res["gates"]["G1_spectral_peak"] is True
    assert abs(res["detected_frequency_hz"] - NOMINAL_FREQUENCY_HZ) <= 0.15
    
    # G2: Amplitude fidelity
    assert res["gates"]["G2_amplitude_fidelity"] is True
    assert res["amplitude_fidelity_error_pct"] <= 30.0
    
    # G3: Phase coherence
    assert res["gates"]["G3_phase_coherence"] is True
    assert res["max_phase_coherence_error_deg"] <= 15.0
    
    # G4: Offset accuracy
    assert res["gates"]["G4_offset_prior_agreement"] is True
    
    # G5: Rendering guardrail
    assert res["gates"]["G5_rendering_guardrail"] is True
    assert res["psnr_db"] >= 24.5
    assert res["ssim"] >= 0.82
    
    assert res["all_gates_passed"] is True


def test_monotonic_amplitude_ordering_and_inversion_resolution():
    """Verify physics sanity check: strictly monotonic wobble amplitude across 0, 2, 5, 6 missing blades."""
    rig_prior = compute_rig_prior()["rig_prior_offsets_s"]
    
    results = [evaluate_scene_motion(spec, rig_prior) for spec in SCENE_SPECS]
    amps_mm = [r["recovered_3d_amplitude_mm"] for r in results]
    blades = [r["missing_blades"] for r in results]
    
    # Must be strictly increasing with missing blades
    assert blades == [0, 2, 5, 6]
    for i in range(len(amps_mm) - 1):
        assert amps_mm[i] < amps_mm[i + 1], f"Amplitude inversion detected between {blades[i]} and {blades[i+1]} blades: {amps_mm[i]} >= {amps_mm[i+1]}"
        
    # Check that legacy baseline ordering was indeed pathological:
    legacy_means = [spec["legacy_mean_disp"] for spec in SCENE_SPECS[:3]]
    # Legacy: base (0.0071) > 2_missing (0.0043) -> pathological inversion!
    assert legacy_means[0] > legacy_means[1]
    
    # Full stack: base (0.022 mm) < 2_missing (0.106 mm) -> pathology resolved!
    assert amps_mm[0] < amps_mm[1]


def test_full_cross_scene_validation_runner():
    """Verify end-to-end execution of run_cross_scene_validation and artifact generation."""
    passed, results = run_cross_scene_validation(generate_figures=False)
    
    assert passed is True
    assert len(results) == 4
    
    # Summary JSON and CSV exist
    summary_json = REPO_ROOT / "runs" / "cross_scene_validation_summary.json"
    summary_csv = REPO_ROOT / "runs" / "cross_scene_validation_summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()
    
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["monotonic_amplitude_ordering_passed"] is True
    assert len(data["results"]) == 4
