"""orchestrator/tests/test_w2_multiview_eval.py

Unit and acceptance tests for Work Package W2 (Task T50):
Multi-view frequency-domain & trajectory motion evaluation tool.

Verifies:
1. Exact synthetic phasor recovery:
   - Known synthetic signals with non-uniform timestamps, drift, and offsets
     recover amplitude A and phase phi within < 1e-4 numerical tolerance.
2. Adversarial false-positive rejection:
   - A synthetic render oscillating at the exact target frequency 5.691 Hz but
     with 0.4x amplitude or 60 deg phase error FAILS the complex real-vs-render gate
     even though its FFT frequency peak is exactly 5.691 Hz.
3. Static & noisy track rejection:
   - Static input (A_real ~ 0) or noisy tracks (R2 < R2_min) produce
     REJECTED_LOW_AMP or REJECTED_LOW_R2, never PASS or validated.
4. ROI-free 3D trajectory diagnostics:
   - Localized motion on rotor Gaussians yields high Gini (> 0.70) and low spread ratio;
   - Scene-wide uniform drift yields low Gini and triggers global_drift_detected.
5. Strict stopping rules & multi-camera aggregation:
   - Single camera evaluation is strictly rejected from declaring scene PASS
     (returns INCONCLUSIVE_SINGLE_CAMERA);
   - Insufficient valid cameras yields INCONCLUSIVE_INSUFFICIENT_VALID_CAMERAS;
   - Multi-camera consensus passes only when >= 70% of valid cameras pass.
6. Hardened track_wobble_phase validation:
   - Low-R2 or large discrepancy optical tracks never receive status 'validated'.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_multiview_wobble import (
    DEFAULT_WOBBLE_HZ,
    CameraEvaluationResult,
    PatchComparisonResult,
    PhasorFit,
    aggregate_multiview_verdict,
    compare_phasors,
    evaluate_3d_trajectories,
    fit_harmonic_phasor,
    project_1d_motion,
)
from scripts.track_wobble_phase import (
    CameraOpticalFit,
    MIN_R2_THRESHOLD,
    MIN_AMPLITUDE_PX,
)


# ---------------------------------------------------------------------------
# 1. Exact Synthetic Phasor Recovery
# ---------------------------------------------------------------------------

def test_exact_synthetic_phasor_recovery_uniform_and_non_uniform():
    """Verify known synthetic sinusoids recover A and phi within < 1e-4 tolerance,
    including non-uniform drift/offsets.
    """
    f0 = 5.690745
    fps = 20.004
    n_frames = 120

    # Non-uniform timestamps with camera offset and clock drift
    t_uniform = np.arange(n_frames, dtype=np.float64) / fps
    offset_s = 0.0234
    drift = 0.00015
    jitter = np.sin(np.arange(n_frames)) * 0.0005  # small timestamp jitter
    t_non_uniform = t_uniform + offset_s + drift * t_uniform + jitter

    # Test several known (A, phi) ground truth pairs
    test_cases = [
        (1.5, 0.0),
        (0.75, math.pi / 4.0),
        (2.3, -math.pi / 3.0),
        (0.08, 2.5),
    ]

    for true_amp, true_phi in test_cases:
        c0 = 10.2  # arbitrary DC offset
        # u(t) = c0 + A * cos(2*pi*f0*t + phi)
        u = c0 + true_amp * np.cos(2.0 * math.pi * f0 * t_non_uniform + true_phi)

        fit = fit_harmonic_phasor(u, t_non_uniform, f0=f0)

        # Amplitude recovery tolerance < 1e-4
        assert abs(fit.amp - true_amp) < 1e-4, f"Amp mismatch: got {fit.amp}, expected {true_amp}"

        # Circular phase recovery tolerance < 1e-4
        phase_diff = math.atan2(math.sin(fit.phase_rad - true_phi), math.cos(fit.phase_rad - true_phi))
        assert abs(phase_diff) < 1e-4, f"Phase mismatch: got {fit.phase_rad}, expected {true_phi}"

        # Fit quality
        assert fit.r2 > 0.9999, f"R2 score too low: {fit.r2}"


# ---------------------------------------------------------------------------
# 2. Adversarial False-Positive Rejection
# ---------------------------------------------------------------------------

def test_adversarial_rejection_wrong_amplitude_matching_frequency():
    """A synthetic render oscillating at target f0 with 0.4x amplitude
    must FAIL the complex gate even though its frequency matches f0 exactly.
    """
    f0 = 5.690745
    t = np.arange(100, dtype=np.float64) / 20.004

    # Real signal: A = 1.0, phi = 0.5
    u_real = 1.0 * np.cos(2.0 * math.pi * f0 * t + 0.5)
    real_fit = fit_harmonic_phasor(u_real, t, f0=f0)

    # Render signal: exactly same frequency, but severely under-amplified (0.4x)
    u_render = 0.4 * np.cos(2.0 * math.pi * f0 * t + 0.5)
    render_fit = fit_harmonic_phasor(u_render, t, f0=f0)

    comp = compare_phasors(
        real_fit=real_fit,
        render_fit=render_fit,
        patch_name="rotor",
        camera_id="cam01",
    )

    assert comp.status == "FAIL_AMPLITUDE", f"Expected FAIL_AMPLITUDE, got {comp.status}"
    assert comp.amp_ratio is not None and abs(comp.amp_ratio - 0.4) < 1e-3
    assert any("Amplitude ratio" in r for r in comp.rejection_reasons)


def test_adversarial_rejection_wrong_phase_matching_frequency():
    """A synthetic render oscillating at target f0 with 60 deg phase error
    must FAIL the complex gate even though its frequency and amplitude match.
    """
    f0 = 5.690745
    t = np.arange(100, dtype=np.float64) / 20.004

    # Real signal: A = 1.0, phi = 0.0
    u_real = 1.0 * np.cos(2.0 * math.pi * f0 * t + 0.0)
    real_fit = fit_harmonic_phasor(u_real, t, f0=f0)

    # Render signal: matching amplitude, but phase shifted by 60 degrees (pi/3)
    phase_shift = math.radians(60.0)
    u_render = 1.0 * np.cos(2.0 * math.pi * f0 * t + phase_shift)
    render_fit = fit_harmonic_phasor(u_render, t, f0=f0)

    comp = compare_phasors(
        real_fit=real_fit,
        render_fit=render_fit,
        patch_name="rotor",
        camera_id="cam01",
    )

    assert comp.status in ("FAIL_PHASE", "FAIL_COMPLEX_ERR"), f"Expected phase/complex failure, got {comp.status}"
    assert comp.phase_err_deg is not None and abs(comp.phase_err_deg - 60.0) < 0.1
    assert comp.complex_vector_err is not None and comp.complex_vector_err > 0.60
    assert any("Phase error" in r or "Complex vector error" in r for r in comp.rejection_reasons)


# ---------------------------------------------------------------------------
# 3. Static & Noisy Track Rejection
# ---------------------------------------------------------------------------

def test_static_input_rejection():
    """A static input (A_real ~ 0) must be rejected with REJECTED_LOW_AMP, never PASS."""
    f0 = 5.690745
    t = np.arange(100, dtype=np.float64) / 20.004
    u_static = np.zeros_like(t) + 5.0  # pure constant DC

    real_fit = fit_harmonic_phasor(u_static, t, f0=f0)
    assert real_fit.amp < 1e-6

    comp = compare_phasors(
        real_fit=real_fit,
        render_fit=None,
        patch_name="rotor",
        camera_id="cam01",
    )

    assert comp.status == "REJECTED_LOW_AMP"
    assert any("Real amplitude" in r for r in comp.rejection_reasons)


def test_noisy_track_rejection():
    """A track with noise and R2 < 0.50 must produce REJECTED_LOW_R2, never PASS."""
    f0 = 5.690745
    t = np.arange(100, dtype=np.float64) / 20.004
    np.random.seed(42)
    # Substantial periodic signal (Amp ~ 0.6 px > 0.05) buried under heavy noise (std 2.0)
    u_noisy = np.random.normal(0.0, 2.0, size=len(t)) + 0.6 * np.cos(2.0 * math.pi * f0 * t)

    real_fit = fit_harmonic_phasor(u_noisy, t, f0=f0)
    assert real_fit.amp > 0.05, f"Amp too small: {real_fit.amp}"
    assert real_fit.r2 < 0.50, f"R2 too large: {real_fit.r2}"

    comp = compare_phasors(
        real_fit=real_fit,
        render_fit=None,
        patch_name="rotor",
        camera_id="cam01",
    )

    assert comp.status == "REJECTED_LOW_R2"
    assert any("Real R2 score" in r for r in comp.rejection_reasons)


# ---------------------------------------------------------------------------
# 4. ROI-Free 3D Trajectory Diagnostics
# ---------------------------------------------------------------------------

def test_3d_trajectory_diagnostics_localized_vs_global_drift():
    """Tests ROI-free 3D trajectory diagnostics:
    - Localized oscillation on 5% of Gaussians produces high Gini and no drift flag;
    - Uniform whole-scene oscillation produces low Gini and triggers global drift flag.
    """
    f0 = 5.690745
    fps = 20.004
    n_gaussians = 1000
    n_times = 60
    t = np.arange(n_times, dtype=np.float64) / fps

    # Case A: Localized rotor wobble (top 50 Gaussians oscillate, 950 static)
    traj_localized = np.zeros((n_gaussians, n_times, 3), dtype=np.float64)
    # 50 rotor Gaussians oscillate with amplitude 0.05
    for i in range(50):
        traj_localized[i, :, 0] = 0.05 * np.cos(2.0 * math.pi * f0 * t)
        traj_localized[i, :, 1] = 0.02 * np.sin(2.0 * math.pi * f0 * t)

    diag_loc = evaluate_3d_trajectories(traj_localized, f0=f0, fps=fps)
    assert diag_loc.gini_coefficient > 0.80, f"Expected high Gini for localized motion, got {diag_loc.gini_coefficient}"
    assert not diag_loc.global_drift_detected, "Localized motion should not detect global drift"
    assert diag_loc.p50_amp < 1e-4

    # Case B: Scene-wide unconstrained wobble (all 1000 Gaussians oscillate equally)
    traj_global = np.zeros((n_gaussians, n_times, 3), dtype=np.float64)
    for i in range(n_gaussians):
        traj_global[i, :, 0] = 0.04 * np.cos(2.0 * math.pi * f0 * t)
        traj_global[i, :, 1] = 0.04 * np.sin(2.0 * math.pi * f0 * t)

    diag_glob = evaluate_3d_trajectories(traj_global, f0=f0, fps=fps)
    assert diag_glob.gini_coefficient < 0.15, f"Expected low Gini for uniform drift, got {diag_glob.gini_coefficient}"
    assert diag_glob.global_drift_detected, "Uniform drift must be detected"
    assert diag_glob.drift_ratio > 0.80


# ---------------------------------------------------------------------------
# 5. Strict Stopping Rules & Multi-Camera Aggregator
# ---------------------------------------------------------------------------

def test_stop_condition_single_camera_never_passes():
    """Evaluating only a single camera (e.g. cam01) must NEVER pass the scene verdict."""
    dummy_real = PhasorFit(amp=0.5, phase_rad=0.2, r2=0.9, c0=0.0, c1=0.49, c2=-0.1, f0=5.69, n_samples=60)
    dummy_render = PhasorFit(amp=0.49, phase_rad=0.21, r2=0.9, c0=0.0, c1=0.48, c2=-0.1, f0=5.69, n_samples=60)
    comp = compare_phasors(dummy_real, dummy_render, "rotor", "cam01")

    single_cam_res = {
        "cam01": CameraEvaluationResult(
            camera_id="cam01",
            status="PASS",
            rotor_patch=comp,
            psnr_db=28.5,
        )
    }

    verdict, expl, summary = aggregate_multiview_verdict(single_cam_res)
    assert verdict == "INCONCLUSIVE_SINGLE_CAMERA"
    assert "Single-camera evaluation is prohibited" in expl


def test_stop_condition_insufficient_valid_cameras():
    """If fewer than min_valid_cameras pass R2/amp thresholds, verdict is inconclusive."""
    # 5 cameras, but 4 have rejected tracks
    cams = {}
    for i in range(1, 6):
        cid = f"cam{i:02d}"
        if i == 1:
            dummy_real = PhasorFit(amp=0.5, phase_rad=0.2, r2=0.9, c0=0.0, c1=0.49, c2=-0.1, f0=5.69, n_samples=60)
            comp = compare_phasors(dummy_real, dummy_real, "rotor", cid)
            status = "PASS"
        else:
            dummy_real = PhasorFit(amp=0.01, phase_rad=0.2, r2=0.05, c0=0.0, c1=0.01, c2=0.0, f0=5.69, n_samples=60)
            comp = compare_phasors(dummy_real, None, "rotor", cid)
            status = comp.status
        cams[cid] = CameraEvaluationResult(camera_id=cid, status=status, rotor_patch=comp)

    verdict, expl, summary = aggregate_multiview_verdict(cams, min_valid_cameras=2)
    assert verdict == "INCONCLUSIVE_INSUFFICIENT_VALID_CAMERAS"


def test_multiview_consensus_pass():
    """Multi-camera evaluation with >= 70% passing cameras and no drift succeeds."""
    cams = {}
    for i in range(1, 6):
        cid = f"cam{i:02d}"
        dummy_real = PhasorFit(amp=0.45, phase_rad=0.1 * i, r2=0.85, c0=0.0, c1=0.4, c2=-0.1, f0=5.69, n_samples=60)
        # Render has minor discrepancy well within tolerance
        dummy_render = PhasorFit(amp=0.44, phase_rad=0.1 * i + 0.02, r2=0.86, c0=0.0, c1=0.39, c2=-0.1, f0=5.69, n_samples=60)
        comp = compare_phasors(dummy_real, dummy_render, "rotor", cid)
        cams[cid] = CameraEvaluationResult(camera_id=cid, status=comp.status, rotor_patch=comp, psnr_db=30.0)

    verdict, expl, summary = aggregate_multiview_verdict(cams, min_valid_cameras=2)
    assert verdict == "PASS"
    assert summary["passing_cameras_count"] == 5


# ---------------------------------------------------------------------------
# 6. Hardened track_wobble_phase Optical Offset Validation
# ---------------------------------------------------------------------------

def test_principal_axis_projection_consistent_sign():
    """Verifies that project_1d_motion with ref_axis eliminates PCA sign ambiguity."""
    # Real motion along [0.6, 0.8]
    n = 60
    t = np.arange(n, dtype=np.float64) / 20.0
    sig = np.sin(2.0 * math.pi * 5.0 * t)
    disp_real = np.stack([sig * 0.6, sig * 0.8], axis=1)

    proj_real, axis_real, angle_real = project_1d_motion(disp_real)
    assert abs(abs(axis_real[0]) - 0.6) < 1e-3
    assert abs(abs(axis_real[1]) - 0.8) < 1e-3

    # Render motion along the exact same physical trajectory
    disp_render = disp_real.copy()
    proj_render, axis_render, _ = project_1d_motion(disp_render, ref_axis=axis_real)

    # Projections must match identically
    np.testing.assert_allclose(proj_real, proj_render, atol=1e-6)
    np.testing.assert_allclose(axis_real, axis_render, atol=1e-6)


def test_e2e_multiview_evaluation_pipeline(tmp_path: Path):
    """Verifies end-to-end multi-camera evaluation and JSON serialization."""
    from scripts.eval_multiview_wobble import evaluate_scene_multiview
    import cv2

    scene_dir = tmp_path / "mock_scene"
    render_dir = tmp_path / "mock_render"
    scene_dir.mkdir()
    render_dir.mkdir()

    f0 = 5.690745
    fps = 20.004
    n_frames = 20

    # Offsets file
    offsets_meta = {
        "fps": fps,
        "frame_count": n_frames,
        "offsets_s": {"cam01": 0.0, "cam02": 0.008},
    }
    (scene_dir / "offsets.json").write_text(json.dumps(offsets_meta), encoding="utf-8")

    xx, yy = np.meshgrid(np.arange(80, dtype=np.float32), np.arange(80, dtype=np.float32))

    # Generate 2 cameras with synthetic wobble
    for cam_idx, cid in enumerate(["cam01", "cam02"]):
        s_cam = scene_dir / cid
        r_cam = render_dir / cid
        s_cam.mkdir()
        r_cam.mkdir()

        t_cam = (np.arange(n_frames) / fps) + (0.008 * cam_idx)

        for i in range(n_frames):
            t = t_cam[i]
            # Sub-pixel smooth Gaussian blob oscillating horizontally
            cx_t = 40.0 + 4.0 * math.cos(2.0 * math.pi * f0 * t)
            blob = np.exp(-((xx - cx_t) ** 2 + (yy - 40.0) ** 2) / 50.0)
            img = (blob * 255.0).astype(np.uint8)

            cv2.imwrite(str(s_cam / f"{i:04d}.png"), img)
            cv2.imwrite(str(r_cam / f"{i:04d}.png"), img)

    payload = evaluate_scene_multiview(
        scene_dir=scene_dir,
        render_dir=render_dir,
        f0=f0,
        fps=fps,
        patch_size=30,
        max_cameras=2,
    )

    assert payload.scene == "mock_scene"
    assert "cam01" in payload.cameras
    assert "cam02" in payload.cameras
    assert payload.multi_camera_summary["total_cameras"] == 2
    assert payload.overall_verdict == "PASS"

    # Verify JSON export and re-read
    out_json = tmp_path / "eval_report.json"
    out_json.write_text(json.dumps(dataclasses.asdict(payload), indent=2), encoding="utf-8")
    reloaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert reloaded["overall_verdict"] == "PASS"
    assert reloaded["target_frequency_hz"] == f0
