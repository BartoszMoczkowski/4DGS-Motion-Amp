#!/usr/bin/env python3
"""scripts/eval_multiview_wobble.py

Work Package W2 (Task T50): Multi-view frequency-domain & trajectory motion evaluation tool.

Gated multi-camera motion evaluation tool for subtle periodic motion (turbine wobble)
in 4DGS reconstructions.

Features:
1. Exact corrected per-camera timebases (offsets, drifts, non-uniform intervals).
2. Least-squares harmonic complex phasor estimation Z = A * exp(i * phi) at target f0.
3. Principal-axis 1D projection with sign alignment against real reference tracks.
4. Per-patch and per-camera phasor, R2, PSNR, SSIM, and temporal std (tstd) metrics.
5. Quality gating: rejects low-R2 tracks, static noise, track divergence, and adversarial
   false-positive frequency matches (wrong amplitude/phase despite matching f0).
6. ROI-free 3D Gaussian trajectory diagnostics (Gini concentration, p50/p90/p99 amplitudes,
   top-decile spatial spread, phase coherence, and background drift check).
7. Strict stopping rules: never passes from a single camera or FFT peak alone;
   never infers timing-sidecar error from rejected optical tracks.

Usage examples:
  # Evaluate pre-rendered directory against real scene:
  uv run python scripts/eval_multiview_wobble.py \\
      --scene data/multipleview/turbine_r1_missing_6 \\
      --render-dir runs/turbine_r1_missing_6_w1_e1a/render_out \\
      --out-json runs/turbine_r1_missing_6_w1_e1a/eval_w2.json \\
      --human

  # Evaluate with 3D trajectories:
  uv run python scripts/eval_multiview_wobble.py \\
      --scene data/multipleview/turbine_r1_missing_6 \\
      --render-dir runs/turbine_r1_missing_6_w1_e1a/render_out \\
      --trajectories runs/turbine_r1_missing_6_w1_e1a/trajectories.npz \\
      --out-json runs/turbine_r1_missing_6_w1_e1a/eval_w2.json
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import asdict, dataclass, field
import datetime
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Generator, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WOBBLE_HZ = 5.690745121784875
DEFAULT_FPS = 20.004025315610498
DEFAULT_PATCH_SIZE = 100
DEFAULT_MIN_R2 = 0.50
DEFAULT_MIN_AMP_PX = 0.05
DEFAULT_MAX_PHASE_ERR_DEG = 45.0
DEFAULT_MAX_COMPLEX_ERR = 0.60
DEFAULT_AMP_RATIO_MIN = 0.50
DEFAULT_AMP_RATIO_MAX = 2.00
DEFAULT_MIN_VALID_CAMERAS = 2


@dataclass
class PhasorFit:
    """Analytical least-squares fit to u(t) = c0 + c1*cos(w*t) + c2*sin(w*t).
    Phasor: Z = c1 - 1j*c2 = A * exp(1j * phi).
    Re(Z * exp(1j * w * t)) = c1*cos(w*t) + c2*sin(w*t) = A*cos(w*t + phi).
    """
    amp: float
    phase_rad: float
    r2: float
    c0: float
    c1: float
    c2: float
    f0: float
    n_samples: int

    @property
    def Z(self) -> complex:
        return complex(self.c1, -self.c2)

    @property
    def phase_deg(self) -> float:
        return float(math.degrees(self.phase_rad))


@dataclass
class PatchComparisonResult:
    patch_name: str
    camera_id: str
    status: str  # VALID, PASS, FAIL_AMPLITUDE, FAIL_PHASE, FAIL_COMPLEX_ERR, REJECTED_LOW_R2, REJECTED_LOW_AMP, REJECTED_DIVERGENCE
    real_fit: PhasorFit
    render_fit: PhasorFit | None = None
    amp_ratio: float | None = None
    abs_amp_err: float | None = None
    phase_diff_rad: float | None = None
    phase_err_deg: float | None = None
    phase_time_err_ms: float | None = None
    complex_vector_err: float | None = None
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class CameraEvaluationResult:
    camera_id: str
    status: str
    rotor_patch: PatchComparisonResult | None = None
    background_patches: list[PatchComparisonResult] = field(default_factory=list)
    psnr_db: float | None = None
    ssim: float | None = None
    tstd_ratio_rotor: float | None = None
    tstd_ratio_bg: float | None = None
    sample_count: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class Trajectory3DDiagnostics:
    n_gaussians: int
    n_timesteps: int
    f0_hz: float
    p50_amp: float
    p90_amp: float
    p99_amp: float
    max_amp: float
    gini_coefficient: float
    top_decile_spread_ratio: float
    phase_coherence_r: float
    bg_displacement_p50: float
    drift_ratio: float
    global_drift_detected: bool
    summary: str


@dataclass
class MultiviewEvaluationPayload:
    eval_tool: str
    version: str
    created_at: str
    scene: str
    render_source: str
    target_frequency_hz: float
    fps: float
    overall_verdict: str
    verdict_explanation: str
    multi_camera_summary: dict[str, Any]
    cameras: dict[str, Any]
    trajectory_3d: dict[str, Any] | None = None
    timing_provenance: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 1. Harmonic Least-Squares Phasor Fit
# ---------------------------------------------------------------------------

def fit_harmonic_phasor(
    u: np.ndarray,
    t: np.ndarray,
    f0: float,
) -> PhasorFit:
    """Least-squares harmonic projection at f0 on arbitrary (non-uniform) timestamps t.
    Solves u(t) ~ c0 + c1*cos(2*pi*f0*t) + c2*sin(2*pi*f0*t).
    Returns PhasorFit with exact amplitude and circularly signed phase.
    """
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    n = len(u_arr)
    if n != len(t_arr):
        raise ValueError(f"Signal length ({n}) does not match time length ({len(t_arr)})")
    if n < 3:
        raise ValueError(f"At least 3 samples required for harmonic fit, got {n}")

    omega = 2.0 * np.pi * f0
    cos_wt = np.cos(omega * t_arr)
    sin_wt = np.sin(omega * t_arr)
    ones = np.ones(n, dtype=np.float64)

    X = np.stack([ones, cos_wt, sin_wt], axis=1)
    coeffs, residuals, rank, s = np.linalg.lstsq(X, u_arr, rcond=None)
    c0, c1, c2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    pred = X @ coeffs
    ss_tot = float(np.sum((u_arr - np.mean(u_arr)) ** 2))
    ss_res = float(np.sum((u_arr - pred) ** 2))

    if ss_tot < 1e-14:
        # Perfectly flat / static signal
        r2 = 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    amp = float(math.sqrt(c1 * c1 + c2 * c2))
    # u(t) = c0 + c1*cos(w*t) + c2*sin(w*t) = c0 + A*cos(w*t + phi)
    # where c1 = A*cos(phi), c2 = -A*sin(phi) => phi = atan2(-c2, c1)
    phi = float(math.atan2(-c2, c1))

    return PhasorFit(
        amp=amp,
        phase_rad=phi,
        r2=r2,
        c0=c0,
        c1=c1,
        c2=c2,
        f0=float(f0),
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# 2. 1D Principal Axis Projection with Consistent Reference Sign
# ---------------------------------------------------------------------------

def project_1d_motion(
    disp_2d: np.ndarray,
    ref_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Projects 2D displacement tracks (N, 2) onto 1D principal axis.
    If ref_axis is provided (from the real reference track), projects along ref_axis
    to maintain a consistent physical sign convention across real and render.
    Returns: (proj_1d [N], unit_axis [2], angle_deg).
    """
    disp = np.asarray(disp_2d, dtype=np.float64)
    if disp.ndim != 2 or disp.shape[1] != 2:
        raise ValueError(f"Expected disp_2d of shape (N, 2), got {disp.shape}")

    n = len(disp)
    d = disp - disp.mean(axis=0, keepdims=True)

    if ref_axis is not None:
        axis = np.asarray(ref_axis, dtype=np.float64).reshape(-1)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            axis = np.array([1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
    else:
        cov = (d.T @ d) / max(n, 1)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
        else:
            axis = np.array([1.0, 0.0], dtype=np.float64)

    proj = d @ axis
    angle_deg = float(math.degrees(math.atan2(axis[1], axis[0])))
    return proj, axis, angle_deg


# ---------------------------------------------------------------------------
# 3. Phasor Comparison & Quality Gating
# ---------------------------------------------------------------------------

def compare_phasors(
    real_fit: PhasorFit,
    render_fit: PhasorFit | None,
    patch_name: str,
    camera_id: str,
    min_r2: float = DEFAULT_MIN_R2,
    min_amp_px: float = DEFAULT_MIN_AMP_PX,
    amp_ratio_bounds: tuple[float, float] = (DEFAULT_AMP_RATIO_MIN, DEFAULT_AMP_RATIO_MAX),
    max_phase_err_deg: float = DEFAULT_MAX_PHASE_ERR_DEG,
    max_complex_err: float = DEFAULT_MAX_COMPLEX_ERR,
    is_background: bool = False,
) -> PatchComparisonResult:
    """Compares real and render harmonic phasors under strict quality gating.
    Catches adversarial false-positives (exact f0 peak but wrong amplitude or phase),
    and rejects noisy/static real tracks.
    """
    reasons = []

    # Check for NaN / divergence
    if not math.isfinite(real_fit.amp) or not math.isfinite(real_fit.phase_rad):
        return PatchComparisonResult(
            patch_name=patch_name,
            camera_id=camera_id,
            status="REJECTED_DIVERGENCE",
            real_fit=real_fit,
            render_fit=render_fit,
            rejection_reasons=["Real fit contains non-finite parameters"],
        )

    # Background patches are expected to have near-zero real amplitude and no wobble
    if is_background:
        if render_fit is None:
            return PatchComparisonResult(
                patch_name=patch_name,
                camera_id=camera_id,
                status="NO_RENDER",
                real_fit=real_fit,
                render_fit=None,
            )
        # For background: render amplitude must be negligible
        bg_max_amp = min_amp_px * 2.0
        if render_fit.amp > bg_max_amp:
            return PatchComparisonResult(
                patch_name=patch_name,
                camera_id=camera_id,
                status="FAIL_BACKGROUND_MOTION",
                real_fit=real_fit,
                render_fit=render_fit,
                rejection_reasons=[f"Background render amplitude {render_fit.amp:.3f} px > {bg_max_amp:.3f} px"],
            )
        return PatchComparisonResult(
            patch_name=patch_name,
            camera_id=camera_id,
            status="PASS",
            real_fit=real_fit,
            render_fit=render_fit,
            rejection_reasons=[],
        )

    # Real track validation
    if real_fit.amp < min_amp_px:
        reasons.append(f"Real amplitude {real_fit.amp:.4f} px < threshold {min_amp_px:.2f} px")
        return PatchComparisonResult(
            patch_name=patch_name,
            camera_id=camera_id,
            status="REJECTED_LOW_AMP",
            real_fit=real_fit,
            render_fit=render_fit,
            rejection_reasons=reasons,
        )

    if real_fit.r2 < min_r2:
        reasons.append(f"Real R2 score {real_fit.r2:.3f} < threshold {min_r2:.2f}")
        return PatchComparisonResult(
            patch_name=patch_name,
            camera_id=camera_id,
            status="REJECTED_LOW_R2",
            real_fit=real_fit,
            render_fit=render_fit,
            rejection_reasons=reasons,
        )

    if render_fit is None:
        return PatchComparisonResult(
            patch_name=patch_name,
            camera_id=camera_id,
            status="VALID_REAL_NO_RENDER",
            real_fit=real_fit,
            render_fit=None,
        )

    # Render fit check
    if render_fit.r2 < min_r2:
        reasons.append(f"Render R2 score {render_fit.r2:.3f} < threshold {min_r2:.2f}")

    amp_ratio = float(render_fit.amp / (real_fit.amp + 1e-12))
    abs_amp_err = float(abs(render_fit.amp - real_fit.amp))

    # Circular phase difference: render - real wrapped to [-pi, pi]
    phase_diff = float(math.atan2(
        math.sin(render_fit.phase_rad - real_fit.phase_rad),
        math.cos(render_fit.phase_rad - real_fit.phase_rad),
    ))
    phase_err_deg = float(abs(math.degrees(phase_diff)))
    phase_time_err_ms = float((phase_diff / (2.0 * math.pi * real_fit.f0)) * 1000.0)

    # Complex vector error: |Z_render - Z_real| / |Z_real|
    Z_real = real_fit.Z
    Z_render = render_fit.Z
    complex_err = float(abs(Z_render - Z_real) / (abs(Z_real) + 1e-12))

    # Gating checks
    if amp_ratio < amp_ratio_bounds[0] or amp_ratio > amp_ratio_bounds[1]:
        reasons.append(f"Amplitude ratio {amp_ratio:.2f} outside [{amp_ratio_bounds[0]}, {amp_ratio_bounds[1]}]")

    if phase_err_deg > max_phase_err_deg:
        reasons.append(f"Phase error {phase_err_deg:.1f} deg > {max_phase_err_deg:.1f} deg")

    if complex_err > max_complex_err:
        reasons.append(f"Complex vector error {complex_err:.3f} > {max_complex_err:.2f}")

    if reasons:
        # Determine specific failure status
        if amp_ratio < amp_ratio_bounds[0] or amp_ratio > amp_ratio_bounds[1]:
            status = "FAIL_AMPLITUDE"
        elif phase_err_deg > max_phase_err_deg:
            status = "FAIL_PHASE"
        elif complex_err > max_complex_err:
            status = "FAIL_COMPLEX_ERR"
        else:
            status = "FAIL_QUALITY"
    else:
        status = "PASS"

    return PatchComparisonResult(
        patch_name=patch_name,
        camera_id=camera_id,
        status=status,
        real_fit=real_fit,
        render_fit=render_fit,
        amp_ratio=amp_ratio,
        abs_amp_err=abs_amp_err,
        phase_diff_rad=phase_diff,
        phase_err_deg=phase_err_deg,
        phase_time_err_ms=phase_time_err_ms,
        complex_vector_err=complex_err,
        rejection_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# 4. Patch Tracking & Image Metrics
# ---------------------------------------------------------------------------

def track_patch_displacements(
    frame_generator: Generator[np.ndarray, None, None] | Sequence[np.ndarray],
    cx: int,
    cy: int,
    patch_size: int,
) -> tuple[np.ndarray, list[float]]:
    """Tracks sub-pixel displacement of a patch across frames using phaseCorrelate.
    Takes a generator of grayscale float32 [0, 1] frames.
    Returns: (displacements [N, 2], correlation_responses [N]).
    """
    h = patch_size // 2
    win = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)
    displacements = []
    responses = []
    tpl = None

    for idx, fr in enumerate(frame_generator):
        img_h, img_w = fr.shape
        y0, y1 = cy - h, cy + h
        x0, x1 = cx - h, cx + h
        if y0 < 0 or y1 > img_h or x0 < 0 or x1 > img_w:
            raise ValueError(f"Patch ({cx}, {cy}, size={patch_size}) exceeds frame dimensions ({img_w}x{img_h})")

        patch = fr[y0:y1, x0:x1]
        if tpl is None:
            tpl = patch.copy()
            displacements.append((0.0, 0.0))
            responses.append(1.0)
            continue

        (dx, dy), resp = cv2.phaseCorrelate(tpl, patch, win)
        displacements.append((float(dx), float(dy)))
        responses.append(float(resp))

    return np.array(displacements, dtype=np.float64), responses


def compute_image_metrics(
    real_generator: Generator[np.ndarray, None, None] | Sequence[np.ndarray],
    render_generator: Generator[np.ndarray, None, None] | Sequence[np.ndarray],
) -> tuple[float, float, float, float]:
    """Computes held-out PSNR, SSIM, and temporal standard-deviation stats across matching series.
    Consumes frames streaming one-by-one to avoid 6 MP accumulation.
    Returns: (mean_psnr_db, mean_ssim, real_temporal_std, render_temporal_std).
    """
    psnr_vals = []
    ssim_vals = []

    sum_real = None
    sum_sq_real = None
    sum_ren = None
    sum_sq_ren = None
    count = 0

    for r_img, g_img in zip(real_generator, render_generator):
        r = np.asarray(r_img, dtype=np.float32)
        g = np.asarray(g_img, dtype=np.float32)

        # Ensure matching shape
        if r.shape != g.shape:
            g = cv2.resize(g, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)

        diff = r - g
        mse = float(np.mean(diff * diff))
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = float(10.0 * np.log10(1.0 / mse))
        psnr_vals.append(psnr)

        # Lightweight luminance-contrast SSIM
        mu_r = float(np.mean(r))
        mu_g = float(np.mean(g))
        var_r = float(np.var(r))
        var_g = float(np.var(g))
        cov_rg = float(np.mean((r - mu_r) * (g - mu_g)))
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        ssim = float(((2 * mu_r * mu_g + c1) * (2 * cov_rg + c2)) /
                     ((mu_r**2 + mu_g**2 + c1) * (var_r + var_g + c2)))
        ssim_vals.append(ssim)

        # Accumulators for temporal standard deviation
        if sum_real is None:
            sum_real = np.zeros_like(r, dtype=np.float64)
            sum_sq_real = np.zeros_like(r, dtype=np.float64)
            sum_ren = np.zeros_like(g, dtype=np.float64)
            sum_sq_ren = np.zeros_like(g, dtype=np.float64)

        sum_real += r
        sum_sq_real += r * r
        sum_ren += g
        sum_sq_ren += g * g
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_psnr = float(np.mean(psnr_vals))
    mean_ssim = float(np.mean(ssim_vals))

    var_real_map = np.maximum(0.0, (sum_sq_real / count) - (sum_real / count) ** 2)
    var_ren_map = np.maximum(0.0, (sum_sq_ren / count) - (sum_ren / count) ** 2)

    mean_tstd_real = float(np.mean(np.sqrt(var_real_map)))
    mean_tstd_ren = float(np.mean(np.sqrt(var_ren_map)))

    return mean_psnr, mean_ssim, mean_tstd_real, mean_tstd_ren


# ---------------------------------------------------------------------------
# 5. ROI-free 3D Native-HexPlane Trajectory Diagnostics
# ---------------------------------------------------------------------------

def evaluate_3d_trajectories(
    traj_3d: np.ndarray,
    f0: float,
    fps: float = DEFAULT_FPS,
    times: np.ndarray | None = None,
    canonical_xyz: np.ndarray | None = None,
) -> Trajectory3DDiagnostics:
    """ROI-free 3D native-HexPlane trajectory diagnostics:
    Evaluates Gini concentration, p50/p90/p99 displacement amplitudes,
    top-decile spatial concentration, circular phase coherence, and global drift.
    """
    traj = np.asarray(traj_3d, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"Expected traj_3d of shape (N, T, 3), got {traj.shape}")

    n_gaussians, n_timesteps, _ = traj.shape
    if times is None:
        t_arr = np.arange(n_timesteps, dtype=np.float64) / fps
    else:
        t_arr = np.asarray(times, dtype=np.float64).reshape(-1)

    omega = 2.0 * np.pi * f0
    cos_wt = np.cos(omega * t_arr)
    sin_wt = np.sin(omega * t_arr)
    ones = np.ones(n_timesteps, dtype=np.float64)

    # Basis: (T, 3)
    X = np.stack([ones, cos_wt, sin_wt], axis=1)
    # Solve X_pinv: (3, T)
    X_pinv = np.linalg.pinv(X)

    # Vectorized least-squares for each dimension (N, T) @ (T, 3) -> (N, 3)
    # Coefficients: [c0, c1, c2]
    Cx = traj[:, :, 0] @ X_pinv.T
    Cy = traj[:, :, 1] @ X_pinv.T
    Cz = traj[:, :, 2] @ X_pinv.T

    # 3D harmonic displacement amplitude for each Gaussian
    amp_x2 = Cx[:, 1] ** 2 + Cx[:, 2] ** 2
    amp_y2 = Cy[:, 1] ** 2 + Cy[:, 2] ** 2
    amp_z2 = Cz[:, 1] ** 2 + Cz[:, 2] ** 2
    amp_3d = np.sqrt(amp_x2 + amp_y2 + amp_z2)

    p50 = float(np.percentile(amp_3d, 50))
    p90 = float(np.percentile(amp_3d, 90))
    p99 = float(np.percentile(amp_3d, 99))
    max_amp = float(np.max(amp_3d))

    # Gini coefficient of displacement amplitudes
    sorted_amp = np.sort(amp_3d)
    tot = float(np.sum(sorted_amp))
    if tot < 1e-12:
        gini = 0.0
    else:
        idx = np.arange(1, n_gaussians + 1, dtype=np.float64)
        gini = float((2.0 * np.sum(idx * sorted_amp)) / (n_gaussians * tot) - (n_gaussians + 1.0) / n_gaussians)

    # Top-decile spatial concentration
    top_thresh = p90
    top_mask = amp_3d >= top_thresh
    n_top = int(np.sum(top_mask))

    if canonical_xyz is not None and len(canonical_xyz) == n_gaussians and n_top > 1:
        canon = np.asarray(canonical_xyz, dtype=np.float64)
        all_std = float(np.mean(np.std(canon, axis=0)))
        top_std = float(np.mean(np.std(canon[top_mask], axis=0)))
        spread_ratio = float(top_std / (all_std + 1e-8))
    else:
        spread_ratio = 1.0

    # Phase coherence among top movers along their principal vibration axes
    # Vibration direction: [c1x, c1y, c1z]
    if n_top > 0:
        c1_top = np.stack([Cx[top_mask, 1], Cy[top_mask, 1], Cz[top_mask, 1]], axis=1)
        c2_top = np.stack([Cx[top_mask, 2], Cy[top_mask, 2], Cz[top_mask, 2]], axis=1)
        # Principal axis projection of cosine and sine components
        dir_norms = np.linalg.norm(c1_top, axis=1, keepdims=True) + 1e-8
        v_top = c1_top / dir_norms
        proj_c1 = np.sum(c1_top * v_top, axis=1)
        proj_c2 = np.sum(c2_top * v_top, axis=1)
        phi_top = np.arctan2(-proj_c2, proj_c1)
        # Circular mean resultant vector length R in [0, 1]
        R_coherence = float(abs(np.mean(np.exp(1j * phi_top))))
    else:
        R_coherence = 0.0

    # Background / global drift check
    drift_ratio = float(p50 / (p90 + 1e-8))
    # Drift is flagged if bottom 50% Gaussians move with significant amplitude relative to top
    drift_detected = bool(p50 > 0.002 and drift_ratio > 0.25)

    summary = (
        f"3D Motion @ {f0:.3f} Hz: p50={p50:.5f}, p90={p90:.5f}, p99={p99:.5f}, "
        f"Gini={gini:.3f}, top10_spread_ratio={spread_ratio:.3f}, "
        f"phase_coherence={R_coherence:.3f}, bg_drift_ratio={drift_ratio:.3f} "
        f"({'GLOBAL DRIFT DETECTED' if drift_detected else 'ok'})"
    )

    return Trajectory3DDiagnostics(
        n_gaussians=n_gaussians,
        n_timesteps=n_timesteps,
        f0_hz=float(f0),
        p50_amp=p50,
        p90_amp=p90,
        p99_amp=p99,
        max_amp=max_amp,
        gini_coefficient=gini,
        top_decile_spread_ratio=spread_ratio,
        phase_coherence_r=R_coherence,
        bg_displacement_p50=p50,
        drift_ratio=drift_ratio,
        global_drift_detected=drift_detected,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 6. Multi-Camera Aggregator & Scene Verdict Gate
# ---------------------------------------------------------------------------

def aggregate_multiview_verdict(
    camera_results: dict[str, CameraEvaluationResult],
    traj_diag: Trajectory3DDiagnostics | None = None,
    min_valid_cameras: int = DEFAULT_MIN_VALID_CAMERAS,
) -> tuple[str, str, dict[str, Any]]:
    """Synthesizes the overall scene verdict under Phase 14 quality gates.

    Strict rules:
    - Never declares PASS from a single camera alone.
    - Never declares PASS from an FFT frequency peak alone.
    - Fails if 3D global drift is detected.
    - Requires valid optical tracks and passing complex vector error across cameras.
    """
    total_cameras = len(camera_results)
    if total_cameras == 0:
        return "INCONCLUSIVE_NO_DATA", "No cameras provided for evaluation", {}

    valid_cams = [cid for cid, res in camera_results.items() if res.rotor_patch and res.rotor_patch.status != "REJECTED_LOW_R2" and res.rotor_patch.status != "REJECTED_LOW_AMP" and res.rotor_patch.status != "REJECTED_DIVERGENCE"]
    passing_cams = [cid for cid, res in camera_results.items() if res.status == "PASS"]

    # Gather metrics across valid cameras
    amp_ratios = []
    phase_errs = []
    complex_errs = []
    psnrs = []
    sin_phases = []
    cos_phases = []

    for cid in valid_cams:
        res = camera_results[cid]
        rp = res.rotor_patch
        if rp and rp.amp_ratio is not None:
            amp_ratios.append(rp.amp_ratio)
            phase_errs.append(rp.phase_err_deg)
            complex_errs.append(rp.complex_vector_err)
            if rp.phase_diff_rad is not None:
                sin_phases.append(math.sin(rp.phase_diff_rad))
                cos_phases.append(math.cos(rp.phase_diff_rad))
        if res.psnr_db is not None:
            psnrs.append(res.psnr_db)

    mean_amp_ratio = float(np.mean(amp_ratios)) if amp_ratios else None
    mean_phase_err = float(np.mean(phase_errs)) if phase_errs else None
    mean_complex_err = float(np.mean(complex_errs)) if complex_errs else None
    mean_psnr = float(np.mean(psnrs)) if psnrs else None

    # Circular mean phase difference across rig
    if sin_phases and cos_phases:
        circ_mean_phase_deg = float(math.degrees(math.atan2(np.mean(sin_phases), np.mean(cos_phases))))
    else:
        circ_mean_phase_deg = None

    summary_stats = {
        "total_cameras": total_cameras,
        "valid_cameras_count": len(valid_cams),
        "passing_cameras_count": len(passing_cams),
        "mean_amp_ratio": mean_amp_ratio,
        "mean_phase_err_deg": mean_phase_err,
        "circular_mean_phase_err_deg": circ_mean_phase_deg,
        "mean_complex_vector_err": mean_complex_err,
        "mean_psnr_db": mean_psnr,
    }

    # GATING DECISION TREE
    # 1. Stop condition: Single-camera evaluation is never sufficient for scene pass
    if total_cameras == 1:
        return (
            "INCONCLUSIVE_SINGLE_CAMERA",
            "Single-camera evaluation is prohibited by W2 gate from declaring scene pass.",
            summary_stats,
        )

    # 2. Insufficient valid optical tracks
    if len(valid_cams) < min_valid_cameras:
        return (
            "INCONCLUSIVE_INSUFFICIENT_VALID_CAMERAS",
            f"Only {len(valid_cams)}/{total_cameras} cameras had valid optical tracks (R2 >= {DEFAULT_MIN_R2}, Amp >= {DEFAULT_MIN_AMP_PX}px). Required >= {min_valid_cameras}.",
            summary_stats,
        )

    # 3. 3D Global drift check
    if traj_diag is not None and traj_diag.global_drift_detected:
        return (
            "FAIL_GLOBAL_DRIFT",
            f"ROI-free 3D diagnostics detected unconstrained scene-wide drift (drift_ratio={traj_diag.drift_ratio:.3f}).",
            summary_stats,
        )

    # 4. Multi-camera complex phasor pass rate
    pass_fraction = len(passing_cams) / len(valid_cams)
    if pass_fraction < 0.70:
        if mean_complex_err is not None and mean_complex_err > DEFAULT_MAX_COMPLEX_ERR:
            return (
                "FAIL_COMPLEX_PHASOR",
                f"Multi-camera complex vector error {mean_complex_err:.3f} exceeds threshold {DEFAULT_MAX_COMPLEX_ERR:.2f} (pass fraction: {pass_fraction:.1%}).",
                summary_stats,
            )
        elif mean_amp_ratio is not None and (mean_amp_ratio < DEFAULT_AMP_RATIO_MIN or mean_amp_ratio > DEFAULT_AMP_RATIO_MAX):
            return (
                "FAIL_AMPLITUDE",
                f"Multi-camera mean amplitude ratio {mean_amp_ratio:.2f} outside bounds [{DEFAULT_AMP_RATIO_MIN}, {DEFAULT_AMP_RATIO_MAX}].",
                summary_stats,
            )
        else:
            return (
                "FAIL_PHASE",
                f"Multi-camera mean phase error {mean_phase_err:.1f} deg exceeds threshold {DEFAULT_MAX_PHASE_ERR_DEG:.1f} deg.",
                summary_stats,
            )

    return (
        "PASS",
        f"Multi-camera motion validated across {len(passing_cams)}/{len(valid_cams)} valid cameras (mean amp ratio: {mean_amp_ratio:.2f}, phase err: {mean_phase_err:.1f} deg).",
        summary_stats,
    )


# ---------------------------------------------------------------------------
# 7. Loading & Frame Streaming Helpers
# ---------------------------------------------------------------------------

def load_timestamps(
    scene_dir: Path,
    camera_ids: list[str],
    fps: float = DEFAULT_FPS,
    n_frames: int = 60,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Loads exact physical timestamps for each camera from offsets.json,
    accounting for individual exposure offsets and clock drift.
    """
    offsets_file = scene_dir / "offsets.json"
    provenance = {"offsets_file": str(offsets_file), "has_offsets": False}
    timestamps: dict[str, np.ndarray] = {}

    offsets_dict = {}
    drift_dict = {}

    if offsets_file.exists():
        try:
            data = json.loads(offsets_file.read_text(encoding="utf-8"))
            offsets_dict = data.get("offsets_s", {})
            drift_dict = data.get("drift_s_per_s", {})
            fps = float(data.get("fps", fps))
            n_frames = int(data.get("frame_count", n_frames))
            provenance["has_offsets"] = True
            provenance["fps"] = fps
            provenance["frame_count"] = n_frames
            import hashlib
            provenance["offsets_sha256"] = hashlib.sha256(offsets_file.read_bytes()).hexdigest()
        except Exception as exc:
            print(f"[WARN] Failed to parse {offsets_file}: {exc}", file=sys.stderr)

    for cid in camera_ids:
        offset_s = float(offsets_dict.get(cid, 0.0))
        drift = float(drift_dict.get(cid, 0.0))
        indices = np.arange(n_frames, dtype=np.float64)
        # Physical time: t = i/fps + offset_s + drift * (i/fps)
        t_cam = (indices / fps) + offset_s + drift * (indices / fps)
        timestamps[cid] = t_cam

    return timestamps, provenance


def stream_disk_series(frame_paths: list[Path]) -> Generator[np.ndarray, None, None]:
    """Streams grayscale float32 images one-by-one from disk."""
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image at {p}")
        yield (img.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# 8. Main Multi-Camera Evaluator
# ---------------------------------------------------------------------------

def evaluate_scene_multiview(
    scene_dir: Path,
    render_dir: Path,
    f0: float = DEFAULT_WOBBLE_HZ,
    fps: float = DEFAULT_FPS,
    trajectories_file: Path | None = None,
    min_r2: float = DEFAULT_MIN_R2,
    min_amp_px: float = DEFAULT_MIN_AMP_PX,
    max_cameras: int = 10,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> MultiviewEvaluationPayload:
    """Evaluates a full multi-camera capture and rendered reconstruction."""
    scene_dir = Path(scene_dir).resolve()
    render_dir = Path(render_dir).resolve()

    # Discover camera directories
    cam_ids = [f"cam{i:02d}" for i in range(1, max_cameras + 1)]
    avail_real_cams = [cid for cid in cam_ids if (scene_dir / cid).is_dir()]
    if not avail_real_cams:
        raise FileNotFoundError(f"No camera folders (cam01..) found in real scene dir: {scene_dir}")

    # Discover render directory layout
    timestamps, time_prov = load_timestamps(scene_dir, avail_real_cams, fps=fps)

    camera_evals: dict[str, CameraEvaluationResult] = {}

    for cid in avail_real_cams:
        real_cam_dir = scene_dir / cid
        real_frame_files = sorted(list(real_cam_dir.glob("*.jpg")) + list(real_cam_dir.glob("*.png")))
        if not real_frame_files:
            continue

        # Look for render frames in render_dir/cid or render_dir/cid_series or render_dir
        render_cam_dir = render_dir / cid
        if not render_cam_dir.is_dir():
            render_cam_dir = render_dir / f"{cid}_series"
        if not render_cam_dir.is_dir() and len(avail_real_cams) == 1:
            render_cam_dir = render_dir

        has_render = render_cam_dir.is_dir()
        render_frame_files = sorted(list(render_cam_dir.glob("*.png")) + list(render_cam_dir.glob("*.jpg"))) if has_render else []

        n_frames = min(len(real_frame_files), len(render_frame_files)) if render_frame_files else len(real_frame_files)
        if n_frames < 10:
            continue

        real_frames_subset = real_frame_files[:n_frames]
        ren_frames_subset = render_frame_files[:n_frames] if render_frame_files else []
        t_arr = timestamps[cid][:n_frames]

        # Read first frame to select patches
        first_img = cv2.imread(str(real_frames_subset[0]), cv2.IMREAD_GRAYSCALE)
        h, w = first_img.shape

        # Default rotor center: middle of frame
        cx, cy = w // 2, h // 2
        # Check if motion mask exists for better rotor patch location
        mask_path = scene_dir / "motion_masks" / f"{cid}.npy"
        if mask_path.exists():
            try:
                mask = np.load(str(mask_path))
                if mask.shape == (h, w) and np.sum(mask) > 0:
                    integral = cv2.integral(mask.astype(np.float64))
                    ph = patch_size // 2
                    best_sum, best_pos = -1.0, (cx, cy)
                    for py in range(ph, h - ph, 20):
                        for px in range(ph, w - ph, 20):
                            s = (integral[py + ph, px + ph] - integral[py - ph, px + ph]
                                 - integral[py + ph, px - ph] + integral[py - ph, px - ph])
                            if s > best_sum:
                                best_sum = s
                                best_pos = (px, py)
                    cx, cy = best_pos
            except Exception:
                pass

        # Track rotor patch in real frames
        real_disp, _ = track_patch_displacements(stream_disk_series(real_frames_subset), cx, cy, patch_size)
        real_proj, ref_axis, angle_deg = project_1d_motion(real_disp)
        real_fit = fit_harmonic_phasor(real_proj, t_arr, f0)

        # Track rotor in render frames if available
        render_fit = None
        if ren_frames_subset:
            ren_disp, _ = track_patch_displacements(stream_disk_series(ren_frames_subset), cx, cy, patch_size)
            ren_proj, _, _ = project_1d_motion(ren_disp, ref_axis=ref_axis)
            render_fit = fit_harmonic_phasor(ren_proj, t_arr, f0)

        rotor_comp = compare_phasors(
            real_fit=real_fit,
            render_fit=render_fit,
            patch_name="rotor",
            camera_id=cid,
            min_r2=min_r2,
            min_amp_px=min_amp_px,
        )

        # Image-level PSNR / SSIM / tstd
        psnr_db, ssim, tstd_real, tstd_ren = (None, None, None, None)
        tstd_ratio = None
        if ren_frames_subset:
            psnr_db, ssim, tstd_real, tstd_ren = compute_image_metrics(
                stream_disk_series(real_frames_subset),
                stream_disk_series(ren_frames_subset),
            )
            tstd_ratio = float(tstd_ren / (tstd_real + 1e-8)) if tstd_real > 0 else 1.0

        camera_evals[cid] = CameraEvaluationResult(
            camera_id=cid,
            status=rotor_comp.status,
            rotor_patch=rotor_comp,
            psnr_db=psnr_db,
            ssim=ssim,
            tstd_ratio_rotor=tstd_ratio,
            sample_count=n_frames,
            rejection_reasons=rotor_comp.rejection_reasons,
        )

    # 3D trajectory diagnostics if file provided
    traj_diag = None
    if trajectories_file and Path(trajectories_file).exists():
        data = np.load(str(trajectories_file))
        if "traj" in data:
            traj_3d = data["traj"]
            canon_xyz = data.get("canonical_xyz", None)
            times_arr = data.get("times", None)
            traj_diag = evaluate_3d_trajectories(
                traj_3d=traj_3d,
                f0=f0,
                fps=fps,
                times=times_arr,
                canonical_xyz=canon_xyz,
            )

    overall_verdict, verdict_explanation, multi_cam_summary = aggregate_multiview_verdict(
        camera_results=camera_evals,
        traj_diag=traj_diag,
    )

    # Serialize camera evals
    cams_dict = {}
    for cid, cr in camera_evals.items():
        c_dict: dict[str, Any] = {
            "camera_id": cid,
            "status": cr.status,
            "sample_count": cr.sample_count,
            "psnr_db": cr.psnr_db,
            "ssim": cr.ssim,
            "tstd_ratio_rotor": cr.tstd_ratio_rotor,
            "rejection_reasons": cr.rejection_reasons,
        }
        if cr.rotor_patch:
            rp = cr.rotor_patch
            c_dict["rotor_patch"] = {
                "status": rp.status,
                "real_amp_px": rp.real_fit.amp,
                "real_phase_deg": rp.real_fit.phase_deg,
                "real_r2": rp.real_fit.r2,
                "render_amp_px": rp.render_fit.amp if rp.render_fit else None,
                "render_phase_deg": rp.render_fit.phase_deg if rp.render_fit else None,
                "render_r2": rp.render_fit.r2 if rp.render_fit else None,
                "amp_ratio": rp.amp_ratio,
                "phase_err_deg": rp.phase_err_deg,
                "phase_time_err_ms": rp.phase_time_err_ms,
                "complex_vector_err": rp.complex_vector_err,
                "rejection_reasons": rp.rejection_reasons,
            }
        cams_dict[cid] = c_dict

    traj_dict = asdict(traj_diag) if traj_diag else None

    return MultiviewEvaluationPayload(
        eval_tool="scripts/eval_multiview_wobble.py",
        version="1.0.0",
        created_at=datetime.datetime.now().isoformat(),
        scene=str(scene_dir.name),
        render_source=str(render_dir),
        target_frequency_hz=float(f0),
        fps=float(fps),
        overall_verdict=overall_verdict,
        verdict_explanation=verdict_explanation,
        multi_camera_summary=multi_cam_summary,
        cameras=cams_dict,
        trajectory_3d=traj_dict,
        timing_provenance=time_prov,
    )


# ---------------------------------------------------------------------------
# 9. Formatters & Summary Table Printer
# ---------------------------------------------------------------------------

def print_human_summary(payload: MultiviewEvaluationPayload):
    print("\n" + "=" * 95)
    print(f"MULTI-VIEW MOTION EVALUATION: {payload.scene}")
    print(f"Target f0: {payload.target_frequency_hz:.4f} Hz | FPS: {payload.fps:.3f} | Verdict: {payload.overall_verdict}")
    print(f"Summary: {payload.verdict_explanation}")
    print("=" * 95)

    print(f"\n{'Camera':<8} | {'Real Amp':<10} | {'Ren Amp':<10} | {'Ratio':<7} | {'Phase Err':<10} | {'Vec Err':<8} | {'R2 Real':<7} | {'PSNR':<6} | {'Status':<15}")
    print("-" * 95)
    for cid, c_data in sorted(payload.cameras.items()):
        rp = c_data.get("rotor_patch", {})
        r_amp = f"{rp.get('real_amp_px', 0.0):.3f} px" if rp.get("real_amp_px") is not None else "N/A"
        g_amp = f"{rp.get('render_amp_px', 0.0):.3f} px" if rp.get("render_amp_px") is not None else "N/A"
        ratio = f"{rp.get('amp_ratio', 0.0):.2f}" if rp.get("amp_ratio") is not None else "N/A"
        p_err = f"{rp.get('phase_err_deg', 0.0):.1f}°" if rp.get("phase_err_deg") is not None else "N/A"
        v_err = f"{rp.get('complex_vector_err', 0.0):.3f}" if rp.get("complex_vector_err") is not None else "N/A"
        r2 = f"{rp.get('real_r2', 0.0):.2f}" if rp.get("real_r2") is not None else "N/A"
        psnr = f"{c_data.get('psnr_db', 0.0):.1f}" if c_data.get("psnr_db") is not None else "N/A"
        status = c_data.get("status", "UNKNOWN")
        print(f"{cid:<8} | {r_amp:<10} | {g_amp:<10} | {ratio:<7} | {p_err:<10} | {v_err:<8} | {r2:<7} | {psnr:<6} | {status:<15}")

    if payload.trajectory_3d:
        t3 = payload.trajectory_3d
        print("\n" + "-" * 95)
        print("3D TRAJECTORY FIELD DIAGNOSTICS (ROI-Free):")
        print(f"  Gaussians: {t3['n_gaussians']} | Frames: {t3['n_timesteps']}")
        print(f"  Amplitudes (px/units): p50={t3['p50_amp']:.5f}, p90={t3['p90_amp']:.5f}, p99={t3['p99_amp']:.5f}, max={t3['max_amp']:.5f}")
        print(f"  Spatial Concentration: Gini={t3['gini_coefficient']:.3f} | Top-10% Spread Ratio: {t3['top_decile_spread_ratio']:.3f}")
        print(f"  Phase Coherence: R={t3['phase_coherence_r']:.3f} | Background Drift Ratio: {t3['drift_ratio']:.3f}")
        print(f"  Global Drift Status: {'FAIL: Scene-wide unconstrained drift' if t3['global_drift_detected'] else 'PASS: Localized motion'}")
    print("=" * 95 + "\n")


# ---------------------------------------------------------------------------
# 10. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="W2 (T50): Multi-view frequency-domain & trajectory motion evaluation tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to real capture scene directory (e.g. data/multipleview/turbine_r1_missing_6)")
    parser.add_argument("--render-dir", type=str, required=True,
                        help="Path to rendered camera series or run root")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Optional path to trajectories.npz for 3D trajectory field diagnostics")
    parser.add_argument("--f0", type=float, default=DEFAULT_WOBBLE_HZ,
                        help="Target wobble fundamental frequency in Hz")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="Capture frame rate in fps")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Path to write machine-readable evaluation JSON")
    parser.add_argument("--human", action="store_true",
                        help="Print human-readable summary table to stdout")
    parser.add_argument("--min-r2", type=float, default=DEFAULT_MIN_R2,
                        help="Minimum R2 score to consider an optical track valid")
    parser.add_argument("--min-amp", type=float, default=DEFAULT_MIN_AMP_PX,
                        help="Minimum real amplitude in px to consider motion detectable")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    render_path = Path(args.render_dir)

    payload = evaluate_scene_multiview(
        scene_dir=scene_path,
        render_dir=render_path,
        f0=args.f0,
        fps=args.fps,
        trajectories_file=Path(args.trajectories) if args.trajectories else None,
        min_r2=args.min_r2,
        min_amp_px=args.min_amp,
    )

    if args.human or not args.out_json:
        print_human_summary(payload)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload_dict = asdict(payload)
        out_p.write_text(json.dumps(payload_dict, indent=2), encoding="utf-8")
        print(f"[W2 EVAL] Wrote evaluation report to {out_p}")

    # Non-zero exit code if evaluation failed outright
    if payload.overall_verdict.startswith("FAIL"):
        sys.exit(1)


if __name__ == "__main__":
    main()

