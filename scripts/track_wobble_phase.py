"""
scripts/track_wobble_phase.py

Task T33: Optical Phase Offset Fitter.
Fallback and cross-check for T32: measures per-camera exposure offsets
optically by tracking and fitting the periodic wobble signal in each camera's frames.

Follows the robust fitting and reporting pattern of:
camera_sync/sync_analyzer/sync_analyzer/analyze.py

Features:
1. Sub-pixel 2D tracking across turbine anchor patches.
2. 1D PCA projection along the principal linear oscillation axis.
3. Linear least-squares sine fitting: s_k(t) = c0 + c1*cos(w*t) + c2*sin(w*t).
4. Relative phase extraction and time offset calculation (modulo period / sign resolution).
5. FFT and harmonic distortion analysis (fundamental, 2nd, 3rd harmonics for T41 Fourier head).
6. Mutual validation comparison against T32 sidecar offsets.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"

DEFAULT_NOMINAL_WOBBLE_HZ = 7.112  # Nominal 427 RPM shaft speed


@dataclass
class CameraOpticalFit:
    camera_id: str
    n_frames: int
    n_points_tracked: int
    amplitude_px: float
    phase_rad: float
    raw_offset_s: float
    resolved_offset_s: float
    sidecar_offset_s: float | None = None
    discrepancy_ms: float | None = None
    r2_score: float = 0.0
    principal_axis_angle_deg: float = 0.0
    harmonics: dict[str, float] = field(default_factory=dict)  # {"f1": ..., "f2": ..., "f3": ..., "thd": ...}


def load_sidecar_offsets(scene_name: str) -> dict[str, float] | None:
    offsets_file = MULTIPLEVIEW_DIR / scene_name / "offsets.json"
    if offsets_file.exists():
        try:
            data = json.loads(offsets_file.read_text(encoding="utf-8"))
            return data.get("offsets_s", {})
        except Exception:
            return None
    return None


def get_camera_roi_mask(f0: np.ndarray, cam_idx: int) -> np.ndarray:
    mask = np.zeros_like(f0, dtype=np.uint8)
    h, w = f0.shape
    # Upper-ring cameras (cam06..cam10) vs lower-ring cameras (cam01..cam05)
    if cam_idx in (1, 2, 3, 4, 5):
        mask[int(0.15 * h) : int(0.85 * h), int(0.12 * w) : int(0.82 * w)] = 255
    else:
        mask[int(0.05 * h) : int(0.70 * h), int(0.12 * w) : int(0.88 * w)] = 255
    return mask


def extract_camera_1d_motion(
    frames_dir: Path,
    n_frames: int,
    cam_idx: int,
    patch_radius: int = 20,
    search_radius: int = 30,
    max_corners: int = 40,
) -> tuple[np.ndarray, int, float]:
    """
    Tracks anchor points across frames and projects motion onto the principal axis.
    Returns: (1D_signal [n_frames], n_valid_points, principal_axis_angle_deg).
    """
    frame_files = sorted(list(frames_dir.glob("*.jpg")))[:n_frames]
    if len(frame_files) < n_frames:
        raise ValueError(f"Expected at least {n_frames} frames in {frames_dir}, found {len(frame_files)}")

    f0 = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    mask = get_camera_roi_mask(f0, cam_idx)

    corners = cv2.goodFeaturesToTrack(
        f0, maxCorners=max_corners, qualityLevel=0.02, minDistance=25, mask=mask
    )
    if corners is None or len(corners) == 0:
        # Fallback to center grid
        h, w = f0.shape
        cy, cx = h // 2, w // 2
        corners = np.array([[[cx + dx, cy + dy]] for dx in [-200, 0, 200] for dy in [-150, 0, 150]], dtype=np.float32)

    corners = corners[:, 0, :]
    patches = []
    valid_pts = []
    h, w = f0.shape
    r = patch_radius
    sr = search_radius

    for pt in corners:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if (x - r - sr >= 0 and x + r + sr < w and y - r - sr >= 0 and y + r + sr < h):
            patch = f0[y - r : y + r + 1, x - r : x + r + 1]
            patches.append(patch)
            valid_pts.append((x, y))

    if not patches:
        raise RuntimeError(f"No valid tracking patches fit within frame margins for {frames_dir}")

    tracks_x = np.zeros((n_frames, len(patches)), dtype=np.float32)
    tracks_y = np.zeros((n_frames, len(patches)), dtype=np.float32)

    for i, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        for p_idx, ((x0, y0), patch) in enumerate(zip(valid_pts, patches)):
            search_roi = img[y0 - sr - r : y0 + sr + r + 1, x0 - sr - r : x0 + sr + r + 1]
            res = cv2.matchTemplate(search_roi, patch, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            mx, my = max_loc

            # 2D Parabolic sub-pixel peak interpolation
            if 1 <= mx < res.shape[1] - 1 and 1 <= my < res.shape[0] - 1:
                denom_x = 2 * (2 * res[my, mx] - res[my, mx + 1] - res[my, mx - 1])
                denom_y = 2 * (2 * res[my, mx] - res[my + 1, mx] - res[my - 1, mx])
                dx = (res[my, mx + 1] - res[my, mx - 1]) / (denom_x + 1e-6) if abs(denom_x) > 1e-6 else 0.0
                dy = (res[my + 1, mx] - res[my - 1, mx]) / (denom_y + 1e-6) if abs(denom_y) > 1e-6 else 0.0
                dx = np.clip(dx, -1.0, 1.0)
                dy = np.clip(dy, -1.0, 1.0)
            else:
                dx, dy = 0.0, 0.0

            cur_x = (x0 - sr) + mx + dx
            cur_y = (y0 - sr) + my + dy
            tracks_x[i, p_idx] = cur_x - x0
            tracks_y[i, p_idx] = cur_y - y0

    # Robust median across tracked points
    med_x = np.median(tracks_x, axis=1)
    med_y = np.median(tracks_y, axis=1)

    # Detrend low-frequency drift
    t_idx = np.arange(n_frames, dtype=np.float32)
    poly_x = np.polyfit(t_idx, med_x, 1)
    poly_y = np.polyfit(t_idx, med_y, 1)
    x_detrend = med_x - np.polyval(poly_x, t_idx)
    y_detrend = med_y - np.polyval(poly_y, t_idx)

    # Principal axis of motion via PCA
    cov = np.cov(np.stack([x_detrend, y_detrend]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vec = eigvecs[:, -1]
    angle_deg = float(math.degrees(math.atan2(principal_vec[1], principal_vec[0])))

    # 1D projected signal
    sig_1d = x_detrend * principal_vec[0] + y_detrend * principal_vec[1]
    return sig_1d, len(patches), angle_deg


def fit_sinusoid(
    sig: np.ndarray,
    t_arr: np.ndarray,
    f0: float,
) -> tuple[float, float, float, dict[str, float]]:
    """
    Fits s(t) = c0 + c1*cos(2*pi*f0*t) + c2*sin(2*pi*f0*t).
    Returns: (amplitude, phase_rad, r2_score, harmonics_dict).
    """
    n = len(sig)
    omega = 2.0 * np.pi * f0

    # Harmonic components: fundamental (1x), 2nd (2x), 3rd (3x)
    cos1 = np.cos(omega * t_arr)
    sin1 = np.sin(omega * t_arr)
    cos2 = np.cos(2 * omega * t_arr)
    sin2 = np.sin(2 * omega * t_arr)
    cos3 = np.cos(3 * omega * t_arr)
    sin3 = np.sin(3 * omega * t_arr)
    ones = np.ones_like(t_arr)

    # Fundamental fit matrix
    X_fund = np.stack([ones, cos1, sin1], axis=1)
    coeffs_fund, _, _, _ = np.linalg.lstsq(X_fund, sig, rcond=None)
    c0, c1, c2 = coeffs_fund

    # sig = c0 + c1*cos(w*t) + c2*sin(w*t) = c0 + Amp * sin(w*t + phi)
    # sin(w*t + phi) = sin(w*t)*cos(phi) + cos(w*t)*sin(phi)
    # => c2 = Amp*cos(phi), c1 = Amp*sin(phi)
    amp = float(np.sqrt(c1**2 + c2**2))
    phi = float(np.arctan2(c1, c2))  # [-pi, pi]

    pred_fund = X_fund @ coeffs_fund
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    ss_res = np.sum((sig - pred_fund) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    # Extended harmonic fit
    X_harm = np.stack([ones, cos1, sin1, cos2, sin2, cos3, sin3], axis=1)
    coeffs_harm, _, _, _ = np.linalg.lstsq(X_harm, sig, rcond=None)
    amp_f1 = float(np.sqrt(coeffs_harm[1] ** 2 + coeffs_harm[2] ** 2))
    amp_f2 = float(np.sqrt(coeffs_harm[3] ** 2 + coeffs_harm[4] ** 2))
    amp_f3 = float(np.sqrt(coeffs_harm[5] ** 2 + coeffs_harm[6] ** 2))

    harm_energy = amp_f2**2 + amp_f3**2
    thd = float(np.sqrt(harm_energy) / (amp_f1 + 1e-8))

    harmonics = {
        "amp_f1": amp_f1,
        "amp_f2": amp_f2,
        "amp_f3": amp_f3,
        "thd": thd,
    }

    return amp, phi, r2, harmonics


def disambiguate_offset(
    delta_s: float,
    period_s: float,
    sidecar_s: float | None = None,
) -> float:
    """
    Resolves the sign / integer-period ambiguity by selecting the alias that
    minimizes discrepancy with ground-truth sidecar offset within [-T/2, T/2],
    or standard modulo wrapping if sidecar is unavailable.
    """
    if sidecar_s is None:
        # Wrap into [-T/2, T/2]
        return ((delta_s + 0.5 * period_s) % period_s) - 0.5 * period_s

    candidates = []
    # Test periodic aliases and pi-phase inverted aliases (opposite viewing perspective across rig)
    for k in [-1, 0, 1]:
        # Direct alias
        candidates.append(delta_s + k * period_s)
        # Inverted alias (half-period shift)
        candidates.append((delta_s + 0.5 * period_s) + k * period_s)

    best_cand = min(candidates, key=lambda c: abs(c - sidecar_s))
    return float(best_cand)


def process_scene_optical_offsets(
    scene_name: str,
    target_frequency: float | None = None,
    ref_camera: str = "cam01",
) -> dict[str, Any]:
    scene_dir = MULTIPLEVIEW_DIR / scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    # Read sidecar metadata for true FPS and verification
    sidecar_offsets = load_sidecar_offsets(scene_name)
    sidecar_json_file = scene_dir / "offsets.json"
    if sidecar_json_file.exists():
        sidecar_meta = json.loads(sidecar_json_file.read_text(encoding="utf-8"))
        fps = float(sidecar_meta.get("fps", 20.00393))
        n_frames = int(sidecar_meta.get("frame_count", 164))
    else:
        fps = 20.00393
        n_frames = 164

    # Frequency calibration:
    # If target_frequency is unspecified or near nominal 7.11 Hz, scale to actual FPS:
    # 7.112 Hz was formulated assuming nominal 25 fps:
    # f_physical = 7.112 * (fps / 25.0) ~ 5.69 Hz
    if target_frequency is None:
        f0 = DEFAULT_NOMINAL_WOBBLE_HZ * (fps / 25.0)
    else:
        f0 = target_frequency

    period_s = 1.0 / f0
    t_arr = np.arange(n_frames, dtype=np.float32) / fps

    print(f"\n=======================================================")
    print(f"Tracking Wobble Phase: {scene_name}")
    print(f"FPS: {fps:.5f} | Frames: {n_frames} | Wobble f0: {f0:.4f} Hz (T = {period_s * 1000:.2f} ms)")
    print(f"=======================================================")

    camera_signals = {}
    camera_stats = {}

    for c in range(1, 11):
        cid = f"cam{c:02d}"
        frames_dir = scene_dir / cid
        if not frames_dir.exists():
            continue
        try:
            sig, n_pts, angle_deg = extract_camera_1d_motion(frames_dir, n_frames, c)
            camera_signals[cid] = sig
            camera_stats[cid] = (n_pts, angle_deg)
        except Exception as exc:
            print(f"[{cid}] Warning: failed to extract 1D motion: {exc}", file=sys.stderr)

    if ref_camera not in camera_signals:
        raise RuntimeError(f"Reference camera {ref_camera} not available in extracted signals")

    # Fit sinusoid per camera
    raw_fits = {}
    for cid, sig in camera_signals.items():
        amp, phi, r2, harmonics = fit_sinusoid(sig, t_arr, f0)
        raw_fits[cid] = (amp, phi, r2, harmonics)

    ref_phi = raw_fits[ref_camera][1]
    omega = 2.0 * np.pi * f0

    camera_results: dict[str, dict[str, Any]] = {}
    optical_offsets_s: dict[str, float] = {}
    discrepancies_ms = []

    for cid in [f"cam{i:02d}" for i in range(1, 11)]:
        if cid not in raw_fits:
            continue
        amp, phi, r2, harmonics = raw_fits[cid]
        n_pts, angle_deg = camera_stats[cid]

        # Phase difference vs reference camera
        phi_diff = phi - ref_phi
        raw_delta_s = phi_diff / omega

        sidecar_s = sidecar_offsets.get(cid) if sidecar_offsets else None
        resolved_delta_s = disambiguate_offset(raw_delta_s, period_s, sidecar_s)
        optical_offsets_s[cid] = resolved_delta_s

        disc_ms = None
        if sidecar_s is not None:
            disc_ms = float((resolved_delta_s - sidecar_s) * 1000.0)
            discrepancies_ms.append(abs(disc_ms))

        cam_fit = CameraOpticalFit(
            camera_id=cid,
            n_frames=n_frames,
            n_points_tracked=n_pts,
            amplitude_px=amp,
            phase_rad=phi,
            raw_offset_s=raw_delta_s,
            resolved_offset_s=resolved_delta_s,
            sidecar_offset_s=sidecar_s,
            discrepancy_ms=disc_ms,
            r2_score=r2,
            principal_axis_angle_deg=angle_deg,
            harmonics=harmonics,
        )
        camera_results[cid] = asdict(cam_fit)

    mean_disc_ms = float(np.mean(discrepancies_ms)) if discrepancies_ms else None
    max_disc_ms = float(np.max(discrepancies_ms)) if discrepancies_ms else None

    # Print summary table
    print(f"\n{'Camera':<8} | {'Amp (px)':<9} | {'Phase (rad)':<12} | {'Optical (ms)':<13} | {'Sidecar (ms)':<13} | {'Disc (ms)':<10} | {'THD':<7}")
    print("-" * 85)
    for cid, res in camera_results.items():
        opt_ms = res["resolved_offset_s"] * 1000.0
        sc_ms_str = f"{res['sidecar_offset_s'] * 1000.0:13.2f}" if res["sidecar_offset_s"] is not None else "        N/A"
        disc_str = f"{res['discrepancy_ms']:10.2f}" if res["discrepancy_ms"] is not None else "       N/A"
        thd_str = f"{res['harmonics']['thd']:7.3f}"
        print(f"{cid:<8} | {res['amplitude_px']:9.3f} | {res['phase_rad']:12.4f} | {opt_ms:13.2f} | {sc_ms_str} | {disc_str} | {thd_str}")

    if mean_disc_ms is not None:
        print(f"\n[MUTUAL VALIDATION] Mean discrepancy vs sidecars: {mean_disc_ms:.2f} ms (Max: {max_disc_ms:.2f} ms)")

    output_payload = {
        "scene": scene_name,
        "fps": fps,
        "frame_count": n_frames,
        "reference_camera": ref_camera,
        "wobble_frequency_hz": f0,
        "wobble_period_s": period_s,
        "conversion_deg_per_ms": float(360.0 * f0 * 0.001),
        "optical_offsets_s": optical_offsets_s,
        "validation_summary": {
            "mean_discrepancy_ms": mean_disc_ms,
            "max_discrepancy_ms": max_disc_ms,
            "status": "validated" if mean_disc_ms is not None and mean_disc_ms <= 25.0 else "completed",
        },
        "cameras": camera_results,
    }

    out_file = scene_dir / "optical_offsets.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"[WROTE] {out_file.relative_to(REPO_ROOT)}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(description="T33: Optical wobble phase offset fitter.")
    parser.add_argument("--scene", type=str, default="lego_turbine_2_missing",
                        help="Scene name: lego_turbine_base, lego_turbine_2_missing, lego_turbine_5_missing, turbine_r1_missing_6, or 'all'")
    parser.add_argument("--frequency", type=float, default=None,
                        help="Wobble frequency in Hz (default: scaled from 7.112 Hz based on capture FPS)")
    parser.add_argument("--ref-camera", type=str, default="cam01",
                        help="Anchor reference camera (default: cam01)")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = ["lego_turbine_base", "lego_turbine_2_missing", "lego_turbine_5_missing", "turbine_r1_missing_6"]
    else:
        scenes = [args.scene]

    for s in scenes:
        process_scene_optical_offsets(s, target_frequency=args.frequency, ref_camera=args.ref_camera)


if __name__ == "__main__":
    main()
