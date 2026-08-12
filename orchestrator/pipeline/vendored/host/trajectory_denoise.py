"""Trajectory denoising for T18 (``segment.rigid2``, proposal 06 in ``docs/proposals/``).

Why this exists: on the real pump01/grid models the trained deformation field's frame-to-frame
position *jitter* is comparable to the true mm-scale motion (``docs/motion-segmentation.md``:
median edge rigidity score 0.00125 vs p90 0.0035), so any per-edge rigidity threshold fails.
But the jitter is approximately *white* in time while the scene's true motion is *narrowband* —
the pump scenes are driven by a known periodic motion (``scene-gen/gen_scenes.py``), the same
property ``core/render_amp.py``'s FFT-based amplification already exploits. Band-passing each
trajectory at the drive frequency and its harmonics removes most of the noise power while
leaving the periodic signal untouched (~93% of noise power gone at T=60, H=3 — proposal 06 §1).

The frequency-bin conventions (rfft over the sampled deformation trajectory, keep DC + the bins
at integer multiples of the drive frequency) are ported from the same FFT usage in
``core/render_amp.py`` (reference only, per the copy-in rule — nothing here imports ``core/``).

Pure numpy — no torch, no GPU, safe to import at module scope in the orchestrator.
"""

from __future__ import annotations

import numpy as np


def trajectory_energy(traj: np.ndarray) -> np.ndarray:
    """Total temporal variance per point, E_i = tr(Cov[p_i(t)]). traj: (N,T,3) -> (N,).

    Used both for drive-frequency auto-detection (the moving points dominate the spectrum)
    and for static/moving gating upstream of noise calibration.
    """
    return float(1.0) * traj.var(axis=1).sum(axis=-1)


def mean_power_spectrum(traj: np.ndarray, energy: np.ndarray | None = None,
                        top_frac: float = 0.1) -> np.ndarray:
    """Mean rfft power spectrum of the centered trajectories, averaged over the moving points
    (top ``top_frac`` by energy — static points contribute only their noise floor). Returns
    power per rfft bin, shape (T//2+1,). Bin 0 is DC."""
    e = trajectory_energy(traj) if energy is None else energy
    n_top = max(1, int(len(traj) * top_frac))
    top = np.argsort(e)[-n_top:]
    centered = traj[top] - traj[top].mean(axis=1, keepdims=True)
    spec = np.fft.rfft(centered, axis=1)  # (n_top, T//2+1, 3)
    power = (np.abs(spec) ** 2).sum(axis=-1).mean(axis=0)
    return power


def detect_drive_freq(traj: np.ndarray, energy: np.ndarray | None = None) -> int:
    """Auto-detect the drive frequency as the argmax (non-DC bin) of the moving points' mean
    power spectrum. Returns the rfft bin index (cycles per clip window), >= 1."""
    power = mean_power_spectrum(traj, energy=energy)
    if len(power) <= 1:
        return 1
    return int(np.argmax(power[1:]) + 1)


def bandpass(
    traj: np.ndarray,
    drive_freq: float | None = None,
    harmonics: int = 3,
    half_width: int = 0,
) -> tuple[np.ndarray, int]:
    """FFT band-pass: keep DC + the bins at h*drive_freq (+/- half_width) for h = 1..harmonics,
    inverse-transform. Returns (traj_denoised (N,T,3), drive_freq_bin_used).

    ``drive_freq`` is in cycles per clip window (rfft bin units); ``None`` => auto-detect via
    :func:`detect_drive_freq`. ``half_width > 0`` widens each kept bin to a small band, for
    motion that isn't exactly periodic in the sampled window.
    """
    n, t, _ = traj.shape
    f0 = detect_drive_freq(traj) if drive_freq is None else int(round(drive_freq))
    f0 = max(1, min(f0, t // 2))

    spec = np.fft.rfft(traj, axis=1)  # (N, T//2+1, 3)
    keep = np.zeros(spec.shape[1], dtype=bool)
    keep[0] = True  # DC = mean position
    for h in range(1, harmonics + 1):
        b = h * f0
        lo, hi = max(1, b - half_width), min(spec.shape[1] - 1, b + half_width)
        keep[lo:hi + 1] = True
    spec_filtered = spec * keep[None, :, None]
    denoised = np.fft.irfft(spec_filtered, n=t, axis=1)
    return denoised, f0


def motion_fingerprint(traj: np.ndarray, drive_freq: int, harmonics: int = 3) -> np.ndarray:
    """Complex FFT coefficients at the kept drive-harmonic bins, flattened per point:
    (N, 3*harmonics) complex. Points on one rigid part share the part's {R(t), tau(t)}, so
    their fingerprints are linearly related — a cheap amplitude/phase clustering feature
    (proposal 06 §5; used as EM initialization by the planned ``segment.kabsch`` impl, T20)."""
    spec = np.fft.rfft(traj, axis=1)  # (N, T//2+1, 3)
    bins = [min(h * drive_freq, spec.shape[1] - 1) for h in range(1, harmonics + 1)]
    fp = spec[:, bins, :]  # (N, H, 3)
    return fp.reshape(len(traj), -1)
