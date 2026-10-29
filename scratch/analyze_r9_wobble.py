"""Analyze r9 turbine run trajectories vs baseline: FFT at 7.11 Hz, amplitude,
spatial distribution of the wobble (whole scene vs turbine ROI)."""
import numpy as np
import json

FPS = 20.003929910241613
F_WOBBLE = 7.112
OUT = {}

def analyze(path, label):
    d = np.load(path)
    traj = d["traj"].astype(np.float64)          # (N, T, 3)
    xyz = d["canonical_xyz"].astype(np.float64)  # (N, 3)
    op = d["opacity"].astype(np.float64)
    times = d["times"].astype(np.float64)
    T = traj.shape[1]

    keep = op > 0.1
    traj_k, xyz_k = traj[keep], xyz[keep]
    N = len(xyz_k)

    disp = traj_k - xyz_k[:, None, :]            # (N, T, 3)
    max_disp = np.linalg.norm(disp, axis=-1).max(axis=-1)

    # FFT of displacement along time, per Gaussian, per axis
    disp_c = disp - disp.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(disp_c, axis=1)            # (N, F, 3)
    freqs = np.fft.rfftfreq(T, d=(times[1] - times[0]) if len(times) > 1 else 1.0)
    # times are normalized 0..1; convert to Hz using capture duration
    duration = 164 / FPS  # frame_count/fps ≈ 8.148 s; times span 0..1
    freqs_hz = freqs / (times[-1] if times[-1] > 0 else 1.0) / duration * duration  # placeholder
    # simpler: times normalized over capture → freq[Hz] = rfftfreq index * (1/(T*dt_norm)) / duration
    freqs_hz = np.fft.rfftfreq(T, d=(times[1]-times[0]) * duration)

    amp = np.abs(fft) * 2.0 / T                  # amplitude spectrum (N, F, 3)
    amp_total = np.linalg.norm(amp, axis=-1)     # (N, F)

    i_wobble = int(np.argmin(np.abs(freqs_hz - F_WOBBLE)))
    peak_freq_per_pt = freqs_hz[np.argmax(amp_total[:, 1:], axis=1) + 1]

    res = {
        "label": label,
        "N_total": int(len(xyz)), "N_kept": int(N),
        "max_disp_mean": float(max_disp.mean()),
        "max_disp_median": float(np.median(max_disp)),
        "max_disp_p95": float(np.percentile(max_disp, 95)),
        "freqs_hz_0_to_nyq": [round(float(f), 2) for f in freqs_hz[:20]],
        "wobble_bin_hz": float(freqs_hz[i_wobble]),
    }

    # population-level spectrum: mean amplitude across Gaussians per frequency
    pop_amp = amp_total.mean(axis=0)
    res["pop_spectrum_top5"] = sorted(
        [(round(float(freqs_hz[i]), 2), round(float(pop_amp[i]), 5)) for i in range(1, len(freqs_hz))],
        key=lambda x: -x[1])[:5]
    res["pop_amp_at_7.11"] = float(pop_amp[i_wobble])

    # fraction of points whose dominant freq is the wobble bin (±1 bin)
    dom = np.argmax(amp_total[:, 1:], axis=1) + 1
    near_wobble = np.abs(dom - i_wobble) <= 1
    res["frac_pts_dominant_at_wobble"] = float(near_wobble.mean())

    # amplitude at wobble per point
    amp_w = amp_total[:, i_wobble]
    res["amp_at_wobble_median"] = float(np.median(amp_w))
    res["amp_at_wobble_p90"] = float(np.percentile(amp_w, 90))
    res["amp_at_wobble_max"] = float(amp_w.max())

    # ---- spatial distribution of wobble amplitude ----
    # scene bbox
    lo, hi = xyz_k.min(axis=0), xyz_k.max(axis=0)
    res["scene_bbox_lo"] = [round(float(v), 3) for v in lo]
    res["scene_bbox_hi"] = [round(float(v), 3) for v in hi]

    # deciles of amp_w: where are high-amplitude points?
    thr90 = np.percentile(amp_w, 90)
    high = amp_w >= thr90
    res["high_amp_spatial"] = {
        "count": int(high.sum()),
        "bbox_lo": [round(float(v), 3) for v in xyz_k[high].min(axis=0)] if high.any() else None,
        "bbox_hi": [round(float(v), 3) for v in xyz_k[high].max(axis=0)] if high.any() else None,
    }
    # correlation of wobble amplitude with distance from scene center
    center = xyz_k.mean(axis=0)
    r = np.linalg.norm(xyz_k - center, axis=1)
    res["corr_amp_vs_radius"] = float(np.corrcoef(r, amp_w)[0, 1])

    # phase coherence: phase of dominant wobble component (projected on dominant axis)
    # use axis of max variance of the mean trajectory
    mean_traj = disp_c.mean(axis=0)              # (T, 3)
    _, _, vt = np.linalg.svd(mean_traj - mean_traj.mean(axis=0), full_matrices=False)
    axis = vt[0]
    proj = disp_c @ axis                          # (N, T)
    fft_p = np.fft.rfft(proj, axis=1)
    phase = np.angle(fft_p[:, i_wobble])
    strong = amp_w > np.percentile(amp_w, 50)
    ph = phase[strong]
    # circular std
    R = np.abs(np.exp(1j * ph).mean())
    res["wobble_axis"] = [round(float(v), 3) for v in axis]
    res["phase_coherence_R_strong_pts"] = float(R)

    return res, xyz_k, amp_w, amp_total, freqs_hz, i_wobble, keep, xyz


r9, xyz_k9, amp_w9, amp_t9, freqs9, iw9, keep9, xyz9 = analyze(
    "runs/lego_turbine_2_missing_r9-415299ad/trajectories.npz", "r9 (fixed stack)")
base, xyz_kb, amp_wb, amp_tb, freqsb, iwb, keepb, xyzb = analyze(
    "runs/lego_turbine_2_missing-b9c0d39f/trajectories.npz", "baseline (collapsed)")

OUT = {"r9": r9, "baseline": base}

# Save amplitude maps for later spatial plotting
np.savez("scratch/r9_amp_maps.npz",
         xyz_r9=xyz_k9, amp_w_r9=amp_w9, amp_total_r9=amp_t9,
         xyz_base=xyz_kb, amp_w_base=amp_wb, amp_total_base=amp_tb,
         freqs=freqs9, i_wobble=iw9)

print(json.dumps(OUT, indent=2))
