"""T38/r11 preflight gate evaluation: does the trained r11 stack recover the
injected 5.691 Hz rotor motion without whole-scene wobble?

Inputs:
  runs/<run_id>/trajectories.npz   (from `amp seg extract <run_id>`)
  data/multipleview/synthetic_turbine_preflight_r11/roi_voxels.npz
  data/multipleview/synthetic_turbine_preflight_r11/gt_motion_info.json

Gates (report §6 step 3):
  G1  ROI Gaussians' dominant spectral peak at 5.691 Hz (+/- 0.25 Hz bin tolerance)
  G2  background (non-ROI) motion ≈ 0: p95 background max displacement < 20% of
      ROI median amplitude  (r9 failure mode was whole-scene wobble)
  G3  ROI median oscillation amplitude within ±30% of injected 3.6 mm
"""
import json
import sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "synthetic_turbine_preflight_r11-b8a7e7b9"
SCENE = "data/multipleview/synthetic_turbine_preflight_r11"
F_WOBBLE = 5.691

gt = json.loads(open(f"{SCENE}/gt_motion_info.json").read())
FPS = float(gt["fps"]); N_FRAMES = int(gt["n_frames"])
AMP_GT_M = float(gt["amplitude_3d_m"])
DURATION = N_FRAMES / FPS

d = np.load(f"runs/{RUN}/trajectories.npz")
traj = d["traj"].astype(np.float64)           # (N, T, 3)
xyz = d["canonical_xyz"].astype(np.float64)   # (N, 3)
op = d["opacity"].astype(np.float64)
times = d["times"].astype(np.float64)
T = traj.shape[1]
keep = op > 0.1
traj, xyz = traj[keep], xyz[keep]
print(f"points kept: {len(xyz)} / {len(op)}  (T={T})")

roi = np.load(f"{SCENE}/roi_voxels.npz")
grid, origin, vs = roi["grid"], roi["origin"], float(roi["voxel_size"])
idx = np.floor(xyz / vs).astype(np.int64) - np.floor(origin / vs).astype(np.int64)
inb = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=1)
in_roi = np.zeros(len(xyz), dtype=bool)
in_roi[inb] = grid[idx[inb, 0], idx[inb, 1], idx[inb, 2]]
print(f"ROI: {in_roi.sum()} pts ({in_roi.mean()*100:.1f}%), background: {(~in_roi).sum()}")

disp = traj - xyz[:, None, :]
max_disp = np.linalg.norm(disp, axis=-1).max(axis=-1)

disp_c = disp - disp.mean(axis=1, keepdims=True)
fft = np.fft.rfft(disp_c, axis=1)
freqs = np.fft.rfftfreq(T, d=(times[1] - times[0]) * DURATION)
amp = np.abs(fft) * 2.0 / T
amp_tot = np.linalg.norm(amp, axis=-1)        # (N, F) single-sided amplitude, scene units

peak_f = freqs[np.argmax(amp_tot[:, 1:], axis=1) + 1]
i_w = int(np.argmin(np.abs(freqs - F_WOBBLE)))

# --- G1: ROI dominant peak frequency
roi_peak = peak_f[in_roi]
g1_hits = np.abs(roi_peak - F_WOBBLE) <= 0.25
g1_frac = g1_hits.mean()
print(f"\nG1  ROI dominant-peak at {F_WOBBLE} Hz: {g1_frac*100:.1f}% of ROI pts "
      f"(median peak {np.median(roi_peak):.2f} Hz)  -> {'PASS' if g1_frac > 0.8 else 'FAIL'}")

# --- G2: background quiescence
roi_med_amp = np.median(amp_tot[in_roi, i_w])
bg_p95 = np.percentile(max_disp[~in_roi], 95)
bg_mean_w = amp_tot[~in_roi, i_w].mean()
print(f"G2  background p95 max-disp: {bg_p95*1000:.2f} mm; bg mean amp@{F_WOBBLE} Hz: "
      f"{bg_mean_w*1000:.3f} mm; ROI median amp: {roi_med_amp*1000:.2f} mm  -> "
      f"{'PASS' if bg_p95 < 0.2*roi_med_amp*2 and bg_mean_w < 0.1*roi_med_amp else 'FAIL'}")

# --- G3: ROI amplitude fidelity (3D oscillation amplitude vs injected 3.6 mm)
g3 = abs(roi_med_amp - AMP_GT_M) / AMP_GT_M
print(f"G3  ROI median amplitude {roi_med_amp*1000:.2f} mm vs injected "
      f"{AMP_GT_M*1000:.2f} mm (err {g3*100:.1f}%)  -> {'PASS' if g3 <= 0.30 else 'FAIL'}")

# global mean displacement spectrum (r9's failure signature)
mean_disp = disp.mean(axis=0) - disp.mean(axis=(0, 1))
mfft = np.abs(np.fft.rfft(mean_disp, axis=0)) * 2.0 / T
m_amp = np.linalg.norm(mfft, axis=-1)
print(f"\n    global-mean amp@{freqs[i_w]:.2f} Hz: {m_amp[i_w]*1000:.3f} mm "
      f"(r9 run: whole scene wobbled; want ≈ 0)")
top = sorted([(round(float(freqs[i]), 2), round(float(m_amp[i]) * 1000, 3))
              for i in range(1, len(freqs))], key=lambda x: -x[1])[:5]
print(f"    global-mean spectrum top5 (Hz, mm): {top}")
