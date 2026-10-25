import cv2
import numpy as np
from pathlib import Path

scene = "lego_turbine_2_missing"
cam = "cam01"
frames_dir = Path(f"data/multipleview/{scene}/{cam}")
frame_files = sorted(list(frames_dir.glob("*.jpg")))
n_frames = len(frame_files)
fps = 20.00393

first_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)

# We want points on the nacelle / tower
mask = np.zeros_like(first_img, dtype=np.uint8)
mask[350:900, 1400:2300] = 255

pts0 = cv2.goodFeaturesToTrack(first_img, maxCorners=100, qualityLevel=0.02, minDistance=20, mask=mask)
curr_pts = pts0
prev_img = first_img

trajectories = [curr_pts.copy()]
lk_params = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

valid_mask = np.ones(len(pts0), dtype=bool)

for i in range(1, n_frames):
    img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, curr_pts, None, **lk_params)
    back_pts, status_b, err_b = cv2.calcOpticalFlowPyrLK(img, prev_img, next_pts, None, **lk_params)
    d = abs(curr_pts - back_pts).reshape(-1, 2).max(-1)
    good = (status.flatten() == 1) & (status_b.flatten() == 1) & (d < 1.0)
    valid_mask = valid_mask & good
    trajectories.append(next_pts.copy())
    curr_pts = next_pts
    prev_img = img

valid_indices = np.where(valid_mask)[0]
print(f"Robustly tracked: {len(valid_indices)} points")

traj_arr = np.array(trajectories)[:, valid_indices, 0, :] # [T, N, 2]

# High-pass filter or detrend to remove low-frequency drift
# Subtract degree-2 polynomial trend per point
t_arr = np.arange(n_frames) / fps
detrended = np.zeros_like(traj_arr)
for p in range(traj_arr.shape[1]):
    for dim in range(2):
        poly = np.polyfit(t_arr, traj_arr[:, p, dim], deg=2)
        trend = np.polyval(poly, t_arr)
        detrended[:, p, dim] = traj_arr[:, p, dim] - trend

rfftfreq = np.fft.rfftfreq(n_frames, d=1.0/fps)
fft_x = np.abs(np.fft.rfft(detrended[..., 0], axis=0)) # [F, N]
fft_y = np.abs(np.fft.rfft(detrended[..., 1], axis=0)) # [F, N]
fft_mag = np.sqrt(fft_x**2 + fft_y**2) # [F, N]
mean_fft = np.mean(fft_mag, axis=1) # [F]

# Normalized amplitude in pixels
amp_px = (mean_fft / n_frames) * 2.0

# Print spectrum between 5 Hz and 9 Hz
print(f"{'Freq (Hz)':<10} | {'Amp (px)':<10}")
print("-" * 25)
for f_idx, freq in enumerate(rfftfreq):
    if 5.5 <= freq <= 8.5:
        print(f"{freq:<10.3f} | {amp_px[f_idx]:<10.4f}")

# Find peak in 6 - 8 Hz band
band_idx = [i for i, f in enumerate(rfftfreq) if 6.0 <= f <= 8.0]
peak_band_idx = band_idx[np.argmax(amp_px[band_idx])]
print(f"\nDominant peak in [6.0, 8.0] Hz: {rfftfreq[peak_band_idx]:.3f} Hz with amplitude {amp_px[peak_band_idx]:.4f} px (raw peak: {mean_fft[peak_band_idx]:.3f})")
