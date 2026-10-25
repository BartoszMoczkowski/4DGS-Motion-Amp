import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import cv2

scene = "lego_turbine_2_missing"
cam = "cam01"
frames_dir = Path(f"data/multipleview/{scene}/{cam}")
frame_files = sorted(list(frames_dir.glob("*.jpg")))
n_frames = len(frame_files)
fps = 20.00393

# Load first frame
f0 = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)

# Let's find 50 good corners on the turbine nacelle/mast
mask = np.zeros_like(f0, dtype=np.uint8)
# Turbine body region
mask[400:1300, 800:1800] = 255

corners = cv2.goodFeaturesToTrack(f0, maxCorners=50, qualityLevel=0.03, minDistance=30, mask=mask)
corners = corners[:, 0, :] # [M, 2]
print(f"Selected {len(corners)} high-contrast anchor points.")

patch_r = 24 # 49x49 patch
search_r = 40 # search window +- 40 px

# Extract patches from frame 0
patches = []
valid_pts = []
for pt in corners:
    x, y = int(round(pt[0])), int(round(pt[1]))
    if x - patch_r - search_r >= 0 and x + patch_r + search_r < f0.shape[1] and \
       y - patch_r - search_r >= 0 and y + patch_r + search_r < f0.shape[0]:
        patch = f0[y - patch_r : y + patch_r + 1, x - patch_r : x + patch_r + 1]
        patches.append(patch)
        valid_pts.append((x, y))

print(f"Valid patches: {len(patches)}")

# Track each patch across all frames
tracks_x = np.zeros((n_frames, len(patches)), dtype=np.float32)
tracks_y = np.zeros((n_frames, len(patches)), dtype=np.float32)

for i in range(n_frames):
    img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
    for p_idx, ((x0, y0), patch) in enumerate(zip(valid_pts, patches)):
        search_roi = img[y0 - search_r - patch_r : y0 + search_r + patch_r + 1,
                         x0 - search_r - patch_r : x0 + search_r + patch_r + 1]
        res = cv2.matchTemplate(search_roi, patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Sub-pixel refinement via 2D quadratic fit around max_loc
        mx, my = max_loc
        if 1 <= mx < res.shape[1] - 1 and 1 <= my < res.shape[0] - 1:
            # 1D parabolic fit for dx, dy
            dx = (res[my, mx + 1] - res[my, mx - 1]) / (2 * (2 * res[my, mx] - res[my, mx + 1] - res[my, mx - 1]) + 1e-6)
            dy = (res[my + 1, mx] - res[my - 1, mx]) / (2 * (2 * res[my, mx] - res[my + 1, mx] - res[my - 1, mx]) + 1e-6)
            dx = np.clip(dx, -1.0, 1.0)
            dy = np.clip(dy, -1.0, 1.0)
        else:
            dx, dy = 0.0, 0.0
            
        cur_x = (x0 - search_r) + mx + dx
        cur_y = (y0 - search_r) + my + dy
        tracks_x[i, p_idx] = cur_x - x0
        tracks_y[i, p_idx] = cur_y - y0

# Mean displacement across patches
mean_dx = np.mean(tracks_x, axis=1)
mean_dy = np.mean(tracks_y, axis=1)

print(f"Mean displacement X: std={mean_dx.std():.3f} px, peak-to-peak={mean_dx.max()-mean_dx.min():.3f} px")
print(f"Mean displacement Y: std={mean_dy.std():.3f} px, peak-to-peak={mean_dy.max()-mean_dy.min():.3f} px")

# Detrend
t_arr = np.arange(n_frames) / fps
dx_dt = mean_dx - np.polyval(np.polyfit(t_arr, mean_dx, 1), t_arr)
dy_dt = mean_dy - np.polyval(np.polyfit(t_arr, mean_dy, 1), t_arr)

rfftfreq = np.fft.rfftfreq(n_frames, d=1.0/fps)
fft_dx = np.abs(np.fft.rfft(dx_dt)) * 2.0 / n_frames
fft_dy = np.abs(np.fft.rfft(dy_dt)) * 2.0 / n_frames

print("\nTop 5 frequencies in mean X:")
for idx in np.argsort(fft_dx)[::-1][:5]:
    print(f"  {rfftfreq[idx]:.3f} Hz -> Amp: {fft_dx[idx]:.4f} px")

print("\nTop 5 frequencies in mean Y:")
for idx in np.argsort(fft_dy)[::-1][:5]:
    print(f"  {rfftfreq[idx]:.3f} Hz -> Amp: {fft_dy[idx]:.4f} px")
