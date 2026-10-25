import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import cv2

from scripts.prep_real_multipleview import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat

scene_dir = Path("data/multipleview/lego_turbine_2_missing")
cams = read_cameras_binary(scene_dir / "sparse_" / "cameras.bin")
imgs = read_images_binary(scene_dir / "sparse_" / "images.bin")
pts3d = read_points3D_binary(scene_dir / "sparse_" / "points3D.bin")

cam_params = cams[1]
f, cx, cy, k = cam_params["params"]
pts_arr = np.array([p["xyz"] for p in pts3d.values()])

fps = 20.00393

# Let's test image1 (which is cam01)
im = [v for v in imgs.values() if v["name"] == "image1.jpg"][0]
R = qvec2rotmat(im["qvec"])
t = im["tvec"]
p_cam = (R @ pts_arr.T + t.reshape(3, 1)).T
z = p_cam[:, 2]
u = p_cam[:, 0] / z
v = p_cam[:, 1] / z
r2 = u**2 + v**2
radial = 1.0 + k * r2
px = f * u * radial + cx
py = f * v * radial + cy

valid = (z > 0) & (px >= 50) & (px < cam_params["width"] - 50) & (py >= 50) & (py < cam_params["height"] - 50)
seed_pts = np.stack([px[valid], py[valid]], axis=-1).astype(np.float32).reshape(-1, 1, 2)
print(f"Cam01 seed points from COLMAP: {len(seed_pts)}")

frames_dir = scene_dir / "cam01"
frame_files = sorted(list(frames_dir.glob("*.jpg")))
n_frames = len(frame_files)

# Track across all frames
first_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
curr_pts = seed_pts
prev_img = first_img

trajectories = [curr_pts.copy()]
lk_params = dict(winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

valid_mask = np.ones(len(seed_pts), dtype=bool)

for i in range(1, n_frames):
    img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, curr_pts, None, **lk_params)
    back_pts, status_b, err_b = cv2.calcOpticalFlowPyrLK(img, prev_img, next_pts, None, **lk_params)
    d = abs(curr_pts - back_pts).reshape(-1, 2).max(-1)
    good = (status.flatten() == 1) & (status_b.flatten() == 1) & (d < 1.5)
    valid_mask = valid_mask & good
    trajectories.append(next_pts.copy())
    curr_pts = next_pts
    prev_img = img

valid_indices = np.where(valid_mask)[0]
print(f"Robustly tracked across {n_frames} frames: {len(valid_indices)} points")

traj_arr = np.array(trajectories)[:, valid_indices, 0, :] # [T, N, 2]
t_arr = np.arange(n_frames) / fps

# Detrend
detrended = np.zeros_like(traj_arr)
for p in range(traj_arr.shape[1]):
    for dim in range(2):
        poly = np.polyfit(t_arr, traj_arr[:, p, dim], deg=1)
        detrended[:, p, dim] = traj_arr[:, p, dim] - np.polyval(poly, t_arr)

# Displacements
rfftfreq = np.fft.rfftfreq(n_frames, d=1.0/fps)
fft_x = np.abs(np.fft.rfft(detrended[..., 0], axis=0))
fft_y = np.abs(np.fft.rfft(detrended[..., 1], axis=0))
fft_mag = np.sqrt(fft_x**2 + fft_y**2)

# Point-by-point peak analysis
# Check per-point max amplitude in [5, 9] Hz
idx_5_9 = [i for i, f in enumerate(rfftfreq) if 5.0 <= f <= 9.0]
top_pts_amp = []
for p in range(fft_mag.shape[1]):
    peak_idx = idx_5_9[np.argmax(fft_mag[idx_5_9, p])]
    amp = (fft_mag[peak_idx, p] / n_frames) * 2.0
    top_pts_amp.append((amp, rfftfreq[peak_idx], p))

top_pts_amp.sort(reverse=True)
print("\nTop 10 highest vibrating tracked points:")
for amp, f_peak, p_idx in top_pts_amp[:10]:
    pt_orig = seed_pts[valid_indices[p_idx]][0]
    print(f"Point {p_idx} at ({pt_orig[0]:.1f}, {pt_orig[1]:.1f}): Amp = {amp:.3f} px, Peak Freq = {f_peak:.2f} Hz")
