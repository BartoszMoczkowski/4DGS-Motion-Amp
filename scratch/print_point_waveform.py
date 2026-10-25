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

frames_dir = scene_dir / "cam01"
frame_files = sorted(list(frames_dir.glob("*.jpg")))
n_frames = len(frame_files)

first_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
curr_pts = seed_pts
prev_img = first_img

trajectories = [curr_pts.copy()]
lk_params = dict(winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

for i in range(1, n_frames):
    img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, curr_pts, None, **lk_params)
    trajectories.append(next_pts.copy())
    curr_pts = next_pts
    prev_img = img

# Point 519
p_idx = 519
pt_traj = np.array(trajectories)[:, p_idx, 0, :] # [164, 2]
x = pt_traj[:, 0] - np.mean(pt_traj[:, 0])
y = pt_traj[:, 1] - np.mean(pt_traj[:, 1])

print(f"Point 519: X range=[{x.min():.3f}, {x.max():.3f}], std={x.std():.3f}")
print(f"Point 519: Y range=[{y.min():.3f}, {y.max():.3f}], std={y.std():.3f}")
print("First 20 X values:", np.round(x[:20], 3))
