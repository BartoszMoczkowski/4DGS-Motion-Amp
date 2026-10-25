import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np

from scripts.prep_real_multipleview import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat

scene_dir = Path("data/multipleview/lego_turbine_2_missing")
cams = read_cameras_binary(scene_dir / "sparse_" / "cameras.bin")
imgs = read_images_binary(scene_dir / "sparse_" / "images.bin")
pts3d = read_points3D_binary(scene_dir / "sparse_" / "points3D.bin")

cam_params = cams[1]
print("Camera model id:", cam_params["model_id"], "w:", cam_params["width"], "h:", cam_params["height"])
print("Params:", cam_params["params"]) # f, cx, cy, k

f, cx, cy, k = cam_params["params"]

pts_arr = np.array([p["xyz"] for p in pts3d.values()])
print(f"Total sparse points: {len(pts_arr)}")

for img_id, im in sorted(imgs.items(), key=lambda kv: kv[1]["name"]):
    R = qvec2rotmat(im["qvec"])
    t = im["tvec"]
    # Project: p_cam = R @ p_world + t
    p_cam = (R @ pts_arr.T + t.reshape(3, 1)).T
    z = p_cam[:, 2]
    u = p_cam[:, 0] / z
    v = p_cam[:, 1] / z
    # SIMPLE_RADIAL distortion
    r2 = u**2 + v**2
    radial = 1.0 + k * r2
    px = f * u * radial + cx
    py = f * v * radial + cy
    
    valid = (z > 0) & (px >= 0) & (px < cam_params["width"]) & (py >= 0) & (py < cam_params["height"])
    px_v = px[valid]
    py_v = py[valid]
    print(f"{im['name']} (cam id {im['camera_id']}): {np.sum(valid)} points visible. X=[{px_v.min():.0f}, {px_v.max():.0f}], Y=[{py_v.min():.0f}, {py_v.max():.0f}], center=({px_v.mean():.0f}, {py_v.mean():.0f})")
