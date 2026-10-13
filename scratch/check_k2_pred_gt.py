from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent

# 1. Check with data GT
data_gt = repo_root / "data" / "multipleview" / "cubes_k2_ring" / "gt_segmentation.npz"
run_traj = repo_root / "runs" / "cubes-cubes_k2_ring" / "trajectories.npz"
run_gt = list((repo_root / "runs" / "cubes-cubes_k2_ring").glob("**/gt_segmentation.npz"))[0]

traj_d = np.load(run_traj)
xyz = traj_d["canonical_xyz"]

d_gt = np.load(data_gt)
tree_data = cKDTree(d_gt["points"])
_, nn_data = tree_data.query(xyz, k=1)
gt_on_pred_data = d_gt["labels"][nn_data]

r_gt = np.load(run_gt)
tree_run = cKDTree(r_gt["points"])
_, nn_run = tree_run.query(xyz, k=1)
gt_on_pred_run = r_gt["labels"][nn_run]

print("=== DATA GT on pred ===")
u_d, c_d = np.unique(gt_on_pred_data, return_counts=True)
for u, c in zip(u_d, c_d):
    print(f"Label {u}: {c} points ({c/len(xyz)*100:.2f}%)")

print("\n=== RUN GT on pred ===")
u_r, c_r = np.unique(gt_on_pred_run, return_counts=True)
for u, c in zip(u_r, c_r):
    print(f"Label {u}: {c} points ({c/len(xyz)*100:.2f}%)")
