from pathlib import Path
import numpy as np

p_data = Path("data/multipleview/pump01/gt_segmentation.npz")
p_run = list(Path("runs/grid-A40mm_M8").glob("**/gt_segmentation.npz"))[0]
p_traj = Path("runs/grid-A40mm_M8/trajectories.npz")

d_data = np.load(p_data)
d_run = np.load(p_run)
d_traj = np.load(p_traj)

print("Data GT points range:")
print("  min:", d_data["points"].min(axis=0))
print("  max:", d_data["points"].max(axis=0))

print("\nRun GT points range:")
print("  min:", d_run["points"].min(axis=0))
print("  max:", d_run["points"].max(axis=0))

print("\nTraj canonical_xyz range:")
print("  min:", d_traj["canonical_xyz"].min(axis=0))
print("  max:", d_traj["canonical_xyz"].max(axis=0))
