import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

run_dir = Path('runs/cubes-cubes_k2_ring')
traj_data = np.load(run_dir / 'trajectories.npz')
xyz = traj_data['canonical_xyz']
traj = traj_data['traj']
opacity = traj_data['opacity'] if 'opacity' in traj_data else np.ones(len(xyz))

gt_data = np.load('data/multipleview/cubes_k2_ring/gt_segmentation.npz')
gt_points = gt_data['points']
gt_labels = gt_data['labels']

tree = cKDTree(gt_points)
dists, nn = tree.query(xyz, k=1)
mapped_labels = gt_labels[nn]

# Cube bounding boxes / centers in canonical space
for l in [0, 1]:
    pts_l = gt_points[gt_labels == l]
    print(f"GT Cube {l}: n={len(pts_l)}, min={pts_l.min(axis=0)}, max={pts_l.max(axis=0)}, center={pts_l.mean(axis=0)}")

# Check distances of Gaussians mapped to cube 0 and 1
print("\n--- Gaussian distance to nearest GT point ---")
for threshold in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    close = dists < threshold
    print(f"Dist < {threshold*100:.0f}cm: {close.sum()} / {len(xyz)} ({close.sum()/len(xyz)*100:.1f}%) | "
          f"Cube 0: {((mapped_labels == 0) & close).sum()}, Cube 1: {((mapped_labels == 1) & close).sum()}")

# What are the 361k Gaussians? Where are they?
print(f"\nTotal Gaussians: {len(xyz)}")
print(f"Gaussian z: min={xyz[:, 2].min():.2f}, max={xyz[:, 2].max():.2f}")
print(f"Gaussian x: min={xyz[:, 0].min():.2f}, max={xyz[:, 0].max():.2f}")
print(f"Gaussian y: min={xyz[:, 1].min():.2f}, max={xyz[:, 1].max():.2f}")

# How many Gaussians are actual cube surfaces vs room/floor?
# Cube size is 0.35m (z is 0 to 0.35m, x/y in [-1.5, 1.5])
# Floor is at z=0, walls at x,y = +/- 5.5m
room_pts = (dists > 0.2)
print(f"\nGaussians far from any cube surface (>20cm, i.e. floor/walls/floaters): {room_pts.sum()} ({room_pts.sum()/len(xyz)*100:.1f}%)")
print(f"Gaussians near cube surfaces (<=20cm): {(~room_pts).sum()} ({(~room_pts).sum()/len(xyz)*100:.1f}%)")
