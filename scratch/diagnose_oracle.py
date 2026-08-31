import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom
from sklearn.cluster import KMeans
from scipy.ndimage import label as nd_label

# Let's inspect everything on cubes_k2_ring
repo_root = Path('.').resolve()
run_dir = repo_root / 'runs' / 'cubes-cubes_k2_ring'
traj_file = run_dir / 'trajectories.npz'
traj_data = np.load(traj_file)
xyz = traj_data['canonical_xyz'] # shape: (N, 3)
traj = traj_data['traj']          # shape: (N, T, 3)
opacity = traj_data['opacity'] if 'opacity' in traj_data else np.ones(len(xyz))

# 1. As-is GT in data/multipleview/cubes_k2_ring/gt_segmentation.npz
gt_as_is = np.load('data/multipleview/cubes_k2_ring/gt_segmentation.npz')
gt_pts_asis = gt_as_is['points']
gt_labs_asis = gt_as_is['labels']

print(f"Total Gaussians: {len(xyz)}")
print(f"As-is GT points: {len(gt_pts_asis)}, labels: {np.unique(gt_labs_asis, return_counts=True)}")

# 2. Extract TRUE GT point cloud from USD at TimeCode 0
usd_path = repo_root / 'omniverse-pipeline/data/scenes/cubes_dataset_60fps/cubes_k2_ring.usd'
stage = Usd.Stage.Open(str(usd_path))
xf_cache_0 = UsdGeom.XformCache(Usd.TimeCode(0))

true_gt_pts = []
true_gt_labs = []
cube_names = []

for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Cubes')):
    if prim.GetTypeName() == 'Cube':
        cube = UsdGeom.Cube(prim)
        size = float(cube.GetSizeAttr().Get() or 0.35)
        hs = size / 2.0
        grid = np.linspace(-hs, hs, 25)
        gx, gy, gz = np.meshgrid(grid, grid, grid)
        mask = (np.abs(gx) == hs) | (np.abs(gy) == hs) | (np.abs(gz) == hs)
        P = np.column_stack([gx[mask], gy[mask], gz[mask]]).astype(np.float64)
        
        M = xf_cache_0.GetLocalToWorldTransform(prim)
        Pw = np.array([M.Transform((x, y, z)) for x, y, z in P])
        
        parent = prim.GetParent()
        name = parent.GetName() if (parent and parent.IsValid()) else prim.GetName()
        cube_id = len(cube_names)
        cube_names.append(name)
        
        true_gt_pts.append(Pw)
        true_gt_labs.append(np.full(len(Pw), cube_id, dtype=np.int32))
        print(f"Extracted True GT for {name} (id={cube_id}): {len(Pw)} points, center={Pw.mean(axis=0)}, bounds={Pw.min(axis=0)} to {Pw.max(axis=0)}")

true_gt_points = np.concatenate(true_gt_pts).astype(np.float32)
true_gt_labels = np.concatenate(true_gt_labs).astype(np.int32)

# Save the true GT point cloud
out_true_gt = run_dir / 'true_gt_segmentation.npz'
np.savez(out_true_gt, points=true_gt_points, labels=true_gt_labels)
print(f"Saved true GT point cloud to {out_true_gt}")
