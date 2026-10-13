import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent
from pipeline.vendored.host.metrics import adjusted_rand_index, best_iou_matching
from partition_swap import fit_spatial_connected_components

run_id = "cubes-cubes_k2_ring"
run_dir = repo_root / "runs" / run_id

# Load Trajectories & Coordinates
traj_d = np.load(run_dir / "trajectories.npz")
xyz = traj_d["canonical_xyz"]
traj = traj_d["traj"]

# Load Run GT
gt_d = np.load(run_dir / "convert_out/data/multipleview/cubes_k2_ring/gt_segmentation.npz")
tree = cKDTree(gt_d["points"])
_, nn = tree.query(xyz, k=1)
gt_on_pred = gt_d["labels"][nn]

print(f"Total Gaussians: {len(xyz)}")
print(f"GT class 0 (Cube 0): {(gt_on_pred == 0).sum()} points")
print(f"GT class 1 (Cube 1): {(gt_on_pred == 1).sum()} points")

# When ROI = labels >= 0:
roi_mask = (gt_on_pred >= 0) # all 361,180 points!
print(f"ROI points (labels >= 0): {roi_mask.sum()}")

# 1. Evaluate Kabsch
# From runs/cubes_seg_kabsch_results.csv line 31:
# global ari was 0.7909658016669537, mean_iou = 0.8856773248984364, k_pred = 4
# Since roi_mask is all points, within_roi ARI is identical to global ARI: 0.7910!

# 2. Evaluate Mask Lift Oracle
# From runs/cubes_seg_mask_lift_oracle_results.csv line 31:
# global ari was 0.51181304618099, mean_iou = 0.6045745756526992, k_pred = 378
# within_roi ARI: 0.5118!

# 3. Evaluate Spatial Connected Components across radii on the 2 cubes (labels >= 0)
print("\n--- Evaluating Spatial CC on both cubes (labels >= 0) ---")
for rad in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    labs_cc, k_cc = fit_spatial_connected_components(xyz[roi_mask], radius=rad)
    ari_cc = adjusted_rand_index(gt_on_pred[roi_mask], labs_cc)
    iou_cc, _ = best_iou_matching(gt_on_pred[roi_mask], labs_cc)
    print(f"Radius {rad:.2f}m: ARI={ari_cc:.4f}, IoU={iou_cc:.4f}, K_pred={k_cc}")
