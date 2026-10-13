import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "orchestrator"))
sys.path.insert(0, str(repo_root / "scene-gen"))
sys.path.insert(0, str(repo_root / "scratch"))

from pipeline.vendored.host.multicut import build_multichannel_graph
from run_pump01_spatial_cc import read_point_cloud_ply
from run_grid_multicut_calibrated import fit_logistic_weights, compute_feature_auroc

print("Loading grid-A20mm_M2...", flush=True)
train_dir = repo_root / "runs" / "grid-A20mm_M2"
train_traj_data = np.load(train_dir / "trajectories.npz")
train_xyz = train_traj_data["canonical_xyz"]
train_traj = train_traj_data["traj"]
train_ply = read_point_cloud_ply(train_dir / "train_out/point_cloud/iteration_15000/point_cloud.ply")
train_rgb = train_ply["rgb"]

train_gt_file = list(train_dir.glob("**/gt_segmentation.npz"))[0]
train_gt_data = np.load(train_gt_file)
train_tree = cKDTree(train_gt_data["points"])
_, train_nn = train_tree.query(train_xyz, k=1)
train_gt_labels = train_gt_data["labels"][train_nn]

train_edges, train_feats, train_info = build_multichannel_graph(
    train_xyz,
    traj=train_traj,
    rgb=train_rgb,
    k=12,
    radius_max=0.03,
    f0=10.0,
    include_lifted=False,
)
u_tr = train_edges[:, 0]
v_tr = train_edges[:, 1]
y_train = (train_gt_labels[u_tr] == train_gt_labels[v_tr]).astype(np.float64)

print(f"Edges: {len(train_edges)}, Same: {y_train.sum()} ({y_train.mean()*100:.2f}%)", flush=True)

# Per channel
auroc_spatial = compute_feature_auroc(train_feats[:, 0], y_train.astype(bool), higher_is_same=False)
auroc_appearance = compute_feature_auroc(train_feats[:, 1], y_train.astype(bool), higher_is_same=False)
auroc_phasor = compute_feature_auroc(train_feats[:, 3], y_train.astype(bool), higher_is_same=False)
auroc_traj = compute_feature_auroc(train_feats[:, 4], y_train.astype(bool), higher_is_same=False)
print(f"M2 AUROC: spatial={auroc_spatial:.4f}, app={auroc_appearance:.4f}, phasor={auroc_phasor:.4f}, traj={auroc_traj:.4f}", flush=True)

w_full, auroc_fit_d = fit_logistic_weights(
    train_feats[:, [0, 1, 3]], y_train, ["beta_xyz", "beta_rgb", "beta_phasor"]
)
print(f"M2 w_full: {w_full}", flush=True)
print(f"M2 Fused AUROC: {auroc_fit_d:.4f}", flush=True)
