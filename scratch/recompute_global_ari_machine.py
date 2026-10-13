import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent
from pipeline.vendored.host.metrics import adjusted_rand_index, best_iou_matching
from run_pump01_spatial_cc import read_point_cloud_ply, spatial_cc

checkpoints = [
    ("grid-A40mm_M8", "A40mm_M8", 0.005), # headline optimal r=0.5cm
    ("grid-A40mm_M8", "A40mm_M8", 0.003), # r=0.3cm
    ("grid-A20mm_M2", "A20mm_M2", 0.003), # headline optimal r=0.3cm
    ("grid-A20mm_M2", "A20mm_M2", 0.005), # r=0.5cm
    ("grid-A20mm_M4", "A20mm_M4", 0.003), # r=0.3cm
]

gt_path = repo_root / "data" / "multipleview" / "pump01" / "gt_segmentation.npz"
gt_data = np.load(gt_path)
gt_pts = gt_data["points"] # (100000, 3)
gt_labs = gt_data["labels"] # (100000,)

tree = cKDTree(gt_pts)

# Machine Bounding Box:
cad_min = gt_pts.min(axis=0)
cad_max = gt_pts.max(axis=0)
print(f"Machine CAD Bounding Box:")
print(f"  Min: {cad_min}")
print(f"  Max: {cad_max}")
print(f"  Extents: {cad_max - cad_min}")

for dir_name, exp_name, radius_m in checkpoints:
    print(f"\n=======================================================")
    print(f"Analyzing {exp_name} with radius {radius_m*100:.1f}cm")
    print(f"=======================================================")
    run_dir = repo_root / "runs" / dir_name
    traj_data = np.load(run_dir / "trajectories.npz")
    xyz = traj_data["canonical_xyz"]
    
    # 1. 1-NN mapping from 100k machine CAD points to ~360k Gaussians
    dists_to_cad, nn_idx = tree.query(xyz, k=1)
    gt_on_pred = gt_labs[nn_idx]
    
    oracle_roi = (gt_on_pred > 0)
    n_total = len(xyz)
    n_roi = int(oracle_roi.sum())
    
    # S6 Spatial CC baseline
    cc_labels, k_cc = spatial_cc(xyz[oracle_roi], radius=radius_m, min_cluster_size=15)
    full_pred_labels = np.full(n_total, -1, dtype=np.int64)
    full_pred_labels[oracle_roi] = cc_labels
    full_pred_labels[~oracle_roi] = 0
    
    # Baseline full-cloud ARI
    ari_global_all = adjusted_rand_index(gt_on_pred, full_pred_labels)
    ari_roi = adjusted_rand_index(gt_on_pred[oracle_roi], full_pred_labels[oracle_roi])
    
    # Label 0 distribution
    is_label_0 = (gt_on_pred == 0)
    n_label_0 = int(is_label_0.sum())
    
    print(f"Total Gaussians: {n_total}")
    print(f"Mapped to Label 0: {n_label_0} ({n_label_0/n_total*100:.2f}%)")
    print(f"Dynamic ROI (Labels > 0): {n_roi} ({n_roi/n_total*100:.2f}%)")
    print(f"Full-cloud Global ARI: {ari_global_all:.6f}")
    print(f"Within-ROI ARI:        {ari_roi:.6f}")
    
    # Subsets of machine points:
    # 1. Inside CAD BBox (with 5cm margin)
    in_bbox = np.all((xyz >= cad_min - 0.05) & (xyz <= cad_max + 0.05), axis=1)
    # 2. Near CAD surfaces (d <= 5cm, 2cm, 1cm)
    near_5cm = (dists_to_cad <= 0.05)
    near_2cm = (dists_to_cad <= 0.02)
    near_1cm = (dists_to_cad <= 0.01)
    
    subsets = [
        ("CAD BBox (+5cm)", in_bbox),
        ("Dist to CAD <= 5cm", near_5cm),
        ("Dist to CAD <= 2cm", near_2cm),
        ("Dist to CAD <= 1cm", near_1cm),
    ]
    
    print("\nRecomputed Global ARI on Machine Subsets:")
    for s_name, s_mask in subsets:
        n_sub = int(s_mask.sum())
        p_sub = n_sub / n_total * 100
        l0_sub = int((gt_on_pred[s_mask] == 0).sum())
        l0_pct = l0_sub / n_sub * 100
        ari_sub = adjusted_rand_index(gt_on_pred[s_mask], full_pred_labels[s_mask])
        print(f"  {s_name:<20}: N={n_sub:6d} ({p_sub:5.2f}% cloud) | Label 0={l0_sub:5d} ({l0_pct:5.2f}%) | ARI={ari_sub:.6f}")
