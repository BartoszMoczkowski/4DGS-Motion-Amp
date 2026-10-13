from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

repo_root = Path(__file__).resolve().parent.parent

# 1. Load M4 Oracle Identity Segmentation
run_dir = repo_root / "runs" / "grid-A20mm_M4"
seg_path = run_dir / "segmentation_identity_oracle.npz"
seg_data = np.load(seg_path)
pred_points = seg_data["points"]
pred_labels = seg_data["labels"]

# 2. Load GT
gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
gt_data = np.load(gt_file)
gt_points = gt_data["points"]
gt_labels = gt_data["labels"]

tree = cKDTree(gt_points)
dists, nn_idx = tree.query(pred_points, k=1)
gt_on_pred = gt_labels[nn_idx]

# 3. Classify Machine vs Non-Machine Gaussians
cad_min = gt_points.min(axis=0)
cad_max = gt_points.max(axis=0)
in_bbox = np.all((pred_points >= cad_min - 0.05) & (pred_points <= cad_max + 0.05), axis=1)
near_cad_5cm = (dists <= 0.05)

is_gt_0 = (gt_on_pred == 0)
is_roi = (gt_on_pred > 0)
is_room = is_gt_0 & (~in_bbox)  # Distant room/floor/walls mapped to label 0
is_base = is_gt_0 & in_bbox     # Actual machine baseplate points mapped to label 0

n_total = len(pred_points)
n_clusters = len(np.unique(pred_labels))

print(f"Total Gaussians: {n_total}")
print(f"Total Clusters:  {n_clusters}")
print(f"GT Label 0 Gaussians: {is_gt_0.sum()} ({is_gt_0.sum()/n_total*100:.2f}%)")
print(f"  - Machine Baseplate (in BBox): {is_base.sum()} ({is_base.sum()/n_total*100:.2f}%)")
print(f"  - Non-Machine Room/Floor:       {is_room.sum()} ({is_room.sum()/n_total*100:.2f}%)")
print(f"GT Dynamic ROI (Labels > 0):    {is_roi.sum()} ({is_roi.sum()/n_total*100:.2f}%)")

# Calculate ARIs
ari_global = adjusted_rand_score(gt_on_pred, pred_labels)
ari_roi = adjusted_rand_score(gt_on_pred[is_roi], pred_labels[is_roi])
ari_bbox = adjusted_rand_score(gt_on_pred[in_bbox], pred_labels[in_bbox])
ari_near5cm = adjusted_rand_score(gt_on_pred[near_cad_5cm], pred_labels[near_cad_5cm])

print(f"\nReported Global ARI:      {ari_global:.6f}")
print(f"Reported Within-ROI ARI:  {ari_roi:.6f}")
print(f"Machine BBox ARI:         {ari_bbox:.6f}")
print(f"Near CAD (<=5cm) ARI:     {ari_near5cm:.6f}")

# 4. Tabulate Per-Cluster Background Counts
u_clusters = np.unique(pred_labels)
cluster_table = []

for c in sorted(u_clusters):
    c_mask = (pred_labels == c)
    c_size = int(c_mask.sum())
    c_l0 = int((c_mask & is_gt_0).sum())
    c_room = int((c_mask & is_room).sum())
    c_base = int((c_mask & is_base).sum())
    c_roi = int((c_mask & is_roi).sum())
    
    # Dominant GT class in this cluster
    gt_in_c = gt_on_pred[c_mask]
    u_g, cnt_g = np.unique(gt_in_c, return_counts=True)
    top_gt = int(u_g[cnt_g.argmax()])
    top_gt_cnt = int(cnt_g.max())
    
    # Distance stats of points in this cluster
    c_dists = dists[c_mask]
    med_dist = float(np.median(c_dists))
    max_dist = float(np.max(c_dists))
    
    cluster_table.append({
        "cluster_id": int(c),
        "total_pts": c_size,
        "gt_label_0_total": c_l0,
        "pct_label_0": c_l0 / c_size * 100,
        "room_floor_pts": c_room,
        "machine_base_pts": c_base,
        "roi_pts": c_roi,
        "top_gt_label": top_gt,
        "top_gt_pct": top_gt_cnt / c_size * 100,
        "median_dist_m": med_dist,
        "max_dist_m": max_dist,
    })

print("\n=== PER-CLUSTER BACKGROUND COUNT TABLE ===")
header = f"{'Cluster':<8} | {'Total':<7} | {'Label 0':<8} | {'% L0':<6} | {'Room/Floor':<10} | {'Baseplate':<9} | {'ROI':<7} | {'Top GT':<6} | {'MedDist(m)':<10} | {'MaxDist(m)':<10}"
print(header)
print("-" * len(header))
for r in cluster_table:
    print(f"{r['cluster_id']:<8d} | {r['total_pts']:<7d} | {r['gt_label_0_total']:<8d} | {r['pct_label_0']:<6.1f} | {r['room_floor_pts']:<10d} | {r['machine_base_pts']:<9d} | {r['roi_pts']:<7d} | {r['top_gt_label']:<6d} | {r['median_dist_m']:<10.3f} | {r['max_dist_m']:<10.3f}")

# Also check Oracle Mask identity training details
# How were oracle masks generated for identity encoding training?
# In OracleMaskProvider, did it have a mask for the background?
