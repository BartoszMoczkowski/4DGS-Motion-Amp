import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import (
    adjusted_rand_index,
    compute_pairs_split_rate,
    PUMP01_BASELINE_MERGED_PAIRS_141,
)

# Test on A40mm_M8
run_dir_m8 = Path("runs/grid-A40mm_M8")
traj_m8 = np.load(run_dir_m8 / "trajectories.npz")
xyz_m8 = traj_m8["canonical_xyz"]

gt_file_m8 = list(run_dir_m8.glob("**/gt_segmentation.npz"))[0]
gt_data_m8 = np.load(gt_file_m8)
tree_m8 = cKDTree(gt_data_m8["points"])
_, nn_m8 = tree_m8.query(xyz_m8, k=1)
gt_on_pred_m8 = gt_data_m8["labels"][nn_m8]
oracle_roi_m8 = (gt_on_pred_m8 > 0)

sub_xyz_m8 = xyz_m8[oracle_roi_m8]
sub_gt_m8 = gt_on_pred_m8[oracle_roi_m8]
n_m8 = len(sub_xyz_m8)

# Load PLY attributes for M8
ply_files_m8 = list((run_dir_m8 / "train_out").glob("**/point_cloud.ply"))
ply_files_m8.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
ply_file_m8 = ply_files_m8[-1]

with open(ply_file_m8, "rb") as f:
    header = b""
    while b"end_header\n" not in header:
        header += f.readline()
    h_len = f.tell()
props = [
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
]
for i in range(45):
    props.append((f"f_rest_{i}", "<f4"))
props.extend([
    ("opacity", "<f4"),
    ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
    ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
])
dt = np.dtype(props)
with open(ply_file_m8, "rb") as f:
    f.seek(h_len)
    data_m8 = np.fromfile(f, dtype=dt)

sub_raw_op_m8 = data_m8["opacity"][oracle_roi_m8]
sub_op_m8 = 1.0 / (1.0 + np.exp(-np.clip(sub_raw_op_m8, -20.0, 20.0)))

tree_sub_m8 = cKDTree(sub_xyz_m8)
r_base = 0.003  # 0.3 cm
pairs_m8 = tree_sub_m8.query_pairs(r=r_base, output_type="ndarray")
op_i_m8 = sub_op_m8[pairs_m8[:, 0]]
op_j_m8 = sub_op_m8[pairs_m8[:, 1]]
op_prod_m8 = op_i_m8 * op_j_m8

print(f"=== Held-out A40mm_M8 at frozen r=0.3cm ({len(pairs_m8)} candidate pairs) ===")

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

for tau_op in [0.0, 0.006, 0.008, 0.010, 0.012, 0.014]:
    mask = op_prod_m8 >= tau_op
    k_pairs = pairs_m8[mask]
    data_adj = np.ones(len(k_pairs), dtype=bool)
    adj = csr_matrix((data_adj, (k_pairs[:, 0], k_pairs[:, 1])), shape=(n_m8, n_m8))
    adj = adj + adj.T

    n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_comp = unique[counts >= 15]

    labels_dilated = np.full(n_m8, -1, dtype=np.int64)
    comp_map = {comp: idx for idx, comp in enumerate(valid_comp)}
    for comp, idx in comp_map.items():
        labels_dilated[raw_labels == comp] = idx
    unassigned = np.where(labels_dilated == -1)[0]
    assigned = np.where(labels_dilated >= 0)[0]
    if len(assigned) > 0 and len(unassigned) > 0:
        sub_tree = cKDTree(sub_xyz_m8[assigned])
        _, nearest = sub_tree.query(sub_xyz_m8[unassigned], k=1)
        labels_dilated[unassigned] = labels_dilated[assigned[nearest]]

    # No dilation: cluster -1
    labels_cminus1 = np.full(n_m8, -1, dtype=np.int64)
    for comp, idx in comp_map.items():
        labels_cminus1[raw_labels == comp] = idx

    # No dilation: singletons
    labels_singletons = labels_cminus1.copy()
    labels_singletons[unassigned] = np.arange(len(valid_comp), len(valid_comp) + len(unassigned))

    ari_dilated = float(adjusted_rand_index(sub_gt_m8, labels_dilated))
    ari_cminus1 = float(adjusted_rand_index(sub_gt_m8, labels_cminus1))
    ari_single = float(adjusted_rand_index(sub_gt_m8, labels_singletons))

    ps_141 = compute_pairs_split_rate(sub_gt_m8, labels_dilated)
    ps_cat_b = compute_pairs_split_rate(sub_gt_m8, labels_dilated, merged_pairs=cat_b_pairs)

    print(
        f"tau_op={tau_op:.3f} | edges={len(k_pairs):4d} | seeds={len(valid_comp):2d} | "
        f"ARI_dilated={ari_dilated:.4f} (gate>=0.3847: {ari_dilated>=0.3847}) | "
        f"ARI_c-1={ari_cminus1:.4f} | ARI_single={ari_single:.6f} | "
        f"CatB_split={ps_cat_b['pairs_split_rate']:.4f} ({ps_cat_b['n_pairs_split']:2d}/102) | "
        f"141_split={ps_141['pairs_split_rate']:.4f} ({ps_141['n_pairs_split']:2d}/141)"
    )
