import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import (
    adjusted_rand_index,
    compute_pairs_split_rate,
    PUMP01_BASELINE_MERGED_PAIRS_141,
)

run_dir = Path("runs/grid-A20mm_M2")
traj = np.load(run_dir / "trajectories.npz")
xyz = traj["canonical_xyz"]

gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
gt_data = np.load(gt_file)
tree = cKDTree(gt_data["points"])
_, nn = tree.query(xyz, k=1)
gt_on_pred = gt_data["labels"][nn]
oracle_roi = (gt_on_pred > 0)

sub_xyz = xyz[oracle_roi]
sub_gt = gt_on_pred[oracle_roi]
n = len(sub_xyz)

# Load PLY attributes
ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
ply_file = ply_files[-1]

with open(ply_file, "rb") as f:
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
with open(ply_file, "rb") as f:
    f.seek(h_len)
    data = np.fromfile(f, dtype=dt)

sub_raw_op = data["opacity"][oracle_roi]
sub_op = 1.0 / (1.0 + np.exp(-np.clip(sub_raw_op, -20.0, 20.0)))
sub_scales = np.exp(np.stack([
    data["scale_0"][oracle_roi],
    data["scale_1"][oracle_roi],
    data["scale_2"][oracle_roi],
], axis=1))
sub_vol = np.cbrt(sub_scales.prod(axis=1))
sub_aniso = sub_scales.max(axis=1) / np.maximum(sub_scales.min(axis=1), 1e-8)

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

tree_sub = cKDTree(sub_xyz)
r_base = 0.003  # 0.3 cm
pairs = tree_sub.query_pairs(r=r_base, output_type="ndarray")
d_ij = np.linalg.norm(sub_xyz[pairs[:, 0]] - sub_xyz[pairs[:, 1]], axis=1)
op_i = sub_op[pairs[:, 0]]
op_j = sub_op[pairs[:, 1]]
op_prod = op_i * op_j
vol_i = sub_vol[pairs[:, 0]]
vol_j = sub_vol[pairs[:, 1]]
max_vol = np.maximum(vol_i, vol_j)
aniso_i = sub_aniso[pairs[:, 0]]
aniso_j = sub_aniso[pairs[:, 1]]
max_aniso = np.maximum(aniso_i, aniso_j)

print(f"Total candidate spatial pairs at r={r_base*100}cm: {len(pairs)}")


def evaluate_graph(valid_edge_mask, name=""):
    k_pairs = pairs[valid_edge_mask]
    if len(k_pairs) > 0:
        data = np.ones(len(k_pairs), dtype=bool)
        adj = csr_matrix((data, (k_pairs[:, 0], k_pairs[:, 1])), shape=(n, n))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n, n), dtype=bool)

    n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_comp = unique[counts >= 15]

    # Dilated
    labels_dilated = np.full(n, -1, dtype=np.int64)
    comp_map = {comp: idx for idx, comp in enumerate(valid_comp)}
    for comp, idx in comp_map.items():
        labels_dilated[raw_labels == comp] = idx
    unassigned = np.where(labels_dilated == -1)[0]
    assigned = np.where(labels_dilated >= 0)[0]
    if len(assigned) > 0 and len(unassigned) > 0:
        sub_tree = cKDTree(sub_xyz[assigned])
        _, nearest = sub_tree.query(sub_xyz[unassigned], k=1)
        labels_dilated[unassigned] = labels_dilated[assigned[nearest]]

    # No dilation: cluster -1
    labels_cminus1 = np.full(n, -1, dtype=np.int64)
    for comp, idx in comp_map.items():
        labels_cminus1[raw_labels == comp] = idx

    # No dilation: singletons
    labels_singletons = labels_cminus1.copy()
    labels_singletons[unassigned] = np.arange(len(valid_comp), len(valid_comp) + len(unassigned))

    ari_dilated = float(adjusted_rand_index(sub_gt, labels_dilated))
    ari_cminus1 = float(adjusted_rand_index(sub_gt, labels_cminus1))
    ari_single = float(adjusted_rand_index(sub_gt, labels_singletons))

    ps_141 = compute_pairs_split_rate(sub_gt, labels_dilated)
    ps_cat_b = compute_pairs_split_rate(sub_gt, labels_dilated, merged_pairs=cat_b_pairs)

    res = {
        "name": name,
        "n_edges": int(len(k_pairs)),
        "n_seeds": int(len(valid_comp)),
        "n_assigned": int(len(assigned)),
        "ari_dilated": ari_dilated,
        "ari_cminus1": ari_cminus1,
        "ari_singletons": ari_single,
        "pairs_split_141": ps_141["pairs_split_rate"],
        "n_split_141": ps_141["n_pairs_split"],
        "cat_b_split": ps_cat_b["pairs_split_rate"],
        "n_split_cat_b": ps_cat_b["n_pairs_split"],
    }
    return res


print("\n--- Sweeping Opacity Product Thresholds (op_prod >= tau) ---")
for tau_op in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    mask = op_prod >= tau_op
    res = evaluate_graph(mask, f"op_prod>={tau_op}")
    print(
        f"tau_op={tau_op:5.3f} | edges={res['n_edges']:4d} | seeds={res['n_seeds']:2d} | "
        f"ARI_dil={res['ari_dilated']:.4f} | ARI_c-1={res['ari_cminus1']:.4f} | ARI_sin={res['ari_singletons']:.6f} | "
        f"CatB_split={res['cat_b_split']:.4f} ({res['n_split_cat_b']:2d}/102) | 141_split={res['pairs_split_141']:.4f} ({res['n_split_141']:2d}/141)"
    )

print("\n--- Sweeping Power Weighting w = d_ij / (op_prod)^alpha <= r ---")
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    w = d_ij / np.maximum(op_prod, 1e-6)**alpha
    mask = w <= r_base
    res = evaluate_graph(mask, f"alpha={alpha}")
    print(
        f"alpha={alpha:4.2f} | edges={res['n_edges']:4d} | seeds={res['n_seeds']:2d} | "
        f"ARI_dil={res['ari_dilated']:.4f} | ARI_c-1={res['ari_cminus1']:.4f} | "
        f"CatB_split={res['cat_b_split']:.4f} ({res['n_split_cat_b']:2d}/102) | 141_split={res['pairs_split_141']:.4f} ({res['n_split_141']:2d}/141)"
    )

print("\n--- Sweeping Scale & Anisotropy Pruning ---")
for max_v in [0.005, 0.003, 0.002, 0.0015, 0.001]:
    for tau_op in [0.0, 0.01, 0.03]:
        mask = (op_prod >= tau_op) & (max_vol <= max_v)
        res = evaluate_graph(mask, f"tau={tau_op}_maxvol={max_v}")
        print(
            f"tau={tau_op:4.2f}, max_vol={max_v*1000:4.1f}mm | edges={res['n_edges']:4d} | seeds={res['n_seeds']:2d} | "
            f"ARI_dil={res['ari_dilated']:.4f} | CatB_split={res['cat_b_split']:.4f} ({res['n_split_cat_b']:2d}/102)"
        )
