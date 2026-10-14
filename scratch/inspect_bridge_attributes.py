import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index, compute_pairs_split_rate, PUMP01_BASELINE_MERGED_PAIRS_141

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

# Let's inspect which specific edges on M8 caused the seeds to drop!
# Baseline M8 at r=0.003 has 11 valid components.
# Let's see which components in M8 are delicate.
n_m8 = d_m8["n"]
k_pairs_m8 = d_m8["pairs"]
adj_m8 = csr_matrix((np.ones(len(k_pairs_m8), dtype=bool), (k_pairs_m8[:, 0], k_pairs_m8[:, 1])), shape=(n_m8, n_m8))
adj_m8 = adj_m8 + adj_m8.T
_, raw_labels_m8 = csgraph.connected_components(adj_m8, directed=False)
u_m8, c_m8 = np.unique(raw_labels_m8, return_counts=True)
valid_comp_m8 = u_m8[c_m8 >= 15]

# Points in the 11 valid components on M8
pts_in_valid_m8 = np.isin(raw_labels_m8, valid_comp_m8)
# What is the minimum opacity on edges internal to the 11 valid components?
internal_edges_m8 = pts_in_valid_m8[k_pairs_m8[:, 0]] & pts_in_valid_m8[k_pairs_m8[:, 1]] & (raw_labels_m8[k_pairs_m8[:, 0]] == raw_labels_m8[k_pairs_m8[:, 1]])
print(f"M8 internal edges in valid seeds: {internal_edges_m8.sum()}")
internal_op_prod = d_m8["op_prod"][internal_edges_m8]
print(f"M8 internal op_prod: min={internal_op_prod.min():.6f}, 1%={np.percentile(internal_op_prod, 1):.6f}, 5%={np.percentile(internal_op_prod, 5):.6f}")

# And what about M2 internal edges?
n_m2 = d_m2["n"]
k_pairs_m2 = d_m2["pairs"]
adj_m2 = csr_matrix((np.ones(len(k_pairs_m2), dtype=bool), (k_pairs_m2[:, 0], k_pairs_m2[:, 1])), shape=(n_m2, n_m2))
adj_m2 = adj_m2 + adj_m2.T
_, raw_labels_m2 = csgraph.connected_components(adj_m2, directed=False)
u_m2, c_m2 = np.unique(raw_labels_m2, return_counts=True)
valid_comp_m2 = u_m2[c_m2 >= 15]
pts_in_valid_m2 = np.isin(raw_labels_m2, valid_comp_m2)
internal_edges_m2 = pts_in_valid_m2[k_pairs_m2[:, 0]] & pts_in_valid_m2[k_pairs_m2[:, 1]] & (raw_labels_m2[k_pairs_m2[:, 0]] == raw_labels_m2[k_pairs_m2[:, 1]])
print(f"M2 internal edges in valid seeds: {internal_edges_m2.sum()}")
internal_op_prod_m2 = d_m2["op_prod"][internal_edges_m2]
print(f"M2 internal op_prod: min={internal_op_prod_m2.min():.6f}, 1%={np.percentile(internal_op_prod_m2, 1):.6f}, 5%={np.percentile(internal_op_prod_m2, 5):.6f}")

# What about bridging edges between Cat B parts on M2?
# Which edges connect different parts in Cat B?
cat_b_set = set()
for p1, p2 in cat_b_pairs:
    cat_b_set.add((min(p1, p2), max(p1, p2)))

gt_i_m2 = d_m2["sub_gt"][k_pairs_m2[:, 0]]
gt_j_m2 = d_m2["sub_gt"][k_pairs_m2[:, 1]]
cat_b_bridging_edges = []
for idx in range(len(k_pairs_m2)):
    g1, g2 = min(gt_i_m2[idx], gt_j_m2[idx]), max(gt_i_m2[idx], gt_j_m2[idx])
    if (g1, g2) in cat_b_set:
        cat_b_bridging_edges.append(idx)
cat_b_bridging_edges = np.array(cat_b_bridging_edges)
print(f"M2 edges bridging Cat B pairs: {len(cat_b_bridging_edges)}")
if len(cat_b_bridging_edges) > 0:
    bridge_op = d_m2["op_prod"][cat_b_bridging_edges]
    bridge_vol = d_m2["vol_max"][cat_b_bridging_edges]
    bridge_aniso = d_m2["aniso_max"][cat_b_bridging_edges]
    bridge_dist = d_m2["d_ij"][cat_b_bridging_edges]
    print(f"Bridge op_prod: min={bridge_op.min():.5f}, median={np.median(bridge_op):.5f}, max={bridge_op.max():.5f}")
    print(f"Bridge dist mm: min={bridge_dist.min()*1000:.2f}, median={np.median(bridge_dist)*1000:.2f}, max={bridge_dist.max()*1000:.2f}")
    print(f"Bridge vol mm: min={bridge_vol.min()*1000:.2f}, median={np.median(bridge_vol)*1000:.2f}, max={bridge_vol.max()*1000:.2f}")
    print(f"Bridge aniso: min={bridge_aniso.min():.1f}, median={np.median(bridge_aniso):.1f}, max={bridge_aniso.max():.1f}")
