import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data
d_m8 = load_data('grid-A40mm_M8')

from scipy.sparse import csgraph, csr_matrix
n = d_m8['n']
pairs = d_m8['pairs']
adj = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
adj = adj + adj.T
_, raw_labels = csgraph.connected_components(adj, directed=False)
u, c = np.unique(raw_labels, return_counts=True)
valid_comp = u[c >= 15]

# For each component of size 15 or 16, let's find which edges if removed break the component!
small_valid = [comp for comp in valid_comp if np.sum(raw_labels == comp) <= 16]
for comp in small_valid:
    comp_pts = np.where(raw_labels == comp)[0]
    in_comp_mask = np.isin(pairs[:, 0], comp_pts) & np.isin(pairs[:, 1], comp_pts)
    comp_edges = np.where(in_comp_mask)[0]
    print(f"Comp {comp}: pts={len(comp_pts)}, edges={len(comp_edges)}")
    critical_edges = []
    for e in comp_edges:
        sub_mask = np.ones(len(pairs), dtype=bool)
        sub_mask[e] = False
        adj_sub = csr_matrix((np.ones(sub_mask.sum(), dtype=bool), (pairs[sub_mask, 0], pairs[sub_mask, 1])), shape=(n, n))
        adj_sub = adj_sub + adj_sub.T
        _, sub_labels = csgraph.connected_components(adj_sub, directed=False)
        sub_u, sub_c = np.unique(sub_labels[comp_pts], return_counts=True)
        if (sub_c >= 15).sum() == 0:
            critical_edges.append(e)
    print(f"  Critical bridge edges in comp {comp}: {len(critical_edges)} out of {len(comp_edges)}")
    for ce in critical_edges:
        d = d_m8['d_ij'][ce] * 1000
        op = d_m8['op_prod'][ce]
        v = d_m8['vol_max'][ce] * 1000
        a = d_m8['aniso_max'][ce]
        print(f"    Edge {ce}: dist={d:.2f}mm, op_prod={op:.6f}, vol={v:.2f}mm, aniso={a:.1f}")
