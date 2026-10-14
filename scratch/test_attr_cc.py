import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index, compute_pairs_split_rate, PUMP01_BASELINE_MERGED_PAIRS_141

run_dir = Path('runs/grid-A20mm_M2')
traj = np.load(run_dir / 'trajectories.npz')
xyz = traj['canonical_xyz']
gt_file = list(run_dir.glob('**/gt_segmentation.npz'))[0]
gt_data = np.load(gt_file)
tree = cKDTree(gt_data['points'])
_, nn = tree.query(xyz, k=1)
gt_on_pred = gt_data['labels'][nn]
oracle_roi = (gt_on_pred > 0)

sub_xyz = xyz[oracle_roi]
sub_gt = gt_on_pred[oracle_roi]
n = len(sub_xyz)

tree_sub = cKDTree(sub_xyz)
pairs = tree_sub.query_pairs(r=0.003, output_type='ndarray')
adj = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
adj = adj + adj.T
n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
unique, counts = np.unique(raw_labels, return_counts=True)
valid_comp = unique[counts >= 15]

comp_map = {comp: idx for idx, comp in enumerate(valid_comp)}
labels_dilated = np.full(n, -1, dtype=np.int64)
for comp, idx in comp_map.items():
    labels_dilated[raw_labels == comp] = idx
unassigned = np.where(labels_dilated == -1)[0]
assigned = np.where(labels_dilated >= 0)[0]
sub_tree = cKDTree(sub_xyz[assigned])
_, nearest = sub_tree.query(sub_xyz[unassigned], k=1)
labels_dilated[unassigned] = labels_dilated[assigned[nearest]]

ari = adjusted_rand_index(sub_gt, labels_dilated)
ps_res = compute_pairs_split_rate(sub_gt, labels_dilated)

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]
cat_b_res = compute_pairs_split_rate(sub_gt, labels_dilated, merged_pairs=cat_b_pairs)

print(f'Baseline S6 r=0.3cm on A20mm_M2:')
print(f'ARI within ROI: {ari:.6f}')
print(f"Pairs-split rate (141 pairs): {ps_res['pairs_split_rate']:.4f} ({ps_res['n_pairs_split']}/{ps_res['n_pairs_merged_base']})")
print(f"Cat B split rate (102 pairs): {cat_b_res['pairs_split_rate']:.4f} ({cat_b_res['n_pairs_split']}/{cat_b_res['n_pairs_merged_base']})")
