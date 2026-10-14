import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

run_dir = Path('runs/grid-A40mm_M8')
traj = np.load(run_dir / 'trajectories.npz')
xyz = traj['canonical_xyz']
gt = np.load(list(run_dir.glob('**/gt_segmentation.npz'))[0])
_, nn = cKDTree(gt['points']).query(xyz, k=1)
roi = (gt['labels'][nn] > 0)
sub_xyz = xyz[roi]

tree_sub = cKDTree(sub_xyz)
pairs = tree_sub.query_pairs(r=0.003, output_type='ndarray')
adj = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(sub_xyz), len(sub_xyz)))
adj = adj + adj.T
n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
unique, counts = np.unique(raw_labels, return_counts=True)
valid_comp = unique[counts >= 15]
print('M8 components >= 15 points:', sorted(counts[counts >= 15], reverse=True))
print('Total points in valid components:', counts[counts >= 15].sum())
