import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import PUMP01_BASELINE_MERGED_PAIRS_141

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

# Load PLY attributes
ply_files = list((run_dir / 'train_out').glob('**/point_cloud.ply'))
ply_files.sort(key=lambda p: int(p.parent.name.split('_')[-1]) if '_' in p.parent.name else 0)
ply_file = ply_files[-1]

with open(ply_file, 'rb') as f:
    header = b''
    while b'end_header\n' not in header:
        header += f.readline()
    h_len = f.tell()
props = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
         ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4')]
for i in range(45): props.append((f'f_rest_{i}', '<f4'))
props.extend([('opacity', '<f4'), ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
              ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')])
dt = np.dtype(props)
with open(ply_file, 'rb') as f:
    f.seek(h_len)
    data = np.fromfile(f, dtype=dt)

sub_raw_op = data['opacity'][oracle_roi]
sub_op = 1.0 / (1.0 + np.exp(-np.clip(sub_raw_op, -20.0, 20.0)))
sub_scales = np.exp(np.stack([data['scale_0'][oracle_roi], data['scale_1'][oracle_roi], data['scale_2'][oracle_roi]], axis=1))
sub_vol = np.cbrt(sub_scales.prod(axis=1))
sub_aniso = sub_scales.max(axis=1) / np.maximum(sub_scales.min(axis=1), 1e-8)

print('ROI points:', len(sub_xyz))
print('Opacity: median', np.median(sub_op), 'mean', np.mean(sub_op), '<0.1:', (sub_op < 0.1).mean())
print('Scale vol (mm): median', np.median(sub_vol)*1000, 'mean', np.mean(sub_vol)*1000)
print('Anisotropy: median', np.median(sub_aniso), 'mean', np.mean(sub_aniso))

# Now find edges at r=0.003
tree_sub = cKDTree(sub_xyz)
pairs = tree_sub.query_pairs(r=0.003, output_type='ndarray')
print('Total spatial pairs at r=3mm:', len(pairs))

# For each pair, check if it bridges different GT parts
p_i = pairs[:, 0]
p_j = pairs[:, 1]
same_part = (sub_gt[p_i] == sub_gt[p_j])
cross_part = ~same_part

print(f'Same part edges: {same_part.sum()} ({same_part.mean()*100:.1f}%)')
print(f'Cross part edges: {cross_part.sum()} ({cross_part.mean()*100:.1f}%)')

# Opacity product on same vs cross
op_prod = sub_op[p_i] * sub_op[p_j]
print(f'Same part op_prod: median {np.median(op_prod[same_part]):.4f}, mean {np.mean(op_prod[same_part]):.4f}')
print(f'Cross part op_prod: median {np.median(op_prod[cross_part]):.4f}, mean {np.mean(op_prod[cross_part]):.4f}')

# Min opacity on same vs cross
op_min = np.minimum(sub_op[p_i], sub_op[p_j])
print(f'Same part op_min: median {np.median(op_min[same_part]):.4f}, mean {np.mean(op_min[same_part]):.4f}')
print(f'Cross part op_min: median {np.median(op_min[cross_part]):.4f}, mean {np.mean(op_min[cross_part]):.4f}')

# Scale volume product or max
vol_max = np.maximum(sub_vol[p_i], sub_vol[p_j]) * 1000
print(f'Same part max vol (mm): median {np.median(vol_max[same_part]):.4f}, mean {np.mean(vol_max[same_part]):.4f}')
print(f'Cross part max vol (mm): median {np.median(vol_max[cross_part]):.4f}, mean {np.mean(vol_max[cross_part]):.4f}')

# Anisotropy max
aniso_max = np.maximum(sub_aniso[p_i], sub_aniso[p_j])
print(f'Same part max aniso: median {np.median(aniso_max[same_part]):.4f}, mean {np.mean(aniso_max[same_part]):.4f}')
print(f'Cross part max aniso: median {np.median(aniso_max[cross_part]):.4f}, mean {np.mean(aniso_max[cross_part]):.4f}')
