import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index

run_dir = Path("runs/cubes-cubes_k2_ring")
traj = np.load(run_dir / "trajectories.npz")
xyz = traj["canonical_xyz"]
gt_file = run_dir / "convert_out/data/multipleview/cubes_k2_ring/gt_segmentation.npz"
gt = np.load(gt_file)
_, nn = cKDTree(gt["points"]).query(xyz, k=1)
gt_on_pred = gt["labels"][nn]

ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
ply_file = ply_files[-1]

with open(ply_file, "rb") as f:
    header = b""
    while b"end_header\n" not in header: header += f.readline()
    h_len = f.tell()
props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
         ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
for i in range(45): props.append((f"f_rest_{i}", "<f4"))
props.extend([("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
              ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")])
dt = np.dtype(props)
with open(ply_file, "rb") as f:
    f.seek(h_len)
    data = np.fromfile(f, dtype=dt)

raw_op = data["opacity"]
op = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1))
vol = np.cbrt(scales.prod(axis=1))

print("Cubes points:", len(xyz))
print("Cubes opacity: median", np.median(op), "mean", np.mean(op), "min", op.min())

# On cubes, what is the separation between the 2 cubes?
# Cube 0 vs Cube 1
c0_pts = xyz[gt_on_pred == 0]
c1_pts = xyz[gt_on_pred == 1]
t0 = cKDTree(c0_pts)
t1 = cKDTree(c1_pts)
dists, _ = t0.query(c1_pts, k=1)
print(f"Cubes inter-cube min distance: {dists.min()*100:.2f} cm ({dists.min()*1000:.2f} mm)")

# At r=0.3cm, are there ANY edges between Cube 0 and Cube 1?
tree_all = cKDTree(xyz)
pairs = tree_all.query_pairs(r=0.003, output_type="ndarray")
print(f"Cubes pairs at r=0.3cm: {len(pairs)}")
cross_cube = (gt_on_pred[pairs[:, 0]] != gt_on_pred[pairs[:, 1]])
print(f"Cross-cube edges at r=0.3cm: {cross_cube.sum()}")
