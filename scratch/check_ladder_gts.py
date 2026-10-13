from pathlib import Path
import numpy as np

ladder_runs = [
    ('cubes-cubes_k2_ring', 2, 'cubes_k2_ring'),
    ('cubes-cubes_k3_ring', 3, 'cubes_k3_ring'),
    ('cubes-cubes_k5_ring', 5, 'cubes_k5_ring'),
    ('cubes-cubes_k10_ring', 10, 'cubes_k10_ring'),
    ('cubes-cubes_k20_ring', 20, 'cubes_k20_ring'),
    ('cubes-cubes_k30_ring', 30, 'cubes_k30_ring'),
]

for run_id, k, name in ladder_runs:
    run_dir = Path("runs") / run_id
    gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
    d = np.load(gt_file)
    u, counts = np.unique(d["labels"], return_counts=True)
    print(f"{run_id}: unique={u.tolist()}, counts={counts.tolist()}")
