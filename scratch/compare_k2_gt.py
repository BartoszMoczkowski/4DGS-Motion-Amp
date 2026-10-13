from pathlib import Path
import numpy as np
import json

p_run = Path("runs/cubes-cubes_k2_ring/convert_out/data/multipleview/cubes_k2_ring/gt_segmentation.npz")
p_data = Path("data/multipleview/cubes_k2_ring/gt_segmentation.npz")

d_run = np.load(p_run)
d_data = np.load(p_data)

print(f"Run GT: {len(d_run['points'])} points, unique={np.unique(d_run['labels'], return_counts=True)}")
print(f"Data GT: {len(d_data['points'])} points, unique={np.unique(d_data['labels'], return_counts=True)}")
