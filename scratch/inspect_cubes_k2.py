import numpy as np
import json
from pathlib import Path

p = Path("data/multipleview/cubes_k2_ring/gt_segmentation.npz")
d = np.load(p)
pts = d["points"]
labs = d["labels"]
print(f"Total points: {len(pts)}")
for l in np.unique(labs):
    sub = pts[labs == l]
    print(f"Label {l}: count={len(sub)}, centroid={sub.mean(axis=0)}, min={sub.min(axis=0)}, max={sub.max(axis=0)}")

p2 = Path("data/multipleview/cubes_k2_ring/gt_motion_classes.npy")
if p2.exists():
    mot = np.load(p2)
    print(f"Motion classes unique: {np.unique(mot)}")
