import numpy as np
from pathlib import Path

for k in [2, 3, 5, 10, 20, 30]:
    p = Path(f"data/multipleview/cubes_k{k}_ring/gt_segmentation.npz")
    if p.exists():
        d = np.load(p)
        labs = d["labels"]
        u, counts = np.unique(labs, return_counts=True)
        print(f"k={k}: unique={u.tolist()}, counts={counts.tolist()}")
    else:
        print(f"k={k}: missing {p}")
