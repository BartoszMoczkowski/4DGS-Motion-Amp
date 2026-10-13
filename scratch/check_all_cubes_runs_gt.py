from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent

for p in sorted(repo_root.glob("runs/cubes*")):
    if p.is_dir():
        gt_files = list(p.glob("**/gt_segmentation.npz"))
        if gt_files:
            d = np.load(gt_files[0])
            u, c = np.unique(d["labels"], return_counts=True)
            print(f"{p.name}: GT points={len(d['points'])}, unique labels={u.tolist()}, counts={c.tolist()}")
