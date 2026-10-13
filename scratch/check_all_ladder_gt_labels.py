import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent

for k in [2, 3, 5, 10, 20, 30]:
    run_id = f"cubes-cubes_k{k}_ring"
    p_run = list((repo_root / "runs" / run_id).glob("**/gt_segmentation.npz"))[0]
    p_data = repo_root / "data" / "multipleview" / f"cubes_k{k}_ring" / "gt_segmentation.npz"
    
    d_run = np.load(p_run)
    d_data = np.load(p_data)
    
    print(f"\n--- K={k} ---")
    print(f"Run GT ({p_run.name}): {len(d_run['points'])} pts, unique labels: {np.unique(d_run['labels'], return_counts=True)}")
    print(f"Data GT ({p_data.name}): {len(d_data['points'])} pts, unique labels: {np.unique(d_data['labels'], return_counts=True)}")
