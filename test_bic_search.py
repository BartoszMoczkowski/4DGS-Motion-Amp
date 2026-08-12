import sys
sys.path.insert(0, 'orchestrator')
import time
import numpy as np
from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.vendored.host.metrics import adjusted_rand_index as ari

data = np.load('runs/grid-A20mm_M2/trajectories.npz')
xyz, traj = data['canonical_xyz'], data['traj']
gt = np.load('runs/grid-A20mm_M2/convert_out/data/multipleview/capture_pump_A20mm_M2/gt_segmentation.npz')['labels']
print(f"Model: {len(xyz)} points, GT parts: {gt.max()+1}")

# Test BIC search
print("\nBIC search...")
t0 = time.time()
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=0, k_range=[20, 150],
    init='fft', fps_subsample=5000, max_iter=15, rng_seed=0,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"best_k={info['best_k']}, n_final={info['n_clusters_final']}")
print(f"BICs: {info.get('bics')}")
score = ari(gt, labels)
print(f"ARI={score:.4f}")
