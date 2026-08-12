import sys
sys.path.insert(0, 'orchestrator')
import time
import numpy as np
from pipeline.vendored.host.kabsch_em import segment_by_kabsch

data = np.load('runs/grid-A20mm_M2/trajectories.npz')
xyz, traj = data['canonical_xyz'], data['traj']
print(f"Model: {len(xyz)} points, {traj.shape[1]} frames")

t0 = time.time()
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=107, init='fft',
    fps_subsample=5000, max_iter=15, rng_seed=0,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"n_clusters_final={info['n_clusters_final']}, converged={info.get('converged')}")
