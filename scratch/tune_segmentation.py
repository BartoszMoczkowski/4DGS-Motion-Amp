"""
Systematic parameter tuning for motion segmentation algorithms:
1. Kabsch sequential scaling: K = 6, 10, 16, 24
2. Rigid tuning: threshold_mult, k, min_size to reduce clusters from ~1400 to 10-50
3. Rigid2 tuning: z_thresh, min_size, spectral partition
4. Multicut tuning: min_size, radius_max
"""

import sys
import numpy as np
from pathlib import Path

from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.vendored.host.rigidity_graph import segment_by_rigidity
from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

def test_tuning(run_name: str):
    traj_path = Path("runs") / run_name / "trajectories.npz"
    print(f"\n================ Loading {run_name} ================", flush=True)
    data = np.load(traj_path)
    xyz, traj, op = data["canonical_xyz"], data["traj"], data["opacity"]
    keep = op > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]
    print(f"Total points: {len(xyz)}, Kept (opacity > 0.1): {len(xyz_k)}", flush=True)

    # 1. KABSCH SEQUENTIAL SCALING
    print("\n--- Kabsch Sequential Scaling ---", flush=True)
    for K in [5, 8, 12, 16, 24]:
        labels, info = segment_by_kabsch(
            xyz_k, traj_k,
            n_clusters=K,
            fps_subsample=10000,
            max_iter=15,
            rng_seed=42,
        )
        uniq, counts = np.unique(labels, return_counts=True)
        print(f"Kabsch Target K={K:2d} -> Produced {len(uniq):2d} clusters. Sizes: {sorted(counts, reverse=True)[:6]}...", flush=True)

    # 2. RIGID TUNING
    print("\n--- Rigid Parameter Tuning ---", flush=True)
    for thresh_mult in [1.5, 2.0, 3.0]:
        for min_sz in [200, 500, 1000]:
            labels, info = segment_by_rigidity(
                xyz_k, traj_k,
                k=16,
                threshold_mult=thresh_mult,
                min_size=min_sz,
            )
            uniq, counts = np.unique(labels, return_counts=True)
            print(f"Rigid thr_mult={thresh_mult:.1f}, min_sz={min_sz:4d} -> {len(uniq):3d} clusters. Top sizes: {sorted(counts, reverse=True)[:5]}", flush=True)

    # 3. RIGID2 TUNING
    print("\n--- Rigid2 Parameter Tuning ---", flush=True)
    for z_th in [4.0, 6.0, 8.0]:
        for min_sz in [200, 500, 1000]:
            labels, info = segment_by_rigidity2(
                xyz_k, traj_k,
                z_thresh=z_th,
                min_size=min_sz,
                partition="components",
            )
            uniq, counts = np.unique(labels, return_counts=True)
            print(f"Rigid2 z_thresh={z_th:.1f}, min_sz={min_sz:4d} -> {len(uniq):3d} clusters. Top sizes: {sorted(counts, reverse=True)[:5]}", flush=True)

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "hand_test_002-a324fdd2"
    test_tuning(run_id)
