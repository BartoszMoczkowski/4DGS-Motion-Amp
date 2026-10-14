import time
import numpy as np
from pathlib import Path
from pipeline.vendored.host.rigidity_graph import segment_by_rigidity
from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

for run_name in ["hand_test_002-a324fdd2", "controler_test_01-0aaf60a0"]:
    traj_path = Path("runs") / run_name / "trajectories.npz"
    data = np.load(traj_path)
    xyz, traj, op = data["canonical_xyz"], data["traj"], data["opacity"]
    keep = op > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]
    print(f"\n==================== {run_name} (points: {len(xyz_k)}) ====================", flush=True)

    print("\n--- Tuning Rigid (thr_mult x min_size) ---", flush=True)
    for mult in [1.0, 1.5, 2.0, 3.0]:
        for sz in [100, 500, 1500, 3000]:
            t0 = time.time()
            labels, info = segment_by_rigidity(
                xyz_k, traj_k, k=16, threshold_mult=mult, min_size=sz
            )
            dt = time.time() - t0
            n_c = len(np.unique(labels))
            print(f"Rigid thr_mult={mult:3.1f}, min_sz={sz:4d} -> {n_c:4d} clusters ({dt:.1f}s)", flush=True)

    print("\n--- Tuning Rigid2 (z_thresh x min_size) ---", flush=True)
    for z in [3.0, 5.0, 8.0, 12.0]:
        for sz in [100, 500, 1500, 3000]:
            t0 = time.time()
            labels, info = segment_by_rigidity2(
                xyz_k, traj_k, k=16, z_thresh=z, min_size=sz, partition="components"
            )
            dt = time.time() - t0
            n_c = len(np.unique(labels))
            print(f"Rigid2 z_thresh={z:4.1f}, min_sz={sz:4d} -> {n_c:4d} clusters ({dt:.1f}s)", flush=True)
