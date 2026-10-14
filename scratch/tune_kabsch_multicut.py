import time
import numpy as np
from pathlib import Path
from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.stages.segment_multicut import SegmentMulticutStage
from pipeline.vendored.host.multicut import (
    build_multichannel_graph,
    compute_edge_logits,
    solve_multicut,
    postprocess_min_cluster_size,
)

for run_name in ["hand_test_002-a324fdd2", "controler_test_01-0aaf60a0"]:
    traj_path = Path("runs") / run_name / "trajectories.npz"
    data = np.load(traj_path)
    xyz, traj, op = data["canonical_xyz"], data["traj"], data["opacity"]
    keep = op > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]
    print(f"\n==================== {run_name} (points: {len(xyz_k)}) ====================", flush=True)

    print("\n--- Kabsch Sequential Scaling ---", flush=True)
    for K in [6, 10, 15, 20]:
        t0 = time.time()
        # Direct EM with max_iter=15
        labels, info = segment_by_kabsch(
            xyz_k, traj_k,
            n_clusters=K,
            max_iter=15,
            rng_seed=42,
        )
        dt = time.time() - t0
        uniq = len(np.unique(labels))
        print(f"Kabsch Target K={K:2d} -> Produced {uniq:2d} clusters in {dt:.1f}s", flush=True)

    print("\n--- Multicut Tuning (min_size) ---", flush=True)
    # Load existing multicut or compute with new min_size
    orig_npz = Path("runs") / run_name / "segmentation_multicut.npz"
    if orig_npz.exists():
        raw_labels = np.load(orig_npz)["labels"][keep]
        for min_sz in [100, 300, 600, 1200]:
            merged = postprocess_min_cluster_size(xyz_k, raw_labels.copy(), min_size=min_sz)
            uniq = len(np.unique(merged[merged != -1]))
            print(f"Multicut min_size={min_sz:4d} -> Produced {uniq:3d} clusters", flush=True)
