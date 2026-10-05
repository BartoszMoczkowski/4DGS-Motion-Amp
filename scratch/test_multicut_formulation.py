"""scratch/test_multicut_formulation.py — Test affinity graph design for multicut."""

from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scene-gen"))

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index
from run_pump01_spatial_cc import read_point_cloud_ply, fast_best_iou_matching

def main():
    run_dir = Path("runs/grid-A20mm_M2")
    traj_data = np.load(run_dir / "trajectories.npz")
    xyz = traj_data["canonical_xyz"]
    traj = traj_data["traj"]
    ply = read_point_cloud_ply(run_dir / "train_out/point_cloud/iteration_15000/point_cloud.ply")
    rgb = ply["rgb"]

    gt_files = list((run_dir / "convert_out/data/multipleview").glob("*/gt_segmentation.npz"))
    gt = np.load(gt_files[0])
    tree = cKDTree(gt["points"])
    _, nn = tree.query(xyz, k=1)
    gt_labels = gt["labels"][nn]
    roi_mask = gt_labels > 0

    N = len(xyz)
    print(f"Loaded {N} points, {roi_mask.sum()} in dynamic ROI")

    # Extract phasors at f0=10
    T = traj.shape[1]
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * 10.0 * t / T)
    traj_c = traj - traj.mean(axis=1, keepdims=True)
    phasors = np.tensordot(traj_c, basis, axes=([1], [0]))
    phasors_6d = np.concatenate([phasors.real, phasors.imag], axis=1).astype(np.float32)
    p_norms = np.linalg.norm(phasors_6d, axis=1, keepdims=True)
    norm_phasors = np.zeros_like(phasors_6d)
    p_valid = p_norms[:, 0] > 1e-9
    norm_phasors[p_valid] = phasors_6d[p_valid] / p_norms[p_valid]

    # Build radius pairs graph
    t_tree = cKDTree(xyz)
    r_search = 0.006 # 6mm
    pairs = t_tree.query_pairs(r=r_search, output_type="ndarray")
    print(f"Found {len(pairs)} pairs within r={r_search*1000:.1f}mm")

    u = pairs[:, 0]
    v = pairs[:, 1]
    d_xyz = np.linalg.norm(xyz[u] - xyz[v], axis=1)
    d_rgb = np.linalg.norm(rgb[u] - rgb[v], axis=1)
    r_phasor = np.sum(norm_phasors[u] * norm_phasors[v], axis=1)
    disc_phasor = np.clip(1.0 - r_phasor, 0.0, 2.0)

    # Sweep r_cut, c_cut, p_cut
    for r_cut in [0.003, 0.0035, 0.004, 0.0045, 0.005]:
        for c_cut in [0.12, 0.15, 0.18, 0.20]:
            for p_weight in [0.0, 0.5, 1.0, 1.5]:
                # Multi-channel logit:
                # Base spatial: +3.0 at d=0, 0.0 at d=r_cut, negative beyond
                # Color penalty: -5.0 * (d_rgb / c_cut)
                # Motion penalty: -p_weight * disc_phasor
                w_e = 3.0 * (1.0 - (d_xyz / r_cut)) - 4.0 * (d_rgb / c_cut) - p_weight * disc_phasor

                # Connected components on positive edges (multicut partition)
                pos_mask = w_e > 0.0
                if not pos_mask.any():
                    continue
                pos_pairs = pairs[pos_mask]
                adj = csr_matrix((np.ones(len(pos_pairs), dtype=bool), (pos_pairs[:, 0], pos_pairs[:, 1])), shape=(N, N))
                n_comp, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)

                # Min cluster size 15
                unique, counts = np.unique(raw_labels, return_counts=True)
                valid_components = unique[counts >= 15]
                if len(valid_components) == 0:
                    continue

                comp_map = {comp: idx for idx, comp in enumerate(valid_components)}
                labels = np.full(N, -1, dtype=np.int64)
                for comp, new_idx in comp_map.items():
                    labels[raw_labels == comp] = new_idx

                unassigned = np.where(labels == -1)[0]
                if len(unassigned) > 0:
                    assigned = np.where(labels >= 0)[0]
                    sub_tree = cKDTree(xyz[assigned])
                    _, nearest_idx = sub_tree.query(xyz[unassigned], k=1)
                    labels[unassigned] = labels[assigned[nearest_idx]]

                ari_roi = float(adjusted_rand_index(gt_labels[roi_mask], labels[roi_mask]))
                ari_glob = float(adjusted_rand_index(gt_labels, labels))
                k_pred = len(np.unique(labels))

                if ari_roi > 0.38:
                    iou, _ = fast_best_iou_matching(gt_labels, labels)
                    print(f"r_cut={r_cut*1000:.1f}mm c_cut={c_cut:.2f} p_w={p_weight:.1f} -> ARI_roi={ari_roi:.4f} ARI_glob={ari_glob:.4f} IoU={iou:.4f} K_pred={k_pred}")

if __name__ == "__main__":
    main()
