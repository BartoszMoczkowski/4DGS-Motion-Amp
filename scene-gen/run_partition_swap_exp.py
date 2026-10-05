#!/usr/bin/env python3
"""run_partition_swap_exp.py — Benchmark baseline graph partitions vs rigid-body-prior partition swaps."""

import sys
from pathlib import Path
import csv
import numpy as np
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / 'scene-gen'))

from pipeline.vendored.host.metrics import adjusted_rand_index, best_iou_matching
from pipeline.vendored.host.trajectory_denoise import trajectory_energy
from partition_swap import fit_fft_kmeans_bic, fit_spatial_connected_components

runs_to_eval = [
    ('cubes-cubes_k2_ring', 2, 'cubes_k2_ring'),
    ('cubes-cubes_k3_ring', 3, 'cubes_k3_ring'),
    ('cubes-cubes_k5_ring', 5, 'cubes_k5_ring'),
    ('cubes-cubes_k10_ring', 10, 'cubes_k10_ring'),
    ('cubes-cubes_k20_ring', 20, 'cubes_k20_ring'),
    ('cubes-cubes_k30_ring', 30, 'cubes_k30_ring'),
]


def run_experiment():
    results = []

    for run_id, k_gt, scene_name in runs_to_eval:
        run_dir = repo_root / 'runs' / run_id
        traj_file = run_dir / 'trajectories.npz'
        gt_file = repo_root / 'data' / 'multipleview' / scene_name / 'gt_segmentation.npz'

        if not traj_file.is_file() or not gt_file.is_file():
            print(f"Skipping {run_id}: trajectories or GT file missing.")
            continue

        print(f"\n=======================================================")
        print(f"Evaluating Partition Swaps on {run_id} (K_gt={k_gt})")
        print(f"=======================================================")

        d = np.load(traj_file)
        traj = d['traj']  # (N, T, 3)
        xyz = d['canonical_xyz']  # (N, 3)

        gt_d = np.load(gt_file)
        gt_points = gt_d['points']
        gt_labels = gt_d['labels']

        # NN label transfer
        tree = cKDTree(gt_points)
        _, nn_idx = tree.query(xyz, k=1)
        gt_on_pred = gt_labels[nn_idx]

        # Dynamic ROI
        gt_roi = (gt_on_pred > 0)
        n_roi = int(gt_roi.sum())
        print(f"Total Gaussians: {len(xyz)}, ROI Gaussians: {n_roi}")

        # -------------------------------------------------------------
        # 1. Baseline Graph Partition (rigid2 inside ROI / global)
        # -------------------------------------------------------------
        # Check if rigid2 segmentation exists
        r2_seg_path = run_dir / 'segmentation_rigid2.npz'
        if not r2_seg_path.is_file():
            r2_seg_path = run_dir / 'segmentation.npz'
        
        if r2_seg_path.is_file():
            r2_data = np.load(r2_seg_path)
            r2_labels = r2_data.get('labels', r2_data.get('pred_labels', None))
            if r2_labels is not None:
                ari_r2_roi = adjusted_rand_index(gt_on_pred[gt_roi], r2_labels[gt_roi])
                ari_r2_glob = adjusted_rand_index(gt_on_pred, r2_labels)
                iou_r2, _ = best_iou_matching(gt_on_pred, r2_labels)
                k_pred_r2 = len(np.unique(r2_labels))
                results.append({
                    'scene': scene_name,
                    'k_gt': k_gt,
                    'method': 'baseline_rigid2_graph',
                    'roi_type': 'oracle_dynamic_roi',
                    'k_pred': k_pred_r2,
                    'ari_within_roi': ari_r2_roi,
                    'ari_global': ari_r2_glob,
                    'mean_iou': iou_r2,
                })
                print(f"Baseline Rigid2: within-ROI ARI={ari_r2_roi:.4f}, global ARI={ari_r2_glob:.4f}, K_pred={k_pred_r2}")

        # -------------------------------------------------------------
        # 2. Partition Swap (a): FFT-fingerprint K-Means with BIC
        # -------------------------------------------------------------
        k_max_search = min(35, max(10, k_gt * 2 + 2))
        labels_bic, k_bic, best_bic = fit_fft_kmeans_bic(
            traj[gt_roi], xyz[gt_roi], k_min=2, k_max=k_max_search
        )
        ari_bic_roi = adjusted_rand_index(gt_on_pred[gt_roi], labels_bic)
        
        # Build global prediction (0 for background, 1..k for clusters)
        full_pred_bic = np.zeros(len(xyz), dtype=np.int64)
        full_pred_bic[gt_roi] = labels_bic + 1
        ari_bic_glob = adjusted_rand_index(gt_on_pred, full_pred_bic)
        iou_bic, _ = best_iou_matching(gt_on_pred, full_pred_bic)

        results.append({
            'scene': scene_name,
            'k_gt': k_gt,
            'method': 'swap_a_fft_kmeans_bic',
            'roi_type': 'oracle_dynamic_roi',
            'k_pred': k_bic + 1,
            'ari_within_roi': ari_bic_roi,
            'ari_global': ari_bic_glob,
            'mean_iou': iou_bic,
        })
        print(f"Swap (a) FFT K-Means (BIC): within-ROI ARI={ari_bic_roi:.4f}, global ARI={ari_bic_glob:.4f}, K_pred={k_bic}")

        # -------------------------------------------------------------
        # 3. Partition Swap (b): Calibrated Spatial Connected Components
        # -------------------------------------------------------------
        best_cc_ari = -1.0
        best_cc_res = None
        for rad in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            labels_cc, k_cc = fit_spatial_connected_components(xyz[gt_roi], radius=rad)
            ari_cc_roi = adjusted_rand_index(gt_on_pred[gt_roi], labels_cc)
            full_pred_cc = np.zeros(len(xyz), dtype=np.int64)
            full_pred_cc[gt_roi] = labels_cc + 1
            ari_cc_glob = adjusted_rand_index(gt_on_pred, full_pred_cc)
            iou_cc, _ = best_iou_matching(gt_on_pred, full_pred_cc)
            
            if ari_cc_roi > best_cc_ari:
                best_cc_ari = ari_cc_roi
                best_cc_res = {
                    'scene': scene_name,
                    'k_gt': k_gt,
                    'method': f'swap_b_spatial_cc_r{int(rad*100):02d}cm',
                    'roi_type': 'oracle_dynamic_roi',
                    'k_pred': k_cc + 1,
                    'ari_within_roi': ari_cc_roi,
                    'ari_global': ari_cc_glob,
                    'mean_iou': iou_cc,
                }
            print(f"   Spatial CC (r={rad:.2f}m): within-ROI ARI={ari_cc_roi:.4f}, K_pred={k_cc}")

        if best_cc_res is not None:
            results.append(best_cc_res)

    # Write results CSV
    out_csv = repo_root / 'runs' / 'cubes_partition_swap_results.csv'
    if results:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote partition swap results -> {out_csv}")


if __name__ == '__main__':
    run_experiment()
