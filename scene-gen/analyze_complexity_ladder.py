#!/usr/bin/env python3
"""analyze_complexity_ladder.py — compile results across the complexity ladder (k=2, 3, 5, 10, 20, 30)."""

import json
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

repo_root = Path(__file__).resolve().parent.parent

ladder_runs = [
    ('cubes-cubes_k2_ring', 2, 'cubes_k2_ring'),
    ('cubes-cubes_k3_ring', 3, 'cubes_k3_ring'),
    ('cubes-cubes_k5_ring', 5, 'cubes_k5_ring'),
    ('cubes-cubes_k10_ring', 10, 'cubes_k10_ring'),
    ('cubes-cubes_k20_ring', 20, 'cubes_k20_ring'),
    ('cubes-cubes_k30_ring', 30, 'cubes_k30_ring'),
]

backends = ['rigid', 'rigid2', 'kabsch', 'rigid2_roi', 'mask_lift_oracle', 'mbs']

master_rows = []
auroc_by_k = {}
ari_roi_by_k_backend = {b: {} for b in backends}
ari_global_by_k_backend = {b: {} for b in backends}
iou_by_k_backend = {b: {} for b in backends}
k_pred_by_k_backend = {b: {} for b in backends}

for run_id, k, name in ladder_runs:
    run_dir = repo_root / 'runs' / run_id
    
    # Read separability AUROC from separability.json
    sep_path = run_dir / 'separability.json'
    sep_auroc = None
    if sep_path.is_file():
        try:
            sep_data = json.loads(sep_path.read_text())
            sep_auroc = float(sep_data.get('denoised_z', {}).get('auroc', 0.0))
        except Exception:
            pass
    if sep_auroc is not None:
        auroc_by_k[k] = sep_auroc
        
    for b in backends:
        csv_file = repo_root / 'runs' / f'cubes_seg_{b}_results.csv'
        latest_row = None
        if csv_file.is_file():
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                for r in reader:
                    if len(r) >= 11 and r[0].strip() == run_id:
                        latest_row = [x.strip() for x in r]
        
        if latest_row is not None:
            ari_global_val = float(latest_row[4]) if latest_row[4] else 0.0
            ari_roi_val = float(latest_row[5]) if latest_row[5] else 0.0
            iou_val = float(latest_row[6]) if latest_row[6] else 0.0
            n_gt_val = int(float(latest_row[7])) if latest_row[7] else k + 1
            n_pred_val = int(float(latest_row[8])) if latest_row[8] else 0
            
            if len(latest_row) >= 12:
                n_roi_pts = int(float(latest_row[10])) if latest_row[10] else 0
                wall_s = float(latest_row[11]) if latest_row[11] else 0.0
            else:
                n_roi_pts = int(float(latest_row[9])) if latest_row[9] else 0
                wall_s = float(latest_row[10]) if latest_row[10] else 0.0
        else:
            ari_global_val = 0.0
            ari_roi_val = 0.0
            iou_val = 0.0
            n_gt_val = k + 1
            n_pred_val = 0
            n_roi_pts = 0
            wall_s = 0.0

        row_dict = {
            'scene': name,
            'run_id': run_id,
            'k_gt_parts': k,
            'backend': b,
            'status': 'success',
            'ari_within_roi': ari_roi_val,
            'ari_global': ari_global_val,
            'mean_iou': iou_val,
            'n_gt_clusters': n_gt_val,
            'n_pred_clusters': n_pred_val,
            'separability_auroc': sep_auroc,
            'n_roi_points': n_roi_pts,
            'segment_s': wall_s,
        }
        master_rows.append(row_dict)
        ari_roi_by_k_backend[b][k] = ari_roi_val
        ari_global_by_k_backend[b][k] = ari_global_val
        iou_by_k_backend[b][k] = iou_val
        k_pred_by_k_backend[b][k] = n_pred_val

# Write master benchmark CSV v2
out_master_csv = repo_root / 'runs' / 'cubes_complexity_ladder_v2.csv'
with open(out_master_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
    w.writeheader()
    w.writerows(master_rows)
print(f'Wrote master benchmark CSV v2 -> {out_master_csv}')

# Add pump01 baseline (k=107)
auroc_by_k[107] = 0.5841

print('\n=== SUMMARY OF COMPLEXITY LADDER BENCHMARK (V2) ===')
header_str = f"{'K':<4} | {'AUROC':<8} | {'Rigid ARI_roi':<14} | {'Rigid2 ARI_roi':<15} | {'Kabsch ARI_roi':<15} | {'Oracle ARI_roi':<15} | {'MBS ARI_roi':<12}"
print(header_str)
print('-' * len(header_str))
for k in [2, 3, 5, 10, 20, 30]:
    auc = auroc_by_k.get(k, 0.0)
    r_ari = ari_roi_by_k_backend['rigid'].get(k, 0.0)
    r2_ari = ari_roi_by_k_backend['rigid2'].get(k, 0.0)
    k_ari = ari_roi_by_k_backend['kabsch'].get(k, 0.0)
    o_ari = ari_roi_by_k_backend['mask_lift_oracle'].get(k, 0.0)
    m_ari = ari_roi_by_k_backend['mbs'].get(k, 0.0)
    print(f"{k:<4} | {auc:<8.4f} | {r_ari:<14.4f} | {r2_ari:<14.4f} | {k_ari:<14.4f} | {o_ari:<14.4f} | {m_ari:<12.4f}")

# Load partition swap results
swap_csv = repo_root / 'runs' / 'cubes_partition_swap_results.csv'
swap_kmeans_ari = {}
swap_spatial_ari = {}
if swap_csv.is_file():
    with open(swap_csv, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            k = int(r['k_gt'])
            m = r['method']
            if 'swap_a' in m:
                swap_kmeans_ari[k] = float(r['ari_within_roi'])
            elif 'swap_b' in m:
                swap_spatial_ari[k] = float(r['ari_within_roi'])

# -------------------------------------------------------------
# Generate Publication-Quality Figure (3 panels)
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

ks = [2, 3, 5, 10, 20, 30]

# 1. Measured Separability AUROC Curve
auc_vals = [auroc_by_k[k] for k in ks]
ax1.plot(ks, auc_vals, marker='o', linewidth=2.5, color='#1f77b4', label='Deformation Separability AUROC', markersize=8)
ax1.axhline(0.80, color='red', linestyle='--', linewidth=1.5, label='Viability Threshold (0.80)')
ax1.scatter([107], [auroc_by_k[107]], color='darkred', marker='X', s=120, label='Pump01 (K=107: 0.5841)', zorder=5)

ax1.set_title('Trajectory Separability vs Part Count (K)\n(Measured Knee Curve)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Part Count (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Mann-Whitney Separability AUROC', fontsize=11, fontweight='bold')
ax1.set_ylim(0.50, 1.02)
ax1.set_xticks([2, 3, 5, 10, 20, 30])
ax1.legend(frameon=True, fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.6)

for k, val in zip(ks, auc_vals):
    ax1.annotate(f'{val:.4f}', (k, val), textcoords='offset points', xytext=(0, 8), ha='center', fontweight='bold', fontsize=8.5)
ax1.annotate('Pump01 (107)\n0.5841', (107, 0.5841), textcoords='offset points', xytext=(-35, -20), ha='center', fontweight='bold', color='darkred', fontsize=8.5)

# 2. Baseline Graph Partitions ARI within Dynamic ROI
colors = {
    'rigid': '#7f7f7f',
    'rigid2': '#2ca02c',
    'kabsch': '#ff7f0e',
    'rigid2_roi': '#17becf',
    'mask_lift_oracle': '#9467bd',
    'mbs': '#d62728',
}
markers = {
    'rigid': 's',
    'rigid2': '^',
    'kabsch': 'D',
    'rigid2_roi': 'v',
    'mask_lift_oracle': '*',
    'mbs': 'x',
}

for b in backends:
    b_vals = [ari_roi_by_k_backend[b].get(k, 0.0) for k in ks]
    ax2.plot(ks, b_vals, marker=markers[b], linewidth=2.0, color=colors[b], label=f'{b}', markersize=6.5)

ax2.set_title('Baseline Graph Partition ARI\n(Within Dynamic ROI)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Part Count (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Adjusted Rand Index (within ROI)', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.02, 0.25)
ax2.set_xticks([2, 3, 5, 10, 20, 30])
ax2.legend(frameon=True, fontsize=8.5, ncol=2)
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Partition Swaps with Rigid-Body Prior
r2_vals = [ari_roi_by_k_backend['rigid2'].get(k, 0.0) for k in ks]
km_vals = [swap_kmeans_ari.get(k, 0.0) for k in ks]
sp_vals = [swap_spatial_ari.get(k, 0.0) for k in ks]

ax3.plot(ks, r2_vals, marker='^', linewidth=2.0, color='#2ca02c', label='Baseline Rigid2 Graph', markersize=7)
ax3.plot(ks, km_vals, marker='o', linewidth=2.5, color='#9467bd', label='Swap (a): FFT K-Means (BIC)', markersize=7)
ax3.plot(ks, sp_vals, marker='s', linewidth=2.5, color='#e377c2', label='Swap (b): Spatial Connected Comp', markersize=7)
ax3.axhline(0.60, color='red', linestyle='--', linewidth=1.5, label='Gate Threshold (0.60)')

ax3.set_title('Rigid-Body-Prior Partition Swaps\n(Within Dynamic ROI)', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Part Count (K)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Adjusted Rand Index (within ROI)', fontsize=11, fontweight='bold')
ax3.set_ylim(-0.05, 1.05)
ax3.set_xticks([2, 3, 5, 10, 20, 30])
ax3.legend(frameon=True, fontsize=9)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plot_path = repo_root / 'runs' / 'cubes_complexity_knee_plot.png'
plt.savefig(plot_path, dpi=300)
plt.close()
print(f'Wrote complexity knee plot -> {plot_path}')

# Also copy to artifacts directory
art_dir = Path('C:/Users/barte/.gemini/antigravity-ide/brain/fde51817-043e-4107-8200-6f242c895a4f')
if art_dir.is_dir():
    shutil.copy2(plot_path, art_dir / 'cubes_complexity_knee_plot.png')
    print(f'Copied plot to artifact directory -> {art_dir / "cubes_complexity_knee_plot.png"}')
