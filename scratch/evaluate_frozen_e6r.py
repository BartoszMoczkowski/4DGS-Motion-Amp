import sys
import time
from pathlib import Path
import numpy as np
import scipy.stats
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "orchestrator"))
sys.path.insert(0, str(repo_root / "scene-gen"))
sys.path.insert(0, str(repo_root / "scratch"))

from pipeline.vendored.host.multicut import build_multichannel_graph
from run_pump01_spatial_cc import read_point_cloud_ply
from run_grid_multicut_calibrated import fit_logistic_weights, compute_feature_auroc

def compute_exact_auroc(score: np.ndarray, y: np.ndarray, higher_is_same: bool = True) -> float:
    """Exact Mann-Whitney U AUROC without subsampling."""
    if not higher_is_same:
        score = -score
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = scipy.stats.rankdata(score)
    u = np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))

models = [
    ("grid-A20mm_M2", "A20mm_M2", True),
    ("grid-A20mm_M4", "A20mm_M4", False),
    ("grid-A40mm_M8", "A40mm_M8", False),
]

# 1. Fit on M2
print("=== [1] Fitting on grid-A20mm_M2 ===", flush=True)
m2_dir = repo_root / "runs" / "grid-A20mm_M2"
m2_traj_data = np.load(m2_dir / "trajectories.npz")
m2_xyz = m2_traj_data["canonical_xyz"]
m2_traj = m2_traj_data["traj"]
m2_ply = read_point_cloud_ply(sorted((m2_dir / "train_out/point_cloud").glob("*/*.ply"))[-1])
m2_rgb = m2_ply["rgb"]

m2_gt_file = list(m2_dir.glob("**/gt_segmentation.npz"))[0]
m2_gt_data = np.load(m2_gt_file)
m2_tree = cKDTree(m2_gt_data["points"])
_, m2_nn = m2_tree.query(m2_xyz, k=1)
m2_gt_labels = m2_gt_data["labels"][m2_nn]

m2_edges, m2_feats, m2_info = build_multichannel_graph(
    m2_xyz, traj=m2_traj, rgb=m2_rgb, k=12, radius_max=0.03, f0=10.0, include_lifted=False
)
u_m2 = m2_edges[:, 0]
v_m2 = m2_edges[:, 1]
y_m2 = (m2_gt_labels[u_m2] == m2_gt_labels[v_m2]).astype(np.float64)

# Fit frozen weights
feature_cols = [0, 1, 3] # d_xyz, d_rgb, disc_phasor
w_full, auroc_fit_d = fit_logistic_weights(
    m2_feats[:, feature_cols], y_m2, ["beta_xyz", "beta_rgb", "beta_phasor"]
)
print(f"Learned frozen weights: {w_full}")
print(f"M2 Fused AUROC (E6R subsampled): {auroc_fit_d:.4f}")

# Sanity check: log-odds
p_same_m2 = y_m2.mean()
log_odds_m2 = np.log(p_same_m2 / (1.0 - p_same_m2))
print(f"M2 class balance: same={y_m2.sum():.0f}/{len(y_m2)} ({p_same_m2*100:.2f}%)")
print(f"M2 marginal log-odds prior: {log_odds_m2:+.3f}")
print(f"Learned beta0 (intercept): {w_full['intercept']:+.3f}")

intercept = w_full["intercept"]
betas = [w_full["beta_xyz"], w_full["beta_rgb"], w_full["beta_phasor"]]

results = []

# Now evaluate on all models
for dir_name, model_name, is_train in models:
    print(f"\n=== Evaluating on {model_name} ({dir_name}) ===", flush=True)
    run_dir = repo_root / "runs" / dir_name
    traj_data = np.load(run_dir / "trajectories.npz")
    xyz = traj_data["canonical_xyz"]
    traj = traj_data["traj"]
    ply_file = sorted((run_dir / "train_out/point_cloud").glob("*/*.ply"))[-1]
    ply = read_point_cloud_ply(ply_file)
    rgb = ply["rgb"]

    gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
    gt_data = np.load(gt_file)
    tree = cKDTree(gt_data["points"])
    _, nn = tree.query(xyz, k=1)
    gt_labels = gt_data["labels"][nn]

    edges, feats, info = build_multichannel_graph(
        xyz, traj=traj, rgb=rgb, k=12, radius_max=0.03, f0=10.0, include_lifted=False
    )
    u = edges[:, 0]
    v = edges[:, 1]
    y = (gt_labels[u] == gt_labels[v]).astype(np.float64)

    # Class balance
    p_same = y.mean()
    log_odds = np.log(p_same / (1.0 - p_same))

    # Per-channel AUROCs (subsampled as in E6R and exact)
    auroc_xyz_sub = compute_feature_auroc(feats[:, 0], y.astype(bool), higher_is_same=False)
    auroc_rgb_sub = compute_feature_auroc(feats[:, 1], y.astype(bool), higher_is_same=False)
    auroc_phasor_sub = compute_feature_auroc(feats[:, 3], y.astype(bool), higher_is_same=False)
    auroc_traj_sub = compute_feature_auroc(feats[:, 4], y.astype(bool), higher_is_same=False)

    auroc_xyz_exact = compute_exact_auroc(feats[:, 0], y, higher_is_same=False)
    auroc_rgb_exact = compute_exact_auroc(feats[:, 1], y, higher_is_same=False)
    auroc_phasor_exact = compute_exact_auroc(feats[:, 3], y, higher_is_same=False)
    auroc_traj_exact = compute_exact_auroc(feats[:, 4], y, higher_is_same=False)

    # Fused logits with frozen weights
    # logit = intercept + beta_xyz * d_xyz + beta_rgb * d_rgb + beta_phasor * disc_phasor
    fused_logits = intercept + betas[0] * feats[:, 0] + betas[1] * feats[:, 1] + betas[2] * feats[:, 3]

    fused_auroc_sub = compute_feature_auroc(fused_logits, y.astype(bool), higher_is_same=True)
    fused_auroc_exact = compute_exact_auroc(fused_logits, y, higher_is_same=True)

    print(f"Edges: {len(edges)}, Same-part: {y.sum():.0f} ({p_same*100:.2f}%), Prior Log-Odds: {log_odds:+.3f}")
    print(f"Per-channel AUROC (subsampled / exact):")
    print(f"  Spatial (d_xyz):      {auroc_xyz_sub:.4f} / {auroc_xyz_exact:.4f}")
    print(f"  Appearance (d_rgb):   {auroc_rgb_sub:.4f} / {auroc_rgb_exact:.4f}")
    print(f"  Phasor (disc_phasor): {auroc_phasor_sub:.4f} / {auroc_phasor_exact:.4f}")
    print(f"  Traj (disc_traj):     {auroc_traj_sub:.4f} / {auroc_traj_exact:.4f}")
    print(f"Fused AUROC (subsampled / exact): {fused_auroc_sub:.4f} / {fused_auroc_exact:.4f}")

    results.append({
        "model": model_name,
        "is_train": is_train,
        "n_edges": len(edges),
        "same_pct": p_same * 100,
        "log_odds_prior": log_odds,
        "auroc_xyz": auroc_xyz_sub,
        "auroc_rgb": auroc_rgb_sub,
        "auroc_phasor": auroc_phasor_sub,
        "auroc_traj": auroc_traj_sub,
        "fused_auroc": fused_auroc_sub,
        "fused_auroc_exact": fused_auroc_exact,
    })

print("\n=== SUMMARY TABLE ===")
for r in results:
    print(f"{r['model']} ({'TRAIN' if r['is_train'] else 'HELD-OUT'}): edges={r['n_edges']}, same={r['same_pct']:.2f}%, "
          f"xyz={r['auroc_xyz']:.4f}, rgb={r['auroc_rgb']:.4f}, phasor={r['auroc_phasor']:.4f}, traj={r['auroc_traj']:.4f} | "
          f"FUSED={r['fused_auroc']:.4f} (exact={r['fused_auroc_exact']:.4f})")
