import os
import sys
import time
import numpy as np
import pandas as pd

from pipeline.vendored.host.kabsch_em import _em_single, _bic
from pipeline.vendored.host.trajectory_denoise import trajectory_energy
from pipeline.vendored.host.rigidity_graph2 import fps_subsample
from pipeline.vendored.host.rigidity_graph import merge_small_components

MODELS = ["grid-A20mm_M2", "grid-A20mm_M4", "grid-A40mm_M8"]
K_CANDIDATES = [2, 3, 5, 8, 12, 16, 20]
N_SUBSAMPLE = 5000
OPACITY_THRESH = 0.1
MAX_ITER = 15
TOL = 1e-4
RNG_SEED = 0

rows = []

print("=== Starting T20 BIC Rerun ===")
print(f"Models: {MODELS}")
print(f"Candidate K: {K_CANDIDATES}")
print(f"N_subsample: {N_SUBSAMPLE}, max_iter: {MAX_ITER}")

for model_id in MODELS:
    traj_path = os.path.join("runs", model_id, "trajectories.npz")
    print(f"\n--- Loading {model_id} from {traj_path} ---")
    data = np.load(traj_path)
    xyz = data["canonical_xyz"]
    traj = data["traj"]
    opacity = data["opacity"] if "opacity" in data.files else None

    if opacity is not None:
        keep = opacity > OPACITY_THRESH
    else:
        keep = np.ones(len(xyz), dtype=bool)

    xyz_valid = xyz[keep]
    traj_valid = traj[keep]
    print(f"Total points: {len(xyz)}, valid (opacity > {OPACITY_THRESH}): {len(xyz_valid)}")

    t0 = time.time()
    sel = fps_subsample(xyz_valid, N_SUBSAMPLE, rng_seed=RNG_SEED)
    dt_fps = time.time() - t0
    print(f"FPS subsampled {len(sel)} points in {dt_fps:.2f}s")

    xyz_sub = xyz_valid[sel]
    traj_sub = traj_valid[sel]
    N, T, _ = traj_sub.shape

    # Sigma auto-estimation per proposal 05 / T20
    energy = trajectory_energy(traj_sub)
    sigma = float(np.sqrt(np.median(energy) / max(3 * T, 1)))
    print(f"Auto-estimated noise sigma: {sigma:.8f}")

    for k in K_CANDIDATES:
        t_em = time.time()
        gamma, R, tau, residuals, it, info = _em_single(
            xyz_sub,
            traj_sub,
            k,
            sigma,
            init="fft",
            max_iter=MAX_ITER,
            tol=TOL,
            rng=np.random.default_rng(RNG_SEED),
        )
        dt_em = time.time() - t_em

        pred_labels = gamma.argmax(axis=1)
        n_pred_raw = len(np.unique(pred_labels))
        merged_labels = merge_small_components(xyz_sub, pred_labels, min_size=15)
        n_pred_merged = len(np.unique(merged_labels))

        weighted_r = float((gamma * residuals).sum())
        dof = 3 * T * N
        sigma2_mle = weighted_r / dof

        bic_gauss = _bic(residuals, gamma, k, T)
        # Legacy formula that produced 61494 at K=20 in original T20 run:
        # bic_legacy = weighted_r + (6TK + K) * ln(N)
        bic_legacy = weighted_r + (6 * T * k + k) * np.log(N)

        row = {
            "model": model_id,
            "k_candidate": k,
            "bic_gaussian": bic_gauss,
            "bic_legacy": bic_legacy,
            "weighted_residual": weighted_r,
            "sigma2_mle": sigma2_mle,
            "n_pred_raw": n_pred_raw,
            "n_pred_merged": n_pred_merged,
            "n_iter": it,
            "converged": info.get("converged", False),
            "max_shift": info.get("max_shift", np.nan),
            "time_s": round(dt_em, 2),
        }
        rows.append(row)
        print(
            f"[{model_id}] K={k:2d} | bic_gauss={bic_gauss:12.2f} | bic_legacy={bic_legacy:8.2f} | "
            f"weighted_r={weighted_r:.4f} | n_pred={n_pred_raw} (merged: {n_pred_merged}) | "
            f"it={it} (conv={info.get('converged', False)}) | time={dt_em:.2f}s"
        )

df = pd.DataFrame(rows)
out_csv = os.path.join("runs", "grid_seg_bic_rerun_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved rerun results to {out_csv}")
