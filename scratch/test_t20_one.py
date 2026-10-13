import time
import numpy as np
from pipeline.vendored.host.kabsch_em import _em_single, _bic
from pipeline.vendored.host.trajectory_denoise import trajectory_energy
from pipeline.vendored.host.rigidity_graph2 import fps_subsample

d = np.load("runs/grid-A20mm_M2/trajectories.npz")
xyz = d["canonical_xyz"]
traj = d["traj"]
opacity = d["opacity"]
keep = opacity > 0.1
xyz_valid = xyz[keep]
traj_valid = traj[keep]

print(f"Total points: {len(xyz)}, valid (opacity>0.1): {len(xyz_valid)}")
t0 = time.time()
sel = fps_subsample(xyz_valid, 5000, rng_seed=0)
print(f"Subsampled 5000 points in {time.time() - t0:.2f}s")
xyz_sub = xyz_valid[sel]
traj_sub = traj_valid[sel]

N, T, _ = traj_sub.shape
energy = trajectory_energy(traj_sub)
sigma = float(np.sqrt(np.median(energy) / max(3 * T, 1)))
print(f"sigma: {sigma:.6f}")

for k in [2, 3, 5, 8]:
    t0 = time.time()
    gamma, R, tau, residuals, it, info = _em_single(
        xyz_sub, traj_sub, k, sigma,
        init="fft", max_iter=15, tol=1e-4, rng=np.random.default_rng(0)
    )
    bic_gauss = _bic(residuals, gamma, k, T)
    weighted_r = float((gamma * residuals).sum())
    bic_legacy = weighted_r + (6 * T * k + k) * np.log(N)
    clusters = len(np.unique(gamma.argmax(axis=1)))
    dt = time.time() - t0
    print(f"K={k}: time={dt:.2f}s, it={it}, converged={info['converged']}, n_pred={clusters}, bic_gauss={bic_gauss:.2f}, bic_legacy={bic_legacy:.2f}, weighted_r={weighted_r:.4f}")
