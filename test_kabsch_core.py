import sys
sys.path.insert(0, 'orchestrator')
import numpy as np
from pipeline.vendored.host.kabsch_em import _em_single, segment_by_kabsch
from pipeline.vendored.host.metrics import adjusted_rand_index as ari

DRIVE_FREQ = 4
T = 60

def _make_scene(seed=0, noise_sigma=0.008):
    rng = np.random.RandomState(seed)
    times = np.linspace(0.0, 1.0, T, endpoint=False)
    xyz_list, traj_list, gt_list = [], [], []
    lid = 0
    base_pts = rng.uniform(-1.0, 1.0, size=(1500, 3)) * np.array([1.5, 0.15, 0.75])
    xyz_list.append(base_pts)
    traj_list.append(np.repeat(base_pts[:, None, :], T, axis=1))
    gt_list.append(np.full(len(base_pts), lid))
    lid += 1
    n_parts = 6
    for p in range(n_parts):
        center = np.array([(p - (n_parts - 1) / 2) * 0.42, 0.6, 0.0])
        pts = center + rng.uniform(-0.15, 0.15, size=(150, 3))
        centroid = pts.mean(axis=0)
        amp_deg = 12.0 + 3.0 * p
        phase = 0.8 * p
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        traj = np.empty((len(pts), T, 3))
        rel = pts - centroid
        for ti, t in enumerate(times):
            theta = np.deg2rad(amp_deg) * np.sin(2 * np.pi * DRIVE_FREQ * t + phase)
            cos, sin = np.cos(theta), np.sin(theta)
            rotated = rel * cos + np.cross(axis, rel) * sin + axis * (rel @ axis)[:, None] * (1 - cos)
            traj[:, ti, :] = centroid + rotated
        xyz_list.append(pts)
        traj_list.append(traj)
        gt_list.append(np.full(len(pts), lid))
        lid += 1
    xyz = np.concatenate(xyz_list)
    traj = np.concatenate(traj_list)
    gt = np.concatenate(gt_list)
    traj = traj + np.random.RandomState(seed + 1).normal(scale=noise_sigma, size=traj.shape)
    return xyz, traj, gt

print("Making scene...")
xyz, traj, gt = _make_scene()
print(f"Scene: {len(xyz)} points, {T} frames, {gt.max()+1} GT parts")

print("\nTest: EM single K=7...")
gamma, R, tau, residuals, it, info = _em_single(
    xyz, traj, 7, 0.01, init="fft", drive_freq=DRIVE_FREQ, harmonics=3,
    max_iter=30, tol=1e-4, rng=np.random.default_rng(0),
)
print(f"  converged={info['converged']}, it={it}, max_shift={info['max_shift']:.6f}")

print("\nTest: segment_by_kabsch K=7...")
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
    harmonics=3, max_iter=30, sigma=0.01, rng_seed=0,
)
score = ari(gt, labels)
print(f"  ARI={score:.4f}, n_clusters_final={info['n_clusters_final']}, converged={info.get('converged')}")
print(f"  PASS" if score >= 0.95 else f"  FAIL (expected >= 0.95)")

print("\nTest: segment_by_kabsch with BIC search...")
labels2, info2 = segment_by_kabsch(
    xyz, traj, init="fft", drive_freq=DRIVE_FREQ,
    harmonics=3, max_iter=20, sigma=0.01, rng_seed=0,
)
score2 = ari(gt, labels2)
print(f"  ARI={score2:.4f}, best_k={info2['best_k']}, n_clusters_final={info2['n_clusters_final']}")
