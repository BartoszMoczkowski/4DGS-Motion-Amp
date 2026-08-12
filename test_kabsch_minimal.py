import sys
sys.path.insert(0, 'orchestrator')
import numpy as np
from pipeline.vendored.host.kabsch_em import _em_single, segment_by_kabsch

# Simple ARI without scipy
def ari(gt, pred):
    n = len(gt)
    classes, class_idx = np.unique(gt, return_inverse=True)
    clusters, cluster_idx = np.unique(pred, return_inverse=True)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i in range(n):
        contingency[class_idx[i], cluster_idx[i]] += 1
    sum_comb_c = sum(c * (c - 1) for c in contingency.ravel())
    sum_comb_t = sum(a * (a - 1) for a in contingency.sum(axis=1))
    sum_comb_p = sum(b * (b - 1) for b in contingency.sum(axis=0))
    prod_combs = sum_comb_t * sum_comb_p / (n * (n - 1)) if n > 1 else 0
    mean_combs = (sum_comb_t + sum_comb_p) / 2
    if mean_combs == prod_combs:
        return 1.0
    return (sum_comb_c - prod_combs) / (mean_combs - prod_combs)

DRIVE_FREQ = 4
T = 60

def make_scene(seed=0, noise_sigma=0.008):
    rng = np.random.RandomState(seed)
    times = np.linspace(0.0, 1.0, T, endpoint=False)
    xyz_list, traj_list, gt_list = [], [], []
    lid = 0
    base_pts = rng.uniform(-1.0, 1.0, size=(1500, 3)) * np.array([1.5, 0.15, 0.75])
    xyz_list.append(base_pts)
    traj_list.append(np.repeat(base_pts[:, None, :], T, axis=1))
    gt_list.append(np.full(len(base_pts), lid))
    lid += 1
    for p in range(6):
        center = np.array([(p - 2.5) * 0.42, 0.6, 0.0])
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

xyz, traj, gt = make_scene()
print(f"Scene: {len(xyz)} points, {gt.max()+1} GT parts")

# Test 1: EM converges
print("\nTest 1: EM K=7 converges...")
gamma, R, tau, residuals, it, info = _em_single(
    xyz, traj, 7, 0.01, init="fft", drive_freq=DRIVE_FREQ, harmonics=3,
    max_iter=30, tol=1e-4, rng=np.random.default_rng(0),
)
print(f"  it={it}, converged={info['converged']}, max_shift={info['max_shift']:.6f}")
assert info["converged"], f"EM did not converge (max_shift={info['max_shift']})"

# Test 2: Full pipeline with fixed K
print("\nTest 2: segment_by_kabsch K=7...")
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
    harmonics=3, max_iter=30, sigma=0.01, rng_seed=0,
)
score = ari(gt, labels)
print(f"  ARI={score:.4f}, n_clusters_final={info['n_clusters_final']}")
assert score >= 0.95, f"ARI {score:.4f} < 0.95"

# Test 3: BIC search
print("\nTest 3: segment_by_kabsch BIC search...")
labels2, info2 = segment_by_kabsch(
    xyz, traj, init="fft", drive_freq=DRIVE_FREQ,
    harmonics=3, max_iter=20, sigma=0.01, rng_seed=0,
)
score2 = ari(gt, labels2)
print(f"  ARI={score2:.4f}, best_k={info2['best_k']}, n_final={info2['n_clusters_final']}")
assert score2 >= 0.90, f"BIC ARI {score2:.4f} < 0.90"

print("\n=== ALL TESTS PASSED ===")
