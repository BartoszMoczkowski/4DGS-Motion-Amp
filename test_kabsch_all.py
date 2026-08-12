import sys
sys.path.insert(0, 'orchestrator')
import json
from pathlib import Path
import numpy as np
from pipeline.vendored.host.metrics import adjusted_rand_index as ari
from pipeline.vendored.host.kabsch_em import (
    weighted_kabsch, _em_single, segment_by_kabsch,
    _compute_residuals, _bic, _greedy_split,
)

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

passed = 0
failed = 0

def check(name, cond, msg=""):
    global passed, failed
    if cond:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {msg}")
        failed += 1

# Test 1: identity Kabsch
print("\nTest 1: weighted_kabsch identity")
pts = np.random.randn(50, 3)
R, tau = weighted_kabsch(pts, pts)
check("R=I", np.allclose(R, np.eye(3)))
check("tau=0", np.allclose(tau, 0.0))

# Test 2: known transform
print("\nTest 2: weighted_kabsch known transform")
theta = np.deg2rad(30)
R_true = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
t_true = np.array([1.0, 2.0, 3.0])
pts_t = (R_true @ pts.T).T + t_true
R, tau = weighted_kabsch(pts, pts_t)
check("R matches", np.allclose(R, R_true))
check("tau matches", np.allclose(tau, t_true))

# Test 3: EM converges and improves
print("\nTest 3: EM converges and improves likelihood")
xyz, traj, gt = _make_scene()
gamma, R, tau, residuals, it, info = _em_single(
    xyz, traj, 7, 0.01, init="fft", drive_freq=DRIVE_FREQ, harmonics=3,
    max_iter=30, tol=1e-4, rng=np.random.default_rng(0),
)
check("shape gamma", gamma.shape == (len(xyz), 7))
check("shape R", R.shape == (7, T, 3, 3))
check("converged", info["converged"], f"max_shift={info['max_shift']}")

# Test 4: full pipeline fixed K
print("\nTest 4: segment_by_kabsch K=7")
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
    harmonics=3, max_iter=30, sigma=0.01, rng_seed=0,
)
score = ari(gt, labels)
check(f"ARI >= 0.95 (got {score:.4f})", score >= 0.95)

# Test 5: BIC prefers correct K
print("\nTest 5: BIC prefers correct K")
bics = {}
for k in [3, 5, 7, 10, 15]:
    g, Rk, tk, res, _, _ = _em_single(
        xyz, traj, k, 0.01, init="fft", drive_freq=DRIVE_FREQ,
        max_iter=20, rng=np.random.default_rng(0),
    )
    bics[k] = _bic(res, g, k, T)
best_k = min(bics, key=bics.get)
check(f"BIC near K=7 (best={best_k})", best_k in [5, 7, 10], str(bics))

# Test 6: greedy split
print("\nTest 6: greedy split")
g, Rk, tk, res, _, _ = _em_single(
    xyz, traj, 6, 0.01, init="fft", drive_freq=DRIVE_FREQ,
    max_iter=20, rng=np.random.default_rng(0),
)
ng, nR, nt, nl, nres, ginfo = _greedy_split(xyz, traj, g, Rk, tk, 0.01, res, rng=np.random.default_rng(0))
check("split attempted", ginfo["n_splits"] == 1)

# Test 7: FPS subsample
print("\nTest 7: FPS subsample")
labels, info = segment_by_kabsch(
    xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
    fps_subsample=800, propagate_q=3, max_iter=20, sigma=0.01, rng_seed=0,
)
check("propagated", info.get("propagated") is True)
check("n_subsample=800", info.get("n_subsample") == 800)
score = ari(gt, labels)
check(f"subsampled ARI >= 0.90 (got {score:.4f})", score >= 0.90)

# Test 8: stage level
print("\nTest 8: segment.kabsch stage end-to-end")
from pipeline.artifacts import Artifact, create_run, update_manifest
from pipeline.api import _stage_config_for
from pipeline.config import validate_config
from pipeline.dag import run_dag

xyz, traj, gt = _make_scene()
traj_path = Path("test_tmp") / "trajectories.npz"
traj_path.parent.mkdir(exist_ok=True)
np.savez(traj_path, canonical_xyz=xyz, traj=traj)
gt_path = Path("test_tmp") / "gt_segmentation.npz"
np.savez(gt_path, points=xyz.astype(np.float32), labels=gt)

resolved = validate_config("pump01_kabsch").model_dump()
resolved["segment"]["kabsch"]["n_clusters"] = 7
resolved["segment"]["kabsch"]["fps_subsample"] = 0

names = ["segment.kabsch", "seg_eval.default"]
stage_configs = {n: _stage_config_for(n, resolved) for n in names}

run_id = "t20-kabsch-test"
create_run(run_id, "pump01_kabsch", resolved, stage_names=names, runs_root=Path("test_tmp"))

def _seed(m):
    m.artifacts["trajectories"] = Artifact(
        name="trajectories", kind="npz", path=str(traj_path), producing_stage="external"
    )
    m.artifacts["gt_segmentation"] = Artifact(
        name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
    )

update_manifest(run_id, _seed, runs_root=Path("test_tmp"))
manifest = run_dag(
    run_id, names, resolved, preset="pump01_kabsch", stage_configs=stage_configs,
    runs_root=Path("test_tmp"),
)

check("manifest success", manifest.status == "success")
check("stages success", [manifest.stages[n].status for n in names] == ["success", "success"])

seg = np.load(manifest.artifacts["segmentation"].path)
check("segmentation artifact", set(seg.files) >= {"points", "labels"})

summary = json.loads(Path(manifest.artifacts["seg_eval_result"].path).read_text())
check(f"eval ARI >= 0.95 (got {summary['ari']:.4f})", summary["ari"] >= 0.95)
check("ari match", ari(gt, seg["labels"]) >= 0.95)

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
    sys.exit(1)
