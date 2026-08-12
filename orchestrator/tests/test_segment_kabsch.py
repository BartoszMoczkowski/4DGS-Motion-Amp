"""Tests for T20: `segment.kabsch` — Kabsch EM rigid-body clustering (proposal 05,
``docs/proposals/05-iterative-kabsch-em.md``).

Acceptance: on a synthetic rigid-body scene with white trajectory noise (jitter ~ 0.3× motion
amplitude, the measured failure ratio of the real grid models), Kabsch EM must recover the
parts with ARI ≥ 0.95.  Also covered: weighted Kabsch correctness, E/M-step consistency,
FFT-fingerprint initialisation, BIC model selection, greedy split, and the stage-level DAG run.

CPU-only, no GPU/containers — runs in the default sandbox suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.vendored.host.metrics import adjusted_rand_index as ari

import numpy as np

from pipeline.vendored.host.metrics import adjusted_rand_index as ari

DRIVE_FREQ = 4  # cycles per clip window
T = 60


def _make_t20_scene(seed: int = 0, noise_sigma: float = 0.008):
    """Static base + 6 adjacent rigid parts rotating sinusoidally about their centroids, all
driven at DRIVE_FREQ with per-part axis/amplitude/phase.  Same fixture as T18 (reused for
comparability), with noise_sigma=0.008 ≈ 0.3× motion amplitude."""
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
            rotated = (
                rel * cos
                + np.cross(axis, rel) * sin
                + axis * (rel @ axis)[:, None] * (1 - cos)
            )
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


# --- unit level -------------------------------------------------------------------------------


def test_weighted_kabsch_recovers_identity_on_aligned_points():
    from pipeline.vendored.host.kabsch_em import weighted_kabsch

    rng = np.random.RandomState(0)
    pts = rng.randn(50, 3)
    R, tau = weighted_kabsch(pts, pts)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(tau, 0.0, atol=1e-6)


def test_weighted_kabsch_recovers_known_rigid_transform():
    from pipeline.vendored.host.kabsch_em import weighted_kabsch

    rng = np.random.RandomState(0)
    pts = rng.randn(50, 3)
    # Known rotation (30° about z) + translation
    theta = np.deg2rad(30)
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
    t_true = np.array([1.0, 2.0, 3.0])
    pts_t = (R_true @ pts.T).T + t_true
    R, tau = weighted_kabsch(pts, pts_t)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(tau, t_true, atol=1e-6)


def test_em_single_converges_and_improves_likelihood():
    from pipeline.vendored.host.kabsch_em import _em_single, _compute_residuals

    xyz, traj, gt = _make_t20_scene()
    K = 7  # 1 base + 6 parts
    sigma = 0.01

    gamma, R, tau, residuals, it, info = _em_single(
        xyz, traj, K, sigma, init="fft", drive_freq=DRIVE_FREQ, harmonics=3,
        max_iter=30, tol=1e-4, rng=np.random.default_rng(0),
    )
    assert gamma.shape == (len(xyz), K)
    assert R.shape == (K, T, 3, 3)
    assert tau.shape == (K, T, 3)
    assert it <= 30
    assert info["converged"]

    # Likelihood should be higher (residuals lower) than a random initialisation
    random_labels = np.random.RandomState(0).randint(0, K, size=len(xyz))
    from pipeline.vendored.host.kabsch_em import _init_from_labels, _m_step
    gamma_rand, R_rand, tau_rand = _init_from_labels(traj, xyz, random_labels, K)
    res_rand = _compute_residuals(traj, R_rand, tau_rand, xyz)
    res_final = _compute_residuals(traj, R, tau, xyz)
    # EM should beat random init in weighted residual
    assert (gamma * res_final).sum() < (gamma_rand * res_rand).sum()


def test_segment_by_kabsch_recovers_parts():
    from pipeline.vendored.host.kabsch_em import segment_by_kabsch

    xyz, traj, gt = _make_t20_scene()
    labels, info = segment_by_kabsch(
        xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
        harmonics=3, max_iter=30, sigma=0.01, rng_seed=0,
    )
    assert labels.shape == (len(xyz),)
    score = ari(gt, labels)
    assert score >= 0.95, f"ARI {score:.4f} < 0.95 (info: {info})"
    assert info["n_clusters_final"] >= 6  # at least the 6 parts + base


def test_bic_prefers_correct_k():
    from pipeline.vendored.host.kabsch_em import _em_single, _bic

    xyz, traj, gt = _make_t20_scene()
    sigma = 0.01
    bics = {}
    for k in [3, 5, 7, 10, 15]:
        gamma, R, tau, residuals, _, _ = _em_single(
            xyz, traj, k, sigma, init="fft", drive_freq=DRIVE_FREQ,
            max_iter=20, rng=np.random.default_rng(0),
        )
        bics[k] = _bic(residuals, gamma, k, T)
    # The true K=7 should be near the minimum
    best_k = min(bics, key=bics.get)
    assert best_k in [5, 7, 10], f"BIC minimum at K={best_k}, not near true K=7 (BICs: {bics})"


def test_greedy_split_can_improve_bic():
    from pipeline.vendored.host.kabsch_em import _em_single, _greedy_split

    xyz, traj, gt = _make_t20_scene()
    sigma = 0.01
    gamma, R, tau, residuals, _, _ = _em_single(
        xyz, traj, 6, sigma, init="fft", drive_freq=DRIVE_FREQ,
        max_iter=20, rng=np.random.default_rng(0),
    )
    # With K=6 (one part missing), greedy split should try to split the worst body
    new_gamma, new_R, new_tau, new_labels, new_res, info = _greedy_split(
        xyz, traj, gamma, R, tau, sigma, residuals, rng=np.random.default_rng(0),
    )
    assert info["n_splits"] == 1
    assert new_gamma.shape[1] == 7  # increased by one


def test_fps_subsample_path():
    from pipeline.vendored.host.kabsch_em import segment_by_kabsch

    xyz, traj, gt = _make_t20_scene()
    labels, info = segment_by_kabsch(
        xyz, traj, n_clusters=7, init="fft", drive_freq=DRIVE_FREQ,
        fps_subsample=800, propagate_q=3, max_iter=20, sigma=0.01, rng_seed=0,
    )
    assert labels.shape == (len(xyz),)
    assert info["propagated"] is True
    assert info["n_subsample"] == 800
    score = ari(gt, labels)
    assert score >= 0.90, f"subsampled ARI {score:.4f} < 0.90"


# --- stage level ------------------------------------------------------------------------------


def test_segment_kabsch_stage_runs_end_to_end(tmp_path):
    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.api import _stage_config_for
    from pipeline.config import validate_config
    from pipeline.dag import run_dag

    xyz, traj, gt = _make_t20_scene()
    traj_path = tmp_path / "trajectories.npz"
    np.savez(traj_path, canonical_xyz=xyz, traj=traj)
    gt_path = tmp_path / "gt_segmentation.npz"
    np.savez(gt_path, points=xyz.astype(np.float32), labels=gt)

    resolved = validate_config("pump01_kabsch").model_dump()
    assert resolved["segment"]["impl"] == "kabsch"
    resolved["segment"]["kabsch"]["n_clusters"] = 7
    resolved["segment"]["kabsch"]["fps_subsample"] = 0  # full set for accuracy

    names = ["segment.kabsch", "seg_eval.default"]
    stage_configs = {n: _stage_config_for(n, resolved) for n in names}

    run_id = "t20-kabsch"
    create_run(run_id, "pump01_kabsch", resolved, stage_names=names, runs_root=tmp_path)

    def _seed(m):
        m.artifacts["trajectories"] = Artifact(
            name="trajectories", kind="npz", path=str(traj_path), producing_stage="external"
        )
        m.artifacts["gt_segmentation"] = Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        )

    update_manifest(run_id, _seed, runs_root=tmp_path)
    manifest = run_dag(
        run_id, names, resolved, preset="pump01_kabsch", stage_configs=stage_configs,
        runs_root=tmp_path,
    )

    assert manifest.status == "success"
    assert [manifest.stages[n].status for n in names] == ["success", "success"]

    seg = np.load(manifest.artifacts["segmentation"].path)
    assert set(seg.files) >= {"points", "labels"}

    summary = json.loads(Path(manifest.artifacts["seg_eval_result"].path).read_text())
    assert summary["ari"] >= 0.95, f"eval ARI {summary['ari']:.4f} < 0.95"
    assert ari(gt, seg["labels"]) >= 0.95
