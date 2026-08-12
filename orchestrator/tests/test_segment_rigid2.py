"""Tests for T18: `segment.rigid2` — FFT denoising + calibrated rigidity z-scores + spectral
partition (proposal 06, `docs/proposals/06-multiscale-snr-multiscale.md`).

Acceptance per `docs/proposals/IMPLEMENTATION_PLAN.md` §T18: on a synthetic rigid-body scene
with white trajectory noise injected at the *measured* failure ratio of the real grid models
(jitter ~ 0.3 x motion amplitude — the regime where the baseline `segment.rigid` collapses on
real data, `runs/grid_seg_results.csv` ARI ~ 0.002-0.009), rigid2 must recover the parts. Also
covered: drive-frequency auto-detection, calibrated z-score separability (AUROC go/no-go bar),
the components fallback partition, subsample+propagate, and the stage-level DAG run.

The fixture is T18-specific (T07's `_make_synthetic_rigid_scene` places its parts 0.9 units
apart, so no k-NN edge ever crosses a part boundary and the z-score separability diagnostic has
nothing to classify): parts here are *adjacent* (gap ~ in-part point spacing, so boundary edges
exist) and share **one drive frequency** with different axes/amplitudes/phases — exactly the
pump scenes' regime and the denoiser's narrowband assumption (`scene-gen/gen_scenes.py`).

CPU-only, no GPU/containers — runs in the default sandbox suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.vendored.host.metrics import adjusted_rand_index as ari

DRIVE_FREQ = 4  # cycles per clip window; bandpass(harmonics=3) keeps bins 4, 8, 12
T = 60


def _make_t18_scene(seed: int = 0, noise_sigma: float = 0.008):
    """Static base + 6 adjacent rigid parts rotating sinusoidally about their centroids, all
    driven at DRIVE_FREQ with per-part axis/amplitude/phase. Motion amplitude ~0.02-0.05;
    noise_sigma=0.008 ≈ 0.3x — the real models' measured failure ratio (proposal 06 §1).
    Returns (xyz (N,3), traj_noisy (N,T,3), gt (N,))."""
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
        center = np.array([(p - (n_parts - 1) / 2) * 0.42, 0.6, 0.0])  # gap ~0.12, parts touch k-NN
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


# --- unit level: the vendored pieces ------------------------------------------------------------


def test_detect_drive_freq_finds_the_drive_bin():
    from pipeline.vendored.host.trajectory_denoise import bandpass, detect_drive_freq

    _, traj, gt = _make_t18_scene()
    f0 = detect_drive_freq(traj[gt > 0])
    assert f0 == DRIVE_FREQ
    denoised, f0_used = bandpass(traj, drive_freq=None, harmonics=3)
    assert f0_used == DRIVE_FREQ
    assert denoised.shape == traj.shape
    rough = lambda x: np.abs(np.diff(x, axis=1)).mean()
    assert rough(denoised) < rough(traj)  # band-passed output is smoother than the noisy input


def test_calibrated_z_scores_separate_same_from_cross_part_edges():
    from pipeline.vendored.host.rigidity_graph import build_knn_edges
    from pipeline.vendored.host.rigidity_graph2 import (
        calibrate_sigma_d,
        edge_zscores,
        separability_auroc,
    )
    from pipeline.vendored.host.trajectory_denoise import bandpass

    xyz, traj, gt = _make_t18_scene()
    edges = build_knn_edges(xyz, k=12)
    sigma_d = calibrate_sigma_d(traj, edges)
    assert sigma_d > 0
    denoised, _ = bandpass(traj, harmonics=3)
    z = edge_zscores(denoised, edges, sigma_d)
    sep = separability_auroc(z, edges, gt)
    assert sep["n_diff_edges"] > 0 and sep["n_same_edges"] > 0  # boundary edges exist here
    # The go/no-go bar from IMPLEMENTATION_PLAN §3.
    assert sep["auroc"] > 0.8
    assert sep["z_median_diff"] > sep["z_median_same"]


def test_segment_by_rigidity2_recovers_parts_under_noise():
    from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

    xyz, traj, gt = _make_t18_scene()
    labels, info = segment_by_rigidity2(xyz, traj, k=12, min_size=15, max_clusters=12, gt_labels=gt)
    assert info["sigma_d"] is not None and info["sigma_d"] > 0
    assert info["separability"]["auroc"] > 0.8
    score = ari(gt, labels)
    # The acceptance bar: ARI >= 0.95 at the measured failure ratio (T18 plan).
    assert score >= 0.95, f"ARI {score:.4f} < 0.95 (info: { {k: v for k, v in info.items() if k != 'eigenvalues'} })"


def test_components_partition_fallback_runs():
    from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

    xyz, traj, gt = _make_t18_scene()
    labels, info = segment_by_rigidity2(
        xyz, traj, k=12, min_size=15, partition="components", z_thresh=3.0
    )
    assert labels.shape[0] == len(xyz)
    assert info["n_components_final"] >= 2
    # Adaptive cut = min(z_thresh, median + 6*MAD) — on this clean-ratio fixture the robust
    # term is the stricter one, so the effective threshold lands below the configured 3.0.
    assert 0 < info["threshold"] <= 3.0


def test_subsample_and_propagate():
    from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

    xyz, traj, gt = _make_t18_scene()
    labels, info = segment_by_rigidity2(
        xyz, traj, k=12, min_size=15, max_clusters=12, n_subsample=800
    )
    assert labels.shape[0] == len(xyz)
    assert info["propagated"] is True and info["n_subsample"] == 800


# --- stage level: the DAG run, same shape as test_stages_cpu.py ---------------------------------


def test_segment_rigid2_stage_runs_end_to_end(tmp_path):
    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.api import _stage_config_for
    from pipeline.config import validate_config
    from pipeline.dag import run_dag

    xyz, traj, gt = _make_t18_scene()
    traj_path = tmp_path / "trajectories.npz"
    np.savez(traj_path, canonical_xyz=xyz, traj=traj)
    gt_path = tmp_path / "gt_segmentation.npz"
    np.savez(gt_path, points=xyz.astype(np.float32), labels=gt)

    resolved = validate_config("pump01_segB2").model_dump()
    assert resolved["segment"]["impl"] == "rigid2"
    resolved["segment"]["rigid2"]["max_clusters"] = 12
    resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)

    names = ["segment.rigid2", "seg_eval.default"]
    stage_configs = {n: _stage_config_for(n, resolved) for n in names}

    run_id = "t18-rigid2"
    create_run(run_id, "pump01_segB2", resolved, stage_names=names, runs_root=tmp_path)

    def _seed(m):
        m.artifacts["trajectories"] = Artifact(
            name="trajectories", kind="npz", path=str(traj_path), producing_stage="external"
        )
        m.artifacts["gt_segmentation"] = Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        )

    update_manifest(run_id, _seed, runs_root=tmp_path)
    manifest = run_dag(
        run_id, names, resolved, preset="pump01_segB2", stage_configs=stage_configs,
        runs_root=tmp_path,
    )

    assert manifest.status == "success"
    assert [manifest.stages[n].status for n in names] == ["success", "success"]

    seg = np.load(manifest.artifacts["segmentation"].path)
    assert set(seg.files) >= {"points", "labels"}

    # Separability diagnostic artifact exists and clears the go/no-go bar.
    sep = json.loads(Path(manifest.artifacts["separability"].path).read_text())
    assert sep["denoised_z"]["auroc"] > 0.8

    summary = json.loads(Path(manifest.artifacts["seg_eval_result"].path).read_text())
    assert summary["ari"] >= 0.95, f"eval ARI {summary['ari']:.4f} < 0.95"
    assert ari(gt, seg["labels"]) >= 0.95
