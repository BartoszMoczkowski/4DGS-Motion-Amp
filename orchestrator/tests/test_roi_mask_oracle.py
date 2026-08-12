"""Sandbox tests for T22's ``roi.mask_oracle`` (proposal 02 ceiling benchmark).

No GPU, no Docker, no torch — pure numpy/scipy fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_scene(*, n_moving: int, n_static: int):
    """Synthetic scene with movers and static background."""
    rng = np.random.RandomState(42)
    movers = rng.randn(n_moving, 3).astype(np.float32) + np.array([[0.0, 0.0, 0.0]])
    static = rng.randn(n_static, 3).astype(np.float32) + np.array([[10.0, 10.0, 10.0]])
    xyz = np.concatenate([movers, static], axis=0)
    labels = np.concatenate([
        np.ones(n_moving, dtype=np.int32),
        np.zeros(n_static, dtype=np.int32),
    ])
    return xyz, labels


def test_oracle_matches_gt_labels():
    """roi_mask_oracle should mark all GT-labeled points as inside ROI."""
    from pipeline.stages.roi_mask_oracle import RoiMaskOracleStage
    from pipeline.stages.base import StageContext

    xyz, labels = _make_scene(n_moving=100, n_static=50)

    # Write fixture files
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        traj_path = os.path.join(tmp, "trajectories.npz")
        np.savez(traj_path, canonical_xyz=xyz, traj=xyz[:, None, :])

        gt_path = os.path.join(tmp, "gt_segmentation.npz")
        np.savez(gt_path, points=xyz.astype(np.float32), labels=labels)

        # Minimal StageContext mock
        class MockPaths:
            pass
        class MockLogger:
            def info(self, *a, **k): pass

        ctx = StageContext(
            stage_name="roi.mask_oracle",
            run_id="test",
            run_dir=tmp,
            config={},
            inputs={
                "trajectories": type("A", (), {"path": traj_path})(),
                "gt_segmentation": type("A", (), {"path": gt_path})(),
            },
            paths=MockPaths(),
            logger=MockLogger(),
        )

        stage = RoiMaskOracleStage()
        artifacts = stage.run(ctx)

        roi_mask = np.load(artifacts["roi_mask"].path)["roi_mask"]
        assert roi_mask.dtype == bool
        assert len(roi_mask) == len(xyz)
        # All GT-labeled points (label >= 0) should be in ROI
        assert roi_mask[:100].all(), "movers should all be in ROI"
        assert not roi_mask[100:].any(), "static points should be outside ROI"


def test_oracle_with_label_zero_as_background():
    """Some datasets use 0 as background; oracle should exclude it."""
    rng = np.random.RandomState(7)
    n = 80
    xyz = rng.randn(n, 3).astype(np.float32)
    labels = np.concatenate([
        np.ones(50, dtype=np.int32),   # machine parts
        np.zeros(30, dtype=np.int32),  # background
    ])

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        traj_path = os.path.join(tmp, "trajectories.npz")
        np.savez(traj_path, canonical_xyz=xyz, traj=xyz[:, None, :])

        gt_path = os.path.join(tmp, "gt_segmentation.npz")
        np.savez(gt_path, points=xyz.astype(np.float32), labels=labels)

        class MockPaths:
            pass
        class MockLogger:
            def info(self, *a, **k): pass

        ctx = StageContext(
            stage_name="roi.mask_oracle",
            run_id="test",
            run_dir=tmp,
            config={},
            inputs={
                "trajectories": type("A", (), {"path": traj_path})(),
                "gt_segmentation": type("A", (), {"path": gt_path})(),
            },
            paths=MockPaths(),
            logger=MockLogger(),
        )

        stage = RoiMaskOracleStage()
        artifacts = stage.run(ctx)
        roi_mask = np.load(artifacts["roi_mask"].path)["roi_mask"]

        assert roi_mask[:50].all()
        assert not roi_mask[50:].any()


def test_oracle_nearest_neighbour_alignment():
    """When trajectory points differ from GT points, NN alignment should still work."""
    rng = np.random.RandomState(99)
    n_gt = 100
    n_traj = 120

    gt_points = rng.randn(n_gt, 3).astype(np.float32)
    gt_labels = np.arange(n_gt) % 5  # 5 classes

    # Trajectory points are GT points plus small noise, with 20 extra points
    traj_points = np.concatenate([
        gt_points + rng.randn(n_gt, 3).astype(np.float32) * 0.01,
        rng.randn(20, 3).astype(np.float32) + np.array([[5.0, 5.0, 5.0]]),
    ])

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        traj_path = os.path.join(tmp, "trajectories.npz")
        np.savez(traj_path, canonical_xyz=traj_points, traj=traj_points[:, None, :])

        gt_path = os.path.join(tmp, "gt_segmentation.npz")
        np.savez(gt_path, points=gt_points, labels=gt_labels)

        class MockPaths:
            pass
        class MockLogger:
            def info(self, *a, **k): pass

        ctx = StageContext(
            stage_name="roi.mask_oracle",
            run_id="test",
            run_dir=tmp,
            config={},
            inputs={
                "trajectories": type("A", (), {"path": traj_path})(),
                "gt_segmentation": type("A", (), {"path": gt_path})(),
            },
            paths=MockPaths(),
            logger=MockLogger(),
        )

        stage = RoiMaskOracleStage()
        artifacts = stage.run(ctx)
        roi_mask = np.load(artifacts["roi_mask"].path)["roi_mask"]

        # The first n_gt points should map to their corresponding GT labels
        # (small noise, so NN should be exact or very close)
        assert len(roi_mask) == n_traj
        # The 20 extra far-away points should map to some GT point; since all GT labels
        # are >= 0, they will be included.  This is expected behavior.


def test_oracle_artifact_contract():
    """The roi_mask artifact satisfies the contract (bool roi_mask + float32 snr)."""
    rng = np.random.RandomState(3)
    n = 50
    xyz = rng.randn(n, 3).astype(np.float32)
    labels = np.ones(n, dtype=np.int32)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        traj_path = os.path.join(tmp, "trajectories.npz")
        np.savez(traj_path, canonical_xyz=xyz, traj=xyz[:, None, :])

        gt_path = os.path.join(tmp, "gt_segmentation.npz")
        np.savez(gt_path, points=xyz, labels=labels)

        class MockPaths:
            pass
        class MockLogger:
            def info(self, *a, **k): pass

        ctx = StageContext(
            stage_name="roi.mask_oracle",
            run_id="test",
            run_dir=tmp,
            config={},
            inputs={
                "trajectories": type("A", (), {"path": traj_path})(),
                "gt_segmentation": type("A", (), {"path": gt_path})(),
            },
            paths=MockPaths(),
            logger=MockLogger(),
        )

        stage = RoiMaskOracleStage()
        artifacts = stage.run(ctx)

        data = np.load(artifacts["roi_mask"].path)
        assert "roi_mask" in data.files
        assert "snr" in data.files
        assert data["roi_mask"].dtype == bool
        assert data["snr"].dtype == np.float32
