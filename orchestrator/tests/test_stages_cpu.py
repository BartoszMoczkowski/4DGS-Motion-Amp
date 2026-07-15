"""Tests for T07 (wrap CPU stages): `convert.default` / `segment.rigid` / `seg_eval.default`.

Per `planning/tasks/T07-wrap-cpu-stages.md`'s acceptance criteria: runs the convert + segment +
eval slice from one `run_dag` call using synthetic fixtures (a tiny fake Omniverse capture for
`convert.default`; the same synthetic rigid-body scene `motion_seg/segment_rigid.py`'s own
`_selftest()` builds, for `segment.rigid`/`seg_eval.default`), then exercises caching: an
unchanged rerun (new run_id) is fully cached, and changing `segment.rigid`'s `threshold_mult`
reruns only `segment.rigid` + `seg_eval.default` while `convert.default` (unaffected — a
different config section, same inputs) stays cached.

Per T07's design notes, `convert` and the `segment.rigid -> seg_eval.default` pair are two
independently-runnable stage chains in the *real* pipeline (their true upstream producers,
`capture.isaac` and `seg_extract`, are GPU stages out of this task's scope) — they're only
exercised together here, in one `run_dag` call, to prove the framework end-to-end on everything
that needs no GPU/containers.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest


# --- fixtures: a tiny fake Omniverse capture for convert.default -------------------------------


def _make_capture_fixture(root: Path, *, n_cams: int = 3, n_frames: int = 2, size: int = 32) -> Path:
    """A minimal capture directory in `omni_capture.py`'s output shape: `cameras_gt.json` +
    `camNN/rgb/rgb_XXXXX.png`. Cameras arranged in a ring looking at the origin, mirroring
    `omni_to_4dgs.py`'s own `_selftest()` camera setup (but far smaller: 3 cams x 2 frames x
    32x32px, since this is only exercising the file-format wrapping, not real geometry)."""
    from PIL import Image

    capture_dir = root / "capture"
    rng = np.random.RandomState(0)
    cams = []
    for i in range(n_cams):
        ang = 2 * np.pi * i / n_cams
        center = np.array([3 * np.cos(ang), 3 * np.sin(ang), 1.0])
        fwd = -center / np.linalg.norm(center)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_c2w = np.stack([right, down, fwd], axis=1)
        c2w = np.eye(4)
        c2w[:3, :3] = r_c2w
        c2w[:3, 3] = center

        folder = f"cam{i + 1:02d}"
        cam_rgb_dir = capture_dir / folder / "rgb"
        cam_rgb_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(1, n_frames + 1):
            arr = rng.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cam_rgb_dir / f"rgb_{fi:05d}.png")

        cams.append({"folder": folder, "c2w": c2w.tolist()})

    gt = {
        "intrinsics": {"width": size, "height": size, "fx": size, "fy": size, "cx": size / 2, "cy": size / 2},
        "meters_per_unit": 1.0,
        "cameras": cams,
    }
    (capture_dir / "cameras_gt.json").write_text(json.dumps(gt))
    return capture_dir


# --- fixture: the same synthetic rigid-body scene segment_rigid.py's own _selftest() builds ----


def _make_synthetic_rigid_scene(seed: int = 0):
    """Mirrors `motion_seg/segment_rigid.py`'s `_selftest()` generator exactly (a static base +
    6 independently-rotating rigid parts, each with its own rotation axis/frequency/amplitude) —
    leaning on an already-verified synthetic scene rather than inventing a new one, per T07's
    design notes. Returns (xyz (N,3), traj (N,T,3), gt_labels (N,))."""
    rng = np.random.RandomState(seed)
    T = 60
    times = np.linspace(0.0, 1.0, T, endpoint=False)

    xyz_list, traj_list, gt_list = [], [], []
    lid = 0

    base_pts = rng.uniform(-1.0, 1.0, size=(2000, 3)) * np.array([3.0, 0.3, 3.0])
    xyz_list.append(base_pts)
    traj_list.append(np.repeat(base_pts[:, None, :], T, axis=1))
    gt_list.append(np.full(len(base_pts), lid))
    lid += 1

    n_parts = 6
    for p in range(n_parts):
        center = np.array([(p - n_parts / 2) * 1.2, 1.0, 0.0])
        pts = center + rng.uniform(-0.15, 0.15, size=(150, 3))
        centroid = pts.mean(axis=0)
        freq = 2 + p
        amp_deg = 3.0 + p
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        traj = np.empty((len(pts), T, 3))
        rel = pts - centroid
        for ti, t in enumerate(times):
            theta = np.deg2rad(amp_deg) * np.sin(2 * np.pi * freq * t)
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

    xyz = np.concatenate(xyz_list).astype(np.float64)
    traj = np.concatenate(traj_list).astype(np.float64)
    gt = np.concatenate(gt_list)
    return xyz, traj, gt


NAMES = ["convert.default", "segment.rigid", "seg_eval.default"]


def _default_resolved_config() -> dict:
    from pipeline.config import validate_config

    return validate_config("base").model_dump()


def _stage_configs(resolved: dict) -> dict:
    return {
        "convert.default": resolved["convert"],
        "segment.rigid": resolved["segment"]["rigid"],
        "seg_eval.default": resolved["seg_eval"],
    }


def _write_common_fixtures(tmp_path: Path):
    """Builds the capture dir + trajectories.npz + gt_segmentation.npz once; returns
    (capture_dir, traj_path, gt_path)."""
    capture_dir = _make_capture_fixture(tmp_path)
    xyz, traj, gt = _make_synthetic_rigid_scene()

    traj_path = tmp_path / "trajectories.npz"
    np.savez(traj_path, canonical_xyz=xyz, traj=traj)

    gt_path = tmp_path / "gt_segmentation.npz"
    # Same points as segment.rigid's own predicted output (both derive from the same `xyz`), so
    # NN label propagation in seg_eval is exact/trivial while still exercising the real code path.
    np.savez(gt_path, points=xyz.astype(np.float32), labels=gt)

    return capture_dir, traj_path, gt_path


def _external_artifacts(capture_dir: Path, traj_path: Path, gt_path: Path) -> dict:
    from pipeline.artifacts import Artifact

    return {
        "capture": Artifact(
            name="capture", kind="dataset", path=str(capture_dir), producing_stage="external"
        ),
        "trajectories": Artifact(
            name="trajectories", kind="npz", path=str(traj_path), producing_stage="external"
        ),
        "gt_segmentation": Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        ),
    }


def _seed_run(run_id: str, preset: str, resolved: dict, stage_names, external: dict, runs_root: Path):
    """Pre-seeds `external`'s artifacts into `run_id`'s manifest before `run_dag` runs — the
    "external input satisfied by resume" path `run_dag`'s `known_artifacts`/`truly_missing` logic
    supports (see `pipeline/dag/scheduler.py`)."""
    from pipeline.artifacts import create_run, update_manifest

    create_run(run_id, preset, resolved, stage_names=list(stage_names), runs_root=runs_root)

    def _mutate(m):
        for name, art in external.items():
            m.artifacts[name] = art

    update_manifest(run_id, _mutate, runs_root=runs_root)


# --- acceptance criterion 1: the slice runs end to end, real output files -----------------------


def test_convert_segment_eval_slice_runs_end_to_end(tmp_path):
    from pipeline.dag import run_dag

    capture_dir, traj_path, gt_path = _write_common_fixtures(tmp_path)
    external = _external_artifacts(capture_dir, traj_path, gt_path)

    resolved = _default_resolved_config()
    # `segment_rigid.py`'s Otsu-log auto-threshold (`threshold_mult=1.0`, the config default)
    # turns out not to work on *this* fixture in this environment: the synthetic scene's rigid
    # parts are exactly rigid up to float64 rounding (~1e-16), and at this k every k-NN edge is
    # already within one true part (no genuine cross-part edge exists to give Otsu a real
    # bimodal signal) — confirmed by running `python -m motion_seg.segment_rigid --selftest`
    # unmodified, which also reports ARI=0.0/FAIL in this sandbox's numpy 2.2.6/scipy 1.15.3.
    # Otsu ends up drawing its cut through pure floating-point noise, fragmenting every part into
    # hundreds of sub-min_size pieces that then merge back together across true part boundaries.
    # A large `threshold_mult` (confirmed empirically: correct above ~1e6, this uses 1e7 for
    # margin) keeps essentially every edge, which is exactly right here since there was never a
    # cross-part edge to cut in the first place. This is a pre-existing fragility in
    # `motion_seg/rigidity_graph.py`'s auto-threshold for near-noiseless synthetic data, not
    # something T07 touches ("wrap, don't rewrite") — flagged in the T07 log, not fixed here.
    resolved["segment"]["rigid"]["threshold_mult"] = 1e7
    stage_configs = _stage_configs(resolved)

    run_id = "t07-slice"
    _seed_run(run_id, "t07", resolved, NAMES, external, tmp_path)
    manifest = run_dag(
        run_id, NAMES, resolved, preset="t07", stage_configs=stage_configs, runs_root=tmp_path
    )

    assert manifest.status == "success"
    assert [manifest.stages[n].status for n in NAMES] == ["success", "success", "success"]

    scene_artifact = manifest.artifacts["scene"]
    scene_dir = Path(scene_artifact.path)
    assert scene_dir.is_dir()
    assert (scene_dir / "sparse_" / "cameras.bin").is_file()
    assert (scene_dir / "sparse_" / "images.bin").is_file()
    assert (scene_dir / "points3D_multipleview.ply").is_file()

    seg_artifact = manifest.artifacts["segmentation"]
    seg_data = np.load(seg_artifact.path)
    assert set(seg_data.files) >= {"points", "labels"}
    assert seg_data["labels"].shape[0] == seg_data["points"].shape[0]

    eval_artifact = manifest.artifacts["seg_eval_result"]
    summary = json.loads(Path(eval_artifact.path).read_text())
    # Same bar segment_rigid.py's own _selftest() holds its recovered labels to.
    assert summary["ari"] > 0.9
    assert summary["n_gt"] == 7  # static base + 6 rotating parts

    # Re-running the *same* run_id is a no-op that keeps the honest "success" status (T05's
    # same-run vs cross-run distinction) rather than downgrading it to "skipped".
    manifest2 = run_dag(
        run_id, NAMES, resolved, preset="t07", stage_configs=stage_configs, runs_root=tmp_path
    )
    assert [manifest2.stages[n].status for n in NAMES] == ["success", "success", "success"]


# --- acceptance criterion 2: manifest + artifacts queryable afterward ---------------------------


def test_manifest_and_artifacts_are_queryable_via_the_store(tmp_path):
    from pipeline.artifacts import get_manifest, list_artifacts
    from pipeline.dag import run_dag

    capture_dir, traj_path, gt_path = _write_common_fixtures(tmp_path)
    external = _external_artifacts(capture_dir, traj_path, gt_path)
    resolved = _default_resolved_config()
    stage_configs = _stage_configs(resolved)

    run_id = "t07-query"
    _seed_run(run_id, "t07", resolved, NAMES, external, tmp_path)
    run_dag(run_id, NAMES, resolved, preset="t07", stage_configs=stage_configs, runs_root=tmp_path)

    manifest = get_manifest(run_id, runs_root=tmp_path)
    assert manifest.run_id == run_id
    assert manifest.status == "success"

    artifacts = list_artifacts(run_id, runs_root=tmp_path)
    names = {a.name for a in artifacts}
    assert {"capture", "trajectories", "gt_segmentation", "scene", "segmentation", "seg_eval_result"} <= names


# --- acceptance criterion 3: caching — unchanged rerun cached, threshold_mult invalidates --------


def test_rerun_is_cached_then_threshold_mult_change_reruns_only_segment_and_eval(tmp_path):
    from pipeline.dag import run_dag

    capture_dir, traj_path, gt_path = _write_common_fixtures(tmp_path)
    resolved = _default_resolved_config()
    stage_configs = _stage_configs(resolved)

    def _external():
        return _external_artifacts(capture_dir, traj_path, gt_path)

    _seed_run("t07-cache-1", "t07", resolved, NAMES, _external(), tmp_path)
    m1 = run_dag(
        "t07-cache-1", NAMES, resolved, preset="t07", stage_configs=stage_configs, runs_root=tmp_path
    )
    assert [m1.stages[n].status for n in NAMES] == ["success", "success", "success"]

    # A fresh run_id, unchanged config -> cross-run cache hit for every stage (T05's cache index).
    _seed_run("t07-cache-2", "t07", resolved, NAMES, _external(), tmp_path)
    m2 = run_dag(
        "t07-cache-2", NAMES, resolved, preset="t07", stage_configs=stage_configs, runs_root=tmp_path
    )
    assert [m2.stages[n].status for n in NAMES] == ["skipped", "skipped", "skipped"]

    # Change segment.rigid's threshold_mult, another fresh run_id -> only segment.rigid +
    # seg_eval.default re-execute; convert.default has a different config section and the same
    # input, so it's unaffected and still hits the cross-run cache.
    resolved2 = copy.deepcopy(resolved)
    resolved2["segment"]["rigid"]["threshold_mult"] = 2.0
    stage_configs2 = _stage_configs(resolved2)
    _seed_run("t07-cache-3", "t07", resolved2, NAMES, _external(), tmp_path)
    m3 = run_dag(
        "t07-cache-3", NAMES, resolved2, preset="t07", stage_configs=stage_configs2, runs_root=tmp_path
    )
    assert m3.stages["convert.default"].status == "skipped"
    assert m3.stages["segment.rigid"].status == "success"
    assert m3.stages["seg_eval.default"].status == "success"


# --- registry sanity: the three stages are actually registered under the expected names ---------


def test_stages_are_registered_under_the_expected_names():
    from pipeline.stages import get_stage, list_stages

    assert {"convert.default", "segment.rigid", "seg_eval.default"} <= set(list_stages())
    for name in ("convert.default", "segment.rigid", "seg_eval.default"):
        cls = get_stage(name)
        assert cls.environment == "host"
        assert cls.resources.needs_gpu is False


# --- pipeline.api: a real preset now auto-plans onto these three real stages --------------------


def test_api_auto_stage_plan_now_includes_the_three_real_roles():
    from pipeline.api import _auto_stage_plan

    resolved = _default_resolved_config()
    plan = _auto_stage_plan(resolved)
    assert set(plan) >= {"convert.default", "segment.rigid", "seg_eval.default"}
