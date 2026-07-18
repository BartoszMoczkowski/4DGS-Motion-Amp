"""Tests for T11 (wrap Isaac stages): `prep_split.default` / `prep_motion.default` /
`capture.isaac`.

None of these can run for real without Isaac Sim + a GPU (same story as T08's
`ContainerManager`/T09's cuda stages -- see `pipeline/containers/MANUAL_CHECKLIST.md`). What *is*
verifiable here in the sandbox, mirroring T09's own "fake the one thing that needs a real
daemon/GPU" approach: the CLI-argument construction each stage builds, the capture-config YAML
`capture.isaac` writes, and the full `prep_split.default -> prep_motion.default -> capture.isaac
-> convert.default` dependency + caching behavior through `run_dag`.

**Revised 2026-07-16, same day as the "adjust the project plan" decision** (see
`.claude_notes/NOTES_pipeline_orchestration.md`): `capture.isaac` (`omni_capture.py`) now runs as a
**native Windows subprocess** against a real Isaac Sim install instead of the `isaac` Docker
container -- WSL2 doesn't support Vulkan, which Isaac Sim's RTX/Hydra renderer needs. So this file
now fakes two different things: `pipeline.containers.exec_in_container` (still used by
`prep_split.default`/`prep_motion.default`, which only need `pxr`/USD bindings and are unaffected)
via `_fake_isaac_exec`/`_FakeContainers`, and `pipeline.stages.capture_isaac.run_native_isaac_script`
(the new native path) via `_FakeNativeIsaac`/`_fake_run_native_isaac`. `pipeline.vendored.isaac.rig`'s
pure-numpy camera-rig math is exercised directly (no container/subprocess needed, unlike its
sibling modules in that package -- see that module's docstring).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.api import _stage_config_for
from pipeline.artifacts import Artifact
from pipeline.config.models import PipelineConfig
from pipeline.containers.manager import ExecResult
from pipeline.stages.base import StageContext
from pipeline.stages.capture_isaac import CaptureIsaacStage
from pipeline.stages.prep_motion import PrepMotionStage
from pipeline.stages.prep_split import PrepSplitStage


# --- pipeline.vendored.isaac.rig: pure numpy, safe to exercise directly (no container) ----------


def test_rig_ring_and_dome_produce_orthonormal_camera_matrices():
    from pipeline.vendored.isaac import rig

    poses = rig.ring(center=[0, 0, 0], radius=3, n=8, height=1)
    assert len(poses) == 8
    for c2w in poses:
        R = c2w[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R) - 1) < 1e-6

    d = rig.dome(center=[0, 0, 0], radius=5, n=10, n_rings=3)
    assert len(d) == 10

    intr = rig.intrinsics_from_fov(1600, 900, 45)
    assert intr["fx"] > 0 and intr["width"] == 1600


def test_rig_build_rig_dispatches_ring_vs_dome_by_layout():
    from pipeline.vendored.isaac import rig

    ring_poses = rig.build_rig({"layout": "ring", "n_cameras": 6}, [0, 0, 0], 10.0)
    dome_poses = rig.build_rig({"layout": "dome", "n_cameras": 6, "n_rings": 2}, [0, 0, 0], 10.0)
    assert len(ring_poses) == 6
    assert len(dome_poses) == 6


# --- shared fixtures/helpers (mirrors tests/test_stages_cuda.py's own pattern) ------------------


def _setup_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PIPELINE_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path / "assets"))
    return repo_root


class _FakeContainers:
    """Stands in for `ctx.containers` -- records every `exec_in_container` call and returns a
    canned result, exactly like `tests/test_stages_cuda.py`'s own fake. Only `prep_split.default`/
    `prep_motion.default` actually call `ctx.containers` now -- `capture.isaac` runs natively (see
    `_FakeNativeIsaac` below) and never touches this.
    """

    def __init__(self, exit_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.exit_code = exit_code

    def exec_in_container(self, env, cmd, *, log_path=None, workdir=None, environment=None):
        self.calls.append(
            {"env": env, "cmd": cmd, "log_path": log_path, "workdir": workdir, "environment": environment}
        )
        return ExecResult(exit_code=self.exit_code, log_path=log_path)


class _FakeNativeIsaac:
    """Stands in for `pipeline.stages.capture_isaac.run_native_isaac_script` -- records every call
    (as `(script_key, args, log_name)`) and drops a stub `cameras_gt.json` plus one empty `camNN/`
    directory per configured `rig.n_cameras` under the `--out` directory, matching
    `CaptureIsaacStage.run`'s own post-hoc success checks (see that method's docstring: it verifies
    both `cameras_gt.json` and the camera-directory *count* exist before declaring success, since
    Isaac Sim can exit 0 without either being real -- a bare no-op fake would make those checks
    correctly fail every capture test in this file).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, ctx, script_key, args, *, log_name=None):
        self.calls.append({"script_key": script_key, "args": args, "log_name": log_name})
        out_host = Path(_flag_value(args, "out"))
        out_host.mkdir(parents=True, exist_ok=True)
        (out_host / "cameras_gt.json").write_text("{}")

        config_host = Path(_flag_value(args, "config"))
        n_cameras = 1
        if config_host.is_file():
            doc = yaml.safe_load(config_host.read_text()) or {}
            n_cameras = int(doc.get("rig", {}).get("n_cameras", 1))
        for i in range(1, n_cameras + 1):
            (out_host / f"cam{i:02d}").mkdir(parents=True, exist_ok=True)


def _ctx(repo_root: Path, *, stage_name: str, config: dict, inputs: dict, run_id: str = "r1"):
    import pipeline.paths as paths_mod

    run_dir = repo_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fake_containers = _FakeContainers()
    ctx = StageContext(
        run_id=run_id,
        stage_name=stage_name,
        config=config,
        run_dir=run_dir,
        logger=logging.getLogger(f"test.{stage_name}"),
        inputs=inputs,
        paths=paths_mod,
        containers=fake_containers,
    )
    return ctx, fake_containers


def _flag_value(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(f"--{name}") + 1]


# --- prep_split.default --------------------------------------------------------------------


def test_prep_split_stage_builds_expected_cli(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    raw_mesh_path = tmp_path / "assets" / "CONJUNTO_BOMBAS.usd"

    cfg = PipelineConfig().model_dump()
    prep_cfg = {
        **_stage_config_for("prep_split.default", cfg),
        "group": "MYGROUP",
        "min_faces": 5,
        "preview": "seg_preview.png",
        "no_usd": True,
    }
    inputs = {
        "raw_mesh": Artifact(name="raw_mesh", kind="usd", path=str(raw_mesh_path), producing_stage="external")
    }
    ctx, fake = _ctx(repo_root, stage_name="prep_split.default", config=prep_cfg, inputs=inputs)

    result = PrepSplitStage().run(ctx)

    assert set(result) == {"segmented_mesh"}
    out_host = Path(result["segmented_mesh"].path)
    assert out_host == ctx.run_dir / "segmented_mesh.usd"
    assert result["segmented_mesh"].kind == "usd"

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["env"] == "isaac"
    cmd = call["cmd"]
    assert cmd[0] == "/isaac-sim/python.sh"
    # split_mesh needs the Kit-bootstrap wrapper ahead of it (T11 real-hardware finding, see
    # pipeline.stages.isaac_common's "Revised 2026-07-16" note) -- the vendored script itself is
    # the *second* .py argument, not the first.
    assert cmd[1].endswith("orchestrator/pipeline/stages/_isaac_kit_bootstrap.py")
    assert cmd[2].endswith("orchestrator/pipeline/vendored/isaac/split_mesh.py")
    assert str(raw_mesh_path) not in " ".join(cmd)  # host path never leaks into a container cmd
    assert _flag_value(cmd, "group") == "MYGROUP"
    assert _flag_value(cmd, "min-faces") == "5"
    assert _flag_value(cmd, "preview") == "seg_preview.png"
    assert "--no-usd" in cmd


def test_prep_split_stage_omits_preview_and_no_usd_flags_by_default(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    raw_mesh_path = tmp_path / "assets" / "CONJUNTO_BOMBAS.usd"

    cfg = PipelineConfig().model_dump()
    prep_cfg = _stage_config_for("prep_split.default", cfg)  # defaults: preview=None, no_usd=False
    inputs = {
        "raw_mesh": Artifact(name="raw_mesh", kind="usd", path=str(raw_mesh_path), producing_stage="external")
    }
    ctx, fake = _ctx(repo_root, stage_name="prep_split.default", config=prep_cfg, inputs=inputs)

    PrepSplitStage().run(ctx)

    cmd = fake.calls[0]["cmd"]
    assert "--preview" not in cmd
    assert "--no-usd" not in cmd


# --- prep_motion.default -------------------------------------------------------------------


def test_prep_motion_stage_builds_expected_cli(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    segmented = repo_root / "runs" / "r1" / "segmented_mesh.usd"
    segmented.parent.mkdir(parents=True)
    segmented.write_text("usd")

    cfg = PipelineConfig().model_dump()
    prep_cfg = {
        **_stage_config_for("prep_motion.default", cfg),
        "num_frames": 12,
        "fps": 30.0,
        "trans_amp_mm": [2.0, 6.0],
        "rot_surface_mm": [1.0, 4.0],
        "freq": [3, 7],
        "groups": 2,
        "exclude": [],  # explicitly exclude nothing -> --exclude with zero following values
        "seed": 7,
    }
    inputs = {
        "segmented_mesh": Artifact(
            name="segmented_mesh", kind="usd", path=str(segmented), producing_stage="prep_split.default"
        )
    }
    ctx, fake = _ctx(repo_root, stage_name="prep_motion.default", config=prep_cfg, inputs=inputs)

    result = PrepMotionStage().run(ctx)

    assert set(result) == {"animated_mesh", "motion_groups"}
    out_host = Path(result["animated_mesh"].path)
    assert out_host == ctx.run_dir / "animated_mesh.usd"
    assert Path(result["motion_groups"].path) == ctx.run_dir / "animated_mesh_motion_groups.json"

    cmd = fake.calls[0]["cmd"]
    assert cmd[0] == "/isaac-sim/python.sh"
    # add_motion also needs the Kit-bootstrap wrapper -- see the prep_split test's own comment.
    assert cmd[1].endswith("orchestrator/pipeline/stages/_isaac_kit_bootstrap.py")
    assert cmd[2].endswith("orchestrator/pipeline/vendored/isaac/add_motion.py")
    assert _flag_value(cmd, "num-frames") == "12"
    assert _flag_value(cmd, "fps") == "30.0"
    idx = cmd.index("--trans-amp-mm")
    assert cmd[idx + 1 : idx + 3] == ["2.0", "6.0"]
    idx = cmd.index("--rot-surface-mm")
    assert cmd[idx + 1 : idx + 3] == ["1.0", "4.0"]
    idx = cmd.index("--freq")
    assert cmd[idx + 1 : idx + 3] == ["3", "7"]
    assert _flag_value(cmd, "groups") == "2"
    assert _flag_value(cmd, "seed") == "7"
    # exclude=[] is a deliberate "exclude nothing" -> the flag IS present with zero values,
    # distinct from not passing --exclude at all (which would fall back to the script's own
    # default=["frame_base"]) -- see pipeline.stages.isaac_common.star_list_flag's docstring.
    assert "--exclude" in cmd
    exclude_idx = cmd.index("--exclude")
    following = cmd[exclude_idx + 1 :]
    assert following[0] == "--seed"  # nothing between --exclude and the next flag


def test_prep_motion_stage_exclude_defaults_to_frame_base(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    segmented = repo_root / "runs" / "r1" / "segmented_mesh.usd"
    segmented.parent.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    prep_cfg = _stage_config_for("prep_motion.default", cfg)  # exclude defaults to ["frame_base"]
    inputs = {
        "segmented_mesh": Artifact(
            name="segmented_mesh", kind="usd", path=str(segmented), producing_stage="prep_split.default"
        )
    }
    ctx, fake = _ctx(repo_root, stage_name="prep_motion.default", config=prep_cfg, inputs=inputs)

    PrepMotionStage().run(ctx)

    cmd = fake.calls[0]["cmd"]
    idx = cmd.index("--exclude")
    assert cmd[idx + 1] == "frame_base"
    assert "--plot" not in cmd  # unset by default


# --- capture.isaac (native subprocess -- see module docstring) --------------------------------


def test_capture_isaac_stage_writes_config_yaml_and_overrides_usd_and_out(tmp_path, monkeypatch):
    import pipeline.stages.capture_isaac as capture_isaac_mod

    repo_root = _setup_roots(tmp_path, monkeypatch)
    animated = repo_root / "runs" / "r1" / "animated_mesh.usd"
    animated.parent.mkdir(parents=True)
    animated.write_text("usd")

    cfg = PipelineConfig().model_dump()
    capture_cfg = _stage_config_for("capture.isaac", cfg)
    capture_cfg = {
        **capture_cfg,
        "scene": {**capture_cfg["scene"], "usd_path": "Q:/Omniverse/should/not/be/used.usd"},
        "rig": {**capture_cfg["rig"], "n_cameras": 4},
    }
    inputs = {
        "animated_mesh": Artifact(
            name="animated_mesh", kind="usd", path=str(animated), producing_stage="prep_motion.default"
        )
    }
    ctx, _ = _ctx(repo_root, stage_name="capture.isaac", config=capture_cfg, inputs=inputs)

    fake_native = _FakeNativeIsaac()
    monkeypatch.setattr(capture_isaac_mod, "run_native_isaac_script", fake_native)

    result = CaptureIsaacStage().run(ctx)

    assert set(result) == {"capture"}
    capture_dir_host = Path(result["capture"].path)
    assert capture_dir_host == ctx.run_dir / "capture"
    assert result["capture"].kind == "dataset"
    assert result["capture"].metadata["n_cameras"] == 4

    assert len(fake_native.calls) == 1
    call = fake_native.calls[0]
    assert call["script_key"] == "omni_capture"
    assert call["log_name"] == "capture"
    args = call["args"]

    # --usd overrides the (irrelevant) static config value with the real animated_mesh input
    # path -- straight through as a plain host path, no container translation (native execution
    # shares this process's own filesystem, see capture_isaac.py's module docstring).
    usd_arg = _flag_value(args, "usd")
    assert "should/not/be/used" not in usd_arg
    assert Path(usd_arg) == animated

    out_arg = _flag_value(args, "out")
    assert Path(out_arg) == capture_dir_host

    config_arg = _flag_value(args, "config")
    yaml_host = Path(config_arg)
    assert yaml_host.is_file()
    doc = yaml.safe_load(yaml_host.read_text())
    assert doc["app"]["headless"] is True
    assert doc["rig"]["n_cameras"] == 4
    # near/far are None (CaptureFrameConfig defaults) -> dropped, not written as `null` (which
    # would shadow omni_capture.py's own radius-based fallback -- see _write_capture_config's
    # docstring / _NONE_SHADOWS_DEFAULT).
    assert "near" not in doc["capture"]
    assert "far" not in doc["capture"]


def test_capture_isaac_stage_keeps_explicit_near_far_when_set(tmp_path, monkeypatch):
    import pipeline.stages.capture_isaac as capture_isaac_mod

    repo_root = _setup_roots(tmp_path, monkeypatch)
    animated = repo_root / "runs" / "r1" / "animated_mesh.usd"
    animated.parent.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    capture_cfg = _stage_config_for("capture.isaac", cfg)
    capture_cfg = {**capture_cfg, "capture": {**capture_cfg["capture"], "near": 90.0, "far": 7000.0}}
    inputs = {
        "animated_mesh": Artifact(
            name="animated_mesh", kind="usd", path=str(animated), producing_stage="prep_motion.default"
        )
    }
    ctx, _ = _ctx(repo_root, stage_name="capture.isaac", config=capture_cfg, inputs=inputs)

    fake_native = _FakeNativeIsaac()
    monkeypatch.setattr(capture_isaac_mod, "run_native_isaac_script", fake_native)

    CaptureIsaacStage().run(ctx)

    yaml_host = Path(_flag_value(fake_native.calls[0]["args"], "config"))
    doc = yaml.safe_load(yaml_host.read_text())
    assert doc["capture"]["near"] == 90.0
    assert doc["capture"]["far"] == 7000.0


def test_capture_isaac_stage_raises_if_cameras_gt_json_never_written(tmp_path, monkeypatch):
    """The post-hoc success check (T11 real-hardware fixup): a native exec that "succeeds" (no
    exception) but never writes `cameras_gt.json` -- Isaac Sim's own shutdown path has been
    observed to still exit 0 after a fatal in-process error -- must fail the stage, not report a
    bogus success (see `CaptureIsaacStage.run`'s own docstring/comments)."""
    import pipeline.stages.capture_isaac as capture_isaac_mod
    from pipeline.stages.isaac_common import IsaacStageError

    repo_root = _setup_roots(tmp_path, monkeypatch)
    animated = repo_root / "runs" / "r1" / "animated_mesh.usd"
    animated.parent.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    capture_cfg = _stage_config_for("capture.isaac", cfg)
    inputs = {
        "animated_mesh": Artifact(
            name="animated_mesh", kind="usd", path=str(animated), producing_stage="prep_motion.default"
        )
    }
    ctx, _ = _ctx(repo_root, stage_name="capture.isaac", config=capture_cfg, inputs=inputs)

    def _noop_native(ctx, script_key, args, *, log_name=None):
        pass  # no filesystem side effect at all -- mirrors a swallowed in-process crash

    monkeypatch.setattr(capture_isaac_mod, "run_native_isaac_script", _noop_native)

    with pytest.raises(IsaacStageError, match="cameras_gt.json"):
        CaptureIsaacStage().run(ctx)


# --- registry sanity -------------------------------------------------------------------------


def test_isaac_stages_are_registered_under_the_expected_names():
    from pipeline.stages import get_stage, list_stages

    expected = {"prep_split.default", "prep_motion.default", "capture.isaac"}
    assert expected <= set(list_stages())
    for name in expected:
        cls = get_stage(name)
        assert cls.environment == "isaac"
    assert get_stage("prep_split.default").resources.needs_gpu is False
    assert get_stage("prep_motion.default").resources.needs_gpu is False
    assert get_stage("capture.isaac").resources.needs_gpu is True


def test_api_auto_stage_plan_includes_the_isaac_roles_with_no_role_collision():
    from pipeline.api import _auto_stage_plan
    from pipeline.config import validate_config

    resolved = validate_config("base").model_dump()
    plan = _auto_stage_plan(resolved)
    assert set(plan) >= {"prep_split.default", "prep_motion.default", "capture.isaac"}
    # exactly one of each -- proves prep_split/prep_motion didn't collide into one ambiguous
    # "prep" role (see pipeline.stages.prep_split's module docstring for the gap this avoids).
    assert plan.count("prep_split.default") == 1
    assert plan.count("prep_motion.default") == 1


# --- end-to-end through run_dag: prep_split -> prep_motion -> capture.isaac -> convert.default --


def _make_capture_dir_at(capture_dir: Path, *, n_cams: int = 3, n_frames: int = 2, size: int = 32) -> None:
    """Writes a real, minimal Omniverse-capture-shaped directory directly at `capture_dir`
    (mirrors `tests/test_stages_cpu.py`'s `_make_capture_fixture`, parameterized by target dir
    instead of always creating a fresh `capture/` subdirectory) so `convert.default`'s real,
    unmocked logic can run against it afterward."""
    from PIL import Image

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


def _fake_isaac_exec(paths_mod):
    """A fake `exec_in_container` that plays `split_mesh`/`add_motion`'s part just well enough
    for the real downstream stage (`convert.default`, via `capture.isaac`'s native output) to run
    against fake output afterward. `omni_capture` no longer routes through here at all -- see
    `_fake_run_native_isaac` below.
    """
    calls: list[list[str]] = []

    def fake_exec(env, cmd, *, log_path=None, workdir=None, environment=None, manager=None):
        calls.append(cmd)
        # split_mesh/add_motion run behind the Kit-bootstrap wrapper (T11 real-hardware finding),
        # so the vendored script's own name may be cmd[1] or cmd[2] -- check the whole cmd instead
        # of a fixed index.
        joined = " ".join(cmd)
        out_arg = cmd[cmd.index("--out") + 1]
        out_host = paths_mod.to_host(out_arg)
        if "split_mesh.py" in joined:
            out_host.parent.mkdir(parents=True, exist_ok=True)
            out_host.write_text("segmented-usd-placeholder")
        elif "add_motion.py" in joined:
            out_host.parent.mkdir(parents=True, exist_ok=True)
            out_host.write_text("animated-usd-placeholder")
            motion_groups = out_host.with_name(out_host.stem + "_motion_groups.json")
            motion_groups.write_text(json.dumps({"num_groups": 1, "segment_of_part": {}}))
        return ExecResult(exit_code=0, log_path=log_path)

    return fake_exec, calls


def _fake_run_native_isaac(calls: list[list[str]]):
    """Fake for `pipeline.stages.capture_isaac.run_native_isaac_script` -- `capture.isaac`'s
    native `omni_capture.py` call, now separate from `_fake_isaac_exec`'s container-exec fake
    (`split_mesh`/`add_motion` still route through the `isaac` container -- see
    `isaac_common.py`'s "Revised again 2026-07-16" note). Drops a real capture-shaped directory
    via `_make_capture_dir_at`, sized from the `--config` YAML's own `rig.n_cameras`/
    `capture.num_frames`, so `convert.default` can run against it for real exactly like
    `_fake_isaac_exec` already does for `split_mesh`/`add_motion`.
    """

    def fake(ctx, script_key, args, *, log_name=None):
        calls.append(args)
        out_host = Path(_flag_value(args, "out"))
        config_host = Path(_flag_value(args, "config"))
        doc = yaml.safe_load(config_host.read_text()) if config_host.is_file() else {}
        doc = doc or {}
        n_cams = int(doc.get("rig", {}).get("n_cameras", 3))
        n_frames = int(doc.get("capture", {}).get("num_frames", 2))
        out_host.mkdir(parents=True, exist_ok=True)
        _make_capture_dir_at(out_host, n_cams=n_cams, n_frames=n_frames)

    return fake


NAMES = ["prep_split.default", "prep_motion.default", "capture.isaac", "convert.default"]


def test_prep_to_convert_chain_runs_end_to_end_via_run_dag(tmp_path, monkeypatch):
    import pipeline.containers as containers_mod
    import pipeline.paths as paths_mod
    import pipeline.stages.capture_isaac as capture_isaac_mod
    from pipeline.artifacts import create_run, update_manifest
    from pipeline.dag import run_dag

    repo_root = _setup_roots(tmp_path, monkeypatch)
    runs_root = repo_root / "runs"
    fake_exec, calls = _fake_isaac_exec(paths_mod)
    monkeypatch.setattr(containers_mod, "exec_in_container", fake_exec)
    native_calls: list[list[str]] = []
    monkeypatch.setattr(
        capture_isaac_mod, "run_native_isaac_script", _fake_run_native_isaac(native_calls)
    )

    cfg = PipelineConfig(name="t11").model_dump()
    stage_configs = {name: _stage_config_for(name, cfg) for name in NAMES}
    raw_mesh_path = tmp_path / "assets" / "CONJUNTO_BOMBAS.usd"
    external = {
        "raw_mesh": Artifact(name="raw_mesh", kind="usd", path=str(raw_mesh_path), producing_stage="external")
    }

    def _seed(run_id: str):
        create_run(run_id, "t11", cfg, stage_names=NAMES, runs_root=runs_root)
        update_manifest(run_id, lambda m: m.artifacts.update(external), runs_root=runs_root)

    _seed("run1")
    m1 = run_dag("run1", NAMES, cfg, preset="t11", stage_configs=stage_configs, runs_root=runs_root)

    assert m1.status == "success"
    assert [m1.stages[n].status for n in NAMES] == ["success", "success", "success", "success"]
    assert len(calls) == 2  # split_mesh, add_motion (container; convert runs in-process, no exec)
    assert len(native_calls) == 1  # omni_capture (native)

    scene_dir = Path(m1.artifacts["scene"].path)
    assert scene_dir.is_dir()
    assert (scene_dir / "sparse_" / "cameras.bin").is_file()
    assert (scene_dir / "points3D_multipleview.ply").is_file()

    # A fresh run_id, unchanged config -> every stage hits the cross-run cache, zero new execs.
    _seed("run2")
    m2 = run_dag("run2", NAMES, cfg, preset="t11", stage_configs=stage_configs, runs_root=runs_root)
    assert [m2.stages[n].status for n in NAMES] == ["skipped", "skipped", "skipped", "skipped"]
    assert len(calls) == 2  # unchanged
    assert len(native_calls) == 1  # unchanged
    assert m2.artifacts["scene"].path == m1.artifacts["scene"].path  # reused, not reconverted

    # Changing prep_split's group re-runs it and everything downstream whose *input* content hash
    # actually changes: segmented_mesh/animated_mesh are `usd`-kind (single-file) artifacts, so
    # their content hash changes and prep_motion/capture.isaac correctly re-run too. convert.default
    # stays cached, though: its only input, `capture`, is a `dataset`-kind artifact (a directory),
    # and `pipeline.artifacts.hashing.hash_path` only ever hashes files, never a directory tree
    # (see that module's docstring: "the caller's decision to hash per-file or leave content_hash
    # as None") -- capture's `content_hash` is `None` both times, so convert's cache key input for
    # it is the same empty string regardless of what actually changed inside the directory. A
    # real, pre-existing T03/T05 cache-granularity limitation this task's fake exec happens to
    # expose for the first time (nothing upstream of convert.default was ever a *directory*
    # artifact before T11), not something T11 fixes -- documented here, same as T07/T09's own
    # "found, not fixed" notes.
    import copy

    cfg2 = copy.deepcopy(cfg)
    cfg2["prep_split"]["group"] = "OTHER_GROUP"
    stage_configs2 = {name: _stage_config_for(name, cfg2) for name in NAMES}
    _seed("run3")
    m3 = run_dag("run3", NAMES, cfg2, preset="t11", stage_configs=stage_configs2, runs_root=runs_root)
    assert [m3.stages[n].status for n in NAMES] == ["success", "success", "success", "skipped"]
    assert len(calls) == 4  # two more real execs (split_mesh/add_motion re-ran, container)
    assert len(native_calls) == 2  # one more (capture.isaac re-ran, native)


def test_api_run_pipeline_seeds_external_artifacts_before_run_dag(tmp_path, monkeypatch):
    """The real gap T11 found in `pipeline.api.run_pipeline` (see that function's docstring):
    previously there was no way to satisfy a fresh auto-planned run's external inputs at all.
    """
    import pipeline.containers as containers_mod
    import pipeline.paths as paths_mod
    import pipeline.stages.capture_isaac as capture_isaac_mod

    _setup_roots(tmp_path, monkeypatch)
    fake_exec, calls = _fake_isaac_exec(paths_mod)
    monkeypatch.setattr(containers_mod, "exec_in_container", fake_exec)
    native_calls: list[list[str]] = []
    monkeypatch.setattr(
        capture_isaac_mod, "run_native_isaac_script", _fake_run_native_isaac(native_calls)
    )
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "repo" / "runs"))

    from pipeline.api import run_pipeline
    from pipeline.artifacts import get_manifest

    raw_mesh_path = tmp_path / "assets" / "CONJUNTO_BOMBAS.usd"
    gt_path = tmp_path / "assets" / "gt_segmentation.npz"
    # `only` (below) narrows which stages *execute*, but run_dag's external-input check (T05,
    # pipeline.dag.graph.external_inputs) still runs over the *whole* auto-planned DAG (every
    # non-test role, T11's own `_auto_stage_plan`), not just the `only` subset -- so
    # `seg_eval.default`'s `gt_segmentation` still needs seeding here even though this test never
    # actually runs `seg_eval.default`. `only` and "which external inputs must be pre-seeded" are
    # deliberately independent checks (run_dag's own docstring: all three structural errors are
    # raised "before touching the manifest", i.e. before `only` gets a chance to narrow anything).
    run_id = run_pipeline(
        "base",
        external_artifacts={
            "raw_mesh": Artifact(name="raw_mesh", kind="usd", path=str(raw_mesh_path), producing_stage="external"),
            "gt_segmentation": Artifact(
                name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
            ),
        },
        only=["prep_split.default", "prep_motion.default", "capture.isaac"],
    )

    manifest = get_manifest(run_id)
    assert manifest.stages["prep_split.default"].status == "success"
    assert manifest.stages["prep_motion.default"].status == "success"
    assert manifest.stages["capture.isaac"].status == "success"
    assert len(calls) == 2
    assert len(native_calls) == 1
