"""Tests for T10 (wrap Option-A segmentation): `segment.mbs`.

Same testing strategy as T09's own `test_stages_cuda.py` (see that file's module docstring): no
GPU/Docker/torch/MBS `ext/` ops available in this sandbox, so what's verified here is everything
*above* the real container-exec call — CLI-argument construction, the checkpoint relative-path
resolution `pipeline/stages/segment_mbs.py` adds, and the config-switch contract this task exists
to prove (`segment.rigid` <-> `segment.mbs` share the same inputs/outputs/artifact shape, and
`pipeline.api._auto_stage_plan` picks the right one from `SegmentConfig.impl`) — against a fake
`pipeline.containers.exec_in_container`, exactly like T09's own tests fake it.

Real GPU/container execution (building the MBS `ext/` CUDA ops, loading an actual checkpoint,
running MotNet) needs Bartosz's machine — see `planning/WINDOWS_SETUP.md`'s "Option-A
segmentation (MBS) setup" step and `planning/tasks/T10-wrap-option-a-segmentation.md`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pipeline.api import _auto_stage_plan, _stage_config_for
from pipeline.artifacts import Artifact
from pipeline.config import validate_config
from pipeline.config.models import PipelineConfig
from pipeline.containers.manager import ExecResult
from pipeline.stages.base import StageContext
from pipeline.stages.segment_mbs import SegmentMbsStage


# --- shared fixtures/helpers (mirrors test_stages_cuda.py's own) ---------------------------------


def _setup_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PIPELINE_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path / "assets"))
    return repo_root


class _FakeContainers:
    def __init__(self, exit_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.exit_code = exit_code

    def exec_in_container(self, env, cmd, *, log_path=None, workdir=None, environment=None):
        self.calls.append(
            {"env": env, "cmd": cmd, "log_path": log_path, "workdir": workdir, "environment": environment}
        )
        return ExecResult(exit_code=self.exit_code, log_path=log_path)


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


def _traj_input(repo_root: Path, run_id: str = "r1") -> dict:
    traj_path = repo_root / "runs" / run_id / "trajectories.npz"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"")  # never actually read by the stage (only path-translated)
    return {
        "trajectories": Artifact(
            name="trajectories", kind="npz", path=str(traj_path), producing_stage="seg_extract.default"
        )
    }


# --- CLI construction ----------------------------------------------------------------------------


def test_segment_mbs_stage_builds_expected_cli_and_out_path(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    ckpt_path = repo_root / "submodules" / "multibody-sync-4dgs" / "ckpt" / "mbs_full.pth.tar"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(b"")

    cfg = {
        **PipelineConfig(segment={"impl": "mbs", "mbs": {"checkpoint": str(ckpt_path)}}).model_dump()["segment"]["mbs"],
        "n_points": 1234,
        "n_views": 3,
    }
    inputs = _traj_input(repo_root)
    ctx, fake = _ctx(repo_root, stage_name="segment.mbs", config=cfg, inputs=inputs)

    result = SegmentMbsStage().run(ctx)

    assert set(result) == {"segmentation"}
    assert Path(result["segmentation"].path) == ctx.run_dir / "segmentation.npz"
    assert result["segmentation"].metadata == {"n_points": 1234, "n_views": 3}

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["env"] == "cuda"
    cmd = call["cmd"]
    assert cmd[0] == "python"
    assert cmd[1].endswith("orchestrator/pipeline/vendored/cuda/mbs_infer.py")
    assert _flag_value(cmd, "n-points") == "1234"
    assert _flag_value(cmd, "n-views") == "3"
    assert _flag_value(cmd, "n-sub") == "256"
    assert _flag_value(cmd, "opacity-thresh") == "0.1"
    assert _flag_value(cmd, "alpha") == "0.05"
    assert _flag_value(cmd, "seed") == "0"
    assert "--out" in cmd
    assert str(repo_root) not in " ".join(cmd)  # host paths never leak into the container-side cmd

    # this stage never needs the train/render/seg_extract/amp bridge file.
    assert "--configs" not in cmd


def test_segment_mbs_does_not_get_the_cuda_bridge_merge():
    cfg = PipelineConfig(segment={"impl": "mbs", "mbs": {"checkpoint": "x"}}).model_dump()
    stage_cfg = _stage_config_for("segment.mbs", cfg)
    assert "_bridge" not in stage_cfg
    assert stage_cfg["checkpoint"] == "x"


# --- checkpoint relative-path resolution -----------------------------------------------------


def test_segment_mbs_resolves_checkpoint_relative_to_repo_root(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    rel = "submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar"
    (repo_root / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / Path(rel)).write_bytes(b"")

    cfg = {**PipelineConfig(segment={"impl": "mbs", "mbs": {"checkpoint": rel}}).model_dump()["segment"]["mbs"]}
    inputs = _traj_input(repo_root)
    ctx, fake = _ctx(repo_root, stage_name="segment.mbs", config=cfg, inputs=inputs)

    SegmentMbsStage().run(ctx)

    checkpoint_container = _flag_value(fake.calls[0]["cmd"], "checkpoint")
    assert checkpoint_container == "/workspace/submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar"


def test_segment_mbs_accepts_absolute_checkpoint_under_repo_root(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    ckpt_path = repo_root / "somewhere" / "checkpoint.pth.tar"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(b"")

    cfg = {**PipelineConfig(segment={"impl": "mbs", "mbs": {"checkpoint": str(ckpt_path)}}).model_dump()["segment"]["mbs"]}
    inputs = _traj_input(repo_root)
    ctx, fake = _ctx(repo_root, stage_name="segment.mbs", config=cfg, inputs=inputs)

    SegmentMbsStage().run(ctx)

    checkpoint_container = _flag_value(fake.calls[0]["cmd"], "checkpoint")
    assert checkpoint_container == "/workspace/somewhere/checkpoint.pth.tar"


def test_segment_mbs_raises_before_exec_if_checkpoint_empty(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    inputs = _traj_input(repo_root)
    ctx, fake = _ctx(repo_root, stage_name="segment.mbs", config={"checkpoint": ""}, inputs=inputs)

    with pytest.raises(ValueError):
        SegmentMbsStage().run(ctx)

    assert fake.calls == []  # failed fast, before ever touching the container


# --- a non-zero exit is a stage failure, not a silent success -----------------------------------


def test_segment_mbs_raises_on_nonzero_exit(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    inputs = _traj_input(repo_root)
    cfg = {"checkpoint": "submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar"}
    ctx, fake = _ctx(repo_root, stage_name="segment.mbs", config=cfg, inputs=inputs)
    fake.exit_code = 1

    with pytest.raises(Exception):
        SegmentMbsStage().run(ctx)


# --- registry sanity / the config-switch contract T10 exists to prove ---------------------------


def test_segment_mbs_is_registered_under_the_expected_name():
    from pipeline.stages import get_stage, list_stages

    assert "segment.mbs" in set(list_stages())
    cls = get_stage("segment.mbs")
    assert cls.environment == "cuda"
    assert cls.resources.needs_gpu is True


def test_segment_impls_share_the_same_artifact_contract():
    """The whole point of T10: `segment.rigid` <-> `segment.mbs` is a config switch, not a DAG
    change — both must declare the identical inputs/outputs tuple."""
    from pipeline.stages import get_stage

    rigid = get_stage("segment.rigid")
    mbs = get_stage("segment.mbs")
    assert rigid.inputs == mbs.inputs == ("trajectories",)
    assert rigid.outputs == mbs.outputs == ("segmentation",)


def test_auto_stage_plan_selects_rigid_for_base_and_mbs_for_pump01_segA():
    base_resolved = validate_config("base").model_dump()
    assert "segment.rigid" in _auto_stage_plan(base_resolved)
    assert "segment.mbs" not in _auto_stage_plan(base_resolved)

    mbs_resolved = validate_config("pump01_segA").model_dump()
    assert "segment.mbs" in _auto_stage_plan(mbs_resolved)
    assert "segment.rigid" not in _auto_stage_plan(mbs_resolved)


def test_pump01_segA_preset_resolves_and_validates():
    cfg = validate_config("pump01_segA")
    assert cfg.segment.impl == "mbs"
    assert cfg.segment.mbs.checkpoint  # non-empty, per SegmentConfig._check_impl_ready
