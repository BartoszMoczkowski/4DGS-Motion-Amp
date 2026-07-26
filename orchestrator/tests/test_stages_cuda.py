"""Tests for T09 (wrap CUDA stages): `train.default` / `render.default` / `seg_extract.default`
/ `amp.default`.

None of these can run for real without a GPU + Docker Desktop (same story as T08's
`ContainerManager` — see `pipeline/containers/MANUAL_CHECKLIST.md`). What *is* verifiable here in
the sandbox, mirroring T08's own "fake the one thing that needs a real daemon/GPU" approach: the
CLI-argument construction each stage builds, the bridge-file content
(`pipeline.config.bridge`), `pipeline.api._stage_config_for`'s new `"_bridge"` merging (T09), and
the `train -> render` dependency + caching behavior through `run_dag`, against a fake
`pipeline.containers.exec_in_container` rather than a fake Docker SDK (T08 already covers the
Docker-talking layer itself; this task's own new logic is everything *above* that call).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pipeline.api import _stage_config_for
from pipeline.artifacts import Artifact
from pipeline.config.bridge import render_bridge_source
from pipeline.config.models import PipelineConfig
from pipeline.containers.manager import ExecResult
from pipeline.stages.amp import AmpFactorNotIntegerError, AmpStage
from pipeline.stages.base import StageContext
from pipeline.stages.render import RenderStage
from pipeline.stages.seg_extract import SegExtractStage
from pipeline.stages.train import TrainStage


# --- shared fixtures/helpers ---------------------------------------------------------------


def _setup_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`PIPELINE_REPO_ROOT`/`PIPELINE_ASSETS_ROOT` pointed at a sandboxed tmp tree, mirroring
    `test_paths.py`'s own override pattern — every path these stages build must resolve under
    one of the two roots or `pipeline.paths.to_container` raises."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PIPELINE_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path / "assets"))
    return repo_root


class _FakeContainers:
    """Stands in for `ctx.containers` (the real `pipeline.containers` module, since T09's
    scheduler wiring) — records every `exec_in_container` call and returns a canned result,
    exactly like T08's `test_containers.py` fakes the Docker SDK one layer down.

    ``write_checkpoint`` (default ``True``): a successful ``train.py`` call also stubs a
    ``point_cloud/iteration_1/point_cloud.ply`` under the exec'd ``--model_path`` -- mirrors what a
    real, correctly-behaving train run writes, since `TrainStage.run` now checks for it (T11 fix,
    2026-07-18, see ``pipeline/stages/train.py``'s docstring). Set ``False`` to exercise the
    missing-checkpoint error path instead.
    """

    def __init__(self, exit_code: int = 0, *, write_checkpoint: bool = True) -> None:
        self.calls: list[dict] = []
        self.exit_code = exit_code
        self.write_checkpoint = write_checkpoint

    def exec_in_container(self, env, cmd, *, log_path=None, workdir=None, environment=None):
        self.calls.append(
            {"env": env, "cmd": cmd, "log_path": log_path, "workdir": workdir, "environment": environment}
        )
        if self.exit_code == 0 and self.write_checkpoint and any(str(c).endswith("train.py") for c in cmd):
            import pipeline.paths as paths_mod

            model_host = paths_mod.to_host(_flag_value(cmd, "model_path"))
            ckpt_dir = model_host / "point_cloud" / "iteration_1"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "point_cloud.ply").write_text("stub")
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


# --- pipeline.config.bridge ------------------------------------------------------------------


def test_bridge_source_has_all_four_groups_and_excludes_source_and_model_path():
    cfg = PipelineConfig().model_dump()
    cfg["model"]["source_path"] = "/should/not/appear/in/bridge"
    cfg["model"]["model_path"] = "/also/should/not/appear"

    src = render_bridge_source(cfg)
    namespace: dict = {}
    exec(compile(src, "<bridge>", "exec"), namespace)  # same trust model as runpy/mmengine

    assert set(namespace["ModelParams"]) == set(cfg["model"]) - {"source_path", "model_path"}
    assert namespace["PipelineParams"] == cfg["pipeline_params"]
    assert namespace["ModelHiddenParams"] == cfg["hidden"]
    assert namespace["OptimizationParams"] == cfg["optim"]
    assert "/should/not/appear/in/bridge" not in src
    assert "/also/should/not/appear" not in src


# --- pipeline.api._stage_config_for: T09's "_bridge" merge ------------------------------------


def test_stage_config_for_adds_bridge_only_for_cuda_roles():
    cfg = PipelineConfig().model_dump()

    for role in ("train", "render", "seg_extract", "amp"):
        stage_cfg = _stage_config_for(f"{role}.default", cfg)
        assert set(stage_cfg["_bridge"]) == {"model", "pipeline_params", "hidden", "optim"}

    convert_cfg = _stage_config_for("convert.default", cfg)
    assert "_bridge" not in convert_cfg
    segment_cfg = _stage_config_for("segment.rigid", cfg)
    assert "_bridge" not in segment_cfg


def test_stage_config_for_train_cache_key_input_unaffected_by_unrelated_section():
    cfg = PipelineConfig().model_dump()
    cfg2 = PipelineConfig().model_dump()
    cfg2["segment"]["rigid"]["k"] = 999  # unrelated to train

    assert _stage_config_for("train.default", cfg) == _stage_config_for("train.default", cfg2)


def test_stage_config_for_train_cache_key_input_changes_with_hidden_params():
    cfg = PipelineConfig().model_dump()
    cfg2 = PipelineConfig().model_dump()
    cfg2["hidden"]["net_width"] = 999

    assert _stage_config_for("train.default", cfg) != _stage_config_for("train.default", cfg2)


# --- train.default -----------------------------------------------------------------------------


def test_train_stage_builds_expected_cli_and_bridge(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    scene_dir = repo_root / "runs" / "r1" / "scene"
    scene_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    train_cfg = {
        **_stage_config_for("train.default", cfg),
        "port": 7000,
        "expname": "myexp",
        "test_iterations": [10, 20],
        "checkpoint_iterations": [5],
    }
    inputs = {
        "scene": Artifact(name="scene", kind="dataset", path=str(scene_dir), producing_stage="convert.default")
    }
    ctx, fake = _ctx(repo_root, stage_name="train.default", config=train_cfg, inputs=inputs)

    result = TrainStage().run(ctx)

    assert set(result) == {"model"}
    model_host = Path(result["model"].path)
    assert model_host == ctx.run_dir / "train_out"
    assert model_host.is_dir()

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["env"] == "cuda"
    assert call["environment"] == {"PYTHONPATH": "/workspace/core"}
    cmd = call["cmd"]
    assert cmd[0] == "python"
    assert cmd[1].endswith("orchestrator/pipeline/vendored/cuda/train.py")
    assert _flag_value(cmd, "port") == "7000"
    assert _flag_value(cmd, "expname") == "myexp"
    assert "--test_iterations" in cmd and "10" in cmd and "20" in cmd
    assert "--checkpoint_iterations" in cmd and "5" in cmd
    assert str(scene_dir) not in " ".join(cmd)  # host path never leaks into a container-side cmd

    # bridge file: written under run_dir, excludes source_path/model_path.
    configs_container = _flag_value(cmd, "configs")
    import pipeline.paths as paths_mod

    bridge_host = paths_mod.to_host(configs_container)
    assert bridge_host.is_file()
    src = bridge_host.read_text()
    assert "ModelHiddenParams" in src
    assert "OptimizationParams" in src
    assert str(scene_dir) not in src


def test_train_stage_defaults_expname_to_run_id_when_unset(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    scene_dir = repo_root / "runs" / "r1" / "scene"
    scene_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    train_cfg = _stage_config_for("train.default", cfg)  # expname == "" by default
    inputs = {
        "scene": Artifact(name="scene", kind="dataset", path=str(scene_dir), producing_stage="convert.default")
    }
    ctx, fake = _ctx(repo_root, stage_name="train.default", config=train_cfg, inputs=inputs)

    TrainStage().run(ctx)

    assert _flag_value(fake.calls[0]["cmd"], "expname") == "r1"


# --- render.default ------------------------------------------------------------------------


def test_render_stage_builds_expected_cli_and_reuses_model_dir(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    model_dir = repo_root / "runs" / "r1" / "train_out"
    model_dir.mkdir(parents=True)
    (model_dir / "cfg_args").write_text("Namespace()")

    cfg = PipelineConfig().model_dump()
    render_cfg = {**_stage_config_for("render.default", cfg), "skip_train": True}
    inputs = {"model": Artifact(name="model", kind="model", path=str(model_dir), producing_stage="train.default")}
    ctx, fake = _ctx(repo_root, stage_name="render.default", config=render_cfg, inputs=inputs)

    result = RenderStage().run(ctx)

    assert Path(result["renders"].path) == model_dir
    cmd = fake.calls[0]["cmd"]
    assert cmd[1].endswith("orchestrator/pipeline/vendored/cuda/render.py")
    assert "--skip_train" in cmd
    assert "--source_path" not in cmd  # render.py reads it back out of the model's own cfg_args
    assert "--iteration" in cmd and _flag_value(cmd, "iteration") == "-1"


# --- seg_extract.default -------------------------------------------------------------------


def test_seg_extract_stage_builds_expected_cli_and_out_path(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    model_dir = repo_root / "runs" / "r1" / "train_out"
    model_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    seg_cfg = {**_stage_config_for("seg_extract.default", cfg), "n_times": 42}
    inputs = {"model": Artifact(name="model", kind="model", path=str(model_dir), producing_stage="train.default")}
    ctx, fake = _ctx(repo_root, stage_name="seg_extract.default", config=seg_cfg, inputs=inputs)

    result = SegExtractStage().run(ctx)

    assert Path(result["trajectories"].path) == ctx.run_dir / "trajectories.npz"
    cmd = fake.calls[0]["cmd"]
    assert cmd[1].endswith("orchestrator/pipeline/vendored/cuda/seg_extract.py")
    assert "--n-times" in cmd and cmd[cmd.index("--n-times") + 1] == "42"
    assert "--out" in cmd


# --- amp.default ---------------------------------------------------------------------------


def test_amp_stage_builds_expected_cli_and_video_path(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    model_dir = repo_root / "runs" / "r1" / "train_out"
    model_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    amp_cfg = {**_stage_config_for("amp.default", cfg)}
    amp_cfg["channels"] = {
        **amp_cfg["channels"],
        "pos3d": {"factor": 3.0, "freq_low": 0.1, "freq_high": 0.9},
    }
    inputs = {"model": Artifact(name="model", kind="model", path=str(model_dir), producing_stage="train.default")}
    ctx, fake = _ctx(repo_root, stage_name="amp.default", config=amp_cfg, inputs=inputs)

    result = AmpStage().run(ctx)

    assert Path(result["amp_video"].path) == model_dir / "video" / "render.mp4"
    cmd = fake.calls[0]["cmd"]
    assert cmd[1].endswith("orchestrator/pipeline/vendored/cuda/amp.py")
    assert "--amp_factors" in cmd
    factors_start = cmd.index("--amp_factors") + 1
    factors = cmd[factors_start : factors_start + 8]
    assert factors[0] == "3"  # pos3d is first in AMP_CHANNELS order
    assert all(f == "-1" for f in factors[1:])  # every other channel defaults to "don't amplify"


def test_amp_stage_rejects_non_integer_factor_before_execing(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    model_dir = repo_root / "runs" / "r1" / "train_out"
    model_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    amp_cfg = {**_stage_config_for("amp.default", cfg)}
    amp_cfg["channels"] = {
        **amp_cfg["channels"],
        "opacity": {"factor": 2.5, "freq_low": 0.0, "freq_high": 1.0},
    }
    inputs = {"model": Artifact(name="model", kind="model", path=str(model_dir), producing_stage="train.default")}
    ctx, fake = _ctx(repo_root, stage_name="amp.default", config=amp_cfg, inputs=inputs)

    with pytest.raises(AmpFactorNotIntegerError):
        AmpStage().run(ctx)

    assert fake.calls == []  # failed fast, before ever touching the container


# --- registry sanity -------------------------------------------------------------------------


def test_cuda_stages_are_registered_under_the_expected_names():
    from pipeline.stages import get_stage, list_stages

    assert {"train.default", "render.default", "seg_extract.default", "amp.default"} <= set(list_stages())
    for name in ("train.default", "render.default", "seg_extract.default", "amp.default"):
        assert get_stage(name).environment == "cuda"


# --- a non-zero exit is a stage failure, not a silent success -----------------------------------


def test_train_stage_raises_on_nonzero_exit(tmp_path, monkeypatch):
    repo_root = _setup_roots(tmp_path, monkeypatch)
    scene_dir = repo_root / "runs" / "r1" / "scene"
    scene_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    train_cfg = _stage_config_for("train.default", cfg)
    inputs = {
        "scene": Artifact(name="scene", kind="dataset", path=str(scene_dir), producing_stage="convert.default")
    }
    ctx, fake = _ctx(repo_root, stage_name="train.default", config=train_cfg, inputs=inputs)
    fake.exit_code = 1

    with pytest.raises(Exception):
        TrainStage().run(ctx)


def test_train_stage_raises_if_exit_zero_but_no_checkpoint_written(tmp_path, monkeypatch):
    """T11 real-hardware finding (2026-07-18): a `save_iterations` ordering bug in the vendored
    `train.py` (fixed separately, see its module docstring) let a run exit 0 having trained its
    full iteration count but never called `scene.save()` -- no `point_cloud/` ever got written, and
    `train.default` had no way to notice. This regression-tests the guard added for that: exit 0
    alone must not be enough to report success."""
    repo_root = _setup_roots(tmp_path, monkeypatch)
    scene_dir = repo_root / "runs" / "r1" / "scene"
    scene_dir.mkdir(parents=True)

    cfg = PipelineConfig().model_dump()
    train_cfg = _stage_config_for("train.default", cfg)
    inputs = {
        "scene": Artifact(name="scene", kind="dataset", path=str(scene_dir), producing_stage="convert.default")
    }
    ctx, fake = _ctx(repo_root, stage_name="train.default", config=train_cfg, inputs=inputs)
    fake.write_checkpoint = False  # exit 0, but no point_cloud/ -- the exact bug found for real

    with pytest.raises(Exception, match="point_cloud"):
        TrainStage().run(ctx)


# --- end-to-end through run_dag: train -> render, with cross-run caching -----------------------


def test_train_then_render_via_run_dag_with_caching(tmp_path, monkeypatch):
    import pipeline.containers as containers_mod
    import pipeline.paths as paths_mod
    from pipeline.artifacts import create_run, update_manifest
    from pipeline.dag import run_dag

    repo_root = _setup_roots(tmp_path, monkeypatch)
    scene_dir = repo_root / "scene"
    scene_dir.mkdir(parents=True)
    runs_root = repo_root / "runs"

    calls: list[list[str]] = []

    def fake_exec(env, cmd, *, log_path=None, workdir=None, environment=None, manager=None):
        calls.append(cmd)
        if "--model_path" in cmd:
            model_host = paths_mod.to_host(cmd[cmd.index("--model_path") + 1])
            Path(model_host).mkdir(parents=True, exist_ok=True)
            (Path(model_host) / "cfg_args").write_text("Namespace()")
            if any(str(c).endswith("train.py") for c in cmd):
                # TrainStage now checks for a real checkpoint (T11 fix, 2026-07-18) -- stub one so
                # this fake still represents a *successful* train run, same as _FakeContainers
                # above.
                ckpt_dir = Path(model_host) / "point_cloud" / "iteration_1"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ckpt_dir / "point_cloud.ply").write_text("stub")
        return ExecResult(exit_code=0, log_path=log_path)

    monkeypatch.setattr(containers_mod, "exec_in_container", fake_exec)

    cfg = PipelineConfig(name="t09").model_dump()
    stage_names = ["train.default", "render.default"]
    stage_configs = {name: _stage_config_for(name, cfg) for name in stage_names}
    external = {
        "scene": Artifact(name="scene", kind="dataset", path=str(scene_dir), producing_stage="external")
    }

    def _seed(run_id: str):
        create_run(run_id, "t09", cfg, stage_names=stage_names, runs_root=runs_root)
        update_manifest(run_id, lambda m: m.artifacts.update(external), runs_root=runs_root)

    _seed("run1")
    m1 = run_dag("run1", stage_names, cfg, preset="t09", stage_configs=stage_configs, runs_root=runs_root)
    assert [m1.stages[n].status for n in stage_names] == ["success", "success"]
    assert len(calls) == 2  # train, then render

    # A fresh run_id, unchanged config -> both stages hit the cross-run cache, no new exec calls.
    _seed("run2")
    m2 = run_dag("run2", stage_names, cfg, preset="t09", stage_configs=stage_configs, runs_root=runs_root)
    assert [m2.stages[n].status for n in stage_names] == ["skipped", "skipped"]
    assert len(calls) == 2  # unchanged

    assert m2.artifacts["model"].path == m1.artifacts["model"].path  # reused, not retrained
