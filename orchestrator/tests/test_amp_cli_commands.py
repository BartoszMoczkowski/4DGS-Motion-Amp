"""T26 Phase A: `amp train` / `amp render` / `amp orch` command parsing, validation, dry-run.

CliRunner-based, in-process. No Docker/GPU/subprocess launches: the execution paths are only
exercised up to the point where they would call `run_pipeline`/`run_stage`/spawn a process —
`--dry-run` is the test surface for everything that would touch hardware, and the error paths
(unknown preset/run, bad amp factors) fail before any execution.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from amp.cli import app

runner = CliRunner()

FACTORS_8 = ["--amp-factors", "2 -1 -1 -1 -1 -1 -1 -1"]


@pytest.fixture()
def fresh_run(tmp_path, monkeypatch):
    """An existing (empty-plan) run for `amp render` to resolve against."""
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    return api.run_pipeline("base")


# --- amp train -----------------------------------------------------------------


def test_train_unknown_preset_fails():
    result = runner.invoke(app, ["train", "definitely-not-a-preset"])
    assert result.exit_code != 0
    assert "unknown preset" in result.output


def test_train_dry_run_prints_stage_slice_without_executing(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    result = runner.invoke(app, ["train", "base", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "train.default" in result.output
    assert "nothing executed" in result.output
    # No run directory was created.
    assert not list(tmp_path.iterdir())


def test_train_scene_must_exist_and_be_confined(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    result = runner.invoke(app, ["train", "base", "--scene", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "--scene" in result.output


def test_train_scene_dry_run_ok(tmp_path, monkeypatch):
    """A scene dir under the (patched) runs root passes confinement."""
    runs_root = tmp_path / "runs"
    scene = runs_root / "seed_scene"
    scene.mkdir(parents=True)
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))
    result = runner.invoke(app, ["train", "base", "--scene", str(scene), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "train.default" in result.output


# --- amp render (plain) ----------------------------------------------------------


def test_render_unknown_run_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    result = runner.invoke(app, ["render", "no-such-run"])
    assert result.exit_code == 1
    assert "unknown run" in result.output


def test_render_plain_dry_run(fresh_run):
    result = runner.invoke(app, ["render", fresh_run, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "render.default" in result.output
    assert "nothing executed" in result.output


def test_render_amp_only_flags_rejected_without_factors(fresh_run):
    result = runner.invoke(app, ["render", fresh_run, "--method", "eulerian_abs"])
    assert result.exit_code != 0
    assert "--amp-factors" in result.output
    result = runner.invoke(app, ["render", fresh_run, "--video", "x.mp4"])
    assert result.exit_code != 0
    assert "--amp-factors" in result.output


def test_render_stage_failure_exits_nonzero_with_stage_name(fresh_run, monkeypatch):
    """api.StageFailedError (T26 follow-up: a failed stage recorded in the manifest) must surface
    as a clean red error + exit 1, never as "finished" / exit 0."""
    from pipeline import api

    def _fail(run_id, stage, **kwargs):
        raise api.StageFailedError(
            f"run {run_id!r}: stage(s) failed: {stage}: simulated failure (log: /x/y.log)"
        )

    monkeypatch.setattr(api, "run_stage", _fail)
    result = runner.invoke(app, ["render", fresh_run])
    assert result.exit_code == 1
    assert "render.default failed" in result.output
    assert "Traceback" not in result.output


# --- amp render (amplified) ------------------------------------------------------


def test_render_amp_factors_wrong_count(fresh_run):
    result = runner.invoke(app, ["render", fresh_run, "--amp-factors", "2 -1 -1"])
    assert result.exit_code != 0
    assert "exactly 8" in result.output


def test_render_amp_factors_illegal_negative(fresh_run):
    result = runner.invoke(
        # Quoted list starting with a negative needs the `=` spelling (a bare "-2 ..." token
        # looks like an option to click).
        app, ["render", fresh_run, "--amp-factors=-2 -1 -1 -1 -1 -1 -1 -1"]
    )
    assert result.exit_code != 0
    # (Rich-wraps long messages at 80 cols, so assert on a short substring.)
    assert "skip sentinel" in result.output


def test_render_amp_quoted_factor_list(fresh_run):
    """The quoted space-separated spelling (render_amp CLI habit) works too."""
    result = runner.invoke(
        app, ["render", fresh_run, "--amp-factors", "2 -1 -1 -1 -1 -1 -1 -1", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "amp.default" in result.output


def test_render_amp_freqs_require_each_other(fresh_run):
    result = runner.invoke(app, ["render", fresh_run, *FACTORS_8, "--freq-low", "0.1"])
    assert result.exit_code != 0
    assert "--freq-low and --freq-high must be given together" in result.output


def test_render_amp_freqs_bad_length(fresh_run):
    result = runner.invoke(
        app, ["render", fresh_run, *FACTORS_8, "--freq-low", "0.1 0.2", "--freq-high", "1.0 0.9"]
    )
    assert result.exit_code != 0
    assert "length 1 (broadcast) or 8" in result.output


def test_render_amp_unknown_method(fresh_run):
    result = runner.invoke(app, ["render", fresh_run, *FACTORS_8, "--method", "wat"])
    assert result.exit_code != 0
    assert "--method" in result.output


def test_render_amp_dry_run_channel_mapping(fresh_run):
    """--amp-factors positional slots map onto amp.channels in render_amp's slot order
    ([means3D, means2D, scales, rotations, ...] -> [pos3d, pos2d, scale, rotation, ...])."""
    result = runner.invoke(
        app,
        [
            "render", fresh_run,
            "--amp-factors", "2 3 4 5 -1 -1 -1 -1",
            "--freq-low", "0.25",
            "--freq-high", "0.75",
            "--method", "eulerian_abs",
            "--video", "out.mp4",
            "--low-vram",
            "--frozen-cam",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "amp.default" in result.output
    import json

    blob = result.output.split("resolved config slice for amp.default:", 1)[1]
    cfg = json.loads(blob.split("[dry-run] nothing executed.", 1)[0])
    channels = cfg["channels"]
    # Slot 2 is *scales* in render_amp's positional order -> "scale" channel.
    assert channels["pos3d"]["factor"] == 2.0
    assert channels["pos2d"]["factor"] == 3.0
    assert channels["scale"]["factor"] == 4.0
    assert channels["rotation"]["factor"] == 5.0
    assert channels["opacity"]["factor"] == -1.0
    # Broadcast freq cutoffs landed on every channel.
    assert channels["pos3d"]["freq_low"] == 0.25
    assert channels["pos3d"]["freq_high"] == 0.75
    assert channels["cov3D"]["freq_low"] == 0.25
    assert cfg["method"] == "eulerian_abs"
    assert cfg["video_path"] == "out.mp4"
    assert cfg["low_vram_mode"] is True
    assert cfg["frozen_cam"] is True


# --- amp orch --------------------------------------------------------------------


def test_orch_mcp_refuses_without_token(monkeypatch):
    monkeypatch.delenv("PIPELINE_MCP_TOKEN", raising=False)
    result = runner.invoke(app, ["orch", "mcp"])
    assert result.exit_code == 2
    assert "PIPELINE_MCP_TOKEN" in result.output
    assert "token_urlsafe" in result.output


def test_orch_ui_rejects_unknown_app():
    result = runner.invoke(app, ["orch", "ui", "--app", "bogus"])
    assert result.exit_code != 0
    assert "--app" in result.output
