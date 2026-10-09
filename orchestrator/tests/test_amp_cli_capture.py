"""T26 Phase C: `amp capture` command parsing, validation, and `verify` on a fixture.

CliRunner-based, sandbox-safe: no pxr/trimesh/imageio/Isaac — the heavy paths (split/motion/
gen-grid/preview execution, capture run's native subprocess) are only exercised through
arg-validation and the missing-launcher error path; `verify` runs fully against a synthetic
capture-dir fixture.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from amp.cli import app

runner = CliRunner()


def _make_capture(root, n_cams=2, n_frames=3, gt_json=True, status=True):
    """Synthetic omni_capture output: camNN/rgb/rgb_XXXXX.png + cameras_gt.json + status.json."""
    cap = root / "capture_test"
    for i in range(1, n_cams + 1):
        rgb_dir = cap / f"cam{i:02d}" / "rgb"
        rgb_dir.mkdir(parents=True)
        for f in range(1, n_frames + 1):
            (rgb_dir / f"rgb_{f:05d}.png").touch()
    if gt_json:
        (cap / "cameras_gt.json").write_text("{}", encoding="utf-8")
    if status:
        (cap / "status.json").write_text(json.dumps({
            "status": "completed", "overall_progress_pct": 100.0,
            "current_frame": n_frames, "total_frames": n_frames,
            "current_chunk": 1, "total_chunks": 1,
        }), encoding="utf-8")
    return cap


# --- amp capture verify ------------------------------------------------------------


def test_capture_verify_ok(tmp_path):
    cap = _make_capture(tmp_path, n_cams=2, n_frames=3)
    result = runner.invoke(app, ["capture", "verify", str(cap), "--cams", "2", "--frames", "3"])
    assert result.exit_code == 0, result.output
    assert "cam01: 3" in result.output
    assert "cam02: 3" in result.output
    assert "verify: OK" in result.output


def test_capture_verify_frame_mismatch_fails(tmp_path):
    cap = _make_capture(tmp_path, n_cams=2, n_frames=3)
    # Drop one frame from cam02.
    (cap / "cam02" / "rgb" / "rgb_00003.png").unlink()
    result = runner.invoke(app, ["capture", "verify", str(cap), "--cams", "2", "--frames", "3"])
    assert result.exit_code == 1
    assert "cam02: 2" in result.output
    assert "MISMATCH" in result.output


def test_capture_verify_cam_count_mismatch_fails(tmp_path):
    cap = _make_capture(tmp_path, n_cams=2, n_frames=3)
    result = runner.invoke(app, ["capture", "verify", str(cap), "--cams", "10"])
    assert result.exit_code == 1
    assert "MISMATCH" in result.output


def test_capture_verify_missing_gt_json_fails(tmp_path):
    cap = _make_capture(tmp_path, gt_json=False)
    result = runner.invoke(app, ["capture", "verify", str(cap)])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_capture_verify_expectations_from_config(tmp_path):
    cap = _make_capture(tmp_path, n_cams=2, n_frames=3)
    cfg = tmp_path / "capture_config.yaml"
    cfg.write_text("rig:\n  n_cameras: 2\ncapture:\n  num_frames: 3\n", encoding="utf-8")
    result = runner.invoke(app, ["capture", "verify", str(cap), "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    # Explicit flags win over the YAML.
    result = runner.invoke(app, ["capture", "verify", str(cap), "--config", str(cfg), "--cams", "9"])
    assert result.exit_code == 1


def test_capture_verify_missing_dir(tmp_path):
    result = runner.invoke(app, ["capture", "verify", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


# --- arg validation for the execution commands --------------------------------------


def test_capture_split_requires_in_out():
    result = runner.invoke(app, ["capture", "split"])
    assert result.exit_code == 2  # missing required options


def test_capture_motion_requires_in_out():
    result = runner.invoke(app, ["capture", "motion"])
    assert result.exit_code == 2


def test_capture_convert_missing_capture_dir(tmp_path):
    result = runner.invoke(app, ["capture", "convert", "--capture", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "--capture" in result.output


def test_capture_preview_missing_capture_dir(tmp_path):
    result = runner.invoke(app, ["capture", "preview", "--capture", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "--capture" in result.output


def test_capture_run_missing_config(tmp_path):
    result = runner.invoke(app, ["capture", "run", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert "--config" in result.output


def test_capture_run_bad_headless(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("app: {headless: true}\n", encoding="utf-8")
    result = runner.invoke(app, ["capture", "run", "--config", str(cfg), "--headless", "2"])
    assert result.exit_code != 0
    assert "--headless" in result.output


def test_capture_run_missing_isaac_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ISAAC_NATIVE_PYTHON", str(tmp_path / "no_such_python.bat"))
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("app: {headless: true}\n", encoding="utf-8")
    result = runner.invoke(app, ["capture", "run", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "PIPELINE_ISAAC_NATIVE_PYTHON" in result.output
    # The launcher check fails before any run directory is created.
    assert not (tmp_path / "runs").exists()


# --- every capture subcommand's help works --------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["split", "motion", "gen-grid", "run", "convert", "preview", "verify"]
)
def test_capture_subcommand_help(cmd):
    result = runner.invoke(app, ["capture", cmd, "--help"])
    assert result.exit_code == 0, result.output
