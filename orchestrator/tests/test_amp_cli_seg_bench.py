"""T26 Phase B: `amp seg` + `amp bench` command parsing, validation, dry-run.

CliRunner-based, in-process, sandbox-safe: seg commands are exercised only through their
validation and --dry-run paths (which resolve the run's manifest and print the stage config
slice without executing); `amp bench renders` is only exercised through flag validation,
which happens before the torch/render_amp-heavy `run_renders_auto` import (that module is
not importable off-Linux and must never load in tests).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from amp.cli import app

runner = CliRunner()


@pytest.fixture()
def fresh_run(tmp_path, monkeypatch):
    """An existing (empty-plan) run for the seg commands to resolve against."""
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    return api.run_pipeline("base")


def _dry_run_slice(output: str, stage: str) -> dict:
    blob = output.split(f"resolved config slice for {stage}:", 1)[1]
    return json.loads(blob.split("[dry-run] nothing executed.", 1)[0])


# --- amp seg extract -------------------------------------------------------------


def test_seg_extract_unknown_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    result = runner.invoke(app, ["seg", "extract", "no-such-run"])
    assert result.exit_code == 1
    assert "unknown run" in result.output


def test_seg_extract_dry_run_with_n_times(fresh_run):
    result = runner.invoke(app, ["seg", "extract", fresh_run, "--n-times", "120", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "seg_extract.default" in result.output
    assert _dry_run_slice(result.output, "seg_extract.default")["n_times"] == 120


def test_seg_extract_dry_run_default_config(fresh_run):
    result = runner.invoke(app, ["seg", "extract", fresh_run, "--dry-run"])
    assert result.exit_code == 0, result.output
    # No --n-times: preset value is untouched.
    assert _dry_run_slice(result.output, "seg_extract.default")["n_times"] == 60


# --- amp seg <impl> ----------------------------------------------------------------


def test_seg_rigid_dry_run_overrides(fresh_run):
    result = runner.invoke(
        app, ["seg", "rigid", fresh_run, "--k", "7", "--threshold-mult", "2.5", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    cfg = _dry_run_slice(result.output, "segment.rigid")
    assert cfg["k"] == 7
    assert cfg["threshold_mult"] == 2.5
    # Unset flags don't clobber config defaults.
    assert cfg["min_size"] == 15


def test_seg_rigid2_bad_partition(fresh_run):
    result = runner.invoke(app, ["seg", "rigid2", fresh_run, "--partition", "bogus"])
    assert result.exit_code != 0
    assert "--partition" in result.output


def test_seg_kabsch_dry_run_and_bad_init(fresh_run):
    result = runner.invoke(
        app, ["seg", "kabsch", fresh_run, "--n-clusters", "107", "--init", "spectral", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    cfg = _dry_run_slice(result.output, "segment.kabsch")
    assert cfg["n_clusters"] == 107
    assert cfg["init"] == "spectral"

    result = runner.invoke(app, ["seg", "kabsch", fresh_run, "--init", "bogus"])
    assert result.exit_code != 0
    assert "--init" in result.output


def test_seg_multicut_dry_run_bool_flags(fresh_run):
    result = runner.invoke(
        app, ["seg", "multicut", fresh_run, "--no-appearance", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert _dry_run_slice(result.output, "segment.multicut")["use_appearance"] is False


def test_seg_mbs_dry_run(fresh_run):
    result = runner.invoke(
        app, ["seg", "mbs", fresh_run, "--n-points", "8000", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert _dry_run_slice(result.output, "segment.mbs")["n_points"] == 8000


# --- amp seg eval ------------------------------------------------------------------


def test_seg_eval_dry_run(fresh_run):
    result = runner.invoke(
        app, ["seg", "eval", fresh_run, "--drop-floaters", "--bg-label", "0", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    cfg = _dry_run_slice(result.output, "seg_eval.default")
    assert cfg["drop_floaters"] is True
    assert cfg["bg_label"] == 0


def test_seg_eval_bad_bg_label(fresh_run):
    result = runner.invoke(app, ["seg", "eval", fresh_run, "--bg-label", "bogus"])
    assert result.exit_code != 0
    assert "--bg-label" in result.output


def test_seg_eval_bg_label_auto(fresh_run):
    result = runner.invoke(app, ["seg", "eval", fresh_run, "--bg-label", "auto", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert _dry_run_slice(result.output, "seg_eval.default")["bg_label"] == "auto"


# --- amp seg run -------------------------------------------------------------------


def test_seg_run_unknown_preset():
    result = runner.invoke(app, ["seg", "run", "definitely-not-a-preset"])
    assert result.exit_code != 0
    assert "unknown preset" in result.output


def test_seg_run_bad_impl():
    result = runner.invoke(app, ["seg", "run", "base", "--impl", "bogus"])
    assert result.exit_code != 0
    assert "--impl" in result.output


def test_seg_run_dry_run_full_chain():
    result = runner.invoke(app, ["seg", "run", "base", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "seg_extract.default" in result.output
    assert "segment.rigid" in result.output
    assert "seg_eval.default" in result.output


def test_seg_run_dry_run_skip_extract():
    result = runner.invoke(app, ["seg", "run", "base", "--skip-extract", "--dry-run"])
    assert result.exit_code == 0, result.output
    plan_line = next(l for l in result.output.splitlines() if "stage plan" in l)
    assert "seg_extract" not in plan_line
    assert "segment.rigid" in plan_line and "seg_eval.default" in plan_line


def test_seg_run_dry_run_impl_override_reaches_plan():
    result = runner.invoke(app, ["seg", "run", "base", "--impl", "kabsch", "--dry-run"])
    assert result.exit_code == 0, result.output
    plan_line = next(l for l in result.output.splitlines() if "stage plan" in l)
    assert "segment.kabsch" in plan_line
    # And the impl override reached the resolved config (kabsch section is sliced for it).
    cfg = _dry_run_slice(result.output, "segment.kabsch")
    assert "k_range" in cfg


def test_seg_run_mbs_requires_checkpoint():
    result = runner.invoke(app, ["seg", "run", "base", "--impl", "mbs", "--dry-run"])
    assert result.exit_code != 0
    assert "checkpoint" in result.output


# --- amp bench renders ---------------------------------------------------------------


def test_bench_renders_bad_method():
    result = runner.invoke(app, ["bench", "renders", "--methods", "bogus"])
    assert result.exit_code != 0
    assert "--methods" in result.output


def test_bench_renders_good_method_labels_pass_validation():
    # Validation passes; execution then fails on the torch-heavy import / missing GPU —
    # so assert only that the failure is NOT a validation error.
    result = runner.invoke(app, ["bench", "renders", "--methods", "base,abs segmented"])
    assert "--methods entries must be among" not in result.output


def test_bench_renders_bad_vram_mode():
    result = runner.invoke(app, ["bench", "renders", "--vram-modes", "medium"])
    assert result.exit_code != 0
    assert "--vram-modes" in result.output


def test_bench_renders_bad_repeats():
    result = runner.invoke(app, ["bench", "renders", "--repeats", "0"])
    assert result.exit_code != 0
    assert "--repeats" in result.output
