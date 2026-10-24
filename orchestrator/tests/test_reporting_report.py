"""Unit tests for pipeline.reporting.report and CLI (T29)."""

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from pipeline.artifacts.models import Artifact, RunManifest, StageRecord
from pipeline.artifacts.paths import manifest_path, run_dir
from pipeline.reporting.cli import app as report_app
from pipeline.reporting.report import build_comparison_report, build_run_report


def _create_mock_run(
    root: Path,
    run_id: str,
    preset: str = "pump01",
    experiment: str = "pump_grid_v2",
    cell: str = "pump_A20mm_M2",
    ari: float = 0.95,
) -> Path:
    r_dir = run_dir(run_id, runs_root=root)
    r_dir.mkdir(parents=True, exist_ok=True)

    # Train curves file
    train_dir = r_dir / "train_out"
    train_dir.mkdir(parents=True, exist_ok=True)
    curves_data = {
        "curves": [
            {"iteration": 10, "loss": 0.5, "l1": 0.4, "elapsed": 1.0},
            {"iteration": 20, "loss": 0.1, "l1": 0.08, "elapsed": 2.5, "test_psnr": 28.5},
        ]
    }
    (train_dir / "training_curves.json").write_text(json.dumps(curves_data), encoding="utf-8")

    # Seg eval file
    seg_eval_data = {
        "ari": ari,
        "mean_iou": 0.88,
        "n_pred": 5,
        "n_gt": 5,
        "pairs_split_rate": 0.015,
    }
    seg_file = r_dir / "seg_eval_result_rigid.json"
    seg_file.write_text(json.dumps(seg_eval_data), encoding="utf-8")

    manifest = RunManifest(
        run_id=run_id,
        preset=preset,
        metadata={"experiment": experiment, "cell": cell, "variant": cell},
        resolved_config={},
        created_at="2026-10-24T00:00:00Z",
        updated_at="2026-10-24T00:05:00Z",
        status="success",
        stages={
            "convert.default": StageRecord(status="success", wall_time_s=10.0, peak_vram_mb=0.0),
            "train.default": StageRecord(status="success", wall_time_s=120.0, peak_vram_mb=6500.0),
            "seg_eval.default": StageRecord(status="success", wall_time_s=5.0, peak_vram_mb=0.0),
        },
        artifacts={
            "seg_eval_result_rigid": Artifact(
                name="seg_eval_result_rigid",
                kind="json",
                path=str(seg_file),
                producing_stage="seg_eval.default",
                metadata={"ari": ari, "mean_iou": 0.88},
            )
        },
    )
    m_path = manifest_path(run_id, runs_root=root)
    m_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return r_dir


def test_build_run_report(tmp_path: Path):
    """build_run_report creates index.md, summary.png, training_curves.png."""
    run_id = "test_run_single"
    _create_mock_run(tmp_path, run_id)

    res = build_run_report(run_id, runs_root=tmp_path)

    assert res["run_id"] == run_id
    assert res["exists"] is True
    assert res["has_index"] is True
    assert "summary.png" in res["charts"]
    assert "training_curves.png" in res["charts"]

    rep_dir = Path(res["report_dir"])
    assert (rep_dir / "index.md").is_file()
    assert (rep_dir / "summary.png").is_file()
    assert (rep_dir / "training_curves.png").is_file()

    md_text = (rep_dir / "index.md").read_text(encoding="utf-8")
    assert f"# Run Report: `{run_id}`" in md_text
    assert "training_curves.png" in md_text
    assert "summary.png" in md_text
    assert "Total Wall Time" in md_text


def test_build_comparison_report(tmp_path: Path):
    """build_comparison_report aggregates runs, plots overlays and heatmaps."""
    _create_mock_run(tmp_path, "pump_grid_v2-cell1-11111111", cell="pump_A20mm_M2", ari=0.96)
    _create_mock_run(tmp_path, "pump_grid_v2-cell2-22222222", cell="pump_A40mm_M4", ari=0.88)

    res = build_comparison_report("pump_grid_v2", metric="ari", runs_root=tmp_path)

    assert res["experiment"] == "pump_grid_v2"
    assert res["run_count"] == 2
    assert "compare_training_curves.png" in res["charts"]
    assert "compare_metrics.png" in res["charts"]

    rep_dir = Path(res["report_dir"])
    assert (rep_dir / "compare_report.md").is_file()
    md_text = (rep_dir / "compare_report.md").read_text(encoding="utf-8")
    assert "Accuracy Pivot Table" in md_text
    assert "pump_A20mm_M2" in md_text
    assert "pump_A40mm_M4" in md_text


def test_cli_report_commands(tmp_path: Path, monkeypatch):
    """CLI subcommands execute smoothly via CliRunner."""
    _create_mock_run(tmp_path, "pump01-cli-test")
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    runner = CliRunner()

    # amp report <run_id>
    r1 = runner.invoke(report_app, ["pump01-cli-test"])
    assert r1.exit_code == 0, r1.output
    assert "Report generated" in r1.output

    # amp report run <run_id> --force
    r2 = runner.invoke(report_app, ["run", "pump01-cli-test", "--force"])
    assert r2.exit_code == 0, r2.output
    assert "Report generated" in r2.output

    # amp report compare
    r3 = runner.invoke(report_app, ["compare", "--experiment", "pump_grid_v2"])
    assert r3.exit_code == 0, r3.output
    assert "Report directory" in r3.output


def test_ast_scan_report_pure_cpu():
    """Ensure report.py and cli.py have NO torch or tensorboard imports."""
    rep_dir = Path(__file__).resolve().parents[1] / "pipeline" / "reporting"
    forbidden = {"torch", "tensorboard"}

    for name in ("report.py", "cli.py"):
        p = rep_dir / name
        assert p.is_file()
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    assert root_mod not in forbidden
