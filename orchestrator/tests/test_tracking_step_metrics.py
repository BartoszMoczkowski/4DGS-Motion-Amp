"""Unit tests for MLflow step-metric tracking and ingestion (T29)."""

import json
from pathlib import Path

import pytest
from pipeline import tracking
from pipeline.artifacts import manifest_path, run_dir
from pipeline.artifacts.models import Artifact, RunManifest, StageRecord


def test_ingest_step_metrics_into_mlflow(tmp_path: Path, monkeypatch):
    """log_stage_metrics ingests training_curves.json into MLflow using native step=iteration."""
    monkeypatch.delenv("PIPELINE_MLFLOW_DISABLE", raising=False)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    runs_dir = tmp_path / "runs"
    mlflow_dir = runs_dir / "mlflow"
    mlflow_dir.mkdir(parents=True)

    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_dir))
    monkeypatch.setenv("PIPELINE_MLFLOW_TRACKING_URI", f"file:///{mlflow_dir.as_posix()}")

    run_id = "test_step_metrics_run"
    r_dir = run_dir(run_id, runs_root=runs_dir)
    r_dir.mkdir(parents=True)

    # 1. Create manifest and start MLflow run
    manifest = RunManifest(
        run_id=run_id,
        preset="test_preset",
        resolved_config={"train": {"iterations": 30}},
        created_at="2026-10-24T00:00:00Z",
        updated_at="2026-10-24T00:05:00Z",
        stages={
            "train.default": StageRecord(status="success", wall_time_s=50.0, peak_vram_mb=4000.0),
        },
        artifacts={},
    )
    m_path = manifest_path(run_id, runs_root=runs_dir)
    m_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    mlflow_run_id = tracking.start_run(run_id, manifest.preset, manifest.resolved_config, manifest.metadata)
    assert mlflow_run_id is not None
    tracking.record_backlink(run_id, mlflow_run_id)

    # 2. Drop training_curves.json into train_out
    train_out = r_dir / "train_out"
    train_out.mkdir()
    curves = [
        {"iteration": 10, "loss": 0.45, "l1": 0.40, "elapsed": 1.2},
        {"iteration": 20, "loss": 0.15, "l1": 0.12, "elapsed": 2.5},
        {"iteration": 30, "loss": 0.04, "l1": 0.03, "elapsed": 3.8, "test_psnr": 28.5, "train_psnr": 29.2},
    ]
    (train_out / "training_curves.json").write_text(json.dumps({"curves": curves}), encoding="utf-8")

    # 3. Log stage metrics
    tracking.log_stage_metrics(run_id, manifest, stage_order=["train.default"])

    # 4. Query metric history via get_metric_history
    history_loss = tracking.get_metric_history(run_id, "stage.train.default.loss")
    assert len(history_loss) == 3
    assert history_loss[0]["step"] == 10
    assert pytest.approx(history_loss[0]["value"]) == 0.45
    assert history_loss[1]["step"] == 20
    assert pytest.approx(history_loss[1]["value"]) == 0.15
    assert history_loss[2]["step"] == 30
    assert pytest.approx(history_loss[2]["value"]) == 0.04

    history_psnr = tracking.get_metric_history(run_id, "stage.train.default.test_psnr")
    assert len(history_psnr) == 1
    assert history_psnr[0]["step"] == 30
    assert pytest.approx(history_psnr[0]["value"]) == 28.5


def test_metric_history_degradation_safety():
    """get_metric_history returns empty list and does not crash when tracking is disabled."""
    assert tracking.get_metric_history("nonexistent_run", "loss") == []
