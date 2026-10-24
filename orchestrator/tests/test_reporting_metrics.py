"""Unit tests for pipeline.reporting.metrics (T29)."""

import ast
from pathlib import Path

import pytest
from pipeline.artifacts.models import Artifact, RunManifest, StageRecord
from pipeline.reporting.metrics import (
    METRIC_CATALOG,
    MetricDef,
    extract_manifest_metrics,
    format_metric,
    get_metric_def,
)


def test_metric_catalog_entries_valid():
    """Every metric in the canonical catalog must have phase, unit, source, and description."""
    assert len(METRIC_CATALOG) >= 20
    valid_phases = {"train", "render", "segment", "research", "bench", "amp"}

    for name, mdef in METRIC_CATALOG.items():
        assert mdef.name == name
        assert mdef.phase in valid_phases, f"Invalid phase {mdef.phase} for metric {name}"
        assert mdef.source, f"Metric {name} missing source"
        assert mdef.description, f"Metric {name} missing description"


def test_format_metric_values():
    """Formatting helper should handle units and None values gracefully."""
    assert format_metric("unknown_metric", None) == "N/A"
    assert format_metric("total_loss", 0.0523456) == "0.05235"
    assert format_metric("test_psnr", 29.85) == "29.85 dB"
    assert format_metric("wall_time_s", 75.5) == "1m 15.5s"
    assert format_metric("wall_time_s", 12.34) == "12.34s"
    assert format_metric("peak_vram_mb", 4096.5) == "4096.5 MB"
    assert format_metric("pairs_split_rate", 0.035) == "3.50%"
    assert format_metric("fft_dominant_freq_hz", 15.6) == "15.60 Hz"
    assert format_metric("estimated_rpm", 1200.0) == "1200.0 RPM"


def test_extract_manifest_metrics(tmp_path: Path):
    """extract_manifest_metrics aggregates metrics from manifest, artifacts, and disk."""
    run_id = "test_run_123"
    r_dir = tmp_path / run_id
    r_dir.mkdir(parents=True)

    manifest = RunManifest(
        run_id=run_id,
        preset="test_preset",
        resolved_config={},
        created_at="2026-10-24T00:00:00Z",
        updated_at="2026-10-24T00:00:00Z",
        stages={
            "convert.default": StageRecord(status="success", wall_time_s=10.0, peak_vram_mb=500.0),
            "train.default": StageRecord(status="success", wall_time_s=120.0, peak_vram_mb=6000.0),
            "seg_eval.default": StageRecord(status="success", wall_time_s=5.0, peak_vram_mb=100.0),
        },
        artifacts={
            "seg_eval_result_rigid": Artifact(
                name="seg_eval_result_rigid",
                kind="json",
                path=str(r_dir / "seg_eval_result_rigid.json"),
                producing_stage="seg_eval.default",
                metadata={"ari": 0.952, "mean_iou": 0.884, "pairs_split_rate": 0.012},
            )
        },
    )

    # Fabricate training_curves.json
    train_dir = r_dir / "train_out"
    train_dir.mkdir()
    (train_dir / "training_curves.json").write_text(
        '{"curves": [{"iteration": 10, "loss": 0.5, "elapsed": 5.0}, {"iteration": 20, "loss": 0.05, "l1": 0.04, "elapsed": 120.0, "test_psnr": 28.5}]}',
        encoding="utf-8",
    )

    extracted = extract_manifest_metrics(manifest, run_dir_path=r_dir)

    assert extracted["bench"]["total_wall_time_s"] == 135.0
    assert extracted["bench"]["peak_vram_mb"] == 6000.0
    assert extracted["segment"]["ari"] == 0.952
    assert extracted["segment"]["mean_iou"] == 0.884
    assert extracted["train"]["total_loss"] == 0.05
    assert extracted["train"]["l1_loss"] == 0.04
    assert extracted["train"]["test_psnr"] == 28.5


def test_ast_scan_pure_cpu():
    """Ensure reporting/metrics.py does not import torch, tensorboard, or matplotlib."""
    file_path = Path(__file__).resolve().parents[1] / "pipeline" / "reporting" / "metrics.py"
    assert file_path.is_file()

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    forbidden = {"torch", "tensorboard", "matplotlib", "cv2"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, f"Forbidden import {alias.name} in metrics.py"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden, f"Forbidden from-import {node.module} in metrics.py"
