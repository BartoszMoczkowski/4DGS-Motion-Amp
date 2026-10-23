"""Tests for the unified experiment navigator (T30: explore_experiment & amp explore)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.artifacts import (
    Artifact,
    create_run,
    record_stage_result,
    record_stage_start,
    update_manifest,
)
from pipeline.explore import explore, format_explore_human


def _seed_sample_run(tmp_path: Path, run_id: str, preset: str = "pump01") -> None:
    runs_root = tmp_path / "runs"
    m = create_run(run_id, preset, {"name": preset}, runs_root=runs_root)

    # Add a stage
    art_path = runs_root / run_id / "test_out.json"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    art_path.write_text(json.dumps({"loss": 0.05, "iterations": 1000}), encoding="utf-8")

    art = Artifact(
        name="test_out",
        kind="json",
        path=str(art_path),
        producing_stage="train.default",
    )
    record_stage_start(run_id, "train.default", runs_root=runs_root)
    record_stage_result(
        run_id,
        "train.default",
        status="success",
        artifacts=[art],
        peak_vram_mb=1024.0,
        runs_root=runs_root,
    )

    # Also add an unregistered file in run_dir
    unreg = runs_root / run_id / "scratch_dump.txt"
    unreg.write_text("temporary scratch notes", encoding="utf-8")

    def _meta(manifest):
        manifest.metadata["experiment"] = "test_exp"
        manifest.metadata["cell"] = "cell_A"
        manifest.metadata["variant"] = "variant_1"
        manifest.metadata["mlflow"] = {
            "run_id": "mlflow-run-123",
            "url": "http://127.0.0.1:5000/#/experiments/1/runs/mlflow-run-123",
        }

    update_manifest(run_id, _meta, runs_root=runs_root)


def test_explore_overview(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _seed_sample_run(tmp_path, "run-001")
    _seed_sample_run(tmp_path, "run-002")

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "test_exp.yaml").write_text(
        "name: test_exp\npipeline:\n  preset: pump01\ncapture:\n  cells: [cell_A]\nsegment:\n  - impl: rigid\n",
        encoding="utf-8",
    )

    data = explore(target=None, runs_root=runs_root, experiments_dir=exp_dir)
    assert data["kind"] == "overview"
    assert data["total_runs"] == 2
    assert len(data["experiments"]) >= 1
    assert data["experiments"][0]["name"] == "test_exp"
    assert len(data["recent_runs"]) == 2

    # Verify human formatter
    human_text = format_explore_human(data)
    assert "# Workspace Overview" in human_text
    assert "test_exp" in human_text
    assert "run-001" in human_text


def test_explore_run(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_id = "test_run_explore"
    _seed_sample_run(tmp_path, run_id)

    data = explore(
        target=run_id, view="full", include_unregistered=True, runs_root=runs_root
    )
    assert data["kind"] == "run"
    assert data["run_id"] == run_id
    assert data["preset"] == "pump01"
    assert data["mlflow_url"] == "http://127.0.0.1:5000/#/experiments/1/runs/mlflow-run-123"
    assert len(data["timeline"]) == 1
    assert data["timeline"][0]["stage"] == "train.default"
    assert data["timeline"][0]["peak_vram_mb"] == 1024.0
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["name"] == "test_out"

    # Verify unregistered file detection
    unreg = data.get("unregistered_files", [])
    assert any("scratch_dump.txt" in u["path"] for u in unreg)

    # Human formatter
    human_text = format_explore_human(data)
    assert f"# Run: {run_id}" in human_text
    assert "train.default" in human_text
    assert "test_out" in human_text
    assert "scratch_dump.txt" in human_text


def test_explore_experiment(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _seed_sample_run(tmp_path, "test_exp-cell_A-001")

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "test_exp.yaml").write_text(
        "name: test_exp\npipeline:\n  preset: pump01\ncapture:\n  cells: [cell_A]\nsegment:\n  - impl: rigid\n",
        encoding="utf-8",
    )

    data = explore(
        target="test_exp", runs_root=runs_root, experiments_dir=exp_dir
    )
    assert data["kind"] == "experiment"
    assert data["name"] == "test_exp"
    assert data["linked_runs_count"] == 1
    assert data["linked_runs"][0]["run_id"] == "test_exp-cell_A-001"

    human_text = format_explore_human(data)
    assert "# Experiment: test_exp" in human_text
    assert "test_exp-cell_A-001" in human_text


def test_explore_capture_directory(tmp_path: Path):
    cap_dir = tmp_path / "dummy_capture"
    cap_dir.mkdir()
    (cap_dir / "images").mkdir()
    (cap_dir / "images" / "frame_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (cap_dir / "images" / "frame_001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (cap_dir / "cameras_gt.json").write_text(json.dumps([{"id": 0}, {"id": 1}]), encoding="utf-8")

    data = explore(target=str(cap_dir))
    assert data["kind"] == "capture"
    assert data["name"] == "dummy_capture"
    assert data["frame_count"] == 2
    assert data["camera_count"] == 2

    human_text = format_explore_human(data)
    assert "# Capture / Dataset: dummy_capture" in human_text
    assert "Frames: 2" in human_text
