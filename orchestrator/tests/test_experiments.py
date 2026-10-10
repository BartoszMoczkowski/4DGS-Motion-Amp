"""T27 Phase B: experiment specs (parsing/layering/validation), the runner's dry-run plan and
metadata-based idempotency, the unified results store, and sweep-variant materialization.

Sandbox-only: no GPU/containers. Runner tests never execute stages — the dry-run path touches
nothing, and idempotency is exercised against fabricated manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def _write_grid_manifest(tmp_path: Path, cells: list[str]) -> Path:
    entries = []
    for cell in cells:
        cap = tmp_path / f"capture_{cell}"
        cap.mkdir(parents=True, exist_ok=True)
        entries.append({
            "cell": cell,
            "base_amp_mm": 20,
            "multiplier": 2,
            "amplified_peak_mm": 23.0,
            "capture_dir": str(cap),
        })
    manifest = tmp_path / "grid_manifest.json"
    manifest.write_text(json.dumps({"seed": 0, "freq_hz": 10.0, "cells": entries}))
    (tmp_path / f"{cells[0]}_motion.json").write_text(json.dumps(
        {"num_frames": 240, "fps": 60.0, "freq_hz": 10.0, "base_amp_mm": 20, "multiplier": 2}
    ))
    return manifest


def _write_spec(tmp_path: Path, manifest: Path, **overrides) -> Path:
    spec = {
        "name": "toy_exp",
        "capture": {"manifest": str(manifest), "cells": ["cellA", "cellB"]},
        "pipeline": {"preset": "base"},
        "segment": [
            {"impl": "rigid", "preset": "grid_seg", "extract": True},
            {"impl": "kabsch", "preset": "pump01_kabsch"},
        ],
        "eval": {"gt": "data/multipleview/pump01/gt_segmentation.npz"},
    }
    spec.update(overrides)
    path = tmp_path / "toy_exp.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


# --- spec parsing / validation -------------------------------------------------------------------


def test_real_pump_grid_v2_spec_parses():
    from pipeline.experiments.spec import load_spec

    spec = load_spec("pump_grid_v2")
    assert spec.name == "pump_grid_v2"
    assert spec.capture.cells == ["pump_A20mm_M2", "pump_A20mm_M4", "pump_A40mm_M8"]
    assert spec.sweep.targets == [10000, 25000, 50000, 100000]
    assert spec.sweep.cell == "pump_A40mm_M8"
    assert spec.pipeline.preset == "pump01"
    assert spec.pipeline.overrides["render"]["skip_train"] is True
    assert [e.impl for e in spec.segment] == [
        "rigid", "mbs", "rigid2", "kabsch", "rigid2_roi", "mask_lift_oracle", "mask_lift",
        "multicut",
    ]
    assert spec.segment[0].extract is True
    assert sum(e.extract for e in spec.segment) == 1
    assert spec.stages == ["convert.default", "train.default", "render.default"]


def test_spec_parse_and_defaults(tmp_path):
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    from pipeline.experiments.spec import load_spec

    spec = load_spec(str(_write_spec(tmp_path, manifest)))
    assert spec.name == "toy_exp"
    assert spec.sweep is None
    assert spec.stages == ["convert.default", "train.default", "render.default"]


def test_spec_rejects_unknown_pipeline_preset(tmp_path):
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    path = _write_spec(tmp_path, manifest, pipeline={"preset": "no_such_preset"})
    from pipeline.experiments.spec import load_spec

    with pytest.raises(ValueError, match="unknown pipeline preset"):
        load_spec(str(path))


def test_spec_rejects_unknown_impl_and_preset(tmp_path):
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    from pipeline.experiments.spec import load_spec

    path = _write_spec(tmp_path, manifest,
                       segment=[{"impl": "not_an_impl", "preset": "grid_seg"}])
    with pytest.raises(ValueError, match="unknown segment impl"):
        load_spec(str(path))

    path = _write_spec(tmp_path, manifest,
                       segment=[{"impl": "rigid", "preset": "no_such_preset"}])
    with pytest.raises(ValueError, match="unknown.*preset"):
        load_spec(str(path))


def test_spec_rejects_missing_cell_and_bad_sweep_cell(tmp_path):
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    from pipeline.experiments.spec import load_spec

    path = _write_spec(tmp_path, manifest)
    data = yaml.safe_load(path.read_text())
    data["capture"]["cells"] = ["cellA", "nope"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="not in"):
        load_spec(str(path))

    data["capture"]["cells"] = ["cellA", "cellB"]
    data["sweep"] = {"cell": "other", "targets": [1000]}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="sweep cell"):
        load_spec(str(path))


def test_spec_extends_layering(tmp_path):
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    base = tmp_path / "base_exp.yaml"
    base.write_text(yaml.safe_dump({
        "name": "base_exp",
        "capture": {"manifest": str(manifest), "cells": ["cellA"]},
        "pipeline": {"preset": "base", "overrides": {"render": {"skip_train": True}}},
        "segment": [{"impl": "rigid", "preset": "grid_seg", "extract": True}],
    }))
    child = tmp_path / "child_exp.yaml"
    child.write_text(yaml.safe_dump({
        "extends": "base_exp.yaml",
        "name": "child_exp",
        "capture": {"cells": ["cellA", "cellB"]},
        "pipeline": {"overrides": {"render": {"skip_test": True}}},
    }))
    from pipeline.experiments.spec import load_spec

    spec = load_spec(str(child))
    assert spec.name == "child_exp"
    assert spec.capture.cells == ["cellA", "cellB"]           # child wins
    assert spec.capture.manifest == str(manifest)             # inherited
    # pipeline.overrides deep-merge: child's skip_test added, parent's skip_train kept.
    assert spec.pipeline.overrides == {"render": {"skip_train": True, "skip_test": True}}
    assert [e.impl for e in spec.segment] == ["rigid"]

    loop = tmp_path / "loop.yaml"
    loop.write_text(yaml.safe_dump({"extends": "loop.yaml", "name": "loop"}))
    with pytest.raises(ValueError, match="circular"):
        load_spec(str(loop))


# --- runner: dry-run + idempotency -----------------------------------------------------------------


def _fake_pipeline_run(tmp_path: Path, run_id: str, experiment: str, cell: str, variant: str):
    """Fabricate a successful pipeline-phase run manifest with experiment metadata."""
    from pipeline.artifacts import StageRecord, create_run, update_manifest

    stages = ["convert.default", "train.default", "render.default"]
    create_run(run_id, "base", {}, stage_names=stages, runs_root=tmp_path,
               metadata={"experiment": experiment, "cell": cell, "variant": variant})

    def _mut(m):
        for s in stages:
            m.stages[s] = StageRecord(status="success", wall_time_s=1.0)

    update_manifest(run_id, _mut, runs_root=tmp_path)


def test_dry_run_plans_without_touching_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    spec_path = _write_spec(
        tmp_path, manifest,
        sweep={"cell": "cellA", "targets": [500, 1000], "seed": 0},
    )
    lines: list[str] = []

    from pipeline.experiments.runner import run_experiment

    summary = run_experiment(str(spec_path), dry_run=True, log=lines.append)

    # 2 cells + 2 sweep variants.
    assert [v["variant"] for v in summary["variants"]] == [
        "cellA", "cellB", "cellA_g500", "cellA_g1000",
    ]
    assert all(not v["reused"] for v in summary["variants"])
    # The plan names the would-be run ids, stages, and the impl matrix with stage lists.
    v0 = summary["variants"][0]
    assert v0["run_id"].startswith("toy_exp-cellA-")
    assert v0["stages"] == ["convert.default", "train.default", "render.default"]
    assert [e["impl"] for e in v0["segment"]] == ["rigid", "kabsch"]
    assert v0["segment"][0]["stages"] == ["seg_extract.default", "segment.rigid", "seg_eval.default"]
    assert v0["segment"][1]["stages"] == ["segment.kabsch", "seg_eval.default"]
    # Sweep variant dirs follow the capture_pump-style convention.
    assert summary["variants"][2]["capture_dir"].endswith("capture_cellA_g500")
    # Nothing was created.
    runs_root = tmp_path / "runs"
    assert not runs_root.exists() or not any(runs_root.iterdir())
    assert not (tmp_path / "capture_cellA_g500").exists()
    assert any("toy_exp" in line for line in lines)


def test_idempotency_reuses_run_by_metadata(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))
    manifest = _write_grid_manifest(tmp_path, ["cellA", "cellB"])
    _fake_pipeline_run(runs_root, "toy_exp-cellA-deadbeef", "toy_exp", "cellA", "cellA")
    spec_path = _write_spec(tmp_path, manifest)

    from pipeline.experiments.runner import run_experiment

    summary = run_experiment(str(spec_path), dry_run=True, log=lambda m: None)
    by_variant = {v["variant"]: v for v in summary["variants"]}
    assert by_variant["cellA"]["reused"] is True
    assert by_variant["cellA"]["run_id"] == "toy_exp-cellA-deadbeef"
    assert by_variant["cellB"]["reused"] is False
    # A run tagged with a DIFFERENT experiment name must not be reused.
    _fake_pipeline_run(runs_root, "other-cellB-12345678", "other_exp", "cellB", "cellB")
    summary = run_experiment(str(spec_path), dry_run=True, log=lambda m: None)
    assert {v["variant"]: v["reused"] for v in summary["variants"]} == {
        "cellA": True, "cellB": False,
    }


# --- results store ---------------------------------------------------------------------------------


def test_results_collect_and_export(tmp_path):
    from pipeline.artifacts import Artifact, StageRecord, create_run, update_manifest
    from pipeline.experiments import results as res

    run_id = "toy_exp-cellA-deadbeef"
    run_dir = tmp_path / run_id
    create_run(run_id, "base", {},
               stage_names=["convert.default", "train.default", "render.default",
                            "seg_extract.default", "segment.rigid", "seg_eval.default"],
               runs_root=tmp_path,
               metadata={"experiment": "toy_exp", "cell": "cellA", "variant": "cellA",
                         "target_gaussians": ""})

    eval_path = run_dir / "seg_eval_result_rigid.json"
    eval_path.write_text(json.dumps({"ari": 0.91, "mean_iou": 0.5, "n_gt": 7, "n_pred": 9}))

    def _mut(m):
        m.stages["convert.default"] = StageRecord(status="success", wall_time_s=2.0)
        m.stages["train.default"] = StageRecord(status="success", wall_time_s=300.0,
                                                peak_vram_mb=8000.0)
        m.stages["render.default"] = StageRecord(status="success", wall_time_s=60.0)
        m.stages["seg_extract.default"] = StageRecord(status="success", wall_time_s=30.0)
        m.stages["segment.rigid"] = StageRecord(status="success", wall_time_s=12.0)
        m.stages["seg_eval.default"] = StageRecord(status="success", wall_time_s=1.0)
        m.artifacts["seg_eval_result_rigid"] = Artifact(
            name="seg_eval_result_rigid", kind="json", path=str(eval_path),
            producing_stage="seg_eval.default",
        )
        m.status = "success"

    update_manifest(run_id, _mut, runs_root=tmp_path)
    # A run WITHOUT experiment metadata must be excluded.
    create_run("adhoc-run", "base", {}, runs_root=tmp_path)

    rows = res.collect_results(runs_root=tmp_path)
    assert {r["run_id"] for r in rows} == {run_id}
    base_row = next(r for r in rows if r["impl"] == "")
    assert base_row["experiment"] == "toy_exp"
    assert base_row["train_s"] == 300.0
    assert base_row["train_peak_vram_mb"] == 8000.0
    seg_row = next(r for r in rows if r["impl"] == "rigid")
    assert seg_row["ari"] == 0.91 and seg_row["n_gt"] == 7
    assert seg_row["segment_s"] == 12.0

    out = res.export_csv(rows, tmp_path / "store.csv")
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0].split(",") == res.COLUMNS
    assert len(text.splitlines()) == 3  # header + pipeline row + rigid row

    # Latest-wins keying on (run_id, impl): appending a replacement row updates in place.
    res.append_result({**seg_row, "ari": 0.95}, tmp_path / "store.csv")
    with open(out, newline="", encoding="utf-8") as f:
        import csv

        reread = list(csv.DictReader(f))
    assert len(reread) == 2
    assert next(r for r in reread if r["impl"] == "rigid")["ari"] == "0.95"


# --- sweep variant materialization -----------------------------------------------------------------


def _write_source_capture(src: Path, n: int = 100) -> None:
    from pipeline.vendored.host.convert import write_ply

    src.mkdir(parents=True)
    rng = np.random.RandomState(0)
    xyz = rng.rand(n, 3).astype(np.float32)
    rgb = rng.randint(0, 256, size=(n, 3)).astype(np.uint8)
    write_ply(str(src / "points3D_gt.ply"), xyz, rgb.astype(np.float32) / 255.0)
    np.save(src / "points3D_labels.npy", np.arange(n) % 5)
    (src / "cameras_gt.json").write_text("{}")
    (src / "cam00").mkdir()
    (src / "cam00" / "000.png").write_bytes(b"png")


def test_make_capture_variant_subsamples_and_is_idempotent(tmp_path, monkeypatch):
    from pipeline.experiments import variants

    src = tmp_path / "capture_cellA"
    _write_source_capture(src, n=100)
    dst = tmp_path / "capture_cellA_g40"

    # Junctions may be unavailable in the sandbox (privileges); stub mklink to a plain mkdir —
    # the subsample/file-writing behavior under test is junction-independent.
    def fake_mklink(cmd, capture_output, text):
        Path(cmd[-2]).mkdir(exist_ok=True)

        class R:
            returncode = 0
            stdout = stderr = ""

        return R()

    monkeypatch.setattr(variants.subprocess, "run", fake_mklink)

    out = variants.make_capture_variant(src, dst, 40, seed=0)
    assert out == dst
    assert variants.count_ply_vertices(dst / "points3D_gt.ply") == 40
    assert len(np.load(dst / "points3D_labels.npy")) == 40
    assert (dst / "cameras_gt.json").is_file()
    assert (dst / "cam00").is_dir()  # junction (stubbed)

    # Idempotent: existing variant with the right count is reused untouched.
    assert variants.make_capture_variant(src, dst, 40, seed=0) == dst
    # A mismatched existing dir is an error, never a silent clobber.
    with pytest.raises(RuntimeError, match="already exists"):
        variants.make_capture_variant(src, dst, 50, seed=0)
    # 100% target reuses the source dir.
    assert variants.make_capture_variant(src, tmp_path / "g100", 100, seed=0) == src


def test_count_gaussians_numeric_iteration_sort(tmp_path):
    from pipeline.experiments.variants import count_gaussians
    from pipeline.vendored.host.convert import write_ply

    pc = tmp_path / "train_out" / "point_cloud"
    for it, n in ((7000, 111), (30000, 222)):  # lexicographic order would pick 7000 last
        d = pc / f"iteration_{it}"
        d.mkdir(parents=True)
        write_ply(str(d / "point_cloud.ply"), np.zeros((n, 3), np.float32),
                  np.zeros((n, 3), np.float32))
    assert count_gaussians(tmp_path) == 222
    assert count_gaussians(tmp_path / "empty") is None
