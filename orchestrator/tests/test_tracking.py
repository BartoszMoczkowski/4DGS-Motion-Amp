"""T28 — local MLflow tracking integration.

All tests use a temp **file-store** URI (``PIPELINE_MLFLOW_TRACKING_URI=file:///<tmp>``) plus a
temp ``PIPELINE_RUNS_ROOT`` — no server, no network, no Docker/GPU. The conftest autouse fixture
sets ``PIPELINE_MLFLOW_DISABLE=1`` for the whole suite; tests here opt back in via the
``file_store`` fixture below.
"""

from __future__ import annotations

import ast
import json

import pytest


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    """Opt back into tracking against a temp MLflow file store + temp runs root."""
    monkeypatch.delenv("PIPELINE_MLFLOW_DISABLE", raising=False)
    store = tmp_path / "mlflow_store"
    monkeypatch.setenv("PIPELINE_MLFLOW_TRACKING_URI", f"file:///{store.as_posix()}")
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    # mlflow 3.x gates the (maintenance-mode) file store behind an explicit opt-in; fine for a
    # sandbox test backend — production uses the SQLite backend via `amp mlflow serve`.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return store


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _all_runs(client):
    runs = []
    for exp in client.search_experiments():
        runs.extend(client.search_runs([exp.experiment_id]))
    return runs


# --- logging lifecycle -------------------------------------------------------


def test_run_pipeline_logs_one_mlflow_run(file_store, monkeypatch):
    from pipeline import api
    from pipeline.artifacts import get_manifest

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    run_id = api.run_pipeline(
        "base",
        metadata={"experiment": "pump_grid_v2", "cell": "pump_A20mm_M2"},
    )

    client = _client()
    exp = client.get_experiment_by_name("pump_grid_v2")
    assert exp is not None, "experiment should be named after metadata['experiment']"
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["pipeline.run_id"] == run_id  # join key
    assert run.data.tags["pipeline.preset"] == "base"
    assert run.data.tags["pipeline.cell"] == "pump_A20mm_M2"
    assert run.data.params, "flattened resolved config should be logged as params"
    # manifest → MLflow backlink (free-form metadata, no schema change)
    backlink = get_manifest(run_id).metadata["mlflow"]
    assert backlink["run_id"] == run.info.run_id
    assert backlink["tracking_uri"].startswith("file:///")


def test_finish_run_marks_success(file_store, tmp_path):
    from pipeline import tracking
    from pipeline.artifacts import create_run, record_stage_result

    create_run("t28-fin", "base", {"a": 1}, stage_names=["test.echo"])
    record_stage_result("t28-fin", "test.echo", status="success")
    mlflow_run_id = tracking.start_run("t28-fin", "base", {"a": 1}, None)
    assert mlflow_run_id is not None
    from pipeline.artifacts import get_manifest

    tracking.track_run_completion("t28-fin")
    run = _client().get_run(mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["pipeline.status"] == "success"
    assert get_manifest("t28-fin").run_id  # manifest untouched otherwise


# --- join key / idempotent reuse ----------------------------------------------


def test_run_stage_reuses_existing_mlflow_run(file_store):
    from pipeline import tracking
    from pipeline.artifacts import create_run, record_stage_result

    create_run("t28-reuse", "base", {"a": 1}, stage_names=["test.echo"])
    record_stage_result("t28-reuse", "test.echo", status="success")
    first = tracking.start_run("t28-reuse", "base", {"a": 1}, None)
    second = tracking.start_run("t28-reuse", "base", {"a": 1}, None)
    assert first is not None and first == second
    assert len(_all_runs(_client())) == 1


# --- stage metrics -------------------------------------------------------------


def test_log_stage_metrics_prefixed_and_incremental(file_store, tmp_path):
    from pipeline import tracking
    from pipeline.artifacts import create_run, get_manifest, record_stage_result, record_stage_start

    create_run("t28-met", "base", {}, stage_names=["train.default", "render.default"])
    record_stage_start("t28-met", "train.default")
    record_stage_result(
        "t28-met", "train.default", status="success", peak_vram_mb=1234.0, peak_ram_mb=512.0
    )
    mlflow_run_id = tracking.start_run("t28-met", "base", {}, None)
    manifest = get_manifest("t28-met")
    order = ["train.default", "render.default"]
    tracking.log_stage_metrics("t28-met", manifest, order)
    tracking.log_stage_metrics("t28-met", manifest, order)  # second call must not duplicate

    run = _client().get_run(mlflow_run_id)
    metrics = run.data.metrics
    assert metrics["stage.train.default.peak_vram_mb"] == 1234.0
    assert metrics["stage.train.default.peak_ram_mb"] == 512.0
    assert metrics["stage.train.default.wall_time_s"] >= 0
    assert run.data.tags["stage.train.default.status"] == "success"
    assert run.data.tags["pipeline.logged_stages"] == "train.default"

    # no metrics for the still-pending stage; failure error lands as a tag
    record_stage_result("t28-met", "render.default", status="failed", error="boom")
    tracking.log_stage_metrics("t28-met", get_manifest("t28-met"), order)
    run = _client().get_run(mlflow_run_id)
    assert not any(k.startswith("stage.render.default.peak") for k in run.data.metrics)
    assert run.data.tags["stage.render.default.status"] == "failed"
    assert run.data.tags["stage.render.default.error"] == "boom"

    # history length proves no duplicate logging
    client = _client()
    hist = client.get_metric_history(mlflow_run_id, "stage.train.default.peak_vram_mb")
    assert len(hist) == 1
    assert hist[0].step == 0  # step = stage index


# --- offline degradation ---------------------------------------------------------


def test_offline_server_never_fails_run(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_MLFLOW_DISABLE", raising=False)
    monkeypatch.setenv("PIPELINE_MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # dead
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    from pipeline import api, tracking

    assert tracking.start_run("t28-off", "base", {}, None) is None
    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    run_id = api.run_pipeline("base")  # completes, warning only
    from pipeline.artifacts import get_manifest

    assert get_manifest(run_id).run_id == run_id


# --- metrics-file contract ------------------------------------------------------


def test_metrics_json_contract_ingested_with_prefix(file_store, tmp_path):
    from pipeline import tracking
    from pipeline.artifacts import create_run, get_manifest, record_stage_result

    create_run("t28-mj", "base", {}, stage_names=["seg_eval.default"])
    record_stage_result("t28-mj", "seg_eval.default", status="success")
    out = tmp_path / "runs" / "t28-mj" / "seg_eval_default_out"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(
        json.dumps({"metrics": {"ari": 0.91, "iou": 0.5}, "step": 3}), encoding="utf-8"
    )
    mlflow_run_id = tracking.start_run("t28-mj", "base", {}, None)
    tracking.log_stage_metrics("t28-mj", get_manifest("t28-mj"))

    client = _client()
    run = client.get_run(mlflow_run_id)
    assert run.data.metrics["stage.seg_eval.default.ari"] == 0.91
    hist = client.get_metric_history(mlflow_run_id, "stage.seg_eval.default.iou")
    assert hist[0].step == 3


# --- artifact policy -------------------------------------------------------------


def test_artifact_policy(file_store, tmp_path):
    from pipeline import tracking
    from pipeline.artifacts import Artifact, create_run, get_manifest, record_stage_result

    create_run("t28-art", "base", {}, stage_names=["amp.default"])
    run_dir = tmp_path / "runs" / "t28-art"
    small_mp4 = run_dir / "out.mp4"
    small_mp4.write_bytes(b"\x00" * 128)
    small_json = run_dir / "summary.json"
    small_json.write_text("{}", encoding="utf-8")
    ply = run_dir / "point_cloud.ply"
    ply.write_bytes(b"\x00" * 128)
    big = run_dir / "big.mp4"
    big.write_bytes(b"\x00" * (50 * 1024 * 1024 + 1))

    arts = [
        Artifact(name="video", kind="video", path=str(small_mp4), producing_stage="amp.default"),
        Artifact(name="summary", kind="json", path=str(small_json), producing_stage="amp.default"),
        Artifact(name="cloud", kind="ply", path=str(ply), producing_stage="amp.default"),
        Artifact(name="huge", kind="video", path=str(big), producing_stage="amp.default"),
    ]
    record_stage_result("t28-art", "amp.default", status="success", artifacts=arts)
    mlflow_run_id = tracking.start_run("t28-art", "base", {}, None)
    tracking.log_artifacts("t28-art", get_manifest("t28-art"))

    client = _client()
    logged = {a.path for a in client.list_artifacts(mlflow_run_id, "stage_artifacts")}
    assert logged == {"stage_artifacts/out.mp4", "stage_artifacts/summary.json"}


# --- env injection ----------------------------------------------------------------


def test_cuda_exec_env_injects_container_uri(monkeypatch):
    monkeypatch.delenv("PIPELINE_MLFLOW_TRACKING_URI", raising=False)
    from pipeline.stages.cuda_common import cuda_exec_env

    env = cuda_exec_env()
    assert env["MLFLOW_TRACKING_URI"] == "http://host.docker.internal:5000"
    assert env["PYTHONPATH"] == "/workspace/core"


def test_native_isaac_env_injects_host_uri(monkeypatch):
    monkeypatch.delenv("PIPELINE_MLFLOW_TRACKING_URI", raising=False)
    from pipeline.stages.isaac_common import native_isaac_exec_env

    env = native_isaac_exec_env()
    assert env["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5000"


# --- light-import rule -------------------------------------------------------------


def test_no_module_scope_mlflow_import():
    """AST-scan every ``pipeline/**/*.py``: no top-level ``import mlflow`` / ``from mlflow``."""
    from pathlib import Path

    import pipeline

    pkg = Path(pipeline.__file__).resolve().parent
    offenders = []
    for path in pkg.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only — nested imports are the allowed pattern
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [
                    node.module or ""
                ]
                if any(n == "mlflow" or n.startswith("mlflow.") for n in names):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"module-scope mlflow imports: {offenders}"


def test_tracking_module_imports_without_mlflow(monkeypatch):
    """``import pipeline.tracking`` must not require mlflow (Layer 1 sandbox import)."""
    import builtins
    import sys

    for mod in [m for m in sys.modules if m == "mlflow" or m.startswith("mlflow.")]:
        monkeypatch.delitem(sys.modules, mod)
    sys.modules.pop("pipeline.tracking", None)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    import importlib

    import pipeline.tracking as tracking

    importlib.reload(tracking)  # must import with mlflow imports blocked
    assert tracking.is_enabled() is False  # conftest set PIPELINE_MLFLOW_DISABLE=1


# --- prune -------------------------------------------------------------------------


def test_prune_deletes_old_runs_only(file_store, tmp_path):
    import time

    from mlflow.tracking import MlflowClient

    from pipeline import tracking

    tracking._mlflow()  # sets tracking URI before the raw client is constructed
    client = MlflowClient()
    exp_id = client.create_experiment("prune-test")
    old = client.create_run(exp_id, tags={"pipeline.run_id": "old"})
    new = client.create_run(exp_id, tags={"pipeline.run_id": "new"})
    old_end = int((time.time() - 40 * 86400) * 1000)
    client.set_terminated(old.info.run_id, status="FINISHED", end_time=old_end)
    client.set_terminated(new.info.run_id, status="FINISHED")

    assert tracking.prune(older_than_days=30) == 1
    remaining = client.search_runs([exp_id])
    assert [r.data.tags["pipeline.run_id"] for r in remaining] == ["new"]
