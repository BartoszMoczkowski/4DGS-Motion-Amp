"""T30 MCP consolidation tests: the full intent-level MCP tool surface and resources.

Tests over real loopback HTTP with real MCP client, seeded with synthetic run data:
- ≤12 tools total;
- explore_experiment (run, experiment, capture, overview);
- run_experiment (succeeds with external artifacts, fails with structured error without);
- concurrency guard (starting second job for same run_id raises error);
- real cancellation (marks manifest cancelled);
- typed get_run_preview (returns VideoPreview structure);
- get_run_report (returns bundle path and MLflow URL);
- manage_preset (list, validate, save);
- manage_container (list, start, stop);
- the 3 resource templates.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import anyio
import httpx
import numpy as np
import pytest
import uvicorn
from pydantic import AnyUrl

from mcp_server.config import ServerSettings
from mcp_server.server import build_app

TEST_TOKEN = "test-token-do-not-use-for-real"

_TINY_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a4944415478da6360000002000155273d060000000049454e44ae426082"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mcp_http_client_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    kwargs: dict = {"trust_env": False, "follow_redirects": True}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


@contextlib.contextmanager
def _running_server(token: str = TEST_TOKEN) -> Iterator[str]:
    settings = ServerSettings(host="127.0.0.1", port=_free_port(), token=token)
    app = build_app(settings)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn server did not report started within 5s"
        yield f"http://{settings.host}:{settings.port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


async def _call(url: str, name: str, arguments: dict | None = None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        url,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        httpx_client_factory=_mcp_http_client_factory,
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments or {})


async def _list_tools(url: str):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        url,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        httpx_client_factory=_mcp_http_client_factory,
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


async def _read_resource(url: str, uri: str):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        url,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        httpx_client_factory=_mcp_http_client_factory,
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.read_resource(AnyUrl(uri))


def _seed_completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path))

    from pipeline.artifacts import Artifact, create_run, record_stage_result, record_stage_start, update_manifest

    run_id = "seeded-run-0001"
    create_run(run_id, "base", {"name": "base"}, stage_names=["seg_eval.default"], runs_root=runs_root)

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)

    json_path = work / "summary.json"
    json_path.write_text(json.dumps({"ari": 0.87, "n_labels": 4}), encoding="utf-8")

    npz_path = work / "trajectories.npz"
    np.savez(npz_path, positions=np.arange(24, dtype="float32").reshape(2, 3, 4))

    png_path = work / "seg_preview.png"
    png_path.write_bytes(_TINY_PNG_BYTES)

    video_path = work / "render.mp4"
    video_path.write_bytes(b"not a real mp4, just bytes for size/path purposes")

    model_dir = work / "train_out"
    (model_dir / "point_cloud" / "iteration_1").mkdir(parents=True, exist_ok=True)
    (model_dir / "point_cloud" / "iteration_1" / "point_cloud.ply").write_text("stub ply")

    artifacts = [
        Artifact(name="summary", kind="json", path=str(json_path), producing_stage="seg_eval.default"),
        Artifact(name="trajectories", kind="npz", path=str(npz_path), producing_stage="seg_eval.default"),
        Artifact(name="seg_preview", kind="png", path=str(png_path), producing_stage="seg_eval.default"),
        Artifact(name="amp_video", kind="video", path=str(video_path), producing_stage="seg_eval.default"),
        Artifact(name="model", kind="model", path=str(model_dir), producing_stage="seg_eval.default"),
    ]
    record_stage_start(run_id, "seg_eval.default", runs_root=runs_root)
    record_stage_result(
        run_id,
        "seg_eval.default",
        status="success",
        artifacts=artifacts,
        peak_vram_mb=512.0,
        runs_root=runs_root,
    )

    from pipeline.artifacts import stage_log_path

    log_path = stage_log_path(run_id, "seg_eval.default", runs_root=runs_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(f"line {i}" for i in range(5)) + "\n", encoding="utf-8")

    def _meta(m):
        m.metadata["mlflow"] = {
            "run_id": "mlflow-run-abc",
            "url": "http://127.0.0.1:5000/#/experiments/0/runs/mlflow-run-abc",
        }

    update_manifest(run_id, _meta, runs_root=runs_root)
    return run_id


# --- Tool count acceptance criterion (≤12 tools) -------------------------------------------


def test_tool_count_is_at_most_12() -> None:
    with _running_server() as url:
        tools_result = anyio.run(_list_tools, url)
    tools = tools_result.tools
    assert len(tools) <= 12, f"Expected <= 12 tools, found {len(tools)}: {[t.name for t in tools]}"
    tool_names = {t.name for t in tools}
    expected = {
        "explore_experiment",
        "run_experiment",
        "run_stage",
        "get_run_status",
        "list_runs",
        "cancel_run",
        "get_run_artifact",
        "get_run_preview",
        "get_run_report",
        "manage_preset",
        "gpu_status",
        "manage_container",
    }
    assert tool_names == expected
    # Verify rich descriptions on all tools
    for t in tools:
        assert len(t.description or "") >= 40, f"Tool {t.name} description is too sparse"


# --- Navigator: explore_experiment over HTTP ------------------------------------------------


def test_explore_experiment_over_http(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        # Overview
        overview = anyio.run(_call, url, "explore_experiment", {})
        assert not overview.isError
        assert overview.structuredContent["kind"] == "overview"

        # Specific run
        run_res = anyio.run(_call, url, "explore_experiment", {"target": run_id, "view": "full"})
        assert not run_res.isError
        res_data = run_res.structuredContent
        assert res_data["kind"] == "run"
        assert res_data["run_id"] == run_id
        assert len(res_data["artifacts"]) == 5
        assert res_data["mlflow_url"] == "http://127.0.0.1:5000/#/experiments/0/runs/mlflow-run-abc"


# --- Presets: manage_preset (list, validate, save) ------------------------------------------


def test_manage_preset_list_and_validate() -> None:
    with _running_server() as url:
        list_res = anyio.run(_call, url, "manage_preset", {"action": "list"})
        assert not list_res.isError
        presets = list_res.structuredContent["presets"]
        assert "base" in presets

        val_res = anyio.run(_call, url, "manage_preset", {"action": "validate", "preset": "base"})
        assert not val_res.isError
        assert isinstance(val_res.structuredContent["config"], dict)

        val_err = anyio.run(_call, url, "manage_preset", {"action": "validate", "preset": "does-not-exist"})
        assert val_err.isError


def test_manage_preset_save(tmp_path, monkeypatch) -> None:
    # Point presets to isolated dir
    from pipeline.config import resolver

    custom_presets = tmp_path / "presets"
    custom_presets.mkdir()
    (custom_presets / "base.yaml").write_text("name: base\n", encoding="utf-8")
    monkeypatch.setattr(resolver, "PRESETS_DIR", custom_presets)

    with _running_server() as url:
        save_res = anyio.run(
            _call,
            url,
            "manage_preset",
            {
                "action": "save",
                "name": "custom_variant",
                "base_preset": "base",
                "overrides": {"amp": {"method": "eulerian"}},
            },
        )
    assert not save_res.isError
    assert (custom_presets / "custom_variant.yaml").is_file()


# --- Execution: run_experiment with & without external artifacts ----------------------------


def test_run_experiment_missing_external_artifacts_fails_fast(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))

    with _running_server() as url:
        # Running base preset directly requires raw_mesh / capture which don't exist in empty tmp_path
        res = anyio.run(_call, url, "run_experiment", {"target": "base"})
        assert res.isError
        err_text = res.content[0].text
        assert "external artifact" in err_text or "requires" in err_text


def test_run_experiment_succeeds_when_external_artifacts_provided(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))

    # Create dummy scene dataset
    scene_dir = tmp_path / "scene_data"
    scene_dir.mkdir(parents=True)
    (scene_dir / "transforms.json").write_text("{}", encoding="utf-8")

    from pipeline.artifacts import Artifact

    ext_arts = {
        "scene": Artifact(
            name="scene",
            kind="dataset",
            path=str(scene_dir),
            producing_stage="external",
        )
    }

    with _running_server() as url:
        # Run only train.default which needs 'scene'
        res = anyio.run(
            _call,
            url,
            "run_experiment",
            {
                "target": "base",
                "stages": ["train.default"],
                "external_artifacts": ext_arts,
            },
        )
        assert not res.isError
        run_id = res.structuredContent["run_id"]
        assert run_id.startswith("base-")


# --- Concurrency guard (T17/T30) ------------------------------------------------------------


def test_concurrent_jobs_guard_rejects_second_call() -> None:
    from mcp_server import jobs

    run_id = "test-concurrent-run-01"

    # Simulate an active job
    event = threading.Event()
    def _long_running():
        event.wait(timeout=2.0)

    jobs._spawn(run_id, "test_job", _long_running)
    assert jobs.is_job_running(run_id)

    # Attempting to start another job with same run_id must raise JobRunningError
    with pytest.raises(jobs.JobRunningError, match="already running"):
        jobs._spawn(run_id, "test_job_2", lambda: None)

    event.set()


# --- Real cancellation (T17/T30) ------------------------------------------------------------


def test_cancel_run_marks_manifest_cancelled(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        res = anyio.run(_call, url, "cancel_run", {"run_id": run_id})
        assert not res.isError
        assert res.structuredContent["cancelled"] is True

        # Check status reflects cancelled
        status_res = anyio.run(_call, url, "get_run_status", {"run_id": run_id})
        assert status_res.structuredContent["status"] == "cancelled"


# --- Previews: Typed video preview & inline image (T17/T30) ----------------------------------


def test_get_preview_png_and_video(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        # Image
        img_res = anyio.run(_call, url, "get_run_preview", {"run_id": run_id, "artifact_name": "seg_preview"})
        assert not img_res.isError
        images = [c for c in img_res.content if c.type == "image"]
        assert len(images) == 1

        # Video: returns typed structured output
        vid_res = anyio.run(_call, url, "get_run_preview", {"run_id": run_id, "artifact_name": "amp_video"})
        assert not vid_res.isError
        vid_data = (
            json.loads(vid_res.content[0].text)
            if vid_res.structuredContent is None
            else vid_res.structuredContent
        )
        assert vid_data["kind"] == "video"
        assert vid_data["resource_uri"] == f"run://{run_id}/artifact/amp_video"


# --- Reports: get_run_report (T29/T30) -------------------------------------------------------


def test_get_run_report_returns_bundle_and_mlflow_url(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    runs_root = tmp_path / "runs"
    rep_dir = runs_root / run_id / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "index.md").write_text("# Test Report\nSummary data.", encoding="utf-8")
    (rep_dir / "loss_curve.png").write_bytes(_TINY_PNG_BYTES)

    with _running_server() as url:
        res = anyio.run(_call, url, "get_run_report", {"run_id": run_id})
        assert not res.isError
        data = res.structuredContent
        assert data["run_id"] == run_id
        assert data["exists"] is True
        assert data["has_index"] is True
        assert "# Test Report" in data["index_markdown"]
        assert "loss_curve.png" in data["charts"]
        assert data["mlflow_url"] == "http://127.0.0.1:5000/#/experiments/0/runs/mlflow-run-abc"


# --- Status & Logs: get_run_status with include_logs (T30) ----------------------------------


def test_get_run_status_includes_log_tail(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        res = anyio.run(
            _call,
            url,
            "get_run_status",
            {"run_id": run_id, "include_logs": True, "stage_log": "seg_eval.default", "max_log_lines": 2},
        )
        assert not res.isError
        data = res.structuredContent
        assert "log_tail" in data
        assert data["log_tail"]["stage"] == "seg_eval.default"
        assert len(data["log_tail"]["lines"]) == 2
        assert data["log_tail"]["lines"] == ["line 3", "line 4"]


# --- Artifacts: get_run_artifact ------------------------------------------------------------


def test_get_run_artifact(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        res = anyio.run(_call, url, "get_run_artifact", {"run_id": run_id, "artifact_name": "summary"})
        assert not res.isError
        assert res.structuredContent["kind"] == "json"
        assert res.structuredContent["content"]["ari"] == 0.87


# --- Containers: manage_container -----------------------------------------------------------


def test_manage_container(monkeypatch) -> None:
    from pipeline import containers

    # Mock container manager calls to be safe in sandbox
    monkeypatch.setattr(
        containers,
        "list_containers",
        lambda: [{"id": "cid-01", "name": "pipeline-cuda", "env": "cuda", "status": "running"}],
    )
    monkeypatch.setattr(containers, "start_container", lambda env: f"mock-{env}")
    monkeypatch.setattr(containers, "stop_container", lambda cid: None)

    with _running_server() as url:
        list_res = anyio.run(_call, url, "manage_container", {"action": "list"})
        assert not list_res.isError
        assert len(list_res.structuredContent["containers"]) == 1

        start_res = anyio.run(_call, url, "manage_container", {"action": "start", "env": "cuda"})
        assert not start_res.isError
        assert start_res.structuredContent["container_id"] == "mock-cuda"

        stop_res = anyio.run(_call, url, "manage_container", {"action": "stop", "container_id": "cid-01"})
        assert not stop_res.isError
        assert stop_res.structuredContent["stopped"] is True


# --- Resources: manifest / log / artifact ---------------------------------------------------


def test_resources_over_http(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        manifest_res = anyio.run(_read_resource, url, f"run://{run_id}/manifest")
        assert json.loads(manifest_res.contents[0].text)["run_id"] == run_id

        log_res = anyio.run(_read_resource, url, f"run://{run_id}/log/seg_eval.default")
        assert log_res.contents[0].text.count("\n") == 5

        art_res = anyio.run(_read_resource, url, f"run://{run_id}/artifact/seg_preview")
        import base64
        blob = base64.b64decode(art_res.contents[0].blob)
        assert blob == _TINY_PNG_BYTES
