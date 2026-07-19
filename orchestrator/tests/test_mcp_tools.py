"""T14 acceptance tests: the full MCP tool/resource set, over the same real HTTP loopback
connection ``tests/test_mcp_server.py`` (T13) already proved the transport/auth over.

Same "there's nothing to fake" reasoning as T13's own test file for the read/discovery tools
(``list_presets``/``validate_config``/``list_runs``/``list_artifacts``/``read_artifact``/
``get_preview``/``tail_logs``/the resources) — they only ever touch the local filesystem/config,
which this sandbox has for real, so these run the actual server + actual MCP client, seeded with
synthetic run data (real files on disk, written directly via ``pipeline.artifacts``, the same
"seed a manifest by hand" pattern ``tests/test_stages_cpu.py``'s ``_seed_run`` helper established).

``run_pipeline``/``run_stage``/``list_containers``/``start_container``/``stop_container`` genuinely
need a GPU/Docker daemon/native Isaac install this sandbox doesn't have — what's verified here for
those is the *async-return-immediately* plumbing (``mcp_server.jobs``) and that a real failure
(missing external input, no reachable Docker daemon) surfaces cleanly instead of hanging or
crashing the server, mirroring every other real-hardware-dependent task's "verified the logic
above the real dependency, not the real dependency itself" story (T08/T09/T10/T11).
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

#: A real, minimal (1x1, transparent) PNG -- small enough to inline, valid enough for `Image` to
#: read a real mime type/size off of, unlike a stub text file with a `.png` name.
_TINY_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a4944415478da6360000002000155273d060000000049454e44ae426082"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mcp_http_client_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    """Same ``trust_env=False`` fix as ``test_mcp_server.py`` -- this sandbox's own proxy env vars
    would otherwise break ``streamablehttp_client``'s default ``httpx.AsyncClient``."""

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


# --- seeding a synthetic completed run, real files on disk ----------------------------------


def _seed_completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Writes a real manifest + real json/npz/png/video/dataset-dir artifacts for one run, under
    an isolated ``PIPELINE_RUNS_ROOT`` -- exactly what ``pipeline.api``'s (and therefore this
    server's) functions read by default, no ``runs_root=`` plumbing needed on the MCP side.
    """
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))

    from pipeline.artifacts import Artifact, create_run, record_stage_result

    run_id = "seeded-run-0001"
    create_run(run_id, "base", {"preset": "base"}, stage_names=["seg_eval.default"])

    work = tmp_path / "work"
    work.mkdir()

    json_path = work / "summary.json"
    json_path.write_text(json.dumps({"ari": 0.87, "n_labels": 4}), encoding="utf-8")

    npz_path = work / "trajectories.npz"
    np.savez(npz_path, positions=np.arange(24, dtype="float32").reshape(2, 3, 4))

    png_path = work / "seg_preview.png"
    png_path.write_bytes(_TINY_PNG_BYTES)

    video_path = work / "render.mp4"
    video_path.write_bytes(b"not a real mp4, just bytes for size/path purposes")

    model_dir = work / "train_out"
    (model_dir / "point_cloud" / "iteration_1").mkdir(parents=True)
    (model_dir / "point_cloud" / "iteration_1" / "point_cloud.ply").write_text("stub ply")

    artifacts = [
        Artifact(name="summary", kind="json", path=str(json_path), producing_stage="seg_eval.default"),
        Artifact(name="trajectories", kind="npz", path=str(npz_path), producing_stage="seg_eval.default"),
        Artifact(name="seg_preview", kind="png", path=str(png_path), producing_stage="seg_eval.default"),
        Artifact(name="amp_video", kind="video", path=str(video_path), producing_stage="seg_eval.default"),
        Artifact(name="model", kind="model", path=str(model_dir), producing_stage="seg_eval.default"),
    ]
    record_stage_result(
        run_id,
        "seg_eval.default",
        status="success",
        artifacts=artifacts,
        log_path=None,
    )

    from pipeline.artifacts import get_runs_root, stage_log_path

    log_path = stage_log_path(run_id, "seg_eval.default", runs_root=get_runs_root())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(f"line {i}" for i in range(5)) + "\n", encoding="utf-8")

    return run_id


# --- discovery: list_presets / validate_config -----------------------------------------------


def test_list_presets_over_http() -> None:
    with _running_server() as url:
        result = anyio.run(_call, url, "list_presets", {})
    assert not result.isError
    assert isinstance(result.structuredContent["result"], list)
    assert "base" in result.structuredContent["result"]


def test_validate_config_over_http() -> None:
    with _running_server() as url:
        result = anyio.run(_call, url, "validate_config", {"preset": "base"})
    assert not result.isError
    assert isinstance(result.structuredContent, dict)


def test_validate_config_unknown_preset_is_a_tool_error() -> None:
    with _running_server() as url:
        result = anyio.run(_call, url, "validate_config", {"preset": "does-not-exist"})
    assert result.isError


# --- seeded run: status / artifacts / read / preview / logs / resources ----------------------


def test_list_runs_and_list_artifacts_over_http(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        runs = anyio.run(_call, url, "list_runs", {})
        artifacts = anyio.run(_call, url, "list_artifacts", {"run_id": run_id})

    assert not runs.isError
    ids = [r["run_id"] for r in runs.structuredContent["result"]]
    assert run_id in ids

    assert not artifacts.isError
    names = {a["name"] for a in artifacts.structuredContent["result"]}
    assert names == {"summary", "trajectories", "seg_preview", "amp_video", "model"}


def test_get_run_status_includes_job_error_key(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "get_run_status", {"run_id": run_id})
    assert not result.isError
    status = result.structuredContent
    assert status["run_id"] == run_id
    assert status["status"] == "success"
    assert status["job_error"] is None  # this run was seeded directly, never went through jobs.py


def test_read_artifact_json_returns_parsed_content(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "read_artifact", {"run_id": run_id, "artifact_name": "summary"})
    assert not result.isError
    body = result.structuredContent
    assert body["kind"] == "json"
    assert body["content"] == {"ari": 0.87, "n_labels": 4}


def test_read_artifact_npz_returns_summary_not_raw_arrays(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(
            _call, url, "read_artifact", {"run_id": run_id, "artifact_name": "trajectories"}
        )
    assert not result.isError
    body = result.structuredContent
    assert body["kind"] == "npz"
    assert "positions" in body["arrays"]
    assert body["arrays"]["positions"]["shape"] == [2, 3, 4]
    assert "raw" not in body and "data" not in body["arrays"]["positions"]


def test_read_artifact_model_directory_returns_listing(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "read_artifact", {"run_id": run_id, "artifact_name": "model"})
    assert not result.isError
    body = result.structuredContent
    assert body["kind"] == "model"
    assert any(e["name"] == "point_cloud" and e["is_dir"] for e in body["entries"])


def test_get_preview_png_returns_inline_image(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "get_preview", {"run_id": run_id, "artifact_name": "seg_preview"})
    assert not result.isError
    images = [c for c in result.content if c.type == "image"]
    assert len(images) == 1
    assert images[0].mimeType == "image/png"


def test_get_preview_video_returns_pointer_not_inline_bytes(tmp_path, monkeypatch) -> None:
    """``get_preview`` has no return-type annotation (it returns either an ``Image`` or a plain
    dict, an intentionally mixed shape FastMCP can't express as one structured schema) -- so a
    dict return comes back as unstructured ``TextContent`` (JSON text), not ``structuredContent``.
    Parse it out of ``result.content`` instead, same as any other unstructured-tool caller would.
    """
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "get_preview", {"run_id": run_id, "artifact_name": "amp_video"})
    assert not result.isError
    assert result.structuredContent is None
    body = json.loads(result.content[0].text)
    assert body["kind"] == "video"
    assert body["resource_uri"] == f"run://{run_id}/artifact/amp_video"


def test_get_preview_non_previewable_kind_is_a_tool_error(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "get_preview", {"run_id": run_id, "artifact_name": "summary"})
    assert result.isError


def test_tail_logs_returns_last_lines(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(
            _call, url, "tail_logs", {"run_id": run_id, "stage": "seg_eval.default", "max_lines": 2}
        )
    assert not result.isError
    body = result.structuredContent
    assert body["lines"] == ["line 3", "line 4"]
    assert body["line_count"] == 5
    assert body["truncated"] is True


def test_cancel_run_reports_not_implemented_honestly(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        result = anyio.run(_call, url, "cancel_run", {"run_id": run_id})
    assert not result.isError
    body = result.structuredContent
    assert body["cancelled"] is False
    assert "not implemented" in body["reason"] or "isn't implemented" in body["reason"]


def test_cancel_run_unknown_run_id_is_a_tool_error() -> None:
    with _running_server() as url:
        result = anyio.run(_call, url, "cancel_run", {"run_id": "no-such-run"})
    assert result.isError


# --- resources: manifest / log / artifact -----------------------------------------------------


def test_manifest_resource_matches_get_run_status(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        status = anyio.run(_call, url, "get_run_status", {"run_id": run_id}).structuredContent
        resource = anyio.run(_read_resource, url, f"run://{run_id}/manifest")

    text = resource.contents[0].text
    manifest_via_resource = json.loads(text)
    assert manifest_via_resource["run_id"] == status["run_id"]
    assert manifest_via_resource["status"] == status["status"]


def test_log_resource_returns_full_untruncated_text(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        resource = anyio.run(_read_resource, url, f"run://{run_id}/log/seg_eval.default")
    text = resource.contents[0].text
    assert text.count("\n") == 5  # all 5 lines, not just tail_logs's default-truncated view


def test_artifact_resource_returns_raw_bytes(tmp_path, monkeypatch) -> None:
    run_id = _seed_completed_run(tmp_path, monkeypatch)
    with _running_server() as url:
        resource = anyio.run(_read_resource, url, f"run://{run_id}/artifact/seg_preview")
    content = resource.contents[0]
    import base64

    raw = base64.b64decode(content.blob) if hasattr(content, "blob") and content.blob else None
    if raw is None:
        # some client-side plumbing may decode text-ish resources as text even for bytes-typed
        # functions if a mime type looks textual; png never should, but assert defensively.
        raise AssertionError(f"expected binary resource content, got: {content!r}")
    assert raw == _TINY_PNG_BYTES


# --- async run lifecycle: real preset, no GPU/Docker touched at all ---------------------------


def test_run_pipeline_returns_run_id_immediately(tmp_path, monkeypatch) -> None:
    """No ``external_artifacts`` supplied for a preset whose auto-planned DAG needs one
    (``prep_split.default``'s ``raw_mesh``) -- ``run_dag`` raises ``MissingDependencyError``
    before any stage runs, entirely without touching Docker/GPU/native Isaac. What this actually
    proves: the tool call returns a ``run_id`` immediately (doesn't block for however long a real
    run would take), and the background failure is captured by ``mcp_server.jobs`` rather than
    vanishing silently.
    """
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(runs_root))

    with _running_server() as url:
        result = anyio.run(_call, url, "run_pipeline", {"preset": "base"})
        assert not result.isError
        run_id = result.structuredContent["run_id"]
        assert run_id  # returned before the background thread could possibly have finished

        deadline = time.monotonic() + 10.0
        job_error = None
        while time.monotonic() < deadline:
            status = anyio.run(_call, url, "get_run_status", {"run_id": run_id})
            if not status.isError and status.structuredContent.get("job_error"):
                job_error = status.structuredContent["job_error"]
                break
            time.sleep(0.1)

    assert job_error is not None, "expected the background MissingDependencyError to be captured"
    assert "MissingDependencyError" in job_error or "raw_mesh" in job_error


def test_run_stage_unknown_run_id_is_a_tool_error() -> None:
    with _running_server() as url:
        result = anyio.run(_call, url, "run_stage", {"run_id": "no-such-run", "stage": "train.default"})
    assert result.isError


# --- machine control: no real Docker daemon reachable here, but must fail cleanly -------------


def test_list_containers_without_a_docker_daemon_is_a_clean_tool_error() -> None:
    """No Docker daemon is reachable in this sandbox -- proves the tool surfaces that as a normal
    MCP tool error rather than crashing the server or hanging (``pipeline.containers`` raising
    ``ContainerError`` is expected and correct here, not a bug to work around).
    """
    with _running_server() as url:
        result = anyio.run(_call, url, "list_containers", {})
    assert result.isError
