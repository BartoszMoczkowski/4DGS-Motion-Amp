"""T13 acceptance tests: the MCP server, over a *real* HTTP loopback connection.

Unlike T09/T10/T11's fake-``exec_in_container`` strategy (needed there because the real
dependency is a GPU/Docker/native Isaac install this sandbox doesn't have), there's nothing to
fake here: ``gpu_status`` is a fast, GPU-less-safe read (T12's ``conftest.py`` autouse fixture
already makes every test's telemetry query return ``None``, the same "can't measure" value a
real GPU-less machine would give), and the transport itself — uvicorn + Starlette + the MCP
streamable-HTTP protocol — is exactly the real thing T13 needs verified, not a mock of it. So
these tests spin up the *actual* server (``mcp_server.server.build_app``) on a real loopback TCP
port in a background thread and talk to it with the *actual* MCP Python client
(``mcp.client.streamable_http`` + ``mcp.ClientSession``), proving the literal acceptance
criteria: "Claude connects over HTTP and calls ``gpu_status``" / "auth rejects unauthenticated
calls" — not an approximation of them.

Real LAN/tunnel reachability from an actual remote Claude session to Bartosz's Windows machine is
explicitly *not* provable from the sandbox (no such network path exists here) — see
``mcp_server/CONNECTING.md`` for what's verified here vs. what still needs confirming on his
machine.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator

import anyio
import httpx
import pytest
import uvicorn

from mcp_server.config import MissingTokenError, ServerSettings, load_settings
from mcp_server.server import build_app

TEST_TOKEN = "test-token-do-not-use-for-real"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_client(**kwargs) -> httpx.Client:
    """A plain ``httpx.Client`` that ignores this *sandbox's* proxy env vars.

    This dev sandbox sets ``ALL_PROXY``/``HTTP_PROXY``/etc. (routing normal outbound traffic
    through a SOCKS/HTTP proxy for network allowlisting) — nothing to do with this test, which
    only ever talks to its own loopback port, but ``httpx`` builds a transport for every proxy
    env var at client-construction time regardless of ``NO_PROXY``, and this sandbox doesn't have
    the optional ``socksio`` dependency the SOCKS one needs. ``trust_env=False`` skips all of
    that. Not needed on a real deployment machine with no such proxy configured, but harmless
    there either way.
    """

    return httpx.Client(trust_env=False, **kwargs)


def _mcp_http_client_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    """Same ``trust_env=False`` fix as :func:`_http_client`, shaped for
    ``streamablehttp_client``'s ``httpx_client_factory=`` parameter (which otherwise builds its
    ``httpx.AsyncClient`` with the default ``trust_env=True``)."""

    import httpx as _httpx

    kwargs: dict = {"trust_env": False, "follow_redirects": True}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return _httpx.AsyncClient(**kwargs)


@contextlib.contextmanager
def _running_server(token: str = TEST_TOKEN) -> Iterator[str]:
    """Runs the real app on a free loopback port in a background thread; yields its /mcp URL."""

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


# --- config -------------------------------------------------------------------


def test_load_settings_raises_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_MCP_TOKEN", raising=False)
    with pytest.raises(MissingTokenError):
        load_settings()


def test_load_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_MCP_TOKEN", "abc123")
    monkeypatch.setenv("PIPELINE_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("PIPELINE_MCP_PORT", "9999")
    settings = load_settings()
    assert settings == ServerSettings(host="0.0.0.0", port=9999, token="abc123")


def test_load_settings_defaults_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_MCP_TOKEN", "abc123")
    monkeypatch.delenv("PIPELINE_MCP_HOST", raising=False)
    monkeypatch.delenv("PIPELINE_MCP_PORT", raising=False)
    settings = load_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765


# --- auth, over a real connection ----------------------------------------------


def test_missing_authorization_header_is_rejected() -> None:
    with _running_server() as url, _http_client() as client:
        resp = client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"accept": "application/json, text/event-stream"},
        )
        assert resp.status_code == 401


def test_wrong_token_is_rejected() -> None:
    with _running_server() as url, _http_client() as client:
        resp = client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "accept": "application/json, text/event-stream",
                "authorization": "Bearer not-the-right-token",
            },
        )
        assert resp.status_code == 401


def test_malformed_scheme_is_rejected() -> None:
    """Not just "wrong token" — a non-Bearer scheme (or a bare token, no scheme) must also 401."""
    with _running_server() as url, _http_client() as client:
        resp = client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "accept": "application/json, text/event-stream",
                "authorization": TEST_TOKEN,  # no "Bearer " prefix
            },
        )
        assert resp.status_code == 401


# --- the real connectivity proof: initialize + call gpu_status -----------------


def test_gpu_status_over_real_http_with_valid_token() -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    with _running_server() as url:

        async def _run() -> dict:
            async with streamablehttp_client(
                url,
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                httpx_client_factory=_mcp_http_client_factory,
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert any(t.name == "gpu_status" for t in tools.tools)
                    result = await session.call_tool("gpu_status", {})
                    assert not result.isError
                    assert result.structuredContent is not None
                    return result.structuredContent

        payload = anyio.run(_run)

    # This is the real, unmocked pipeline.api.gpu_status() call (T12) — conftest.py's autouse
    # fixture only patches the query functions pipeline.resources.gating/monitor reference, not
    # gpu_status() itself, so it genuinely queries this sandbox's real psutil-visible RAM. No GPU
    # exists here, so "gpu" is None; "ram" comes back a real, live reading — proving the full
    # Claude -> HTTP -> auth -> MCP tool -> pipeline.api -> pipeline.resources chain actually ran,
    # not just that some canned value was echoed back.
    assert payload["gpu"] is None
    assert isinstance(payload["ram"], dict)
    for key in ("total_mb", "free_mb", "used_mb"):
        assert isinstance(payload["ram"][key], (int, float))
        assert payload["ram"][key] >= 0


def test_gpu_status_rejected_without_token_even_at_protocol_level() -> None:
    """The auth gate runs before any MCP framing is parsed — initialize() itself must fail."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    with _running_server() as url:

        async def _run() -> None:
            async with streamablehttp_client(
                url, httpx_client_factory=_mcp_http_client_factory
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()

        with pytest.raises(Exception):
            anyio.run(_run)
