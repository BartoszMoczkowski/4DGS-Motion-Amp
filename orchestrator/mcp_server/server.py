"""Builds the FastMCP app + auth wrapper, and a ``python -m mcp_server`` entry point (T13).

T13's whole job is the transport/auth skeleton plus **one** tool (``gpu_status``) proving the
Claude <-> this server <-> ``pipeline.api`` path works end to end over real HTTP. The full tool set
(run pipeline/stage, tail logs, list/read artifacts, previews, container control, cancel) is T14 —
see ``planning/tasks/T14-mcp-tools-and-resources.md``; adding one there means calling
``@mcp.tool()`` again in :func:`build_mcp`, nothing about ``auth``/``config``/the app-wiring below
needs to change.

Async-job note for T14 (not needed by this one fast, synchronous tool): per
``ARCHITECTURE.md``'s Layer 2 section, any tool that can run long (``run_pipeline``, ``run_stage``)
must return a ``run_id`` immediately rather than blocking the request for the run's whole duration
— Claude polls ``get_status``/tails logs separately. ``gpu_status`` itself is a fast local read
(``pynvml``/``nvidia-smi``/``psutil`` queries, T12), so it just answers directly.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp

from .auth import BearerAuthMiddleware
from .config import ServerSettings, load_settings


def build_mcp() -> FastMCP:
    """A fresh :class:`FastMCP` instance with just the ``gpu_status`` connectivity-proof tool.

    ``stateless_http=True``: each HTTP request is self-contained rather than requiring a
    server-held session to persist across calls — this server has no per-session state of its own
    (``gpu_status`` reads live machine state on every call), and statelessness means restarting the
    process never strands a client mid-session. The bundled Python MCP client
    (``mcp.client.streamable_http`` + ``mcp.ClientSession``) handles this transparently either way.
    """

    mcp = FastMCP(
        "4dgs-pipeline",
        instructions=(
            "Layer 2 MCP server for the 4DGS motion-amp orchestrator. Wraps orchestrator/"
            "pipeline/api.py (Layer 1) so Claude can drive the pipeline on a machine with the "
            "GPU/Docker/Isaac Sim the Claude sandbox itself doesn't have."
        ),
        stateless_http=True,
    )

    @mcp.tool()
    def gpu_status() -> dict[str, Any]:
        """Current VRAM/RAM usage and free headroom on this machine.

        Delegates to ``pipeline.api.gpu_status`` (T12). ``None`` sub-dicts mean that dimension
        couldn't be measured here (no GPU, or no working ``pynvml``/``nvidia-smi``/``psutil``) —
        not an error.
        """
        from pipeline.api import gpu_status as _gpu_status

        return _gpu_status()

    return mcp


def build_app(settings: ServerSettings) -> ASGIApp:
    """The full ASGI app: FastMCP's streamable-HTTP app wrapped in the bearer-token gate."""

    mcp = build_mcp()
    mcp.settings.host = settings.host
    mcp.settings.port = settings.port
    inner_app = mcp.streamable_http_app()
    return BearerAuthMiddleware(inner_app, settings.token)


def main() -> None:
    """Entry point for ``python -m mcp_server``. Reads settings from the environment."""

    import uvicorn

    settings = load_settings()
    app = build_app(settings)
    print(
        f"4DGS orchestrator MCP server listening on http://{settings.host}:{settings.port}/mcp "
        "(streamable HTTP, bearer-token auth required) — see mcp_server/CONNECTING.md."
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
