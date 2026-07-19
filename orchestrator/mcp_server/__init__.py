"""Layer 2 — MCP server over HTTP (T13).

Thin server meant to run natively on Bartosz's Windows machine, wrapping Layer 1's public API
(``pipeline.api``) so Claude — local or remote — can drive the pipeline without the Claude sandbox
needing CUDA/Isaac/Docker itself (see ``orchestrator/planning/ARCHITECTURE.md``'s "Layer 2" section).

T13's scope is deliberately narrow: stand up the HTTP/SSE transport + bearer-token auth + exactly
one tool (``gpu_status``) as an end-to-end connectivity proof. The full tool/resource set (run
pipeline/stage, tail logs, browse artifacts/previews, container control, cancel) is T14 — see
``orchestrator/planning/tasks/T14-mcp-tools-and-resources.md``.

Submodules:

- :mod:`mcp_server.config` — env-var-driven settings (host/port/bearer token); fails fast with no
  default token, same "no sensible default, fail loud" pattern as T10's MBS checkpoint config.
- :mod:`mcp_server.auth` — a small ASGI (not Starlette ``BaseHTTPMiddleware``) bearer-token gate.
  Plain ASGI is deliberate: ``BaseHTTPMiddleware`` buffers the whole response before handing it
  back, which breaks streamable-HTTP/SSE's long-lived streaming responses; a pass-through ASGI
  wrapper doesn't touch the body at all once a request is authorized.
- :mod:`mcp_server.server` — builds the ``FastMCP`` app, registers ``gpu_status``, wraps it in the
  auth middleware, and exposes a ``main()`` entry point for ``python -m mcp_server``.

Nothing in this package imports ``pipeline.api`` at module scope beyond what ``server.py``'s tool
body needs at call time — keeps ``mcp_server`` importable (for tests, introspection) even before
any pipeline run has ever happened, mirroring ``pipeline.api``'s own "lazy import per call" style.
"""
