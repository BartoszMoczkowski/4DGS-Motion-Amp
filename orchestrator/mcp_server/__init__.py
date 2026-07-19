"""Layer 2 — MCP server over HTTP (T13 transport/auth, T14 full tool/resource set).

Thin server meant to run natively on Bartosz's Windows machine, wrapping Layer 1's public API
(``pipeline.api``) so Claude — local or remote — can drive the pipeline without the Claude sandbox
needing CUDA/Isaac/Docker itself (see ``orchestrator/planning/ARCHITECTURE.md``'s "Layer 2" section).

T13 stood up the HTTP/SSE transport + bearer-token auth + exactly one tool (``gpu_status``) as an
end-to-end connectivity proof. T14 (``orchestrator/planning/tasks/T14-mcp-tools-and-resources.md``)
filled in the rest: run pipeline/stage (async — see :mod:`mcp_server.jobs`), status/log polling,
artifact discovery + shaped summaries/previews (:mod:`mcp_server.artifact_view`), container
control, and manifest/log/artifact MCP resources. See ``mcp_server/TOOLS.md`` for the full tool
reference (a usage doc written for Claude, not just this codebase's own contributors).

Submodules:

- :mod:`mcp_server.config` — env-var-driven settings (host/port/bearer token); fails fast with no
  default token, same "no sensible default, fail loud" pattern as T10's MBS checkpoint config.
- :mod:`mcp_server.auth` — a small ASGI (not Starlette ``BaseHTTPMiddleware``) bearer-token gate.
  Plain ASGI is deliberate: ``BaseHTTPMiddleware`` buffers the whole response before handing it
  back, which breaks streamable-HTTP/SSE's long-lived streaming responses; a pass-through ASGI
  wrapper doesn't touch the body at all once a request is authorized.
- :mod:`mcp_server.jobs` (T14) — turns ``pipeline.api``'s synchronous ``run_pipeline``/``run_stage``
  into fire-and-forget background work, so those two tools can return a ``run_id`` immediately
  instead of blocking the request for however long the DAG takes.
- :mod:`mcp_server.artifact_view` (T14) — per-artifact-kind result shaping for ``read_artifact``/
  ``get_preview``: summaries and previews, never a raw multi-GB blob.
- :mod:`mcp_server.server` — builds the ``FastMCP`` app, registers every tool/resource, wraps it in
  the auth middleware, and exposes a ``main()`` entry point for ``python -m mcp_server``.

Nothing in this package imports ``pipeline.api``/``pipeline.dag``/``pipeline.containers`` at module
scope — only the two leaf, docker/torch/pynvml-free packages tool type hints actually need
resolvable at registration time (``pipeline.artifacts``, ``pipeline.paths``); everything else is a
lazy ``from pipeline... import ...`` inside each tool body, mirroring ``pipeline.api``'s own "lazy
import per call" style. Keeps ``mcp_server`` importable (for tests, introspection) even before any
pipeline run has ever happened, and without a reachable Docker daemon/GPU.
"""
