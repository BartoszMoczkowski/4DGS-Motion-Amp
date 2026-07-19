"""Server settings, all read from the environment (T13).

Deliberately tiny and dependency-free (no ``pydantic``/``pydantic-settings`` — this is three
scalars) so it can be imported and unit-tested without pulling in the ``mcp`` package at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Same convention as ``PIPELINE_ISAAC_NATIVE_PYTHON`` etc. (T06/T11) — env vars, no config-file.
HOST_ENV_VAR = "PIPELINE_MCP_HOST"
PORT_ENV_VAR = "PIPELINE_MCP_PORT"
TOKEN_ENV_VAR = "PIPELINE_MCP_TOKEN"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class MissingTokenError(RuntimeError):
    """Raised when no bearer token is configured.

    There is deliberately no default token (same "no sensible default, fail fast" rule T10's
    ``SegmentMbsConfig.checkpoint`` established) — a server with a baked-in or empty default would
    be one accidental deploy away from an unauthenticated pipeline-control endpoint reachable from
    anywhere the host binds to.
    """


@dataclass(frozen=True)
class ServerSettings:
    """Resolved server settings for one run of the MCP server."""

    host: str
    port: int
    token: str


def load_settings() -> ServerSettings:
    """Read :class:`ServerSettings` from the environment.

    ``PIPELINE_MCP_HOST`` defaults to ``127.0.0.1`` (loopback-only — see
    ``mcp_server/CONNECTING.md`` for why this is the recommended default and what changes if
    Bartosz needs LAN/tunnel reachability instead). ``PIPELINE_MCP_PORT`` defaults to ``8765``.
    ``PIPELINE_MCP_TOKEN`` has **no default** — raises :class:`MissingTokenError` if unset.
    """

    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise MissingTokenError(
            f"{TOKEN_ENV_VAR} is not set — refusing to start the MCP server without an auth "
            "token configured. Generate one, e.g.:\n"
            '    python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            f"then set {TOKEN_ENV_VAR} to that value before starting the server. Every client "
            'must send the same value as an `Authorization: Bearer <token>` header — see '
            "mcp_server/CONNECTING.md."
        )
    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    port = int(os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT)))
    return ServerSettings(host=host, port=port, token=token)
