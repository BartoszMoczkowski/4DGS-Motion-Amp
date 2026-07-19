"""Bearer-token auth gate for the MCP HTTP app (T13).

A plain ASGI middleware, not Starlette's ``BaseHTTPMiddleware``. ``BaseHTTPMiddleware`` buffers an
entire response (reads it all into memory, or an internal queue) before it can inspect/forward it,
which breaks a streaming response — and MCP's streamable-HTTP transport's ``GET`` (server-initiated
notifications) responses, and its ``json_response=False`` (SSE) mode, are exactly that. A pass-
through ASGI wrapper only has to look at the *request* headers before either rejecting the call
outright or handing the untouched ``scope``/``receive``/``send`` straight to the wrapped app, so it
never has to buffer or re-emit a streamed body itself.
"""

from __future__ import annotations

import hmac

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """Rejects any HTTP request whose ``Authorization: Bearer <token>`` header doesn't match.

    Non-HTTP scopes (``lifespan``, and ``websocket`` if ever added) are passed through untouched —
    there's no per-request header to check on a ``lifespan`` scope, and FastMCP's own session
    manager needs its ``lifespan`` to run for the app to start up / shut down cleanly at all.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        scheme, _, credential = headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(credential, self._token):
            # hmac.compare_digest (constant-time) so a wrong-length/wrong-content guess can't be
            # distinguished by response-time side channel from a near-miss.
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
