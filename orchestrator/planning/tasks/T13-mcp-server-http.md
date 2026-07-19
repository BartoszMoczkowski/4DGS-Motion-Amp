# T13 — MCP server over HTTP (transport + auth)

- Status: done (2026-07-19) — sandbox-verifiable parts done and verified; real-machine
  reachability still open, see "Implementation notes" below.
- Phase: 4
- Depends on: T05
- Environment: host (Windows)

## Goal
Stand up the Layer 2 MCP server on the Windows host over **HTTP** (streamable HTTP / SSE) so Claude —
local or remote — can drive the pipeline. Solves problem #2's access. (Toward Milestone M4.)

## In scope
- MCP server skeleton using an HTTP/SSE transport, bound on the Windows host, reachable from where
  Claude runs.
- Auth (token/bearer) and a bind/reachability plan (localhost vs. LAN vs. tunnel) — the main
  integration unknown; document how the Claude sandbox connects.
- Wire to the Layer 1 API (T05's `run_pipeline`/`get_status`/etc.) with one trivial tool
  (`gpu_status`) as an end-to-end connectivity proof.
- Async model: long jobs return a `run_id`; no long-blocking calls.

## Out of scope
The full tool/resource set (T14).

## Deliverables
`orchestrator/mcp_server/` with an HTTP MCP server + auth + one working tool; connection docs.

## Acceptance criteria
- Claude connects over HTTP and calls `gpu_status`, receiving live GPU info from the Windows host.
- Auth rejects unauthenticated calls; reachability documented and reproducible.

## Relevant existing files
Layer 1 `pipeline/api.py`. Decision recorded in INSTRUCTIONS.md (transport = HTTP).

## Notes / gotchas
Reachability from the sandbox to the Windows host is the crux — settle localhost vs. tunnel early.
Whitelist operations; never expose arbitrary shell.

## Implementation notes (2026-07-19)

`orchestrator/mcp_server/` — `config.py` (env-driven `ServerSettings`; `PIPELINE_MCP_TOKEN` has
**no default**, raises `MissingTokenError` if unset — same "no sensible default, fail fast" rule
T10's MBS checkpoint config established), `auth.py` (`BearerAuthMiddleware`, a plain ASGI wrapper
— deliberately *not* Starlette's `BaseHTTPMiddleware`, which buffers a whole response before
forwarding it and would break streamable-HTTP/SSE's streaming responses; `hmac.compare_digest` for
a constant-time token check), `server.py` (`build_mcp()` registers the one `gpu_status` tool
delegating to `pipeline.api.gpu_status`; `build_app()` wraps FastMCP's `streamable_http_app()` in
the auth middleware; `main()` is the `python -m mcp_server` entry point), `__main__.py`.

Packaging: added an opt-in `mcp` extra to `orchestrator/pyproject.toml` (`mcp>=1.28`, the official
Python MCP SDK) and `mcp_server*` to its `packages.find.include` — kept separate from `pipeline`'s
base dependencies so importing/testing Layer 1 alone never needs `mcp`/starlette/uvicorn. Root
`pyproject.toml` gained a matching `orchestrator-mcp = ["pipeline[mcp]"]` extra alongside the
existing `orchestrator` one.

Tests (`tests/test_mcp_server.py`, 8 new, pass deterministically every run — the whole-suite total
fluctuates run to run only in `test_containers.py`'s pre-existing, unrelated sandbox-permission
errors, see T12's own log entry for that) run the **real** server — real `uvicorn` on a real loopback port in a background thread, real
`mcp.client.streamable_http`/`ClientSession` — rather than mocking anything, since nothing here
needs a GPU/Docker/native install this sandbox lacks (unlike T09-T11's fake-`exec_in_container`
strategy). Covers: `load_settings()`'s fail-fast/defaults; missing/wrong-token/wrong-scheme
requests all get a real `401`; a correctly authenticated client can `initialize()`/`list_tools()`/
`call_tool("gpu_status")` and get back this machine's real live RAM reading (not a canned value —
conftest.py's autouse "no real telemetry" fixture only patches `pipeline.resources.gating`/
`monitor`'s references, not `gpu_status()`'s own, so this genuinely round-trips through
`pipeline.api` end to end). Had to work around one sandbox-specific artifact, not a real bug:
`httpx` builds a transport for every proxy env var this dev sandbox happens to set
(`ALL_PROXY=socks5h://...` etc., for its own network allowlisting) at client-construction time
regardless of `NO_PROXY`, and the sandbox lacks the optional `socksio` dependency that scheme
needs — fixed by constructing every test's own `httpx` client/factory with `trust_env=False`
(harmless on a real machine with no such proxy configured).

Docs: `mcp_server/CONNECTING.md` (bind options — localhost/LAN/tunnel — token setup, security
notes, and an explicit "verified vs. still open" split) plus a new step 9 in
`planning/WINDOWS_SETUP.md` pointing to it.

**What's genuinely still open, matching every other real-hardware-dependent task's honest status
in this project:** starting the server for real on Bartosz's machine, and — the task's own
"main integration unknown" — which reachability option (localhost/LAN/tunnel) his actual Claude
setup needs and whether it works. This can't be resolved from inside a coding session: it depends
on where his Claude client actually runs and, for the tunnel option, on Anthropic's own sandbox
network allowlist plus how a custom MCP connector gets configured in whatever client he uses —
neither of which this repository controls. `CONNECTING.md` documents all three options and what
each needs; the acceptance criteria's own network round-trip is fully proven within what the
sandbox *can* reach (real loopback HTTP, real auth, real tool call, real payload).
