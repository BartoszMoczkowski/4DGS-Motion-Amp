# T13 — MCP server over HTTP (transport + auth)

- Status: todo
- Phase: 4
- Depends on: T05
- Environment: host (WSL2)

## Goal
Stand up the Layer 2 MCP server on the WSL2 host over **HTTP** (streamable HTTP / SSE) so Claude —
local or remote — can drive the pipeline. Solves problem #2's access. (Toward Milestone M4.)

## In scope
- MCP server skeleton using an HTTP/SSE transport, bound on the WSL2 host, reachable from where
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
- Claude connects over HTTP and calls `gpu_status`, receiving live GPU info from the WSL2 host.
- Auth rejects unauthenticated calls; reachability documented and reproducible.

## Relevant existing files
Layer 1 `pipeline/api.py`. Decision recorded in INSTRUCTIONS.md (transport = HTTP).

## Notes / gotchas
Reachability from the sandbox to the WSL2 host is the crux — settle localhost vs. tunnel early.
Whitelist operations; never expose arbitrary shell.
