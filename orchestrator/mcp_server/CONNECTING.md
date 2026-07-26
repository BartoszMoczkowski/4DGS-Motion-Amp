# Connecting to the MCP server (T13)

How to start the Layer 2 server on Bartosz's Windows machine and reach it from wherever Claude
runs. Read this alongside `planning/ARCHITECTURE.md`'s "Layer 2" section and
`planning/INSTRUCTIONS.md`'s locked "MCP transport" decision.

## What's actually verified vs. still open

**Verified in the sandbox (this task):** the server itself — FastMCP's streamable-HTTP transport,
the bearer-token auth gate, and the one `gpu_status` tool — all work correctly over a real
loopback HTTP connection (`tests/test_mcp_server.py`, real `uvicorn` + real MCP Python client, not
mocked). Unauthenticated/wrong-token/wrong-scheme requests all get a real `401`; a correctly
authenticated client can `initialize()`, `list_tools()`, and `call_tool("gpu_status", {})` and get
back this machine's real live RAM reading (see that test file's own docstring for why the
sandbox's *own* RAM shows up as a real number rather than a canned value).

**Still open — needs Bartosz's own machine to confirm, same "not yet verified for real" status
every other real-hardware-dependent task in this project has carried until someone actually ran
it:**
1. Starting the server for real on his Windows box (native Windows Python, not a container —
   nothing about this server needs Docker/GPU/Isaac, so no container setup is needed here at
   all).
2. Whichever reachability option below he actually needs, reached from wherever his Claude
   session actually runs. **This is the one genuine integration unknown flagged in
   `ARCHITECTURE.md`'s "Cross-cutting risks"** — it depends on details this repo can't see
   (whether Claude runs directly on that Windows machine, on his LAN, or in a hosted sandbox like
   this one whose outbound network is allowlisted by Anthropic) and can't be settled from inside
   this coding session.

## Starting the server

```powershell
# one-time: install the Layer 2 extra (Layer 1's `pipeline` plus `mcp`)
uv sync --package pipeline --extra mcp

# generate a token once, save it somewhere safe (a password manager, not a repo file)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# every time you start the server
$env:PIPELINE_MCP_TOKEN = "<paste the token>"
uv run --package pipeline --extra mcp python -m mcp_server
```

Reads three env vars (`mcp_server/config.py`):

| Var | Default | Notes |
|---|---|---|
| `PIPELINE_MCP_TOKEN` | *(none — required)* | Server refuses to start without it. Every client sends it back as `Authorization: Bearer <token>`. |
| `PIPELINE_MCP_HOST` | `127.0.0.1` | Loopback-only by default — see "Bind options" below. |
| `PIPELINE_MCP_PORT` | `8765` | |

The server logs `http://<host>:<port>/mcp` on startup — that full URL (including the `/mcp` path)
is what a client connects to.

## Bind options — pick based on where Claude actually runs

**1. Localhost (`127.0.0.1`, the default).** Use this if Claude runs as a process on the *same*
Windows machine (e.g. Claude Code, or a local MCP-capable client configured to reach
`http://127.0.0.1:8765/mcp`). Simplest and most secure — nothing is exposed to the network at all,
the bearer token is really just defense-in-depth against another local process. **This is the
recommended default whenever it's an option.**

**2. LAN (`PIPELINE_MCP_HOST=0.0.0.0`, plus a Windows Firewall inbound rule for the port).** Use if
Claude runs on a different device on the same home network (e.g. a laptop/phone client on the same
Wi-Fi). The bearer token now matters for real — anything on that LAN could otherwise reach it.
Still plain HTTP, not HTTPS: acceptable for a private home LAN, not for anything less trusted.

**3. Tunnel (Tailscale Funnel / Cloudflare Tunnel / ngrok, terminated in front of this server).**
Needed if Claude runs somewhere with no direct network path to the Windows machine at all — e.g.
**this Cowork sandbox**, whose outbound network access is allowlisted by Anthropic rather than
open to arbitrary hosts. Two separate things have to line up before this option works, neither of
which this repo controls:
   - The tunnel's public hostname needs to be reachable from wherever Claude's sandbox runs (an
     Anthropic-side network allowlist decision, not a setting in this project).
   - The MCP client itself needs to be told to connect to that tunnel URL — for Cowork/Claude
     Desktop, that means adding this server as a custom MCP connector in the app's own settings,
     not something `orchestrator/` can configure from inside the repo.

   Keep the server bound to `127.0.0.1` and let the tunnel software do the actual internet-facing
   listening (most tunnel tools support this — e.g. `cloudflared tunnel --url http://127.0.0.1:8765`)
   rather than binding this server itself to `0.0.0.0` *and* tunnelling it, so there's exactly one
   process actually exposed to the internet, and it's one built for that job.

## Tools & resources

T13 verified only the transport/auth + one connectivity-proof tool (`gpu_status`). T14 added the
full control + read surface — see `TOOLS.md` for the complete tool/resource reference (written for
Claude to read directly, not just this file's own human audience).

## Security notes

- Whitelisted operations only (T14's full tool set — `TOOLS.md`) — never a generic shell/exec tool
  (see `ARCHITECTURE.md`'s Layer 2 bullet and `INSTRUCTIONS.md`'s ground rules).
- The bearer check (`mcp_server/auth.py`) uses `hmac.compare_digest` (constant-time) specifically
  so a wrong guess can't be distinguished from a near-miss by response timing.
- Plain HTTP (no TLS) is fine for options 1–2 above; anything crossing an untrusted network
  (option 3, if not already terminated by the tunnel's own TLS) needs the tunnel tool's TLS, not
  this server's — it deliberately doesn't implement its own TLS termination, one job at a time.
- Rotate the token by generating a new one and restarting the server (`PIPELINE_MCP_TOKEN`) — no
  token-rotation/expiry mechanism exists yet (not needed for T13's scope; revisit if this ever
  becomes multi-user).
