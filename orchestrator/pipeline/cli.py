"""`amp orch` CLI (owned by the `pipeline` / orchestrator member).

Lazy-import rule: typer only at module scope; os/sys/subprocess & friends are
imported inside the command functions (T26).
"""

import typer

app = typer.Typer(help="Orchestrator: MCP server and Streamlit UIs.")


@app.command("mcp")
def mcp(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (loopback by default)."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
) -> None:
    """Start the Layer-2 HTTP MCP server (``python -m mcp_server``).

    Requires ``PIPELINE_MCP_TOKEN`` to already be set in the environment — the token is the
    server's only auth, so it is never defaulted or generated here. Generate one with
    ``python -c "import secrets; print(secrets.token_urlsafe(32))"``.
    """
    import os
    import subprocess
    import sys

    if not os.environ.get("PIPELINE_MCP_TOKEN"):
        typer.secho(
            "PIPELINE_MCP_TOKEN is not set — refusing to start the MCP server without an "
            "auth token (it would be an unauthenticated pipeline-control endpoint). "
            "Generate one, e.g.:\n"
            '    python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "then set PIPELINE_MCP_TOKEN to that value before starting the server.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)

    env = {
        **os.environ,
        "PIPELINE_MCP_HOST": host,
        "PIPELINE_MCP_PORT": str(port),
    }
    typer.echo(f"Starting MCP server on http://{host}:{port}/mcp (Ctrl+C to stop)")
    proc = subprocess.run([sys.executable, "-m", "mcp_server"], env=env)
    raise typer.Exit(proc.returncode)


@app.command("ui")
def ui(
    app_name: str = typer.Option(
        "orchestrator",
        "--app",
        help="Which UI to launch: 'orchestrator' (orchestrator/ui/app.py) or 'amp' "
        "(amp-ui/amp_ui/ampUI.py, the standalone motion-amp UI).",
    ),
) -> None:
    """Start one of the Streamlit UIs."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    targets = {
        "orchestrator": repo_root / "orchestrator" / "ui" / "app.py",
        "amp": repo_root / "amp-ui" / "amp_ui" / "ampUI.py",
    }
    target = targets.get(app_name)
    if target is None:
        raise typer.BadParameter(
            f"--app must be one of {', '.join(sorted(targets))}, got {app_name!r}"
        )
    if not target.is_file():
        typer.secho(f"UI entry point not found: {target}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"Starting Streamlit UI: {target}")
    proc = subprocess.run([sys.executable, "-m", "streamlit", "run", str(target)])
    raise typer.Exit(proc.returncode)
