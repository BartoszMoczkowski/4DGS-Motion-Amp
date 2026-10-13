"""`amp mlflow` CLI (T28) — serve / prune / status for the local MLflow tracking server.

Lazy-import rule: typer only at module scope; subprocess/mlflow/pipeline imports happen
inside the command functions (mirrors ``pipeline/cli.py``, enforced by
``tests/test_amp_cli.py``).
"""

import typer

app = typer.Typer(help="Local MLflow tracking: serve, prune, status (T28).")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (loopback by default)."),
    port: int = typer.Option(5000, "--port", help="Bind port."),
) -> None:
    """Start the local MLflow server (locked layout: SQLite backend store +
    ``runs/mlflow/artifacts`` file artifact root, ``--no-serve-artifacts``).
    UI at http://127.0.0.1:5000. Ctrl+C to stop."""
    import subprocess

    from pipeline.tracking import build_serve_command, get_mlflow_root

    get_mlflow_root().mkdir(parents=True, exist_ok=True)
    cmd = build_serve_command(host=host, port=port)
    typer.echo(f"Starting MLflow server on http://{host}:{port} (Ctrl+C to stop)")
    proc = subprocess.run(cmd)
    raise typer.Exit(proc.returncode)


@app.command("prune")
def prune(
    older_than: str = typer.Option(
        ...,
        "--older-than",
        help="Hard-delete MLflow runs older than this, e.g. '30d'. Never touches runs/<id>/ manifests.",
    ),
) -> None:
    """Hard-delete old MLflow runs (and their artifact dirs under runs/mlflow/artifacts)."""
    if not older_than.endswith("d") or not older_than[:-1].isdigit():
        raise typer.BadParameter(f"--older-than must look like '30d', got {older_than!r}")
    days = int(older_than[:-1])
    from pipeline.tracking import prune as _prune

    n = _prune(days)
    typer.echo(f"Pruned {n} MLflow run(s) older than {days}d.")


@app.command("status")
def status() -> None:
    """Show the resolved tracking URI, server reachability, and run count."""
    from pipeline.tracking import server_status

    info = server_status()
    typer.echo(f"tracking URI : {info['tracking_uri']}")
    typer.echo(f"mlflow root  : {info['mlflow_root']}")
    if not info["enabled"]:
        typer.echo("tracking is DISABLED (PIPELINE_MLFLOW_DISABLE=1 or empty URI)")
        return
    if info["reachable"]:
        typer.echo(f"server       : reachable — {info['run_count']} run(s) across all experiments")
    else:
        typer.echo(f"server       : NOT reachable ({info.get('error', 'unknown error')})")
        typer.echo("start it with: amp mlflow serve")
