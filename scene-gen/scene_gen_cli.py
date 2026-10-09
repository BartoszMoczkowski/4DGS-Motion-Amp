"""`amp exp` CLI stub (owned by the `scene-gen` member).

Lazy-import rule: typer only at module scope. Command bodies land in T26/T27
and must import numpy/matplotlib/pxr & friends inside the command functions.
"""

import typer

app = typer.Typer(help="Experiments: scene generation, grid runs, sweeps. Commands land in T26/T27.")


@app.command("status")
def status() -> None:
    """Placeholder — exp commands land in T26/T27."""
    typer.echo("The 'exp' group is a T25 skeleton; commands land in T26/T27.")
