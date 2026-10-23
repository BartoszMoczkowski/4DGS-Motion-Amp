"""CLI command for `amp explore` (T30).

Provides structured JSON exploration by default, or formatted markdown/tables with `--human`.
"""

from __future__ import annotations

import json
from typing import Optional

import typer


def explore(
    target: Optional[str] = typer.Argument(
        None,
        help="Target experiment name, run ID, or capture path (omit for workspace overview).",
    ),
    view: str = typer.Option(
        "default",
        "--view",
        help="Detail level: 'default', 'full', 'diff', or 'metrics'.",
    ),
    human: bool = typer.Option(
        False,
        "--human",
        help="Print human-readable tables/markdown instead of structured JSON.",
    ),
    include_unregistered: bool = typer.Option(
        False,
        "--include-unregistered",
        help="Scan run directories for unregistered/un-manifested files.",
    ),
) -> None:
    """Traverse experiments, run configs, stage progress, and results in ONE call."""
    from pipeline.explore import explore as _explore
    from pipeline.explore import format_explore_human

    data = _explore(
        target=target,
        view=view,
        include_unregistered=include_unregistered,
    )

    if human:
        typer.echo(format_explore_human(data))
    else:
        typer.echo(json.dumps(data, indent=2, default=str))
