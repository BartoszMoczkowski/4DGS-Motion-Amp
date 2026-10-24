"""`amp report` CLI subcommands (T29).

Provides single-run report generation (`amp report <run_id>`) and multi-run comparative
analysis (`amp report compare --experiment <name>`).

Lazy-import rule: typer only at module scope.
"""

from __future__ import annotations

from typing import Optional

import typer
from typer.core import TyperGroup


class ReportDefaultGroup(TyperGroup):
    """Custom TyperGroup that routes `amp report <run_id>` directly to the `run` command."""

    def get_command(self, ctx, cmd_name):
        rv = TyperGroup.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        return TyperGroup.get_command(self, ctx, "run")

    def resolve_command(self, ctx, args):
        if not args:
            return super().resolve_command(ctx, args)
        cmd_name = args[0]
        cmd = self.get_command(ctx, cmd_name)
        if cmd_name not in self.commands and cmd == self.get_command(ctx, "run"):
            return "run", cmd, args
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=ReportDefaultGroup,
    help="Generate standardized single-run and comparative evaluation reports (T29).",
    no_args_is_help=True,
)


@app.command("run")
def report_run(
    run_id: str = typer.Argument(..., help="Run ID to generate report for (e.g. pump01-12345)."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate all figures and summary markdown even if already present.",
    ),
) -> None:
    """Build single-run report bundle: house-style charts, metric table, index.md."""
    from pipeline.reporting.report import build_run_report

    typer.echo(f"Building report bundle for run '{run_id}'...")
    res = build_run_report(run_id, force=force)

    typer.echo(f"Report generated : {res['report_dir']}")
    if res.get("charts"):
        typer.echo(f"Charts generated : {', '.join(res['charts'])}")
    if res.get("mlflow_url"):
        typer.echo(f"MLflow Run Link  : {res['mlflow_url']}")


@app.command("compare")
def report_compare(
    experiment: str = typer.Option(
        ...,
        "--experiment",
        "-e",
        help="Experiment name to compare runs across (e.g. pump_grid_v2, capture_limitation).",
    ),
    impls: Optional[str] = typer.Option(
        None,
        "--impls",
        help="Comma-separated list of segmentation implementations to include (e.g. 'rigid,rigid2,kabsch').",
    ),
    metric: str = typer.Option(
        "ari",
        "--metric",
        "-m",
        help="Primary comparison metric ('ari', 'mean_iou', 'pairs_split_rate', etc.).",
    ),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        "-o",
        help="Custom output directory for comparison bundle (defaults to runs/reports/<experiment>).",
    ),
) -> None:
    """Generate comparative evaluation summary, overlaid curves, and MLflow compare UI link."""
    from pathlib import Path
    from pipeline.reporting.report import build_comparison_report

    impl_list = [i.strip() for i in impls.split(",") if i.strip()] if impls else None
    out_dir = Path(out).resolve() if out else None

    typer.echo(f"Building comparison report for experiment '{experiment}' (metric={metric})...")
    res = build_comparison_report(
        experiment=experiment,
        impls=impl_list,
        metric=metric,
        out_dir=out_dir,
    )

    if res.get("error"):
        typer.secho(f"Error: {res['error']}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Runs analyzed    : {res['run_count']}")
    typer.echo(f"Report directory : {res['report_dir']}")
    if res.get("charts"):
        typer.echo(f"Charts generated : {', '.join(res['charts'])}")
    if res.get("mlflow_compare_url"):
        typer.echo(f"MLflow Compare   : {res['mlflow_compare_url']}")
