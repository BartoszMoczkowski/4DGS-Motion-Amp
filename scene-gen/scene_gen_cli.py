"""`amp exp` CLI (owned by the `scene-gen` member) — T27 declarative experiments.

Lazy-import rule: typer only at module scope. ``pipeline.experiments`` and anything heavier
are imported inside the command functions (the ``amp --help`` probe forbids numpy & friends at
import time).
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Declarative experiments: run/status/compare/export (T27).")


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


def _load(spec: str):
    from pipeline.experiments.spec import load_spec

    try:
        return load_spec(spec)
    except (FileNotFoundError, ValueError) as e:
        raise _bad(str(e)) from e


@app.command("run")
def run(
    spec: str = typer.Argument(..., help="Experiment name (in experiments/) or YAML path"),
    only_cells: Optional[str] = typer.Option(None, "--only-cells", help="Comma-separated cells"),
    only_impls: Optional[str] = typer.Option(None, "--only-impls", help="Comma-separated impls"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan, touch nothing"),
) -> None:
    """Run (or plan) an experiment: cells x sweep variants x segment impls."""
    cells = [c.strip() for c in only_cells.split(",")] if only_cells else None
    impls = [i.strip() for i in only_impls.split(",")] if only_impls else None
    from pipeline.experiments.runner import run_experiment

    try:
        run_experiment(spec, dry_run=dry_run, only_cells=cells, only_impls=impls,
                       log=lambda m: typer.echo(m))
    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    if dry_run:
        typer.echo("[dry-run] nothing executed.")


@app.command("list")
def list_specs() -> None:
    """List experiment specs found under experiments/."""
    from pipeline.experiments.spec import list_experiments

    names = list_experiments()
    if not names:
        typer.echo("no experiments found (experiments/*.yaml)")
        return
    for name in names:
        typer.echo(name)


@app.command("status")
def status(spec: str = typer.Argument(...)) -> None:
    """Per-variant run status, read from run manifests (metadata experiment/cell/variant)."""
    loaded = _load(spec)
    from pipeline.experiments.runner import _find_runs, _pipeline_done

    found = _find_runs(loaded.name)
    if not found:
        typer.echo(f"no runs tagged experiment={loaded.name!r}")
        return
    for (cell, variant), manifest in sorted(found.items()):
        done = _pipeline_done(manifest, loaded.stages)
        impls = sorted(
            a.removeprefix("seg_eval_result_")
            for a in manifest.artifacts
            if a.startswith("seg_eval_result_")
        )
        typer.echo(
            f"{variant}: run_id={manifest.run_id} status={manifest.status} "
            f"pipeline={'done' if done else 'incomplete'} seg_impls={','.join(impls) or '-'}"
        )


@app.command("compare")
def compare(
    spec: str = typer.Argument(...),
    metric: str = typer.Option("ari", "--metric", help="Store column to pivot on"),
) -> None:
    """Impl x variant pivot table of a metric, read from the results store."""
    loaded = _load(spec)
    from pipeline.experiments import results as res

    rows = [r for r in res.collect_results() if r["experiment"] == loaded.name and r["impl"]]
    if not rows:
        typer.echo(f"no seg results for experiment {loaded.name!r} (run `amp exp export` first?)")
        return
    variants = sorted({r["variant"] for r in rows})
    impls = sorted({r["impl"] for r in rows})
    table = {(r["impl"], r["variant"]): r.get(metric, "") for r in rows}
    typer.echo("impl\t" + "\t".join(variants))
    for impl in impls:
        cells = []
        for v in variants:
            val = table.get((impl, v), "")
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        typer.echo(impl + "\t" + "\t".join(cells))


@app.command("export")
def export(
    out: Optional[Path] = typer.Option(
        None, "--out", help="Store path (default: <runs_root>/experiment_results.csv)"
    ),
) -> None:
    """Rebuild the unified results store from run manifests (source of truth)."""
    from pipeline.experiments import results as res

    path = res.export_csv(res.collect_results(), out)
    typer.echo(f"wrote {path}")
