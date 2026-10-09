"""`amp seg` CLI (owned by the `motion-seg` member).

Lazy-import rule: typer only at module scope; ``pipeline.api``/``pipeline.artifacts`` and
anything heavier are imported inside the command functions (T26 Phase B).

All commands are thin wrappers over the orchestrator's Layer-1 API: ``run_stage(run_id,
"segment.<impl>")`` for the individual segmentation impls (kabsch/multicut exist only as
vendored orchestrator stages — going through run_stage is the only correct path), and
``run_pipeline(preset, only=[...])`` for ``amp seg run`` (which replaces
``motion-seg/motion_seg/run.sh``: its ``SKIP_EXTRACT=1`` env var is now ``--skip-extract``).
"""

import json
from typing import Optional

import typer

app = typer.Typer(help="Motion segmentation (trajectory extraction, clustering, evaluation).")

#: Registered segment impls (mirrors `SegmentConfig.impl`'s Literal).
IMPLS = ("rigid", "rigid2", "kabsch", "multicut", "mbs")


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


def _resolve_run(run_id: str, overrides: Optional[dict]):
    """Load the run's manifest and apply config overrides; clear CLI errors on failure."""
    from pipeline import api
    from pipeline.artifacts import get_manifest

    try:
        manifest = get_manifest(run_id)
    except FileNotFoundError as e:
        typer.secho(f"unknown run {run_id!r}: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    try:
        resolved = api._apply_overrides(manifest.resolved_config, overrides)
    except ValueError as e:
        raise _bad(f"invalid config override: {e}") from e
    return api, resolved


def _echo_dry_run(stage: str, config_slice: dict) -> None:
    typer.echo(f"[dry-run] stage: {stage}")
    typer.echo(f"[dry-run] resolved config slice for {stage}:")
    typer.echo(json.dumps(config_slice, indent=2, default=str))
    typer.echo("[dry-run] nothing executed.")


def _run_seg_stage(run_id: str, stage: str, overrides: Optional[dict],
                   dry_run: bool, force: bool) -> None:
    """Shared body for the per-impl commands: validate, dry-run, or run_stage."""
    api, resolved = _resolve_run(run_id, overrides)
    if dry_run:
        _echo_dry_run(stage, api._stage_config_for(stage, resolved))
        return
    from pipeline.stages.registry import StageNotFoundError

    try:
        api.run_stage(run_id, stage, force=force, overrides=overrides)
    except StageNotFoundError as e:
        typer.secho(f"unknown stage {stage!r}: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except api.StageFailedError as e:
        typer.secho(f"{stage} failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except (ValueError, FileNotFoundError) as e:
        typer.secho(f"{stage} failed to start: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    typer.echo(f"{stage} finished; run_id: {run_id}")


def _prune(d: dict) -> dict:
    """Drop None values so unset flags never clobber preset/config defaults."""
    return {k: v for k, v in d.items() if v is not None}


@app.command("extract")
def extract(
    run_id: str = typer.Argument(..., help="Existing run id with a trained model."),
    n_times: Optional[int] = typer.Option(
        None, "--n-times", help="Number of trajectory timesteps (default from preset, usually 60)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the resolved stage + config slice and exit."),
    force: bool = typer.Option(False, "--force", help="Re-run even if the stage's cache key is unchanged."),
) -> None:
    """Extract per-Gaussian trajectories (``seg_extract.default``, GPU)."""
    overrides = {"seg_extract": _prune({"n_times": n_times})} if n_times is not None else None
    _run_seg_stage(run_id, "seg_extract.default", overrides, dry_run, force)


@app.command("rigid")
def rigid(
    run_id: str = typer.Argument(...),
    k: Optional[int] = typer.Option(None, "--k", help="kNN graph degree."),
    min_size: Optional[int] = typer.Option(None, "--min-size"),
    threshold_mult: Optional[float] = typer.Option(None, "--threshold-mult"),
    opacity_thresh: Optional[float] = typer.Option(None, "--opacity-thresh"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Rigidity-graph clustering (``segment.rigid``, CPU — the default Option B)."""
    overrides = {"segment": {"rigid": _prune({
        "k": k, "min_size": min_size, "threshold_mult": threshold_mult,
        "opacity_thresh": opacity_thresh,
    })}}
    _run_seg_stage(run_id, "segment.rigid", overrides, dry_run, force)


@app.command("rigid2")
def rigid2(
    run_id: str = typer.Argument(...),
    k: Optional[int] = typer.Option(None, "--k"),
    min_size: Optional[int] = typer.Option(None, "--min-size"),
    opacity_thresh: Optional[float] = typer.Option(None, "--opacity-thresh"),
    z_thresh: Optional[float] = typer.Option(None, "--z-thresh"),
    drive_freq: Optional[float] = typer.Option(None, "--drive-freq"),
    partition: Optional[str] = typer.Option(None, "--partition", help="spectral|components"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """FFT-denoised calibrated rigidity segmentation (``segment.rigid2``)."""
    if partition is not None and partition not in ("spectral", "components"):
        raise _bad(f"--partition must be spectral|components, got {partition!r}")
    overrides = {"segment": {"rigid2": _prune({
        "k": k, "min_size": min_size, "opacity_thresh": opacity_thresh,
        "z_thresh": z_thresh, "drive_freq": drive_freq, "partition": partition,
    })}}
    _run_seg_stage(run_id, "segment.rigid2", overrides, dry_run, force)


@app.command("kabsch")
def kabsch(
    run_id: str = typer.Argument(...),
    n_clusters: Optional[int] = typer.Option(None, "--n-clusters", help="0 = BIC search over k-range."),
    init: Optional[str] = typer.Option(None, "--init", help="fft|kmeans|spectral"),
    max_iter: Optional[int] = typer.Option(None, "--max-iter"),
    min_size: Optional[int] = typer.Option(None, "--min-size"),
    opacity_thresh: Optional[float] = typer.Option(None, "--opacity-thresh"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Iterative Kabsch EM rigid-body clustering (``segment.kabsch``)."""
    if init is not None and init not in ("fft", "kmeans", "spectral"):
        raise _bad(f"--init must be fft|kmeans|spectral, got {init!r}")
    overrides = {"segment": {"kabsch": _prune({
        "n_clusters": n_clusters, "init": init, "max_iter": max_iter,
        "min_size": min_size, "opacity_thresh": opacity_thresh,
    })}}
    _run_seg_stage(run_id, "segment.kabsch", overrides, dry_run, force)


@app.command("multicut")
def multicut(
    run_id: str = typer.Argument(...),
    k: Optional[int] = typer.Option(None, "--k"),
    drive_freq: Optional[float] = typer.Option(None, "--drive-freq"),
    min_size: Optional[int] = typer.Option(None, "--min-size"),
    opacity_thresh: Optional[float] = typer.Option(None, "--opacity-thresh"),
    use_appearance: Optional[bool] = typer.Option(None, "--use-appearance/--no-appearance"),
    use_motion: Optional[bool] = typer.Option(None, "--use-motion/--no-motion"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Multi-channel affinity fusion + lifted multicut (``segment.multicut``)."""
    overrides = {"segment": {"multicut": _prune({
        "k": k, "drive_freq": drive_freq, "min_size": min_size,
        "opacity_thresh": opacity_thresh,
        "use_appearance": use_appearance, "use_motion": use_motion,
    })}}
    _run_seg_stage(run_id, "segment.multicut", overrides, dry_run, force)


@app.command("mbs")
def mbs(
    run_id: str = typer.Argument(...),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="MultiBodySync .pth.tar path."),
    n_points: Optional[int] = typer.Option(None, "--n-points", help="Working-set size (default 4000)."),
    n_views: Optional[int] = typer.Option(None, "--n-views"),
    opacity_thresh: Optional[float] = typer.Option(None, "--opacity-thresh"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """MultiBodySync MotNet inference (``segment.mbs`` — Option A)."""
    overrides = {"segment": {"mbs": _prune({
        "checkpoint": checkpoint, "n_points": n_points, "n_views": n_views,
        "opacity_thresh": opacity_thresh,
    })}}
    _run_seg_stage(run_id, "segment.mbs", overrides, dry_run, force)


@app.command("eval")
def seg_eval(
    run_id: str = typer.Argument(...),
    drop_floaters: Optional[bool] = typer.Option(
        None, "--drop-floaters/--keep-floaters",
        help="Exclude GT-label -1 points from scoring (recommended for MBS output).",
    ),
    bg_label: Optional[str] = typer.Option(
        None, "--bg-label", help="GT label to exclude, or 'auto' (legacy heuristic)."
    ),
    top_n: Optional[int] = typer.Option(None, "--top-n"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Score a segmentation against ground truth (``seg_eval.default``)."""
    bg: object = None
    if bg_label is not None:
        if bg_label == "auto":
            bg = "auto"
        else:
            try:
                bg = int(bg_label)
            except ValueError:
                raise _bad(f"--bg-label must be an integer label or 'auto', got {bg_label!r}") from None
    section = _prune({"drop_floaters": drop_floaters, "bg_label": bg, "top_n": top_n})
    overrides = {"seg_eval": section} if section else None
    _run_seg_stage(run_id, "seg_eval.default", overrides, dry_run, force)


@app.command("run")
def run(
    preset: str = typer.Argument(..., help="Config preset name."),
    impl: str = typer.Option(
        "rigid", "--impl", help=f"Segmentation implementation: {', '.join(IMPLS)}."
    ),
    skip_extract: bool = typer.Option(
        False, "--skip-extract",
        help="Reuse an existing trajectories artifact (replaces run.sh's SKIP_EXTRACT=1).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the stage plan + config slices and exit."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the full segmentation chain: extract -> segment -> eval (replaces run.sh)."""
    from pipeline import api

    if impl not in IMPLS:
        raise _bad(f"--impl must be one of {', '.join(IMPLS)}, got {impl!r}")
    try:
        resolved = api.validate_config(preset)
    except (FileNotFoundError, ValueError) as e:
        raise _bad(str(e)) from e

    overrides = {"segment": {"impl": impl}}
    try:
        resolved = api._apply_overrides(resolved, overrides)
    except ValueError as e:
        raise _bad(f"invalid config override: {e}") from e

    stages = [f"segment.{impl}", "seg_eval.default"]
    if not skip_extract:
        stages.insert(0, "seg_extract.default")

    if dry_run:
        typer.echo(f"[dry-run] preset: {preset}")
        typer.echo(f"[dry-run] stage plan (only): {', '.join(stages)}")
        for stage in stages:
            _echo_dry_run(stage, api._stage_config_for(stage, resolved))
        return

    try:
        run_id = api.run_pipeline(preset, only=stages, overrides=overrides, force=force)
    except api.StageFailedError as e:
        typer.secho(f"seg run failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        typer.secho(f"seg run failed to start: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    typer.echo(f"segmentation chain finished; run_id: {run_id}")
