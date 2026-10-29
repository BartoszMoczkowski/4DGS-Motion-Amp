"""`amp train` / `amp render` CLI commands (owned by the `4dgs-core` member).

Lazy-import rule: typer (+ light stdlib needed for the option annotations) at
module scope only; ``pipeline.api``/``pipeline.artifacts`` and anything heavier
are imported inside the command functions (enforced by
``orchestrator/tests/test_amp_cli.py``'s clean-subprocess probe).

These are thin wrappers over the orchestrator's Layer-1 API (``pipeline.api``):
``amp train`` runs ``run_pipeline(preset, only=["train.default"])``; ``amp
render`` runs ``run_stage(run_id, "render.default")``, or — when
``--amp-factors`` is given — ``run_stage(run_id, "amp.default", overrides=...)``
(T26 Phase A).

Amp-factor validation reimplements ``core/render_amp.py:validate_amp_args``'s
contract locally (exactly 8 factors, -1 the only negative / skip sentinel;
freq lists len 1 (broadcast) or 8) instead of importing render_amp: render_amp
is torch-heavy at module scope, and the validation must stay light enough to
run on a CPU-only host — and inside ``amp --help``'s lazy-import guarantee.
render_amp re-validates inside the container anyway, so the two checks can't
silently diverge into a wrong render, only into an earlier error.
"""

import json
import re
from pathlib import Path
from typing import Optional

import typer

#: Positional ``--amp-factors``/``--freq-low``/``--freq-high`` order -> ``AmpConfig.channels``
#: keys. This is ``core/render_amp.py``'s parameter-slot order
#: ([means3D, means2D, scales, rotations, opacity, shs, colors_precomp, cov3D_precomp] — see its
#: ``N_AMP_PARAMS`` comment and amp-ui/amp_ui/ampUI.py's ``chanels_list``).
#: NOTE: fixed 2026-10-06 — ``pipeline.config.models.AMP_CHANNELS`` previously listed ``rotation``
#: before ``scale`` (swapping the two slots in ``AmpStage``'s positional emission); it now matches
#: this tuple. The keyed mapping here is kept anyway so the CLI is immune to any future reorder.
_AMP_FACTOR_CHANNELS = (
    "pos3d",
    "pos2d",
    "scale",
    "rotation",
    "opacity",
    "SHs",
    "color",
    "cov3D",
)

_AMP_METHODS = ("eulerian", "eulerian_abs", "eulerian_mod", "eulerian_abs_mod")


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


def _flatten_values(values: list[str], flag: str) -> list[str]:
    """Flatten a list-typed option whose items may be repeated single values
    (``--f 2 --f -1``) or quoted space/comma-separated lists (``--f "2 -1"``).

    click can't give one option occurrence ``nargs="+"`` (argparse-style
    ``--f 2 -1 -1``), so the quoted form is how the render_amp CLI's
    space-separated habit is carried over; both spellings land here.
    """
    out: list[str] = []
    for v in values:
        out.extend(tok for tok in re.split(r"[,\s]+", v.strip()) if tok)
    return out


def _parse_ints(values: list[str], flag: str) -> list[int]:
    try:
        return [int(v) for v in _flatten_values(values, flag)]
    except ValueError as e:
        raise _bad(f"{flag} takes integer values: {e}") from e


def _parse_floats(values: list[str], flag: str) -> list[float]:
    try:
        return [float(v) for v in _flatten_values(values, flag)]
    except ValueError as e:
        raise _bad(f"{flag} takes numeric values: {e}") from e


def _validate_amp_factors(factors: list[int]) -> list[int]:
    """Local reimplementation of render_amp.validate_amp_args' factor contract (see module
    docstring for why render_amp itself isn't imported)."""
    if len(factors) != len(_AMP_FACTOR_CHANNELS):
        raise _bad(
            f"--amp-factors must have exactly {len(_AMP_FACTOR_CHANNELS)} values (one per "
            f"parameter slot: {', '.join(_AMP_FACTOR_CHANNELS)}), got {len(factors)}"
        )
    for a in factors:
        if a < 0 and a != -1:
            raise _bad(
                f"negative amplification factor {a} is not allowed: -1 is the skip sentinel "
                "and the only permitted negative value"
            )
    return list(factors)


def _validate_freqs(
    freq_low: Optional[list[float]], freq_high: Optional[list[float]]
) -> Optional[list[tuple[float, float]]]:
    """Local reimplementation of render_amp.build_freq_cutoffs' contract: both-or-neither,
    equal length, len 1 (broadcast) or 8. Returns per-slot (low, high) pairs, or None."""
    n = len(_AMP_FACTOR_CHANNELS)
    if (freq_low is None) != (freq_high is None):
        raise _bad("--freq-low and --freq-high must be given together (or neither)")
    if freq_low is None:
        return None
    assert freq_high is not None
    if len(freq_low) != len(freq_high):
        raise _bad(
            f"--freq-low and --freq-high must have equal length, got {len(freq_low)} and "
            f"{len(freq_high)}"
        )
    if len(freq_low) == 1:
        return [(freq_low[0], freq_high[0])] * n
    if len(freq_low) != n:
        raise _bad(f"--freq-low/--freq-high must have length 1 (broadcast) or {n}, "
                   f"got {len(freq_low)}")
    return list(zip(freq_low, freq_high))


def _echo_dry_run(stage: str, config_slice: dict) -> None:
    typer.echo(f"[dry-run] stage: {stage}")
    typer.echo(f"[dry-run] resolved config slice for {stage}:")
    typer.echo(json.dumps(config_slice, indent=2, default=str))
    typer.echo("[dry-run] nothing executed.")


def train(
    preset: str = typer.Argument(..., help="Config preset name (see `pipeline list-presets`)."),
    scene: Optional[Path] = typer.Option(
        None,
        "--scene",
        help="4DGS multipleview scene directory, seeded as the run's external `scene` "
        "artifact. Must be an existing directory under the repo/assets/runs roots.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved stage + config slice and exit."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even if the stage's cache key is unchanged."
    ),
) -> None:
    """Train a 4DGS model (stage ``train.default``) from a config preset."""
    from pipeline import api

    try:
        resolved = api.validate_config(preset)
    except (FileNotFoundError, ValueError) as e:
        raise _bad(str(e)) from e

    from pipeline.artifacts import Artifact, validate_external_artifact_path

    external_artifacts = {}
    metadata = {}
    target_scene_dir: Optional[Path] = None

    if scene is not None:
        try:
            valid_scene_path = validate_external_artifact_path(str(scene.resolve()), "dataset")
            target_scene_dir = Path(valid_scene_path)
            external_artifacts["scene"] = Artifact(
                name="scene",
                kind="dataset",
                path=str(valid_scene_path),
                producing_stage="external",
            )
        except ValueError as e:
            raise _bad(f"--scene: {e}") from e
    else:
        model_src = resolved.get("model", {}).get("source_path", "")
        if model_src:
            candidate = Path(model_src).resolve()
            if candidate.exists():
                try:
                    valid_scene_path = validate_external_artifact_path(str(candidate), "dataset")
                    target_scene_dir = Path(valid_scene_path)
                    external_artifacts["scene"] = Artifact(
                        name="scene",
                        kind="dataset",
                        path=str(valid_scene_path),
                        producing_stage="external",
                    )
                except ValueError:
                    pass

    import hashlib
    if target_scene_dir and target_scene_dir.exists():
        offsets_file = target_scene_dir / "offsets.json"
        if offsets_file.exists():
            h = hashlib.sha256(offsets_file.read_bytes()).hexdigest()
            metadata["offsets_json_hash"] = h
            metadata["offsets_json_path"] = str(offsets_file.resolve())
            external_artifacts["offsets"] = Artifact(
                name="offsets",
                kind="json",
                path=str(offsets_file.resolve()),
                producing_stage="external",
                content_hash=f"sha256:{h}",
            )

    if dry_run:
        _echo_dry_run("train.default", api._stage_config_for("train.default", resolved))
        if "offsets_json_hash" in metadata:
            typer.echo(f"[dry-run] offsets.json hash: {metadata['offsets_json_hash']}")
        return

    if not external_artifacts:
        external_artifacts = None

    try:
        run_id = api.run_pipeline(
            preset,
            only=["train.default"],
            external_artifacts=external_artifacts,
            metadata=metadata if metadata else None,
            force=force,
        )
    except api.StageFailedError as e:
        typer.secho(f"train.default failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        typer.secho(f"train failed to start: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    typer.echo(f"train.default finished; run_id: {run_id}")


def render(
    run_id: str = typer.Argument(..., help="Existing run id (see the run registry)."),
    amp_factors: Optional[list[str]] = typer.Option(
        None,
        "--amp-factors",
        help=f"8 per-slot amplification factors ({', '.join(_AMP_FACTOR_CHANNELS)}); "
        "-1 skips a slot. Pass as one quoted list (\"2 -1 -1 -1 -1 -1 -1 -1\") or repeat the "
        "flag per value. Presence switches the stage from render.default to amp.default.",
    ),
    method: Optional[str] = typer.Option(
        None, "--method", help=f"Amplification method: one of {', '.join(_AMP_METHODS)}."
    ),
    freq_low: Optional[list[str]] = typer.Option(
        None, "--freq-low", help="Lower freq cutoff(s): 1 value (broadcast) or 8 "
        "(same quoting/repeat rules as --amp-factors)."
    ),
    freq_high: Optional[list[str]] = typer.Option(
        None, "--freq-high", help="Upper freq cutoff(s): 1 value (broadcast) or 8."
    ),
    video: Optional[str] = typer.Option(
        None, "--video", help="Output video filename (written under <model>/video/)."
    ),
    low_vram: Optional[bool] = typer.Option(
        None, "--low-vram/--no-low-vram", help="Chunked low-VRAM processing."
    ),
    frozen_cam: Optional[bool] = typer.Option(
        None, "--frozen-cam/--no-frozen-cam", help="Hold the camera fixed while amplifying."
    ),
    cluster: bool = typer.Option(
        False, "--cluster", help="Render motion clusters video (runs stage render.cluster)."
    ),
    camera: Optional[str] = typer.Option(
        None, "--camera", help="Camera name (e.g. cam01) or index for cluster video rendering."
    ),
    side_by_side: Optional[bool] = typer.Option(
        None, "--side-by-side/--no-side-by-side", help="Render RGB + cluster colored side by side."
    ),
    add_labels: Optional[bool] = typer.Option(
        None, "--add-labels/--no-add-labels", help="Overlay text labels on cluster video."
    ),
    resolution_scale: Optional[float] = typer.Option(
        None, "--resolution-scale", help="Resolution scale factor for cluster video (default: 0.5)."
    ),
    fps: Optional[int] = typer.Option(
        None, "--fps", help="Frames per second for cluster video (default: 25)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved stage + config slice and exit."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even if the stage's cache key is unchanged."
    ),
) -> None:
    """Render a run — plain (``render.default``), motion-amplified (``amp.default``),
    or cluster-colored (``render.cluster``).

    Without ``--amp-factors`` or ``--cluster`` this is a plain ``render.default`` re-render.
    With ``--amp-factors`` it runs ``amp.default``. With ``--cluster`` it runs ``render.cluster``.
    """
    from pipeline import api

    if cluster and amp_factors is not None:
        raise _bad("--cluster and --amp-factors are mutually exclusive")

    overrides: Optional[dict] = None
    stage = "render.default"
    if cluster:
        for flag_name, value in (
            ("--method", method),
            ("--freq-low/--freq-high", freq_low or freq_high),
            ("--low-vram", low_vram),
            ("--frozen-cam", frozen_cam),
        ):
            if value is not None:
                raise _bad(f"{flag_name} only makes sense together with --amp-factors, not --cluster")

        stage = "render.cluster"
        cluster_cfg: dict = {
            "camera": camera or "cam01",
            "fps": 25 if fps is None else fps,
            "resolution_scale": 0.5 if resolution_scale is None else resolution_scale,
            "side_by_side": True if side_by_side is None else side_by_side,
            "add_labels": True if add_labels is None else add_labels,
        }
        if video is not None:
            cluster_cfg["output_video"] = video
        overrides = {
            "render": {
                "impl": "cluster",
                "cluster": cluster_cfg,
            }
        }
    elif amp_factors is not None:
        for flag_name, value in (
            ("--camera", camera),
            ("--side-by-side", side_by_side),
            ("--add-labels", add_labels),
            ("--resolution-scale", resolution_scale),
            ("--fps", fps),
        ):
            if value is not None:
                raise _bad(f"{flag_name} only makes sense together with --cluster")

        stage = "amp.default"
        factors = _validate_amp_factors(_parse_ints(amp_factors, "--amp-factors"))
        freqs = _validate_freqs(
            _parse_floats(freq_low, "--freq-low") if freq_low is not None else None,
            _parse_floats(freq_high, "--freq-high") if freq_high is not None else None,
        )
        if method is not None and method not in _AMP_METHODS:
            raise _bad(f"--method must be one of {', '.join(_AMP_METHODS)}, got {method!r}")

        channels: dict[str, dict] = {}
        for i, name in enumerate(_AMP_FACTOR_CHANNELS):
            channel: dict = {"factor": float(factors[i])}
            if freqs is not None:
                channel["freq_low"], channel["freq_high"] = freqs[i]
            channels[name] = channel

        amp: dict = {"channels": channels}
        if method is not None:
            amp["method"] = method
        if video is not None:
            amp["video_path"] = video
        if low_vram is not None:
            amp["low_vram_mode"] = low_vram
        if frozen_cam is not None:
            amp["frozen_cam"] = frozen_cam
        overrides = {"amp": amp}
    else:
        for flag_name, value in (
            ("--method", method),
            ("--freq-low/--freq-high", freq_low or freq_high),
            ("--video", video),
            ("--low-vram", low_vram),
            ("--frozen-cam", frozen_cam),
        ):
            if value is not None:
                raise _bad(f"{flag_name} only makes sense together with --amp-factors")
        for flag_name, value in (
            ("--camera", camera),
            ("--side-by-side", side_by_side),
            ("--add-labels", add_labels),
            ("--resolution-scale", resolution_scale),
            ("--fps", fps),
        ):
            if value is not None:
                raise _bad(f"{flag_name} only makes sense together with --cluster")

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

    if dry_run:
        _echo_dry_run(stage, api._stage_config_for(stage, resolved))
        return

    try:
        api.run_stage(run_id, stage, force=force, overrides=overrides)
    except api.StageFailedError as e:
        typer.secho(f"{stage} failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except (ValueError, FileNotFoundError) as e:
        typer.secho(f"{stage} failed to start: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    typer.echo(f"{stage} finished; run_id: {run_id}")
