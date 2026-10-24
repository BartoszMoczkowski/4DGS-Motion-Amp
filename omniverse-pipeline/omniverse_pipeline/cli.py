"""`amp capture` CLI (owned by the `omniverse-pipeline` member).

Lazy-import rule: typer (+ light stdlib for annotations) at module scope only;
the reference modules (``omniverse_pipeline.split_mesh``/``add_motion``/
``omni_to_4dgs``/``capture_check``, scene-gen's ``gen_scenes``/``frames_to_mp4``)
and ``pipeline.*`` are imported inside the command functions (T26 Phase C).

Most commands are in-process calls into the light reference modules. The
argparse-``main()`` ones (split/motion/preview/gen-grid) are invoked via a
temporary ``sys.argv`` swap — in-process, no subprocess, no shelling out.
``capture run`` is the exception: it drives the vendored ``omni_capture.py``
through the Layer-1 native mechanism (``run_native_isaac_script``) and remains
native-Windows-Isaac-subprocess-only (Vulkan/WSL2 gap — see
``pipeline.stages.isaac_common``).
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Synthetic capture (USD prep, Isaac Sim capture, 4DGS conversion).")


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


def _call_argv_main(main_fn, argv: list[str], prog: str) -> None:
    """Call an argparse-style ``main()`` in-process with a temporary ``sys.argv``.

    argparse usage errors surface as ``SystemExit(2)``; convert to typer's exit so the
    CLI reports them uniformly.
    """
    import sys

    old_argv = sys.argv
    sys.argv = [prog, *argv]
    try:
        main_fn()
    except SystemExit as e:
        raise typer.Exit(int(e.code or 0)) from e
    finally:
        sys.argv = old_argv


def _flatten(values: Optional[list[str]]) -> list[str]:
    """Flatten repeated-flag / quoted-list option values (see core_cli._flatten_values)."""
    import re

    out: list[str] = []
    for v in values or []:
        out.extend(tok for tok in re.split(r"[,\s]+", v.strip()) if tok)
    return out


@app.command("split")
def split(
    in_path: Path = typer.Option(..., "--in", help="Input fused mesh (.usd/.stl/...)."),
    out: Path = typer.Option(..., "--out", help="Output segmented .usd."),
    group: str = typer.Option("CONJUNTO_BOMBAS", "--group", help="Parent Xform name."),
    min_faces: int = typer.Option(1, "--min-faces", help="Smaller components merge into nearest part."),
    preview: Optional[Path] = typer.Option(None, "--preview", help="Also write a colored preview PNG."),
    no_usd: bool = typer.Option(False, "--no-usd", help="Analysis/preview only, skip USD write."),
) -> None:
    """Split a fused mesh into per-part rigid components (split_mesh.py)."""
    from omniverse_pipeline import split_mesh

    argv = ["--in", str(in_path), "--out", str(out), "--group", group,
            "--min-faces", str(min_faces)]
    if preview is not None:
        argv += ["--preview", str(preview)]
    if no_usd:
        argv.append("--no-usd")
    _call_argv_main(split_mesh.main, argv, "split_mesh")


@app.command("motion")
def motion(
    in_path: Path = typer.Option(..., "--in", help="Input segmented .usd."),
    out: Path = typer.Option(..., "--out", help="Output animated .usd."),
    group: str = typer.Option("CONJUNTO_BOMBAS", "--group"),
    num_frames: int = typer.Option(60, "--num-frames"),
    fps: float = typer.Option(24.0, "--fps"),
    trans_amp_mm: Optional[tuple[float, float]] = typer.Option(None, "--trans-amp-mm", help="MIN MAX"),
    rot_surface_mm: Optional[tuple[float, float]] = typer.Option(None, "--rot-surface-mm", help="MIN MAX"),
    rot_deg_max: Optional[float] = typer.Option(None, "--rot-deg-max"),
    freq: Optional[tuple[int, int]] = typer.Option(None, "--freq", help="LO HI cycles per clip"),
    groups: Optional[int] = typer.Option(None, "--groups", help="K shared-motion groups (0 = per-part)."),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", help="Part names to keep static (repeatable/quoted list)."),
    seed: Optional[int] = typer.Option(None, "--seed"),
    plot: Optional[Path] = typer.Option(None, "--plot", help="Write a motion-QA image."),
) -> None:
    """Author per-part periodic motion onto a segmented USD (add_motion.py)."""
    from omniverse_pipeline import add_motion

    argv = ["--in", str(in_path), "--out", str(out), "--group", group,
            "--num-frames", str(num_frames), "--fps", str(fps)]
    if trans_amp_mm is not None:
        argv += ["--trans-amp-mm", *(str(v) for v in trans_amp_mm)]
    if rot_surface_mm is not None:
        argv += ["--rot-surface-mm", *(str(v) for v in rot_surface_mm)]
    if rot_deg_max is not None:
        argv += ["--rot-deg-max", str(rot_deg_max)]
    if freq is not None:
        argv += ["--freq", *(str(v) for v in freq)]
    if groups is not None:
        argv += ["--groups", str(groups)]
    if exclude is not None:
        argv += ["--exclude", *_flatten(exclude)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if plot is not None:
        argv += ["--plot", str(plot)]
    _call_argv_main(add_motion.main, argv, "add_motion")


@app.command("gen-grid")
def gen_grid(
    segmented_usd: Optional[str] = typer.Option(None, "--segmented-usd"),
    env: Optional[str] = typer.Option(None, "--env"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir"),
    base_amp_mm: Optional[list[str]] = typer.Option(None, "--base-amp-mm", help="Repeatable/quoted floats."),
    multipliers: Optional[list[str]] = typer.Option(None, "--multipliers", help="Repeatable/quoted floats."),
    only_cell: Optional[str] = typer.Option(None, "--only-cell", help="e.g. 16x8 — one cell only."),
    num_frames: Optional[int] = typer.Option(None, "--num-frames"),
    fps: Optional[float] = typer.Option(None, "--fps"),
    freq_hz: Optional[float] = typer.Option(None, "--freq-hz"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    capture_root: Optional[str] = typer.Option(None, "--capture-root"),
) -> None:
    """Generate the parametric grid of pump test scenes (scene-gen gen_scenes.py)."""
    import gen_scenes

    argv: list[str] = []
    for flag_name, value in (
        ("--segmented-usd", segmented_usd),
        ("--env", env),
        ("--out-dir", out_dir),
        ("--only-cell", only_cell),
        ("--num-frames", num_frames),
        ("--fps", fps),
        ("--freq-hz", freq_hz),
        ("--seed", seed),
        ("--capture-root", capture_root),
    ):
        if value is not None:
            argv += [flag_name, str(value)]
    for flag_name, values in (("--base-amp-mm", base_amp_mm), ("--multipliers", multipliers)):
        flat = _flatten(values)
        if flat:
            argv += [flag_name, *flat]
    _call_argv_main(gen_scenes.main, argv, "gen_scenes")


@app.command("run")
def capture_run(
    config: Path = typer.Option(..., "--config", help="omni_capture YAML config."),
    usd: Optional[Path] = typer.Option(None, "--usd", help="Override scene.usd_path."),
    out: Optional[Path] = typer.Option(None, "--out", help="Override output.capture_dir."),
    n_cameras: Optional[int] = typer.Option(None, "--n-cameras", help="Override rig.n_cameras."),
    frames: Optional[int] = typer.Option(None, "--frames", help="Override capture.num_frames."),
    headless: Optional[int] = typer.Option(None, "--headless", help="0|1 — override app.headless."),
) -> None:
    """Run the Isaac Sim capture (native Windows subprocess only — Vulkan/WSL2 gap).

    Uses the Layer-1 native mechanism (``run_native_isaac_script``) against the Isaac Sim
    install pointed at by ``PIPELINE_ISAAC_NATIVE_PYTHON`` (default:
    ``Q:\\Omniverse\\isaac-sim-standalone-6.0.1-windows-x86_64\\python.bat``). Logs land in a
    fresh ``runs/capture-*`` run directory.
    """
    import logging

    if not config.is_file():
        raise _bad(f"--config not found: {config}")
    if headless is not None and headless not in (0, 1):
        raise _bad(f"--headless must be 0 or 1, got {headless}")

    from pipeline import paths as paths_mod
    from pipeline.artifacts import create_run
    from pipeline.api import new_run_id
    from pipeline.stages.base import StageContext
    from pipeline.stages.isaac_common import (
        IsaacStageError,
        native_isaac_python,
        run_native_isaac_script,
    )

    launcher = native_isaac_python()
    if not launcher.is_file():
        typer.secho(
            f"native Isaac Sim python launcher not found at {launcher} — set "
            "PIPELINE_ISAAC_NATIVE_PYTHON to the python.bat under your Isaac Sim install "
            "(see orchestrator/planning/WINDOWS_SETUP.md).",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    run_id = new_run_id("capture")
    create_run(run_id, "capture", {"config": str(config), "usd": str(usd), "out": str(out)})

    argv = ["--config", str(config)]
    if usd is not None:
        argv += ["--usd", str(usd)]
    if out is not None:
        argv += ["--out", str(out)]
    if n_cameras is not None:
        argv += ["--n-cameras", str(n_cameras)]
    if frames is not None:
        argv += ["--frames", str(frames)]
    if headless is not None:
        argv += ["--headless", str(headless)]

    from pipeline.artifacts.paths import run_dir

    ctx = StageContext(
        run_id=run_id,
        stage_name="capture.isaac",
        config={},
        run_dir=run_dir(run_id),
        logger=logging.getLogger("amp.capture.run"),
        paths=paths_mod,
    )
    typer.echo(f"capture run {run_id}: launching native Isaac Sim subprocess (logs under the run dir)")
    try:
        run_native_isaac_script(ctx, "omni_capture", argv, log_name="capture")
    except IsaacStageError as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    typer.echo(f"capture finished; run_id: {run_id}")


@app.command("convert")
def convert(
    capture: Path = typer.Option(..., "--capture", help="Capture dir produced by omni_capture."),
    out: Path = typer.Option(Path("."), "--out", help="4DGS repo root (writes data/multipleview/<name>)."),
    name: str = typer.Option("omni_scene", "--name", help="Scene name."),
    near: Optional[float] = typer.Option(None, "--near"),
    far: Optional[float] = typer.Option(None, "--far"),
    target_radius: float = typer.Option(4.0, "--target-radius"),
) -> None:
    """Convert a capture dir to the 4DGS multipleview format (omni_to_4dgs.py)."""
    from omniverse_pipeline import omni_to_4dgs

    if not capture.is_dir():
        raise _bad(f"--capture directory not found: {capture}")
    kwargs = {}
    if near is not None:
        kwargs["default_near"] = near
    if far is not None:
        kwargs["default_far"] = far
    omni_to_4dgs.convert(str(capture), str(out), name, target_radius=target_radius, **kwargs)


@app.command("preview")
def preview(
    capture: Path = typer.Option(..., "--capture", help="omni_capture output dir."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Default: <capture>/videos."),
    fps: Optional[int] = typer.Option(None, "--fps"),
    tile_cols: Optional[int] = typer.Option(None, "--tile-cols"),
    tile_width: Optional[int] = typer.Option(None, "--tile-width"),
    no_per_cam: bool = typer.Option(False, "--no-per-cam", help="Skip per-camera mp4s."),
) -> None:
    """Compose capture RGB frames into mp4 previews (scene-gen frames_to_mp4.py)."""
    import frames_to_mp4

    if not capture.is_dir():
        raise _bad(f"--capture directory not found: {capture}")
    argv = ["--capture", str(capture)]
    if out_dir is not None:
        argv += ["--out-dir", str(out_dir)]
    if fps is not None:
        argv += ["--fps", str(fps)]
    if tile_cols is not None:
        argv += ["--tile-cols", str(tile_cols)]
    if tile_width is not None:
        argv += ["--tile-width", str(tile_width)]
    if no_per_cam:
        argv.append("--no-per-cam")
    _call_argv_main(frames_to_mp4.main, argv, "frames_to_mp4")


@app.command("verify")
def verify(
    capture_dir: Path = typer.Argument(..., help="Capture directory to check."),
    cams: Optional[int] = typer.Option(None, "--cams", help="Expected camera count."),
    frames: Optional[int] = typer.Option(None, "--frames", help="Expected frames per camera."),
    config: Optional[Path] = typer.Option(
        None, "--config",
        help="omni_capture YAML to read rig.n_cameras / capture.num_frames from "
        "(explicit --cams/--frames win).",
    ),
) -> None:
    """Verify a capture: cameras_gt.json, camera count, per-camera frame counts.

    Replaces the post-mortem's ad-hoc ``python -c`` frame counting. Exits 1 with a
    per-camera report when anything mismatches.
    """
    from omniverse_pipeline.capture_check import (
        load_expectations_from_config,
        verify_capture_dir,
    )

    if config is not None:
        if not config.is_file():
            raise _bad(f"--config not found: {config}")
        cfg_cams, cfg_frames = load_expectations_from_config(config)
        cams = cams if cams is not None else cfg_cams
        frames = frames if frames is not None else cfg_frames

    ok, lines = verify_capture_dir(capture_dir, expected_cams=cams, expected_frames=frames)
    for line in lines:
        typer.echo(line)
    if not ok:
        raise typer.Exit(1)
    typer.echo("verify: OK")


@app.command("prep-real")
def prep_real(
    capture: Path = typer.Option(..., "--capture", help="Directory of synchronized multi-camera frames."),
    out: str = typer.Option(..., "--out", help="Scene name (written under data/multipleview/) or destination path."),
    colmap_source: Optional[str] = typer.Option(
        None, "--colmap-source", help="Optional source scene/sparse directory to borrow calibration from."
    ),
    camera_model: str = typer.Option("SIMPLE_RADIAL", "--camera-model", help="COLMAP camera model."),
    colmap_bin: Optional[Path] = typer.Option(None, "--colmap-bin", help="Explicit COLMAP binary path."),
    force: bool = typer.Option(False, "--force", help="Re-run calibration even if outputs already exist."),
) -> None:
    """Prepare real multi-camera capture into standard 4DGS multipleview dataset format."""
    from omniverse_pipeline.prep_real import prep_real_dataset

    if not capture.is_dir():
        raise _bad(f"--capture directory not found: {capture}")

    try:
        dst = prep_real_dataset(
            capture_dir=capture,
            out_name_or_dir=out,
            colmap_source=colmap_source,
            camera_model=camera_model,
            colmap_bin=colmap_bin,
            force=force,
        )
    except FileNotFoundError as e:
        typer.secho(f"prep-real error: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e
    except subprocess.CalledProcessError as e:
        typer.secho(f"COLMAP command failed with exit code {e.returncode}", err=True, fg=typer.colors.RED)
        raise typer.Exit(e.returncode) from e
    except Exception as e:
        typer.secho(f"prep-real failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e

    typer.echo(f"prep-real finished; dataset prepared at: {dst}")
