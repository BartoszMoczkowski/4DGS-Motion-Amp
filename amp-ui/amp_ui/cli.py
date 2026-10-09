"""`amp bench` CLI (owned by the `amp-ui` member).

Lazy-import rule: typer only at module scope. ``amp_ui.run_renders_auto`` is
torch/pandas/render_amp-heavy at module scope, so it is imported inside the
command function only (T26 Phase B).
"""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Benchmark harness (run_renders_auto).")

#: run_renders_auto's method labels (UI naming: base/base segmented/abs/abs segmented).
METHODS = ("base", "base segmented", "abs", "abs segmented")

#: --vram-modes spellings -> run_benchmark's low_vram_mode bools.
_VRAM_MODES = {"full": False, "low": True}


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


@app.command("renders")
def renders(
    models: Optional[list[str]] = typer.Option(
        None, "--models",
        help="Models to benchmark as '<dataset>/<name>' (repeatable or quoted list). "
        "Default: auto-discover all trained models under output/.",
    ),
    methods: Optional[list[str]] = typer.Option(
        None, "--methods",
        help=f"Amplification methods to test (repeatable or quoted list): {', '.join(METHODS)}.",
    ),
    repeats: Optional[int] = typer.Option(None, "--repeats", help="Iterations per combo (default 5)."),
    vram_modes: Optional[list[str]] = typer.Option(
        None, "--vram-modes", help="Subset of: full, low (default: both)."
    ),
    out_csv: Optional[Path] = typer.Option(
        None, "--out-csv", help="Results CSV path (default: <repo>/results.csv)."
    ),
    no_registry: bool = typer.Option(
        False, "--no-registry", help="Skip recording the results into the run registry."
    ),
) -> None:
    """Run the render benchmark (methods x VRAM modes x models x repeats).

    Wraps ``amp_ui.run_renders_auto.run_benchmark``; results are written to the CSV
    atomically after every combo (a crash keeps completed rows) and additionally recorded
    as a run-registry artifact (a JSON summary under a fresh ``bench-*`` run) unless
    ``--no-registry`` is given.
    """
    import re

    def flatten(values):
        out = []
        for v in values or []:
            out.extend(tok for tok in re.split(r"[,\s]+", v.strip()) if tok)
        return out

    models = flatten(models) or None
    # Method labels contain spaces ("base segmented"), so only comma-split there —
    # whitespace-splitting would corrupt them.
    method_list = None
    if methods:
        method_list = []
        for v in methods:
            method_list.extend(tok.strip() for tok in v.split(",") if tok.strip())
    vram_list = flatten(vram_modes) or None

    if method_list is not None:
        for m in method_list:
            if m not in METHODS:
                raise _bad(f"--methods entries must be among {', '.join(METHODS)}; got {m!r}")
    if vram_list is not None:
        for v in vram_list:
            if v not in _VRAM_MODES:
                raise _bad(f"--vram-modes entries must be full|low; got {v!r}")
        vram_list = [_VRAM_MODES[v] for v in vram_list]
    if repeats is not None and repeats < 1:
        raise _bad(f"--repeats must be >= 1, got {repeats}")

    try:
        from amp_ui.run_renders_auto import run_benchmark
    except ImportError as e:
        # run_renders_auto imports torch + core/render_amp.py (the 4DGS renderer chain),
        # which only exists in the Linux/CUDA environment.
        typer.secho(
            f"the render benchmark needs the Linux/CUDA 4DGS environment (import failed: {e})",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from e

    try:
        rows = run_benchmark(
            models=models,
            methods=method_list,
            vram_modes=vram_list,
            repeats=repeats,
            out_csv=str(out_csv) if out_csv is not None else None,
        )
    except SystemExit as e:
        # resolve_model_specs' "model not found / no config" errors.
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e

    if no_registry:
        return

    # Record the benchmark into the run registry: a fresh `bench-*` run whose manifest
    # carries the benchmark spec as its resolved config and a JSON summary artifact
    # (the CSV itself stays where run_benchmark wrote it and is referenced from metadata).
    import json as _json
    from datetime import datetime, timezone

    from pipeline import api
    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.artifacts.paths import run_dir

    run_id = api.new_run_id("bench")
    csv_path = str(out_csv) if out_csv is not None else None
    create_run(
        run_id,
        "bench",
        {
            "kind": "bench.renders",
            "models": models,
            "methods": method_list,
            "vram_modes": vram_list,
            "repeats": repeats,
            "out_csv": csv_path,
        },
    )
    summary = {
        "benchmark": "run_renders_auto",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "methods": method_list,
        "vram_modes": vram_list,
        "repeats": repeats,
        "results_csv": csv_path,
        "rows": len(rows),
        "errors": sum(1 for r in rows if r[-1]),
    }
    summary_path = run_dir(run_id) / "bench_results.json"
    summary_path.write_text(_json.dumps(summary, indent=2), encoding="utf-8")

    def _record(m):
        m.artifacts["bench_results"] = Artifact(
            name="bench_results",
            kind="json",
            path=str(summary_path),
            producing_stage="bench.renders",
            metadata={"results_csv": csv_path, "rows": len(rows), "errors": summary["errors"]},
        )
        m.status = "success"

    update_manifest(run_id, _record)
    typer.echo(f"Recorded benchmark as run {run_id} (artifact: {summary_path})")
