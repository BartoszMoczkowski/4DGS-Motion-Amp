"""Root `amp` CLI — mounts each workspace member's Typer sub-app.

Lazy-import rule: this module (and every member `cli.py` it mounts) must stay
importable without torch/docker/mcp/streamlit/numpy. Heavy imports happen
inside command functions only (command bodies land in T26+). A sandbox test
(`orchestrator/tests/test_amp_cli.py`) enforces this.
"""

import typer

from amp_ui.cli import app as bench_app
from core_cli import render as render_command
from core_cli import train as train_command
from motion_seg.cli import app as seg_app
from omniverse_pipeline.cli import app as capture_app
from pipeline.cli import app as orch_app
from pipeline.explore_cli import explore as explore_command
from pipeline.reporting.cli import app as report_app
from pipeline.tracking_cli import app as mlflow_app
from scene_gen_cli import app as exp_app

app = typer.Typer(
    name="amp",
    help="4DGS motion amplification — unified CLI (capture, train, render, seg, amp).",
    no_args_is_help=True,
)

app.add_typer(capture_app, name="capture")
# T26: `amp train` / `amp render` are single commands (each takes a positional argument), not
# command groups — register core_cli's command functions directly on the root app.
app.command("train")(train_command)
app.command("render")(render_command)
# T30: `amp explore` single unified inspection navigator
app.command("explore")(explore_command)
app.add_typer(seg_app, name="seg")
app.add_typer(exp_app, name="exp")
app.add_typer(orch_app, name="orch")
app.add_typer(bench_app, name="bench")
app.add_typer(mlflow_app, name="mlflow")
app.add_typer(report_app, name="report")
