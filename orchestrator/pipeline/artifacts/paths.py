"""Run directory layout: where runs, manifests, config snapshots, and logs live on disk.

This is **not** the path-*translation* module (mapping host <-> container paths is T06's job,
"the only place that logic may live" per ``planning/INSTRUCTIONS.md``) — this module
only knows about one convention on the host: everything for a run lives under
``runs/<run_id>/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: orchestrator/pipeline/artifacts/paths.py -> repo root (4DGS-Motion-Amp/).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default root for all run directories, sibling to ``output/``/``data/``. Overridable per-call
#: (tests, alternate hosts) via the ``runs_root`` kwarg most functions here take, or globally via
#: the ``PIPELINE_RUNS_ROOT`` env var (see :func:`get_runs_root`).
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

_ENV_VAR = "PIPELINE_RUNS_ROOT"


def get_runs_root() -> Path:
    """The runs root to use: ``$PIPELINE_RUNS_ROOT`` if set, else :data:`DEFAULT_RUNS_ROOT`.

    Read at call time (not import time) so tests can flip it via ``monkeypatch.setenv`` without
    reloading this module, and so importing ``pipeline`` never touches the filesystem.
    """

    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else DEFAULT_RUNS_ROOT


def run_dir(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return (runs_root or get_runs_root()) / run_id


def manifest_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "manifest.json"


def config_snapshot_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "config_snapshot.json"


def log_dir(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "logs"


def stage_log_path(run_id: str, stage: str, *, runs_root: Optional[Path] = None) -> Path:
    # stage names look like "segment.mbs" — keep the dot out of the filename's extension slot.
    return log_dir(run_id, runs_root=runs_root) / f"{stage.replace('.', '_')}.log"


def ensure_run_dirs(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    """Create ``runs/<run_id>/`` and its ``logs/`` subdir if missing. Returns the run dir."""

    d = run_dir(run_id, runs_root=runs_root)
    log_dir(run_id, runs_root=runs_root).mkdir(parents=True, exist_ok=True)
    return d
