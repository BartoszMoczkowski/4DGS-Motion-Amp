"""Thin adapter between the Streamlit app (``app.py``) and Layer 1's public API.

T15's own scope note ("no pipeline logic lives in the UI — it only calls the API") means this
module must not reimplement anything ``pipeline.api`` (or the read-only helpers it delegates to)
already does. Concretely, every function here does one of:

- Call straight into ``pipeline.api`` (the same module Layer 2's MCP server calls).
- Reuse ``mcp_server.jobs``'s background-thread wrapper so ``run_pipeline``/``run_stage`` (which
  block for the run's own duration — training alone can take hours) don't freeze the whole
  Streamlit script thread. ``mcp_server.jobs`` has no ``mcp`` package import at module scope (pure
  ``threading``/``dataclasses``) — importing it here does not pull in the optional ``mcp`` extra
  ``orchestrator/pyproject.toml`` declares for the HTTP server.
- Reuse ``mcp_server.artifact_view``'s per-kind artifact summarization (also ``mcp``-free) so the
  "browse an artifact" view shows the exact same shape Claude sees over MCP, rather than a second,
  possibly-drifting implementation.
- Read run/log data via ``pipeline.artifacts``'s existing read surface (``get_runs_root``,
  ``stage_log_path``) — the same functions T14's own ``tail_logs`` tool uses.

**Direct-import decision (T15, documented per the task spec's "pick one and document"):** this UI
talks to Layer 1 by importing ``pipeline``/``mcp_server`` directly, in the same Python process as
Streamlit — not over the T14 HTTP/MCP server. Rationale: the UI's whole reason to exist is as a
thin panel for Bartosz on the *same* Windows machine that already runs Docker Desktop/the GPU (see
``ARCHITECTURE.md``'s Layer 3 note) — unlike an MCP client, which might be remote, this UI has no
reason to hop through a network+auth boundary to reach code running on the same box. This also
means the UI has no dependency on the MCP server process being up at all. The trade-off: this
module (and therefore the UI) can only run in an environment with Layer 1's own dependencies
installed (``pipeline``'s base deps — docker, pynvml, psutil, ...), same as running the pipeline
directly would require anyway.

One new, genuinely-UI-only piece of state lives here: :func:`save_preset_variant` writes a new
preset YAML file. This is *config*, not pipeline logic (``INSTRUCTIONS.md``'s "config is the
single source of truth" rule already treats presets as data files a human — or a thin UI on their
behalf — edits directly), so it doesn't violate the "no pipeline logic in the UI" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# --- bootstrap: make `pipeline`/`mcp_server` importable regardless of how this app is launched --
#
# If `pipeline` is already installed (editable, as the uv workspace member `pipeline` is meant to
# be — see orchestrator/pyproject.toml), this is a no-op. If someone instead runs
# `streamlit run orchestrator/ui/app.py` against a bare checkout with nothing pip-installed,
# `sys.path[0]` is `orchestrator/ui/` (the script's own directory, same convention T09's
# `PYTHONPATH` fix documented for container execs) — neither `pipeline` nor `mcp_server` would be
# importable without this.
_ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[1]
if str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))


# --- discovery ----------------------------------------------------------------------------------


def list_presets() -> list[str]:
    from pipeline.api import list_presets as _list_presets

    return _list_presets()


def validate_config(preset: str) -> dict[str, Any]:
    """Fully-resolved config for ``preset``, or raises if it doesn't validate."""
    from pipeline.api import validate_config as _validate_config

    return _validate_config(preset)


# --- preset editing (config, not pipeline logic — see module docstring) -------------------------


def save_preset_variant(
    name: str,
    *,
    extends: str,
    amp_method: Optional[str] = None,
    amp_channels: Optional[dict[str, dict[str, float]]] = None,
    overwrite: bool = False,
) -> Path:
    """Write a new preset YAML (``extends: <extends>`` plus an ``amp:`` override block).

    Mirrors ``pipeline/config/presets/pump01_segB_tuned.yaml``'s own shape (a small experiment
    preset extending a base one). Raises ``FileExistsError`` if ``name`` already exists and
    ``overwrite`` is false. Returns the written path.
    """
    import yaml

    from pipeline.config.resolver import PRESETS_DIR

    path = PRESETS_DIR / f"{name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"preset {name!r} already exists at {path}")

    doc: dict[str, Any] = {"name": name, "extends": extends}
    amp: dict[str, Any] = {}
    if amp_method is not None:
        amp["method"] = amp_method
    if amp_channels:
        amp["channels"] = {ch: dict(vals) for ch, vals in amp_channels.items()}
    if amp:
        doc["amp"] = amp

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path


# --- run lifecycle (async — reuses mcp_server.jobs' background-thread wrapper) ------------------


def start_pipeline_run(
    preset: str,
    *,
    external_artifacts: Optional[dict[str, Any]] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    only: Optional[list[str]] = None,
    force: bool = False,
) -> str:
    """Launch ``preset``'s auto-planned DAG on a background thread; returns ``run_id`` immediately.

    Validates ``preset`` synchronously first (same as the MCP tool) so a bad preset name surfaces
    right in the UI instead of silently inside a background thread.
    """
    from mcp_server.jobs import start_pipeline_run as _start

    return _start(
        preset,
        external_artifacts=external_artifacts,
        from_stage=from_stage,
        to_stage=to_stage,
        only=only,
        force=force,
    )


def start_stage_run(run_id: str, stage: str, *, force: bool = False) -> str:
    from mcp_server.jobs import start_stage_run as _start

    return _start(run_id, stage, force=force)


def get_status(run_id: str) -> dict[str, Any]:
    """Per-stage status, plus ``job_error`` for a background-thread failure that happened before
    any stage record existed — same shape ``get_run_status`` returns over MCP.
    """
    from mcp_server.jobs import job_error
    from pipeline.api import get_status as _get_status

    status = _get_status(run_id)
    status["job_error"] = job_error(run_id)
    return status


def cancel_run(run_id: str) -> dict[str, Any]:
    """Best-effort cancel — honest about ``pipeline.api.cancel`` still being unimplemented
    (T12/T17's scope), same behavior as the MCP ``cancel_run`` tool.
    """
    from pipeline.api import cancel as _cancel
    from pipeline.artifacts import get_manifest

    get_manifest(run_id)  # real 404 if run_id doesn't exist
    try:
        _cancel(run_id)
        return {"cancelled": True}
    except NotImplementedError:
        return {
            "cancelled": False,
            "reason": (
                "cancellation isn't implemented in the pipeline engine yet (T12/T17's scope) — "
                "the run will continue; refresh status for its outcome."
            ),
        }


def list_runs() -> list[dict[str, Any]]:
    from pipeline.api import list_runs as _list_runs

    return _list_runs()


def get_resolved_config(run_id: str) -> dict[str, Any]:
    """The resolved config a run was actually started with (for the Compare-runs diff view).

    Not exposed by ``pipeline.api.get_status``/the MCP ``get_run_status`` tool (deliberately kept
    small) — reads it straight off the manifest, the same record both of those already read.
    """
    from pipeline.artifacts import get_manifest

    return get_manifest(run_id).resolved_config


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    from pipeline.api import list_artifacts as _list_artifacts

    return _list_artifacts(run_id)


def tail_logs(run_id: str, stage: str, max_lines: int = 200) -> dict[str, Any]:
    """Same body as ``mcp_server.server``'s ``tail_logs`` tool — last ``max_lines`` of a stage's
    own log file."""
    from pipeline.artifacts import get_runs_root, stage_log_path

    path = stage_log_path(run_id, stage, runs_root=get_runs_root())
    if not path.is_file():
        return {"path": str(path), "lines": [], "line_count": 0, "truncated": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    all_lines = text.splitlines()
    tail = all_lines[-max_lines:] if max_lines > 0 else all_lines
    return {
        "path": str(path),
        "lines": tail,
        "line_count": len(all_lines),
        "truncated": len(all_lines) > len(tail),
    }


def read_artifact_summary(run_id: str, artifact_name: str) -> dict[str, Any]:
    """Same per-kind summary shape Claude sees over MCP (``mcp_server.artifact_view``)."""
    from mcp_server.artifact_view import read_artifact_summary as _summary
    from pipeline.artifacts import get_artifact as _get_artifact

    return _summary(_get_artifact(run_id, artifact_name))


def artifact_preview_info(run_id: str, artifact_name: str) -> dict[str, Any]:
    """``{"kind": "image"|"video", "path": ...}`` for a previewable artifact, ``{"kind": None}``
    for anything else (browse it via :func:`read_artifact_summary` instead).

    Unlike ``get_preview``'s MCP form, this UI runs on the same machine as the files themselves, so
    it just returns the local path — ``st.image``/``st.video`` read it directly, no need to
    base64/pointer-ize anything the way an MCP tool result must.
    """
    from mcp_server.artifact_view import ArtifactNotPreviewableError, preview_kind
    from pipeline.artifacts import get_artifact as _get_artifact

    artifact = _get_artifact(run_id, artifact_name)
    try:
        kind = preview_kind(artifact)
    except ArtifactNotPreviewableError:
        return {"kind": None, "path": artifact.path}
    return {"kind": kind, "path": artifact.path}


# --- machine / containers -------------------------------------------------------------------


def gpu_status() -> dict[str, Any]:
    from pipeline.api import gpu_status as _gpu_status

    return _gpu_status()


def list_containers() -> list[dict[str, Any]]:
    from pipeline.api import list_containers as _list_containers

    return _list_containers()


def start_container(env: str) -> str:
    from pipeline.api import start_container as _start_container

    return _start_container(env)


def stop_container(container_id: str) -> None:
    from pipeline.api import stop_container as _stop_container

    _stop_container(container_id)
