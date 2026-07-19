"""Background execution for MCP's async-job tools (T14): ``run_pipeline``/``run_stage``.

Per ``planning/ARCHITECTURE.md``'s Layer 2 "Async jobs" note: a stage can run for hours (training
in particular), so the MCP tool that starts one must return a ``run_id`` immediately rather than
blocking the whole HTTP request for however long the DAG takes — Claude polls
``get_run_status``/``tail_logs`` separately afterward. ``pipeline.api.run_pipeline``/``run_stage``
themselves are synchronous, blocking calls (T05/T09's own scope never needed otherwise); this
module is the one place that turns them into fire-and-forget background work without changing
anything about Layer 1's own contract.

The manifest itself (``pipeline.artifacts``, T03) already updates incrementally per-stage as
``run_dag`` executes, atomically (write-temp-rename) — so ``get_run_status`` polling the manifest
from a different thread than the one running the DAG is already a safe, supported pattern that
predates this module (T13's own connectivity proof relies on nothing state-holding in-process).
The one real gap this module closes: a background call can raise *before* the manifest reflects
anything at all — e.g. ``run_stage``'s own ``MissingDependencyError`` if an input isn't ready yet,
or ``run_pipeline``'s ``MissingDependencyError`` for a DAG whose external inputs were never
supplied, both raised by ``pipeline.dag.scheduler.run_dag`` before any per-stage record is
written. With no in-process caller waiting on the call's return value, that exception would
otherwise vanish silently the moment the background thread dies — a run would look permanently
stuck ``"pending"`` with no explanation. ``job_error`` is the one piece of extra state this module
keeps, purely to surface that one failure mode; every other status question is answered by reading
the manifest, same as always.

Deliberately *not* a solution for concurrent writers: starting two jobs against the same
``run_id`` at once (e.g. ``run_stage`` called twice before the first finishes) is the caller's own
responsibility to avoid — ``pipeline.artifacts.manifest``'s own per-path lock (see that module's
``_lock_for``) keeps any such race from corrupting the manifest file itself, but two stages
stepping on each other's ``StageContext``/container state is not something this module (or
Layer 1) guards against.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Optional

#: One entry per run_id this process has ever started a background job for. Never cleared — one
#: small dataclass per run for the life of the process is negligible, and evicting entries would
#: reopen a window where a still-relevant `job_error` disappears right when something wants to
#: check it. Mirrors `pipeline.artifacts.manifest`'s own `_manifest_locks` registry's reasoning.
@dataclass
class _Job:
    thread: threading.Thread
    kind: str  # "run_pipeline" | "run_stage"
    error: Optional[str] = None


_jobs: dict[str, _Job] = {}
_jobs_guard = threading.Lock()


def job_error(run_id: str) -> Optional[str]:
    """The most recent background-job exception (full traceback text) for ``run_id``, if a call
    started via this module raised one — ``None`` if no job is tracked for this id, or its thread
    hasn't raised. See the module docstring for exactly which failures this catches.
    """
    with _jobs_guard:
        job = _jobs.get(run_id)
        return job.error if job is not None else None


def _run_and_capture(run_id: str, fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 - a background thread has no caller to propagate to
        with _jobs_guard:
            job = _jobs.get(run_id)
            if job is not None:
                job.error = traceback.format_exc()


def _spawn(run_id: str, kind: str, fn) -> None:
    thread = threading.Thread(
        target=_run_and_capture, args=(run_id, fn), name=f"pipeline-{kind}-{run_id}", daemon=True
    )
    with _jobs_guard:
        _jobs[run_id] = _Job(thread=thread, kind=kind)
    thread.start()


def start_pipeline_run(
    preset: str,
    *,
    external_artifacts: Optional[dict[str, Any]] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    only: Optional[list[str]] = None,
    force: bool = False,
) -> str:
    """Validate ``preset`` synchronously — so a bad preset name/config fails the MCP call
    immediately instead of silently dying inside a background thread — then start the real
    (possibly long-running) DAG execution on a background thread and return its ``run_id`` right
    away.

    The id is generated here (:func:`pipeline.api.new_run_id`) rather than read from
    ``pipeline.api.run_pipeline``'s own return value, which is only available once the *entire*
    run finishes — waiting for it would defeat the whole point of returning immediately.
    """
    from pipeline.config import validate_config

    validate_config(preset)  # raises synchronously on an unknown/invalid preset — see docstring

    from pipeline.api import new_run_id, run_pipeline

    run_id = new_run_id(preset)

    def _run() -> None:
        run_pipeline(
            preset,
            run_id=run_id,
            external_artifacts=external_artifacts,
            from_stage=from_stage,
            to_stage=to_stage,
            only=only,
            force=force,
        )

    _spawn(run_id, "run_pipeline", _run)
    return run_id


def start_stage_run(run_id: str, stage: str, *, force: bool = False) -> str:
    """Confirm ``run_id`` actually exists synchronously (same fail-fast reasoning as
    :func:`start_pipeline_run`), then run just ``stage`` on a background thread.
    """
    from pipeline.artifacts import get_manifest

    get_manifest(run_id)  # raises FileNotFoundError/ManifestCorruptError synchronously if bad

    from pipeline.api import run_stage

    def _run() -> None:
        run_stage(run_id, stage, force=force)

    _spawn(run_id, "run_stage", _run)
    return run_id
