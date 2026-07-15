"""Public API surface for Layer 1 (the pipeline execution engine).

This is the *only* module Layer 2 (MCP server) and Layer 3 (UI) are meant to
call into. Every function here is a typed stub for now — signatures follow
``planning/ARCHITECTURE.md`` ("Public API" under Layer 1) so later tasks slot
real implementations in without changing the interface.

``list_presets``/``validate_config`` delegate to ``pipeline.config`` (T02); ``list_runs``/
``get_status``/``list_artifacts``/``get_artifact`` delegate to ``pipeline.artifacts`` (T03);
``run_pipeline``/``run_stage`` delegate to ``pipeline.dag`` (T05) — the DAG scheduler is generic
over the stage registry (T04) and doesn't know ``PipelineConfig``'s shape, so *this* module owns
the one bit of glue that does: turning a resolved preset into the ordered stage-name plan
``run_dag`` executes (see ``_auto_stage_plan`` below), and — since T07 — slicing that resolved
config into the specific section each stage actually wants (see ``_stage_config_for`` below).
``list_containers``/``start_container``/``stop_container`` now delegate to ``pipeline.containers``
(T08). ``cancel``/``gpu_status`` remain stubs (T12 — the resource manager, not this module, owns
gating/cancellation). All delegated calls use a lazy import inside the function so this module's
own top-level imports stay light. Nothing in this module imports torch/CUDA/docker/pynvml at
module scope — it must import cleanly on a CPU-only host with no GPU and no Docker daemon running.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional


def _auto_stage_plan(resolved_config: dict[str, Any]) -> list[str]:
    """Every currently-registered "real" stage (role != ``"test"``), resolved to one impl each.

    A role with exactly one registered impl uses it unconditionally. A role with more than one
    (only ``segment``, today — ``SegmentConfig.impl``) is disambiguated by looking for a
    same-named top-level key in ``resolved_config`` that itself has an ``"impl"`` field, mirroring
    how ``pipeline.config.models.SegmentConfig`` resolves to ``segment.rigid``/``segment.mbs`` via
    the registry (T04's stated design). Raises ``ValueError`` if a multi-impl role has no such
    selector — a config gap, not something to guess at.

    Right now this returns ``[]`` for every preset: no real role besides ``test`` (deliberately
    excluded) is registered until T07 wraps the first real stages. That's fine — an empty DAG is
    a valid, trivially-successful run (see ``pipeline.dag.scheduler.run_dag``); the plan becomes
    meaningful the moment T07/T09/T10/T11 register real stage classes, with no change needed here.
    """
    from .stages import list_roles

    names: list[str] = []
    for role, impls in sorted(list_roles().items()):
        if role == "test":
            continue
        if len(impls) == 1:
            names.append(f"{role}.{impls[0]}")
            continue
        selector = resolved_config.get(role)
        impl = selector.get("impl") if isinstance(selector, dict) else None
        if not impl:
            raise ValueError(
                f"role {role!r} has multiple registered impls {impls} and resolved_config[{role!r}] "
                f"has no 'impl' selector to disambiguate"
            )
        names.append(f"{role}.{impl}")
    return names


#: T09: roles whose stage needs `model`/`pipeline_params`/`hidden`/`optim` to build their
#: `--configs` bridge file (`pipeline.config.bridge`) on top of their own section — see
#: `_stage_config_for`'s "_bridge" handling below.
_CUDA_BRIDGE_ROLES = frozenset({"train", "render", "seg_extract", "amp"})
_BRIDGE_SECTIONS = ("model", "pipeline_params", "hidden", "optim")


def _stage_config_for(name: str, resolved_config: dict[str, Any]) -> dict[str, Any]:
    """This stage's own config section, not the whole ``PipelineConfig`` dict (T07).

    T05 defaulted every stage's ``ctx.config`` to the *whole* resolved config (there was no real
    per-role section to slice yet — ``run_dag``'s own docstring flagged this as "T07+ can start
    passing ``stage_configs`` overrides once real per-stage sections matter"). Now that
    ``convert``/``segment.rigid``/``seg_eval`` are real, wholesale-passing the entire config would
    force every stage to know the top-level schema shape; slice it here instead, mirroring how
    ``SegmentConfig.impl`` already nests an implementation's own section under its role
    (``resolved_config["segment"]["rigid"]`` for ``"segment.rigid"``).

    Falls back to the whole ``resolved_config`` dict when the role has no matching top-level
    section (e.g. ``"test.echo"``) — preserves T05's original default for any stage that doesn't
    have its own section.

    T09 addition: ``train``/``render``/``seg_extract``/``amp`` also need
    ``model``/``pipeline_params``/``hidden``/``optim`` — the 4DGS core param groups
    ``pipeline.config.bridge`` writes into a temp ``arguments/multipleview/<name>.py``-style file
    for these stages' ``--configs`` flag. Merging the *whole* resolved config in (like the
    no-section fallback above) would defeat T05's cache-key scoping — an unrelated section
    changing (``capture``, ``segment``, ...) must not invalidate ``train``'s cache — so these four
    sections are merged in under a reserved ``"_bridge"`` key instead, alongside the stage's own
    section, keeping the cache key exactly as sensitive as it needs to be and no more.
    """
    role, impl = name.split(".", 1)
    section = resolved_config.get(role)
    if not isinstance(section, dict):
        return resolved_config
    if "impl" in section and isinstance(section.get(impl), dict):
        cfg = section[impl]
    else:
        cfg = section
    if role not in _CUDA_BRIDGE_ROLES:
        return cfg
    return {**cfg, "_bridge": {k: resolved_config[k] for k in _BRIDGE_SECTIONS}}


# --- run lifecycle ----------------------------------------------------------

def run_pipeline(
    preset: str,
    *,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    only: Optional[list[str]] = None,
    force: bool = False,
) -> str:
    """Run (or resume) the full DAG for ``preset``. Returns a ``run_id``.

    Every call starts a *new* run (a fresh ``run_id``): cross-run caching (``pipeline.dag.cache``)
    means an unchanged stage still gets skipped even though it's a different run, so re-running
    for run history/comparison (T15) doesn't cost a real re-execution. Resuming a specific crashed
    run by its own ``run_id`` isn't exposed here yet — use ``run_stage`` to retry one stage of an
    existing run, or call ``pipeline.dag.run_dag`` directly with that ``run_id``.
    """
    from .config import validate_config
    from .dag import run_dag

    config = validate_config(preset)
    resolved = config.model_dump()
    stage_names = _auto_stage_plan(resolved)
    stage_configs = {name: _stage_config_for(name, resolved) for name in stage_names}
    run_id = f"{preset}-{uuid.uuid4().hex[:8]}"
    manifest = run_dag(
        run_id,
        stage_names,
        resolved,
        preset=preset,
        stage_configs=stage_configs,
        from_stage=from_stage,
        to_stage=to_stage,
        only=only,
        force=force,
    )
    return manifest.run_id


def run_stage(
    run_id: str,
    stage: str,
    *,
    force: bool = False,
) -> str:
    """Run a single stage within an existing run. Returns a ``run_id``.

    ``stage``'s declared inputs must already be present among ``run_id``'s recorded artifacts
    (from earlier stages of that same run) — this doesn't (re)run anything upstream. Raises
    ``FileNotFoundError`` if ``run_id`` doesn't exist yet (call ``run_pipeline`` first to create
    it) and ``pipeline.dag.MissingDependencyError`` if an input isn't available.
    """
    from .artifacts import get_manifest
    from .dag import run_dag

    manifest = get_manifest(run_id)
    stage_cfg = _stage_config_for(stage, manifest.resolved_config)
    run_dag(
        run_id,
        [stage],
        manifest.resolved_config,
        preset=manifest.preset,
        stage_configs={stage: stage_cfg},
        force=force,
    )
    return run_id


def cancel(run_id: str) -> None:
    """Cancel an in-flight run."""
    raise NotImplementedError


def get_status(run_id: str) -> dict[str, Any]:
    """Per-stage status/timing/logs/outputs/peak-mem for a run.

    Reads straight from the manifest (T03) — no scheduling knowledge needed, so this works for
    any run a hand-written or T03-level caller created, even before T05's scheduler exists.
    """
    from .artifacts import get_manifest

    manifest = get_manifest(run_id)
    return {
        "run_id": manifest.run_id,
        "preset": manifest.preset,
        "status": manifest.status,
        "git_sha": manifest.git_sha,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "stages": {name: rec.model_dump() for name, rec in manifest.stages.items()},
    }


# --- discovery ---------------------------------------------------------------

def list_runs() -> list[dict[str, Any]]:
    """List known runs (most recent first)."""
    from .artifacts import list_runs as _list_runs

    return _list_runs()


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    """List artifacts produced by a run."""
    from .artifacts import list_artifacts as _list_artifacts

    return [a.model_dump() for a in _list_artifacts(run_id)]


def get_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    """Fetch a single artifact record (path, type, hash, producing stage)."""
    from .artifacts import get_artifact as _get_artifact

    return _get_artifact(run_id, artifact_id).model_dump()


def list_presets() -> list[str]:
    """List available config presets (base <- scene <- experiment)."""
    from .config import list_presets as _list_presets

    return _list_presets()


def validate_config(preset: str) -> dict[str, Any]:
    """Resolve + validate a preset without running anything."""
    from .config import validate_config as _validate_config

    return _validate_config(preset).model_dump()


# --- resources / containers ---------------------------------------------------

def gpu_status() -> dict[str, Any]:
    """Current VRAM/RAM usage and free headroom."""
    raise NotImplementedError


def list_containers() -> list[dict[str, Any]]:
    """List managed containers (image, state, mounts)."""
    from .containers import list_containers as _list_containers

    return _list_containers()


def start_container(env: str) -> str:
    """Start (or reuse a warm) container for environment ``env`` (cuda|isaac)."""
    from .containers import start_container as _start_container

    return _start_container(env)


def stop_container(container_id: str) -> None:
    """Stop a managed container."""
    from .containers import stop_container as _stop_container

    _stop_container(container_id)
