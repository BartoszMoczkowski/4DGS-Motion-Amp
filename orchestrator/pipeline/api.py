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
(T08). ``gpu_status`` now delegates to ``pipeline.resources`` (T12 — VRAM/RAM query, serial
gating, adaptive knobs, and OOM-retry are all wired into ``pipeline.dag.scheduler`` directly,
rather than through this module). ``cancel`` remains a stub — cancelling an in-flight run is
explicitly out of scope for T12 (see ``planning/tasks/T12-resource-manager.md``'s scope), not yet
Top-level imports follow PEP 8: standard library and internal pipeline submodules are imported
at module scope. None of the internal submodules import heavy or platform-restricted libraries
(torch, CUDA, docker, pynvml, psutil) at module scope, so this module imports cleanly on a
CPU-only host with no GPU and no Docker daemon running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import uuid

from . import (
    artifacts,
    config,
    containers,
    dag,
    explore as explore_module,
    paths,
    resources,
    stages,
    tracking,
)
from .config import models as config_models
from .config import resolver as config_resolver
from .dag import graph as dag_graph
from .experiments import runner as experiments_runner



def _auto_stage_plan(resolved_config: dict[str, Any]) -> list[str]:
    """Every currently-registered "real" stage (role != ``"test"``), resolved to one impl each.

    A role with exactly one registered impl uses it unconditionally. A role with more than one
    (only ``segment``, today — ``SegmentConfig.impl``) is disambiguated by looking for a
    same-named top-level key in ``resolved_config`` that itself has an ``"impl"`` field, mirroring
    how ``pipeline.config.models.SegmentConfig`` resolves to ``segment.rigid``/``segment.mbs`` via
    the registry (T04's stated design). Raises ``ValueError`` if a multi-impl role has no such
    selector — a config gap, not something to guess at.

    A selector impl of ``"none"`` (``RoiConfig``'s default — "the DAG contains no ``roi`` stage",
    ``pipeline/config/models.py``) means the role is *disabled*, not a stage to run: no stage is
    emitted for that role at all (``"none"`` is never a registered impl, so emitting
    ``f"{role}.none"`` would fail stage resolution for every preset that just left the default).
    """
    names: list[str] = []
    for role, impls in sorted(stages.list_roles().items()):
        if role == "test":
            continue
        if len(impls) == 1:
            names.append(f"{role}.{impls[0]}")
            continue
        selector = resolved_config.get(role)
        impl = selector.get("impl") if isinstance(selector, dict) else None
        if impl == "none":
            continue  # role explicitly disabled — no stage emitted (see docstring)
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


class StageFailedError(RuntimeError):
    """A stage executed by :func:`run_pipeline`/:func:`run_stage` failed.

    ``pipeline.dag.scheduler.run_dag`` deliberately records a stage failure in the run manifest
    (``status="failed"``) and returns instead of raising — the manifest is the durable record, and
    a background scheduler must not crash. But the *synchronous* API callers (the `amp` CLI) used
    to get a normal return either way, so a failed stage printed "finished" and exited 0 (found
    live in the 2026-10-09 T26 follow-up: an `amp.default` RAM-preflight rejection reported as
    success). This exception restores failure propagation at the API boundary while leaving the
    manifest record untouched. Layers 2/3 need no special-casing: the MCP server already captures
    background-job exceptions into ``job_error``, and the UI displays caught exceptions.
    """


def _raise_for_failed_stages(run_id: str, manifest: Any, executed: list[str]) -> None:
    """Raise :class:`StageFailedError` if any of ``executed`` (this call's selected stages) has a
    ``failed`` record in ``manifest``. Stages never selected by this call (``from_stage``/
    ``to_stage``/``only`` windows) have no fresh record and are ignored; a failure stops the DAG,
    so later selected stages simply stay pending."""
    failed = [
        (name, manifest.stages[name])
        for name in executed
        if name in manifest.stages and manifest.stages[name].status == "failed"
    ]
    if not failed:
        return
    parts = []
    for name, rec in failed:
        part = f"{name}: {rec.error or 'unknown error'}"
        if rec.log_path:
            part += f" (log: {rec.log_path})"
        parts.append(part)
    raise StageFailedError(
        f"run {run_id!r}: stage(s) failed: {'; '.join(parts)} — full details in the run manifest"
    )


def _apply_overrides(
    resolved: dict[str, Any], overrides: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto a resolved config dict, then re-validate (T26).

    Re-validation through :class:`pipeline.config.models.PipelineConfig` keeps
    ``extra="forbid"`` enforcement: an unknown/misspelled override key fails here with a
    pydantic ``ValidationError`` naming the bad key instead of being silently ignored, and
    value types are coerced/validated exactly as if the override had been written into the
    preset YAML. Returns ``resolved`` unchanged when ``overrides`` is ``None``/empty.
    """
    if not overrides:
        return resolved
    return config_models.PipelineConfig(
        **config_resolver._deep_merge(resolved, overrides)
    ).model_dump()


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

def new_run_id(preset: str) -> str:
    """Generate a fresh, human-legible run id for ``preset`` (``"<preset>-<8 hex chars>"``).

    Factored out of :func:`run_pipeline` (T14) so a caller that needs the id *before* a run
    actually starts executing — e.g. ``mcp_server.jobs``, whose whole job is handing a ``run_id``
    back to Claude immediately while the real (possibly hours-long) execution continues on a
    background thread — can generate one with the exact same scheme rather than reimplementing it
    and risking the two drifting apart.
    """
    return f"{preset}-{uuid.uuid4().hex[:8]}"


def run_pipeline(
    preset: str,
    *,
    run_id: Optional[str] = None,
    external_artifacts: Optional[dict[str, Any]] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    only: Optional[list[str]] = None,
    force: bool = False,
    overrides: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Run (or resume) the full DAG for ``preset``. Returns a ``run_id``.

    ``run_id`` (T14): normally left ``None`` and generated here via :func:`new_run_id`. A caller
    that needs to know the id *before* this (synchronous, blocking) call returns — concretely,
    ``mcp_server.jobs``'s async-job wrapper, which must hand a ``run_id`` back over MCP right away
    and run this on a background thread — can pass one in explicitly instead.

    Every call starts a *new* run (a fresh ``run_id``): cross-run caching (``pipeline.dag.cache``)
    means an unchanged stage still gets skipped even though it's a different run, so re-running
    for run history/comparison (T15) doesn't cost a real re-execution. Resuming a specific crashed
    run by its own ``run_id`` isn't exposed here yet — use ``run_stage`` to retry one stage of an
    existing run, or call ``pipeline.dag.run_dag`` directly with that ``run_id``.

    ``external_artifacts`` (T11): a real gap found while wiring ``prep_split.default`` in — this function
    previously had no way to satisfy a DAG's *external* inputs (declared by some stage, produced
    by none in this run — see ``pipeline.dag.graph.external_inputs``) before T05's own
    ``MissingDependencyError`` check runs, so every caller that needed one (every test so far) had
    to bypass this function and call ``pipeline.dag.run_dag`` directly against a hand-seeded
    manifest (see ``tests/test_stages_cpu.py``'s ``_seed_run`` helper). That was fine while
    ``convert``/``seg_eval`` were the DAG's only entry points and only tests ever ran them. Now
    that ``prep_split.default`` (T11) needs an external ``raw_mesh`` (a real asset with no in-repo
    producer, e.g. an Omniverse CAD export) for a preset's *auto-planned* full run to work at all,
    this is a real caller-facing gap, not just a test-fixture inconvenience — so
    :func:`run_pipeline` itself now accepts pre-resolved :class:`~pipeline.artifacts.Artifact`
    values for any of a run's external inputs (``{"raw_mesh": Artifact(...), "gt_segmentation":
    Artifact(...)}``, keyed by artifact name) and seeds them into the fresh run's manifest before
    ``run_dag`` executes — the exact same seed-then-run_dag sequence ``_seed_run`` already proved
    out, just promoted from a test helper into the real API.

    External artifact paths are validated at seeding time
    (:func:`pipeline.artifacts.validate_external_artifact_path`): each must be absolute, must
    exist (a directory for ``dataset``/``model`` kinds, a file otherwise), and must resolve under
    one of the pipeline's known roots (repo/assets/runs — see
    ``pipeline.artifacts.paths.allowed_artifact_roots``). An unconstrained path would otherwise
    let an MCP client turn this parameter into an arbitrary-host-file read via the artifact
    resources.

    ``run_id`` collision: passing a ``run_id`` that already exists on disk is an error
    (``FileExistsError``), not a silent reset — ``create_run`` would otherwise overwrite the old
    run's ``manifest.json``/``config_snapshot.json`` with an all-pending manifest, destroying its
    recorded history before ``run_dag``'s resume logic ever sees it. To deliberately re-run, mint
    a fresh id (or delete the old run directory first).

    ``overrides`` (T26): an optional nested config dict (``PipelineConfig``-shaped, e.g.
    ``{"amp": {"channels": {"pos3d": {"factor": 2.0}}}}``) deep-merged onto the resolved preset
    *before* the stage plan is computed and before ``PipelineConfig`` re-validation — so
    ``extra="forbid"`` still catches misspelled override keys (pydantic ``ValidationError``), and
    an override that flips e.g. ``segment.impl`` changes the auto-planned stage list too. This is
    the mechanism CLI flags (``amp train --scene``, ``amp render --amp-factors``) use instead of
    writing throwaway YAML presets.

    ``metadata`` (T27): an optional free-form dict recorded verbatim on the run manifest
    (``RunManifest.metadata``) — experiment/cell/variant identity, capture provenance, ... — so
    runs are queryable by *what they are* (e.g. the experiment runner's find-or-create by
    ``{experiment, cell, variant}``) rather than only by run_id. Not part of the resolved config
    and never affects cache keys.

    Raises :class:`StageFailedError` if any stage this call executed ends up ``failed`` in the
    manifest (run_dag itself records-and-returns; the failure is re-raised here so synchronous
    callers exit non-zero).
    """
    cfg = config.validate_config(preset)
    resolved = _apply_overrides(cfg.model_dump(), overrides)
    stage_names = _auto_stage_plan(resolved)
    stage_configs = {name: _stage_config_for(name, resolved) for name in stage_names}
    run_id = run_id or new_run_id(preset)

    artifacts.validate_run_id(run_id)  # fail fast with a clear error before any directory is created
    if artifacts.manifest_path(run_id).exists():
        raise FileExistsError(
            f"run_id {run_id!r} already exists ({artifacts.manifest_path(run_id)}); run_pipeline always "
            "starts a NEW run — pass a fresh run_id (or none) rather than clobbering an existing "
            "run's manifest"
        )
    if external_artifacts:
        normalized: dict[str, Any] = {}
        for name, art in external_artifacts.items():
            art = art if isinstance(art, artifacts.Artifact) else artifacts.Artifact.model_validate(art)
            artifacts.validate_external_artifact_path(art.path, art.kind)
            normalized[name] = art
        external_artifacts = normalized

    artifacts.create_run(run_id, preset, resolved, stage_names=stage_names, metadata=metadata)
    if external_artifacts:
        def _seed(m: Any) -> None:
            m.artifacts.update(external_artifacts)

        artifacts.update_manifest(run_id, _seed)

    # T28: local MLflow tracking. Failure-isolated by design (tracking.py swallows all its own
    # errors — a down server warns, never fails the run). The finally block records the failure
    # to MLflow *before* StageFailedError propagates out of _raise_for_failed_stages.
    mlflow_run_id = tracking.start_run(run_id, preset, resolved, metadata)
    tracking.record_backlink(run_id, mlflow_run_id)
    manifest = None
    try:
        manifest = dag.run_dag(
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
        _raise_for_failed_stages(run_id, manifest, stage_names)
        return manifest.run_id
    finally:
        tracking.track_run_completion(run_id, manifest, stage_order=stage_names)


def run_stage(
    run_id: str,
    stage: str,
    *,
    force: bool = False,
    overrides: Optional[dict[str, Any]] = None,
) -> str:
    """Run a single stage within an existing run. Returns a ``run_id``.

    ``stage``'s declared inputs must already be present among ``run_id``'s recorded artifacts
    (from earlier stages of that same run) — this doesn't (re)run anything upstream. Raises
    ``FileNotFoundError`` if ``run_id`` doesn't exist yet (call ``run_pipeline`` first to create
    it) and ``pipeline.dag.MissingDependencyError`` if an input isn't available.

    ``overrides`` (T26): same deep-merge + re-validation semantics as
    :func:`run_pipeline`'s, applied on top of the run's *recorded* resolved config for this
    invocation only — the manifest's ``resolved_config`` is deliberately NOT rewritten, so a
    later plain ``run_stage`` call still sees the original config (and the changed stage config
    naturally produces a different cache key, so the stage re-runs rather than cache-hitting on
    stale parameters).

    Raises :class:`StageFailedError` if the stage's record ends up ``failed`` (see
    :func:`run_pipeline`).
    """
    manifest = artifacts.get_manifest(run_id)
    resolved = _apply_overrides(manifest.resolved_config, overrides)
    stage_cfg = _stage_config_for(stage, resolved)

    # T28: start-or-reuse the MLflow run (same join key as run_pipeline), then incremental
    # stage-metric/artifact logging around the single-stage execution — needed for T27's
    # per-impl segmentation matrix, which re-enters existing runs via run_stage.
    mlflow_run_id = tracking.start_run(run_id, manifest.preset, resolved, manifest.metadata)
    tracking.record_backlink(run_id, mlflow_run_id)
    result_manifest = None
    try:
        result_manifest = dag.run_dag(
            run_id,
            [stage],
            resolved,
            preset=manifest.preset,
            stage_configs={stage: stage_cfg},
            force=force,
        )
        _raise_for_failed_stages(run_id, result_manifest, [stage])
        return run_id
    finally:
        tracking.track_run_completion(run_id, result_manifest, stage_order=[stage])


def cancel(run_id: str) -> dict[str, Any]:
    """Cancel an in-flight run (T17/T30).

    Stops managed containers if running (the locked cancellation approach), updates the run
    manifest status to ``"cancelled"``, and records terminal status in MLflow tracking.
    """
    # Raises FileNotFoundError synchronously if run_id does not exist
    artifacts.get_manifest(run_id)

    stopped_containers: list[str] = []
    try:
        container_list = containers.list_containers()
        for c in container_list:
            if c.get("status") == "running":
                cid = c.get("id") or c.get("name")
                if cid:
                    try:
                        containers.stop_container(cid)
                        stopped_containers.append(cid)
                    except Exception:
                        pass
    except Exception:
        pass

    def _mutate(m: Any) -> None:
        m.status = "cancelled"
        for name, rec in m.stages.items():
            if rec.status == "running":
                rec.status = "failed"
                rec.error = "cancelled by user"

    artifacts.update_manifest(run_id, _mutate)

    if tracking.is_enabled():
        try:
            m_updated = artifacts.get_manifest(run_id)
            tracking.finish_run(run_id, m_updated, status="cancelled")
        except Exception:
            pass

    return {
        "run_id": run_id,
        "cancelled": True,
        "stopped_containers": stopped_containers,
    }


def get_status(run_id: str) -> dict[str, Any]:
    """Per-stage status/timing/logs/outputs/peak-mem for a run.

    Reads straight from the manifest (T03) — no scheduling knowledge needed, so this works for
    any run a hand-written or T03-level caller created, even before T05's scheduler exists.
    """
    manifest = artifacts.get_manifest(run_id)
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
    return artifacts.list_runs()


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    """List artifacts produced by a run."""
    return [a.model_dump() for a in artifacts.list_artifacts(run_id)]


def get_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    """Fetch a single artifact record (path, type, hash, producing stage)."""
    return artifacts.get_artifact(run_id, artifact_id).model_dump()


def list_presets() -> list[str]:
    """List available config presets (base <- scene <- experiment)."""
    return config.list_presets()


def validate_config(preset: str) -> dict[str, Any]:
    """Resolve + validate a preset without running anything."""
    return config.validate_config(preset).model_dump()


def save_preset_variant(
    name: str,
    base_preset: str,
    overrides: Optional[dict[str, Any]] = None,
    *,
    overwrite: bool = False,
) -> str:
    """Save a new preset YAML extending ``base_preset`` with ``overrides``."""
    path = config.save_preset_variant(name, base_preset, overrides=overrides, overwrite=overwrite)
    return str(path)


# --- resources / containers ---------------------------------------------------

def gpu_status() -> dict[str, Any]:
    """Current VRAM/RAM usage and free headroom (T12).

    Delegates to :func:`pipeline.resources.gpu_status` — ``None`` sub-dicts mean that dimension
    couldn't be measured on this machine (no GPU / no working ``pynvml``+``nvidia-smi``/
    ``psutil``), not an error; see that module's docstring.
    """
    return resources.gpu_status()


def list_containers() -> list[dict[str, Any]]:
    """List managed containers (image, state, mounts)."""
    return containers.list_containers()


def start_container(env: str) -> str:
    """Start (or reuse a warm) container for environment ``env`` (cuda|isaac)."""
    return containers.start_container(env)


def stop_container(container_id: str) -> None:
    """Stop a managed container."""
    containers.stop_container(container_id)


# --- experiment & exploration (T30) -------------------------------------------


class MissingExternalArtifactError(RuntimeError):
    """Raised when an external artifact required by a pipeline stage cannot be located."""

    def __init__(
        self,
        target: str,
        stage: str,
        artifact_name: str,
        artifact_kind: str,
        checked_paths: list[str],
    ) -> None:
        msg = (
            f"Cannot run '{target}': stage '{stage}' requires external artifact "
            f"'{artifact_name}' (kind={artifact_kind!r}) which was not found.\n"
            f"Checked locations:\n  - " + "\n  - ".join(checked_paths) + "\n"
            f"Action needed: Provide '{artifact_name}' via external_artifacts or ensure the asset exists."
        )
        super().__init__(msg)
        self.target = target
        self.stage = stage
        self.artifact_name = artifact_name
        self.artifact_kind = artifact_kind
        self.checked_paths = checked_paths


def resolve_external_artifacts_for_preset(
    preset: str,
    stages: Optional[list[str]] = None,
    provided: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Auto-discover and seed external artifacts required by ``preset``'s stage plan."""
    provided_artifacts = dict(provided or {})

    resolved_cfg = config.validate_config(preset).model_dump()
    stage_names = stages or _auto_stage_plan(resolved_cfg)

    dag_nodes = dag_graph.resolve_nodes(stage_names)
    ext_inputs = dag_graph.external_inputs(dag_nodes)

    external_needed: list[tuple[str, str]] = []
    for inp, needing_stages in ext_inputs.items():
        if inp not in provided_artifacts:
            stage_name = sorted(needing_stages)[0] if needing_stages else "unknown"
            external_needed.append((stage_name, inp))

    if not external_needed:
        return provided_artifacts

    roots = paths.get_roots()
    repo_root = roots.repo_root_host
    assets_root = roots.assets_root_host

    artifact_kind_map: dict[str, str] = {
        "raw_mesh": "usd",
        "capture": "dataset",
        "scene": "dataset",
        "gt_segmentation": "npz",
    }

    result = dict(provided_artifacts)
    for sname, inp in external_needed:
        kind = artifact_kind_map.get(inp, "dataset")
        candidates: list[Path] = []
        if inp == "scene":
            candidates = [
                repo_root / "data" / "multipleview" / preset,
                repo_root / "data" / preset,
                repo_root / "output" / preset,
            ]
        elif inp == "capture":
            candidates = [
                repo_root / "omniverse-pipeline" / "data" / "scenes" / "grid" / preset,
                repo_root / "data" / "multipleview" / preset,
                repo_root / "data" / "captures" / preset,
                repo_root / "data" / preset,
                assets_root / "renders" / f"capture_{preset}",
                assets_root / "renders" / preset,
            ]
        elif inp == "raw_mesh":
            candidates = [
                assets_root / f"{preset}.usd",
                repo_root / "omniverse-pipeline" / "data" / "assets" / f"{preset}.usd",
                repo_root / "data" / f"{preset}.usd",
            ]
        elif inp == "gt_segmentation":
            candidates = [
                repo_root / "data" / "multipleview" / preset / "gt_segmentation.npz",
                repo_root / "data" / preset / "gt_segmentation.npz",
            ]

        found_path: Optional[Path] = None
        for cand in candidates:
            if kind in ("dataset", "model") and cand.is_dir():
                found_path = cand
                break
            elif kind not in ("dataset", "model") and cand.is_file():
                found_path = cand
                break

        if found_path is not None:
            result[inp] = artifacts.Artifact(
                name=inp,
                kind=kind,
                path=str(found_path.resolve()),
                producing_stage="external",
            )
        else:
            raise MissingExternalArtifactError(
                target=preset,
                stage=sname,
                artifact_name=inp,
                artifact_kind=kind,
                checked_paths=[str(c) for c in candidates],
            )

    return result


def run_experiment(
    target: str,
    *,
    stages: Optional[list[str]] = None,
    overrides: Optional[dict[str, Any]] = None,
    external_artifacts: Optional[dict[str, Any]] = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run an experiment spec (e.g. ``pump_grid_v2``) or preset with auto-seeded external artifacts."""
    repo_root = paths.get_roots().repo_root_host
    exp_file = repo_root / "experiments" / f"{target}.yaml"
    if not exp_file.is_file() and target.endswith(".yaml"):
        exp_file = Path(target)

    if exp_file.is_file():
        return experiments_runner.run_experiment(str(exp_file), dry_run=dry_run)

    # Target is a preset
    seeded = resolve_external_artifacts_for_preset(
        target, stages=stages, provided=external_artifacts
    )
    if dry_run:
        resolved_cfg = config.validate_config(target).model_dump()
        stage_names = stages or _auto_stage_plan(resolved_cfg)
        return {
            "preset": target,
            "stages": stage_names,
            "external_artifacts": {k: getattr(a, "path", str(a)) for k, a in seeded.items()},
            "dry_run": True,
        }

    run_id = run_pipeline(
        target,
        only=stages,
        overrides=overrides,
        external_artifacts=seeded,
        force=force,
    )
    return {
        "run_id": run_id,
        "preset": target,
        "status": "success",
    }


def explore(
    target: Optional[str] = None,
    view: str = "default",
    include_unregistered: bool = False,
) -> dict[str, Any]:
    """Unified inspection navigator (T30)."""
    return explore_module.explore(target=target, view=view, include_unregistered=include_unregistered)
