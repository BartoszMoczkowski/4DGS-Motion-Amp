"""The scheduler: topo-order + cache-skip + execution controls + resume, over the graph module.

This is the T05 deliverable's core — ``run_dag`` is what ``pipeline.api``'s ``run_pipeline``/
``run_stage`` (T05's wiring) ultimately call. It ties together the previously-independent leaf
modules: the stage registry (T04, via ``pipeline.dag.graph``), the run manifest (T03, via
``pipeline.artifacts``), and this package's own cache index (``pipeline.dag.cache``).

Design choices (see ``planning/tasks/T05-dag-scheduler-and-cache.md``):

- **Serial execution.** One stage at a time, in topological order — correct for a single-GPU host
  where the two GPU images never run concurrently (``planning/ARCHITECTURE.md``). The per-stage
  loop body below is the one hook T12's resource gating slots into later (check headroom right
  before ``stage_cls().run(ctx)``).
- **Caching is cross-run.** A stage is "fresh" if its cache key matches either this run's own
  manifest record (cheap, same-run resume) or a *different* run's success recorded in
  ``pipeline.dag.cache``'s index (cache reuse across runs of the same/similar preset). Either way
  it's recorded as ``status="skipped"`` in *this* run's manifest, referencing the same artifacts
  (never copied).
- **Resume is just caching.** There's no separate "resume" code path: calling ``run_dag`` again
  for the same ``run_id`` re-checks every selected stage's freshness. A stage that previously
  failed, or was left ``running`` by a crash, is never "fresh" (no matching ``success``/``skipped``
  record), so it naturally reruns — "restart at the first stale stage" falls out of the cache
  check rather than needing its own logic.
- **A failed stage stops the run.** Downstream stages stay ``pending`` in the manifest (visible,
  not silently skipped) rather than the scheduler guessing whether it's safe to continue.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from .. import containers as _containers
from .. import paths as _paths
from ..artifacts import (
    FAST_ALGO,
    Artifact,
    RunManifest,
    StageRecord,
    create_run,
    get_git_sha,
    hash_path,
    load_manifest,
    record_stage_result,
    record_stage_start,
    run_dir,
    stage_log_path,
    update_manifest,
)
from ..stages import StageContext
from .cache import compute_cache_key, get_cached, put_cached
from .graph import DAGNode, MissingDependencyError, external_inputs, resolve_nodes, topo_sort


def _select(
    order: list[str],
    stage_names: set[str],
    *,
    from_stage: Optional[str],
    to_stage: Optional[str],
    only: Optional[list[str]],
) -> list[str]:
    """Apply ``only``/``from_stage``/``to_stage`` to the full topo ``order``, preserving order.

    ``only`` and the ``from_stage``/``to_stage`` window compose (AND, not OR) if both are given —
    each just narrows the set further.
    """

    selected = list(order)

    if only is not None:
        unknown = set(only) - stage_names
        if unknown:
            raise ValueError(f"`only` names not in this DAG's stage_names: {sorted(unknown)}")
        only_set = set(only)
        selected = [n for n in selected if n in only_set]

    if from_stage is not None or to_stage is not None:
        if from_stage is not None and from_stage not in stage_names:
            raise ValueError(f"from_stage {from_stage!r} not in this DAG's stage_names")
        if to_stage is not None and to_stage not in stage_names:
            raise ValueError(f"to_stage {to_stage!r} not in this DAG's stage_names")
        start = order.index(from_stage) if from_stage is not None else 0
        end = order.index(to_stage) if to_stage is not None else len(order) - 1
        if start > end:
            raise ValueError(
                f"from_stage {from_stage!r} occurs after to_stage {to_stage!r} in topo order "
                f"{order}"
            )
        window = set(order[start : end + 1])
        selected = [n for n in selected if n in window]

    return selected


def _stage_logger(run_id: str, name: str, *, runs_root: Optional[Path]) -> logging.Logger:
    """A stdlib logger writing to this stage's own log file (``StageContext.logger``'s contract)."""

    log_path = stage_log_path(run_id, name, runs_root=runs_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pipeline.run.{run_id}.{name}")
    logger.setLevel(logging.INFO)
    # Avoid piling up duplicate handlers if a stage is re-run (retry/resume) within one process.
    logger.handlers = [h for h in logger.handlers if getattr(h, "_pipeline_log_path", None) != log_path]
    handler = logging.FileHandler(log_path)
    handler._pipeline_log_path = log_path  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _already_recorded(name: str, cache_key: str, manifest: RunManifest) -> bool:
    """This run's own manifest already has a terminal, matching-cache-key record for ``name``.

    Checked *before* the cross-run cache index so re-running the same ``run_id`` twice is a true
    no-op: a stage that already succeeded here keeps its honest ``"success"`` status rather than
    being overwritten with ``"skipped"`` just because it also happens to satisfy the freshness
    check.
    """

    rec = manifest.stages.get(name)
    return rec is not None and rec.status in ("success", "skipped") and rec.cache_key == cache_key


def run_dag(
    run_id: str,
    stage_names: Sequence[str],
    resolved_config: dict[str, Any],
    *,
    preset: str = "adhoc",
    stage_configs: Optional[dict[str, dict[str, Any]]] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    only: Optional[list[str]] = None,
    force: bool = False,
    runs_root: Optional[Path] = None,
) -> RunManifest:
    """Run (or resume) ``run_id`` over ``stage_names``, writing status/timing into its manifest.

    ``stage_names`` is the full DAG for this call — every name must already be registered (T04).
    ``stage_configs`` optionally overrides a stage's ``StageContext.config`` (defaults to the
    whole ``resolved_config`` dict, letting a stage pick out whatever section it needs; a caller
    that already knows the per-stage slice, e.g. a toy graph in tests, can pass it directly).

    Raises ``pipeline.stages.StageNotFoundError`` for an unregistered name,
    :class:`pipeline.dag.graph.CycleError` if ``stage_names`` has no valid order, and
    :class:`pipeline.dag.graph.MissingDependencyError` if some stage's declared input is neither
    produced by another stage in ``stage_names`` nor already present in an existing (resumed)
    run's artifacts. All three are raised *before* touching the manifest, so a bad call never
    leaves a partially-created run behind.
    """

    stage_names = list(stage_names)
    names_set = set(stage_names)
    nodes = resolve_nodes(stage_names)
    order = topo_sort(nodes)  # CycleError propagates

    try:
        existing: Optional[RunManifest] = load_manifest(run_id, runs_root=runs_root)
    except FileNotFoundError:
        existing = None

    known_artifacts = set(existing.artifacts) if existing is not None else set()
    ext = external_inputs(nodes)
    truly_missing = {inp: sorted(names) for inp, names in ext.items() if inp not in known_artifacts}
    if truly_missing:
        raise MissingDependencyError(
            f"run {run_id!r}: input(s) {truly_missing} required but not produced by any stage in "
            f"{sorted(stage_names)} and not already present in the run's existing artifacts"
        )

    selected = _select(order, names_set, from_stage=from_stage, to_stage=to_stage, only=only)

    if existing is None:
        manifest = create_run(run_id, preset, resolved_config, stage_names=stage_names, runs_root=runs_root)
    else:
        manifest = existing
        missing_slots = [n for n in stage_names if n not in manifest.stages]
        if missing_slots:
            def _add_slots(m: RunManifest) -> None:
                for n in missing_slots:
                    m.stages.setdefault(n, StageRecord())

            manifest = update_manifest(run_id, _add_slots, runs_root=runs_root)

    git_sha = manifest.git_sha or get_git_sha()

    for name in selected:
        node: DAGNode = nodes[name]
        stage_cfg = (stage_configs or {}).get(name, resolved_config)

        input_artifacts = {inp: manifest.artifacts[inp] for inp in node.inputs if inp in manifest.artifacts}
        missing_now = [inp for inp in node.inputs if inp not in input_artifacts]
        if missing_now:
            raise MissingDependencyError(
                f"stage {name!r} requires {missing_now} but no upstream stage has produced "
                f"them yet in run {run_id!r} — include their producing stage(s) in this call "
                f"(e.g. via `only`/`from_stage`) or run them first"
            )
        input_hashes = {inp: (art.content_hash or "") for inp, art in input_artifacts.items()}
        cache_key = compute_cache_key(node.stage_cls, stage_cfg, input_hashes, git_sha)

        if not force and _already_recorded(name, cache_key, manifest):
            continue  # this run's own record is already correct; nothing to write

        cached = None if force else get_cached(cache_key, runs_root=runs_root)
        if cached is not None:
            manifest = record_stage_result(
                run_id,
                name,
                status="skipped",
                artifacts=list(cached.values()),
                cache_key=cache_key,
                runs_root=runs_root,
            )
            continue

        manifest = record_stage_start(run_id, name, runs_root=runs_root)
        logger = _stage_logger(run_id, name, runs_root=runs_root)
        ctx = StageContext(
            run_id=run_id,
            stage_name=name,
            config=stage_cfg,
            run_dir=run_dir(run_id, runs_root=runs_root),
            logger=logger,
            inputs=input_artifacts,
            # T09: `cuda`/`isaac` stages (train/render/seg_extract/amp, ...) need real path
            # translation and container exec, not just the "reserved slot" T04/T08 left on
            # `StageContext` — pass the whole `pipeline.paths`/`pipeline.containers` modules
            # (mirrors T07's fix for `ctx.inputs`, which T05 also left unwired). Cheap and safe to
            # set unconditionally: neither import touches `torch`/`docker` at module scope (see
            # `pipeline.containers`'s own package docstring), and a `host`-environment stage simply
            # never reads either attribute.
            paths=_paths,
            containers=_containers,
        )
        try:
            result = node.stage_cls().run(ctx)
        except Exception as exc:  # noqa: BLE001 - a failing stage must not crash the scheduler
            manifest = record_stage_result(
                run_id,
                name,
                status="failed",
                error=str(exc),
                log_path=str(stage_log_path(run_id, name, runs_root=runs_root)),
                runs_root=runs_root,
            )
            return manifest  # stop scheduling; remaining selected stages stay "pending"

        for art in result.values():
            if art.content_hash is None:
                p = Path(art.path)
                if p.is_file():
                    art.content_hash = hash_path(p)
                    art.hash_algo = FAST_ALGO

        manifest = record_stage_result(
            run_id,
            name,
            status="success",
            artifacts=list(result.values()),
            cache_key=cache_key,
            log_path=str(stage_log_path(run_id, name, runs_root=runs_root)),
            runs_root=runs_root,
        )
        put_cached(cache_key, run_id, name, result, runs_root=runs_root)

    if not manifest.stages:
        # An empty DAG (e.g. no real stages registered for any role yet) has nothing to roll its
        # status up from `record_stage_result` — treat "nothing to do" as trivially done.
        def _mark_trivially_done(m: RunManifest) -> None:
            if m.status == "pending":
                m.status = "success"

        manifest = update_manifest(run_id, _mark_trivially_done, runs_root=runs_root)

    return manifest
