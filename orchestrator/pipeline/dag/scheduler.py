"""The scheduler: topo-order + cache-skip + execution controls + resume, over the graph module.

This is the T05 deliverable's core — ``run_dag`` is what ``pipeline.api``'s ``run_pipeline``/
``run_stage`` (T05's wiring) ultimately call. It ties together the previously-independent leaf
modules: the stage registry (T04, via ``pipeline.dag.graph``), the run manifest (T03, via
``pipeline.artifacts``), and this package's own cache index (``pipeline.dag.cache``).

Design choices (see ``planning/tasks/T05-dag-scheduler-and-cache.md``):

- **Serial execution.** One stage at a time, in topological order — correct for a single-GPU host
  where the two GPU images never run concurrently (``planning/ARCHITECTURE.md``). T12 slots into
  the per-stage loop body exactly where this docstring always said it would: a
  ``pipeline.resources.check_headroom`` call right before a real (non-cached) stage runs, a
  ``pipeline.resources.ResourceMonitor`` wrapping the run to fill ``StageRecord.peak_vram_mb``/
  ``peak_ram_mb``, and ``pipeline.resources.run_with_oom_retry`` in place of a bare
  ``stage_cls().run(ctx)`` call so an apparent CUDA OOM gets one reduced-memory retry before the
  stage is recorded as failed.
- **Caching is cross-run.** A stage is "fresh" if its cache key matches either this run's own
  manifest record (cheap, same-run resume) or a *different* run's success recorded in
  ``pipeline.dag.cache``'s index (cache reuse across runs of the same/similar preset). Either way
  it's recorded as ``status="skipped"`` in *this* run's manifest, referencing the same artifacts
  (never copied). Since 2026-10-05 (``reviews/orchestrator-correctness-review.md`` bugs
  1.3/1.4/1.8/1.9) freshness additionally requires that (a) directory artifacts carry a real
  content fingerprint (``pipeline.artifacts.hash_directory``) so forced upstream reruns
  invalidate downstream keys, (b) a hit's recorded artifact paths still exist (file hashes
  re-checked; directories existence-only, since ``render``/``amp`` legitimately write into their
  input model dir in place), (c) a stage's declared outputs exist on disk before a success is
  recorded, and (d) OOM-fallback successes are recorded/cached under their *effective*
  (reduced-memory) config's key, not the original config's.
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
    FULL_ALGO,
    Artifact,
    RunManifest,
    StageRecord,
    create_run,
    get_git_sha,
    hash_directory,
    hash_path,
    load_manifest,
    record_stage_result,
    record_stage_start,
    run_dir,
    stage_log_path,
    update_manifest,
)
from ..resources import InsufficientResourcesError, ResourceMonitor, check_headroom, run_with_oom_retry
from ..stages import StageContext
from .cache import compute_cache_key, get_cached, put_cached
from .graph import DAGNode, MissingDependencyError, external_inputs, resolve_nodes, topo_sort


class StageOutputError(RuntimeError):
    """A stage exited 0 / returned normally but one of its declared outputs is missing.

    Raised by the scheduler's post-success output verification (2026-10-05, review bug 1.8):
    before this, only ``train.default``/``capture.isaac`` checked their own outputs, and a
    script that exited 0 without writing anything got recorded (and cross-run cached!) as a
    ``success`` with dead artifact paths — see ``pipeline.stages.train``'s comment for the
    real-hardware incident that established this pattern.
    """


def _hash_artifact_path(p: Path, *, fast: bool = True) -> Optional[str]:
    """Content hash/fingerprint for ``p``, whether it's a file or a directory; ``None`` if the
    path doesn't exist at all (callers treat that as "unknown", never as a hash)."""

    if p.is_file():
        return hash_path(p, fast=fast)
    if p.is_dir():
        return hash_directory(p)
    return None


def _input_hash(art: Artifact) -> str:
    """The hash ``compute_cache_key`` sees for one input artifact.

    Uses the artifact's recorded ``content_hash`` when present (the common case for
    stage-produced artifacts, hashed right after the producing stage succeeds). Artifacts with
    no recorded hash — externally seeded ones (``capture``/``gt_segmentation``/``raw_mesh``) —
    are hashed on the fly so a re-captured or re-seeded *directory* input (kind
    ``dataset``/``model``, previously always ``""``) now invalidates downstream cache keys
    (review bug 1.3). Missing paths keep the old ``""`` behavior: the missing-dependency check
    upstream is what guards those.
    """

    if art.content_hash:
        return art.content_hash
    return _hash_artifact_path(Path(art.path)) or ""


def _artifacts_intact(artifacts: Sequence[Artifact]) -> bool:
    """``True`` iff every artifact's path still exists and (for files with a recorded hash)
    still matches it — the revalidation a cache hit never used to do (review bug 1.4).

    Directories are checked for *existence only*: ``render.default``/``amp.default`` write
    *into* their input model directory in place (see ``pipeline.stages.render``'s docstring),
    so a recorded directory hash legitimately goes stale while the artifact is still perfectly
    valid — re-hash-comparing directories would force spurious reruns of ``train`` after every
    ``render``. File artifacts (``.npz``/``.ply``/``.json``/...) are never mutated in place by
    downstream stages, so their recorded hash is safe and cheap to recheck (the fast
    fingerprint reads at most 2 MiB).
    """

    for art in artifacts:
        p = Path(art.path)
        if not p.exists():
            return False
        if art.content_hash and p.is_file():
            current = hash_path(p, fast=(art.hash_algo != FULL_ALGO))
            if current != art.content_hash:
                return False
    return True


def _verify_declared_outputs(
    stage_cls: type, stage_name: str, result: dict[str, Artifact]
) -> None:
    """Fail loudly if a nominally successful stage's declared outputs aren't actually on disk.

    Every artifact the stage returned must exist (a non-empty directory for the directory
    kinds, a file otherwise), and every name in the stage's declared ``outputs`` contract must
    be present in ``result``. Extra, undeclared keys are allowed (e.g.
    ``seg_eval.default``'s optional ``recolored_ply``) but must exist too — they're recorded
    and cached like any other artifact, so a dead path would poison the cache identically.
    """

    missing_keys = [o for o in getattr(stage_cls, "outputs", ()) if o not in result]
    if missing_keys:
        raise StageOutputError(
            f"stage {stage_name!r} exited successfully but did not return its declared "
            f"output(s) {missing_keys} (returned: {sorted(result)}) — refusing to record a "
            f"success without the contracted artifacts"
        )
    for art_name, art in result.items():
        p = Path(art.path)
        if not p.exists():
            raise StageOutputError(
                f"stage {stage_name!r} exited successfully but its output {art_name!r} does "
                f"not exist at {p} — refusing to record/cache a dead artifact path"
            )
        if p.is_dir() and not any(p.iterdir()):
            raise StageOutputError(
                f"stage {stage_name!r} exited successfully but its output {art_name!r} is an "
                f"empty directory ({p}) — refusing to record/cache it"
            )


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

    Since 2026-10-05 (review bug 1.4) the record alone isn't enough: the recorded artifact
    *paths* must also still exist on disk (and file hashes must still match). A run directory
    that was partially deleted externally no longer counts as fresh — the stage reruns instead
    of downstream stages consuming dead paths.
    """

    rec = manifest.stages.get(name)
    if rec is None or rec.status not in ("success", "skipped") or rec.cache_key != cache_key:
        return False
    if any(a not in manifest.artifacts for a in rec.artifacts):
        return False
    return _artifacts_intact([manifest.artifacts[a] for a in rec.artifacts])


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

        declared_inputs = {inp: manifest.artifacts[inp] for inp in node.inputs if inp in manifest.artifacts}
        missing_now = [inp for inp in node.inputs if inp not in declared_inputs]
        if missing_now:
            raise MissingDependencyError(
                f"stage {name!r} requires {missing_now} but no upstream stage has produced "
                f"them yet in run {run_id!r} — include their producing stage(s) in this call "
                f"(e.g. via `only`/`from_stage`) or run them first"
            )
        input_hashes = {inp: _input_hash(art) for inp, art in declared_inputs.items()}
        cache_key = compute_cache_key(node.stage_cls, stage_cfg, input_hashes, git_sha)

        if not force and _already_recorded(name, cache_key, manifest):
            continue  # this run's own record is already correct; nothing to write

        cached = None if force else get_cached(cache_key, runs_root=runs_root)
        if cached is not None:
            if _artifacts_intact(list(cached.values())):
                manifest = record_stage_result(
                    run_id,
                    name,
                    status="skipped",
                    artifacts=list(cached.values()),
                    cache_key=cache_key,
                    runs_root=runs_root,
                )
                continue
            # Review bug 1.4 (2026-10-05): a valid cache index entry whose artifact paths were
            # deleted/modified externally used to be reused verbatim as dead paths. Treat it as
            # a cache miss instead and re-run the producing stage (the rerun's put_cached
            # overwrites the stale entry).
            stale_logger = _stage_logger(run_id, name, runs_root=runs_root)
            stale_logger.warning(
                "cache hit for key %s... but recorded artifact paths no longer exist/match; "
                "treating as a cache miss and re-running", cache_key[:12]
            )

        manifest = record_stage_start(run_id, name, runs_root=runs_root)
        logger = _stage_logger(run_id, name, runs_root=runs_root)
        ctx = StageContext(
            run_id=run_id,
            stage_name=name,
            config=stage_cfg,
            run_dir=run_dir(run_id, runs_root=runs_root),
            logger=logger,
            inputs=dict(manifest.artifacts),  # T19: all artifacts for defensive reading
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
        # T12: pre-flight resource gate, right before this stage actually starts — a too-large
        # estimate raises `InsufficientResourcesError` here, *before* anything has run (no
        # monitor started, nothing to measure), so it's recorded exactly like any other stage
        # failure: a clean "failed" manifest entry with a clear message, never a bare crash.
        try:
            check_headroom(node.stage_cls.resources)
        except InsufficientResourcesError as exc:
            manifest = record_stage_result(
                run_id,
                name,
                status="failed",
                error=str(exc),
                log_path=str(stage_log_path(run_id, name, runs_root=runs_root)),
                runs_root=runs_root,
            )
            return manifest

        monitor = ResourceMonitor()
        monitor.start()
        try:
            result, oom_fallback = run_with_oom_retry(node.stage_cls, ctx, name)
            # Post-success output verification (review bug 1.8, 2026-10-05): a 0 exit / clean
            # return is not enough — every declared output must actually exist before the stage
            # may be recorded (and cross-run cached) as a success. Runs inside this try so a
            # missing output is recorded exactly like any other stage failure.
            _verify_declared_outputs(node.stage_cls, name, result)
        except Exception as exc:  # noqa: BLE001 - a failing stage must not crash the scheduler
            peak_vram_mb, peak_ram_mb = monitor.stop()
            manifest = record_stage_result(
                run_id,
                name,
                status="failed",
                error=str(exc),
                log_path=str(stage_log_path(run_id, name, runs_root=runs_root)),
                peak_vram_mb=peak_vram_mb,
                peak_ram_mb=peak_ram_mb,
                runs_root=runs_root,
            )
            return manifest  # stop scheduling; remaining selected stages stay "pending"
        peak_vram_mb, peak_ram_mb = monitor.stop()

        for art in result.values():
            if art.content_hash is None:
                # Files get the fast fingerprint; directory artifacts (capture/model/scene/
                # renders — kinds dataset/model) get a real directory fingerprint now (review
                # bug 1.3, 2026-10-05) instead of silently staying unhashed, so a forced rerun
                # of the producing stage invalidates every downstream cache key.
                h = _hash_artifact_path(Path(art.path))
                if h is not None:
                    art.content_hash = h
                    art.hash_algo = FAST_ALGO

        # Review bug 1.9 (2026-10-05): when the stage only succeeded via run_with_oom_retry's
        # reduced-memory fallback, ctx.config has been swapped to the fallback config (see
        # pipeline.resources.oom_retry.run_with_oom_retry — left applied on success). Record and
        # cache under a key computed from that EFFECTIVE config, not the original one, so a
        # degraded-output success never poisons the cache entry for identical future runs. The
        # deliberate consequence: a resume re-runs the stage (its manifest cache_key no longer
        # matches the original-config key) — it OOMs and falls back again rather than silently
        # reusing fallback outputs as if they were full-config ones.
        effective_key = cache_key
        if oom_fallback is not None:
            effective_key = compute_cache_key(node.stage_cls, ctx.config, input_hashes, git_sha)

        manifest = record_stage_result(
            run_id,
            name,
            status="success",
            artifacts=list(result.values()),
            cache_key=effective_key,
            log_path=str(stage_log_path(run_id, name, runs_root=runs_root)),
            peak_vram_mb=peak_vram_mb,
            peak_ram_mb=peak_ram_mb,
            oom_fallback=oom_fallback,
            runs_root=runs_root,
        )
        put_cached(effective_key, run_id, name, result, runs_root=runs_root)

    if not manifest.stages:
        # An empty DAG (e.g. no real stages registered for any role yet) has nothing to roll its
        # status up from `record_stage_result` — treat "nothing to do" as trivially done.
        def _mark_trivially_done(m: RunManifest) -> None:
            if m.status == "pending":
                m.status = "success"

        manifest = update_manifest(run_id, _mark_trivially_done, runs_root=runs_root)

    return manifest
