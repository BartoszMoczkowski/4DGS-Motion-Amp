"""Read/write the per-run manifest.json — atomic (write-temp-rename), corrupt-safe reads.

This is the module T05 (scheduler) calls into while running stages, and what Layer 2's MCP tools
(list runs, get status, read artifact) ultimately read through ``pipeline.artifacts.store`` /
``pipeline.api``. See ``planning/tasks/T03-artifacts-and-manifest.md`` for scope.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .models import Artifact, RunManifest, StageRecord
from .paths import config_snapshot_path, ensure_run_dirs, manifest_path, REPO_ROOT


class ManifestError(Exception):
    """Base for manifest read/write errors."""


class ManifestCorruptError(ManifestError):
    """``manifest.json`` exists but isn't valid JSON, or doesn't match the schema.

    Raised instead of letting a raw ``json.JSONDecodeError``/pydantic ``ValidationError``
    propagate, so callers (T05's resume logic, MCP tools) can catch one clear exception type for
    "this run's manifest is unreadable" — a partial write caught mid-flight, disk corruption, or
    schema drift — and decide what to do, rather than crashing.
    """

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"corrupt or partial manifest at {path}: {cause!r}")
        self.path = path
        self.cause = cause


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Best-effort ``git rev-parse HEAD``. ``None`` if git isn't available or it's not a repo."""

    root = repo_root or REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_manifest(
    run_id: str,
    preset: str,
    resolved_config: dict[str, Any],
    *,
    stage_names: list[str] = (),
    git_sha: Optional[str] = None,
) -> RunManifest:
    """Build a fresh, all-``pending`` manifest. Doesn't touch disk — pair with :func:`save_manifest`."""

    now = _utc_now_iso()
    return RunManifest(
        run_id=run_id,
        preset=preset,
        resolved_config=resolved_config,
        git_sha=git_sha,
        created_at=now,
        updated_at=now,
        status="pending",
        stages={name: StageRecord() for name in stage_names},
        artifacts={},
    )


def create_run(
    run_id: str,
    preset: str,
    resolved_config: dict[str, Any],
    *,
    stage_names: list[str] = (),
    runs_root: Optional[Path] = None,
) -> RunManifest:
    """Create ``runs/<run_id>/`` (+ ``logs/``), write the config snapshot, and an initial manifest."""

    ensure_run_dirs(run_id, runs_root=runs_root)
    manifest = new_manifest(
        run_id, preset, resolved_config, stage_names=list(stage_names), git_sha=get_git_sha()
    )
    _atomic_write_json(config_snapshot_path(run_id, runs_root=runs_root), resolved_config)
    save_manifest(manifest, runs_root=runs_root)
    return manifest


#: Belt-and-suspenders retry for a transient `PermissionError` on `os.replace` -- see
#: `_replace_with_retry`'s docstring. Real protection against *this process's own* concurrent
#: readers/writers now comes from `_lock_for`'s per-path lock (below); this only remains to cover
#: a handle held by something outside this process (antivirus, a second orchestrator instance, an
#: editor with the file open) that a Python-level lock can't see.
_REPLACE_RETRIES = 25
_REPLACE_RETRY_DELAY_S = 0.05


def _replace_with_retry(src: str, dst: Path) -> None:
    """``os.replace(src, dst)``, retrying briefly on a transient ``PermissionError``.

    **Found on Bartosz's real Windows machine (2026-07-19)**:
    ``test_concurrent_writes_never_produce_a_torn_file`` (multiple threads hammering
    ``save_manifest``/``load_manifest`` against the same ``manifest.json``) failed with a raw
    ``PermissionError(13, 'Access is denied')`` from this call -- invisible in the sandbox's Linux
    runs, where ``os.replace`` renames over an open file regardless of who else has it open.
    Windows enforces mandatory file locking: ``MoveFileEx`` (what ``os.replace`` uses there) can
    be transiently denied while another thread has ``dst`` open, even though nothing is actually
    wrong -- the rename becomes valid again the instant that other handle's read completes, which
    is typically sub-millisecond for a small JSON file. Retrying a few times with a short sleep is
    the standard cross-platform mitigation for this exact behavior (POSIX essentially never raises
    ``PermissionError`` here, so the loop exits on the first attempt there; this is a Windows-only
    code path in practice). Re-raises the last ``PermissionError`` if every attempt fails --a
    genuinely locked file (e.g. antivirus, a stuck handle) should still surface as an error, not
    hang or silently drop the write.

    **Follow-up (2026-07-19, same day):** the retry alone wasn't enough under this test's real
    stress load (4 writers x 50 iterations racing 4 continuously-looping readers) -- on Bartosz's
    machine it still failed, both as a raw ``PermissionError`` here (contention outlasting the
    retry budget) *and* as ``ManifestCorruptError`` from :func:`load_manifest`'s read hitting the
    same transient sharing violation from the *reader* side, which this function never covered.
    See :func:`_lock_for` for the actual fix -- a per-path lock that serializes this *process's
    own* concurrent access so the OS-level race can't happen at all between our own threads; this
    retry now only matters for a handle held by something outside this process.
    """
    last_exc: Optional[PermissionError] = None
    for _ in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(_REPLACE_RETRY_DELAY_S)
    assert last_exc is not None
    raise last_exc


_manifest_locks: dict[str, threading.Lock] = {}
_manifest_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """A process-wide lock keyed by resolved manifest path -- one lock per run's ``manifest.json``,
    shared by every :func:`save_manifest`/:func:`load_manifest` call against that same path.

    **Added 2026-07-19** alongside :func:`_replace_with_retry`'s retry loop, which turned out not
    to be enough by itself: under real concurrent-thread load on Windows, both writers (``os.
    replace``) *and* readers (``path.read_text``, wrapped into ``ManifestCorruptError`` by
    :func:`load_manifest`'s existing ``OSError`` handling) hit transient sharing-violation
    ``PermissionError``s, and retrying a fixed number of times is inherently racy under sustained
    contention (there's always some load high enough to exhaust the budget). A lock is
    deterministic instead of probabilistic: as long as only *this process's* threads are involved
    (the actual scenario ``test_concurrent_writes_never_produce_a_torn_file`` exercises, and the
    only one this module has ever claimed to make safe -- see :func:`save_manifest`'s docstring),
    serializing every read and write against a given manifest path means no two threads ever have
    it open at the same instant, so the Windows-specific race simply can't occur. Doesn't replace
    :func:`_replace_with_retry` -- a lock can't protect against a handle held by something outside
    this process, which the retry loop still covers.

    Keyed by ``str(path)`` (not the run_id alone) since callers can point different run_ids at
    different ``runs_root`` overrides (tests do this constantly) -- two different physical files
    must never share a lock, and two calls for the *same* physical file (even reached via a
    differently-spelled but equal path) should. The registry itself is never cleared -- one small
    ``Lock`` per distinct manifest path for the life of the process is negligible, and removing
    entries would reopen a tiny window for two threads to each create a *different* lock for the
    same path right as an entry gets evicted.
    """
    key = str(path)
    with _manifest_locks_guard:
        lock = _manifest_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _manifest_locks[key] = lock
        return lock


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # Explicit UTF-8 -- these files (manifest.json, config_snapshot.json) get written from
        # Bartosz's native Windows process but may be read back from inside a Linux container (or
        # by any other locale); without this, `open`'s default falls back to the OS locale's
        # preferred encoding (cp1252 on Windows), which silently mis-encodes any non-ASCII
        # character. Same bug class as `pipeline.config.bridge.write_bridge`'s 2026-07-18 fix.
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        # atomic on the same filesystem, POSIX and Windows alike -- but Windows can transiently
        # deny the rename while another thread has `path` open for reading; see
        # `_replace_with_retry`'s docstring.
        _replace_with_retry(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_manifest(manifest: RunManifest, *, runs_root: Optional[Path] = None) -> None:
    """Atomic write-temp-rename.

    Concurrent-safe in the sense that matters here: every write lands whole or not at all, so a
    reader never observes a truncated/partial file. It does *not* protect a read-modify-write
    against a lost update if two callers race (see :func:`update_manifest`'s docstring) —
    serialize your own read-modify-write calls if that matters.

    Guarded by :func:`_lock_for` (2026-07-19) so this process's own concurrent
    ``save_manifest``/``load_manifest`` calls against the same path never race at the OS level —
    see that function's docstring for why the write-temp-rename dance alone wasn't enough on
    Windows under real concurrent load.
    """

    path = manifest_path(manifest.run_id, runs_root=runs_root)
    with _lock_for(path):
        _atomic_write_json(path, manifest.model_dump())


def load_manifest(run_id: str, *, runs_root: Optional[Path] = None) -> RunManifest:
    """Read + validate ``manifest.json``.

    Raises ``FileNotFoundError`` if the run/manifest doesn't exist yet, or
    :class:`ManifestCorruptError` if it exists but is unreadable/invalid.

    Guarded by :func:`_lock_for` (2026-07-19) — see :func:`save_manifest`'s docstring.
    """

    path = manifest_path(run_id, runs_root=runs_root)
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    with _lock_for(path):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ManifestCorruptError(path, exc) from exc
    try:
        return RunManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestCorruptError(path, exc) from exc


def update_manifest(
    run_id: str,
    mutate: Callable[[RunManifest], None],
    *,
    runs_root: Optional[Path] = None,
) -> RunManifest:
    """Read-modify-write: load the manifest, apply ``mutate`` in place, bump ``updated_at``, save.

    Not itself a cross-process lock (see :func:`save_manifest`'s note) — callers that need true
    multi-writer safety (T05, if it ever parallelizes stage execution) should serialize their own
    writes; this only guarantees each individual write is atomic, never a torn file.
    """

    manifest = load_manifest(run_id, runs_root=runs_root)
    mutate(manifest)
    manifest.updated_at = _utc_now_iso()
    save_manifest(manifest, runs_root=runs_root)
    return manifest


def record_stage_start(
    run_id: str, stage: str, *, runs_root: Optional[Path] = None
) -> RunManifest:
    """Mark ``stage`` running and the overall run running. Convenience wrapper T04/T05 stages use."""

    def _mutate(m: RunManifest) -> None:
        rec = m.stages.setdefault(stage, StageRecord())
        rec.status = "running"
        rec.start_time = _utc_now_iso()
        rec.end_time = None
        rec.error = None
        m.status = "running"

    return update_manifest(run_id, _mutate, runs_root=runs_root)


def record_stage_result(
    run_id: str,
    stage: str,
    *,
    status: str,
    artifacts: list[Artifact] = (),
    error: Optional[str] = None,
    log_path: Optional[str] = None,
    cache_key: Optional[str] = None,
    peak_vram_mb: Optional[float] = None,
    peak_ram_mb: Optional[float] = None,
    oom_fallback: Optional[dict[str, Any]] = None,
    runs_root: Optional[Path] = None,
) -> RunManifest:
    """Finish a stage: set its terminal status/timing, register any artifacts it produced.

    Rolls the overall run status up to ``failed``/``success`` once every stage is terminal —
    ``running``/``pending`` in between are left alone (T05 owns the scheduling decisions; this
    just keeps the manifest's own summary consistent with what it's been told).

    ``cache_key`` (T05) is stored on the record for ``success``/``skipped`` results so a later
    call can compare it against a freshly-computed cache key to decide whether to skip. Left
    ``None`` for a ``failed`` result — a failed attempt has no valid cache key to reuse.

    ``peak_vram_mb``/``peak_ram_mb``/``oom_fallback`` (T12): only ever passed by
    ``pipeline.dag.scheduler`` for a stage that actually ran (never for a ``"skipped"`` cache hit,
    which has no execution to have measured) — each is only written when not ``None``, the same
    "only set if given" pattern ``cache_key`` already uses, so a caller that doesn't know about
    these fields yet (e.g. a hand-built test call) leaves them untouched rather than clobbering a
    previous value with ``None``.
    """

    def _mutate(m: RunManifest) -> None:
        rec = m.stages.setdefault(stage, StageRecord())
        rec.status = status  # type: ignore[assignment]
        rec.end_time = _utc_now_iso()
        if rec.start_time is not None:
            start = datetime.fromisoformat(rec.start_time)
            end = datetime.fromisoformat(rec.end_time)
            rec.wall_time_s = (end - start).total_seconds()
        rec.error = error
        if log_path is not None:
            rec.log_path = log_path
        if cache_key is not None:
            rec.cache_key = cache_key
        if peak_vram_mb is not None:
            rec.peak_vram_mb = peak_vram_mb
        if peak_ram_mb is not None:
            rec.peak_ram_mb = peak_ram_mb
        if oom_fallback is not None:
            rec.oom_fallback = oom_fallback
        for art in artifacts:
            m.artifacts[art.name] = art
            if art.name not in rec.artifacts:
                rec.artifacts.append(art.name)

        if status == "failed":
            m.status = "failed"
        elif m.stages and all(s.status in ("success", "skipped") for s in m.stages.values()):
            m.status = "success"

    return update_manifest(run_id, _mutate, runs_root=runs_root)
