"""Cache key computation + a small cross-run cache index ("skip when a fresh output exists").

Cache key = hash(resolved stage config + input-artifact content hashes + code version), per
``planning/tasks/T05-dag-scheduler-and-cache.md``. "Code version" = git SHA + a hash of the
stage class's own source file, so editing either the repo (a commit) or just the stage's script
in a dirty tree invalidates it — matches the task's explicit gotcha ("include code version so a
script edit reruns").

The index (``runs/.cache/index.json``) is what makes caching work *across* runs, not just within
one: :func:`pipeline.dag.scheduler.run_dag` always checks the current run's own manifest first
(cheap, no lookup needed) and falls back to this index so a brand-new run for an unchanged preset
can still skip stages whose last successful execution was a different run. Same atomic
write-temp-rename discipline as ``pipeline.artifacts.manifest`` (a leaf module in its own right —
this file intentionally doesn't import that module's private helper, it's small enough to repeat).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..artifacts import Artifact, get_runs_root

_INDEX_FILENAME = "index.json"


def _cache_dir(runs_root: Optional[Path] = None) -> Path:
    return (runs_root or get_runs_root()) / ".cache"


def _index_path(runs_root: Optional[Path] = None) -> Path:
    return _cache_dir(runs_root) / _INDEX_FILENAME


def stage_source_hash(stage_cls: type) -> str:
    """Full SHA-256 of the source *file* defining ``stage_cls``.

    Whole-file, not just the method body: a stage class is small and this is the simplest way to
    catch "the script this stage wraps changed" without trying to hash imported modules
    transitively. ``"unknown"`` if the source can't be located (e.g. a class defined dynamically
    in a test) — that still invalidates consistently within one process, it just isn't a real
    content hash.
    """

    try:
        src_file = inspect.getsourcefile(stage_cls) or inspect.getfile(stage_cls)
        if not src_file or not Path(src_file).is_file():
            return "unknown"
        digest = hashlib.sha256(Path(src_file).read_bytes()).hexdigest()
        return f"sha256:{digest}"
    except (TypeError, OSError):
        return "unknown"


def code_version(stage_cls: type, git_sha: Optional[str]) -> str:
    """``"<git-sha-or-'nogit'>:<stage-source-hash>"`` — the "code version" half of a cache key."""

    return f"{git_sha or 'nogit'}:{stage_source_hash(stage_cls)}"


def compute_cache_key(
    stage_cls: type,
    resolved_stage_config: dict[str, Any],
    input_hashes: dict[str, str],
    git_sha: Optional[str],
) -> str:
    """Hash resolved config + input-artifact hashes + code version into one cache key.

    ``input_hashes`` should be ``{artifact_name: content_hash}`` for every declared input
    (empty-string hash for an artifact whose ``content_hash`` is ``None``, e.g. an un-hashed
    directory artifact — deliberately still part of the key rather than skipped, so at least the
    *set* of inputs is captured even when content can't be).
    """

    payload = {
        "stage": getattr(stage_cls, "name", stage_cls.__qualname__),
        "config": resolved_stage_config,
        "inputs": dict(sorted(input_hashes.items())),
        "code_version": code_version(stage_cls, git_sha),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_index(runs_root: Optional[Path] = None) -> dict[str, Any]:
    path = _index_path(runs_root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # A corrupt cache index is never worth crashing over — treat it as empty (cold cache).
        return {}


def get_cached(cache_key: str, *, runs_root: Optional[Path] = None) -> Optional[dict[str, Artifact]]:
    """The artifact set recorded for ``cache_key`` by an earlier successful run, if any."""

    entry = _load_index(runs_root).get(cache_key)
    if entry is None:
        return None
    return {name: Artifact.model_validate(data) for name, data in entry["artifacts"].items()}


def put_cached(
    cache_key: str,
    run_id: str,
    stage_name: str,
    artifacts: dict[str, Artifact],
    *,
    runs_root: Optional[Path] = None,
) -> None:
    """Record a successful stage's outputs under ``cache_key`` for future runs to reuse."""

    index = _load_index(runs_root)
    index[cache_key] = {
        "run_id": run_id,
        "stage": stage_name,
        "artifacts": {name: art.model_dump() for name, art in artifacts.items()},
    }
    _atomic_write_json(_index_path(runs_root), index)
