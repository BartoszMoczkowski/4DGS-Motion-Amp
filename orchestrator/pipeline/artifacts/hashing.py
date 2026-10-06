"""Content hashing for artifact identity/caching.

Full-file SHA-256 is correct but too slow to run on every manifest touch for the multi-GB point
clouds / videos this pipeline produces — flagged as a real risk in
``planning/tasks/T05-dag-scheduler-and-cache.md`` ("hash big artifacts by size+mtime+partial-hash
if full hashing is too slow; document the choice"). This is that documented choice: the default
strategy is a **fingerprint**, not a cryptographic hash of the full content — size + mtime + a
SHA-256 of the first/last chunk. That is enough to detect "this file changed" for T05's cache-key
use case and to give artifacts a stable identity for dedup, without reading gigabytes on every
call. Pass ``fast=False`` when a real full-content hash is actually needed (e.g. verifying a
byte-for-byte copy).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Bytes read from the start and end of the file in fast mode.
_FAST_CHUNK_BYTES = 1024 * 1024  # 1 MiB
#: Streaming read size for the full-content hash.
_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB

FAST_ALGO = "fast-sha256-v1"
FULL_ALGO = "sha256"

#: Files larger than this contribute only size+mtime to a directory fingerprint; smaller
#: files get their full content hashed in. Keeps ``hash_directory`` cheap on multi-GB capture
#: dirs (thousands of frame PNGs) while still content-verifying the small metadata files
#: (``cameras_gt.json``, ``*.npy``, ...) that actually define a dataset's identity.
_DIR_CONTENT_HASH_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB


def hash_path(path: str | Path, *, fast: bool = True) -> str:
    """Return a content hash/fingerprint for a file.

    Raises ``FileNotFoundError`` if ``path`` doesn't exist or isn't a regular file — only
    meaningful for files, not directories. A ``dataset``/``model`` artifact that's a whole
    directory tree is the caller's decision to hash per-file or leave
    :attr:`pipeline.artifacts.models.Artifact.content_hash` as ``None``.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"hash_path: not a file: {p}")

    if fast:
        return _fast_fingerprint(p)
    return f"{FULL_ALGO}:{_full_sha256(p)}"


def _fast_fingerprint(p: Path) -> str:
    stat = p.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    with p.open("rb") as f:
        digest.update(f.read(_FAST_CHUNK_BYTES))
        if size > _FAST_CHUNK_BYTES:
            f.seek(max(size - _FAST_CHUNK_BYTES, 0))
            digest.update(f.read(_FAST_CHUNK_BYTES))
    return f"{FAST_ALGO}:{size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


def hash_directory(path: str | Path) -> str:
    """Return a content fingerprint for a whole directory tree (``dataset``/``model`` artifacts).

    ``hash_path`` raises on directories by design, which left directory artifacts with
    ``content_hash=None`` — and the scheduler's ``art.content_hash or ""`` then contributed a
    constant ``""`` to every downstream cache key, so force-rerunning ``capture.isaac``/
    ``train.default`` never invalidated ``convert``/``render``/``seg_extract`` (see
    ``reviews/orchestrator-correctness-review.md`` bug 1.3). This is the pragmatic fix promised
    there: a SHA-256 over the sorted relative file paths, each with its size, mtime, and — for
    files up to ``_DIR_CONTENT_HASH_MAX_BYTES`` — full content hash; larger files (the actual
    frame/megabyte payload) contribute size+mtime only, mirroring ``_fast_fingerprint``'s
    "detect change, don't read gigabytes" tradeoff.

    Raises ``FileNotFoundError`` if ``path`` doesn't exist or isn't a directory.
    """

    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"hash_directory: not a directory: {p}")

    digest = hashlib.sha256()
    n_files = 0
    total_bytes = 0
    files = sorted(
        (f for f in p.rglob("*") if f.is_file()),
        key=lambda f: f.relative_to(p).as_posix(),
    )
    for f in files:
        stat = f.stat()
        rel = f.relative_to(p).as_posix()
        n_files += 1
        total_bytes += stat.st_size
        digest.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}".encode("utf-8"))
        if stat.st_size <= _DIR_CONTENT_HASH_MAX_BYTES:
            digest.update(_full_sha256(f).encode("ascii"))
        digest.update(b"\n")
    return f"{FAST_ALGO}:dir:{n_files}:{total_bytes}:{digest.hexdigest()}"


def _full_sha256(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
