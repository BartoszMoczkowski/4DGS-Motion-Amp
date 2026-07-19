"""Shaping artifact content for MCP tool responses (T14): small summaries, never big blobs.

Per the task's own scope note ("Large artifacts never dumped wholesale; npz returns a summary")
and ``planning/ARCHITECTURE.md``'s Layer 2 "result shaping" bullet, this module is the one place
that decides, per :class:`pipeline.artifacts.Artifact` kind, what ``read_artifact``/``get_preview``
actually return to Claude over MCP — never the raw file content of something that could be
gigabytes (a point cloud, a training checkpoint, a multi-GB dataset directory).

Deliberately independent of ``mcp_server.server``/the ``mcp`` package itself (no ``FastMCP``/
``Image`` import here) so this summarization logic can be unit-tested directly against real files
on disk, without spinning up a server — mirrors ``pipeline.config.bridge``'s own "pure function,
plumbing lives elsewhere" split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pipeline.artifacts import Artifact

#: A JSON artifact larger than this is still read and parsed (so a caller reliably gets structured
#: data back for anything reasonably sized), but the parsed content is dropped from the response —
#: keeps an unusually large JSON file from itself becoming the "big blob" this module exists to
#: avoid. Fetch the ``run://<run_id>/artifact/<name>`` resource for the full content instead.
_MAX_INLINE_JSON_BYTES = 64 * 1024

#: Shallow (one-level) directory listings for `dataset`/`model` artifacts are capped here so a
#: checkpoint directory with thousands of small files doesn't itself become a wall of text.
_MAX_DIR_ENTRIES = 200

#: kinds `get_preview` can actually render, and how.
_PREVIEW_IMAGE_KINDS = frozenset({"png"})
_PREVIEW_VIDEO_KINDS = frozenset({"video"})


class ArtifactNotPreviewableError(ValueError):
    """Raised by :func:`preview_kind` for a kind ``get_preview`` has no rendering for."""


def preview_kind(artifact: Artifact) -> str:
    """``"image"`` or ``"video"`` for a previewable artifact; raises
    :class:`ArtifactNotPreviewableError` for anything else (npz/ply/json/dataset/model/usd — use
    ``read_artifact`` for those instead).
    """
    if artifact.kind in _PREVIEW_IMAGE_KINDS:
        return "image"
    if artifact.kind in _PREVIEW_VIDEO_KINDS:
        return "video"
    raise ArtifactNotPreviewableError(
        f"artifact {artifact.name!r} has kind={artifact.kind!r} — not previewable (only "
        f"{sorted(_PREVIEW_IMAGE_KINDS | _PREVIEW_VIDEO_KINDS)} are); use read_artifact for a "
        "text/JSON/npz/directory summary instead."
    )


def read_artifact_summary(artifact: Artifact) -> dict[str, Any]:
    """A small, JSON-safe summary of ``artifact``'s content — shape depends on its ``kind``:

    - ``json``: parsed content (dropped if the raw file exceeds
      :data:`_MAX_INLINE_JSON_BYTES` — see that constant's docstring) plus the raw byte size.
    - ``npz``: **not raw arrays** — per-key ``shape``/``dtype``/``min``/``max``/``mean``/
      (``nan_count`` for float arrays). A stage's own real ``.npz`` outputs (``trajectories.npz``,
      ``segmentation.npz``, ...) are exactly the "big blob" this function must never dump
      wholesale.
    - ``dataset``/``model`` (directories): a shallow, one-level file listing with per-entry sizes,
      capped at :data:`_MAX_DIR_ENTRIES`, plus a total size and entry count.
    - ``ply``: vertex/face counts parsed from the PLY header only — never the vertex/face data.
    - ``png``/``video``/``usd``: no text summary here (visual/binary) — use ``get_preview`` for
      png/video; still reports path/size so the artifact's existence is confirmable.

    Always includes ``name``/``kind``/``path``/``producing_stage``/``content_hash`` so the caller
    can tell *what* was summarized even for a kind with nothing more specific to say.
    """
    base: dict[str, Any] = {
        "name": artifact.name,
        "kind": artifact.kind,
        "path": artifact.path,
        "producing_stage": artifact.producing_stage,
        "content_hash": artifact.content_hash,
    }
    path = Path(artifact.path)

    if artifact.kind == "json":
        base.update(_summarize_json(path))
    elif artifact.kind == "npz":
        base.update(_summarize_npz(path))
    elif artifact.kind in ("dataset", "model"):
        base.update(_summarize_directory(path))
    elif artifact.kind == "ply":
        base.update(_summarize_ply(path))
    else:
        base["note"] = (
            f"kind={artifact.kind!r} has no text summary here — use get_preview for png/video."
        )
        if path.is_file():
            base["size_bytes"] = path.stat().st_size
    return base


def _summarize_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"error": f"no such file: {path}"}
    size = path.stat().st_size
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"size_bytes": size, "error": f"invalid JSON: {exc}"}
    result: dict[str, Any] = {"size_bytes": size}
    if size <= _MAX_INLINE_JSON_BYTES:
        result["content"] = parsed
    else:
        result["truncated"] = True
        result["note"] = (
            f"{size} bytes exceeds the {_MAX_INLINE_JSON_BYTES}-byte inline limit — fetch the "
            "run://<run_id>/artifact/<name> MCP resource for the full content."
        )
    return result


def _summarize_npz(path: Path) -> dict[str, Any]:
    import numpy as np  # local import — this module must stay importable without numpy loaded

    if not path.is_file():
        return {"error": f"no such file: {path}"}
    arrays: dict[str, Any] = {}
    with np.load(path) as data:
        for key in data.files:
            arr = data[key]
            entry: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            if np.issubdtype(arr.dtype, np.number) and arr.size:
                entry["min"] = float(np.nanmin(arr))
                entry["max"] = float(np.nanmax(arr))
                entry["mean"] = float(np.nanmean(arr))
                if np.issubdtype(arr.dtype, np.floating):
                    entry["nan_count"] = int(np.isnan(arr).sum())
            arrays[key] = entry
    return {"size_bytes": path.stat().st_size, "arrays": arrays}


def _summarize_directory(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"error": f"not a directory (or doesn't exist): {path}"}
    entries = []
    total_size = 0
    total_count = 0
    for child in sorted(path.iterdir()):
        total_count += 1
        size = child.stat().st_size if child.is_file() else _dir_size(child)
        total_size += size
        if len(entries) < _MAX_DIR_ENTRIES:
            entries.append({"name": child.name, "is_dir": child.is_dir(), "size_bytes": size})
    return {
        "entries": entries,
        "entry_count": total_count,
        "truncated": total_count > _MAX_DIR_ENTRIES,
        "total_size_bytes": total_size,
    }


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _summarize_ply(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"error": f"no such file: {path}"}
    header_lines: list[str] = []
    with path.open("rb") as f:
        for _ in range(200):  # a PLY header is always short; this bounds a malformed file's read
            line = f.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
    return {
        "size_bytes": path.stat().st_size,
        "vertex_count": _ply_element_count(header_lines, "vertex"),
        "face_count": _ply_element_count(header_lines, "face"),
    }


def _ply_element_count(header_lines: list[str], element: str) -> Optional[int]:
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == element:
            try:
                return int(parts[2])
            except ValueError:
                return None
    return None
