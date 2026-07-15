"""Query helpers over ``runs/`` — the read surface ``pipeline.api`` exposes to Layers 2/3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .manifest import ManifestCorruptError, load_manifest
from .models import Artifact, RunManifest
from .paths import get_runs_root


class ArtifactNotFoundError(KeyError):
    """No artifact named ``artifact_name`` in the given run's manifest."""


def list_runs(*, runs_root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Summaries of all known runs, most recently updated first.

    Runs with an unreadable manifest (:class:`pipeline.artifacts.manifest.ManifestCorruptError`)
    are skipped rather than raising — one bad run directory shouldn't take down a listing of all
    the others. Call :func:`get_manifest` directly on that ``run_id`` to see the corruption.
    """

    root = runs_root or get_runs_root()
    if not root.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            manifest = load_manifest(entry.name, runs_root=root)
        except (FileNotFoundError, ManifestCorruptError):
            continue
        summaries.append(
            {
                "run_id": manifest.run_id,
                "preset": manifest.preset,
                "status": manifest.status,
                "created_at": manifest.created_at,
                "updated_at": manifest.updated_at,
            }
        )
    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return summaries


def get_manifest(run_id: str, *, runs_root: Optional[Path] = None) -> RunManifest:
    """The full manifest for one run. Propagates ``FileNotFoundError``/``ManifestCorruptError``."""

    return load_manifest(run_id, runs_root=runs_root or get_runs_root())


def list_artifacts(run_id: str, *, runs_root: Optional[Path] = None) -> list[Artifact]:
    """All artifacts recorded for ``run_id``, in manifest order."""

    return list(get_manifest(run_id, runs_root=runs_root).artifacts.values())


def get_artifact(
    run_id: str, artifact_name: str, *, runs_root: Optional[Path] = None
) -> Artifact:
    """One artifact by name. Raises :class:`ArtifactNotFoundError` if unknown."""

    manifest = get_manifest(run_id, runs_root=runs_root)
    try:
        return manifest.artifacts[artifact_name]
    except KeyError:
        raise ArtifactNotFoundError(
            f"run {run_id!r} has no artifact named {artifact_name!r} "
            f"(has: {sorted(manifest.artifacts)})"
        ) from None
