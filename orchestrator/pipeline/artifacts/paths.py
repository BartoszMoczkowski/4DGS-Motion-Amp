"""Run directory layout: where runs, manifests, config snapshots, and logs live on disk.

This is **not** the path-*translation* module (mapping host <-> container paths is T06's job,
"the only place that logic may live" per ``planning/INSTRUCTIONS.md``) — this module
only knows about one convention on the host: everything for a run lives under
``runs/<run_id>/``.

It is also the **single validation choke point** for two trust boundaries (round-2 correctness
fixes, 2026-10-05):

- :func:`validate_run_id` — every ``run_id`` that gets interpolated into a path under the runs
  root. Called by :func:`run_dir`, so *every* function here (and therefore every caller:
  manifest read/write, the DAG scheduler, the MCP tools/resources) rejects a traversal-flavored
  id (``"../../x"``, ``".."``, anything with a path separator) before any filesystem touch.
- :func:`validate_external_artifact_path` / :func:`resolve_servable_artifact_path` — the
  confinement policy for artifact paths that leave the host (seeded ``external_artifacts`` in
  ``pipeline.api.run_pipeline``; raw serving in the MCP layer): absolute, existing, and under
  one of the pipeline's known roots (:func:`allowed_artifact_roots`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from ..paths import get_roots

#: orchestrator/pipeline/artifacts/paths.py -> repo root (4DGS-Motion-Amp/).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default root for all run directories, sibling to ``output/``/``data/``. Overridable per-call
#: (tests, alternate hosts) via the ``runs_root`` kwarg most functions here take, or globally via
#: the ``PIPELINE_RUNS_ROOT`` env var (see :func:`get_runs_root`).
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

_ENV_VAR = "PIPELINE_RUNS_ROOT"


def get_runs_root() -> Path:
    """The runs root to use: ``$PIPELINE_RUNS_ROOT`` if set, else :data:`DEFAULT_RUNS_ROOT`.

    Read at call time (not import time) so tests can flip it via ``monkeypatch.setenv`` without
    reloading this module, and so importing ``pipeline`` never touches the filesystem.
    """

    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else DEFAULT_RUNS_ROOT


class InvalidRunIdError(ValueError):
    """A ``run_id`` that isn't safe to interpolate into a path under the runs root."""


#: What :func:`pipeline.api.new_run_id` mints (``"<preset>-<8 hex>"``) plus anything a human would
#: reasonably type — letters, digits, ``_``, ``-``, ``.``. Crucially no path separators, and no
#: ``..`` (checked separately below, since a pure charset check would still allow ``".."``).
_RUN_ID_CHARSET = re.compile(r"[A-Za-z0-9_.\-]+")


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged if it's safe to use as a directory name under the runs root;
    raise :class:`InvalidRunIdError` otherwise.

    The charset alone blocks every path separator (``/``, ``\\``, ``:``); the explicit ``..``
    rejection covers the one traversal spelling the charset would still admit, and the all-dots
    rejection covers ``"."`` (which would alias the runs root itself, so ``manifest.json`` would
    be written straight into it). This is the ONLY place run ids are validated — every path
    built under the runs root goes through :func:`run_dir`, which calls this.
    """
    if (
        not isinstance(run_id, str)
        or not _RUN_ID_CHARSET.fullmatch(run_id)
        or ".." in run_id
        or not run_id.strip(".")
    ):
        raise InvalidRunIdError(
            f"invalid run_id {run_id!r}: must match [A-Za-z0-9_.-]+, contain no '..', and not be all dots"
        )
    return run_id


def run_dir(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    validate_run_id(run_id)
    root = runs_root or get_runs_root()
    d = root / run_id
    # Belt-and-suspenders behind validate_run_id's charset check: prove the resolved path stays
    # under the intended root even if the charset rule is ever loosened (or a weird filesystem
    # alias/junction is involved).
    if not d.resolve().is_relative_to(root.resolve()):
        raise InvalidRunIdError(f"run_id {run_id!r} resolves outside the runs root {root}")
    return d


def manifest_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "manifest.json"


def config_snapshot_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "config_snapshot.json"


def log_dir(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_root=runs_root) / "logs"


def stage_log_path(run_id: str, stage: str, *, runs_root: Optional[Path] = None) -> Path:
    # stage names look like "segment.mbs" — keep the dot out of the filename's extension slot.
    return log_dir(run_id, runs_root=runs_root) / f"{stage.replace('.', '_')}.log"


def ensure_run_dirs(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    """Create ``runs/<run_id>/`` and its ``logs/`` subdir if missing. Returns the run dir."""

    d = run_dir(run_id, runs_root=runs_root)
    log_dir(run_id, runs_root=runs_root).mkdir(parents=True, exist_ok=True)
    return d


# --- artifact path confinement (round-2 correctness fixes, 2026-10-05) --------------------------
#
# Two trust boundaries share one policy: (a) `pipeline.api.run_pipeline`'s `external_artifacts`
# seeding (an MCP client supplies raw paths that stages then open natively), and (b) the MCP
# layer serving artifact bytes back (`artifact_resource`/`get_preview`/`read_artifact`). The
# policy: an artifact path must be absolute, must exist, and — once resolved (no `..` tricks,
# symlink/junction hops included) — must live under one of the pipeline's known roots. Those
# roots are exactly the ones `pipeline.paths` (T06) already defines as "every path the pipeline
# cares about": the repo root, the assets root, plus the runs root (stage outputs under
# `runs/<run_id>/` and externally-produced assets under e.g. `Q:/Omniverse` are both legitimate;
# `C:/Users/<anything-else>` is not).


class ArtifactPathError(ValueError):
    """An artifact path failed confinement/existence validation."""


#: Artifact kinds whose ``path`` is a whole directory tree (``Artifact``'s own docstring: "a
#: directory for ``dataset``/``model`` kinds, a file otherwise"). Kept as a literal tuple rather
#: than imported from ``.models`` so this module stays a leaf (``models`` already imports nothing
#: from here; importing it back would be fine, but the two-kind list is the entire coupling).
_DIRECTORY_KINDS = ("dataset", "model")


def allowed_artifact_roots(*, runs_root: Optional[Path] = None) -> list[Path]:
    """The resolved roots an artifact path may live under: runs root, repo root, assets root.

    Repo/assets come from :func:`pipeline.paths.get_roots` (T06 — the one module that may know
    about them), so the ``PIPELINE_REPO_ROOT``/``PIPELINE_ASSETS_ROOT`` overrides apply here too.
    """
    roots = get_roots()
    return [
        (runs_root or get_runs_root()).resolve(),
        roots.repo_root_host.resolve(),
        roots.assets_root_host.resolve(),
    ]


def _resolve_under_allowed_roots(
    raw_path: str, *, runs_root: Optional[Path] = None
) -> Path:
    """Resolve ``raw_path`` and require it to stay under :func:`allowed_artifact_roots`."""

    p = Path(raw_path)
    if not p.is_absolute():
        raise ArtifactPathError(
            f"artifact path {raw_path!r} must be absolute (no relative paths, no '..' guessing)"
        )
    resolved = p.resolve()
    roots = allowed_artifact_roots(runs_root=runs_root)
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ArtifactPathError(
            f"artifact path {raw_path!r} resolves to {resolved}, which is outside every allowed "
            f"root ({', '.join(str(r) for r in roots)}); put the asset under the repo or the "
            "assets root (PIPELINE_ASSETS_ROOT), or adjust those roots"
        )
    return resolved


def validate_external_artifact_path(
    raw_path: str, kind: str, *, runs_root: Optional[Path] = None
) -> Path:
    """Seeding-time gate for ``run_pipeline``'s ``external_artifacts``: absolute, confined to the
    allowed roots, existing, and of the on-disk shape ``kind`` declares (directory for
    ``dataset``/``model``, a file otherwise). Returns the resolved path.
    """
    resolved = _resolve_under_allowed_roots(raw_path, runs_root=runs_root)
    if kind in _DIRECTORY_KINDS:
        if not resolved.is_dir():
            raise ArtifactPathError(
                f"external artifact path {raw_path!r} (kind={kind!r}) must be an existing "
                f"directory; got {resolved}"
            )
    elif not resolved.is_file():
        raise ArtifactPathError(
            f"external artifact path {raw_path!r} (kind={kind!r}) must be an existing file; "
            f"got {resolved}"
        )
    return resolved


def resolve_servable_artifact_path(
    raw_path: str, *, runs_root: Optional[Path] = None
) -> Path:
    """Serving-time gate for the MCP layer: the same confinement as
    :func:`validate_external_artifact_path`, minus the existence/shape check (a legitimately
    recorded artifact may have been deleted since; the caller's own ``is_file()`` handles that).
    """
    return _resolve_under_allowed_roots(raw_path, runs_root=runs_root)
