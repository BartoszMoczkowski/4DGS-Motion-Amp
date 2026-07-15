"""Path translation: the *only* module that maps host <-> container paths.

Per ``planning/INSTRUCTIONS.md``: "Path translation lives in exactly one module (T06). Never
hardcode ``Q:\\`` / ``/omniverse`` / ``/workspace`` anywhere else." Every other module (config,
artifacts, stages, the container manager in T08) asks *this* module for a path in the space it
needs; none of them parse a drive letter or build a container path themselves.

Two path spaces, two canonical roots
-------------------------------------
**Revised 2026-07-14** (see ``.claude_notes/NOTES_pipeline_orchestration.md``'s "Runtime host
moved off WSL2" entry): the pipeline's execution host is now **native Windows** — the Python
process running the DAG/CPU stages and driving Docker Desktop runs directly on Windows, not from
inside a WSL2 Linux venv. Docker Desktop is reachable from Windows directly (it always was; WSL2
was never a requirement for *talking to* Docker Desktop, only an assumption about where our own
code happened to run), so the three-space (host/wsl/container) model T06 originally shipped is
now two spaces:

- **host** — wherever this process's own filesystem actually is. On the real target (Bartosz's
  Windows machine) that's a Windows drive-letter path (``C:\\Users\\...``, ``Q:\\Omniverse\\...``)
  — also exactly the form Docker Desktop's bind-mount ``source=`` wants (see
  ``.devcontainer/devcontainer.json``). Nothing here hardcodes "Windows" as a type, though:
  ``Roots`` just uses ``pathlib.Path`` (native, OS-dispatching), so if this ever *does* run from a
  Linux/WSL2 shell again, "host" simply becomes whatever a Linux path looks like there — no code
  change needed, it was only ever a matter of which OS the interpreter happens to run on.
- **container** — the path as seen from inside the ``cuda``/``isaac`` Docker containers, which
  bind-mount fixed locations: the repo at ``/workspace``, the Omniverse assets at ``/omniverse``
  (see ``.devcontainer/devcontainer.json`` and ``omniverse_pipeline/.devcontainer/devcontainer.json``).
  Always POSIX, regardless of host OS — that's just what's inside the Linux container images.

**Deferred, not gone:** running this from inside a WSL2 distro (or eventually bundling a whole
WSL2 + Docker setup as a packaged feature) is explicitly future work, not something this module
needs to support today — see ``planning/ARCHITECTURE.md``'s phasing notes. If that ever comes
back, it re-enters as a third space the same way ``container`` already works, not as a redesign.

Every path the pipeline cares about falls under exactly one of two roots — **repo** or
**assets** — each of which has a host/container form. ``Roots`` holds those two forms for both;
:func:`get_roots` builds a default set with no hardcoded drive letter for the repo root:

- The repo root's host form is derived from ``__file__`` (this module already lives inside the
  repo, wherever it's actually checked out) — no hardcoded drive letter or username.
- The assets root has no such anchor (it's an external asset library, e.g. ``Q:\\Omniverse``) —
  its default matches the existing ``omniverse_pipeline/capture_config_pump.yaml`` convention and
  is overridable via the ``PIPELINE_ASSETS_ROOT`` env var (``PIPELINE_REPO_ROOT`` for the repo
  root).

Both are read at *call* time (:func:`get_roots`), not import time, so importing this module never
touches the filesystem or a env var snapshot — mirrors ``artifacts/paths.get_runs_root``.

Public API
----------
``to_host(path)`` / ``to_container(path, env)`` each accept a path in *either* space (auto-
detected against the known roots) and return it in the target space. This makes every conversion
a round trip: ``to_host(to_container(x)) == x`` for any ``x`` already under a known root, in
either starting space.

``env`` on :func:`to_container` (and :func:`container_mounts`) selects which container
(``"cuda"`` or ``"isaac"``) the path is destined for. Both currently mount the repo/assets roots
identically (see the two ``devcontainer.json`` files) so it has no effect on the mapping today —
it exists so a future container with different mounts doesn't need an API change, and it still
validates its value so a typo fails fast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Optional, Union

PathLike = Union[str, Path]
Env = Literal["cuda", "isaac"]

#: orchestrator/pipeline/paths.py -> repo root (4DGS-Motion-Amp/), in whatever form this process's
#: own filesystem uses (native ``pathlib.Path`` -- a Windows path on the real target machine).
_THIS_FILE_HOST = Path(__file__).resolve().parents[2]

_REPO_ROOT_ENV_VAR = "PIPELINE_REPO_ROOT"
_ASSETS_ROOT_ENV_VAR = "PIPELINE_ASSETS_ROOT"

#: Sensible defaults. Repo root comes from where this file actually is; the assets root has no
#: such anchor so it falls back to the path already used in
#: ``omniverse_pipeline/capture_config_pump.yaml`` (``Q:/Omniverse``).
_DEFAULT_REPO_ROOT_HOST = _THIS_FILE_HOST
_DEFAULT_ASSETS_ROOT_HOST = Path("Q:/Omniverse")

#: Fixed container-side mount targets (see both ``.devcontainer/devcontainer.json`` files).
REPO_ROOT_CONTAINER = PurePosixPath("/workspace")
ASSETS_ROOT_CONTAINER = PurePosixPath("/omniverse")


# --- roots --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Roots:
    """The repo/assets root mappings every path in the pipeline is translated relative to."""

    repo_root_host: Path
    assets_root_host: Path
    repo_root_container: PurePosixPath = REPO_ROOT_CONTAINER
    assets_root_container: PurePosixPath = ASSETS_ROOT_CONTAINER


def get_roots() -> Roots:
    """Roots resolved at call time: env-var overrides if set, else the sensible defaults.

    ``PIPELINE_REPO_ROOT`` / ``PIPELINE_ASSETS_ROOT`` override the host form of either root (e.g.
    for a differently-laid-out machine, or a test that wants a host form the sandbox actually
    has). Read at call time, not import time, so importing this module is always safe and tests
    can flip them via ``monkeypatch.setenv`` without reloading the module.
    """
    repo = os.environ.get(_REPO_ROOT_ENV_VAR)
    assets = os.environ.get(_ASSETS_ROOT_ENV_VAR)
    return Roots(
        repo_root_host=Path(repo) if repo else _DEFAULT_REPO_ROOT_HOST,
        assets_root_host=Path(assets) if assets else _DEFAULT_ASSETS_ROOT_HOST,
    )


@dataclass(frozen=True)
class _RootSpec:
    host: Path
    container: PurePosixPath


def _root_specs(roots: Roots) -> dict[str, _RootSpec]:
    return {
        "repo": _RootSpec(roots.repo_root_host, roots.repo_root_container),
        "assets": _RootSpec(roots.assets_root_host, roots.assets_root_container),
    }


# --- matching: which root (repo/assets) and space does a path belong to? -----------------------


def _strip_host_prefix(raw: str, base: Path) -> Optional[str]:
    """If ``raw`` (any casing/separator) is ``base`` or a descendant, return the ``/``-joined
    remainder (``""`` for an exact match), else ``None``. Backslash-tolerant and case-insensitive
    (matches Windows path semantics — the real target machine) even when a test's override
    happens to look like a POSIX path; that's harmless since it only ever makes matching more
    permissive, never wrong for an exact-cased path.
    """
    raw_n = raw.replace("\\", "/")
    base_n = str(base).replace("\\", "/").rstrip("/")
    if raw_n.rstrip("/").lower() == base_n.lower():
        return ""
    prefix = base_n + "/"
    if raw_n[: len(prefix)].lower() == prefix.lower():
        return raw_n[len(prefix) :]
    return None


def _strip_posix_prefix(raw: str, base: PurePosixPath) -> Optional[str]:
    """Same as :func:`_strip_host_prefix` for the (always-POSIX) container base, case-sensitive —
    real Linux paths inside the container images.
    """
    raw_n = raw.replace("\\", "/")
    base_n = str(base).rstrip("/")
    if raw_n.rstrip("/") == base_n:
        return ""
    prefix = base_n + "/"
    if raw_n.startswith(prefix):
        return raw_n[len(prefix) :]
    return None


def _match(path: PathLike, roots: Roots) -> tuple[str, str]:
    """Find which root (``"repo"`` or ``"assets"``) ``path`` is under, in whichever of the two
    spaces it's expressed in, and the ``/``-joined remainder relative to that root.
    """
    raw = str(path)
    for name, spec in _root_specs(roots).items():
        for rel in (
            _strip_host_prefix(raw, spec.host),
            _strip_posix_prefix(raw, spec.container),
        ):
            if rel is not None:
                return name, rel
    raise ValueError(
        f"path {raw!r} is not under a known root (repo={roots.repo_root_host!s}, "
        f"assets={roots.assets_root_host!s}) in host or container form"
    )


def _validate_env(env: Optional[Env]) -> None:
    if env is not None and env not in ("cuda", "isaac"):
        raise ValueError(f"unknown environment {env!r}; expected 'cuda', 'isaac', or None")


# --- public API -----------------------------------------------------------------------------


def to_host(path: PathLike, *, roots: Optional[Roots] = None) -> Path:
    """Convert a path in either space (host/container) to its host-path form."""
    roots = roots or get_roots()
    name, rel = _match(path, roots)
    base = _root_specs(roots)[name].host
    return base / rel if rel else base


def to_container(
    path: PathLike, env: Optional[Env] = None, *, roots: Optional[Roots] = None
) -> PurePosixPath:
    """Convert a path in either space (host/container) to its in-container path form.

    ``env`` (``"cuda"`` / ``"isaac"`` / ``None``) is validated but currently doesn't change the
    mapping — see the module docstring.
    """
    _validate_env(env)
    roots = roots or get_roots()
    name, rel = _match(path, roots)
    base = _root_specs(roots)[name].container
    return base / rel if rel else base


# --- mount-spec builders (consumed by the container manager, T08) ------------------------------


@dataclass(frozen=True)
class MountSpec:
    """One bind-mount, in the shape Docker / devcontainer ``mounts`` entries use.

    ``source`` is the host path in the ``C:/foo/bar`` (forward-slash) form used by the existing
    ``.devcontainer/devcontainer.json`` files, e.g. ``"source=Q:/Omniverse,target=/omniverse,...".
    """

    source: str
    target: str
    type: Literal["bind", "volume"] = "bind"
    consistency: Optional[str] = "cached"

    def as_docker_mount_string(self) -> str:
        """Render as a ``docker run --mount`` / devcontainer ``mounts[]`` entry string."""
        parts = [f"source={self.source}", f"target={self.target}", f"type={self.type}"]
        if self.type == "bind" and self.consistency:
            parts.append(f"consistency={self.consistency}")
        return ",".join(parts)


def _mount_source(host_path: Path) -> str:
    # Forward-slash form, matching "source=Q:/Omniverse,..." in the existing devcontainer.json.
    return str(host_path).replace("\\", "/")


def container_mounts(env: Env, *, roots: Optional[Roots] = None) -> list[MountSpec]:
    """The repo + assets bind mounts every ``cuda``/``isaac`` container needs.

    Mirrors ``.devcontainer/devcontainer.json`` and
    ``omniverse_pipeline/.devcontainer/devcontainer.json``: repo -> ``/workspace``, Omniverse
    assets -> ``/omniverse``. Cache/auth *volume* mounts (Isaac shader cache, Claude Code config,
    etc.) are the container manager's own concern (T08), not a path-translation one, and aren't
    built here.
    """
    _validate_env(env)
    roots = roots or get_roots()
    return [
        MountSpec(
            source=_mount_source(roots.repo_root_host), target=str(roots.repo_root_container)
        ),
        MountSpec(
            source=_mount_source(roots.assets_root_host),
            target=str(roots.assets_root_container),
        ),
    ]
