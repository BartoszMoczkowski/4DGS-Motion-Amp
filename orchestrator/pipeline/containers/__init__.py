"""Container manager: Docker SDK/CLI over Docker Desktop, driven directly from Windows.

T08 (``planning/tasks/T08-container-manager.md``): lets stages run inside the ``cuda``/``isaac``
images with the correct mounts + GPU passthrough — the mechanism that gives Claude/automation
access to the GPU-bound code (problem #2's mechanics, ``planning/pipeline-orchestration-plan.md``).
Config (images/mounts/env vars, mirrored from the existing devcontainer defs) lives in
``config.py``; the Docker-talking logic lives in ``manager.py``'s ``ContainerManager``.

Free functions here (``ensure_image``/``start_container``/``exec_in_container``/
``stop_container``/``list_containers``) mirror the public-function style ``pipeline.config``/
``pipeline.artifacts`` already use: a lazily-created module-level :class:`ContainerManager`
singleton backs them so callers (``pipeline.api``, a stage's ``ctx.containers``) don't need to
construct one themselves, while still being able to inject their own (tests use this to pass a
fake Docker client) via the ``manager=`` kwarg.

Must not import the ``docker`` SDK at module scope so the package stays importable without it
installed / without a reachable daemon (``ContainerManager``/``manager.py`` only import it inside
methods) — this is what ``tests/test_import.py``'s ``test_no_heavy_imports_at_module_scope``
checks for the package as a whole.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from ..paths import Env
from .config import (
    CACHE_VOLUMES,
    CONTAINER_ENV,
    CUDA_DOCKERFILE,
    CUDA_IMAGE,
    GPU_ALL,
    IMAGES,
    IPC_MODE,
    ISAAC_IMAGE,
    KEEP_ALIVE_CMD,
    container_name,
    mounts_for,
)
from .manager import (
    ContainerError,
    ContainerInfo,
    ContainerManager,
    ContainerNotRunningError,
    ExecResult,
    ImageNotAvailableError,
)

__all__ = [
    "Env",
    "ContainerManager",
    "ContainerError",
    "ImageNotAvailableError",
    "ContainerNotRunningError",
    "ExecResult",
    "ContainerInfo",
    "IMAGES",
    "CUDA_IMAGE",
    "ISAAC_IMAGE",
    "CUDA_DOCKERFILE",
    "GPU_ALL",
    "IPC_MODE",
    "CONTAINER_ENV",
    "KEEP_ALIVE_CMD",
    "CACHE_VOLUMES",
    "container_name",
    "mounts_for",
    "ensure_image",
    "start_container",
    "exec_in_container",
    "stop_container",
    "list_containers",
]

_manager: Optional[ContainerManager] = None


def _get_manager(manager: Optional[ContainerManager] = None) -> ContainerManager:
    global _manager
    if manager is not None:
        return manager
    if _manager is None:
        _manager = ContainerManager()
    return _manager


def ensure_image(env: Env, *, manager: Optional[ContainerManager] = None) -> str:
    """Build (``cuda``) or pull (``isaac``) ``env``'s image if it isn't present locally."""
    return _get_manager(manager).ensure_image(env)


def start_container(env: Env, *, manager: Optional[ContainerManager] = None) -> str:
    """Return a running container id for ``env``, reusing a warm one if there is one."""
    return _get_manager(manager).start(env)


def exec_in_container(
    env: Env,
    cmd: list[str],
    *,
    log_path: Optional[str] = None,
    workdir: Optional[str] = None,
    environment: Optional[dict[str, str]] = None,
    manager: Optional[ContainerManager] = None,
) -> ExecResult:
    """Run ``cmd`` inside ``env``'s container, streaming output into ``log_path``.

    Stage-facing (T09/T11's ``ctx.containers``), not part of ``pipeline.api``'s public surface —
    the MCP server only ever exposes whitelisted ops (``planning/ARCHITECTURE.md``), never
    arbitrary exec. ``environment`` (T09) sets extra env vars for this one exec call — see
    :meth:`pipeline.containers.manager.ContainerManager.exec`'s docstring.
    """
    return _get_manager(manager).exec(env, cmd, log_path=log_path, workdir=workdir, environment=environment)


def stop_container(
    container_id: str, *, remove: bool = False, manager: Optional[ContainerManager] = None
) -> None:
    """Stop (and optionally remove) a managed container by id (``pipeline.api.stop_container``)."""
    _get_manager(manager).stop_by_id(container_id, remove=remove)


def list_containers(*, manager: Optional[ContainerManager] = None) -> list[dict[str, Any]]:
    """List managed containers as plain dicts (``pipeline.api.list_containers``)."""
    return [dataclasses.asdict(c) for c in _get_manager(manager).list_containers()]
