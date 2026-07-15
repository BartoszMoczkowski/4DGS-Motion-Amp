"""``ContainerManager``: Docker SDK over Docker Desktop, driven directly from Windows (T08).

Implements ``planning/tasks/T08-container-manager.md``: build/pull the ``cuda``/``isaac`` images
(:meth:`ContainerManager.ensure_image`), start a warm long-lived container per image with the
right mounts + GPU passthrough (:meth:`ContainerManager.start`), ``exec`` a command inside it with
log streaming into the run dir (:meth:`ContainerManager.exec`), and a clean
:meth:`ContainerManager.stop`/:meth:`ContainerManager.stop_by_id`.

**Revised 2026-07-14:** this talks to Docker Desktop's engine from a native Windows Python
process (``docker.from_env()`` finds it the same way the ``docker`` CLI does on Windows — a named
pipe, no different in kind from how it'd find a Unix socket on Linux) rather than from inside a
WSL2 distro. Nothing in this module changed to make that true — it never assumed a specific OS,
only ``pipeline.paths`` (T06) did, and that's been revised too. WSL2/Linux-distro-level bundling
is deferred future work, not something this module needs today (see ``planning/ARCHITECTURE.md``).

Per the package docstring (mirrors ``pipeline.resources`` not importing ``pynvml`` at module
scope): ``docker`` is only ever imported *inside* a method, never at module scope, so importing
this module — and therefore ``pipeline`` as a whole — stays safe with no ``docker`` package
installed and no Docker daemon reachable (``tests/test_import.py``'s
``test_no_heavy_imports_at_module_scope`` enforces this).

GPU/Isaac behavior can only be exercised for real on Bartosz's machine, with Docker Desktop + GPU
support set up (see ``planning/WINDOWS_SETUP.md`` and the task spec's acceptance checklist); this
module's own logic — mount/GPU-kwarg construction, warm-container reuse bookkeeping, exec
exit-code handling, label-based listing — is unit-tested against a fake Docker client in
``tests/test_containers.py`` so it doesn't need a real daemon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..paths import Env, get_roots
from . import config as _config

#: every container/image this module creates is labelled with this, so `list_containers`/the
#: warm-reuse lookup never mistakes an unrelated container for one of ours.
MANAGED_LABEL = "pipeline.managed"
ENV_LABEL = "pipeline.env"


class ContainerError(Exception):
    """Base class for every error this module raises."""


class ImageNotAvailableError(ContainerError):
    """:meth:`ContainerManager.ensure_image` couldn't build (``cuda``) or pull (``isaac``)."""


class ContainerNotRunningError(ContainerError):
    """An operation needed a running container for this env, and starting one failed."""


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one :meth:`ContainerManager.exec` call."""

    exit_code: int
    log_path: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class ContainerInfo:
    """One managed container, in the shape ``pipeline.api.list_containers`` returns."""

    id: str
    env: str
    image: str
    status: str
    name: str


def _docker_client() -> Any:
    import docker  # local import — see module docstring

    try:
        return docker.from_env()
    except Exception as exc:  # noqa: BLE001 - surface as our own error type, not docker's
        raise ContainerError(f"could not reach the Docker daemon: {exc}") from exc


def _to_docker_mounts(mounts: list[Any]) -> list[Any]:
    from docker.types import Mount  # local import — see module docstring

    return [
        Mount(target=m.target, source=m.source, type=m.type, read_only=False) for m in mounts
    ]


def _device_requests(env: Env) -> list[Any]:
    from docker.types import DeviceRequest  # local import — see module docstring

    if not _config.GPU_ALL[env]:
        return []
    # `count=-1, capabilities=[["gpu"]]` is docker-py's spelling of `--gpus all`.
    return [DeviceRequest(count=-1, capabilities=[["gpu"]])]


def _image_tag(container: Any) -> str:
    tags = getattr(container.image, "tags", None) or []
    return tags[0] if tags else container.image.id


class ContainerManager:
    """Owns the Docker client and all ``cuda``/``isaac`` container lifecycle for the pipeline.

    Docker Desktop itself is the source of truth for what's running (containers are found by the
    deterministic name from :func:`pipeline.containers.config.container_name`, not a remembered
    id), so a fresh ``ContainerManager()`` — e.g. after a process restart — still finds and reuses
    an already-warm container rather than starting a duplicate.

    ``client`` is injectable so tests can pass a fake Docker client without a real daemon; real
    callers leave it ``None`` and get a lazily-created ``docker.from_env()`` on first use.
    """

    def __init__(self, *, client: Any = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _docker_client()
        return self._client

    # --- images ----------------------------------------------------------------------------

    def ensure_image(self, env: Env) -> str:
        """Make sure ``env``'s image is present locally, building (``cuda``) or pulling
        (``isaac``) it if not. Returns the image tag. Idempotent — once the image exists, a
        later call is just one cheap ``images.get``.
        """
        image = _config.IMAGES[env]
        if self._image_present(image):
            return image

        if env == "cuda":
            repo_root = get_roots().repo_root_host
            try:
                self.client.images.build(
                    path=str(repo_root),
                    dockerfile=_config.CUDA_DOCKERFILE,
                    tag=image,
                    rm=True,
                )
            except Exception as exc:  # noqa: BLE001 - our own error type, with context
                raise ImageNotAvailableError(f"failed to build {image!r}: {exc}") from exc
        else:
            try:
                self.client.images.pull(image)
            except Exception as exc:  # noqa: BLE001
                raise ImageNotAvailableError(f"failed to pull {image!r}: {exc}") from exc
        return image

    def _image_present(self, image: str) -> bool:
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    # --- containers --------------------------------------------------------------------------

    def _find_container(self, env: Env) -> Optional[Any]:
        try:
            return self.client.containers.get(_config.container_name(env))
        except Exception:
            return None

    def start(self, env: Env) -> str:
        """Return a running container id for ``env``, starting or reusing a warm one.

        Reuse order: a container already running for this env is returned untouched (the "warm
        container" the task asks for — no restart, no lost in-container state); one that exists
        but is stopped is started in place (its cache *volumes* already persisted anything worth
        keeping); otherwise a fresh one is created from :meth:`ensure_image`'s image with T06's +
        this env's mounts, GPU passthrough, and the keep-alive command.
        """
        self.ensure_image(env)
        container = self._find_container(env)
        if container is not None:
            container.reload()
            if container.status != "running":
                container.start()
            return container.id

        container = self.client.containers.run(
            _config.IMAGES[env],
            _config.KEEP_ALIVE_CMD,
            name=_config.container_name(env),
            detach=True,
            mounts=_to_docker_mounts(_config.mounts_for(env)),
            device_requests=_device_requests(env),
            ipc_mode=_config.IPC_MODE[env],
            environment=_config.CONTAINER_ENV[env],
            labels={MANAGED_LABEL: "true", ENV_LABEL: env},
        )
        return container.id

    def exec(
        self,
        env: Env,
        cmd: list[str],
        *,
        log_path: Optional[Path] = None,
        workdir: Optional[str] = None,
        environment: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Run ``cmd`` inside ``env``'s (warm, started-if-needed) container.

        Streams combined stdout/stderr into ``log_path`` as it arrives (append mode — a stage may
        call ``exec`` more than once against the same log file) rather than buffering the whole
        thing in memory; a non-zero exit is reported via :attr:`ExecResult.exit_code`, never
        raised — the caller (a stage's ``run(ctx)``, T09/T11) decides whether that means the stage
        failed. Uses the low-level ``client.api`` calls rather than the high-level
        ``container.exec_run`` wrapper because the high-level one only exposes an exit code once
        the *whole* stream has been consumed and demuxed; here we want both the streamed write and
        the exit code from the same call.

        ``environment`` (added T09): extra env vars for *this exec call only* (docker-py's
        ``exec_create`` supports this directly) — e.g. ``pipeline.stages.cuda_common``'s
        ``PYTHONPATH=/workspace`` for a vendored ``pipeline/vendored/cuda/*.py`` script, whose own
        directory (not the repo root) is what Python would otherwise put on ``sys.path[0]``. Not
        the same thing as :data:`pipeline.containers.config.CONTAINER_ENV` (set once, at container
        *creation*, e.g. Isaac's EULA vars) — this is scoped to one ``exec``, not persisted on the
        container.
        """
        self.start(env)
        container = self._find_container(env)
        if container is None:
            raise ContainerNotRunningError(f"no running container for env={env!r} after start()")

        api = self.client.api
        exec_id = api.exec_create(container.id, cmd, workdir=workdir, environment=environment)["Id"]
        stream = api.exec_start(exec_id, stream=True)

        log_file = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("ab")
        try:
            for chunk in stream:
                if log_file is not None:
                    log_file.write(chunk)
        finally:
            if log_file is not None:
                log_file.close()

        exit_code = api.exec_inspect(exec_id)["ExitCode"]
        return ExecResult(exit_code=exit_code, log_path=str(log_path) if log_path else None)

    def stop(self, env: Env, *, remove: bool = False) -> None:
        """Stop (and optionally remove) ``env``'s managed container, if one exists. A no-op if
        there isn't one — callers don't need to check ``list_containers`` first just to tear down.
        """
        container = self._find_container(env)
        if container is None:
            return
        container.stop()
        if remove:
            container.remove()

    def stop_by_id(self, container_id: str, *, remove: bool = False) -> None:
        """Stop (and optionally remove) a managed container by id — what
        ``pipeline.api.stop_container(container_id)`` (Layer 2/3-facing) uses, since callers there
        only ever saw an id from :meth:`list_containers`, not which ``env`` it was.
        """
        container = self.client.containers.get(container_id)
        container.stop()
        if remove:
            container.remove()

    def list_containers(self) -> list[ContainerInfo]:
        """Every container this manager created (labelled with :data:`MANAGED_LABEL`), running or
        not — what ``pipeline.api.list_containers`` exposes to Layers 2/3.
        """
        containers = self.client.containers.list(all=True, filters={"label": MANAGED_LABEL})
        infos = []
        for c in containers:
            c.reload()
            infos.append(
                ContainerInfo(
                    id=c.id,
                    env=c.labels.get(ENV_LABEL, ""),
                    image=_image_tag(c),
                    status=c.status,
                    name=c.name,
                )
            )
        return infos
