"""``ContainerManager``: Docker SDK over Docker Desktop, driven directly from Windows (T08).

Implements ``planning/tasks/T08-container-manager.md``: build/pull the ``cuda``/``isaac`` images
(:meth:`ContainerManager.ensure_image`), start a warm long-lived container per image with the
right mounts + GPU passthrough (:meth:`ContainerManager.start`), ``exec`` a command inside it with
log streaming into the run dir (:meth:`ContainerManager.exec`), and a clean
:meth:`ContainerManager.stop`/:meth:`ContainerManager.stop_by_id`.

**Revised 2026-07-14:** this talks to Docker Desktop's engine from a native Windows Python
process (``docker.from_env()`` finds it the same way the ``docker`` CLI does on Windows -- a named
pipe, no different in kind from how it'd find a Unix socket on Linux) rather than from inside a
WSL2 distro. Nothing in this module changed to make that true -- it never assumed a specific OS,
only ``pipeline.paths`` (T06) did, and that's been revised too. WSL2/Linux-distro-level bundling
is deferred future work, not something this module needs today (see ``planning/ARCHITECTURE.md``).

Per the package docstring (mirrors ``pipeline.resources`` not importing ``pynvml`` at module
scope): ``docker`` is only ever imported *inside* a method, never at module scope, so importing
this module -- and therefore ``pipeline`` as a whole -- stays safe with no ``docker`` package
installed and no Docker daemon reachable (``tests/test_import.py``'s
``test_no_heavy_imports_at_module_scope`` enforces this).

GPU/Isaac behavior can only be exercised for real on Bartosz's machine, with Docker Desktop + GPU
support set up (see ``planning/WINDOWS_SETUP.md`` and the task spec's acceptance checklist); this
module's own logic -- mount/GPU-kwarg construction, warm-container reuse bookkeeping, exec
exit-code handling, label-based listing -- is unit-tested against a fake Docker client in
``tests/test_containers.py`` so it doesn't need a real daemon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..paths import Env, get_roots
from . import config as _config

#: every container/image this module creates is labelled with this, so `list_containers`/the
#: warm-reuse lookup never mistakes an unrelated container for one of ours.
MANAGED_LABEL = "pipeline.managed"
ENV_LABEL = "pipeline.env"

#: Label storing a content hash of the files that determine what's inside the `cuda` image
#: (Dockerfile + pyproject.toml + uv.lock). Lets `ensure_image` distinguish "the tag exists" from
#: "the tag exists *and* matches what's on disk right now" -- see `ensure_image`'s docstring and
#: `_cuda_build_hash`'s docstring for the bug this exists to stop from recurring silently.
CUDA_BUILD_HASH_LABEL = "pipeline.cuda_build_hash"

#: Files whose content determines what ends up inside the `cuda` image. Anything that changes what
#: `uv sync` installs, or what the Dockerfile itself does, belongs here.
_CUDA_BUILD_HASH_INPUTS = ("Dockerfile", "pyproject.toml", "uv.lock")


def _cuda_build_hash(repo_root: Path) -> str:
    """Hash of the `cuda` image's build inputs, stored on the built image as
    :data:`CUDA_BUILD_HASH_LABEL` and recomputed on every :meth:`ContainerManager.ensure_image`
    call so a changed Dockerfile/`pyproject.toml`/`uv.lock` triggers a rebuild automatically.

    **Found on T11's second real-hardware run (2026-07-18):** the Dockerfile's `uv venv .venv &&
    uv sync --frozen` lines were commented out, so the built `cuda` image never had a working
    Python -- `train.default` failed with exit code 127. Fixing the Dockerfile alone wasn't
    enough: `ensure_image` only checked "does the tag exist," so the already-built, still-broken
    image kept being reused until a manual `docker rm -f pipeline-cuda` + `docker rmi
    4dgs-motion-amp-cuda:latest` forced a rebuild. This closes that gap -- see
    ``.claude_notes/NOTES_pipeline_orchestration.md``'s "cuda image never had a real Python" entry.
    """
    digest = hashlib.sha256()
    for name in _CUDA_BUILD_HASH_INPUTS:
        path = repo_root / name
        try:
            data = path.read_bytes()
        except OSError:
            data = b"<missing>"
        digest.update(data)
        digest.update(b"\x00")
    return digest.hexdigest()


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
    import docker  # local import -- see module docstring

    try:
        return docker.from_env()
    except Exception as exc:  # noqa: BLE001 - surface as our own error type, not docker's
        raise ContainerError(f"could not reach the Docker daemon: {exc}") from exc


def _to_docker_mounts(mounts: list[Any]) -> list[Any]:
    from docker.types import Mount  # local import -- see module docstring

    return [
        Mount(target=m.target, source=m.source, type=m.type, read_only=False) for m in mounts
    ]


def _device_requests(env: Env) -> list[Any]:
    from docker.types import DeviceRequest  # local import -- see module docstring

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
    id), so a fresh ``ContainerManager()`` -- e.g. after a process restart -- still finds and reuses
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
        """Make sure ``env``'s image is present locally *and* up to date, building (``cuda``) or
        pulling (``isaac``) it if not. Returns the image tag.

        For ``cuda`` specifically, "present" isn't enough -- :func:`_cuda_build_hash` is compared
        against the label the last successful build stored (:data:`CUDA_BUILD_HASH_LABEL`), so a
        Dockerfile/``pyproject.toml``/``uv.lock`` change since the last build triggers an automatic
        rebuild instead of silently reusing a stale image (see ``_cuda_build_hash``'s docstring).
        ``isaac`` stays on the simple presence check -- its image is pulled by pinned tag from NGC,
        never built locally, so there's no local "staleness" to detect.

        Idempotent in the common case -- once the image exists and matches the current build
        inputs, a later call is just one cheap ``images.get`` plus a hash comparison.
        """
        image = _config.IMAGES[env]
        if env == "cuda":
            repo_root = get_roots().repo_root_host
            if self._cuda_image_up_to_date(image, repo_root):
                return image
            try:
                _, build_log = self.client.images.build(
                    path=str(repo_root),
                    dockerfile=_config.CUDA_DOCKERFILE,
                    tag=image,
                    rm=True,
                    labels={CUDA_BUILD_HASH_LABEL: _cuda_build_hash(repo_root)},
                )
            except Exception as exc:  # noqa: BLE001 - our own error type, with context
                log_path = self._persist_cuda_build_log(repo_root, exc=exc)
                detail = f"{exc}" + (f" -- full build log: {log_path}" if log_path else "")
                raise ImageNotAvailableError(f"failed to build {image!r}: {detail}") from exc
            # Persisted on a *successful* build too, not just a failed one: a build that exits 0
            # doesn't guarantee the image actually works (see T11's 2026-07-18 real-hardware
            # finding -- the Dockerfile fixes so far have each addressed a build-time *failure*,
            # but nothing yet has confirmed the venv this produces is genuinely usable at exec
            # time). Cheap insurance against needing yet another slow rebuild just to see the log.
            self._persist_cuda_build_log(repo_root, chunks=build_log)
            return image

        if self._image_present(image):
            return image
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

    def _persist_cuda_build_log(
        self,
        repo_root: Path,
        *,
        exc: Optional[Exception] = None,
        chunks: Optional[Any] = None,
    ) -> Optional[Path]:
        """Best-effort: write a `cuda` image build's log to ``runs/.cache/cuda_build.log``, and
        return the path (or ``None`` if there was nothing to write / the write itself failed).

        Pass exactly one of ``exc`` (a failed build's exception -- real docker-py's
        ``docker.errors.BuildError``, raised on a failed ``RUN`` step, carries the *entire* build
        log on its ``.build_log`` attribute even though ``str(exc)`` alone is just the generic
        final-line message, e.g. ``"...returned a non-zero code: 1"``) or ``chunks`` (the log
        generator ``images.build()`` returns alongside the image on *success*). Both are the same
        shape: an iterable of ``{"stream": ...}``/``{"status": ...}``/``{"error": ...}`` dicts, the
        same the Docker CLI itself prints.

        **Found on T11's third real-hardware run (2026-07-18):** a rebuild triggered by the new
        build-hash staleness check (see :func:`_cuda_build_hash`) failed after a ~22-minute `uv
        sync` with nothing to go on but that one generic line -- this closed that gap for a failed
        build. Also called on a *successful* build (not just a failed one): a build exiting 0
        doesn't guarantee the resulting image actually works end to end (a `RUN` step can succeed
        while still producing something broken, e.g. a venv missing a `python` symlink) --this way
        that build's log is available too, without needing to force a rebuild just to see it.
        """
        if chunks is None:
            chunks = getattr(exc, "build_log", None)
        if chunks is None:
            return None
        log_path = repo_root / "runs" / ".cache" / "cuda_build.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8", errors="replace") as f:
                for chunk in chunks:
                    text = chunk.get("stream") or chunk.get("status") or chunk.get("error") or ""
                    if text:
                        f.write(text if text.endswith("\n") else text + "\n")
        except Exception:  # noqa: BLE001 - best-effort diagnostics, must never mask the real error
            return None
        return log_path

    def _cuda_image_up_to_date(self, image: str, repo_root: Path) -> bool:
        """``True`` only if ``image`` exists locally *and* its stored build-hash label matches
        :func:`_cuda_build_hash` right now. Any error (image missing, no such label, docker-py
        raising) is treated as "not up to date" -- rebuilding is always safe, just possibly
        redundant, whereas skipping a needed rebuild reintroduces the T11 stale-image bug.
        """
        try:
            existing = self.client.images.get(image)
            labels = existing.labels
        except AttributeError:
            # Fake/older docker-py image objects may expose labels via .attrs instead.
            try:
                labels = existing.attrs.get("Config", {}).get("Labels") or {}
            except Exception:
                return False
        except Exception:
            return False
        return labels.get(CUDA_BUILD_HASH_LABEL) == _cuda_build_hash(repo_root)

    # --- containers --------------------------------------------------------------------------

    def _find_container(self, env: Env) -> Optional[Any]:
        try:
            return self.client.containers.get(_config.container_name(env))
        except Exception:
            return None

    def start(self, env: Env) -> str:
        """Return a running container id for ``env``, starting or reusing a warm one.

        Reuse order: a container already running for this env is returned untouched (the "warm
        container" the task asks for -- no restart, no lost in-container state); one that exists
        but is stopped is started in place (its cache *volumes* already persisted anything worth
        keeping); otherwise a fresh one is created from :meth:`ensure_image`'s image with T06's +
        this env's mounts, GPU passthrough, and the keep-alive command.

        A container built from a since-rebuilt image (:meth:`ensure_image` just rebuilt ``cuda``
        because its build-hash label was stale, see :func:`_cuda_build_hash`) is *not* reused --
        the old container is still running the old image's filesystem underneath it, so warm-reuse
        would quietly resurrect the exact bug the rebuild was meant to fix. Such a container is
        stopped and removed here, then recreated fresh, same as if none had existed.
        """
        image = self.ensure_image(env)
        container = self._find_container(env)
        if container is not None:
            container.reload()
            if self._container_is_stale(container, image):
                self._recreate_stale_container(container)
                container = None
            else:
                if container.status != "running":
                    container.start()
                return container.id

        container = self.client.containers.run(
            image,
            _config.KEEP_ALIVE_CMD,
            name=_config.container_name(env),
            detach=True,
            mounts=_to_docker_mounts(_config.mounts_for(env)),
            device_requests=_device_requests(env),
            ipc_mode=_config.IPC_MODE[env],
            environment=_config.CONTAINER_ENV[env],
            labels={MANAGED_LABEL: "true", ENV_LABEL: env},
        )
        if env == "isaac":
            self._fixup_isaac_cache_permissions(container.id)
        return container.id

    def _container_is_stale(self, container: Any, image: str) -> bool:
        """``True`` if ``container`` was created from an image id other than ``image``'s *current*
        id -- i.e. the image tag was rebuilt (new id) since this container was created from it.

        Best-effort: any error (missing ``.image``, docker-py quirks on a fake test client, races
        with the daemon) is treated as "not stale," matching :meth:`_fixup_isaac_cache_permissions`'s
        stance that a lookup failure here must never block a normal ``start()``.
        """
        try:
            current_id = self.client.images.get(image).id
            return container.image.id != current_id
        except Exception:
            return False

    def _recreate_stale_container(self, container: Any) -> None:
        """Stop and remove a container found stale by :meth:`_container_is_stale`, clearing the
        way for :meth:`start` to create a fresh one from the now-current image.

        Unlike the lookup helpers above, failures here are *not* swallowed -- if ``remove()`` were
        silently skipped, the subsequent ``containers.run(..., name=...)`` would collide on the
        still-present container name and fail with a much more confusing error.
        """
        container.stop()
        container.remove()

    def _fixup_isaac_cache_permissions(self, container_id: str) -> None:
        """Best-effort ``chmod`` on the persisted ``isaac`` cache-volume mount points, right after
        this container was freshly created.

        **Found on T11's first real-hardware run (2026-07-16):** a brand-new named Docker volume
        gets its initial content/ownership copied from whatever the image already has at that path
        -- ``nvcr.io/nvidia/isaac-sim``'s own ``/isaac-sim/.cache`` ends up not writable by whatever
        UID ``exec_create`` runs commands as, causing a silent ``PermissionError`` deep inside Kit's
        startup (``omni.warp.core``'s kernel-cache init) that cascades into ``omni.replicator.core``
        failing to start -- and, because Kit still exits 0 afterward, a stage relying on that
        extension (``capture.isaac``) reports manifest "success" while quietly writing nothing (see
        ``.claude_notes/NOTES_pipeline_orchestration.md``). Chmod-ing the volume's actual on-disk
        permissions (not just this container's view) only needs to happen once per volume's
        lifetime, but re-running it on every fresh-container creation is cheap and avoids needing
        extra state to track "have I already fixed this volume." Never raises -- worst case here is
        the pre-existing cold-cache-every-time behavior, not a broken pipeline.
        """
        targets = [m.target for m in _config.CACHE_VOLUMES["isaac"]]
        if not targets:
            return
        api = self.client.api
        cmd = ["chmod", "-R", "0777", *targets]
        try:
            exec_id = api.exec_create(container_id, cmd, workdir=None, environment=None, user="root")[
                "Id"
            ]
            for _ in api.exec_start(exec_id, stream=True):
                pass
        except Exception:  # noqa: BLE001 - best-effort fixup, never fatal to start()
            pass

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

        Streams combined stdout/stderr into ``log_path`` as it arrives (append mode -- a stage may
        call ``exec`` more than once against the same log file) rather than buffering the whole
        thing in memory; a non-zero exit is reported via :attr:`ExecResult.exit_code`, never
        raised -- the caller (a stage's ``run(ctx)``, T09/T11) decides whether that means the stage
        failed. Uses the low-level ``client.api`` calls rather than the high-level
        ``container.exec_run`` wrapper because the high-level one only exposes an exit code once
        the *whole* stream has been consumed and demuxed; here we want both the streamed write and
        the exit code from the same call.

        ``environment`` (added T09): extra env vars for *this exec call only* (docker-py's
        ``exec_create`` supports this directly) -- e.g. ``pipeline.stages.cuda_common``'s
        ``PYTHONPATH=/workspace`` for a vendored ``pipeline/vendored/cuda/*.py`` script, whose own
        directory (not the repo root) is what Python would otherwise put on ``sys.path[0]``. Not
        the same thing as :data:`pipeline.containers.config.CONTAINER_ENV` (set once, at container
        *creation*, e.g. Isaac's EULA vars) -- this is scoped to one ``exec``, not persisted on the
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
        there isn't one -- callers don't need to check ``list_containers`` first just to tear down.
        """
        container = self._find_container(env)
        if container is None:
            return
        container.stop()
        if remove:
            container.remove()

    def stop_by_id(self, container_id: str, *, remove: bool = False) -> None:
        """Stop (and optionally remove) a managed container by id -- what
        ``pipeline.api.stop_container(container_id)`` (Layer 2/3-facing) uses, since callers there
        only ever saw an id from :meth:`list_containers`, not which ``env`` it was.
        """
        container = self.client.containers.get(container_id)
        container.stop()
        if remove:
            container.remove()

    def list_containers(self) -> list[ContainerInfo]:
        """Every container this manager created (labelled with :data:`MANAGED_LABEL`), running or
        not -- what ``pipeline.api.list_containers`` exposes to Layers 2/3.
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
