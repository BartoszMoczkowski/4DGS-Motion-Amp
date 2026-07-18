"""T08 -- container manager, tested against a fake Docker client (no real daemon needed).

Everything GPU/Isaac-specific can only be verified for real on Bartosz's machine, with Docker
Desktop + GPU support set up (``planning/tasks/T08-container-manager.md``'s acceptance checklist,
mirrored in ``pipeline/containers/MANUAL_CHECKLIST.md``). What *is* verifiable here in the sandbox
is this module's own logic: the mount/GPU-kwarg construction, warm-container reuse bookkeeping, exec
exit-code/log-streaming handling, and label-based listing -- exactly the parts that would be wrong
in a way a human skimming the code might not catch, and the parts ``planning/INSTRUCTIONS.md``'s
"every task ends with verification" rule requires for a CPU-only piece.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.containers import config as cfg
from pipeline.containers.manager import (
    CUDA_BUILD_HASH_LABEL,
    ContainerManager,
    ContainerNotRunningError,
    ExecResult,
    ImageNotAvailableError,
    _cuda_build_hash,
)
from pipeline.paths import MountSpec, get_roots


# --- fake Docker SDK -----------------------------------------------------------------------


class _FakeImageRef:
    def __init__(self, tag: str, *, id_: str | None = None, labels: dict | None = None) -> None:
        self.tags = [tag]
        self.id = id_ or f"sha256:{tag}"
        self.labels = labels or {}


class _AutoPresentSet(set):
    """Backs ``_FakeImages.present`` -- ``.add(tag)`` is the shortcut most pre-existing tests use
    to mean "this image already exists and is fine, don't build/pull it," bypassing ``build()``/
    ``pull()`` entirely. Once ``ensure_image`` started comparing a build-hash *label* (T11's
    stale-cuda-image fix) rather than just checking presence, that shortcut needed to also stamp a
    matching label -- otherwise every pre-existing test using it would look "stale" under the new
    check purely as a side effect of not going through the labelled build path, which isn't what
    those tests are about. Tests that specifically want to exercise staleness detection still set
    ``fake_client.images.labels[...]`` directly after this, which overrides the auto-stamp.
    """

    def __init__(self, images: "_FakeImages") -> None:
        super().__init__()
        self._images = images

    def add(self, tag: str) -> None:  # type: ignore[override]
        super().add(tag)
        self._images.ids.setdefault(tag, f"sha256:{tag}")
        if tag == cfg.CUDA_IMAGE:
            self._images.labels.setdefault(
                tag, {CUDA_BUILD_HASH_LABEL: _cuda_build_hash(get_roots().repo_root_host)}
            )


class _FakeBuildError(Exception):
    """Stands in for docker-py's real `docker.errors.BuildError` -- same shape that matters here:
    `str(exc)` is just the short reason, and the full per-line output lives on `.build_log`."""

    def __init__(self, reason: str, build_log: list[dict]) -> None:
        super().__init__(reason)
        self.build_log = build_log


class _FakeImages:
    def __init__(self) -> None:
        self.present: set[str] = _AutoPresentSet(self)
        self.build_calls: list[dict] = []
        self.pull_calls: list[str] = []
        self.labels: dict[str, dict] = {}
        self.ids: dict[str, str] = {}
        self._build_counter = 0
        self._next_build_failure: _FakeBuildError | None = None

    def fail_next_build(self, reason: str, build_log: list[dict]) -> None:
        """Make the next `build()` call raise `_FakeBuildError(reason, build_log)` instead of
        succeeding -- simulates a real `uv sync`/Dockerfile `RUN` step failing partway through."""
        self._next_build_failure = _FakeBuildError(reason, build_log)

    def get(self, tag: str) -> _FakeImageRef:
        if tag not in self.present:
            raise LookupError(f"no such image: {tag}")
        return _FakeImageRef(tag, id_=self.ids.get(tag), labels=self.labels.get(tag))

    def build(self, *, path: str, dockerfile: str, tag: str, rm: bool = True, labels: dict | None = None):
        self.build_calls.append(
            {"path": path, "dockerfile": dockerfile, "tag": tag, "rm": rm, "labels": labels or {}}
        )
        if self._next_build_failure is not None:
            failure, self._next_build_failure = self._next_build_failure, None
            raise failure
        self.present.add(tag)
        self._build_counter += 1
        self.ids[tag] = f"sha256:{tag}-build{self._build_counter}"
        self.labels[tag] = dict(labels or {})
        return _FakeImageRef(tag, id_=self.ids[tag], labels=self.labels[tag]), iter([])

    def pull(self, tag: str):
        self.pull_calls.append(tag)
        self.present.add(tag)
        self.ids.setdefault(tag, f"sha256:{tag}")
        return _FakeImageRef(tag, id_=self.ids.get(tag), labels=self.labels.get(tag))


class _FakeContainer:
    def __init__(
        self,
        id_: str,
        name: str,
        image: str,
        labels: dict,
        status: str = "running",
        *,
        image_id: str | None = None,
    ) -> None:
        self.id = id_
        self.name = name
        self.image = _FakeImageRef(image, id_=image_id)
        self.labels = labels
        self.status = status
        self.stop_called = False
        self.start_called = False
        self.remove_called = False

    def reload(self) -> None:
        pass

    def start(self) -> None:
        self.start_called = True
        self.status = "running"

    def stop(self) -> None:
        self.stop_called = True
        self.status = "exited"

    def remove(self) -> None:
        self.remove_called = True


class _FakeContainers:
    def __init__(self, images: _FakeImages) -> None:
        self._images = images
        self._by_name: dict[str, _FakeContainer] = {}
        self._by_id: dict[str, _FakeContainer] = {}
        self.run_calls: list[dict] = []
        self._next_id = 0

    def seed(self, name: str, image: str, *, status: str = "running", labels: dict | None = None) -> _FakeContainer:
        # Reflect whatever image id is currently on record for `image` (set by `build()`/`pull()`,
        # or by the `present.add()` shortcut) -- this is what lets `_container_is_stale` tell a
        # container created from an old build apart from one matching the current image.
        image_id = self._images.ids.get(image)
        c = _FakeContainer(f"cid-{name}", name, image, labels or {}, status=status, image_id=image_id)
        self._by_name[name] = c
        self._by_id[c.id] = c
        return c

    def get(self, name: str) -> _FakeContainer:
        if name not in self._by_name and name not in self._by_id:
            raise LookupError(f"no such container: {name}")
        return self._by_name.get(name) or self._by_id[name]

    def run(self, image, cmd, *, name, detach, mounts, device_requests, ipc_mode, environment, labels):
        self.run_calls.append(
            {
                "image": image,
                "cmd": cmd,
                "name": name,
                "detach": detach,
                "mounts": mounts,
                "device_requests": device_requests,
                "ipc_mode": ipc_mode,
                "environment": environment,
                "labels": labels,
            }
        )
        c = self.seed(name, image, status="running", labels=labels)
        return c

    def list(self, *, all: bool, filters: dict):
        label = filters["label"]
        return [c for c in self._by_name.values() if label in c.labels]


class _FakeExecAPI:
    """Fake for ``client.api`` -- the low-level exec_create/exec_start/exec_inspect trio."""

    def __init__(self) -> None:
        self._next = 0
        self._chunks: dict[str, list[bytes]] = {}
        self._exit_codes: dict[str, int] = {}
        self.create_calls: list[dict] = []

    def script(self, chunks: list[bytes], exit_code: int) -> None:
        """Queue what the *next* exec_create/exec_start/exec_inspect trio should return."""
        self._pending_chunks = chunks
        self._pending_exit = exit_code

    def exec_create(self, container_id: str, cmd, workdir=None, environment=None, user=None):
        self._next += 1
        exec_id = f"exec-{self._next}"
        self.create_calls.append(
            {
                "container_id": container_id,
                "cmd": cmd,
                "workdir": workdir,
                "environment": environment,
                "user": user,
            }
        )
        chunks = getattr(self, "_pending_chunks", [])
        exit_code = getattr(self, "_pending_exit", 0)
        self._chunks[exec_id] = chunks
        self._exit_codes[exec_id] = exit_code
        return {"Id": exec_id}

    def exec_start(self, exec_id: str, stream: bool = True):
        return iter(self._chunks[exec_id])

    def exec_inspect(self, exec_id: str):
        return {"ExitCode": self._exit_codes[exec_id]}


class _FakeClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers(self.images)
        self.api = _FakeExecAPI()


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def manager(fake_client: _FakeClient) -> ContainerManager:
    return ContainerManager(client=fake_client)


@pytest.fixture(autouse=True)
def _clean_cuda_build_log():
    """`ensure_image("cuda")` now persists a build log to the *real* repo's `runs/.cache/
    cuda_build.log` on every build (success or failure, T11 2026-07-18) -- these tests build
    against the real `get_roots().repo_root_host` (same pattern the existing `path == str(
    get_roots().repo_root_host)` assertions already rely on), so without this they'd leave a stray
    log file behind in the actual working tree after every test run."""
    log_path = get_roots().repo_root_host / "runs" / ".cache" / "cuda_build.log"
    log_path.unlink(missing_ok=True)
    yield
    log_path.unlink(missing_ok=True)


# --- config.py: pure, no Docker involved --------------------------------------------------


def test_container_name_deterministic():
    assert cfg.container_name("cuda") == "pipeline-cuda"
    assert cfg.container_name("isaac") == "pipeline-isaac"
    assert cfg.container_name("cuda") != cfg.container_name("isaac")


def test_mounts_for_cuda_is_just_repo_and_assets():
    mounts = cfg.mounts_for("cuda")
    assert [m.target for m in mounts] == ["/workspace", "/omniverse"]
    assert all(m.type == "bind" for m in mounts)


def test_mounts_for_isaac_adds_persisted_cache_volumes_after_binds():
    mounts = cfg.mounts_for("isaac")
    assert [m.target for m in mounts] == [
        "/workspace",
        "/omniverse",
        "/isaac-sim/.cache",
        "/isaac-sim/.nv/ComputeCache",
        "/isaac-sim/.local/share/ov/data",
    ]
    binds, volumes = mounts[:2], mounts[2:]
    assert all(m.type == "bind" for m in binds)
    assert all(m.type == "volume" for m in volumes)


def test_isaac_container_env_has_eula_and_privacy_consent():
    assert cfg.CONTAINER_ENV["isaac"] == {
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    assert cfg.CONTAINER_ENV["cuda"] == {}


def test_cuda_gets_ipc_host_isaac_does_not():
    assert cfg.IPC_MODE["cuda"] == "host"
    assert cfg.IPC_MODE["isaac"] is None


# --- ensure_image ---------------------------------------------------------------------------


def test_ensure_image_builds_cuda_when_missing(manager, fake_client):
    tag = manager.ensure_image("cuda")

    assert tag == cfg.CUDA_IMAGE
    assert len(fake_client.images.build_calls) == 1
    call = fake_client.images.build_calls[0]
    assert call["tag"] == cfg.CUDA_IMAGE
    assert call["dockerfile"] == "Dockerfile"
    assert call["path"] == str(get_roots().repo_root_host)
    assert fake_client.images.pull_calls == []


def test_ensure_image_pulls_isaac_when_missing(manager, fake_client):
    tag = manager.ensure_image("isaac")

    assert tag == cfg.ISAAC_IMAGE
    assert fake_client.images.pull_calls == [cfg.ISAAC_IMAGE]
    assert fake_client.images.build_calls == []


def test_ensure_image_is_idempotent(manager, fake_client):
    manager.ensure_image("cuda")
    manager.ensure_image("cuda")

    assert len(fake_client.images.build_calls) == 1  # second call was just a cheap `.get`


# --- ensure_image / start: stale-image detection (T11 fix, 2026-07-18) ----------------------
#
# Found on T11's second real-hardware run: the Dockerfile's `uv sync` lines were commented out, so
# the built `cuda` image never had a working Python. Fixing the Dockerfile wasn't enough on its
# own -- `ensure_image` only checked "does the tag exist," so the already-built, still-broken
# image kept being reused until a manual `docker rm`/`rmi`. These tests cover the fix: a build-hash
# label on the image (`_cuda_build_hash`) that triggers an automatic rebuild when it no longer
# matches, and `start()` recreating (rather than reusing) a container whose image has since moved.


def test_ensure_image_rebuilds_cuda_when_build_hash_stale(manager, fake_client):
    manager.ensure_image("cuda")
    assert len(fake_client.images.build_calls) == 1

    # Simulate a Dockerfile/pyproject.toml/uv.lock edit landing after the image was built -- the
    # stored label no longer matches what `_cuda_build_hash` computes from what's on disk now.
    fake_client.images.labels[cfg.CUDA_IMAGE] = {CUDA_BUILD_HASH_LABEL: "stale-hash"}

    manager.ensure_image("cuda")

    assert len(fake_client.images.build_calls) == 2  # rebuilt automatically, no manual docker rm/rmi


def test_ensure_image_stamps_the_current_build_hash_on_a_fresh_build(manager, fake_client):
    manager.ensure_image("cuda")

    call = fake_client.images.build_calls[0]
    expected = _cuda_build_hash(get_roots().repo_root_host)
    assert call["labels"] == {CUDA_BUILD_HASH_LABEL: expected}


def test_ensure_image_isaac_has_no_staleness_check(manager, fake_client):
    """`isaac` is pulled by pinned tag from NGC, never built locally -- there's no local build
    hash to compare, so it stays on the simple "does the tag exist" check."""
    manager.ensure_image("isaac")
    manager.ensure_image("isaac")

    assert len(fake_client.images.pull_calls) == 1


# --- ensure_image: build-failure diagnostics (T11 fix, 2026-07-18) --------------------------
#
# Found on T11's third real-hardware run: a rebuild triggered by the new staleness check above
# failed after a ~22-minute `uv sync` with nothing to go on but docker-py's generic
# `"...returned a non-zero code: 1"` -- no indication what actually failed inside the build. These
# tests cover `_persist_cuda_build_log`, which pulls the full log off `docker.errors.BuildError`'s
# `.build_log` attribute (real docker-py always populates this on a failed `RUN` step) and writes
# it to `runs/.cache/cuda_build.log` so a failure is diagnosable without re-running the build.


def test_ensure_image_persists_full_build_log_on_failure(manager, fake_client):
    fake_client.images.fail_next_build(
        "The command '/bin/sh -c uv venv .venv &&     uv sync --frozen' returned a non-zero code: 1",
        build_log=[
            {"stream": "Step 5/8 : RUN uv venv .venv \\&\\& uv sync --frozen\n"},
            {"stream": "Resolved 42 packages\n"},
            {"stream": "error: failed to build `diff-gaussian-rasterization`\n"},
            {"error": "The command '/bin/sh -c uv venv .venv &&     uv sync --frozen' returned a non-zero code: 1"},
        ],
    )

    with pytest.raises(ImageNotAvailableError) as excinfo:
        manager.ensure_image("cuda")

    log_path = get_roots().repo_root_host / "runs" / ".cache" / "cuda_build.log"
    assert "full build log" in str(excinfo.value)
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "failed to build `diff-gaussian-rasterization`" in content
    log_path.unlink()  # don't leak into the real repo's runs/.cache across test runs


def test_ensure_image_failure_without_build_log_still_raises_cleanly(manager, fake_client):
    """A failure with no `.build_log` (e.g. a connection error, not a failed `RUN` step) must not
    crash the diagnostics helper itself -- `_persist_cuda_build_log` returns `None` and the
    original exception still propagates as `ImageNotAvailableError`."""
    fake_client.images.fail_next_build("could not connect to Docker daemon", build_log=None)

    with pytest.raises(ImageNotAvailableError) as excinfo:
        manager.ensure_image("cuda")

    assert "full build log" not in str(excinfo.value)


def test_start_recreates_a_container_built_from_a_stale_cuda_image(manager, fake_client):
    manager.ensure_image("cuda")
    first_id = manager.start("cuda")
    existing = fake_client.containers.get("pipeline-cuda")
    assert len(fake_client.containers.run_calls) == 1

    # A Dockerfile fix lands and gets picked up on the next ensure_image -- the running container
    # is still backed by the old (stale) image underneath it.
    fake_client.images.labels[cfg.CUDA_IMAGE] = {CUDA_BUILD_HASH_LABEL: "stale-hash"}

    second_id = manager.start("cuda")

    assert existing.stop_called is True
    assert existing.remove_called is True
    assert len(fake_client.containers.run_calls) == 2  # old container replaced, not reused
    # (the fake mints container ids from the deterministic name, so `second_id == first_id` here
    # is a fake-only artifact -- what matters, and what's asserted above, is that a *new*
    # `containers.run()` call happened and the stale one was stopped/removed first.)
    assert second_id == first_id


def test_start_reuses_a_container_when_the_image_has_not_changed(manager, fake_client):
    manager.ensure_image("cuda")
    first_id = manager.start("cuda")

    second_id = manager.start("cuda")

    assert second_id == first_id
    assert len(fake_client.containers.run_calls) == 1  # no spurious recreation


# --- start: warm-container reuse -------------------------------------------------------------


def test_start_creates_new_container_with_mounts_and_gpu(manager, fake_client):
    container_id = manager.start("cuda")

    assert len(fake_client.containers.run_calls) == 1
    call = fake_client.containers.run_calls[0]
    assert call["image"] == cfg.CUDA_IMAGE
    assert call["cmd"] == cfg.KEEP_ALIVE_CMD
    assert call["name"] == "pipeline-cuda"
    assert call["ipc_mode"] == "host"
    assert len(call["device_requests"]) == 1  # `--gpus all` equivalent
    assert [m["Target"] for m in call["mounts"]] == ["/workspace", "/omniverse"]
    assert call["labels"] == {"pipeline.managed": "true", "pipeline.env": "cuda"}
    assert container_id == "cid-pipeline-cuda"


def test_start_reuses_already_running_container(manager, fake_client):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)

    container_id = manager.start("cuda")

    assert fake_client.containers.run_calls == []  # no duplicate container created
    assert container_id == "cid-pipeline-cuda"


def test_start_restarts_a_stopped_container_instead_of_recreating(manager, fake_client):
    existing = fake_client.containers.seed(
        "pipeline-isaac", cfg.ISAAC_IMAGE, status="exited", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.ISAAC_IMAGE)

    container_id = manager.start("isaac")

    assert existing.start_called is True
    assert fake_client.containers.run_calls == []
    assert container_id == existing.id


def test_start_fixes_up_isaac_cache_permissions_only_on_fresh_creation(manager, fake_client):
    """T11 real-hardware finding (2026-07-16): a brand-new ``isaac-cache`` volume isn't writable by
    whatever UID ``exec`` runs as -- ``start()`` chmods the cache-volume mount points right after
    creating a fresh ``isaac`` container, but must not do this (or anything else) for ``cuda``, and
    must not repeat it for an already-running/just-restarted container (see
    :meth:`ContainerManager._fixup_isaac_cache_permissions`'s own docstring)."""
    manager.start("isaac")

    chmod_calls = [c for c in fake_client.api.create_calls if c["cmd"][0] == "chmod"]
    assert len(chmod_calls) == 1
    assert chmod_calls[0]["user"] == "root"
    assert chmod_calls[0]["cmd"][1:3] == ["-R", "0777"]
    assert set(chmod_calls[0]["cmd"][3:]) == {
        "/isaac-sim/.cache",
        "/isaac-sim/.nv/ComputeCache",
        "/isaac-sim/.local/share/ov/data",
    }


def test_start_never_chmods_for_cuda(manager, fake_client):
    manager.start("cuda")

    assert [c for c in fake_client.api.create_calls if c["cmd"][0] == "chmod"] == []


def test_start_does_not_rechmod_an_already_running_isaac_container(manager, fake_client):
    fake_client.containers.seed(
        "pipeline-isaac", cfg.ISAAC_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.ISAAC_IMAGE)

    manager.start("isaac")

    assert [c for c in fake_client.api.create_calls if c["cmd"][0] == "chmod"] == []


# --- exec ------------------------------------------------------------------------------------


def test_exec_streams_output_to_log_and_returns_exit_code(manager, fake_client, tmp_path):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)
    fake_client.api.script([b"line one\n", b"line two\n"], exit_code=0)

    log_path = tmp_path / "logs" / "train.log"
    result = manager.exec("cuda", ["python3", "train.py"], log_path=log_path, workdir="/workspace")

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.ok is True
    assert log_path.read_bytes() == b"line one\nline two\n"
    assert fake_client.api.create_calls[0]["cmd"] == ["python3", "train.py"]
    assert fake_client.api.create_calls[0]["workdir"] == "/workspace"


def test_exec_reports_nonzero_exit_without_raising(manager, fake_client, tmp_path):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)
    fake_client.api.script([b"boom\n"], exit_code=1)

    result = manager.exec("cuda", ["false"], log_path=tmp_path / "log.txt")

    assert result.exit_code == 1
    assert result.ok is False


def test_exec_passes_extra_environment_through_to_exec_create(manager, fake_client, tmp_path):
    """T09: `pipeline.stages.cuda_common.run_cuda_script` sets `PYTHONPATH=/workspace` for every
    vendored `pipeline/vendored/cuda/*.py` call -- this is the plumbing that carries it down to
    docker-py's `exec_create`."""
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)
    fake_client.api.script([b""], exit_code=0)

    manager.exec(
        "cuda",
        ["python", "train.py"],
        log_path=tmp_path / "log.txt",
        environment={"PYTHONPATH": "/workspace"},
    )

    assert fake_client.api.create_calls[0]["environment"] == {"PYTHONPATH": "/workspace"}


def test_exec_environment_defaults_to_none(manager, fake_client, tmp_path):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)
    fake_client.api.script([b""], exit_code=0)

    manager.exec("cuda", ["true"], log_path=tmp_path / "log.txt")

    assert fake_client.api.create_calls[0]["environment"] is None


def test_exec_starts_a_container_if_none_is_running_yet(manager, fake_client, tmp_path):
    fake_client.api.script([b"ok\n"], exit_code=0)

    result = manager.exec("isaac", ["echo", "hi"], log_path=tmp_path / "log.txt")

    assert len(fake_client.containers.run_calls) == 1  # start() created it on demand
    assert result.exit_code == 0


def test_exec_appends_across_calls_rather_than_truncating(manager, fake_client, tmp_path):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )
    fake_client.images.present.add(cfg.CUDA_IMAGE)
    log_path = tmp_path / "log.txt"

    fake_client.api.script([b"first\n"], exit_code=0)
    manager.exec("cuda", ["echo", "first"], log_path=log_path)
    fake_client.api.script([b"second\n"], exit_code=0)
    manager.exec("cuda", ["echo", "second"], log_path=log_path)

    assert log_path.read_bytes() == b"first\nsecond\n"


# --- stop / list -------------------------------------------------------------------------------


def test_stop_is_a_noop_when_nothing_is_running(manager):
    manager.stop("cuda")  # must not raise


def test_stop_stops_and_optionally_removes(manager, fake_client):
    existing = fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )

    manager.stop("cuda", remove=True)

    assert existing.stop_called is True
    assert existing.remove_called is True


def test_stop_by_id_looks_up_directly(manager, fake_client):
    existing = fake_client.containers.seed(
        "pipeline-isaac", cfg.ISAAC_IMAGE, status="running", labels={"pipeline.managed": "true"}
    )

    manager.stop_by_id(existing.id)

    assert existing.stop_called is True
    assert existing.remove_called is False


def test_list_containers_reports_only_labelled_ones(manager, fake_client):
    fake_client.containers.seed(
        "pipeline-cuda", cfg.CUDA_IMAGE, status="running", labels={"pipeline.managed": "true", "pipeline.env": "cuda"}
    )
    fake_client.containers.seed("someone-elses-container", "nginx:latest", status="running", labels={})

    infos = manager.list_containers()

    assert [i.name for i in infos] == ["pipeline-cuda"]
    assert infos[0].env == "cuda"
    assert infos[0].image == cfg.CUDA_IMAGE
    assert infos[0].status == "running"


# --- full lifecycle "smoke test" (fake-backed CPU analog of the manual GPU checklist) --------


def test_full_lifecycle_smoke(manager, fake_client, tmp_path):
    """ensure_image -> start -> exec (twice, warm reuse) -> stop, all in one go."""

    manager.ensure_image("cuda")
    first_id = manager.start("cuda")

    fake_client.api.script([b"nvidia-smi output\n"], exit_code=0)
    r1 = manager.exec("cuda", ["nvidia-smi"], log_path=tmp_path / "smoke.log")
    assert r1.ok

    second_id = manager.start("cuda")
    assert second_id == first_id  # warm reuse, no second container created
    assert len(fake_client.containers.run_calls) == 1

    manager.stop("cuda", remove=True)
