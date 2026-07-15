"""T08 — container manager, tested against a fake Docker client (no real daemon needed).

Everything GPU/Isaac-specific can only be verified for real on Bartosz's machine, with Docker
Desktop + GPU support set up (``planning/tasks/T08-container-manager.md``'s acceptance checklist,
mirrored in ``pipeline/containers/MANUAL_CHECKLIST.md``). What *is* verifiable here in the sandbox
is this module's own logic: the mount/GPU-kwarg construction, warm-container reuse bookkeeping, exec
exit-code/log-streaming handling, and label-based listing — exactly the parts that would be wrong
in a way a human skimming the code might not catch, and the parts ``planning/INSTRUCTIONS.md``'s
"every task ends with verification" rule requires for a CPU-only piece.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.containers import config as cfg
from pipeline.containers.manager import (
    ContainerManager,
    ContainerNotRunningError,
    ExecResult,
)
from pipeline.paths import MountSpec, get_roots


# --- fake Docker SDK -----------------------------------------------------------------------


class _FakeImageRef:
    def __init__(self, tag: str) -> None:
        self.tags = [tag]
        self.id = f"sha256:{tag}"


class _FakeImages:
    def __init__(self) -> None:
        self.present: set[str] = set()
        self.build_calls: list[dict] = []
        self.pull_calls: list[str] = []

    def get(self, tag: str) -> _FakeImageRef:
        if tag not in self.present:
            raise LookupError(f"no such image: {tag}")
        return _FakeImageRef(tag)

    def build(self, *, path: str, dockerfile: str, tag: str, rm: bool = True):
        self.build_calls.append({"path": path, "dockerfile": dockerfile, "tag": tag, "rm": rm})
        self.present.add(tag)
        return _FakeImageRef(tag), iter([])

    def pull(self, tag: str):
        self.pull_calls.append(tag)
        self.present.add(tag)
        return _FakeImageRef(tag)


class _FakeContainer:
    def __init__(self, id_: str, name: str, image: str, labels: dict, status: str = "running") -> None:
        self.id = id_
        self.name = name
        self.image = _FakeImageRef(image)
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
    def __init__(self) -> None:
        self._by_name: dict[str, _FakeContainer] = {}
        self._by_id: dict[str, _FakeContainer] = {}
        self.run_calls: list[dict] = []
        self._next_id = 0

    def seed(self, name: str, image: str, *, status: str = "running", labels: dict | None = None) -> _FakeContainer:
        c = _FakeContainer(f"cid-{name}", name, image, labels or {}, status=status)
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
    """Fake for ``client.api`` — the low-level exec_create/exec_start/exec_inspect trio."""

    def __init__(self) -> None:
        self._next = 0
        self._chunks: dict[str, list[bytes]] = {}
        self._exit_codes: dict[str, int] = {}
        self.create_calls: list[dict] = []

    def script(self, chunks: list[bytes], exit_code: int) -> None:
        """Queue what the *next* exec_create/exec_start/exec_inspect trio should return."""
        self._pending_chunks = chunks
        self._pending_exit = exit_code

    def exec_create(self, container_id: str, cmd, workdir=None, environment=None):
        self._next += 1
        exec_id = f"exec-{self._next}"
        self.create_calls.append(
            {"container_id": container_id, "cmd": cmd, "workdir": workdir, "environment": environment}
        )
        self._chunks[exec_id] = self._pending_chunks
        self._exit_codes[exec_id] = self._pending_exit
        return {"Id": exec_id}

    def exec_start(self, exec_id: str, stream: bool = True):
        return iter(self._chunks[exec_id])

    def exec_inspect(self, exec_id: str):
        return {"ExitCode": self._exit_codes[exec_id]}


class _FakeClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.api = _FakeExecAPI()


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def manager(fake_client: _FakeClient) -> ContainerManager:
    return ContainerManager(client=fake_client)


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
    vendored `pipeline/vendored/cuda/*.py` call — this is the plumbing that carries it down to
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
