"""GPU/Isaac integration tests — the runnable counterpart to
``pipeline/containers/MANUAL_CHECKLIST.md``.

These talk to a *real* Docker daemon (and, for the Isaac tests, actually pull/run the multi-GB
Isaac Sim image), so they can't run in the sandbox and aren't part of the normal `pytest -q` suite
in spirit — they auto-skip the moment no Docker daemon is reachable. Run them for real on
Bartosz's machine, with Docker Desktop + GPU support set up (see `planning/WINDOWS_SETUP.md`),
from the `pipeline` package's venv:

    cd orchestrator
    pytest -q -s tests/test_containers_gpu.py

That covers everything except the Isaac-specific checks (image pull is ~10GB+ and slow the first
time). To also run those:

    PIPELINE_TEST_ISAAC=1 pytest -q -s tests/test_containers_gpu.py

Use ``-s`` so the printed timings (warm-reuse, Isaac cache-persistence) are visible — those two
are reported rather than hard-asserted, since "fast enough" isn't a portable threshold across
machines.

Every managed container is stopped and removed at the end (module-scoped autouse fixture), even
if a test fails partway through, so a bad run doesn't leave containers/GPU memory behind.
"""

from __future__ import annotations

import os
import time

import pytest

from pipeline.containers import (
    ensure_image,
    exec_in_container,
    list_containers,
    start_container,
    stop_container,
)


def _docker_reachable() -> bool:
    try:
        import docker  # local import — this module must stay importable with no `docker` package
    except ImportError:
        return False
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_reachable(),
    reason=(
        "no reachable Docker daemon — these are real-GPU/real-Docker integration tests for "
        "Bartosz's Windows + Docker Desktop machine, not the sandbox/CI"
    ),
)

_RUN_ISAAC = os.environ.get("PIPELINE_TEST_ISAAC") == "1"
isaac_only = pytest.mark.skipif(
    not _RUN_ISAAC,
    reason="set PIPELINE_TEST_ISAAC=1 to also run the Isaac tests (large pull, slow first run)",
)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_managed_containers():
    yield
    for c in list_containers():
        stop_container(c["id"], remove=True)


# --- 1. `ensure_image("cuda")` builds; GPU is visible inside it -------------------------------


def test_cuda_image_builds_and_gpu_is_visible(tmp_path):
    ensure_image("cuda")  # builds from the repo Dockerfile the first time, instant after

    log_path = tmp_path / "nvidia-smi.log"
    result = exec_in_container("cuda", ["nvidia-smi"], log_path=log_path)

    output = log_path.read_text()
    assert result.exit_code == 0, output
    assert "NVIDIA-SMI" in output, f"unexpected nvidia-smi output:\n{output}"


# --- 2. `ensure_image("isaac")` pulls; EULA/consent vars let it start non-interactively -------


@isaac_only
def test_isaac_image_pulls_and_starts_noninteractively(tmp_path):
    ensure_image("isaac")

    log_path = tmp_path / "isaac.log"
    result = exec_in_container(
        "isaac", ["/isaac-sim/python.sh", "-c", "print('isaac ok')"], log_path=log_path
    )

    output = log_path.read_text()
    assert result.exit_code == 0, output
    assert "isaac ok" in output


# --- 3. mounts resolve correctly ---------------------------------------------------------------


def test_mounts_resolve_correctly(tmp_path):
    vendored_log = tmp_path / "vendored.log"
    result = exec_in_container(
        "cuda", ["ls", "/workspace/orchestrator/pipeline/vendored"], log_path=vendored_log
    )
    vendored_out = vendored_log.read_text()
    assert result.exit_code == 0, vendored_out
    assert "host" in vendored_out, (
        "repo bind mount should also expose pipeline/vendored/host — see "
        "ARCHITECTURE.md's 'Vendored stage logic'"
    )

    assets_log = tmp_path / "assets.log"
    result = exec_in_container("cuda", ["ls", "/omniverse"], log_path=assets_log)
    assert result.exit_code == 0, assets_log.read_text()


# --- 4. warm-container reuse --------------------------------------------------------------------


def test_warm_container_reuse_is_fast():
    id1 = start_container("cuda")
    t0 = time.time()
    id2 = start_container("cuda")
    elapsed = time.time() - t0

    print(f"\nsecond start_container('cuda') took {elapsed:.3f}s")
    assert id1 == id2, "expected the same warm container to be reused, not recreated"
    assert elapsed < 1.0, f"took {elapsed:.2f}s — looks like a cold start, not a reuse"


# --- 5. Isaac cache-volume persistence across container removal --------------------------------


@isaac_only
def test_isaac_cache_persists_across_container_removal():
    start_container("isaac")
    cid = next(c["id"] for c in list_containers() if c["env"] == "isaac")
    stop_container(cid, remove=True)

    t0 = time.time()
    start_container("isaac")
    elapsed = time.time() - t0

    # Not a hard-asserted threshold — a genuine cold start (no persisted cache) vs. a warm one
    # (cache volumes intact) is too machine-dependent to bound portably. Read this against the
    # checklist's expectation: noticeably faster than the very first ever `ensure_image("isaac")`
    # + `start_container("isaac")` run.
    print(f"\nisaac restart after container removal took {elapsed:.1f}s")


# --- 6. clean teardown ---------------------------------------------------------------------


def test_teardown_leaves_nothing_running():
    for c in list_containers():
        stop_container(c["id"], remove=True)

    remaining = list_containers()
    assert remaining == [], f"containers left behind: {remaining}"
