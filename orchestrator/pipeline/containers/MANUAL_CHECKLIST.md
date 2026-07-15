# T08 manual verification checklist (run on Bartosz's Windows + Docker Desktop machine)

**Verified 2026-07-15:** all 6 automated checks below passed for real
(`PIPELINE_TEST_ISAAC=1 uv run -m pytest -q -s tests/test_containers_gpu.py`, 6 passed in 1088s).
Checkboxes below reflect that run.

GPU/Isaac behavior can't be exercised in the sandbox (no Docker daemon, no GPU). The unit tests in
`tests/test_containers.py` cover this module's own logic (mount/GPU-kwarg construction, warm-reuse
bookkeeping, exec exit-code handling) against a fake Docker client — this checklist is the
GPU-real counterpart, straight from `planning/tasks/T08-container-manager.md`'s acceptance
criteria.

**Prefer running `tests/test_containers_gpu.py` instead of typing these snippets by hand** — it's
an automated `pytest` file covering every step below (asserts pass/fail instead of "eyeball the
output"), and auto-skips harmlessly anywhere Docker isn't reachable (so it's safe to leave in the
normal suite):

```bash
cd orchestrator
pytest -q -s tests/test_containers_gpu.py                     # everything except Isaac
PIPELINE_TEST_ISAAC=1 pytest -q -s tests/test_containers_gpu.py  # + the Isaac pull/EULA/cache checks
```

The manual steps below are the same checks spelled out by hand, useful if you want to poke at
`ContainerManager` interactively (e.g. in a REPL) rather than run the test file, or if a test
fails and you want to reproduce the exact call that broke.

Run each step from a normal Windows terminal (PowerShell/cmd) with Docker Desktop + GPU support set
up (see `planning/WINDOWS_SETUP.md`), inside the `pipeline` package's venv (`orchestrator/.venv` or
wherever it's installed). No WSL2 shell needed — see `planning/WINDOWS_SETUP.md` for the full
first-time setup.

## 1. `ensure_image("cuda")` builds; GPU is visible inside it

```python
from pipeline.containers import ensure_image, exec_in_container

ensure_image("cuda")  # builds from the repo Dockerfile the first time; instant after that
result = exec_in_container("cuda", ["nvidia-smi"], log_path="nvidia-smi.log")
print(result.exit_code)  # expect 0
print(open("nvidia-smi.log").read())  # expect a normal nvidia-smi table, your GPU listed
```

- [x] `ensure_image("cuda")` returns without error the first time (image build) and instantly the
      second time (cached `images.get`).
- [x] `nvidia-smi` exits 0 and its logged output lists the GPU — confirms `--gpus all`
      passthrough is wired correctly.

## 2. `ensure_image("isaac")` pulls; EULA/consent vars let it start non-interactively

```python
ensure_image("isaac")
result = exec_in_container("isaac", ["/isaac-sim/python.sh", "-c", "print('isaac ok')"])
print(result.exit_code)  # expect 0, no EULA/interactive prompt hang
```

- [x] Pull succeeds (large image — expect several minutes the first time).
- [x] The Isaac Python entrypoint runs without hanging on an EULA prompt (confirms
      `ACCEPT_EULA`/`PRIVACY_CONSENT`/`OMNI_KIT_ACCEPT_EULA` are actually reaching the container).

## 3. Mounts resolve correctly

```python
exec_in_container("cuda", ["ls", "/workspace/orchestrator/pipeline/vendored"])
exec_in_container("cuda", ["ls", "/omniverse"])
```

- [x] `/workspace` is the repo root — `orchestrator/pipeline/vendored/{host,cuda,isaac}` is visible
      (this is *also* how the container gets the orchestrator's own copied-in stage logic, per
      `planning/ARCHITECTURE.md`'s "Vendored stage logic" — no separate mount for it).
- [x] `/omniverse` lists your actual Omniverse assets (whatever `PIPELINE_ASSETS_ROOT` points at,
      `Q:\Omniverse` by default).

## 4. Warm-container reuse

```python
from pipeline.containers import start_container
import time

id1 = start_container("cuda")
t0 = time.time()
id2 = start_container("cuda")  # should be instant -- no new container, no cold start
print(id1 == id2, time.time() - t0)
```

- [x] `id1 == id2` (same container reused, not recreated).
- [x] Second `start_container` call returns in well under a second (no image pull/container
      create/first-run warm-up repeated) — measured 0.028s.
- [x] `docker ps` (in a separate shell) shows exactly one `pipeline-cuda` container, not two.

## 5. Isaac cold-start cache persistence

```python
start_container("isaac")  # first run: slow shader/asset cache warm-up
# ... stop and remove the container ...
from pipeline.containers import stop_container, list_containers
cid = [c for c in list_containers() if c["env"] == "isaac"][0]["id"]
stop_container(cid, remove=True)
start_container("isaac")  # recreated container, but cache *volumes* persisted
```

- [x] The second `start_container("isaac")` (after removal) still runs noticeably faster than a
      true first-ever run — the `isaac-cache`/`isaac-compute`/`isaac-ovdata` volumes survived
      container removal (`docker volume ls` shows them) — measured 0.3s.

## 6. Clean teardown

```python
from pipeline.containers import list_containers, stop_container

for c in list_containers():
    stop_container(c["id"], remove=True)
```

- [x] `docker ps -a --filter label=pipeline.managed` shows nothing left behind.
