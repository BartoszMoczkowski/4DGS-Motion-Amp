# T08 — Container manager

- Status: done (2026-07-14)
- Phase: 1
- Depends on: T05, T06
- Environment: host (drives Docker Desktop)

## Goal
Let stages run inside the `cuda` and `isaac` images with correct mounts + GPU passthrough — the
mechanism that gives Claude/automation access to the GPU-bound code (problem #2, mechanics).

## In scope
- Docker control, driven directly from Windows (docker SDK or CLI) against Docker Desktop's
  engine. (Revised 2026-07-14 — was "from WSL2"; see the revision note below.)
- `ensure_image(env)` (build `cuda` from repo `Dockerfile`; pull `isaac`); `start`, `exec`,
  `stop`, `logs` with mounts from T06 + `--gpus all`.
- Warm long-lived container per image (esp. Isaac) + persisted cache volumes to avoid cold-start.
- Log streaming into the run dir; non-zero exit surfaces as stage failure.
- Env abstraction so a stage just says `environment="cuda"` and the manager does the rest.

## Out of scope
The stage bodies (T09/T11); VRAM gating (T12).

## Deliverables
`pipeline/containers/` manager + config for images/volumes; a smoke test.

## Acceptance criteria (run on Bartosz's Windows + Docker Desktop machine — documented checklist)
- `ensure_image("cuda")` builds; a trivial `exec` (`nvidia-smi`) sees the GPU.
- Mounts resolve: container sees repo at `/workspace` and assets at `/omniverse`.
- Warm-container reuse verified (second exec skips startup); clean teardown.

## Relevant existing files
`Dockerfile`, `.devcontainer/devcontainer.json`, `omniverse_pipeline/.devcontainer/devcontainer.json`
+ `run_capture.sh` (image, mounts, `/isaac-sim/python.sh`, cache volumes).

## Notes / gotchas
Reuse the devcontainer definitions as the source of truth rather than re-specifying images/mounts.
Isaac's ENTRYPOINT is overridden to keep the container alive; replicate that. GPU stages can't be
verified in the sandbox — provide a manual checklist Bartosz runs.

## Done (2026-07-14)

`pipeline/containers/`: `config.py` (pure data — images, T06's `mounts_for(env)` + Isaac's
persisted cache volumes, GPU/`ipc_mode`/env-var settings, all mirrored 1:1 from the two existing
`devcontainer.json` files and `run_capture.sh`, per this task's own "reuse the devcontainer defs"
note) and `manager.py`'s `ContainerManager` (`ensure_image`/`start`/`exec`/`stop`/`stop_by_id`/
`list_containers`, Docker SDK only imported inside methods — same pattern as `pipeline.resources`
not importing `pynvml` at module scope, verified by `tests/test_import.py`).

`start` finds a container by the deterministic name `pipeline-<env>` rather than a remembered id,
so warm-container reuse survives a process restart: already-running -> returned as-is; exists but
stopped -> restarted in place; otherwise created fresh with `container_mounts` (T06) + GPU
`device_requests` + the `sleep infinity` keep-alive (replicating `overrideCommand: true`). `exec`
uses the low-level `client.api.exec_create/exec_start/exec_inspect` trio (not the high-level
`container.exec_run` wrapper) so it can stream output into a log file *and* get a real exit code
from the same call — a stage decides pass/fail from `ExecResult.exit_code`, `exec` itself never
raises on a non-zero exit. No separate mount for `pipeline/vendored/cuda|isaac`: it already lives
under the repo root, which T06 binds to `/workspace`, so the container sees it for free
(`planning/ARCHITECTURE.md`'s "Vendored stage logic").

`pipeline.api`'s `list_containers`/`start_container`/`stop_container` now delegate here (lazy
import, same pattern as T02/T03/T05's wiring); `cancel`/`gpu_status` stay stubs (T12 scope).
`exec` itself (`exec_in_container`) is *not* exposed via `pipeline.api` — it's stage-facing
(`ctx.containers`, for T09/T11), matching `ARCHITECTURE.md`'s "whitelisted ops only, no arbitrary
shell" rule for what Layer 2/3 can reach.

Verified: 20 new tests (126 total) against a fake Docker client covering mount/GPU-kwarg
construction, warm-reuse across all three states (new/running/stopped), exec streaming + exit
code + append-not-truncate, label-based `list_containers`, and a full
ensure_image->start->exec->reuse->stop lifecycle "smoke test" — green in two independently
rebuilt isolated venvs (fresh `pip install -e .` each time, `docker`/`pynvml` install fine, no
GPU/daemon needed for these). Actual GPU/Isaac behavior can't run in this sandbox — see
`pipeline/containers/MANUAL_CHECKLIST.md` for the 6-step checklist Bartosz runs on his WSL2 +
Docker Desktop machine (covers every acceptance-criteria line: cuda build + `nvidia-smi`, isaac
pull + non-interactive EULA accept, mount resolution, warm-reuse timing, Isaac cache-volume
persistence across container removal, clean teardown).

Hit the known sandbox mount-staleness bug repeatedly and badly this time — worth flagging in
`[[cowork-mount-staleness-bug]]`: bash's view of `pipeline/containers/__init__.py` and `api.py`
came back truncated mid string-literal after `Edit` (the usual pattern), fixed by rewriting the
bash-visible copy from a heredoc. Separately (not the same bug — a self-inflicted `cat file >
file` shell redirection, which classically truncates the destination before the read completes)
`tests/test_containers.py`'s tail got wiped down to one truncated line while trying to "refresh" a
stale bash-side read; re-created via `Read` (authoritative) + a fresh heredoc write, not a
self-redirect, to recover it.

## GPU/Isaac verified for real (2026-07-15)

Bartosz ran `uv run -m pytest -q -s tests/test_containers_gpu.py` (with `PIPELINE_TEST_ISAAC=1`,
since both Isaac-gated tests ran, not skipped) on his real Windows + Docker Desktop + GPU machine.
**All 6 passed in 1088s (~18 min, dominated by the Isaac image pull the first time):** cuda image
build + `nvidia-smi` GPU passthrough, Isaac image pull + non-interactive EULA/consent accept,
mount resolution (`/workspace`, `/omniverse`), warm-container reuse (second `start_container`
0.028s vs. a cold start), Isaac cache-volume persistence across container removal (0.3s restart),
clean teardown. This closes out every acceptance-criteria line and every box in
`pipeline/containers/MANUAL_CHECKLIST.md` — T08 is now verified end-to-end on real hardware, not
just against the fake-client unit tests. No code changes needed; the runtime-host revision above
held up under real GPU/Docker conditions.

## Revised (2026-07-14) — runtime host moved off WSL2

Bartosz asked to run/test the orchestrator directly from Windows for now (WSL2/Docker "bundling"
deferred to a later, unscheduled `T16`). This module needed exactly one line changed:
`ensure_image`'s cuda build-context path source, `get_roots().repo_root_wsl` → `.repo_root_host`
(T06's field rename — see `T06-path-translation.md`'s own revision note). Everything else about
`ContainerManager` was already OS-agnostic — it never hardcoded WSL2, only assumed whatever
`pipeline.paths` handed it, and `docker.from_env()` finds Docker Desktop's engine the same way
from a native Windows process as from a WSL2 shell (named pipe vs. Unix socket, same idea).
Docstrings/comments referencing "WSL2 machine" reworded to "Bartosz's machine, with Docker
Desktop + GPU support set up" throughout (`manager.py`, `__init__.py`, `tests/test_containers.py`,
`tests/test_containers_gpu.py`, `MANUAL_CHECKLIST.md`). Full test suite re-verified green (120
passed, 6 skipped) after the change. See `planning/WINDOWS_SETUP.md` (replaces the now-retired
`WSL_SETUP.md`) for the first-time machine setup this task's acceptance checklist assumes.
