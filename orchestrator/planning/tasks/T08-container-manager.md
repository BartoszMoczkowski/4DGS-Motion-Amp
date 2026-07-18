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

## Revised (2026-07-18) — stale cuda-image auto-detection (T11's second real-hardware bug)

Found while re-testing T11 on real hardware: the repo `Dockerfile`'s `uv venv .venv && uv sync
--frozen` lines had been commented out, so the built `cuda` image never had a working Python —
`train.default` failed with exit code 127 ("python" not found). Fixing the Dockerfile wasn't
enough on its own: `ensure_image` only ever checked "does the tag exist," so the already-built,
still-broken image kept being reused silently, requiring a manual `docker rm -f pipeline-cuda` +
`docker rmi 4dgs-motion-amp-cuda:latest` to force a rebuild — the same "reuse-by-design defeats a
one-time fixup" shape as the isaac cache-permission bug and the cache-poisoning bug (see
`.claude_notes/NOTES_pipeline_orchestration.md`).

`manager.py` now closes this gap without a manual step: `_cuda_build_hash(repo_root)` hashes
`Dockerfile` + `pyproject.toml` + `uv.lock`; `ensure_image("cuda")` stamps that hash as a
`pipeline.cuda_build_hash` Docker label on every build and compares it against the *current* hash
on every call, rebuilding automatically on a mismatch instead of just checking presence. `isaac`
is untouched — it's pulled by pinned tag from NGC, never built locally, so there's nothing local
to go stale. `start()` also now checks whether an existing named container's image id still
matches the current image id for that env (`_container_is_stale`); if `ensure_image` just rebuilt
`cuda`, the old warm container (still running the old image's filesystem) is stopped + removed
and recreated fresh, rather than reused out from under the rebuild.

Verified: 5 new tests in `tests/test_containers.py` (160 total, 9 skipped) — rebuild-on-stale-hash,
label stamped correctly on a fresh build, isaac's simple presence check untouched, a stale
container's replacement on the next `start()`, and no spurious rebuild/recreation when nothing
changed (extending `_FakeImages`/`_FakeContainer` to track per-build image ids and labels, with a
`present.add(tag)` shortcut that auto-stamps a matching label so pre-existing tests using it don't
spuriously look stale). Not yet re-verified against a real Docker daemon on Bartosz's machine —
the actual `docker rm`/`rmi` workaround this replaces was only ever done manually there once.

## Revised (2026-07-18, same day) — persist the full `cuda` build log on failure

The staleness fix above worked exactly as designed on the next real-hardware run — it triggered an
automatic rebuild with no manual `docker rm`/`rmi` needed — but the rebuild itself failed after a
~22-minute `uv sync --frozen`, and all `ensure_image` had to report was docker-py's generic
`"...returned a non-zero code: 1"`. No indication what inside `uv sync` actually failed, and no log
to inspect without re-running the (slow) build again.

Added `_persist_cuda_build_log(repo_root, exc)`: real docker-py raises `docker.errors.BuildError`
on a failed `RUN` step, which carries the *entire* build output on `.build_log` (an iterable of
`{"stream": ...}`/`{"error": ...}` chunks, same shape the Docker CLI itself prints) even though
`str(exc)` only surfaces the short final-line reason. `ensure_image`'s except-branch now writes
that log to `runs/.cache/cuda_build.log` and appends the path to the raised
`ImageNotAvailableError`'s message. Best-effort — a failure with no `.build_log` (e.g. a daemon
connection error, not a failed build step) just returns `None` and the original exception still
propagates untouched.

Verified: 2 new tests (162 total) — a fake `docker.errors.BuildError` stand-in with a `.build_log`
list confirms the log file gets written and referenced in the exception message; a
`build_log=None` case confirms the diagnostics helper degrades cleanly without masking the real
error. Root cause of the actual real-hardware `uv sync` failure is still unknown — this makes the
*next* occurrence diagnosable, it doesn't retroactively explain today's. Recommended to Bartosz: run
`docker build -f Dockerfile -t 4dgs-motion-amp-cuda:latest .` manually to see today's live error
without waiting through another 22-minute automated rebuild; flagged the base image
(`nvidia/cuda:12.4.1-devel`) vs. the pinned `pytorch-cu126` wheel index as one thing worth checking
in that output, since `diff-gaussian-rasterization`/`simple-knn` compile CUDA extensions against
whatever toolkit/torch combination ends up installed.

## Revised (2026-07-18, later same day) -- build log now persisted on success too

That hypothesis was wrong (confirmed via the real build log: an unrelated `torch.utils.
cpp_extension` arch-detection crash, fixed by adding `TORCH_CUDA_ARCH_LIST` to the Dockerfile --
see `.claude_notes/NOTES_pipeline_orchestration.md`). After that fix, the exact same "python not
found" error came back, this time surfacing after the build (not as a build failure), meaning
either the rebuilt image's container never got recreated (a bug in this file's own
`_container_is_stale` staleness check) or the build still silently doesn't produce a working venv.
`_persist_cuda_build_log` (added earlier today for failures) now also runs on a successful build,
so if this happens again there's a log to inspect either way, not just on an outright build
failure. No test changes needed (fakes already return an empty log generator on success); added an
autouse cleanup fixture to `tests/test_containers.py` so this doesn't leave a stray
`runs/.cache/cuda_build.log` in the real working tree from test runs. 162 tests, all green.

**Root-caused later the same day:** Bartosz's diagnostic commands ruled out a staleness-detection
bug in this file (`docker images`'s id and `docker inspect pipeline-cuda --format "{{.Image}}"`
matched exactly, confirming the same-day `_container_is_stale` fix worked correctly) but confirmed
`/workspace/.venv/bin/` genuinely didn't exist inside the running container. Root cause: this
file's own `mounts_for("cuda")` bind-mounts the entire live host repo over `/workspace` at
container runtime, which is exactly where the Dockerfile had been building the venv — a bind mount
replaces the underlying image layer, it doesn't merge with it, so the build's own output was
structurally unreachable the moment the container started, independent of whether the build itself
succeeded. Fixed in the Dockerfile (moved the whole build to `/opt/build`, outside the bind-mounted
path) — no change needed here in `manager.py`/`config.py`, since the live `/workspace` mount is
correct and load-bearing for `pipeline/vendored/cuda/*.py` visibility; the bug was in what else got
built into that same shadowed path. Full detail:
`.claude_notes/NOTES_pipeline_orchestration.md`'s "Root cause found and fixed (2026-07-18, later
same day)" entry.

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
