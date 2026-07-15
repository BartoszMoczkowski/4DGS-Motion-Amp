# T06 — Path-translation module

- Status: done
- Phase: 0
- Depends on: T01
- Environment: host

## Goal
One module that owns every host ↔ WSL2 ↔ container path mapping, so it never scatters again
(a root cause of problem #3).

## In scope
- Canonical mappings: repo root (host `C:\...\4DGS-Motion-Amp` / WSL2 `/mnt/...` / container
  `/workspace`), assets (`Q:\Omniverse` / `/mnt/q/Omniverse` / container `/omniverse`), run/output
  dirs, uploads.
- API: `to_container(path, env)`, `to_wsl(path)`, `to_host(path)`, plus mount-spec builders the
  container manager (T08) consumes.
- Config-driven roots (no hardcoded drive letters); sensible WSL2 defaults.

## Out of scope
Actually mounting anything (T08).

## Deliverables
`pipeline/paths.py` with pure functions + a small config block for the roots.

## Acceptance criteria
- Table-driven unit tests cover every mapping both directions (incl. `Q:\` ↔ `/mnt/q` ↔
  `/omniverse` and repo ↔ `/workspace`).
- Round-trip `to_container(to_host(x)) == x` for representative paths.

## Relevant existing files
`.devcontainer/devcontainer.json` and `omniverse_pipeline/.devcontainer/devcontainer.json` (the
mount specs to encode); `capture_config_pump.yaml` (uses `Q:/Omniverse/...`).

## Notes / gotchas
Windows drive-letter ↔ WSL2 `/mnt/<drive>` ↔ container mount is the classic breakage point; make
this the *only* code that knows those strings.

## Implementation notes (2026-07-13)
`pipeline/paths.py`. Two canonical roots (`repo`, `assets`), each with host/wsl/container forms
via `Roots`/`get_roots()` (env-var overrides `PIPELINE_REPO_ROOT_WSL` / `PIPELINE_ASSETS_ROOT_WSL`,
sensible WSL2 defaults — repo root derived from `__file__`, assets root defaults to
`/mnt/q/Omniverse`). `to_host`/`to_wsl`/`to_container(path, env)` auto-detect which root+space an
input path is in and convert; `windows_to_wsl`/`wsl_to_windows` are the one generic (drive-letter-
agnostic) regex pair every other conversion is built from. `container_mounts(env)` +
`MountSpec.as_docker_mount_string()` give T08 ready-made bind-mount specs matching the existing
`.devcontainer/devcontainer.json` convention (`source=Q:/Omniverse,target=/omniverse,...`).
`env` on `to_container`/`container_mounts` is validated but currently a no-op — cuda/isaac mount
identically today.
Tests: `tests/test_paths.py`, table-driven over 5 representative host/wsl/container triples ×
both roots, every pairwise direction + 3 round-trip compositions, plus the generic-mapping
helpers, env-var overrides, and the mount-spec builder. 46 tests, all passing (full suite: 101).

## Revised (2026-07-14) — runtime host moved off WSL2, model collapses to host/container

Bartosz asked to run and test the whole orchestrator directly from Windows for now (WSL2/Docker
"bundling" deferred to a later, unscheduled `T16`). Docker Desktop is reachable from a native
Windows Python process the same way it's reachable from a WSL2 shell — nothing about *using*
Docker required WSL2, only this module's own assumption about where the code executing it lived.

- Three spaces (host/wsl/container) → **two** (host/container). `Roots.repo_root_host` /
  `assets_root_host` are now plain `pathlib.Path` (OS-native — a Windows path on the real target,
  whatever the interpreter's own OS gives on any other), derived directly from `__file__`/env-var
  overrides with no WSL2 intermediate. `windows_to_wsl`/`wsl_to_windows` and `to_wsl` are gone —
  there's no second host-side space to translate between anymore.
- Env vars renamed: `PIPELINE_REPO_ROOT_WSL` → `PIPELINE_REPO_ROOT`, `PIPELINE_ASSETS_ROOT_WSL` →
  `PIPELINE_ASSETS_ROOT`.
- Acceptance criteria re-verified under the new model: `to_container(to_host(x)) == x` still holds
  (now a 2-space round trip); every `Q:\` ↔ `/omniverse` and repo ↔ `/workspace` mapping still has
  table-driven coverage. Host-side assertions compare `Path` objects (not raw strings) so the same
  test file is correct whether it actually runs on Windows or this sandbox's Linux — `str()` of a
  `Path` renders differently per OS, `Path.__eq__` doesn't.
- Tests: `tests/test_paths.py` rewritten for the 2-space model — 34 tests (down from 46; dropped
  the now-meaningless wsl-round-trip and drive-letter-generic-helper tests, added backslash-input-
  tolerance and case-insensitive-matching tests instead). Full suite green (120 total, 6 skipped —
  the GPU-real tests, see T08).
- No task reopened: this module still centralizes all path translation (T06's actual acceptance
  bar), just over two spaces instead of three. Full rationale in
  `.claude_notes/NOTES_pipeline_orchestration.md`'s "Runtime host moved off WSL2" entry.
