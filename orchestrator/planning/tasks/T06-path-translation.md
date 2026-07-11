# T06 — Path-translation module

- Status: todo
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
