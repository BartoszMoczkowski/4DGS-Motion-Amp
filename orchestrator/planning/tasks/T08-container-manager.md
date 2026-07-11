# T08 — Container manager

- Status: todo
- Phase: 1
- Depends on: T05, T06
- Environment: host (drives Docker Desktop)

## Goal
Let stages run inside the `cuda` and `isaac` images with correct mounts + GPU passthrough — the
mechanism that gives Claude/automation access to the GPU-bound code (problem #2, mechanics).

## In scope
- Docker control from WSL2 (docker SDK or CLI) against Docker Desktop's engine.
- `ensure_image(env)` (build `cuda` from repo `Dockerfile`; pull `isaac`); `start`, `exec`,
  `stop`, `logs` with mounts from T06 + `--gpus all`.
- Warm long-lived container per image (esp. Isaac) + persisted cache volumes to avoid cold-start.
- Log streaming into the run dir; non-zero exit surfaces as stage failure.
- Env abstraction so a stage just says `environment="cuda"` and the manager does the rest.

## Out of scope
The stage bodies (T09/T11); VRAM gating (T12).

## Deliverables
`pipeline/containers/` manager + config for images/volumes; a smoke test.

## Acceptance criteria (run on Bartosz's WSL2 machine — documented checklist)
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
