# Orchestrator subproject

Building the three-layer system that runs the whole 4DGS motion-amp pipeline from one call,
lets Claude drive the GPU-bound stages, and gives a consistent way to add new ideas.

- **Layer 1** — `pipeline/` execution module (custom lightweight DAG, runs natively on Windows,
  spins up containers via Docker Desktop, manages VRAM/RAM).
- **Layer 2** — MCP server over **HTTP** (streamable HTTP / SSE) wrapping Layer 1's API.
- **Layer 3** — thin UI (deprioritized).

## Where things live

```
orchestrator/
  README.md              <- you are here (status + how to navigate)
  planning/
    INSTRUCTIONS.md       <- working agreement / conventions for Claude + Bartosz
    ARCHITECTURE.md       <- the design (single source of truth for how it fits together)
    TASKS.md              <- index of all contained tasks + dependency graph + status board
    WINDOWS_SETUP.md      <- one-time Windows machine setup for the container manager / GPU stages
    WSL_SETUP.md          <- superseded by WINDOWS_SETUP.md; kept as a pointer + history
    tasks/
      T01-*.md ... T16-*.md   <- one self-contained spec per task
  pipeline/              <- (future) the Layer 1 package code
  mcp_server/            <- (future) the Layer 2 server code
  ui/                    <- (future) the Layer 3 UI code
```

## How to use this

1. Read `planning/ARCHITECTURE.md` for the whole picture.
2. `planning/TASKS.md` is the board — pick the next unblocked task.
3. Each `planning/tasks/Txx-*.md` is a contained unit of work with its own goal, scope,
   deliverables, and acceptance criteria. Work one at a time; update its status header.

Background/rationale predating this folder: `../.claude_notes/NOTES_pipeline_orchestration.md`
(the original plan), plus `NOTES_omniverse_pipeline.md` and `NOTES_4dgs_motion_segmentation.md`.

## Status

T01–T08 done (scaffold, config, artifacts/manifest, stage base & registry, DAG scheduler &
caching, path translation, wrap CPU stages, container manager). **Phase 0 / M1 reached**
(2026-07-14) — CPU stages run end-to-end with caching, no containers. **T09** (wrap CUDA stages:
train/render/seg_extract/amp) is the critical path's next stop.

Runtime host **revised 2026-07-14**: the orchestrator runs and is tested directly from Windows
(Docker Desktop driven natively), not from inside a WSL2 distro — see
`planning/INSTRUCTIONS.md`'s locked "Runtime host" decision and `planning/WINDOWS_SETUP.md` for
first-time machine setup. Bundling a WSL2/Docker environment as an alternative is deferred,
unscheduled future work (`planning/tasks/T16-wsl-docker-bundling.md`).

T08's GPU/Isaac behavior **verified for real on Bartosz's machine (2026-07-15)**: all 6
`tests/test_containers_gpu.py` checks passed (cuda build + GPU passthrough, Isaac pull +
non-interactive EULA, mount resolution, warm-reuse, Isaac cache persistence, clean teardown) —
`pipeline/containers/MANUAL_CHECKLIST.md` is now fully checked off.

Full history (including T07's brief reopen/redo under the "copy the logic in, don't call the
original script" rule) in `.claude_notes/NOTES_pipeline_orchestration.md`.
