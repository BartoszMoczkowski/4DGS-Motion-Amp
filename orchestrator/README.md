# Orchestrator subproject

Building the three-layer system that runs the whole 4DGS motion-amp pipeline from one call,
lets Claude drive the GPU-bound stages, and gives a consistent way to add new ideas.

- **Layer 1** — `pipeline/` execution module (custom lightweight DAG, runs on WSL2, spins up
  containers, manages VRAM/RAM).
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
    tasks/
      T01-*.md ... T15-*.md   <- one self-contained spec per task
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

Planning complete. No implementation started. Next unblocked task: **T01**.
