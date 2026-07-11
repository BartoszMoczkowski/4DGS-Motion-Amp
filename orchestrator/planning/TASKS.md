# Tasks — index & board

Each task is a contained unit of work with its own spec in `tasks/`. Work top-down within a phase;
respect the dependency graph. Update the `Status` line in each task file **and** this board.

## Status board

| ID | Title | Phase | Depends on | Status |
|----|-------|-------|-----------|--------|
| T01 | Subproject scaffold & tooling | 0 | — | todo |
| T02 | Unified config schema & presets | 0 | T01 | todo |
| T03 | Artifact store & run manifest | 0 | T01 | todo |
| T04 | Stage base class & registry | 0 | T01 | todo |
| T05 | DAG scheduler & caching | 0 | T03, T04 | todo |
| T06 | Path-translation module | 0 | T01 | todo |
| T07 | Wrap CPU stages (vertical slice) | 0 | T02, T05, T06 | todo |
| T08 | Container manager | 1 | T05, T06 | todo |
| T09 | Wrap CUDA stages (train/render/seg_extract/amp) | 1 | T07, T08 | todo |
| T10 | Wrap Option-A segmentation (mbs_infer) | 1 | T09 | todo |
| T11 | Wrap Isaac stages (split/motion/capture) | 2 | T08, T09 | todo |
| T12 | Resource manager (VRAM/RAM + adaptive retry) | 3 | T09 | todo |
| T13 | MCP server over HTTP (transport + auth) | 4 | T05 | todo |
| T14 | MCP tools & resources | 4 | T13, T09 | todo |
| T15 | UI (Streamlit over Layer 1 API) | 5 | T09 | todo |

## Dependency graph

```
T01 ─┬─ T02 ─┐
     ├─ T03 ─┼─ T05 ─┬─ T07 ─┬─ T09 ─┬─ T10
     ├─ T04 ─┘        │        │       ├─ T11
     └─ T06 ──────────┴────────┘       ├─ T12
                                        ├─ T14 (also needs T13)
                                        └─ T15
T05 ── T13 ── T14
T05, T06 ── T08 ── T09
```

## Critical path

T01 → T04/T03 → T05 → T07 → T08 → T09 → (then T11/T12/T14/T15 fan out).
T02 (config) and T06 (paths) are prerequisites for T07 and can be done in parallel with T03–T05.

## Milestones

- **M1 (end of Phase 0):** full pipeline's CPU stages run from `run_pipeline(preset=...)` with
  caching, no containers. Framework proven.
- **M2 (end of Phase 1):** reconstruction → segmentation → amplification runs end-to-end from one
  call using containers. Solves most of problem #1.
- **M3 (end of Phase 2):** true end-to-end from USD asset through amplified render.
- **M4 (end of Phase 4):** Claude can drive everything over HTTP MCP. Solves problem #2.
- **M5 (end of Phase 5):** UI for Bartosz. Solves the remainder of problem #1.
