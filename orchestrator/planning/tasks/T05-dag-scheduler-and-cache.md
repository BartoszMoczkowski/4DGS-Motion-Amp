# T05 — DAG scheduler & caching

- Status: done (2026-07-13)
- Phase: 0
- Depends on: T03, T04
- Environment: host

## Goal
Build and run the stage graph with skip-if-fresh caching and resume — the core of "run the whole
thing from one call" (problem #1).

## In scope
- Build the DAG by topo-sorting declared input/output artifact deps; detect cycles/missing deps.
- **Cache key** per stage = hash(resolved stage config + input-artifact content hashes + code
  version[git SHA + stage source file hash]). Skip when an up-to-date output artifact exists.
- Execution controls: `run_all`, `from_stage`, `to_stage`, `only=[...]`, `force`.
- Resume: a crashed/cancelled run restarts at the first stale stage.
- Serial execution (single GPU) with a clear hook where T12's resource gating slots in.
- Write status/timing into the manifest (T03) as it goes.

## Out of scope
Resource-aware scheduling internals (T12); containers (T08) — call stages directly for now
(CPU stages in T07 exercise this).

## Deliverables
`pipeline/dag/` scheduler; `run_pipeline`/`run_stage` in `api.py` wired to it.

## Acceptance criteria
- On a toy 3-stage graph: first run executes all; second run skips all (cache hit); editing a
  stage's config or source invalidates exactly that stage and its descendants.
- `from_stage`/`to_stage`/`only`/`force` behave as specified; cycle/missing-dep detected.

## Relevant existing files
Handoff conventions the DAG encodes: `train_pump.sh`, `motion_seg/run.sh` (their implicit
ordering and file deps become explicit edges here).

## Notes / gotchas
Getting cache invalidation right is the main risk — include code version so a script edit reruns.
Hash big artifacts by size+mtime+partial-hash if full hashing is too slow; document the choice.
