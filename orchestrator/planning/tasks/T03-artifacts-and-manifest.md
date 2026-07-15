# T03 — Artifact store & run manifest

- Status: done
- Phase: 0
- Depends on: T01
- Environment: host

## Goal
Typed artifacts and a per-run manifest that is the single read surface for Layers 2/3.

## In scope
- `Artifact` record: name, kind (dataset|model|npz|ply|png|video|json), path, producing-stage,
  metadata (shapes/counts/etc.), content hash.
- Run directory layout under a `runs/<run_id>/` root (config snapshot, per-stage logs, outputs or
  links to canonical output dirs like `output/multipleview/<name>`).
- `manifest.json`: resolved config, git SHA, per-stage {status, start/end, wall time, peak
  VRAM/RAM, log path, produced artifacts}, overall status. Atomic updates (write-temp-rename).
- Query helpers: `list_runs`, `get_manifest`, `list_artifacts`, `get_artifact`.

## Out of scope
Scheduling (T05); how peak-mem is measured (T12 fills it — leave the field nullable).

## Deliverables
`pipeline/artifacts/` module; a manifest schema; helpers that read/write it.

## Acceptance criteria
- A hand-constructed run dir round-trips through the manifest reader/writer.
- Concurrent-safe write (temp+rename) verified; corrupt/partial manifest handled gracefully.

## Relevant existing files
Current output conventions: `output/multipleview/<name>/`, `data/multipleview/<name>/`,
`motion_seg` outputs (`trajectories.npz`, `segmentation.npz`, `*_preview.png`).

## Notes / gotchas
Don't duplicate large artifacts — reference existing output paths by path+hash. The manifest is
what the MCP `read_artifact`/`get_preview` tools serve, so keep preview-able paths explicit.
