# T04 — Stage base class & registry

- Status: done
- Phase: 0
- Depends on: T01
- Environment: host

## Goal
The plugin mechanism: a `Stage` interface and a registry so new ideas are added by registering an
implementation, not by editing core code.

## In scope
- `Stage` ABC declaring: `inputs` (named artifact deps), `outputs` (named artifacts),
  `environment` (host|cuda|isaac), `resources` (ResourceRequest: needs_gpu, vram_gb, ram_gb),
  and `run(ctx) -> dict[str, Artifact]`. `ctx` gives resolved config, path translator, container
  manager handle, logger, run dir.
- `@register("role.impl")` decorator + a registry with lookup by role and by full name.
- Role concept: a role (e.g. `segment`) can have multiple impls (`rigid`, `mbs`); config picks one.
- Discovery via decorator import (and optional entry-points) so out-of-tree experimental stages
  can register.

## Out of scope
Concrete stage bodies (T07/T09/T10/T11); scheduling (T05).

## Deliverables
`pipeline/stages/base.py` + `registry.py`; one dummy `EchoStage` registered for tests.

## Acceptance criteria
- Registering two impls under one role and selecting via config returns the right class.
- Duplicate-name registration errors; unknown role/impl lookup errors clearly.
- A dummy stage runs through `run(ctx)` with a fake ctx and produces a valid artifact.

## Relevant existing files
Conceptual mapping in `ARCHITECTURE.md` DAG table.

## Notes / gotchas
Keep the `ctx` object small and explicit — it's the contract every future stage depends on;
changing it later is expensive.
