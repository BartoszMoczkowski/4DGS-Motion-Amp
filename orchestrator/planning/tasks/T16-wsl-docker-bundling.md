# T16 — WSL2/Linux-distro bundling (deferred)

- Status: deferred (not scheduled)
- Phase: 6
- Depends on: T08
- Environment: n/a (planning placeholder)

## Goal

Package a proper WSL2 + Docker setup so the orchestrator can run from inside a managed Linux
distro (or ship as a one-command environment) instead of assuming Docker Desktop is already
installed and reachable from native Windows. This was the *original* runtime-host assumption
(`planning/INSTRUCTIONS.md`'s locked decision, 2026-07-11) before it was revised 2026-07-14 to run
natively on Windows first — see `.claude_notes/NOTES_pipeline_orchestration.md`'s "Runtime host
moved off WSL2" entry.

## Why deferred

Bartosz asked to get the whole thing running and tested from the Windows host machine first,
without the added setup burden of a manually-configured WSL2 distro (`nvidia-ctk`, mount
translation, etc. — see the now-retired `planning/WSL_SETUP.md`, superseded by
`planning/WINDOWS_SETUP.md`). Running natively on Windows turned out to need no WSL2 at all —
Docker Desktop is reachable directly, and `pipeline.paths` (T06) doesn't hardcode an OS. WSL2
support isn't blocking anything today; it only matters again if/when this needs to run somewhere
that isn't Bartosz's Windows machine (a different OS, a headless Linux box, CI, etc.).

## In scope (when picked up)

- A reproducible way to stand up a WSL2 distro + Docker (or Docker-in-WSL2) environment for this
  project — likely a devcontainer-style definition or a setup script, not manual steps.
- Re-introducing a third path space in `pipeline/paths.py` if the execution host and the Docker
  Desktop mount source ever genuinely differ again (they don't today under the 2-space model —
  see `T06-path-translation.md`'s revision note for exactly what would need to come back).
- Deciding whether this is "an alternative way to run the same code" (no `pipeline` changes needed
  beyond path space) or "a packaged product feature" (installer/onboarding flow) — that scoping
  question is itself part of this task, not resolved yet.

## Out of scope

Anything already working under the native-Windows model (T06/T08 as they stand today). This task
is additive — bundling a *second* supported way to run things, not replacing the first.

## Deliverables

Not yet defined — this is a placeholder acknowledging the deferred work exists, not a contained,
ready-to-start task per `planning/INSTRUCTIONS.md`'s "Definition of contained task". Scope it
properly (goal/in-out-scope/deliverables/acceptance criteria) before starting.

## Notes

If this is ever picked up, read `T06-path-translation.md`'s and `T08-container-manager.md`'s
2026-07-14 revision notes first — they document exactly what was WSL2-specific before this task's
predecessor state, and are the fastest way to see what "adding WSL2 back" would actually touch.
