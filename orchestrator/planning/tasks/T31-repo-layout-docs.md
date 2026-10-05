# T31 — Repo layout cleanup & docs alignment

- Status: todo
- Phase: 11 (streamlining)
- Depends on: T26
- Environment: sandbox-testable (docs/consistency checks, link and command-reference greps);
  final newcomer-flow validation is a manual pass on Bartosz's machine

## Goal

One convention everywhere; docs describe reality. After this task a newcomer (or a fresh
session reading only docs) can go from clone to a running experiment using only `uv sync` +
`amp` commands.

## In scope

- **Unify the run-ID convention** (T27's `<experiment>-<cell>-<8hex>`) across docs, scripts,
  and the manifest store.
- **Resolve the two coexisting output roots:** `output/multipleview/pump01/` (direct-script
  legacy) vs `runs/<id>/` (orchestrator). Pick one (the orchestrator layout is the expected
  winner — confirm in-task), document the migration path for legacy dirs.
- **Delete/relocate remaining loose scripts:** scene-gen stragglers either become commands
  (T26) or move to a clearly-marked `scratch/` with a README stating non-support.
- **Update `AGENTS.md`:** §5 commands (now `amp` forms), §4 build/runtime (plain `uv sync`
  installs everything — per T24), §9 gotchas (remove entries fixed by T24–T30, e.g. "plain
  `uv sync` installs almost nothing", the `requirements.txt` staleness note, the run.sh/SKIP_EXTRACT
  references if T26 deleted it).
- **Update `docs/overview.md` + `docs/orchestrator.md`** milestone history: M6 (workspace/CLI
  streamlining — T24–T26), M7 (experiment standardization — T27–T30).
- Remove `python -c` patterns from docs wherever an `amp` command exists (e.g. the capture
  frame-count QA → `amp capture verify`).

## Out of scope

Any code/CLI change (T24–T30 own those; this task documents them). Rewriting historical
post-mortems (`.claude_notes/`, `docs/post-mortem`) — they stay as the record, with a pointer
to what replaced the painful pattern.

## Deliverables

Updated `AGENTS.md` (§4, §5, §9), updated `docs/overview.md` + `docs/orchestrator.md`
(milestones M6/M7), migration note for legacy `output/multipleview/` dirs, scratch/ README,
consistency-check script or test grepping docs for raw-python command references that now have
`amp` equivalents.

## Acceptance criteria

- Fresh-clone walkthrough documented and manually verified on Bartosz's machine: `uv sync` →
  `amp --help` → `amp exp run` on the example spec → MLflow UI shows runs.
- No doc references raw-python paths for routine operations (enforced by the consistency
  grep/test, with an explicit allowlist for the legitimate cases: Docker stage execs invoke
  `core/train.py` etc. inside the container, and scratch/ scripts).
- `AGENTS.md` §9 gotchas re-verified line-by-line against reality; stale entries removed or
  rewritten.
- Legacy-dir migration note exists and was exercised on at least the pump01 dir on Bartosz's
  machine.

## Relevant existing files

`AGENTS.md` (§4 build setup, §5 key commands, §9 gotchas), `docs/overview.md`,
`docs/orchestrator.md` (milestone history M1–M5), `docs/post-mortem` (pain record — source of
the `python -c` patterns to replace), `output/multipleview/` vs `runs/` (two output roots),
`scene-gen/` (loose scripts), T26's command catalog and deletion decisions, T27's run-ID
convention.

## Notes / gotchas

This task is last because it documents the others — running it early guarantees rework. Do not
rewrite history paragraphs in `docs/orchestrator.md`; append M6/M7. The consistency grep needs
an allowlist from day one or it will flag the intentional in-container script invocations.
Keep the voice of AGENTS.md (terse, factual, aimed at an AI coding agent reading the repo
cold).
