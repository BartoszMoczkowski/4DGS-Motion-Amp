# T27 — Declarative experiment specs & `amp exp run` (2026-10-06)

Status: done (sandbox + dry-run verified; real pump-grid reproduction pending, see below).
Spec: `orchestrator/planning/tasks/T27-experiment-specs.md`. Plan: session plan
`doctor-mid-nite-havok-kid-flash.md` (Phases A/B/C executed in order).

## What landed

**Phase A — foundations**

- `RunManifest.metadata: dict` (defaulted; old manifests deserialize) + `run_pipeline(...,
  metadata=)` → `create_run`. Runs are now queryable by *what they are* (experiment/cell/
  variant/capture provenance), which the experiment runner's idempotency is built on.
- Per-impl seg keying: `segment.<impl>` stages write `segmentation_<impl>.npz`; `seg_eval`
  derives the impl from `ctx.inputs["segmentation"].producing_stage` and writes
  `seg_eval_result_<impl>.json` + artifact `seg_eval_result_<impl>` (legacy fixed names kept
  when the segmentation wasn't produced by a `segment.*` stage). The `"segmentation"` manifest
  key is unchanged — it's the DAG contract.
- `_artifacts_intact` now re-hashes directory artifacts with a recorded hash
  (`hash_directory`), not existence-only: changing a file inside a directory artifact
  invalidates the producer's cache entry and (via the rerun's new fingerprint) all downstream
  keys. Acceptance test in `test_scheduler_hardening.py`.

**Phase B — spec system + runner + CLI**

- `pipeline/experiments/spec.py`: Pydantic `ExperimentSpec`, `extends:` layering (reuses
  `config.resolver._deep_merge`, cycle-checked), eager validation (presets exist, impls known
  via a static table — the stage registry pulls numpy and would break the `amp --help`
  lazy-import probe, cells ∈ grid manifest, sweep cell ∈ cells, ≤1 `extract: true`).
- `pipeline/experiments/variants.py`: junction-based capture subsampler + numeric-iteration
  `count_gaussians`, moved out of the deleted `run_grid_4dgs.py`. Idempotent (right-count reuse,
  mismatch = error, never clobber), 100% target returns the source dir.
- `pipeline/experiments/results.py`: unified store `runs/experiment_results.csv`, 22-column
  wide nullable schema, keyed `(run_id, impl)` latest-wins, fully re-derivable from manifests
  (`collect_results` / `amp exp export`).
- `pipeline/experiments/runner.py`: cells × sweep variants × segment impls through
  `run_pipeline`/`run_stage` — no `run_dag` bypass, no hand-built run IDs, **no `force=True`**
  anywhere. Idempotency by manifest metadata + stage caching.
- `scene-gen/scene_gen_cli.py`: real `amp exp run|list|status|compare|export` bodies
  (typer-only module scope preserved).
- `experiments/pump_grid_v2.yaml`: faithful repro of the old grid (3 cells, sweep
  10k/25k/50k/100k on A40mm_M8, pump01 + render skip_train/skip_test, the 8-impl segment
  matrix with their exact presets, eval.gt fallback).

**Phase C — deletions + docs**: `run_grid_4dgs.py` / `run_grid_seg.py` deleted (`git rm`);
references updated in `scene-gen/pyproject.toml`, `scripts/smoke_workspace_imports.py`,
`AGENTS.md` (§3 + §5.7), `grid_seg.yaml` comment, `segment_rigid2.py` comment, and the
load-bearing docs (`motion-segmentation-research-summary.md`, `oracle_roi_benchmark_handoff.md`,
`omniverse-pipeline.md`, proposals 01/04/05/06, HANDOFF_rigid2, IMPLEMENTATION_PLAN §0.2.7).
Historical statements in `.claude_notes/`, `reviews/`, and planning task files left as history.

## Design decisions worth remembering

1. **`seg_eval.result_impl` config field.** Four segment entries (`rigid2`, `rigid2_roi`,
   `mask_lift_oracle`, `mask_lift`) share stage impl `segment.rigid2`; the producing-stage
   derivation alone would key all their evals to `seg_eval_result_rigid2.json` — the old
   `_backup_eval` collision all over again. The runner sets `seg_eval.result_impl` to the
   *entry* name, so all 8 results coexist per run and the store's `(run_id, impl)` key is
   collision-free.
2. **`dir_superseded_hash` instead of hash replacement.** Naive directory re-hashing in
   `_artifacts_intact` would re-run `train` after every `render` (render/amp write into their
   input model dir in place), and *replacing* the recorded hash would drift every downstream
   cache key (including render's own). Solution: the recorded `content_hash` is never touched;
   a stage that re-registers a same-path directory marks older artifacts (manifest + cross-run
   cache index) with `metadata["dir_superseded_hash"]`, and intactness accepts either value.
   External corruption still invalidates; in-place renders don't; keys stay stable.
3. **mtime kept in the directory fingerprint.** Regenerated-identical dirs invalidate the
   cache — safety over reuse, deliberate (a wrong cache hit is far more expensive than a
   spurious rerun for experiments).

## Scope calls (recorded per plan)

- Cubes drivers (`run_cubes_*`) stay: the spec schema is cell-general, so a cubes experiment is
  a YAML-only follow-up, not new code.
- `rescore_motion_classes.py` stays standalone (dual-GT scoring doesn't fit seg_eval's model);
  updated for the per-impl `segmentation_<impl>.npz` filenames.
- Junctions stay Windows-only (`mklink /J`, same-volume), documented in `variants.py`.
- Old per-impl `runs/grid_*_results.csv` files are frozen history; the unified store is
  additive/derived.
- `scene_gen.enabled: true` raises a clear NotImplementedError pointing at
  `amp capture gen-grid` — scene generation itself wasn't in scope for the runner.

## Verification

- Sandbox: `cd orchestrator && uv run --group test pytest -q` → **402 passed, 9 skipped, 2
  failed** — the 2 are the known environmental ones (`test_mcp_server.py` GPU-present,
  `test_mcp_tools.py` Docker-present). New coverage: `test_run_metadata.py` (4),
  `test_experiments.py` (11), scheduler dir-mutation/supersede tests (3), seg_eval per-impl
  keying (2).
- `uv lock --check` clean; `scripts/smoke_workspace_imports.py` 26/26 green after the
  pyproject/py-modules removal.
- `amp exp run experiments/pump_grid_v2.yaml --dry-run` prints 7 variants × 8 impls, correct
  capture dirs (g100000 → source dir), creates nothing.
- `grep -rn "run_grid_4dgs\|run_grid_seg"` → only historical notes (`.claude_notes/`,
  `reviews/`, planning tasks) and the experiments modules' "moved from/mirrors" references.

## Follow-up (Bartosz's machine)

Real pump-grid reproduction: `uv run amp exp run experiments/pump_grid_v2.yaml` with Docker
desktop up (convert/train/render are GPU stages; seg_extract GPU; the rest host). Not run here —
a fresh cell is hours of GPU and old grid runs carry no experiment metadata to reuse, so there
is no cheap smoke. First real run should do
`--only-cells pump_A20mm_M2 --only-impls rigid` as a canary before the full matrix.
