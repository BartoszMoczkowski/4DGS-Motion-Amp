# T42 — Cross-scene validation & amplitude-ordering report

**Status:** todo
**Phase:** 12 / Phase 3 (structured motion)
**Depends on:** T40, T41

## Goal

Prove the recovered motion is physics, not fitting artifacts, by exploiting
the designed experiment already sitting in the repo: three scenes of the same
turbine with 0 / 2 / 5 missing blades.

## The physics sanity check

Current trained models show the pathological ordering — learned motion does
not track imbalance (mean max-displacement 0.0071 / **0.0043** / 0.0095 for
base / 2-missing / 5-missing; `runs/lego_turbine_evaluation_summary.csv`).
With the full stack (T35–T41), wobble amplitude **must increase with
imbalance**. This inversion is the single strongest evidence that the
averaging basin is gone.

## Work items

1. Retrain base / 2-missing / 5-missing (and `turbine_r1_missing_6` if data
   quality allows) with the final configuration, each as a named preset, each
   manifest recording its `offsets.json` hash.
2. Cross-scene leverage: same rig → offsets fitted on one scene are a prior
   for the others; per-scene fitting is a refinement (triples the effective
   data behind each estimate).
3. Produce the auditable per-scene motion report: axis, amplitude, phase at
   7.11 Hz, cross-view phase coherence, offset table, PSNR/SSIM guardrail —
   the form in which "capture even subtle movements" becomes a *measurement*
   capability rather than a rendering-quality metric.

## Verification

- Amplitude ordering correct (monotonic in missing-blade count).
- All five validation metrics pass per scene (see `00-index.md`).
- Report committed under `docs/` with figures.

## Deliverable

Final cross-scene report + reproducible preset ladder. Closes the program.
