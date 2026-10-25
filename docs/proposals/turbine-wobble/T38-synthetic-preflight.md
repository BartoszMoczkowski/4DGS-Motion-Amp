# T38 — Synthetic sync pre-flight scene

**Status:** todo
**Phase:** 12 / Phase 0.5 (pre-flight)
**Depends on:** —
**Cross-cutting:** gates the T39 retrain

## Goal

Before burning real retrains, prove the modified pipeline end-to-end on a toy
scene where ground truth is known.

## Design

1. Construct (or render via the existing Omniverse pipeline:
   `omniverse-pipeline/`, `scene-gen/`) a multi-view sequence with a known
   rigid 7.11 Hz linear oscillation of a test object.
2. Inject the measured 15–25 ms per-camera offsets into the exported frame
   timing (simulate the nearest-frame export defect).
3. Run the T35–T37 + T39 pipeline on it.

## Diagnostic value

- Synthetic fails → implementation bug in the new code.
- Synthetic passes, real scene fails → data assumption wrong (exposure,
  drift, non-rigidity) — investigate data, not code.

Follows the pre-registration habit of `docs/capture_limitation_readouts.md`.

## Verification

- Recovered motion peaks at the injected frequency; recovered offsets match
  injected offsets within ~2 ms; amplitude within ±30 %.

## Deliverable

One synthetic scene + a repeatable pre-flight script + pass/fail record in the
run manifest.
