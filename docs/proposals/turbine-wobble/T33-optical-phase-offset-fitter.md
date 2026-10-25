# T33 — `track_wobble_phase.py` optical offset fitter

**Status:** done
**Phase:** 12 / Phase 0 (timing recovery)
**Depends on:** —
**Remedy:** R4

## Goal

Fallback and cross-check for T32: measure per-camera offsets optically by
fitting the known 7.11 Hz wobble independently in each camera's frames.

## Approach

The wobble is a broadcast clock: every camera observes the same rigid
oscillation. New script `scripts/track_wobble_phase.py`, following the
robust-fit pattern of `camera_sync/sync_analyzer/sync_analyzer/analyze.py`:

1. Per camera (`data/multipleview/<scene>/camXX/`, all frames): compute a 1-D
   motion projection of the turbine ROI — centroid of a tracked patch set, or
   phase of a dominant spatial-frequency band in the ROI.
2. Fit A·sin(2π·7.11·t + φₖ) per camera over the full sequence.
3. Difference the φₖ → relative offsets; convert via 2.56°/ms at 7.11 Hz.

Expected accuracy ≪ 1 ms (3 px signal, sub-pixel noise, ~164 frames at 3.52
samples/period; phase-estimator variance falls with sample count).

## Known ambiguities (handed to T34)

- Offsets only modulo the 140.6 ms wobble period (3.52 frames).
- Sign/π-phase ambiguity of a rigid linear oscillation (half-period delay ≡
  mirrored motion direction).
- Gauge: anchor cam01 at δ = 0; only nine offsets estimable.

## Verification

- Agreement with T32 sidecar offsets within ~2 ms = mutual validation of both
  paths; disagreement → investigate before Phase 2.
- Sanity: fitted amplitude per camera consistent with the measured ±2.5–3 px.

## Deliverable

`scripts/track_wobble_phase.py` + per-scene optical offset table + per-camera
wobble FFTs (harmonic content answers whether the T41 Fourier head needs
n > 3).
