# T32 — Timing recovery: sidecar archaeology + `offsets.json`

**Status:** done
**Phase:** 12 / Phase 0 (timing recovery)
**Depends on:** —
**Remedy:** R3

## Goal

Recover per-camera exposure-time offsets for the four turbine captures from the
original RTSP recording sidecars — pure arithmetic, no optics, no new algorithm.

## Background

The capture chain (`camera_sync/rtsp_capture/`) timestamps every frame against
NTP wall-clock via piecewise-linear interpolation between RTCP Sender Reports.
Misalignment enters only at export, where
`camera_sync/rtsp_capture/rtsp_capture/sync_extract.py:149,219` snaps each
camera onto a reference timeline by nearest-frame selection (±½ frame = ±20 ms
at 25 fps residual). The per-frame NTP tables in the original
`recordings/run_*` directories therefore contain effective ground-truth
exposure instants; pairwise offsets follow by subtraction.

## Work items

1. Locate the `recordings/run_*` directories for `lego_turbine_base`,
   `lego_turbine_2_missing`, `lego_turbine_5_missing`, `turbine_r1_missing_6`.
2. Verify the sidecar time base is *sensor frame-start*, not packet arrival.
3. Record per-camera exposure duration; apply the mid-exposure correction
   (observation instant = frame start + exposure/2 — shifts all cameras by
   half their possibly-different exposure times).
4. Compute the full pairwise offset table; anchor cam01 at δ = 0.
5. **Reconcile the frame-count discrepancy** (164 frames on disk vs
   `n_times: 160` in `scripts/run_lego_turbine_pipeline.py:27-29` vs 261 in
   the diagnosis) — a 4-frame drop = 160 ms timing shift, blocking if
   unresolved.
6. Write `data/multipleview/<scene>/offsets.json`:
   `{fps, exposure_s, per-camera offset_s, drift, provenance}`.

## Verification

- Offsets reproducible from two independent frame ranges of the same sidecars.
- Methodology note committed alongside the JSON.
- Drift fit per camera (offset + rate); if > ~0.1 ms/s, flag for the affine
  time model in T39.

## Deliverable

`offsets.json` per scene + provenance note. Its hash goes into every
downstream run `manifest.json`.
