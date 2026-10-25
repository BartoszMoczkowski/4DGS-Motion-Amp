# T39 — Per-camera timestamps in the loader (R6) — PIVOTAL

**Status:** todo
**Phase:** 12 / Phase 2 (the timestamp fix)
**Depends on:** T32 or T33 (an offset table)
**Remedy:** R6

## Goal

Fix the root defect: frame *i* of every camera is currently asserted to be
simultaneous. Give each camera its own continuous timestamp tₖ = t + δₖ.

## Change points

- **Root defect line:** `core/scene/multipleview_dataset.py:54` —
  `image_times.append(float(i/image_length))`, identical for cam01…cam10.
- Fix: load `offsets.json` from the scene dir;
  `image_times[k] = (i + δₖ·fps)/image_length` (normalized), cam01 anchored
  at δ = 0.
- **Also store fps and physical seconds in the dataset** and normalize once,
  centrally. This permanently fixes the normalized-frequency FFT caveat
  (`core/render_amp.py:237-239` never scales by fps) and makes every
  Hz-denominated diagnostic well-defined.
- If T32 found drift > ~0.1 ms/s: accept an affine model tₖ = aₖ·t + δₖ.
- No change needed in `Camera` (time stays a per-image scalar) or the HexPlane
  (already a continuous function of t — why Sync-NeRF-style offsets graft on
  without architectural surgery).

## Why pivotal

Every other layer (motion-mask loss, Fourier head, regularizer anneal)
operates on a signal the shared-timestamp bug actively destroys. This is the
one non-optional change.

## Verification

- Loader unit test: offset table in → per-camera distinct times out.
- Dataset round-trip test.
- **Pivotal retrain** on `lego_turbine_2_missing` (most collapsed baseline),
  fixed offsets, Phase-1 config. **Gate G2: ROI-averaged trajectory FFT peaks
  at 7.11 Hz.** If the peak is absent: stop, diagnose data assumptions before
  spending on T40/T41.

## Deliverable

Loader change + offsets.json consumption + G2 retrain with full metrics
(see `00-index.md` global rules).
