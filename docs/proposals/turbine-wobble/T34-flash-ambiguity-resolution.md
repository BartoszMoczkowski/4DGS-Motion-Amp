# T34 — Flash / common-event ambiguity resolution

**Status:** skipped
**Phase:** 12 / Phase 0 (timing recovery)
**Depends on:** T33
**Remedy:** R5

## Goal

Break the modulo-period and sign ambiguities left by any purely sinusoidal
offset fit (T33) with one unambiguous common event visible in all cameras.

## Options in preference order

1. **T32 sidecars exist** → they are absolute (NTP wall-clock); ambiguity is
   already resolved, this task becomes a confirmation check.
2. **Serendipitous common event** in existing footage — auto-exposure
   corrections, LED flicker transients. Scan for one.
3. **Flash capture (3 s):** the Dahua rig's built-in white-light illuminators
   are addressable per camera via the existing CLI
   (`uv run rtsp-capture flash -c cameras.yaml --camera camXX --duration 3`,
   `camera_sync/rtsp_capture/README.md:44`). Fire with all cameras watching
   the scene; the flash pins every camera's frame index to one physical
   instant, converting T33 phase measurements into absolute offsets.

Constraint compliance: the illuminator is part of the camera assembly — no
equipment is added to the scene.

## Why it is not optional (when sidecars are absent)

A one-period slip misaligns that camera's supervision with every other
temporal phenomenon on the shared HexPlane time axis; a sign flip supervises
the mirrored motion. Both poison T39/T40 silently.

## Verification

- One event frame index per camera, consistent with T33 phases modulo period.
- Offsets.json updated with the absolute anchor.

## Deliverable

Absolute per-camera offsets (or a documented re-shoot decision feeding the
Phase-4 capture protocol).
