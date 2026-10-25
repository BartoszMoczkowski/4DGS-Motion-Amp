# T41 — Fourier-rigid deformation head (R9)

**Status:** todo
**Phase:** 12 / Phase 3 (structured motion)
**Depends on:** T39
**Remedy:** R9

## Goal

Give the wobble a parameterization that matches its physics instead of asking
~300 000 independent decoder queries to rediscover one rigid sinusoid.

## Design

- New module beside `core/scene/deformation.py`, gated to the turbine ROI by a
  mask (from `motion-seg` clusters or a manual box).
- All ROI Gaussians share one rigid transform (R(t), T(t)):
  - T(t) = Σₙ aₙ·sin(2πn·7.11·t + φₙ), n = 1..3, along a learnable dominant
    axis (harmonic count confirmed by T33's per-camera FFTs);
  - R(t) = small-angle rotation about a learnable pivot;
  - frequency fixed at 7.11 Hz (measured to three significant figures) or
    learnable with a strong prior; initialized from the optical tracks.
- ~20 parameters instead of ~10⁶ — resists the conflicting-gradient regime and
  makes the motion **auditable** (amplitude/phase/axis readable, directly
  comparable against optical tracks).
- Everything outside the ROI (blur-disc rotor, incidental shake) keeps the
  stock HexPlane path — strictly additive.
- Staged unfreezing: freeze the head for the first few hundred fine iterations
  while ROI densification catches up (SC-GS schedule pattern); with the head
  active, exempt the ROI from `l1_time_planes` (its motion does not live in
  the time planes).

## Verification

- Head parameters match optical tracks: amplitude within ±30 %, phase
  consistent (validation metrics 2–3).
- No geometry-error absorption into motion parameters (monitor head parameter
  drift during unfreezing).
- Rotor masked out of both loss and evaluation (unrecoverable at 25 fps long
  exposure; BAD-Gaussians-style exposure integration explicitly rejected as
  poor investment).

## Deliverable

Fourier-rigid head module + ROI gating + Phase-3 fidelity retrain.
