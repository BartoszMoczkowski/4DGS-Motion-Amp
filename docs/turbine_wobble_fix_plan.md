# Turbine Wobble Recovery — Implementation Plan

> **Status:** Proposed plan, ready for task-board registration.
> **Inputs:** [`docs/turbine_wobble_codebase_brief.md`](turbine_wobble_codebase_brief.md)
> (verified codebase evidence) + [`docs/4DGS_Turbine_Wobble_Solutions.md`](4DGS_Turbine_Wobble_Solutions.md)
> (solution architecture, layers L1–L5, remedies R1–R11).
> **Goal:** recover the measured 7.11 Hz, ±2.5–3 px rigid turbine wobble in the
> 4DGS reconstruction of the existing `lego_turbine_*` captures, as an auditable
> motion measurement — without new hardware beyond the cameras' own features.

---

## 0. Strategy in one paragraph

The collapse is an optimization failure, not an information floor (80–93 % of
coherent amplitude survives even ±25 ms offsets). The plan therefore spends as
little as possible on data and as much as necessary on the model: **first
recover per-camera timing (cheap, no GPU), then fix the loader's shared
timestamp (one line, pivotal), and only then add motion capacity and
supervision.** Every phase ends in a retrain with a frequency-domain
acceptance gate, so a failure at any step is attributable to that step.
Remedies map to the solutions doc as R1–R11; each phase below lists them.

---

## 1. Phase map and dependency spine

```
Phase 0  Timing recovery (no GPU)          R3, R4, R5
   │           deliverable: data/multipleview/<scene>/offsets.json (+fps, exposure)
   ▼
Phase 1  Config & loss quick wins          R1, R2, R10        ─┐ independent of
   │           one retrain, new baseline                        │ Phase 0; can run
   ▼                                                            │ in parallel
Phase 2  Timestamp fix (PIVOTAL)           R6 → R7 + R8  ◄──────┘ (needs Phase 0)
   │           two retrains on lego_turbine_2_missing first
   ▼
Phase 3  Structured motion                 R9
   │           fidelity retrain, cross-scene physics check
   ▼
Phase 4  Next-capture protocol             R11 (+ R5 as standard pre-roll)
```

**Hard dependencies:** R6 requires R3-or-R4 (an offset table). R7/R8 require R6.
R9 requires Phase 2 to have produced a non-collapsed deformation field.
R1/R2/R10 require nothing and should land first because they put the training
infrastructure (presets, mask plumbing, stage gating) into final shape.

**The one dangerous ordering:** never enable learnable offsets (R7) before the
cam01 anchor, ±40 ms clamp, and trajectory loss (R8) are in place.

---

## 2. Phase 0 — Timing recovery (days, no GPU)

**Exit deliverable:** `data/multipleview/<scene>/offsets.json` per scene:
`{fps, exposure_s, cam01: {offset_s, ...}, ...}`, hashed and recorded in every
downstream `manifest.json`.

| # | Task | Detail | Verify |
|---|------|--------|--------|
| 0.1 | **Sidecar archaeology (R3)** | Locate the original `recordings/run_*` RTSP capture dirs for the four turbine scenes; per-frame NTP tables → pairwise offsets by subtraction. Confirm sidecar time base is *sensor frame-start*, not packet arrival; record exposure per camera (mid-exposure correction = offset + exposure/2). | Offsets reproducible from two independent frame ranges; methodology note committed |
| 0.2 | **Optical fallback + cross-check (R4)** | New `scripts/track_wobble_phase.py` in the pattern of `camera_sync/sync_analyzer/analyze.py`: track the turbine ROI per camera over all frames, fit A·sin(2π·7.11·t + φₖ), difference φₖ → offsets. Expected accuracy ≪ 1 ms given 3 px signal, ~164 frames. | Agreement with 0.1 within ~2 ms = mutual validation; disagreement → investigate before proceeding |
| 0.3 | **Ambiguity resolution (R5)** | Sinusoidal fits are ambiguous mod 140.6 ms and in sign. If sidecars exist, they break it. Otherwise: check footage for serendipitous common events (AE transients, flicker); failing that, 3 s flash capture via `rtsp-capture flash` with the current rig. | One unambiguous common event pinning frame index ↔ physical instant for all cameras |
| 0.4 | **Reconcile frame counts** | 164 frames on disk vs `n_times: 160` in `scripts/run_lego_turbine_pipeline.py` vs 261 in the diagnosis. A 4-frame drop = 160 ms timing shift — must be understood before trusting phase fits. | Written reconciliation in the offsets JSON header |
| 0.5 | **Drift check (free)** | Fit offset *and* drift per camera (the QR analyzer already does both). If drift > ~0.1 ms/s, the loader (Phase 2) takes an affine model tₖ = aₖ·t + δₖ. | Drift values in offsets.json (even if ~0) |

**Decision gate G0:** sidecars found → offsets authoritative, 0.2 is the
cross-check. Sidecars lost → 0.2 becomes primary and 0.3 is mandatory.
Neither possible on existing data → re-shoot decision (Phase 4 protocol) or
the bounded hypothesis-grid fallback (solutions doc §11).

---

## 3. Phase 1 — Config and loss quick wins (one retrain)

Still on shared timestamps; expected effect on motion is small — the point is
to land the infrastructure and establish the new baseline.

| # | Task | Change point | Verify |
|---|------|--------------|--------|
| 1.1 | **Regularizer schedule (R1)** | `time_smoothness_weight 1e-3 → 1e-4`, `l1_time_planes → 0`, as a *schedule* (stock until fine-stage iter ~3 000, then anneal). New preset YAMLs under `orchestrator/pipeline/config/presets/` — one per ladder step, version-controlled. | `amp train <scene> --dry-run` shows resolved values; sandbox test of the schedule logic |
| 1.2 | **Motion-mask loss ×20–50 (R2)** | `core/train.py:272-288`: temporal-variance weight map per camera (offline precompute, band-passed 6–9 Hz frame differences, dilated, clamped [1, 50]) multiplied into L1. Plumbing exists: `core/scene/dataset.py:28,38-40`. Fallback: hard ROI rectangle + soft falloff. | Weight-map visualizations inspected per scene; unit test: masked loss reduces to stock on static scenes |
| 1.3 | **Coarse-stage gating (R10)** | Option A (minimal): `coarse_iterations` 3 000 → 0–500 in preset. Option B (better): ROI Gaussians query deformation from iteration 0 — `core/gaussian_renderer/__init__.py:80-81` + stage switch `core/train.py:505-507`. Ship A now, B with Phase 3. | Coarse output PLY no longer the blurred average (visual diff on turbine ROI) |
| 1.4 | **Baseline retrain** | `lego_turbine_2_missing` with R1+R2+R10, stock timestamps. Register manifest with config hashes. | Run completes; §7 metrics logged; nonzero structured deformation at 7.11 Hz would already confirm the coherent-amplitude budget |

**Anti-pattern guard:** do not toggle R1 and R2 between separate runs of the
ladder — they interact (stronger motion signal → regularizers bite harder).
They land together here precisely so later ablations are single-factor.

---

## 4. Phase 2 — The timestamp fix (pivotal)

Run on `lego_turbine_2_missing` first (most collapsed baseline).

| # | Task | Change point | Verify |
|---|------|--------------|--------|
| 2.1 | **Per-camera timestamps, fixed mode (R6)** | `core/scene/multipleview_dataset.py:54`: load `offsets.json`, `image_times[k] = (i + δₖ·fps)/image_length` (normalized), anchoring cam01 at δ=0. **Also store fps + physical seconds in the dataset** — this permanently fixes the normalized-frequency FFT caveat (`render_amp.py:237-239`) and the unsourced "0.10 Hz" class of diagnostic. | Loader unit test: offset table in → per-camera distinct times out; dataset round-trip test |
| 2.2 | **Pivotal retrain** | Phase-1 config + R6 fixed offsets. **Acceptance gate G2: ROI-averaged trajectory FFT peaks at 7.11 Hz.** | §7 metrics; compare vs Phase-1 baseline run |
| 2.3 | **Learnable refinement (R7) — guarded** | δₖ as optimizable params, init from offsets.json, cam01 anchored, clamped ±40 ms, dedicated optimizer group at 0.1–0.5× backbone LR, frozen for first ~10 k iters. `core/train.py` optimizer groups. | Offsets stay within clamp; final δ̂ within ~2 ms of Phase-0 values |
| 2.4 | **Trajectory reprojection loss (R8)** | The 55 tracked 2D wobble points already exist: project deformed ROI Gaussian positions into each camera at corrected timestamps, L2-align 2D *displacement* against tracks. Supervises amplitude, phase, axis, and offsets simultaneously. Supervise point displacement, never dense flow (DispFlow-GS domain-gap warning). | Loss converges; enabling it improves metric (3) cross-view phase coherence |
| 2.5 | **Second retrain** | R6+R7+R8 together. | All §7 metrics; PSNR/SSIM guardrail vs static-collapse baseline must not regress |

**Gate G2 is the program's pivot:** if fixed timestamps alone do not produce a
7.11 Hz spectral peak, stop and diagnose (data assumptions: exposure, drift,
non-rigidity) before spending on R7–R9.

---

## 5. Phase 3 — Structured motion (fidelity)

| # | Task | Detail | Verify |
|---|------|--------|--------|
| 3.1 | **Fourier-rigid head (R9)** | New module beside `core/scene/deformation.py`: ROI Gaussians (mask from `motion-seg` clusters or manual box) share one rigid transform; T(t) = Σₙ aₙ·sin(2πn·7.11·t + φₙ), n=1..3, learnable axis/pivot, frequency fixed at 7.11 Hz or learnable with strong prior. ~20 parameters instead of ~10⁶. Outside ROI: stock HexPlane. Strictly additive. | Head parameters readable and match optical tracks (amplitude within ±30 %, phase consistent) |
| 3.2 | **Staged unfreezing** | Freeze the Fourier head for the first few hundred fine iterations while ROI densification catches up (SC-GS schedule pattern). | No geometry-error absorption into motion params (monitor head param drift) |
| 3.3 | **Cross-scene physics check** | Retrain base / 2-missing / 5-missing with the full stack. **Wobble amplitude must now increase with imbalance**, inverting the current pathological ordering (0.0071 / 0.0043 / 0.0095 → monotonic). | Amplitude ordering correct; per-scene auditable (axis, amplitude, phase) report |
| 3.4 | **Rotor policy** | Mask the front rotor out of loss *and* evaluation (unrecoverable at 25 fps long exposure). BAD-Gaussians-style exposure integration explicitly rejected as poor investment. | Rotor excluded from all §7 metrics |

---

## 6. Phase 0.5 — Synthetic pre-flight (cheap, do early)

Before burning real retrains: construct a toy multi-view scene with a known
7.11 Hz rigid oscillation, inject the measured 15–25 ms offsets, run the
modified pipeline end-to-end. **If synthetic fails → implementation bug; if
synthetic passes and real fails → data assumption wrong.** Substrate already
exists: the Omniverse/pump01 synthetic pipeline (`omniverse-pipeline/`,
`scene-gen/`) and the pre-registration habit of
`docs/capture_limitation_readouts.md`. Cost: one synthetic scene; saves every
later ambiguous result.

Recommended position: build during Phase 1 (CPU/GPU-light), gate Phase 2 on it.

---

## 7. Validation protocol (gates for every retrain)

All metrics computable from existing `runs/<run>/trajectories.npz` artifacts +
renders. Logged to MLflow (`amp mlflow`) per run.

| # | Metric | Pass bar |
|---|--------|----------|
| 1 | **Spectral peak test** — dominant peak of ROI-averaged displacement FFT | At 7.11 Hz (physical Hz, enabled by Phase-0 fps storage) |
| 2 | **Amplitude fidelity** — spectral amplitude projected through intrinsics | Matches optical ±2.5–3 px within ±30 % |
| 3 | **Cross-view phase coherence** — wobble phase at each camera's corrected timestamps | Consistent across all 10 views (re-tests the root defect directly) |
| 4 | **Offset accuracy** — \|δ̂ₖ − δₖ\| where ground truth exists | ≤ 2 ms (≈ 5° at 7.11 Hz) |
| 5 | **Rendering guardrail** — held-out PSNR/SSIM | No regression vs static-collapse baseline (image metrics are a guardrail, never the objective) |

---

## 8. Ablation and bookkeeping discipline

- **Strict single-factor ladder** after Phase 2: order R6 → R2 → R1 → R10 →
  R7 → R8 → R9, each run toggling exactly one factor against best-known config.
- **Every config = a named preset YAML** in
  `orchestrator/pipeline/config/presets/lego_turbine_*.yaml`; every run's
  `manifest.json` records the **hash of the offsets.json it consumed** (the
  offset table evolves through Phases 0–2; runs against different revisions
  are not comparable otherwise).
- **Cross-scene leverage:** same rig → offsets fitted on one scene are a prior
  for the others; reserve per-scene fitting as refinement.

---

## 9. Phase 4 — Next-capture protocol (camera-only, deferred)

Principle: **measure, don't synchronize** — NTP on a LAN tops out at ±2–5 ms
(5–26° of wobble phase); commodity PHYs won't do PTP. So:

1. **Lock shutter across all cameras** (`rtsp-capture config --all --shutter
   1/…`): 1/100 s keeps wobble blur ≈1.3 px; 1/250–1/500 makes the rotor
   reconstructable in principle. **Keep resolution, lock shutter** — 25 fps is
   already adequate at 7.11 Hz; the capture-limitation study says pixel
   density binds, not frame rate.
2. **Flash calibration as standard pre/post-roll**: fire each camera's
   illuminator at capture start and end; fit offset + drift; ship a per-capture
   **timing certificate** with the dataset so sync state is never again
   forensically rediscovered.
3. Optional: deliberately stagger camera phases (asynchronous interleaved
   sampling) — tenfold effective temporal sampling, free once offsets are
   measured rather than assumed.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Sidecars lost; no common event in footage | Medium | Optical fit + flash re-shoot (3 s); last resort: bounded sign/period hypothesis grid selected by held-out loss |
| Rigid *linear* oscillation → weak observability along motion axis | Medium | R8 tracks spread across visible surface; degeneracy shows up as systematic amplitude underestimate in metric 2 |
| Learnable offsets chase noise on weak signal | Low (if guarded) | Anchor + clamp + R8 before R7; never unguarded |
| Motion fidelity and image metrics decouple | Known pattern | Metric 5 as guardrail; frequency-domain metrics 1–3 as objective |
| Frame-count discrepancy unresolved | Must fix in Phase 0 | Task 0.4 blocks phase fits |

---

## 11. Proposed task-board registration

Suggested as Phase 12 in `orchestrator/planning/TASKS.md` (IDs free from T32):

| ID | Title | Phase | Depends on |
|----|-------|-------|-----------|
| T32 | Timing recovery: sidecar archaeology + offsets.json | 12 | — |
| T33 | `track_wobble_phase.py` optical offset fitter | 12 | — |
| T34 | Flash/common-event ambiguity resolution | 12 | T33 |
| T35 | Regularizer schedule presets (R1) | 12 | — |
| T36 | Motion-mask weighted loss (R2) | 12 | — |
| T37 | Coarse-stage gating (R10) | 12 | — |
| T38 | Synthetic sync pre-flight scene | 12 | — |
| T39 | Per-camera timestamps in loader (R6) | 12 | T32 or T33 |
| T40 | Learnable offsets + trajectory loss (R7+R8) | 12 | T39, T36 |
| T41 | Fourier-rigid deformation head (R9) | 12 | T39 |
| T42 | Cross-scene validation & amplitude-ordering report | 12 | T40, T41 |

Estimated calendar: Phase 0 = 1–3 days (mostly file archaeology + one script);
Phase 1 = 1–2 days + 1 retrain; Phase 2 = 2–4 days + 2 retrains; Phase 3 =
3–5 days + 3 retrains. GPU time dominated by the ~6 retrains of the ladder.
