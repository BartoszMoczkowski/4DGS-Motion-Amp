# Turbine Wobble Recovery — Task Board & Implementation Plan

> **Current Active Direction (Phase 14 — `W0`–`W7`):** Segmentation-free subtle-motion recovery: per-camera timing correction first, continuous soft loss weighting second, and learned low-dimensional soft motion components with a persistent HexPlane residual third.
>
> **Supersedes for implementation:** the binary-ROI / `pure_rigid` / `static_background` route explored in Phase 13 (`r9`–`r14`, documented in [`../../turbine_wobble_r9_attempt_report.md`](../../turbine_wobble_r9_attempt_report.md)). Those runs remain diagnostic evidence, not the architectural baseline.
>
> **Verified physical parameters (2026-10-27):** Capture frame rate is **20.004 fps** (not 25 fps), exposure is **1/100 s**, and the true turbine wobble fundamental on the corrected timebase is **5.691 Hz** (not 7.112 Hz). At 20.004 fps, $h_2 = 11.382\text{ Hz}$ and $h_3 = 17.073\text{ Hz}$ exceed the $\approx 10.002\text{ Hz}$ Nyquist limit and must not be enabled for the initial turbine model.

---

## 1. Active Status Board — Phase 14: Segmentation-Free Programme (`W0`–`W7`)

| ID | Task ID | Title & Spec | Depends on | Experiment | Gate before proceeding | Status |
|---|---|---|---|---|---|---|
| **W0** | T48 | [Audit and freeze the native-4DGS baseline](W0-native-baseline-audit.md) | — | Baseline | Disabled config is numerically identical to native HexPlane; dry-run matches across both `train.py` entry points; no ROI/head files auto-loaded when disabled | **done** |
| **W1** | T49 | [Harden fixed/learnable per-camera timing](W1-camera-timing-hardening.md) | W0 | **E1a**, **E1b** | Loader/optimizer tests pass; train/eval/render timestamps match identically; fixed-vs-learnable presets differ only in offset optimization | **done** |
| **W2** | T50 | [Multi-view frequency-domain & trajectory motion evaluation](W2-multiview-motion-evaluation.md) | W1 | Evaluates **E1a/E1b+** | Recovers synthetic complex $Z = A e^{i\phi}$; rejects wrong phase/amplitude even at target frequency; rejects low-$R^2$ optical tracks loudly | **done** |
| **W3** | T51 | [Continuous normalized soft loss weighting](W3-soft-loss-weighting.md) | W1, W2 | **E2** | Static sequence gives weights $\approx 1.0$; smooth normalized maps ($\mathbb{E}[w]=1$, floor $\ge 1$, cap $10\text{–}20$); no hard masks | **done** |
| **W4** | T52 | [Fixed-$K$ soft learned motion components](W4-soft-motion-components.md) | W3 | **E3** | Zero-init is exact W3 no-op; densification/prune/checkpoint tests pass; synthetic 1- and 2-mover pilots beat W3 baseline | **done** |
| **W5** | T53 | [Reversible outer-loop component split/merge](W5-reversible-split-merge.md) | W4 | **E4** | Synthetic $K=1 \to 2$ discovers two movers; static scene adds 0 components; rejected split restores bit-exact prior state | **done** |
| **W6** | T54 | [Production integration, streaming render & experiment ladder](W6-production-integration-ladder.md) | W1–W5 | **E1a–E4** ladder | Training/render/`seg_extract`/`amp` deformation parity; streaming bounded-RAM render (no 600-frame OOM); single-factor attribution | **done** |
| **W7** | T55 | [Optional validated per-track trajectory supervision](W7-validated-trajectory-supervision.md) | W6 | **E5** (contingency) | Off by default; cannot produce scene-wide supervision; enabled only if W4/W5 evidence shows a remaining deficit | **deferred** |

### Dependency graph (Phase 14)

```text
W0 (T48: native baseline audit)
 └──► W1 (T49: fixed/learnable timing) ──┬──► W3 (T51: soft loss weights) ──► W4 (T52: fixed-K soft components) ──► W5 (T53: split/merge) ──► W6 (T54: production ladder) ──► W7 (T55: optional trajectory loss)
                                         └──► W2 (T50: multi-view phasor eval) ─────────▲──────────────────────────────▲─────────────────────────▲
```

---

## 2. Compact Run Board & Frozen Experiment Order

| Run | Work Package | New factor introduced | Required OFF | Primary gate evidence |
|---|---|---|---|---|
| **E1a** | W1 | Fixed sidecar timestamps (`offsets.json`) | Learnable offsets, soft weights, components, trajectory loss, all hard ROI/coarse/bg gates | Native deformation spectra; W2 multi-view real/render complex amplitude & phase; held-out PSNR/SSIM |
| **E1b** | W1 | Guarded learnable offsets (cam01 structural anchor + quadratic sidecar prior) | Soft weights, components, trajectory loss, all hard ROI/coarse/bg gates | Same as E1a + offset drift/prior telemetry and single-factor comparison vs E1a |
| **E2** | W3 | Continuous normalized soft image-loss weights | Motion components, trajectory loss, all hard ROI/coarse/bg gates | Improvement over better of E1a/E1b without static-region artifacts |
| **E3** | W4 | Fixed-$K$ soft motion components + persistent HexPlane residual | Split/merge, trajectory loss, `pure_rigid`, `static_background`, hard ROI gates | Synthetic component recovery first; real multi-view complex coefficients, responsibility mass, and residual diagnostics |
| **E4** | W5 | Reversible outer-loop component split/merge | Trajectory loss, all hard ROI/coarse/bg gates | Synthetic discovery & exact-rollback tests; real component count and held-out improvement |
| **E5** | W7 | Validated per-track trajectory supervision | All hard ROI/coarse/bg gates | Dedicated ablation only if E3/E4 evidence identifies a remaining amplitude/phase gap |

**Decision rule:** If a stage fails, diagnose and resolve it at that stage. Never compensate for an unclear timing result with stronger motion priors, or for a component/responsibility defect with trajectory supervision.

---

## 3. Non-Negotiable Model & Engineering Rules

1. **No hard spatial gating:** No binary ROI, bounding box, voxel grid (`roi_voxels.npz`), or thresholded motion mask may decide which Gaussians are allowed to deform. ROIs may be used only for post-hoc diagnostics and plots.
2. **Hard overrides disabled:** `pure_rigid`, `fourier_head_static_background`, `gated_coarse`, and `bypass_hexplane` are **off** for this programme.
3. **Zero motion confidence means native deformation, never freeze:** Every Gaussian retains its native HexPlane free-form residual pathway $h_i(t)$. At null responsibility $q_{i0} = 1$, a Gaussian is ordinary native 4DGS.
4. **Nyquist compliance:** At **20.004 fps**, use the measured **5.691 Hz** fundamental with `num_harmonics: 1`. Do not enable $h_2$ ($11.382\text{ Hz}$) or $h_3$ ($17.073\text{ Hz}$).
5. **Frequency peak $\neq$ pass:** A spectral peak at the configured oscillator frequency is not evidence of successful reconstruction; complex amplitude and phase agreement across views (W2) plus held-out PSNR guardrails are mandatory.
6. **Loud failure & full provenance:** Invalid timing files, low-$R^2$ optical tracks, or incompatible checkpoints must fail loudly. Every run manifest must record git revision, resolved config, `offsets.json` SHA-256 hash, weight-map hash, and component initialization/structural history.

---

## 4. Historical & Diagnostic Task Board (Phases 12 & 13)

> Kept for provenance and reference. See [`../../turbine_wobble_r9_attempt_report.md`](../../turbine_wobble_r9_attempt_report.md) for the full post-mortem of `r9`–`r14`.

| ID | Title & Spec | Phase | Depends on | Status & Notes |
|---|---|---|---|---|
| T32 | [Timing recovery: sidecar archaeology + `offsets.json`](T32-sidecar-offset-recovery.md) ([methodology](T32-methodology-note.md)) | 12 | — | **done** — authoritative RTP/RTCP timing (`20.004 fps`, `1/100 s` exposure) |
| T33 | [`track_wobble_phase.py` optical offset fitter](T33-optical-phase-offset-fitter.md) | 12 | — | **done (caveat)** — measured `5.691 Hz` fundamental; validator status check hardened in W2 |
| T34 | [Flash / common-event ambiguity resolution](T34-flash-ambiguity-resolution.md) | 12 | T33 | **todo** (hardware protocol) |
| T35 | [Regularizer schedule presets (R1)](T35-regularizer-schedule.md) | 12 | — | **done** |
| T36 | [Motion-mask weighted loss (R2)](T36-motion-mask-loss.md) | 12 | — | **done** (superseded by normalized continuous W3 maps) |
| T37 | [Coarse-stage gating (R10)](T37-coarse-stage-gating.md) | 12 | — | **done** (superseded — disabled in Phase 14) |
| T38 | [Synthetic sync pre-flight scene](T38-synthetic-preflight.md) | 12 | — | **done** |
| T39 | [Per-camera timestamps in loader (R6)](T39-per-camera-timestamps.md) | 12 | T32 | **done** (hardened in W1) |
| T40 | [Learnable offsets + trajectory reprojection loss (R7+R8)](T40-learnable-offsets-trajectory-loss.md) | 12 | T39, T36 | **done (diagnostic)** — R8 without ROI caused global wobble in `r9`; superseded by W1 & W7 |
| T41 | [Fourier-rigid deformation head (R9)](T41-fourier-rigid-head.md) | 12 | T39 | **done (diagnostic)** — math reused in W4; ROI gating superseded |
| T42 | [Cross-scene validation & amplitude-ordering report](T42-cross-scene-validation-report.md) | 12 | T40, T41 | **superseded** — pre-`r9` analytical/7.112 Hz report; superseded by W2/W6 real multi-view evaluation |
| T43 | Turbine 3D ROI: mask regeneration (4.5–7 Hz) + voxel-grid lift (`roi_voxels.npz`) | 13 | T36 | **done (diagnostic)** — proved 2D-lifted 3D voxel ROI is too fat (17.6–22.8% of Gaussians) |
| T44 | Corrective preset `r11` (ROI-gated Fourier head @ 5.691 Hz) | 13 | T43 | **done (diagnostic)** |
| T45 | T38 preflight `r11`–`r14` @ 5.691 Hz; `fourier_head_static_background` flag | 13 | T44 | **done (diagnostic)** — `r14` passed synthetic preflight by freezing background |
| T46 | Head-aware render/seg tooling & streaming render OOM fix | 13 | T45 | **superseded by W6 (T54)** |
| T47 | Real-scene `r11` retrain + image-level eval (`turbine_r1_missing_6_r11-9d7aa4a4`) | 13 | T45 | **done (diagnostic)** — exposed fat-ROI × `pure_rigid` failure; motivated Phase 14 (`W0`–`W7`) |

---

## 5. Shared Context Block & Adversarial Review Protocol

### Shared Context Block (prepend to every W0–W7 agent prompt)

```text
PROJECT: 4DGS-Motion-Amp, turbine-wobble recovery and general subtle-motion reconstruction.

OBJECTIVE
Recover small, physically coherent motion in multi-camera 4DGS without adding hardware and without allowing pre-training segmentation to dictate the reconstruction. The architecture under implementation is:
1. corrected per-camera timing with native HexPlane deformation everywhere;
2. continuous soft loss weighting;
3. learned soft motion components with shared rigid/Fourier trajectories and a persistent free-form residual;
4. reversible component split/merge during training.

VERIFIED CURRENT DATA FACTS
- The capture is 20.004 fps, not 25 fps. Exposure is 1/100 s.
- The turbine wobble measured on the corrected timebase is 5.691 Hz, not 7.11 Hz.
- At 20.004 fps, h2=11.382 Hz and h3=17.073 Hz are above Nyquist. Initial component models must use the fundamental only.
- Original RTSP/RTCP sidecars were recovered and per-camera offset tables exist. They are the best timing initialization, but fit residuals do not prove absolute exposure-instant truth.
- Optical offset fitting is not currently trustworthy: the old validator wrote "validated" despite large disagreement, and one scene had R²<0.05 for 6/10 cameras. Quality-gate all optical evidence.

IMPORTANT PRIOR RUNS — DO NOT REPEAT THEIR MISTAKES
- r9 enabled the full stack at once. R8 trajectory loss had no ROI and supervised the mean displacement of scene-wide points, directly encouraging whole-scene motion. R9 Fourier head was silently inert. Offsets drifted through offset/motion degeneracy. PSNR and motion gates failed.
- r11–r14 proved that pure_rigid + static_background can eliminate HexPlane bleed on synthetic data, but the real scene had a fat ROI, static objects moved rigidly, amplitude was under-recovered, and PSNR regressed.
- Binary ROI membership is rejected for the new model. ROIs may be used only for diagnostics and plots.
- Existing render/seg_extract paths may ignore non-HexPlane motion. Offline output must be checked against the training-time deformation path.
- Long render jobs previously accumulated 6 MP frames and OOMed. Render/export must stream one view/frame at a time.

RELEVANT CODE SITES (verify current code before editing; do not assume old line numbers)
- core/scene/multipleview_dataset.py — frame times and camera metadata.
- core/scene/dataset.py, core/scene/cameras.py — camera/frame attribution and masks.
- core/train.py — losses, optimizer groups, densification, checkpointing.
- orchestrator/pipeline/vendored/cuda/train.py — second training entry point that must remain behaviorally synchronized.
- core/gaussian_renderer/__init__.py — deformation dispatch; remove/bypass hard ROI gates for this programme.
- core/scene/deformation.py — native HexPlane deformation.
- core/scene/fourier_rigid.py — prior rigid-head implementation; reuse useful math, not its binary ROI control flow.
- core/utils/trajectory_loss.py — known global-mean failure; disabled unless W7 is explicitly requested.
- orchestrator/pipeline/config/{models.py,presets/} — configuration and named presets.

ENGINEERING RULES
- Inspect the repository and current tests first. Preserve unrelated user changes. Make the smallest coherent diff.
- Keep both training entry points working. Prefer shared helpers over copying logic; if both must be edited, test both.
- All new behavior must be opt-in and disabled by default. Disabled paths must be exact no-ops.
- Do not launch a full real-scene GPU retrain in a coding task unless the prompt explicitly says to do so. Run unit tests, focused integration tests, dry-run config resolution, and a short smoke train only.
- No silent fallback: invalid timing files, poor optical evidence, incompatible checkpoints, or missing component state must fail loudly.
- Every preset/run manifest must record git revision, resolved config, timing-file hash, and relevant data hashes.
- Do not claim physical success from implementation tests. Report exactly what was run and what remains unverified.
```

### Adversarial Code-Review Checklist (run after each work package W0–W7)

1. Can any ROI, mask, or missing config value still change which Gaussians may deform? (Specifically check `Scene.__init__` and `_resolve_roi_mask`.)
2. Is disabled behavior a true numerical no-op, including optimizer state and checkpoint format?
3. Do `core` and `vendored` training paths, offline rendering, trajectory export (`get_state_at_time`), and motion amplification (`render_mod`) use identical physical timestamps and deformation dispatch?
4. Can densification, pruning, or resume lose/reset offset, responsibility, component, or structural-history state?
5. Can a configured Fourier component pass evaluation merely by emitting its configured frequency despite wrong amplitude/phase?
6. Can low-quality optical tracks or empty evidence be labeled `"validated"` or trigger a broad fallback loss?
7. Does any full-image tensor accumulation create a memory regression on 600+ frame splits?
8. Are run comparisons genuinely single-factor and fully hashed in manifests?
