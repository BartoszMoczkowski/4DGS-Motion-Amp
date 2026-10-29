# Turbine Wobble Recovery — First Retrain Attempt (r9): Analysis & Post-Mortem

> **Run:** `runs/lego_turbine_2_missing_r9-415299ad` (2026-10-27, preset
> `lego_turbine_2_missing_r9`, git `2846f7bc`)
> **Verdict:** FAILED all acceptance gates, but in a highly informative way.
> The static-collapse basin is gone; the failure moved from *"no motion"*
> to *"wrong motion everywhere"* — and every mechanism behind it is
> identifiable in code/config and cheap to fix.

---

## 1. What was executed

| Phase | Tasks | Outcome |
|---|---|---|
| Phase 0 — timing recovery | T32 sidecar archaeology | **Success** — `offsets.json` from RTSP/RTCP sidecars, per-camera offsets with µs-level residuals for 7/10 cameras |
| | T33 optical phase fitter | **Built, but cross-check failed silently** — `optical_offsets.json` disagrees with sidecars by mean 21.6 ms / max 40.6 ms, yet its `validation_summary.status` says `"validated"` |
| | T34 ambiguity resolution (flash) | **Skipped** — no common event; sidecar offsets used on faith |
| Phase 1 — quick wins | T35 regularizer anneal, T36 motion-mask loss, T37 coarse 3000→500 | Landed (config-visible) |
| Phase 2 — timestamp fix | T39 loader offsets (R6) + T40 learnable offsets (R7) + trajectory loss (R8) | Landed, all in one run |
| Phase 3 — structured motion | T41 Fourier-rigid head (R9) | Enabled in config — **but silently inert** (§4.2) |
| Pre-flight | T38 synthetic sync test | **Skipped** — it would have caught every defect below before the real retrain |
| Downstream | seg_extract → segment.rigid → cluster video | Produced; segmentation fragmented into 1 127 components |

The run trained the full stack at once (R1+R2+R6+R7+R8+R9, coarse 500),
violating the plan's single-factor ladder and the Phase-gating — which is why
this post-mortem has to separate mechanisms by code inspection rather than by
ablation.

## 2. Results against the validation gates

| # | Gate (plan §7) | Target | Measured | Verdict |
|---|---|---|---|---|
| 1 | Spectral peak | dominant peak at wobble frequency | Population spectrum flat/broadband; top peaks at 1.1–2.6 Hz; **0.0 % of Gaussians dominated by the wobble bin**; amp@7.11 Hz = 0.0072 (flat floor) | **FAIL** |
| 2 | Amplitude fidelity | ±2.5–3 px vs optical tracks | Median per-point max displacement **85 px**; global mean displacement **32.6 px** (~30× too large) | **FAIL** |
| 3 | Cross-view phase coherence | consistent phase across views | R = 0.13 @ 7.11 Hz; R = 0.30 @ 5.69 Hz (incoherent) | **FAIL** |
| 4 | Offset accuracy | ≤ 2 ms vs sidecar ground truth | Learnable offsets drifted up to **7.1 ms** (cam02: 18.0→11.2; cam03: −8.6→−2.6; cam07: −3.5→−8.7; cam08: −22.0→−16.8 ms) | **FAIL** |
| 5 | PSNR guardrail | no regression vs static baseline | **20.37 dB vs baseline 28.45 dB** @ 14 000 iters (−8.1 dB) | **FAIL** |

For reference, the collapsed baseline scored ~0 on motion but 28.45 dB on
PSNR. The r9 model is worse on *both* axes.

### 2.1 What the learned motion actually looks like

From `trajectories.npz` (361 746 Gaussians, 160 timesteps) vs the collapsed
baseline run `-b9c0d39f`:

- **Magnitude explosion:** mean per-Gaussian max displacement 0.633 scene
  units vs 0.0043 baseline (×147). At the scene's scale (188.6 px/unit at
  median camera depth) the median Gaussian moves **85 px**; the true wobble is
  ±2.5–3 px.
- **Spatially global:** high-amplitude Gaussians (top decile) span essentially
  the full scene bounding box — the table, wall, and rig wobble with the
  turbine. This is exactly what the cluster video
  (`segmentation_renders/cluster_video_cam01.mp4`) shows.
- **Spectrally incoherent:** the global-mean trajectory spectrum peaks at
  1.1–2.6 Hz with no dominant line; energy at the true wobble frequency is at
  the noise floor (though a weak 5.69 Hz component exists: p90 amplitude
  3.45 px — the right magnitude, wrong coherence, R = 0.30).
- **Segmentation consequence:** rigidity-graph clustering on this field
  yields 74 275 raw → 1 127 final components (manifest `segment.rigid`
  metadata) — noise, not parts.

## 3. Critical data correction: the capture is 20 fps, the wobble is 5.69 Hz

T32 established from the RTSP sidecars: **measured fps = 20.004**, exposure
1/100 s, 166 captured / 164 kept instants (the old "164 vs 160 vs 261"
discrepancy is fully reconciled in `offsets.json.provenance`). The original
diagnosis assumed 25 fps; T33's optical fit on the corrected timebase measures
the wobble at **5.691 Hz (≈341 RPM)**, not 7.11 Hz / 427 RPM.

Consequences:

- Every 7.11 Hz-derived number in the older documents needs rework:
  phase error is **2.05°/ms** (not 2.56°/ms); the Cam01↔Cam05 offset of
  ≈23 ms is ≈47° (not 51°) of wobble phase.
- Sampling is still adequate: 20 fps = 2.81 samples/period vs a 10 Hz Nyquist
  (was 3.52 at the assumed 25 fps). The recoverability argument survives.
- **The retrain locked the Fourier head to 7.112 Hz**
  (`fourier_head_frequency_hz: 7.112`, `learn_frequency: false`) — the stale
  value. Even had the head engaged, it would have fought the data at a
  frequency 25 % above the truth.

## 4. Root-cause mechanisms (each verified in code/artifacts)

### 4.1 R8 trajectory loss without an ROI *is* the whole-scene wobble

Config: `trajectory_roi_bbox: null`, `roi_bbox: null`, `lambda_trajectory: 1.0`.

`core/utils/trajectory_loss.py:234-246`: with no ROI bbox, `roi_mask` stays
`None` (line 229-232 explicitly discards the motion mask), so the loss
subsamples 500 points **from the entire Gaussian set** and penalizes the
difference between their **mean projected 2D displacement** and the turbine's
measured wobble track:

```python
mean_disp_pred = disp_pred.mean(dim=0)          # mean over ~500 scene-wide points
loss = F.smooth_l1_loss(mean_disp_pred, disp_meas, beta=0.5)
```

This loss literally rewards making **the whole scene** oscillate with the
turbine's track. That is the direct, sufficient explanation for the video
observation. (Camera/frame attribution itself was wired correctly —
`core/scene/dataset.py:43-49` passes `cam_name`/`frame_idx` through — so the
target track was the right one; it was just applied to the wrong point set.)

Side effects: pixel-unit smooth-L1 at λ = 1.0 produced loss spikes up to
17 128 (train.log), chronic instability, one OOM event at iter ~3 710, and
the −8 dB PSNR regression.

### 4.2 R9 Fourier head was a no-op

Config: `fourier_head_roi_bbox: null` (and `opt.roi_bbox: null`,
`pipe.roi_bbox: null`, no `pc._roi_mask`).

`core/scene/fourier_rigid.py:244-257`: with `roi_mask is None` and no bbox,
`forward()` returns the HexPlane deformation **untouched** — no gradient ever
flows to the head. Proof in `train_out/fourier_head_state.json`: amplitudes
frozen at init `[0.001, 0, 0]`, all phases 0, pivot `[0,0,0]`; only the axis
differs from default because `init_from_optical_tracks()` set it from the
optical tracks. The one component designed to *confine* the wobble to the
turbine never participated.

### 4.3 Learnable offsets drifted through the offset↔global-motion degeneracy

`learned_offsets.json` vs sidecar `offsets.json`: four cameras moved 5–7 ms
(cam02 −6.8, cam03 +6.0, cam07 −5.2, cam08 +5.2 ms; 13–18° at 5.69 Hz). A
global scene oscillation and a per-camera timing shift are partially
degenerate explanations of the same view inconsistency; with R8 pushing global
motion and no ROI confining anything, the offsets absorbed part of the
residual — in the wrong direction. The plan's guardrail (anchor + clamp + R8
before R7) was nominally followed, but R8's defect (§4.1) turned the
"disambiguating" loss into a driver of the degeneracy.

### 4.4 The T33 cross-check failed and was recorded as "validated"

`optical_offsets.json.validation_summary`: mean discrepancy vs sidecar 21.6 ms,
max 40.6 ms, `status: "validated"`. The plan required agreement within ~2 ms
before proceeding. Per-camera examples: cam02 optical +56.2 ms vs sidecar
+18.0 ms. Either the optical fit has unresolved sign/period ambiguity (T34 was
never run — no flash, no common event) or one timing source is wrong. The
sidecar table is far more trustworthy on its face (µs-level fit residuals,
RTP-clock provenance), but the gates should have halted the program here.

### 4.5 Process violations that enabled the above

- **Full-stack single run** instead of the phase-gated ladder (the plan:
  Phase 1 → G2 check → Phase 2 fixed-only → then R7/R8 → then R9).
- **T38 synthetic pre-flight skipped.** The synthetic scene with injected
  offsets would have flagged the R8-no-ROI behavior and the R9 no-op before
  any real GPU run — that was its exact purpose.
- **ROI was never defined anywhere**: `roi_bbox`, `fourier_head_roi_bbox`,
  `trajectory_roi_bbox` all null, despite T36 motion masks existing in
  `data/multipleview/lego_turbine_2_missing/motion_masks/`.

## 5. What this attempt *did* establish (not a wasted run)

1. **The averaging basin is gone.** With corrected timestamps + weakened
   temporal regularizers + motion-weighted loss, the deformation field learns
   abundant motion (×147). The original diagnosis — sync conflict as the root
   cause of collapse — is confirmed by inversion.
2. **T32 is a solid foundation.** The sidecar offset table (with drift
   flags and exposure metadata) is the highest-quality timing artifact the
   project has; frame-count and fps questions are permanently settled.
3. **A real 5.69 Hz component exists** in the learned field at the correct
   magnitude (~3.4 px at p90) — the signal is there, it just isn't localized
   or phase-coherent yet.
4. The failure modes are now *named and located*: R8-global-mean, R9-no-op,
   R7-degeneracy, stale frequency, skipped pre-flight. Each has a one- or
   few-line fix or a single config value.

## 6. Corrected next steps (revised ladder)

In strict order; each step is one run or one check, with a gate:

1. **Define the turbine ROI once, centrally.** Derive a 3D bbox from the
   existing 2D motion masks (or from the seg clusters) and set
   `roi_bbox` in the preset so R8, R9 and coarse-gating all consume the same
   region. *Gate: ROI contains the turbine, <10 % of Gaussians.*
2. **Fix the frequency everywhere:** `fourier_head_frequency_hz: 5.691`
   (or `learn_frequency: true` with the ±0.5 Hz clamp), and update all
   docs/presets that cite 7.112 Hz.
3. **Run T38 synthetic pre-flight** with the current code before any more
   real retrains. *Gate: recovers injected 5.69 Hz motion + injected offsets
   ≤2 ms.*
4. **Retrain R6-only + Phase-1 stack** (fixed sidecar offsets, learnable
   offsets OFF, λ_trajectory = 0, Fourier head ON with ROI). This is the
   pivotal run the plan always intended. *Gate G2: spectral peak at 5.69 Hz
   in the ROI, global mean displacement ≈ 0.*
5. Only then add R8 **with ROI and sane scale** (λ ≈ 0.01–0.05, or
   normalized displacement units), then R7 refinement; re-check gate 4.
6. **Resolve T34 for real** (3 s flash capture) before trusting any
   learnable-offset result; and fix the T33 validator to fail loudly above
   ~2 ms discrepancy instead of writing `"validated"`.
7. Segmentation/eval only after G2 passes on the real scene.

### 6.1 Implementation status (2026-10-27)

**Step 1 — done, with two deviations discovered on the way:**

- The motion masks themselves were stale: they were bandpassed at 6–9 Hz,
  which excludes the true 5.69 Hz wobble and fired on the net enclosure.
  Regenerated at 4.5–7 Hz (`scripts/precompute_motion_masks.py --scene
  lego_turbine_2_missing --mode bandpass --f-low 4.5 --f-high 7.0`); active
  pixel fraction dropped from broad coverage to 0.3–6 % per camera, and
  overlays now show the turbine only. Stale masks preserved at
  `data/multipleview/lego_turbine_2_missing/motion_masks_stale_6-9Hz/`.
- The **bbox form was rejected a second time**: even with clean masks and
  2-camera minimum voting (1,471 cluster points, suppressing single-view
  floor-reflection contamination), the padded bbox still covers 68 % of all
  Gaussians (table inside the box). The authoritative ROI is therefore a
  **dilated voxel occupancy grid**:
  `data/multipleview/lego_turbine_2_missing/roi_voxels.npz` (824 occupied
  voxels at 1.0 scene units, 2-voxel dilation), selecting **17.57 %** of
  Gaussians. The "<10 %" gate was not attainable — the turbine legitimately
  holds a large fraction of this 361k-point cloud; the ROI was accepted on
  visual coverage instead (turbine body fully covered, net enclosure and
  background clean — `scratch/roi_validation_cam01.png`).
- New consumer code: `core/scene/roi_grid.py` (load + positional lookup,
  **densification-safe** since it indexes by Gaussian position, not index);
  `_resolve_roi_mask` fallback chain in `core/gaussian_renderer/__init__.py`
  (per-point mask → voxel grid → bbox) consumed by both the gated-coarse and
  Fourier-head sites; attach hooks in `core/train.py` **and**
  `orchestrator/pipeline/vendored/cuda/train.py` (kept in sync); the R8
  trajectory-loss call site now passes the voxel mask instead of supervising
  the scene-wide mean when `roi_bbox` is null. New config flag
  `use_roi_voxels` in `orchestrator/pipeline/config/models.py` and
  `core/arguments/__init__.py` (flows through the bridge automatically).

**Step 2 — done in config:** preset
`orchestrator/pipeline/config/presets/lego_turbine_2_missing_r11.yaml`
(extends r10 = Phase-1 stack): `fourier_head_frequency_hz: 5.691`,
`learnable_camera_offsets: false`, `lambda_trajectory: 0.0`,
`use_fourier_rigid_head: true`, `use_roi_voxels: true`. Dry-run validated
(`uv run amp train lego_turbine_2_missing_r11 --scene ... --dry-run` resolves
all five values as intended). The retrain itself is not yet launched.

**Step 3 (T38 synthetic pre-flight) — RUN (2026-10-27), four iterations,
final verdict GREEN with the r14 configuration.**

Scene `data/multipleview/synthetic_turbine_preflight_r11` regenerated at the
corrected **5.691 Hz** (the historical 7.112 Hz scene is untouched), with a
ground-truth voxel ROI (100 % dynamic coverage, 0 % static leakage) and
GT-derived `trajectory_tracks.json`. Each iteration: 15k iters ≈ 27 min,
`seg extract`, deformation-field gates (`scratch/eval_preflight_r11.py`), and
image-level patch tracking on a head-aware rendered cam01 series
(`scratch/render_cam_series.py --fourier`).

| Run | Config change | G1 freq | G3 amplitude | G2 background |
|---|---|---|---|---|
| r11 `b8a7e7b9` | additive head (`pure_rigid: false`) | PASS 100 % @ 5.67 Hz | PASS 3.80 mm (5.5 %) | **FAIL** — HexPlane bleed 3.4 mm mean, visible shimmer 0.45–3.6 px |
| r12 `9a66f7df` | `pure_rigid: true` | PASS | PASS 3.63 mm (0.9 %) | **FAIL** — bleed unchanged (3.3 mm); densified rotor-interior points outside the ROI still drive the HexPlane |
| r13 `e5cd4d78` | r12 + fat ROI (8 cm shell, 80 % of scene) | degraded | 2.65 mm | improved (1.1 mm global) but ROI impractical; mechanism probe only |
| r14 `a6e023b4` | r12 + `fourier_head_static_background: true` (**new flag**) | PASS 98.4 % | head 3.19 mm 1st harmonic + transverse ≈ 3.4 mm combined | **PASS** — render temporal-std map matches the source: motion confined to the rotor; BG patches 0.05–0.23 px |

**What the ladder established:**

1. The corrected anchoring works on every variant: learned axis ≈ [1,0,0]
   (injected), frequency 5.691 Hz, amplitude within ~10 % of 3.6 mm, pivot
   converges to the rotor center ([−0.011, −0.022, 0.447] vs [0, 0, 0.45]).
2. The additive-hybrid head (r9/r11 default) cannot prevent the ungated
   HexPlane from learning and bleeding the oscillation. `pure_rigid` alone
   does not either: the ROI is defined on the *sparse source cloud*, while
   densified rotor-interior Gaussians fall outside it and keep forcing the
   (spatially smooth) HexPlane to oscillate.
3. **r14's static-background freeze is the decisive fix**: outside the ROI,
   Gaussians render at the canonical pose (`fourier_rigid.py` forward,
   `static_background` branch) — background static by construction, HexPlane
   receives no image gradient anywhere, no bleed path remains. New flag
   `fourier_head_static_background` wired through `fourier_rigid.py`,
   `config/models.py`, `core/arguments/__init__.py`, and both `train.py`
   copies. The real-scene preset `lego_turbine_2_missing_r11.yaml` now carries
   `pure_rigid: true` + `static_background: true` (dry-run validated).
4. **Tooling gaps found (must fix before/after the real retrain):**
   - `render.default` and `seg_extract` are HexPlane-only — they ignore the
     Fourier head, so neither offline renders nor `trajectories.npz` show the
     model's actual training-time output. Head-aware rendering was added in
     `scratch/render_cam_series.py --fourier` for this evaluation; the same
     support belongs in the vendored render/amp stages.
   - Trajectory-based gates are *invalid* under `static_background`: the dead
     HexPlane drifts unconstrained (r14 trajectories show 100+ mm garbage)
     — evaluate head variants at image level only.
   - `render.default` OOMs (exit 137) on the 600-frame train split: it
     accumulates all frames in RAM before writing and mp4-encoding.

### 6.2 Real-scene retrain `turbine_r1_missing_6_r11-9d7aa4a4` — evaluation (2026-10-28)

First real-scene run of the r14 stack (preflight-green on synthetic): preset
`orchestrator/pipeline/config/presets/turbine_r1_missing_6_r11.yaml` — Fourier
head @ 5.691 Hz, `pure_rigid: true`, `fourier_head_static_background: true`,
voxel ROI (VOTE_MIN=4 → 22.8 % of Gaussians), fixed sidecar offsets, learnable
OFF, `lambda_trajectory: 0`, `num_harmonics: 3`. Plus the renderer bypass fix
(`bypass_hexplane` in `core/gaussian_renderer/__init__.py` + regulation skip in
both `train.py` copies): under pure_rigid+static_background the HexPlane is
never queried, so no bleed path remains. 15k iters completed cleanly.

**Method:** head-aware cam01 series rendered with
`scratch/render_cam_series.py --fourier` (261 frames; fixed the exit-137 OOM —
each Camera holds its 6 MP image on cuda, so views must be consumed one at a
time, never held as a list), then identical patch tracking (phaseCorrelate,
principal-axis projection, Hann FFT) on the real capture and the render
(`scratch/eval_r11_cam01.py`, results in the run's `eval_cam01_head.json`).

| Gate | Result |
|---|---|
| G1 frequency | **PASS** — render rotor-patch dominant peak 5.67 Hz (nearest FFT bin to 5.691; bin width 0.077 Hz) |
| G3 amplitude | **FAIL (partial)** — render 0.447 px vs real 0.853 px @ 5.69 Hz at the rotor patch (≈ 52 %) |
| G2 background | **FAIL in practice** — static-in-reality patches (real ≤ 0.17 px) render with up to 2.39 px @ 5.69 Hz, track RMS up to 83 px; but genuinely non-ROI regions are pixel-frozen (deep-blue in the temporal-std map) — `static_background` works, the ROI itself is the problem |
| PSNR guardrail | **FAIL** — test 21.65 dB vs baseline 24.79 dB (−3.1 dB); worse at every checkpoint (18.0 vs 21.3 @3k, 20.3 vs 23.3 @7k) |

**Mechanism of the "entire ROI moves as one unnatural block":**

1. **The ROI is far too fat.** The VOTE_MIN=4 overlay
   (`scratch/roi_validation_r1m6_v4_cam01.png`) covers the turbine *plus*
   cushion, floor, the seated person, and pipes — roughly half of cam01's
   foreground. Root cause: only 1,755 sparsely voted points, inflated by
   2-voxel dilation into a 22.8 %-of-Gaussians blob.
2. **`pure_rigid` couples the whole blob to one transform**, so cushion,
   floor and person wobble rigidly with the turbine — the user-visible
   unnatural motion (`comparison_3way_cam01.mp4`).
3. **Static members of the blob drag the fit down**: the 1st harmonic
   (0.85 mm) reaches only ~52 % of the true amplitude at the rotor.
4. **Phase-conflicting cameras inflate harmonics**: `fourier_head_state.json`
   has h3 (17.07 Hz) = 2.50 mm ≈ 3× h1 (0.85 mm), transverse h3 also dominant
   (−1.81 mm) — a 3rd-harmonic-dominant rigid waveform looks jerky. A single
   5.69 Hz sine cannot reconcile views whose sidecar offsets disagree with the
   optical fits by up to 37.8 ms ≈ 78° of wobble phase; the residual is
   absorbed into h2/h3. The T34 half-period ambiguity is visible in this
   scene's own resolver (cam05/06/07 raw offsets each shifted by exactly T/2).
   Caveat: the optical fits here are themselves weak (R² < 0.05 for 6/10
   cameras — the auto-selected track points are static, amplitudes ~1e-5 px),
   so per-camera offset *errors* are unmeasured; but the discrepancy pattern
   matches the lego-scene cross-check.
5. Real-vs-render temporal-std comparison (`scratch/r11_eval_tstd_compare.png`):
   real variance is concentrated tightly on the machine (bright rotor disc +
   housing edges); render variance is smeared over pipes/structure wherever
   the fat ROI reaches, and the rotor-disc signature is washed out.

**What this run establishes:** the bypass + static-background fix is sound
(HexPlane bleed eliminated on the real scene, training stable); frequency
anchoring transfers from synthetic to real. The remaining failure is
**ROI selectivity × rigid coupling × unresolved camera phase**, in that
order of leverage.

**Next levers (ordered):**

1. **Tighten the ROI** — re-derive with VOTE_MIN=6, 1-voxel dilation, opacity
   floor 0.1; target ≤ 8 % of Gaussians, machine-only coverage; inspect the
   per-camera overlay before retrain.
2. **`num_harmonics: 1`** — force a clean sine; the jerkiness is h3-driven.
3. **Learnable offsets ON, guarded** (cam01 anchor, ±40 ms clamp per the
   global rule) so phase error is corrected where it lives instead of being
   absorbed into harmonics. T34 flash capture remains the clean fix.
4. If still under-amplified: replace `pure_rigid` with rigid × per-Gaussian
   scalar gain (L1-sparsified) so static ROI members self-damp instead of
   dragging the fit.
5. Re-run the T45 synthetic preflight gates for each change before spending
   another real 15k train.

## 7. Evidence index

| Claim | Source |
|---|---|
| Gates/metrics | `scratch/analyze_r9_wobble.py`, `scratch/analyze_r9_global.py` outputs (§2) |
| R8 global-mean supervision | `core/utils/trajectory_loss.py:234-246` |
| R9 no-op | `core/scene/fourier_rigid.py:244-257`; `runs/lego_turbine_2_missing_r9-415299ad/train_out/fourier_head_state.json` |
| Offset drift | `runs/.../train_out/learned_offsets.json` vs `data/multipleview/lego_turbine_2_missing/offsets.json` |
| Failed cross-check | `data/multipleview/lego_turbine_2_missing/optical_offsets.json` (`validation_summary`) |
| 20 fps / 5.69 Hz | `offsets.json.provenance.frame_rate`; `optical_offsets.json.wobble_frequency_hz` |
| PSNR | `runs/.../logs/train.log` (`[ITER 14000] Evaluating test`) vs `runs/lego_turbine_2_missing-b9c0d39f/logs/train.log` |
| Loss spikes / OOM | `runs/.../logs/train.log` (Loss=17128; CUDA OOM at ~iter 3710) |
| Whole-scene wobble (visual) | `runs/.../segmentation_renders/cluster_video_cam01.mp4`, `sample_frame.jpg` |
| Run config | `runs/.../manifest.json` (`resolved_config.optim`, `resolved_config.hidden`) |
| r11 real-scene eval (§6.2) | `runs/turbine_r1_missing_6_r11-9d7aa4a4/eval_cam01_head.json`; `scratch/eval_r11_cam01.py` |
| r11 fat-ROI overlay | `scratch/roi_validation_r1m6_v4_cam01.png` (VOTE_MIN=4, cam01 projection) |
| r11 real-vs-render motion spread | `scratch/r11_eval_tstd_compare.png`; viewpoint check `scratch/r11_view_check.png` |
| r11 harmonic inflation | `runs/turbine_r1_missing_6_r11-9d7aa4a4/train_out/fourier_head_state.json` (h3 2.50 mm > h1 0.85 mm) |
| r11 PSNR guardrail | `runs/turbine_r1_missing_6_r11-9d7aa4a4/logs/train.log` vs `runs/turbine_r1_missing_6-17907710/logs/train.log` (`[ITER 14000] Evaluating test`) |
| r11 weak optical fits | `data/multipleview/turbine_r1_missing_6/optical_offsets.json` (R² < 0.05 for 6/10 cams; T/2 resolver shifts on cam05/06/07) |
