# 4DGS Turbine Wobble — Codebase Evidence Pack

> Purpose: everything in this repository that bears on the "LEGO turbine wobble
> collapses to a static 4DGS reconstruction" problem, with exact file:line
> evidence, measured run results, existing tooling, and the concrete change
> points for the proposed fixes. Self-contained — safe to hand to anyone.
>
> Repo root: `C:\Users\barte\Code\PythonScripts\4DGS-Motion-Amp`

---

## 1. Problem statement (as diagnosed from the captured frames)

- A LEGO turbine with an unbalanced shaft weight (missing blades) shows clear
  wobble in the captured video (10 synchronized static cameras, 25 fps, ~6 MP),
  but the 4DGS reconstruction is essentially static: learned deformation ≈ 0.
- Measured from the frames: rigid whole-body oscillation at **7.11 Hz**
  (~427 RPM = shaft speed), amplitude **±2.5–3 px**, linear (not orbital),
  coherent across the whole assembly (55 tracked points within 0.57 px).
- **Primary cause — camera sync error.** Cam01↔Cam05 measured offset ≈ 20 ms,
  consistent with NTP residuals of 15–25 ms. At 7.11 Hz that is 38–64° of
  wobble-phase disagreement between views that the model believes are
  simultaneous. The deformation field shares **one timestamp across all
  views**, so the time-averaged static pose is the only consistent solution.
- Secondary causes: sub-pixel/edge-scale wobble vs. full-image loss; temporal
  smoothness + L1-on-time-planes regularizers suppress small motion; limited
  deformation-decoder capacity; the static coarse stage bakes the average pose
  into canonical space.
- Front rotor is unrecoverable in this dataset (severe motion blur from long
  exposure at 25 fps → reconstructs as a static blurred disc).

Every one of these claims maps to concrete code/config in this repo — see §3.

---

## 2. Dataset and run facts (verified from the repo)

| Fact | Value | Where |
|---|---|---|
| Scenes | `lego_turbine_base`, `lego_turbine_2_missing`, `lego_turbine_5_missing` (+ `turbine_r1_missing_6`) | `data/multipleview/` |
| Layout | `cam01`…`cam10`, 164 frames each (`frame_00001.jpg`…) | `data/multipleview/lego_turbine_base/cam01/` (164 files) |
| Resolution | 3200×1800 (≈5.8 MP), SIMPLE_RADIAL, f = 1846.1 px, k = −0.24 | `data/multipleview/lego_turbine_base/sparse_/cameras.bin` |
| Calibration | COLMAP on a static reference frame, single shared camera model | `scripts/prep_real_multipleview.py:301` (`--ImageReader.single_camera 1`) |
| Trained runs | `lego_turbine_base-0a0a74b8`, `lego_turbine_2_missing-b9c0d39f`, `lego_turbine_5_missing-76ed4b89`, `turbine_r1_missing_6-17907710` | `runs/` |
| Training config | 15 000 iters total, 3 000 coarse (static) iters, HexPlane grid [64,64,64,**200**], MLP width 128, depth 0 | `core/arguments/multipleview/lego_turbine_base.py` |
| Regularizers | `plane_tv_weight = 2e-4`, `time_smoothness_weight = 1e-3`, `l1_time_planes = 1e-4` | `core/arguments/multipleview/lego_turbine_base.py:12-14` |
| Orchestrator presets | `lego_turbine_base.yaml`, `lego_turbine_2_missing.yaml`, `lego_turbine_5_missing.yaml` | `orchestrator/pipeline/config/presets/` |
| Driver scripts | `scripts/run_lego_turbine_pipeline.py`, `scripts/evaluate_lego_turbine_series.py` | `scripts/` |
| Capture rig | 4–16 Dahua IP cameras over RTSP; RTCP Sender Reports → NTP wall-clock per frame | `camera_sync/rtsp_capture/` |

### 2.1 The trained models really are static (measured)

From `runs/lego_turbine_evaluation_summary.csv` (per-Gaussian trajectory
excursion over the full sequence, scene units):

| Scene | Gaussians | Kept (op>0.1) | Mean max-disp | p95 max-disp | Kabsch clusters |
|---|---|---|---|---|---|
| base (0 missing) | 360 426 | 312 811 | 0.0071 | 0.0129 | 8 |
| 2 missing | 361 272 | 317 511 | **0.0043** | 0.0064 | 8 |
| 5 missing | 357 836 | 309 609 | 0.0095 | 0.0185 | 8 |

The 2-blade-missing model (the imbalance case) has the *smallest* learned
motion — motion magnitude does not track physical imbalance, consistent with
"the deformation collapsed to the temporal mean". Artifacts per run:
`runs/<run_id>/{trajectories.npz, segmentation_*.npz, train_out/, manifest.json}`.

---

## 3. Evidence map — each finding → exact code

### 3.1 PRIMARY: one shared timestamp per frame index, across all cameras

`core/scene/multipleview_dataset.py:49-54`:

```python
for i in image_range:
    num=i+1
    image_path=os.path.join(images_folder,"frame_"+str(num).zfill(5)+".jpg")
    image_paths.append(image_path)
    image_poses.append((R,T))
    image_times.append(float(i/image_length))   # ← same value for every camera
```

- Time is `frame_index / num_frames`, identical for cam01…cam10. There is **no
  per-camera offset field anywhere** in the dataset, the `Camera` object
  (`core/scene/cameras.py:20,32`), or the training loop. A camera that actually
  exposed 20 ms later is trained as if simultaneous.
- At 7.11 Hz, 20 ms ≈ 51° of wobble phase. Two views 51° apart in phase of a
  rigid ±3 px oscillation disagree by ~±2.5 px on where the turbine was. The
  photometric loss (§3.4) resolves this contradiction by driving deformation
  to ~0 — the temporal mean is the loss minimum. This is the "averaging basin".

**The multipleview dataset format itself discards all real timing.** The
capture chain *has* per-frame NTP wall-clock timestamps (§3.7), but
`scripts/prep_real_multipleview.py` only writes frames + COLMAP poses; no
timestamp sidecar survives into `data/multipleview/<scene>/`. Retraining with
corrected timestamps therefore requires both (a) recovering/recording the
per-camera offsets and (b) extending the loader at
`multipleview_dataset.py:54` to accept them.

### 3.2 Coarse static stage bakes the average pose (3 000 iterations)

- `core/gaussian_renderer/__init__.py:80-81` — during the coarse stage the
  renderer returns the canonical parameters unchanged (no deformation query):

```python
if "coarse" in stage:
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
```

- `core/train.py:505-507` runs stage `"coarse"` for `opt.coarse_iterations`
  (3 000 in the turbine configs, of 15 000 total).
- Densification also runs during this static stage (`core/train.py:392-394`),
  so the Gaussian *population* — count, placement, scales — is optimized for
  the blurred time-average of the wobbling turbine. The fine stage inherits
  this biased canonical geometry.

### 3.3 Regularizers actively suppress small-amplitude temporal motion

- Values: `time_smoothness_weight = 1e-3`, `l1_time_planes = 1e-4`,
  `plane_tv_weight = 2e-4` (`core/arguments/multipleview/lego_turbine_base.py:12-14`;
  defaults in `core/arguments/__init__.py:85-87`).
- Applied **every fine-stage iteration** on top of the photometric loss —
  `core/train.py:278-285`:

```python
if stage == "fine" and hyper.time_smoothness_weight != 0:
    tv_loss = gaussians.compute_regulation(
        hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
    loss += tv_loss
```

- Implementation: `core/scene/gaussian_model.py:576-577` — TV smoothness along
  the time axis of the HexPlane + L1 sparsity on the time planes. Both push
  exactly the kind of small, coherent, periodic signal the wobble is toward 0,
  and they only need to win against a ≤3 px photometric gradient (already
  conflicted by the sync error in §3.1).
- Proposed fix values (`time_smoothness_weight → 1e-4`, `l1_time_planes → 0`)
  map to the preset YAML `hidden:` block or the per-scene config module.

### 3.4 Loss is full-image L1 + DSSIM — no motion masking anywhere

`core/train.py:272-288`:

```python
Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
loss = Ll1
...
if opt.lambda_dssim != 0:
    ssim_loss = ssim(image_tensor, gt_image_tensor)
    loss += opt.lambda_dssim * (1.0 - ssim_loss)
```

- The turbine occupies a small fraction of 3200×1800 px; static background
  dominates gradients (confirms the "loss-based suspicion").
- A `mask` field exists in the dataloader (`core/scene/dataset.py:28,38-40`)
  but is always `None` on the multipleview path and is never used in the loss.
  Motion-mask loss weighting (×20–50) is a clean, localized addition here.

### 3.5 Deformation decoder capacity

`core/scene/deformation.py`:

- HexPlane grid `resolution [64, 64, 64, 200]` — 200 temporal bins over the
  sequence (`lego_turbine_base.py:7`). 164 frames → <1 bin per frame; fine for
  7 Hz, but the spatial grid is 64³, so the wobble displacement field must be
  encoded through coarse trilinear planes plus
- `defor_depth = 0`, `net_width = 128` MLP heads (`deformation.py:61-65`):
  separate small heads for Δposition(3), Δscale(3), Δrotation(4), Δopacity(1),
  ΔSH(48). No rigid-body or Fourier parameterization exists anywhere — motion
  must be discovered per-Gaussian from scratch, which is exactly what a
  conflicting-multi-view signal prevents.
- A rigid-body + Fourier prior at 7.11 Hz for the turbine assembly would be a
  new module; nothing in the repo currently supports per-region motion priors
  (motion *segmentation* exists downstream — `motion-seg/` — but nothing feeds
  priors back into training).

### 3.6 FFT / diagnostics

- The repo's FFT machinery is in `core/render_amp.py:233-246` (and chunked
  variants ~295-339, 385-412): mean-anchored Eulerian displacement
  amplification, `torch.fft.rfft` over the time axis per Gaussian parameter.
- **Important:** the frequency bounds are in *normalized* units —
  `render_amp.py:237-239` explicitly does **not** scale by fps
  (`freq_cutoff` fractions × `rfftfreq(n_frames).max()`). Any "0.10 Hz"
  reading from this pipeline is in normalized-frequency units, not physical
  Hz; converting requires the capture fps, which the dataset does not store
  (see §3.1).
- `amp render --amp-factors` validation: exactly 8 channel factors, `-1` =
  skip (`render_amp.py` `validate_amp_args`).
- Post-training trajectory extraction + FFT-adjacent analysis:
  `motion-seg/motion_seg/extract_trajectories.py` → `runs/<run>/trajectories.npz`
  (`canonical_xyz`, `traj`, `opacity`); displacement stats in
  `scripts/evaluate_lego_turbine_series.py:137-146`.

### 3.7 Capture & sync chain (where the 15–25 ms enters)

`camera_sync/rtsp_capture/` (the real rig used for the turbine scenes):

- Dahua cameras emit RTCP Sender Reports mapping RTP → NTP wall-clock; frames
  are timestamped by piecewise-linear interpolation between SRs
  (`rtsp_capture/README.md:7-13`).
- Frame alignment into the "synchronized" export: **nearest-frame selection on
  a reference timeline**, dropping instants beyond `--tolerance` —
  `rtsp_capture/sync_extract.py:7-10, 149, 219`. Nearest-frame alignment
  leaves up to ±½ frame (±20 ms at 25 fps) residual per camera — this is the
  structural source of the 15–25 ms residuals, on top of NTP interpolation
  error. There is no sub-frame compensation.
- QR-display post-hoc sync toolkit (independent measurement path):
  `camera_sync/sync_display` (QR + clock @ 5 Hz), `camera_sync/recorder`,
  `camera_sync/sync_analyzer` (robust linear offset+drift fit per camera,
  reports RMSE in ms, `--remux` realignment) — `camera_sync/README.md`,
  `sync_analyzer/analyze.py:17-26`. Plan doc: `docs/manual_camera_sync_plan.md`.
- **LED flash for sign ambiguity already exists**: the cameras have hardware
  white-light illuminators —
  `uv run rtsp-capture flash -c cameras.yaml --camera cam02 --duration 3`
  (`rtsp_capture/README.md:44`). A flash visible in all cameras gives an
  unambiguous common event in seconds of capture.
- Exposure/shutter can be locked across all cameras
  (`rtsp-capture config --all --shutter 1/100`, README:47) — directly relevant
  to the "faster shutter to freeze the rotor" next-capture item.

### 3.8 Related prior work in the repo worth knowing

- `docs/capture_limitation_readouts.md` — pre-registered experiment series on
  exactly these capture constraints (resolution vs. proximity, motion
  detectability ladders, material realism), on synthetic `pump01`. Conclusion
  there: amplitude alone is not the binding constraint; pixel density and
  texture are.
- `docs/motion-segmentation.md` — downstream segmentation on these models;
  Kabsch EM collapses to ~3 dominant clusters (static frame + motion blob),
  consistent with collapsed deformation.
- USB single-PC recorder `amp-ui/amp_ui/cameras.py` uses barrier-synced
  threads + per-frame timestamp sidecars — *not* the rig used for the turbine
  (that is the RTSP/Dahua path).

---

## 4. Discrepancies to resolve before handing this around

1. **"Sub-pixel after training downsampling" — no downsampling exists on this
   path.** MultipleView scenes train at full 3200×1800 (verified: frames are
   3200×1800, `cameras.bin` matches, no resize anywhere in
   `multipleview_dataset.py` / `dataset.py` / `cameras.py`). At f ≈ 1846 px a
   ±2.5–3 px wobble is *above* pixel scale at training resolution. The
   sub-pixel argument may hold for the synthetic pump01 pipeline (1600×900,
   ~1 mm ≈ 1 px — `docs/capture_limitation_readouts.md:16`) but should not be
   cited as a cause for the turbine; sync conflict + regularizers + static-
   dominated loss are sufficient and better-supported.
2. **"The model's FFT reports 0.10 Hz"** — no Hz-calibrated FFT diagnostic was
   found in the repo; `render_amp.py` works in normalized frequency (§3.6).
   Whoever produced that number should state the conversion used.
3. **Capture fps (25) is not recorded in the dataset** — it lives only in the
   RTSP recording metadata/sidecars, which are not linked from
   `data/multipleview/lego_turbine_*`. Locating the original
   `recordings/run_*` directory (manifest + per-frame NTP tables) would
   immediately provide per-camera offsets for step 1 without any new optical
   tracking — check `camera_sync` output dirs / recording archives first.
4. Config header comment says 164 frames; the pipeline driver uses
   `n_times: 160` (`scripts/run_lego_turbine_pipeline.py:27-29`) — 4 frames
   are dropped somewhere in seg-extract; harmless but worth noting when
   aligning optical tracks (261 frames mentioned in the diagnosis does not
   match either number — verify which scene/cut was measured).

---

## 5. Next steps → what exists and what must be built

### Step 1 — Measure all pairwise camera offsets

- **Fastest path:** find the original RTSP recording sidecars (per-frame NTP
  timestamps from `rtsp_capture`); pairwise offsets are then pure arithmetic —
  no new code.
- **Optical path (frames only):** track the 7.11 Hz oscillation per camera over
  all frames in `data/multipleview/lego_turbine_*/camXX/`, fit sines, extract
  relative phases. No such script exists in the repo yet; closest existing
  machinery is the QR offset fitter in `camera_sync/sync_analyzer/analyze.py`
  (robust linear fit, reporting) — a new `track_wobble_phase.py` would follow
  the same pattern.

### Step 2 — Resolve sign ambiguity with one flash

- Hardware flash command exists: `uv run rtsp-capture flash` (§3.7). Needs
  ~seconds of new capture with all cameras watching the scene.

### Step 3 — Retrain on existing data with corrections

Concrete change points:

| Fix | File / location | Nature |
|---|---|---|
| Per-camera timestamps (fixed or learnable offsets) | `core/scene/multipleview_dataset.py:54` (+ `Camera.time`) | Loader change; learnable variant needs the offset in the optimizer (`core/train.py`) |
| Rigid-body + Fourier prior @ 7.11 Hz | new module beside `core/scene/deformation.py` | New code; mask from `motion-seg` clusters or manual ROI |
| Motion-mask loss weighting ×20–50 | `core/train.py:272-288` | Localized; `mask` plumbing already half-present (`core/scene/dataset.py:28,38-40`) |
| Exclude turbine from coarse stage | `core/gaussian_renderer/__init__.py:80-81`, `core/train.py:505-507` | Region-gated deformation during coarse, or skip coarse for masked Gaussians |
| `time_smoothness_weight → 1e-4`, `l1_time_planes → 0` | preset YAML `hidden:` block (`orchestrator/pipeline/config/presets/lego_turbine_*.yaml`) or `core/arguments/multipleview/lego_turbine_*.py:12-14` | Config-only |
| Retrain entry point | `uv run amp train lego_turbine_2_missing --scene data/multipleview/lego_turbine_2_missing` | Existing CLI |

### Step 4 — Next capture

- Shutter lock (`rtsp-capture config --all --shutter 1/…`), flash-sync
  calibration (`sync_display` + `sync_analyzer` QR path gives per-camera
  offset *and drift* with ms RMSE), or PTP if the cameras support it.
- Higher fps at lower resolution is a camera-config decision, not a repo one.

---

## 6. Key file index

| Topic | Path |
|---|---|
| Shared-timestamp bug site | `core/scene/multipleview_dataset.py:54` |
| Turbine training config | `core/arguments/multipleview/lego_turbine_base.py` |
| Loss (no masking) | `core/train.py:272-288` |
| Coarse static stage | `core/gaussian_renderer/__init__.py:80-81`, `core/train.py:505-507` |
| Regularizer application | `core/train.py:278-285`, `core/scene/gaussian_model.py:576-577` |
| Deformation decoder | `core/scene/deformation.py` |
| FFT amp + normalized-freq caveat | `core/render_amp.py:233-246` |
| Capture + sync (RTSP/NTP) | `camera_sync/rtsp_capture/` (`sync_extract.py:149,219`) |
| QR sync toolkit | `camera_sync/` (`README.md`, `sync_analyzer/analyze.py`) |
| LED flash command | `camera_sync/rtsp_capture/README.md:44` |
| Real-capture → multipleview prep | `scripts/prep_real_multipleview.py` |
| Turbine pipeline driver + results | `scripts/run_lego_turbine_pipeline.py`, `runs/lego_turbine_evaluation_summary.csv` |
| Trained run artifacts | `runs/lego_turbine_{base,2_missing,5_missing}-*/` |
| Prior capture-limitation study | `docs/capture_limitation_readouts.md` |
