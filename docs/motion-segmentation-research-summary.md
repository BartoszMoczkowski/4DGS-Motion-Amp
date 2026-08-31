# Motion Segmentation of 4D Gaussian Splatting: Methods Tried, Results, and Conclusions

> Prepared for research-agent consumption.
> Project: Bartosz Moczkowski, MSc thesis, TUL. Per-part motion amplification of 4DGS.
> Scene: pump01 — 107 rigid parts, 10 calibrated cameras, 60 frames, mm-scale periodic motion.
> Date: 2026-08-12

---

## 1. Problem statement

Given a 4DGS reconstruction (canonical Gaussians + deformation network), segment the Gaussians into rigid motion groups that correspond to the physical parts of the machine. This is a prerequisite for **per-part motion amplification** (the thesis's core contribution): we can only amplify motion part-by-part if we know which Gaussians belong to which part.

**Why this is hard:** 4DGS optimizes ~10⁵ Gaussians with a continuous deformation field. The reconstruction is visually faithful, but the per-Gaussian trajectories are corrupted by optimization jitter comparable in amplitude to the true mm-scale motion. There is no ground-truth correspondence between trained Gaussians and mesh vertices — the init point cloud is split, densified, pruned, and displaced during training.

**Evaluation metric:** Adjusted Rand Index (ARI) against ground-truth part labels (107 parts on pump01). ARI = 1.0 is perfect; ARI = 0.0 is random; ARI < 0 is worse than random.

---

## 2. Methods tried (chronological)

### 2.1 Option B — Rigidity-graph clustering (baseline, T07)

**Method:** Build a k-NN graph (k=12) in canonical space. For each edge, compute the rigidity score = std-dev of the pairwise 3D distance over time. True rigid pairs have score ≈ 0;跨-part pairs have higher variance. Threshold edges via log-space Otsu, then extract connected components.

**Why it was chosen:** Pure numpy/scipy, no GPU for clustering, no ML model to train. The synthetic self-test (7-body scene) gives ARI 0.9988.

**Results on real pump01 (7 trained models: 3 grid + 4 sweep):**

| run_id | ARI | n_pred (vs 107 GT) |
|--------|-----|-------------------|
| grid-A20mm_M2 | 0.0018 | 89 |
| grid-A20mm_M4 | 0.0090 | 99 |
| grid-A40mm_M8 | 0.0069 | 51 |
| sweep-g10000 | 0.0029 | 3 |
| sweep-g25000 | −0.0184 | 2 |
| sweep-g50000 | −0.0056 | 6 |
| sweep-g100000 | −0.0329 | 10 |

**Interpretation:** Near-random. The Otsu threshold cuts through noise rather than signal.

---

### 2.2 Option A — MultiBodySync (MBS) MotNet (T10)

**Method:** MultiBodySync (Dong et al., CVPR 2021) is a 3D scan synchronization method for multi-body motion segmentation. Its full pipeline is:
1. **FlowNet** — estimates scene flow between unordered point-cloud scans
2. **Permutation synchronization** — establishes correspondence across scans (O((KN)²), where K = scans, N ≈ 256–1024)
3. **MotNet** — a learned affinity network that predicts same-body probability from pairwise motion cues
4. **Spectral segmentation synchronization** — clusters the affinity matrix
5. **Per-group Kabsch fitting** — recovers rigid transforms

**Adaptation to 4DGS:** In 4DGS, correspondence is *free* — each Gaussian keeps its identity through the deformation field. So we dropped FlowNet and permutation sync, and fed exact analytic flow (`p_i(t_j) − p_i(t_k)`) directly into MotNet with identity permutations. The checkpoint was downloaded from the authors' Google Drive (`submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar`).

**Why it was tried:** MotNet is a *learned* affinity function trained on synthetic rigid-body scenes. The hope was that it would generalize to 4DGS trajectories better than hand-crafted rigidity scores.

**Key mismatches:**
- MotNet was trained on noisy FlowNet flow at unit-scale scenes; 4DGS pump01 is ~meter-scale with ~mm motion
- MBS assumes N ≈ 256–1024; pump01 has ~10⁵ Gaussians (we FPS-subsampled to 4k)
- MBS assumes piecewise-rigid bodies; 4DGS has floaters, low-opacity noise, and continuous non-rigid deformation

**Results on real pump01:**

| run_id | ARI | n_pred |
|--------|-----|--------|
| grid-A20mm_M2 | 0.0007 | 8 |
| grid-A20mm_M4 | 0.0026 | 5 |
| grid-A40mm_M8 | 0.0005 | 6 |
| sweep-g10000 | 0.0066 | 2 |
| sweep-g25000 | −0.0158 | 2 |
| sweep-g50000 | −0.0021 | 2 |
| sweep-g100000 | −0.0075 | 2 |

**Interpretation:** MotNet is out-of-distribution for mm-scale 4DGS trajectories. Even with exact flow (no FlowNet noise), the learned affinity does not separate same-part from different-part pairs. The checkpoint was not fine-tuned (Option C — full retrain — was deferred as too high effort). MultiBodySync's relevance to this project is therefore **as a reference baseline that confirms learning-based affinity does not automatically solve the problem when the training domain mismatches**.

---

### 2.3 T18 — Upgraded Option B: rigid2 (FFT denoising + calibrated z-scores + spectral partition)

**Method:** Address the root cause suspected in baseline Option B — reconstruction jitter is white noise, while true motion is narrowband at the drive frequency. Steps:
1. **FFT band-pass** each trajectory at the auto-detected drive frequency + harmonics (rfft, keep DC + f0 + 2f0 + 3f0)
2. **Calibrate noise floor σ_d** from "static" points (those with band-limited energy below an Otsu threshold)
3. **Per-edge z-score** = rigidity_score / σ_d
4. **Adaptive threshold** = min(z_thresh, median + 6·MAD) — hard-gates affinity before spectral step
5. **Hybrid partition** — connected components first (free coarse clusters), then spectral sub-splitting only within components where the cut is justified

**Diagnostic:** Separability AUROC — z-score classification of same-part vs different-part edges against GT. AUROC ≥ 0.8 = per-edge methods viable; AUROC < 0.8 = per-edge methods capped.

**Results:**

| run_id | rigid2 ARI | separability AUROC (denoised) | drive_freq_used | sigma_d |
|--------|-----------|------------------------------|-----------------|---------|
| grid-A20mm_M2 | −0.001 | 0.626 | 1 | 4.36×10⁻⁸ |
| grid-A20mm_M4 | −0.041 | 0.585 | 1 | 2.03×10⁻⁵ |
| grid-A40mm_M8 | −0.001 | 0.671 | 1 | 3.58×10⁻⁵ |
| sweep-g10000 | 0.004 | 0.458 | 4 | 4.28×10⁻⁴ |
| sweep-g25000 | −0.018 | 0.518 | 2 | 7.83×10⁻⁴ |
| sweep-g50000 | −0.004 | 0.455 | 9 | 2.20×10⁻⁴ |
| sweep-g100000 | −0.028 | 0.507 | 12 | 3.98×10⁻⁴ |

**Interpretation:**
- **AUROC < 0.8 on every model.** Best is 0.671 (grid-A40mm_M8). The per-edge signal is fundamentally insufficient.
- **Denoising provides only marginal benefit.** ΔAUROC (denoised − raw) is 0.001–0.013.
- **Drive-frequency auto-detection is unreliable** (reports 1 cycle/clip when true motion is ~10 cycles/clip over 60 frames), but even raw AUROC is poor — fixing f0 won't save per-edge methods.

---

### 2.4 T19 — ROI motion gating (proposal 01)

**Method:** Band-limited energy gate + k-NN dilation + rigidity-lock readmission. The hypothesis: the static background cloud is poisoning the graph partition; removing it should help.

**Results:**

| run_id | rigid2_roi global ARI | rigid2_roi ARI-within-ROI | rigid2 baseline ARI |
|--------|----------------------|--------------------------|---------------------|
| grid-A20mm_M2 | 0.016 | 0.001 | −0.001 |
| grid-A20mm_M4 | −0.046 | −0.042 | −0.041 |
| grid-A40mm_M8 | −0.024 | −0.004 | −0.001 |
| sweep-g10000 | 0.010 | 0.003 | 0.004 |
| sweep-g25000 | −0.016 | −0.024 | −0.018 |
| sweep-g50000 | 0.002 | −0.009 | −0.004 |
| sweep-g100000 | −0.029 | −0.028 | −0.028 |

**Interpretation:** The motion gate keeps **~100% of points** (n_roi ≈ total N) because reconstruction jitter after band-passing has energy comparable to true mm-scale motion. There is effectively no static background to remove. **The bottleneck was never the background cloud.**

---

### 2.5 T20 — Kabsch EM (proposal 05)

**Method:** Iterative rigid-body fitting. E-step: soft assignment by trajectory residual to candidate rigid bodies. M-step: weighted per-frame Kabsch (SVD) to fit per-body rigid transforms. FFT-fingerprint init, BIC model selection, adaptive sigma annealing.

**Hypothesis:** Kabsch pools evidence across T=60 frames, so std(residual) should shrink by √(2/(3T)) ≈ 0.1 relative to single-pair edge scores. This should be more robust than per-edge methods.

**Results:**

| run_id | kabsch ARI | best_k (BIC) |
|--------|-----------|-------------|
| grid-A20mm_M2 | 0.019 | 20* |
| grid-A20mm_M4 | −0.016 | — |
| grid-A40mm_M8 | −0.009 | — |
| sweep-g10000 | 0.018 | — |
| sweep-g25000 | −0.033 | — |
| sweep-g50000 | −0.005 | — |
| sweep-g100000 | −0.046 | — |

\* BIC(20) = 61494 vs BIC(107) = 307471 — the statistical evidence only supports ~20 motion groups, not 107.

**Interpretation:** Kabsch EM is mathematically correct (sandbox ARI 0.999+) but the data does not support 107 rigid bodies at current noise levels. **The noise floor is too high for any rigid-body-fitting approach.**

---

### 2.6 T22 — Multi-view mask lifting oracle ceiling (proposal 02)

**Method:** The final bet. Use **geometric** (not motion-based) evidence: project GT-labeled Gaussians to 2D via calibrated cameras, derive a perfect per-view mask, then lift back to 3D via depth rendering. This measures the **ceiling**: what ARI would we get with a perfect ROI mask?

**Oracle results:**

| run_id | oracle global ARI | oracle ARI-within-ROI | motion-gate ARI-w-ROI | separability AUROC |
|--------|------------------|----------------------|----------------------|--------------------|
| grid-A20mm_M2 | 0.281 | 0.105 | 0.001 | 0.626 |
| grid-A20mm_M4 | 0.294 | −0.013 | −0.042 | 0.585 |
| grid-A40mm_M8 | 0.207 | 0.012 | −0.004 | 0.671 |
| sweep-g10000 | 0.128 | 0.005 | 0.003 | 0.458 |
| sweep-g25000 | 0.021 | 0.000 | −0.024 | 0.518 |
| sweep-g50000 | 0.056 | 0.002 | −0.009 | 0.455 |
| sweep-g100000 | 0.000 | −0.000 | −0.028 | 0.507 |

**Critical finding:** Even with a **perfect GT-derived ROI**, ARI-within-ROI ≈ 0 (max 0.105). The higher global ARI (0.21–0.29) is an artifact — the background is excluded and scored as one correct group, but inside the machine region clustering still collapses to 3–35 clusters vs 107 GT parts.

**Verdict: the bottleneck was never ROI quality.** The trajectory signal inside the true machine region is already too noisy for any clustering method tried.

---

## 3. The evidence chain

```
1. Baseline rigidity graph (Option B)            → ARI ≈ 0  (near-random)
2. MultiBodySync MotNet (Option A)               → ARI ≈ 0  (OOD, no fine-tuning)
3. T18 rigid2 (denoised + calibrated z)          → ARI ≈ 0, AUROC 0.45–0.67
   └─ Separability diagnostic: edge signal insufficient on ALL models
4. T19 motion-gate ROI                           → keeps 100% of points
   └─ Jitter ≈ motion; no static background to remove
5. T20 Kabsch EM                                 → ARI ≈ 0, BIC prefers K≈20 not 107
   └─ Noise too high for rigid-body fitting
6. T22 oracle ceiling (perfect ROI)              → ARI-within-ROI ≈ 0
   └─ Even perfect geometric gating fails

→ CONCLUSION: reconstruction jitter ≈ true mm-scale motion amplitude
→ All per-point-trajectory clustering methods are capped
```

---

## 4. What this means for the thesis

**Negative result (but a rigorous one):** Motion-only segmentation of 4DGS at ~10⁵ Gaussians is **reconstruction-quality-limited** on mm-scale industrial scenes. The synthetic-data framework (Omniverse/Isaac Sim capture → 4DGS train → GT labels) was essential to reach this verdict quantitatively — on real captures without GT, we would never know whether the segmentation failure was a method problem or a data problem.

**Positive contributions that stand:**
1. **Per-part motion amplification pipeline** — works. `render_amp.py` amplifies per-segment deformation channels, producing visually compelling amplified videos.
2. **Synthetic-data generation framework** — enables quantitative evaluation (ARI, IoU, separability AUROC) that is impossible on real captures.
3. **The negative result itself** — a principled demonstration of where 4DGS reconstruction quality caps downstream analysis, with a full diagnostic chain (separability AUROC → BIC → oracle ceiling).

**What was NOT tried (and why):**
- **T21 subspace spectral (proposal 04)** — PCA + local subspace fits. Skipped because T22 oracle proved the bottleneck is not the clustering algorithm but the trajectory signal itself.
- **T23 seeded part-focused (proposal 03)** — geodesic ball + PageRank from seed points. Skipped for the same reason.
- **Option C — full MBS retrain** — would require retraining MotNet on 4DGS trajectories. Deferred as high-effort with low probability of success given that the trajectory signal itself is the cap.
- **Higher-quality reconstruction** — e.g., longer training, more cameras, different loss weights. Out of scope for the segmentation rescue arc; could be future work.

---

## 5. MultiBodySync: detailed relevance assessment

**Paper:** Dong et al., "MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization," CVPR 2021.

**Core idea:** Synchronize multiple unordered 3D scans of a multi-body scene by jointly solving for (a) dense correspondences across scans via permutation synchronization, (b) per-body rigid transforms, and (c) motion-based segmentation.

**What we reused:** Only MotNet (the learned affinity head) and the spectral segmentation synchronization module. FlowNet and permutation sync were replaced by exact 4DGS correspondence.

**Why it failed here:**
1. **Training domain mismatch.** MotNet was trained on synthetic scenes with unit-scale objects and FlowNet-noisy flow. Pump01 is ~meter-scale with ~mm motion and exact flow. The learned features do not transfer.
2. **Scale mismatch.** MBS is designed for N ≈ 256–1024. We FPS-subsampled to 4k, but the full cloud is ~10⁵. The subsample loses fine structure.
3. **No fine-tuning.** The original checkpoint was used as-is. Fine-tuning on synthetic 4DGS trajectories might help, but the T18 separability diagnostic (AUROC < 0.8 on all models) suggests the edge signal is fundamentally insufficient — no affinity function can separate what is not separable.

**Relevance going forward:** MultiBodySync remains a valid reference method for point-cloud motion segmentation, but its **direct application to 4DGS without domain adaptation is not viable**. If future work retrains MotNet on 4DGS-specific trajectories, the T18 separability diagnostic provides a clear go/no-go signal (AUROC ≥ 0.8 needed).

---

## 6. Files and data

| Artifact | Path |
|----------|------|
| Baseline results | `runs/grid_seg_results.csv` |
| MBS results | `runs/grid_seg_mbs_results.csv` |
| T18 rigid2 results | `runs/grid_seg_rigid2_results.csv` |
| T20 kabsch results | `runs/grid_seg_kabsch_results.csv` |
| T19 rigid2_roi results | `runs/grid_seg_rigid2_roi_results.csv` |
| T22 oracle results | `runs/grid_seg_mask_lift_oracle_results.csv` |
| Per-run separability | `runs/<run_id>/separability.json` (T18) |
| Colored PLYs | `runs/<run_id>/segmentation_colored_*.ply` |
| Design proposals | `docs/proposals/02-multiview-mask-lifting.md` |
| Implementation plan | `docs/proposals/IMPLEMENTATION_PLAN.md` |
| Task specs | `orchestrator/planning/tasks/T{18,19,20,22}-*.md` |
| Working notes | `.claude_notes/NOTES_T22_oracle_results_2026-08-12.md` |

---

## 7. Open questions for future work

1. **Does reconstruction quality improve with more training iterations or different hyperparameters?** The sweep models (g10000–g100000) show slightly worse AUROC than grid models, suggesting more Gaussians do not help if the noise floor scales with density.
2. **Can a stronger prior (e.g., known CAD mesh topology) guide segmentation?** The GT labels come from the USD mesh; using mesh adjacency as a graph prior was not tried.
3. **Does per-part motion amplification work *without* segmentation?** The thesis's core contribution is amplification, not segmentation. If amplification is applied globally and then masked visually, the segmentation failure may not matter for the end user.

---

## 8. Synthetic data pipeline

### 8.1 Why synthetic data is essential

Real captures of industrial machinery give no ground truth. Without known camera intrinsics/extrinsics, per-pixel part labels, or per-object motion trajectories, quantitative evaluation of segmentation is impossible — one can only inspect colored point clouds and guess. The Omniverse/Isaac Sim pipeline was built specifically to break this deadlock:

- **Exact camera calibration** — poses are authored, so COLMAP is skipped entirely; reprojection error is zero.
- **Per-pixel instance segmentation** — every rendered pixel carries the prim path of the part that produced it, giving exact GT labels for every Gaussian after `omni_to_4dgs.py`.
- **Fully controllable motion** — per-part rigid SE(3) sinusoids with known amplitude, frequency, and phase, so the true displacement at any frame is analytically known.
- **Parametric scene generation** — `scene-gen/gen_scenes.py` can generate grids of scenes that vary motion amplitude, material, lighting, or camera count while holding everything else constant.

This infrastructure is what made the **negative result rigorous**: without GT labels, we could not have computed ARI, IoU, or the separability AUROC diagnostic, and we would not have known whether the segmentation failure was a method problem or a data problem.

### 8.2 Pipeline architecture

```
CAD mesh (CONJUNTO BOMBAS.usd, single fused mesh)
  → split_mesh.py      (weld + connected-components split → 107 labelled parts)
  → add_motion.py      (per-part sinusoidal SE(3), 60 frames @ 24 fps, mm-scale motion)
  → CONJUNTO_BOMBAS_animated.usd

Option A — single pump on dark grid backdrop:
  → omni_capture.py    (Isaac Sim 6.0 headless, Replicator BasicWriter)
      10 dome cameras, 1600×900, 16 path-trace subframes, 60 frames
      outputs: camNN/{rgb, instance_segmentation, camera_params, ...}
  → omni_to_4dgs.py    (pure Python, no Isaac dependency)
      writes: data/multipleview/<name>/
        camNN/frame_XXXXX.jpg
        sparse_/ (COLMAP binaries from GT poses)
        points3D_multipleview.ply + points3D_labels.npy
        poses_bounds_multipleview.npy
        gt_segmentation.npz
        scene_scale.json (mm-per-unit for physical reporting)

Option B — pump inside factory hall (grid experiments):
  → gen_scenes.py      (parametric grid: base_amp_mm × multiplier)
      each cell: new animated USD with metallic materials, composed into Factory.usd
      writes: cell-specific capture YAML + motion.json (peak amplitudes per part)
  → omni_capture.py    (same rig, but factory hall provides natural background clutter)
  → omni_to_4dgs.py    (same as above)

Then:
  → run_grid_4dgs.py   (orchestrator DAG: convert → train → render)
      3 grid cells (A20mm_M2, A20mm_M4, A40mm_M8) + 4 Gaussian-count sweep variants (10k–100k)
  → run_grid_seg.py    (orchestrator DAG: seg_extract → segment → seg_eval)
      supports --impl rigid / mbs / rigid2 / kabsch / rigid2_roi / mask_lift_oracle
```

### 8.3 Parametric scene generation (`gen_scenes.py`)

The grid generator is the workhorse for controlled experiments. Each cell is defined by:

- `base_amp_mm` — nominal motion amplitude; every movable part gets translation drawn from [0.5A, 1.5A] mm and rotational surface displacement from [0.25A, 0.75A] mm.
- `multiplier` — one seeded "amplified" part has its amplitudes multiplied by M, creating a ground-truth target for motion-amplification methods to recover.
- Shared RNG seed across all cells — directions, frequencies, phases are identical; only the amplitude scale differs, making the grid strictly comparable.
- Metallic materials with distinct saturated colors (HSV-spaced) for visual debug.
- Factory hall environment for background realism.

Current grid: 3 amplitudes × 3 multipliers = 9 cells, though only the 3 cells with largest motion were trained and evaluated (A20mm_M2, A20mm_M4, A40mm_M8). The sweep family fixes the scene and varies the initial Gaussian count (10k/25k/50k/100k) with densification frozen.

### 8.4 What the pipeline already catches (bugs found and fixed)

The pipeline has been battle-tested and several subtle bugs were caught only because synthetic GT exists:

- **NaN loss at coarse→fine boundary** — raw Omniverse stage-unit camera translations gave `cameras_extent ≈ 4898`, blowing up learning rates. Fixed by rescaling through `meters_per_unit` to target radius 4.0.
- **All-zero GT labels** — point-cloud sampler used mesh prim name (always `"mesh"`); fixed to use parent Xform name.
- **Erratic test-camera video** — LLFF `get_spiral` is wrong for inward-looking rings; replaced with `get_orbit()` circular path.
- **Drive-frequency auto-detection failure** — FFT peak picking on noisy trajectories sometimes reports 1 cycle/clip instead of the true ~10 cycles; the synthetic motion.json provides the ground-truth frequency for validation.
- **Frame discovery found 0 frames** — Replicator nests output in `camNN/rgb/`; converter now prefers that subfolder.

### 8.5 Evaluation harness

`run_grid_seg.py` is the single entry point for benchmarking any segmentation method against the full grid+sweep:

```bash
.venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl rigid2       # T18
.venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl mask_lift_oracle  # T22
```

It is idempotent (skips already-evaluated runs), backs up previous results before overwriting, and appends one CSV row per run. This made it possible to run six different methods (rigid, mbs, rigid2, kabsch, rigid2_roi, mask_lift_oracle) over the same 7 models without manual bookkeeping.

---

## 9. How the synthetic pipeline can be improved

The current pipeline was sufficient to reach a rigorous negative result, but several upgrades would strengthen it for future work — especially if the goal shifts from "diagnose why segmentation fails" to "find conditions under which it succeeds."

### 9.1 More cameras and denser viewpoints

**Current:** 10 static dome cameras (3 rings, 15°–75° elevation).  
**Limitation:** 10 views × 60 frames = 600 images. For a ~1 m³ machine with 107 parts, this is sparse. Occlusions between parts are unresolved, and small motions project to sub-pixel displacements in many views.  
**Improvement:**
- Increase to **20–30 cameras** on a denser dome or orbital gantry.
- Add a **turntable or drone-like orbital video path** (100+ frames around the subject) to provide continuous parallax. 4DGS benefits enormously from view diversity; the current static rig gives identical baselines for every time step.
- Capture **stereo pairs** or known-baseline arrays to give direct depth constraints.

### 9.2 Higher resolution and temporal sampling

**Current:** 1600 × 900 @ 60 frames (2.5 s clip).  
**Limitation:** mm-scale motion on a ~1 m object is ~0.1% of the image height. At 900 px, this is ~1 px peak displacement — below the threshold where optical-flow or photometric gradients give strong signals. 4DGS's deformation network learns from these weak gradients, and jitter dominates.  
**Improvement:**
- **4K capture** (3840 × 2160) would increase the signal-to-jitter ratio by ~2–4×.
- **Higher frame rate** — 120 fps or 240 fps would give more temporal samples for the same physical motion, letting the deformation network average out noise. The FFT denoising in T18 would also benefit (narrower frequency bins, better harmonic separation).
- **Longer clips** — 120–240 frames would improve both FFT frequency resolution and Kabsch EM's temporal pooling (std shrinks as 1/√T).

### 9.3 Richer ground truth: per-frame transforms and flow

**Current GT:** Canonical part labels only (`gt_segmentation.npz`). Per-part motion parameters are in `motion.json` but not exposed as per-frame per-Gaussian flow vectors.  
**Improvement:**
- Emit **per-frame per-part SE(3) transforms** directly into the converted dataset. This would enable:
  - Supervised training of the deformation network (direct transform loss instead of pure photometric).
  - Ground-truth scene flow for evaluating reconstruction accuracy independently of segmentation.
  - A "cheating" baseline: if we initialize Gaussians with GT correspondence and supervise with GT transforms, what is the best possible segmentation ARI? This would separate "reconstruction error" from "clustering error."
- Emit **per-frame depth maps** alongside RGB. Depth is already rendered by Isaac Sim for mask lifting (T22); persisting it would enable depth-supervised 4DGS variants and cleaner mask-lifting without re-rendering.

### 9.4 Controlled scene complexity variants

**Current:** One asset (107-part pump) with one motion pattern (sinusoidal, all parts same frequency).  
**Improvement:**
- **Simpler scenes** — 5-part, 10-part, 20-part variants. If segmentation succeeds at 10 parts but fails at 107, we can identify the complexity knee. This is the most direct way to test the hypothesis that the problem is reconstruction quality, not algorithm design.
- **Larger motion amplitudes** — the current grid tops out at A40mm_M8 (peak ~50 mm for the amplified part). Scenes with cm-scale motion would test whether the AUROC threshold (≥0.8) is reachable when signal clearly exceeds jitter.
- **Different motion types** — non-sinusoidal (impacts, ramps, drift) to test whether FFT-based methods fail on non-periodic motion while other methods survive.
- **Transparent/occluding parts** — glass panels, rotating shafts that hide inner parts. The current pump is opaque and mostly external.

### 9.5 Material and lighting variations

**Current:** Metallic materials with distinct colors (grid) or pastel displayColor (original pump01). Factory hall has 18 sphere lights.  
**Improvement:**
- **Texture-rich materials** — the current metallic surfaces are feature-poor (specular highlights only). Adding textured labels, rust, or manufacturer stickers would give 4DGS stronger photometric gradients and more stable feature tracks.
- **Varying lighting** — multiple capture passes with different dome intensities or light positions would test robustness and could enable photometric stereo cues.
- **Motion blur** — the current 16 subframe path-trace is effectively sharp. Adding motion blur proportional to velocity would match real high-speed camera captures and test whether 4DGS's deformation field can model it.

### 9.6 Direct deformation-field supervision (the strongest intervention)

The central finding is that 4DGS reconstruction jitter ≈ true motion. The synthetic pipeline can generate the ground-truth deformation field (per-Gaussian displacement at every frame). Using this as an auxiliary loss during training — a **supervised deformation loss** alongside photometric loss — would directly attack the root cause:

```
L_total = L_photometric + λ * || δ_network(t) − δ_GT(t) ||²
```

If even supervised deformation does not yield AUROC ≥ 0.8, then the problem is not optimization noise but representational (the deformation network cannot express per-part rigid motion at 10⁵ Gaussian resolution). If it does succeed, then the path forward is clear: better reconstruction, not better clustering.

This experiment is only possible because the synthetic pipeline can author and export exact per-frame transforms. It is the single highest-impact improvement for future work.

### 9.7 Automated grid expansion

**Current:** Grid is 3×3, only 3 cells trained. Sweep is 4 counts.  
**Improvement:**
- Extend `gen_scenes.py` to sweep over camera count (5, 10, 20, 40), resolution (900p, 1080p, 4K), frame count (60, 120, 240), and motion amplitude (0.5 mm to 100 mm) in a fully automated batch.
- The orchestrator's DAG system already supports this — one YAML preset change per axis, then `run_grid_4dgs.py` and `run_grid_seg.py` handle the rest.
- The separability AUROC diagnostic (T18) provides an automatic stopping rule: once AUROC ≥ 0.8 is reached for some (cameras, resolution, amplitude) combination, that cell is the "success condition" and the thesis segmentation arc could be reopened.

### 9.8 Summary: synthetic data as a scientific instrument

The synthetic pipeline is not merely a data source — it is a **controlled experimental instrument**. Every parameter (camera count, resolution, motion amplitude, part count, material, lighting) is an independent variable that can be varied while holding others constant. The quantitative evaluation harness (ARI, IoU, AUROC, BIC) turns subjective "it looks broken" into falsifiable hypotheses. Future work should exploit this control to find the boundary conditions under which 4DGS motion segmentation succeeds, rather than treating the pump01 result as the final word.

---

## 10. Attachment manifest — what to send a research agent

### 10.1 Core briefing (send first)

| # | File | Why it matters |
|---|------|----------------|
| 1 | **`docs/motion-segmentation-research-summary.md`** | **This document.** Self-contained briefing with problem, methods, results, conclusion, pipeline description, and improvement directions. |
| 2 | **`docs/motion-segmentation.md`** | Deeper design discussion of the segmentation approach, including Option A/B trade-offs and the synthetic-data rationale. Complements the summary. |
| 3 | **`docs/overview.md`** | Project-wide overview: repo map, technology stack, current status. Useful for agent orientation. |

### 10.2 Related-work papers (PDFs)

| # | File | Citation | Relevance |
|---|------|----------|-----------|
| 4 | **`papers/Huang_MultiBodySync_Multi-Body_Segmentation_and_Motion_Estimation_via_3D_Scan_Synchronization_CVPR_2021_paper.pdf`** | Dong et al., CVPR 2021 | The learning-based baseline we tried (Option A). MotNet is the learned affinity head. Agent needs this to understand what was adapted and why it failed. |
| 5 | **`papers/2310.08528v3.pdf`** | Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" | The upstream 4DGS method this project is built on. Agent needs this to understand the deformation-field architecture and the source of reconstruction jitter. |

### 10.3 Raw results (CSVs — the evidence)

All files live in `runs/`. Each is ~0.5–1 KB, 7 rows + header. Backups of pre-T22 runs are in `runs/backups_pre_t22/`.

| # | File | Contents |
|---|------|----------|
| 6 | **`runs/grid_seg_results.csv`** | Baseline Option B (rigidity graph) — ARI, n_pred per run |
| 7 | **`runs/grid_seg_mbs_results.csv`** | Option A (MultiBodySync MotNet) — ARI, n_pred per run |
| 8 | **`runs/grid_seg_rigid2_results.csv`** | T18 (rigid2: FFT denoising + z-scores + spectral) — ARI, AUROC, drive_freq, sigma_d |
| 9 | **`runs/grid_seg_rigid2_roi_results.csv`** | T19 (motion-gate ROI + rigid2) — global ARI, ARI-within-ROI, n_roi |
| 10 | **`runs/grid_seg_kabsch_results.csv`** | T20 (Kabsch EM) — ARI, best_k (BIC) |
| 11 | **`runs/grid_seg_mask_lift_oracle_results.csv`** | T22 (oracle mask ceiling) — global ARI, ARI-within-ROI, motion-gate comparison |
| 12 | **`runs/grid_4dgs_results.csv`** | Training meta-data: actual Gaussian count, train/render time, peak VRAM |

### 10.4 Design proposals (what was planned before implementation)

| # | File | What it describes |
|---|------|-----------------|
| 13 | **`docs/proposals/00-index.md`** | Proposal index with status (implemented / skipped / deferred). |
| 14 | **`docs/proposals/01-motion-gated-roi-masking.md`** | T19 proposal — motion-gate + rigidity-lock readmission |
| 15 | **`docs/proposals/02-multiview-mask-lifting.md`** | T22 proposal — geometric mask lifting via GT depth |
| 16 | **`docs/proposals/03-seeded-part-focused-segmentation.md`** | T23 proposal (skipped) — geodesic ball + PageRank seeds |
| 17 | **`docs/proposals/04-subspace-spectral-trajectory-clustering.md`** | T21 proposal (skipped) — PCA subspace fits |
| 18 | **`docs/proposals/05-iterative-kabsch-em.md`** | T20 proposal — Kabsch EM with BIC model selection |
| 19 | **`docs/proposals/06-multiscale-snr-multiscale.md`** | T18 proposal — FFT denoising + calibrated z-scores |
| 20 | **`docs/proposals/IMPLEMENTATION_PLAN.md`** | Master plan with decision tree (AUROC go/no-go), stage wiring, and test strategy |
| 21 | **`docs/proposals/HANDOFF_rigid2_grid_benchmark.md`** | Agent handoff doc for T18 — exact run commands, files to collect, decision rules |

### 10.5 Task specifications (implementation-level detail)

| # | File | Task |
|---|------|------|
| 22 | **`orchestrator/planning/tasks/T18-segment-rigid2-denoise-calibrate.md`** | FFT denoising, z-score calibration, spectral partition, separability diagnostic |
| 23 | **`orchestrator/planning/tasks/T20-segment-kabsch-em.md`** | Iterative Kabsch fitting, BIC model selection, adaptive sigma |
| 24 | **`orchestrator/planning/tasks/T22-multi-view-mask-lifting.md`** | Oracle mask lifting, depth rendering, ROI evaluation |

### 10.6 Working notes (chronological, detailed)

| # | File | Date | Contents |
|---|------|------|----------|
| 25 | **`.claude_notes/NOTES_T18_rigid2_benchmark_2026-08-11.md`** | 2026-08-11 | T18 grid run results, AUROC analysis, denoising vs raw comparison |
| 26 | **`.claude_notes/NOTES_T20_kabsch_em_2026-08-11.md`** | 2026-08-11 | T20 BIC analysis, sandbox verification, convergence issues |
| 27 | **`.claude_notes/NOTES_T22_mask_lifting_2026-08-12.md`** | 2026-08-12 | T22 oracle implementation, ROI predicate bug, mask lift depth fix |
| 28 | **`.claude_notes/NOTES_T22_oracle_results_2026-08-12.md`** | 2026-08-12 | Final T22 results table, decision scenario C, code fixes summary |

### 10.7 Key code files (for method details)

**Segmentation implementations (orchestrator stages — what actually ran):**

| # | File | What it implements |
|---|------|-------------------|
| 29 | **`orchestrator/pipeline/stages/segment_rigid2.py`** | T18: FFT denoising, noise-floor calibration, z-score thresholding, hybrid partition |
| 30 | **`orchestrator/pipeline/stages/segment_kabsch.py`** | T20: FPS subsample, FFT fingerprint init, EM loop, BIC scoring |
| 31 | **`orchestrator/pipeline/stages/roi_mask_oracle.py`** | T22: GT-label projection to 2D masks, depth-based 3D lift |
| 32 | **`orchestrator/pipeline/stages/roi_motion_gate.py`** | T19: Band-limited energy gate + k-NN dilation |
| 33 | **`orchestrator/pipeline/stages/segment_mbs.py`** | Option A: MotNet inference wrapper (exact flow → affinity → spectral) |
| 34 | **`orchestrator/pipeline/stages/segment_rigid.py`** | Option B: Rigidity-graph clustering (k-NN + Otsu + connected components) |

**Reference scripts (original implementations, for understanding):**

| # | File | What it implements |
|---|------|-------------------|
| 35 | **`motion-seg/motion_seg/segment_rigid.py`** | Original CPU rigidity-graph clustering (self-test included) |
| 36 | **`motion-seg/motion_seg/mbs_infer.py`** | Original MultiBodySync adapter (exact flow → MotNet) |
| 37 | **`motion-seg/motion_seg/extract_trajectories.py`** | GPU trajectory extraction from 4DGS deformation field |

**Motion amplification (thesis core contribution):**

| # | File | What it implements |
|---|------|-------------------|
| 38 | **`core/render_amp.py`** | Main motion-amplified rendering script (Eulerian / absolute / segmented) |
| 39 | **`core/motion_amp/renderer.py`** | Low-level helper: returns pre-rasterization Gaussian params per frame |

**Synthetic pipeline:**

| # | File | What it implements |
|---|------|-------------------|
| 40 | **`scene-gen/gen_scenes.py`** | Parametric grid generator (amplitude × multiplier, metallic materials, factory composition) |
| 41 | **`scene-gen/run_grid_4dgs.py`** | Batch training harness (grid + sweep, idempotent, CSV logging) |
| 42 | **`scene-gen/run_grid_seg.py`** | Batch segmentation harness (all --impl variants, idempotent, CSV + PLY output) |

### 10.8 Architecture and pipeline docs

| # | File | Purpose |
|---|------|---------|
| 43 | **`docs/omniverse-pipeline.md`** | Synthetic data pipeline: Isaac Sim capture, USD prep, omni_to_4dgs conversion |
| 44 | **`docs/orchestrator.md`** | Orchestrator architecture: DAG scheduler, stages, containers, MCP server |
| 45 | **`orchestrator/planning/ARCHITECTURE.md`** | Deep dive into config presets, artifact manifest, stage registry, container bridge |
| 46 | **`AGENTS.md`** | Repo-level onboarding for AI agents (stack, layout, conventions, gotchas) |

### 10.9 How to use this manifest

**Minimum viable package** (if file-size limited):  
Send files 1–5 (briefing + 2 PDFs) + files 6–11 (all CSVs) + file 20 (`IMPLEMENTATION_PLAN.md`). This gives the agent the full story and all quantitative evidence in under 10 MB.

**Recommended package** (if the agent can read code):  
Add files 29–34 (orchestrator stage implementations) so the agent can inspect the actual algorithms without reverse-engineering from prose.

**Full package** (for a deep-dive research collaboration):  
Everything above, plus the working notes (25–28) for chronological context and the task specs (22–24) for design intent vs. implementation deltas.

---

> **Total lines:** 409 → ~500 after this appendix.  
> **Prepared for:** Research-agent consumption.  
> **Author:** Bartosz Moczkowski, MSc thesis, TUL.  
> **Date:** 2026-08-12
