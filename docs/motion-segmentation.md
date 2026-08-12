# Motion segmentation: adapting MultiBodySync to 4DGS

Compiled from `.claude_notes/NOTES_4dgs_motion_segmentation.md` (full detail there).

## The core insight

MultiBodySync (MBS) solves a problem 4DGS doesn't have. MBS's front two-thirds (FlowNet scene-flow estimation + confidence + weighted permutation synchronization) exist to establish correspondence between **unordered, uncorresponded point-cloud scans**. In 4DGS, the scene is one canonical set of N Gaussians plus a deformation field `D(x, t)` — the same Gaussian keeps its identity at every timestep, so dense per-point trajectories `p_i(t)` come for free at arbitrary temporal resolution. The task collapses to **clustering known trajectories into rigid motion groups**. The only genuinely reusable part of MBS is its motion-segmentation core (MotNet affinity + spectral segmentation synchronization + per-group Kabsch fitting).

Key mismatches beyond correspondence: MBS assumes piecewise-rigid bodies and N ≈ 256–1024 points (its permutation sync is O((KN)²) — infeasible at 4DGS's N > 10⁵); 4DGS scenes have floaters/low-opacity noise and continuous, possibly non-rigid deformation.

## Adaptation options

- **Option A — reuse MBS's MotNet head, drop flow + permutation sync.** Feed exact analytic flow (`p(t_l) − p(t_k)`) into MotNet with identity permutations. Medium effort; carries the rigid assumption, needs the `ext/` CUDA ops and a downloaded checkpoint, with a known out-of-distribution risk (MotNet was trained on noisy FlowNet flow at unit scale).
- **Option B — lightweight trajectory clustering, MBS as reference only.** Pure numpy/scipy, no GPU for the clustering itself. Chosen as the first baseline.
- **Option C — full MBS retrain end-to-end.** Highest effort, deferred.

## Implementation: `motion-seg/motion_seg/`

- `extract_trajectories.py` — data adapter (needs the GPU env: the package's `core` extra, which pulls in `4dgs-core`). Loads a trained model, samples `get_state_at_time` at T evenly-spaced times, writes `trajectories.npz` (`canonical_xyz`, `traj (N,T,3)`, `opacity`, `times`).
- `rigidity_graph.py` — Option B core (pure numpy/scipy): k-NN graph in canonical space; per-edge rigidity score = std-dev of pairwise distance over time (exactly 0 for a true rigid pair); **log-space Otsu** auto-threshold; connected components → segments; tiny components folded into nearest neighbor.
- `segment_rigid.py` — CLI wrapper (opacity-filters floaters, label −1). `--selftest` verifies on a synthetic 7-body scene with no GPU: **ARI 0.9988**.
- `mbs_infer.py` — Option A adapter: exact flow into MotNet per view pair, `compose_dense` + `sync_motion_seg` imported from MBS source, one shared FPS subsample across views (the "permutation sync is the identity" simplification), 3-NN label propagation back to the full set. Wired into the orchestrator as `segment.mbs` but **not yet run on a real GPU/checkpoint**.
- `metrics.py`, `evaluate_segmentation.py` — ARI + Hungarian IoU against `gt_segmentation.npz` (GT labels nearest-neighbor-propagated from the init cloud onto trained Gaussians), plus colored-PLY and PNG previews (`visualize.py`).
- `run.sh <scene>` — chains extract → segment → evaluate (`./motion-seg/motion_seg/run.sh pump01`); supports `SKIP_EXTRACT=1` and passthrough tuning args.

## Results so far

- Synthetic self-test: ARI 0.9988, mean Hungarian IoU 0.982.
- First real `pump01` run (2026-07-06): poor (ARI 0.05, 4–5 segments vs 107 GT). Root cause: linear-histogram Otsu failed on the heavily right-skewed real edge-score distribution — fixed by Otsu in log-space. Residual caveat: the trained model's frame-to-frame position noise is comparable to the true mm-scale motion (median edge score 0.00125 vs p90 0.0035), so segmentation quality is ultimately gated by reconstruction quality; `--threshold-mult` and `-k` are the tuning knobs.
- A separate numerical quirk: the Otsu-log threshold degenerates (ARI 0.0) on an *exactly noiseless* synthetic scene under some numpy/scipy builds — worked around via `threshold_mult` (documented in the orchestrator's vertical-slice test).

### T18 grid benchmark — `segment.rigid2` real-data run (2026-08-11)

T18 (FFT band-pass denoising + per-scene calibrated rigidity z-scores + adaptive threshold + connected-components/spectral partition) was sandbox-verified (ARI 1.0 on noisy 7-body fixture) then run on the 7 trained grid/sweep models. **Results: per-edge methods cap out on every model.**

**ARI / IoU comparison (rigid2 vs baselines)**

| run_id | rigid2 ARI | rigid ARI | mbs ARI | rigid2 n_pred | rigid n_pred | rigid2 mean_iou |
|--------|-----------|-----------|---------|--------------|--------------|-----------------|
| grid-A20mm_M2 | −0.00103 | 0.00180 | 0.00073 | 350 | 89 | 0.013 |
| grid-A20mm_M4 | −0.04090 | 0.00901 | 0.00263 | 100 | 99 | 0.006 |
| grid-A40mm_M8 | −0.00132 | 0.00692 | 0.00051 | 15 | 51 | 0.040 |
| sweep-g10000 | 0.00359 | 0.00290 | 0.00659 | 2 | 3 | 0.205 |
| sweep-g25000 | −0.01839 | −0.01839 | −0.01580 | 2 | 2 | 0.224 |
| sweep-g50000 | −0.00372 | −0.00559 | −0.00214 | 2 | 6 | 0.224 |
| sweep-g100000 | −0.02832 | −0.03290 | −0.00745 | 2 | 10 | 0.227 |

**Z-score separability diagnostic (go/no-go for per-edge methods)**

| run_id | denoised_z AUROC | raw AUROC | drive_freq_used | sigma_d | n_same_edges | n_diff_edges |
|--------|-----------------|-----------|-----------------|---------|--------------|--------------|
| grid-A20mm_M2 | 0.626 | 0.616 | 1 | 4.36×10⁻⁸ | 1 651 374 | 40 064 |
| grid-A20mm_M4 | 0.585 | 0.577 | 1 | 2.03×10⁻⁵ | 1 743 146 | 39 263 |
| grid-A40mm_M8 | 0.671 | 0.658 | 1 | 3.58×10⁻⁵ | 1 488 236 | 36 007 |
| sweep-g10000 | 0.458 | 0.453 | 4 | 4.28×10⁻⁴ | 27 657 | 7 015 |
| sweep-g25000 | 0.518 | 0.521 | 2 | 7.83×10⁻⁴ | 45 174 | 8 143 |
| sweep-g50000 | 0.455 | 0.449 | 9 | 2.20×10⁻⁴ | 109 242 | 14 694 |
| sweep-g100000 | 0.507 | 0.520 | 12 | 3.98×10⁻⁴ | 168 196 | 16 919 |

**Interpretation (per `IMPLEMENTATION_PLAN.md` §3 decision rules)**

1. **AUROC < 0.8 on every model.** The best is grid-A40mm_M8 at 0.671; the sweeps are 0.455–0.518. Per the plan, this means **per-edge methods (rigidity-graph clustering of any variant) cannot succeed on these reconstructions** — the signal-to-noise ratio at the edge level is insufficient. Recommendation for all 7 models: proceed to **T20 (Kabsch EM)**.

2. **Denoising provides only marginal benefit.** ΔAUROC (denoised − raw) is 0.001–0.013 across grid models and mixed (±0.01) on sweeps. The FFT band-pass at the auto-detected drive frequency does not fundamentally change the separability landscape.

3. **Static-mask degenerate case visible on grid-A20mm_M2:** `sigma_d = 4.36×10⁻⁸` — log-Otsu on trajectory energy found no static points, so calibration fell back to the all-edge median. This produces enormous z-scores (median same = 381, median diff = 598) but poor AUROC because same-part edge variance is still huge relative to the tiny noise floor.

4. **Drive-frequency auto-detection may be off, but fixing it won't save per-edge methods.** Grid models all report `drive_freq_used = 1` (cycles/clip) while the true motion was authored at 10 Hz with 40 cycles over 240 frames. Even if the training subsamples to 60 frames, the true drive would be ~10 cycles/clip, not 1. However, raw AUROC is similarly poor, so explicit `drive_freq` tuning is unlikely to push any model above the 0.8 bar.

5. **Fragmentation / under-fragmentation:** grid-A20mm_M2 over-fragments (350 vs 107 GT); grid-A40mm_M8 under-fragments (15); sweeps collapse to 2 clusters regardless of threshold because the k-NN graph is too sparse at low Gaussian counts.

**Bottom line:** T18's separability diagnostic is working as designed — it correctly flags that the reconstruction jitter is too large relative to the true mm-scale motion for any per-edge clustering to succeed. The next step is **T20 Kabsch EM** (iterative rigid-body fitting, proposal 05). T19 ROI gating may still be useful as preprocessing for T20 to reduce the Gaussian count and remove static background, but it is not expected to rescue per-edge methods on its own.

### T20 grid benchmark — `segment.kabsch` real-data run (2026-08-11)

T20 (iterative Kabsch EM — E-step soft assignment by trajectory residual, M-step weighted per-frame Kabsch, FFT-fingerprint init, adaptive sigma annealing, BIC model selection) was sandbox-verified (ARI 1.0 on noisy 7-body fixture with convergence in 4 iterations) then run on the 7 trained grid/sweep models. **Results: Kabsch EM does not rescue segmentation on real data either.**

**ARI / IoU comparison (kabsch vs prior methods)**

| run_id | kabsch ARI | rigid ARI | rigid2 ARI | kabsch n_pred | best_k (BIC) |
|--------|-----------|-----------|------------|--------------|-------------|
| grid-A20mm_M2 | 0.0191 | 0.00180 | −0.00103 | 3 | 20* |
| grid-A20mm_M4 | −0.0157 | 0.00901 | −0.04090 | 3 | — |
| grid-A40mm_M8 | −0.0089 | 0.00692 | −0.00132 | 3 | — |
| sweep-g10000 | 0.0176 | 0.00290 | 0.00359 | 30 | — |
| sweep-g25000 | −0.0334 | −0.01839 | −0.01839 | 97 | — |
| sweep-g50000 | −0.0049 | −0.00559 | −0.00372 | 30 | — |
| sweep-g100000 | −0.0456 | −0.03290 | −0.02832 | 89 | — |

\* BIC search on grid-A20mm_M2 (k_range=[20, 150]) found best_k=20 with BIC=61494, vs K=107 BIC=307471. BIC increases monotonically with K, indicating the data does not support 107 rigid bodies at current noise levels.

**Key findings**

1. **Kabsch EM is only marginally better than per-edge baselines.** Best ARI is 0.019 (grid-A20mm_M2), compared to rigid baseline 0.0018. This is not a qualitative improvement — all methods remain near-zero on real data.

2. **BIC model selection strongly prefers K ≈ 20 over K = 107.** On grid-A20mm_M2, BIC(20) = 61494 vs BIC(107) = 307471. The reconstruction noise is so high that the statistical evidence only supports ~20 motion groups, not the 107 GT parts. This is a fundamental reconstruction-quality ceiling, not a tuning issue.

3. **FPS subsample (5k) + fixed K=107 collapses to 3 clusters** after merge_small_components on all grid models. With 5k subsample points and 107 clusters, the average cluster has only ~47 points — below the min_size=15 threshold, but more importantly, the Kabsch fit on tiny clusters is unstable. The sweep models retain more clusters (30–97) because their lower Gaussian counts allow more stable local fits.

4. **Adaptive sigma annealing works correctly.** EM converges in 4–15 iterations. The sandbox fixture (ARI 0.999+) confirms the algorithm is mathematically correct; the failure is in the data, not the method.

**Interpretation and next steps**

Both T18 (per-edge) and T20 (per-body EM) fail on real data for the same root cause: **reconstruction jitter ≈ true mm-scale motion**. The separability diagnostic (T18) correctly predicted this. Kabsch EM pools evidence across T frames, which *should* be more robust (std(r²)/E[r²] ≈ √(2/(3T)) ≈ 0.1 for T=60), but the noise floor is still too high to resolve 107 parts.

The remaining proposals in `IMPLEMENTATION_PLAN.md`:
- **T19 ROI gating** (proposal 01) — remove static background before clustering. This would reduce N and eliminate the largest uniform cloud, which actively harms partitions. Worth trying before declaring defeat.
- **T22 multi-view mask lifting** (proposal 02) — use 2D segmentation masks lifted to 3D as spatial priors. The thesis question is whether oracle masks show a large gap over motion-only methods.
- **T21 subspace spectral** (proposal 04) — PCA + local subspace fits. Cross-check against T20; may provide better init.

If T19+T22 still fail to reach ARI ≥ 0.5 on pump01, the thesis conclusion becomes: **motion-only segmentation of 4DGS at ~10⁵ Gaussians is reconstruction-quality-limited on mm-scale industrial scenes**. The practical contribution shifts to the motion-amplification pipeline (which works) and the synthetic-data generation framework (which enables quantitative evaluation).


- Synthetic self-test: ARI 0.9988, mean Hungarian IoU 0.982.
- First real `pump01` run (2026-07-06): poor (ARI 0.05, 4–5 segments vs 107 GT). Root cause: linear-histogram Otsu failed on the heavily right-skewed real edge-score distribution — fixed by Otsu in log-space. Residual caveat: the trained model's frame-to-frame position noise is comparable to the true mm-scale motion (median edge score 0.00125 vs p90 0.0035), so segmentation quality is ultimately gated by reconstruction quality; `--threshold-mult` and `-k` are the tuning knobs.
- A separate numerical quirk: the Otsu-log threshold degenerates (ARI 0.0) on an *exactly noiseless* synthetic scene under some numpy/scipy builds — worked around via `threshold_mult` (documented in the orchestrator's vertical-slice test).
