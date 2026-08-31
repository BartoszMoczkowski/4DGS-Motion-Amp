# T22 Oracle Ceiling Benchmark — Results (2026-08-12)

Task source: `HANDOFF_T22_oracle_benchmark.md`. Benchmark: `scene-gen/run_grid_seg.py --impl mask_lift_oracle`
(stages `roi.mask_oracle` → `segment.rigid2` → `seg_eval.default`, preset `pump01_mask_oracle`).

## 1. Run summary

All 7 grid/sweep runs succeeded (`status=success`, no errors). Runtime ~3 min total (host-side oracle stage
+ rigid2 + eval; no GPU). Results appended to `runs/grid_seg_mask_lift_oracle_results.csv`.

Three code fixes were required before the benchmark could run (all sandbox-verified, 11/11 T22 tests pass):

1. **`orchestrator/pipeline/config/models.py`** — `RoiConfig.impl` literal did not include `"mask_oracle"`;
   pydantic rejected the preset. Added the literal.
2. **`orchestrator/pipeline/stages/roi_mask_oracle.py`** — the ROI predicate had a dead line followed by
   `roi_mask = mapped_labels >= 0`. Since pump01 GT label 0 is the background class (49,872 of ~86k points,
   labels 1–106 are parts), `>= 0` would mark the *entire* cloud as ROI, making the oracle meaningless.
   Fixed to `mapped_labels > 0` (matches the stage docstring and the sandbox tests). Also wrapped
   `np.load` calls in `with` blocks — open NpzFile handles caused `PermissionError` on Windows tempdir cleanup.
3. **`orchestrator/pipeline/vendored/cuda/mask_lift.py`** — module-scope `import torch` + 4DGS core imports
   made the module unimportable on the host, breaking the 6 sandbox helper tests. Deferred those imports
   into the functions that use them (matches the "light package imports" convention). No runtime behaviour
   change inside the cuda container.

Also fixed `orchestrator/tests/test_roi_mask_oracle.py`: `StageContext.run_dir` is typed `Path` but tests
passed a `str`; `StageContext`/`RoiMaskOracleStage` were only imported inside the first test. Full sandbox
suite: 224 passed; 10 failures in `test_roi_motion_gate.py` / `test_segment_kabsch.py` / `test_stages_isaac.py`
are pre-existing (numeric assertions and a missing `roi.none` registration — none touch files changed here).

## 2. Results table

| run_id | oracle global ARI | oracle ARI-within-ROI | rigid2 ARI (T18) | motion-gate ARI-within-ROI (T19) | kabsch ARI (T20) | separability AUROC (denoised) | n_roi_points | n_pred |
|--------|------------------|----------------------|------------------|----------------------------------|------------------|------------------------------|-------------|--------|
| grid-A20mm_M2 | 0.281 | 0.105 | −0.001 | 0.001 | 0.019 | 0.626 | 51,599 | 35 |
| grid-A20mm_M4 | 0.294 | −0.013 | −0.041 | −0.042 | −0.016 | 0.585 | 52,853 | 12 |
| grid-A40mm_M8 | 0.207 | 0.012 | −0.001 | −0.004 | −0.009 | 0.671 | 48,277 | 6 |
| sweep-g10000 | 0.128 | 0.005 | 0.004 | 0.003 | 0.018 | 0.458 | 4,922 | 3 |
| sweep-g25000 | 0.021 | 0.000 | −0.018 | −0.024 | −0.033 | 0.518 | 12,163 | 3 |
| sweep-g50000 | 0.056 | 0.002 | −0.004 | −0.009 | −0.005 | 0.455 | 24,137 | 3 |
| sweep-g100000 | 0.000 | −0.000 | −0.028 | −0.028 | −0.046 | 0.507 | 48,395 | 3 |

`n_roi_points` is sane everywhere (5–28% of the cloud on grid runs; scales with model size on the sweep) —
the oracle mask is neither empty nor degenerate.

## 3. Interpretation — Scenario C (reconstruction-quality-limited)

Decision rules from the handoff:

- **Oracle ARI-within-ROI is low (< 0.3) on every run** (max 0.105) **and motion-gate ARI-within-ROI is
  also low (< 0.3)** → **Scenario C: the bottleneck is segmentation itself, not ROI quality.**
- Decision 2 confirms it: oracle ARI-within-ROI ≈ rigid2 global ARI (both ≈ 0) — restricting to the true
  machine region does not help clustering at all.
- Decision 3 confirms the root cause: separability AUROC is **0.45–0.67 on all 7 models, far below the 0.8
  threshold** → per-edge rigidity methods are fundamentally capped on these reconstructions. The edge
  signal simply does not separate same-part from different-part pairs.

The one superficially positive number — oracle *global* ARI rising to 0.21–0.29 on the grid runs — is an
artifact of the label −2 convention: the oracle removes the background cloud from clustering, so the
background is scored as one big correct group. Inside the machine region (the metric that matters),
clustering still collapses to 3–35 clusters vs 107 GT parts with ARI ≈ 0.

**Caveat on scope:** the oracle measures the ceiling of *rigid2-with-perfect-ROI*. Strictly, a
hypothetically better clustering algorithm might do more within a perfect ROI — but combined with T18
(rigid2 fails), T20 (Kabsch EM fails), and the sub-0.7 AUROC on every model, the evidence that the
trajectory signal itself is the cap is strong.

## 4. Recommendation

**Do not proceed to mask generation (clean-plate / SAM) for segmentation purposes.** A perfect ROI already
fails to yield meaningful part segmentation, so real (imperfect) lifted masks cannot beat it — T22 as a
segmentation rescue is dead. The thesis conclusion stands:

> Motion-only segmentation of 4DGS at ~10⁵ Gaussians is reconstruction-quality-limited on mm-scale
> industrial scenes (pump01: 107 parts, ~mm-scale periodic motion, reconstruction jitter comparable to
> true motion amplitude).

Practical contributions shift to the motion-amplification pipeline (works) and the synthetic-data
generation + evaluation framework (enables the quantitative verdict above). The `roi.mask_oracle` stage
remains useful as a diagnostic ceiling for any future segmentation method.

## 5. Code fixes needed / made

All fixes listed in §1 were made and sandbox-verified during this run. The depth-sign issue flagged in the
handoff for `mask_lift.py` was **not exercised** (oracle mode does not render depth); it remains a watch
item if `roi.mask_lift` is ever run for non-segmentation purposes.
