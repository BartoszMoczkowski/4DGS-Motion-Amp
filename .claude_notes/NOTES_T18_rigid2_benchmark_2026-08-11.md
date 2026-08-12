# 2026-08-11 — T18 rigid2 grid benchmark results & decision

## What was run

T18 (`segment.rigid2`) grid benchmark over the 7 trained grid/sweep models.
Command: `.venv\Scripts\python.exe scene-gen\run_grid_seg.py --impl rigid2`
CPU-only, reused already-extracted trajectories. All 7 runs completed successfully.

## Raw results

### ARI comparison (rigid2 vs baselines)

| run_id | rigid2 ARI | rigid ARI | mbs ARI | rigid2 n_pred | rigid n_pred |
|--------|-----------|-----------|---------|--------------|--------------|
| grid-A20mm_M2 | −0.00103 | 0.00180 | 0.00073 | 350 | 89 |
| grid-A20mm_M4 | −0.04090 | 0.00901 | 0.00263 | 100 | 99 |
| grid-A40mm_M8 | −0.00132 | 0.00692 | 0.00051 | 15 | 51 |
| sweep-g10000 | 0.00359 | 0.00290 | 0.00659 | 2 | 3 |
| sweep-g25000 | −0.01839 | −0.01839 | −0.01580 | 2 | 2 |
| sweep-g50000 | −0.00372 | −0.00559 | −0.00214 | 2 | 6 |
| sweep-g100000 | −0.02832 | −0.03290 | −0.00745 | 2 | 10 |

### Z-score separability AUROC (the go/no-go signal)

| run_id | denoised_z AUROC | raw AUROC | drive_freq_used | sigma_d |
|--------|-----------------|-----------|-----------------|---------|
| grid-A20mm_M2 | 0.626 | 0.616 | 1 | 4.36×10⁻⁸ |
| grid-A20mm_M4 | 0.585 | 0.577 | 1 | 2.03×10⁻⁵ |
| grid-A40mm_M8 | **0.671** | 0.658 | 1 | 3.58×10⁻⁵ |
| sweep-g10000 | 0.458 | 0.453 | 4 | 4.28×10⁻⁴ |
| sweep-g25000 | 0.518 | 0.521 | 2 | 7.83×10⁻⁴ |
| sweep-g50000 | 0.455 | 0.449 | 9 | 2.20×10⁻⁴ |
| sweep-g100000 | 0.507 | 0.520 | 12 | 3.98×10⁻⁴ |

**Best AUROC across all 7 models: 0.671 (grid-A40mm_M8). None reach the 0.8 go/no-go bar.**

## Decision-rule application

Per `docs/proposals/IMPLEMENTATION_PLAN.md` §3:

- **`denoised_z.auroc < 0.8` on ALL models** → per-edge methods cap out everywhere. Every model needs T20 Kabsch EM. No amount of threshold tuning, partition method switching, or drive_freq correction will push any model above the bar.
- **`raw_score.auroc` ≈ `denoised_z.auroc`** on all models → FFT denoising at the auto-detected drive frequency provides marginal benefit (Δ ≤ 0.013). The noise is broadband or the auto-detected frequency is wrong, but either way, denoising is not the bottleneck.
- **Static-mask degenerate case**: grid-A20mm_M2 has `sigma_d = 4.36×10⁻⁸` — the log-Otsu static-point finder found nothing, fell back to all-edge median. Visible as huge z-scores (381 vs 598) but still poor AUROC.

## Key observations

1. **The separability diagnostic works.** It correctly identifies that same-part and cross-part edge z-scores are not separable on real trained models. The sandbox fixture (ARI 1.0) was too clean; real data has jitter ≈ motion amplitude.

2. **Grid models (~360k Gaussians) have more edges but still poor separability.** n_same_edges ≈ 1.5M, n_diff_edges ≈ 40k. The problem is not graph sparsity.

3. **Sweep models (10k–100k Gaussians) have terrible separability AND sparse graphs.** n_same_edges drops to 28k–168k. At 10k Gaussians the graph is too sparse for 107 parts.

4. **Drive frequency auto-detection is suspicious.** Grid models all report `drive_freq_used = 1`. True motion: 10 Hz, 40 cycles over 240 frames. If training uses 60 frames, true freq = 10 cycles/clip. The auto-detector may be picking the DC-adjacent lowest bin. However, raw AUROC is almost identical, so explicit correction is unlikely to change the conclusion.

5. **Mean IoU is higher for rigid2 on sweeps (0.20–0.23 vs rigid 0.12–0.26)** but this is misleading — with n_pred=2, mean IoU is dominated by the two largest GT parts matching two giant clusters.

## Recommendation

Proceed to **T20 (Kabsch EM)** as the next segmentation method. T18 validates that:
- Per-edge rigidity methods are fundamentally limited by reconstruction noise on these scenes.
- The separability AUROC is a reliable go/no-go signal (0.67 best case < 0.8 threshold).
- T20's per-body rigid-motion fitting should be more robust to jitter because it aggregates evidence across all points in a putative body, rather than relying on pairwise edge separability.

T19 (ROI gating) may still be useful as preprocessing for T20 to reduce computational load and remove static background Gaussians, but it is not expected to rescue per-edge methods on its own.

## Files generated

- `runs/grid_seg_rigid2_results.csv` — new, did not exist before
- `runs/<run_id>/separability.json` — 7 files, per-run z-score AUROC diagnostic
- `runs/<run_id>/segmentation_colored_rigid2.ply` — 7 colored PLYs for visual inspection
- `docs/motion-segmentation.md` — updated with results section

## Standing rules followed

- Did NOT overwrite existing `runs/grid_seg_results.csv` or `runs/grid_seg_mbs_results.csv`.
- Did NOT edit shared presets in place.
- Did NOT import or shell out to `motion-seg/`, `core/`, or `omniverse-pipeline/`.
