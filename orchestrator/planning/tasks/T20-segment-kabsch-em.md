# T20 — Kabsch EM rigid-body clustering (`segment.kabsch`)

- Status: done (2026-08-11) — implemented, sandbox-verified, grid benchmark run
- Phase: 7 (segmentation rescue — `docs/proposals/IMPLEMENTATION_PLAN.md`)
- Depends on: T18
- Environment: host (pure numpy/scipy)

## Goal

Add proposal-05's iterative Kabsch EM as a fourth impl under the `segment` role:
E-step soft-assigns each Gaussian to the body whose rigid motion best explains its full
T-frame trajectory; M-step re-fits each body's motion by weighted Kabsch per frame.
Pools evidence across all T frames (std(r²)/E[r²] ≈ √(2/(3T)) ~ 7× tighter than per-frame
for T=60), making it more robust to the reconstruction jitter that caps per-edge methods.

## In scope

- `pipeline/vendored/host/kabsch_em.py` — `weighted_kabsch`, `_compute_residuals`,
  `_e_step` (soft responsibilities with adaptive sigma annealing), `_m_step` (batch SVD
  over T frames), `_init_fft` (FFT-fingerprint + k-means++), `_init_kmeans_spatial`,
  `_em_single` (fixed-K EM with annealing), `_greedy_split` (BIC-guided), `segment_by_kabsch`
  (full pipeline with BIC model selection, FPS subsample + propagate)
- `pipeline/stages/segment_kabsch.py` — `segment.kabsch`, host env, same I/O contract
- `SegmentKabschConfig` in `pipeline/config/models.py`
- Preset `pump01_kabsch.yaml`
- `scene-gen/run_grid_seg.py --impl kabsch`

## Out of scope

Spatial prior / Potts smoothness (future T19+T23); GPU acceleration.

## Verification (sandbox, no GPU)

`orchestrator/tests/test_segment_kabsch.py` — 8 tests: identity/known-transform Kabsch,
EM convergence, full pipeline ARI ≥ 0.95 on noisy 7-body fixture, BIC preference for correct K,
greedy split, FPS subsample path, and end-to-end DAG run through `segment.kabsch` +
`seg_eval.default`.

## Real-hardware grid benchmark

Run: `.venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl kabsch`

Results: marginal improvement over baselines (best ARI 0.019 vs 0.0018 rigid baseline), but
BIC model selection shows the data only supports ~20 motion groups, not 107 GT parts.
Reconstruction jitter ≈ true motion amplitude is the fundamental ceiling.

## Log (2026-08-11)

- First implementation used wrong einsum in batch Kabsch (V·U instead of V·Uᵀ), causing
  EM to fail to converge. Fixed by using explicit transpose + matrix multiply.
- Second issue: sigma too small caused E-step to underflow to hard assignment immediately.
  Fixed with adaptive sigma annealing: start at max(sigma, 1.0), update each iteration
  via ML estimate from weighted residuals.
- EM converges in 4–15 iterations on the fixture. Sandbox ARI ≥ 0.999.
- Real data: Kabsch EM does not qualitatively rescue segmentation. BIC increases
  monotonically with K, indicating only ~20 motion groups are statistically justified.
- Next: T19 ROI gating or T22 mask lifting to break through the motion-only ceiling.
