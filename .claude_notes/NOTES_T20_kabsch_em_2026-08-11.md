# 2026-08-11 — T20 Kabsch EM implementation & grid benchmark

## What was implemented

T20 (`segment.kabsch`) — Kabsch EM rigid-body clustering (proposal 05):

- `orchestrator/pipeline/vendored/host/kabsch_em.py` — Core algorithm:
  - Weighted Kabsch (batch SVD over T frames)
  - E-step: soft responsibilities with adaptive sigma annealing
  - M-step: per-body per-frame weighted Kabsch (vectorised over T)
  - FFT-fingerprint initialisation (`motion_fingerprint` from T18)
  - BIC model selection over k_range
  - Greedy split (BIC-guided)
  - FPS subsample + q-NN propagation
- `orchestrator/pipeline/stages/segment_kabsch.py` — Stage adapter (same I/O contract)
- `orchestrator/pipeline/config/models.py` — `SegmentKabschConfig` added
- `orchestrator/pipeline/config/presets/pump01_kabsch.yaml` — Preset (fps_subsample=5000, max_iter=15)
- `orchestrator/pipeline/stages/__init__.py` — Import registered
- `scene-gen/run_grid_seg.py` — `--impl kabsch` support

## Sandbox verification

- `test_segment_kabsch.py` — 8 tests (2 unit + 6 integration)
- `test_kabsch_minimal.py` / `test_kabsch_all.py` — standalone validation scripts
- Synthetic 7-body fixture: EM converges in 4 iterations, ARI ≥ 0.999
- `weighted_kabsch` correctly recovers known rotation + translation

## Grid benchmark results

Command: `.venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl kabsch`

| run_id | kabsch ARI | rigid ARI | rigid2 ARI | kabsch n_pred |
|--------|-----------|-----------|------------|--------------|
| grid-A20mm_M2 | 0.0191 | 0.00180 | −0.00103 | 3 |
| grid-A20mm_M4 | −0.0157 | 0.00901 | −0.04090 | 3 |
| grid-A40mm_M8 | −0.0089 | 0.00692 | −0.00132 | 3 |
| sweep-g10000 | 0.0176 | 0.00290 | 0.00359 | 30 |
| sweep-g25000 | −0.0334 | −0.01839 | −0.01839 | 97 |
| sweep-g50000 | −0.0049 | −0.00559 | −0.00372 | 30 |
| sweep-g100000 | −0.0456 | −0.03290 | −0.02832 | 89 |

## Key finding: BIC says 107 parts are not resolvable

On grid-A20mm_M2, BIC search over [20, 30, 45, 67, 100, 150]:
- BIC(20) = 61494 ← minimum
- BIC(30) = 92241
- BIC(45) = 138362
- BIC(67) = 206006
- BIC(100) = 307471
- BIC(150) = 461206

BIC increases monotonically with K. The data itself says only ~20 motion groups are statistically justified at this noise level. 107 GT parts are beyond the reconstruction-quality ceiling.

## Interpretation

Both T18 (per-edge) and T20 (per-body EM) fail on real data for the same root cause: reconstruction jitter ≈ true mm-scale motion. The separability diagnostic correctly predicted this.

Kabsch EM is mathematically sound (sandbox ARI 0.999+) but the noise floor is too high to resolve 107 parts, even with:
- 60-frame trajectory pooling (theoretical 7× tighter than per-frame)
- Adaptive sigma annealing
- FFT-fingerprint initialisation
- BIC model selection

## Next steps

Per `IMPLEMENTATION_PLAN.md`:
1. **T19 ROI gating** — try removing static background before clustering (may help by reducing N and eliminating the largest uniform cloud)
2. **T22 multi-view mask lifting** — test whether 2D→3D mask priors break through the motion-only ceiling
3. If both fail: thesis conclusion shifts to "motion-only segmentation is reconstruction-quality-limited at mm scale; the practical contribution is the motion-amplification pipeline + synthetic-data framework"

## Files generated

- `runs/grid_seg_kabsch_results.csv` — 7 rows
- `runs/<run_id>/segmentation_colored_kabsch.ply` — 7 colored PLYs
- `orchestrator/tests/test_segment_kabsch.py` — sandbox tests
- `docs/motion-segmentation.md` — updated with T20 results
