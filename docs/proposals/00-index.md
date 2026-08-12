# Motion segmentation rescue proposals — index

Status of the existing methods (from `runs/grid_seg_results.csv`, `runs/grid_seg_mbs_results.csv`, and `docs/motion-segmentation.md`):

- **Option B** (`segment_rigid.py`, rigidity graph): ARI ≈ 0.002–0.009 on the real grid models vs 107 GT parts. Works only on the noiseless self-test (ARI 0.9988).
- **Option A** (`mbs_infer.py`, MotNet): ARI ≈ 0 on the same models — MotNet is out-of-distribution for mm-scale 4DGS trajectories.
- **Root blocker:** the trained models' frame-to-frame position noise is comparable to the true mm-scale motion (median edge score 0.00125 vs p90 0.0035). Any method that thresholds raw per-edge rigidity scores will fail; solutions must either **shrink the problem** (fewer, cleaner points) or **average out the noise** (more evidence per decision).

The proposals below are organized by the three requested focus areas. Each file is self-contained with the mathematical derivation and an integration plan against `motion-seg/motion_seg/`.

## Focus 1 — restrict analysis to the machine, not the background

| # | File | Idea |
|---|------|------|
| 01 | [Motion-gated ROI masking](01-motion-gated-roi-masking.md) | Static/dynamic decomposition from trajectory energy; run segmentation only on the moving subset; graph morphology to close the ROI. |
| 02 | [Multi-view mask lifting](02-multiview-mask-lifting.md) | Lift 2D machine masks (SAM / background plate) through the 10 calibrated cameras into per-Gaussian foreground votes. |

## Focus 2 — focus on a specific part

| # | File | Idea |
|---|------|------|
| 03 | [Seeded part-focused segmentation](03-seeded-part-focused-segmentation.md) | Seed from a GT part / click, grow geodesically on the k-NN graph with heat-diffusion weights; local high-resolution rigidity only inside the seeded ball. |

## Focus 3 — cheaper methods inspired by MBS

| # | File | Idea |
|---|------|------|
| 04 | [Subspace/spectral trajectory clustering](04-subspace-spectral-trajectory-clustering.md) | Classic multibody motion segmentation: rank-≤4 motion subspaces, local subspace affinity, spectral clustering — no learned network. |
| 05 | [Iterative Kabsch EM](05-iterative-kabsch-em.md) | Replace MotNet affinity with analytic SE(3) Kabsch residuals inside an EM loop; soft assignments average out trajectory noise. |
| 06 | [Multiscale subsample-and-propagate + SNR-aware scoring](06-multiscale-snr-multiscale.md) | Denoise trajectories before scoring (FFT low-pass, exploiting the known periodic drive), weight edges by rigidity SNR, segment an FPS subsample, propagate labels by k-NN. |

## Recommended order of attack

1. **06** (denoising + SNR weighting) — cheapest to bolt onto the existing Option B and directly attacks the measured noise floor; it also benefits 01/03/05.
2. **01** (motion-gated ROI) — pure numpy, immediate reduction of N and of spurious background edges.
3. **05** (Kabsch EM) — the principled cheap-MBS core; soft assignments are exactly what a noise-dominated regime needs.
4. **02 / 03** — require more plumbing (rendering/projection or UI seeding) but exploit information unique to this setup (calibrated cameras, GT init cloud).
5. **04** — strong classical baseline, useful as a sanity cross-check against 05.
