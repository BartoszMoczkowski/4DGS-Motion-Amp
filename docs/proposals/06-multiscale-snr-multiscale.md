# Proposal 06 — SNR-aware rigidity scoring, trajectory denoising, and multiscale subsample-and-propagate

**Focus area:** 3 — cheaper methods inspired by MBS (and the direct fix for the measured noise floor). This is the cheapest proposal to implement and a force multiplier for 01/03/04/05.

## Motivation

`docs/motion-segmentation.md` records the core blocker: median edge rigidity score 0.00125 vs p90 0.0035 — the noise floor rivals the true motion. But that statistic throws away two things we know:

1. **The motion is periodic with a known drive frequency** (we authored it in `gen_scenes.py`; the FFT machinery in `render_amp.py` already exploits this). Reconstruction jitter is approximately white; the true motion is narrowband. Signal processing separates them.
2. **A per-edge decision is the weakest possible evidence unit.** MBS-style synchronization works because it averages many weak affinities; we can get the same averaging analytically.

Additionally, both MBS (FPS subsample + propagation) and our own `mbs_infer.py` already concede that $N = 10^5$ is unnecessary: segment a clean subsample, propagate labels back. Done carefully, this both accelerates and *denoises* (subsampled trajectories get spatially averaged).

## Mathematical formulation

### 1. Trajectory denoising by band-passing

Each trajectory is periodic up to reconstruction noise: $p_i(t) = \bar p_i + s_i(t) + \epsilon_i(t)$ with $s_i$ supported at the drive frequency $\omega_0$ and its harmonics, $\epsilon_i$ approximately white with per-axis variance $\sigma^2$. DFT-filter:

$$
\tilde p_i(\omega) = \hat p_i(\omega)\cdot \mathbb{1}\big[\, \omega \in \{\omega_0, 2\omega_0, \dots, H\omega_0\} \cup \{0\} \,\big],
\qquad
\tilde p_i = \mathcal{F}^{-1}\{\tilde p_i(\omega)\}.
$$

Noise suppression factor: keeping $H+1$ of $T$ bins removes a fraction $1 - (H+1)/T$ of noise power. For $T=60$, $H=3$: **93% of noise power gone, 0% of periodic signal gone** — the effective SNR improves by ~11 dB before any clustering decision is made. (If the motion is not exactly periodic in the window, widen each kept bin to a small band; estimate $\omega_0$ as the argmax of the mean power spectrum over high-energy points, or read it from the scene preset.)

### 2. Per-edge rigidity SNR (replaces raw std thresholding)

For edge $(i,j)$ with relative distance series $d_{ij}(t) = \lVert p_i(t) - p_j(t) \rVert$, Option B thresholds $\mathrm{std}_t\, d_{ij}$. Upgrade to a hypothesis test. Under $H_0$ (rigid), $d_{ij}(t) = d_0 + \eta(t)$ where $\eta$ has a known (non-central-chi-ish) distribution induced by $\sigma$; under $H_1$ (non-rigid), $d_{ij}$ has additional low-frequency power. On the denoised series $\tilde d_{ij}$, define

$$
z_{ij} = \frac{\mathrm{std}_t\, \tilde d_{ij}(t)}{\hat\sigma_{d}},
\qquad
\hat\sigma_{d}^2 = \text{calibrated per-edge noise variance from static points (median of } \mathrm{std}\, d_{ij} \text{ over static-static edges).}
$$

Calibrating $\hat\sigma_d$ **per scene from static points** turns the threshold into a significance level ($z > 3$ ≈ p < 0.003) instead of a magic `threshold_mult` knob — and it adapts automatically to each grid cell's reconstruction quality, which is currently what breaks the sweep models (ARI goes negative at low Gaussian counts where noise is worst).

### 3. Graph statistics instead of edge thresholding

Replace "cut every edge with $z > \tau$, then connected components" with a noise-aware partition on the $z$-weighted graph:

- **Affinity:** $W_{ij} = \exp(-z_{ij}^2 / 2)$.
- **Partition:** Leiden community detection (as in Intrinsic-GS) or spectral clustering on $W$ with eigengap-chosen $K$ (proposal 04 §3.4). Both aggregate evidence over all paths; a boundary needs many independent high-$z$ edges to survive, so single-edge noise flips no longer merge/split parts.

### 4. Multiscale subsample-and-propagate (MBS's scalability trick, kept)

1. FPS subsample $M \approx 5\,000$–$20\,000$ points from the ROI.
2. Optionally **spatially smooth** trajectories before scoring: $\bar p_i(t) = \sum_{j \in \mathrm{kNN}(i)} w_{ij} p_j(t)$ (Gaussian weights over canonical distance) — averages $\sigma$ down by $\sqrt{k}$ for the subsample decision.
3. Segment the subsample (04 or 05 or upgraded Option B above).
4. Propagate labels to all $N$ Gaussians by $q$-NN majority vote in canonical space (already implemented in `mbs_infer.py`'s 3-NN propagation).

### 5. FFT motion fingerprint as a clustering feature (bonus)

The kept-band coefficients form a per-point feature $f_i = [\hat p_i(\omega_0), \dots, \hat p_i(H\omega_0)] \in \mathbb{C}^{3H}$ — an amplitude/phase fingerprint. Points on one rigid part share $R(t), \tau(t)$ so their fingerprints are related linearly; clustering $\{f_i\}$ gives an excellent EM initialization for proposal 05 or a standalone coarse segmentation at essentially zero extra cost.

## Integration

- All changes land in existing files: `rigidity_graph.py` (denoise → $z$-scores → weighted partition), `segment_rigid.py` (`--denoise`, `--calibrate-sigma`, `--leiden` flags; keep old path as default for regression).
- No new dependencies (FFT/argmax = numpy; Leiden via `igraph` optional — fall back to scipy spectral).
- Benefits everything downstream: proposals 01 (SNR gate), 03 (conductance $W_{ij}$), 05 (init + calibrated $\sigma$).

## Validation plan

1. Report, per grid/sweep model: calibrated $\hat\sigma_d$, and the $z$-score histogram for GT-same-part vs GT-different-part edges → the **separability curve** (AUROC of same/different classification). This single number tells us whether *any* per-edge method can work on each model, before clustering.
2. Re-run `run_grid_seg.py` (Option B + denoise + Leiden): target ARI ≫ 0.01 on the three grid models; expect the biggest gains on the sweep models where noise dominates.
3. Ablate: denoise only / calibration only / Leiden only, to attribute gains.
