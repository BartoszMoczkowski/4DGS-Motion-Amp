# Proposal 01 — Motion-gated ROI masking (restrict to the machine)

**Focus area:** 1 — limit what is analyzed to the machine itself, not the background.

## Motivation

The pump01/grid scenes contain a static environment (table, walls, rig) plus the machine. Background Gaussians are useless for motion segmentation but they:

1. inflate N (every k-NN graph, threshold, and connected-component pass pays for them),
2. contribute edges whose "rigidity score" is near zero *because they don't move*, which confuses the log-Otsu threshold — a static-static edge and a rigidly-moving edge are both perfectly rigid, so background clumps merge with machine parts that happen to be adjacent,
3. physically touch the machine (the pump sits on a surface), creating cross-boundary rigid edges.

The fix is to decide **which Gaussians belong to the machine at all** before any rigidity scoring, using the one signal background points cannot fake: they do not move.

## Mathematical formulation

From `extract_trajectories.py` we have trajectories $p_i(t) \in \mathbb{R}^3$, $t = 1..T$, for $N$ Gaussians.

### 1. Per-Gaussian motion energy

Define the centered trajectory and its energy:

$$
\bar{p}_i = \frac{1}{T}\sum_{t=1}^{T} p_i(t), \qquad
E_i \;=\; \frac{1}{T}\sum_{t=1}^{T} \lVert p_i(t) - \bar{p}_i \rVert_2^2
\;=\; \mathrm{tr}\!\big(\mathrm{Cov}[p_i(t)]\big).
$$

$E_i$ is the total variance of the trajectory. For an ideally static point $E_i = \sigma_{\text{noise}}^2$ (the reconstruction jitter floor); for a machine part moving with amplitude $A$ at frequency $f$, $E_i \approx \tfrac{1}{2}A^2 + \sigma_{\text{noise}}^2$.

### 2. Separating signal from the noise floor

The measured distribution (docs/motion-segmentation.md) is median edge jitter $\approx 1.25\,\mu m$ vs true motion at p90 $\approx 3.5\,\mu m$ — they overlap per-sample, but $E_i$ **integrates over all T frames**, and the noise is (approximately) temporally uncorrelated while the drive is periodic. Decompose in frequency via the DFT $\hat{p}_i(\omega) = \mathcal{F}\{p_i(t) - \bar p_i\}$:

$$
E_i = \frac{1}{T}\sum_{\omega} \lVert \hat{p}_i(\omega) \rVert^2
\quad\Longrightarrow\quad
S_i = \sum_{\omega \in \Omega_{\text{drive}} \cup \text{harmonics}} \lVert \hat{p}_i(\omega) \rVert^2,
\qquad
\mathrm{SNR}_i = \frac{S_i}{E_i - S_i + \varepsilon}.
$$

$\Omega_{\text{drive}}$ is known exactly — we authored the scene, the drive frequency is in `gen_scenes.py`. Band-passed energy $S_i$ rejects the white jitter floor; the gating statistic is $\mathrm{SNR}_i$, not raw $E_i$.

### 3. Gating rule

Two-component model on $\log S_i$ (static cluster vs moving cluster). With $S_i$ band-limited, the two modes are far better separated than raw edge scores, so a simple log-space Otsu on $\{ \log S_i \}$ (already the mechanism in `rigidity_graph.py`) or a 2-component GMM suffices:

$$
\mathcal{M} = \{\, i : \log S_i > \tau_{\text{Otsu}} \,\}.
$$

Note the asymmetry vs. the current failure: today Otsu runs on **edge** scores (one scalar per pair, dominated by noise); here it runs on **per-point integrated band-limited energy** (T frames of evidence per scalar).

### 4. Graph morphology — closing the ROI

A hard per-point gate tears apart parts whose amplitude is small (a rigid flange bolted to a moving housing has $A \approx 0$ but must stay in the ROI). Fix on the canonical k-NN graph $G$:

- **Dilation:** $\mathcal{M}^{+} = \{ j : \exists\, i \in \mathcal{M},\; (i,j) \in G,\; \text{within } d \text{ hops} \}$ — pull in low-amplitude points attached to movers.
- **Conditional readmission:** a static point $j$ is readmitted iff it is *rigidly locked* to a mover: $\mathrm{std}_t \lVert p_j(t) - p_i(t)\rVert < \kappa\,\sigma_{\text{noise}}$ for some neighbor $i \in \mathcal{M}$. This is the existing rigidity edge test, used only as a readmission gate, never as the primary threshold.

This is the discrete analogue of morphological closing with a rigidity-constrained structuring element.

### 5. Consequence for downstream segmentation

Segmentation (Option B, or proposals 04/05) runs on the induced subgraph $G[\mathcal{M}^{+}]$ only:

- N drops (typically 3–10× in these scenes) — every $O(Nk)$ or $O(N^2)$ step shrinks accordingly;
- the Otsu threshold inside the ROI sees only moving-motion scores, removing the static-edge contamination that merges parts;
- static background is reported as label $-2$ ("static"), distinct from floaters ($-1$), which is itself a useful deliverable for `render_amp.py` (amplify only the machine).

## Integration

- New module `motion-seg/motion_seg/motion_gate.py` (pure numpy/scipy): inputs `trajectories.npz` + drive frequency (or auto-detected as argmax of the mean spectrum); outputs `roi_mask` (+ `snr` per point) appended into the npz.
- `segment_rigid.py` gains `--roi` flag: filter to mask, build k-NN graph on the subset, label the rest $-2$.
- `extract_trajectories.py` unchanged.
- Evaluation: extend `evaluate_segmentation.py` to (a) report ROI precision/recall vs GT (background GT labels vs $-2$), (b) compute ARI/IoU **within the GT machine subset** so ROI errors and clustering errors are attributed separately.

## Risks

- A truly static machine part (mounting frame) is excluded by design — acceptable if the goal is amplifiable parts; mitigate with the conditional readmission (§4) and by checking GT: pump01's 107 GT parts let us measure exactly how many are static.
- SNR gating depends on the deformation field actually encoding the periodic motion (it does — `render_amp.py` FFT amplification works on the same signal).

## Validation plan

1. Re-run on `output/multipleview/pump01` and the 7 grid/sweep models via `scene-gen/run_grid_seg.py` with `--roi`.
2. Report: ROI precision/recall, ARI-within-machine, and the before/after edge-score histogram (expect a cleaner bimodal split).
3. Sanity: ARI on the synthetic self-test must stay ≈ 0.999 (gate must pass everything there).
