# Proposal 05 — Iterative Kabsch EM rigid-body clustering (cheap MBS core)

**Focus area:** 3 — cheaper methods inspired by MBS.

## Motivation

The genuinely reusable part of MBS (per `docs/motion-segmentation.md`) is its segmentation core: an affinity between points that measures *"do these two points move as one rigid body?"*, plus spectral synchronization and per-group rigid fitting. MBS computes that affinity with MotNet — a learned net that fails OOD here (ARI ≈ 0). But for 4DGS the affinity is computable **in closed form**: two points are rigidly related iff their trajectories are related by a *fixed relative pose*, and the deviation is measured by the Kabsch algorithm. Around this we build an EM that:

- **E-step:** softly assign each Gaussian to the rigid-body model that best explains its trajectory;
- **M-step:** re-fit each body's motion by weighted Kabsch per frame.

Soft assignment is exactly what our noise regime needs: instead of thresholding a noisy per-edge score (Option B's failure), every assignment pools the whole $T$-frame trajectory and is weighted by residual — noise averages out across frames and across the EM iterations.

## Mathematical formulation

### 1. Generative model

Gaussian $i$ belongs to body $k$ with prior $\pi_k$. Body $k$ has motion $\{(R_k(t), \tau_k(t))\}_{t=1..T}$, $R_k(t) \in SO(3)$. Observation model:

$$
p_i(t) = R_k(t)\, \mu_i + \tau_k(t) + \epsilon_i(t),
\qquad \epsilon_i(t) \sim \mathcal{N}(0, \sigma^2 I_3),
$$

where $\mu_i$ is the canonical position (known — a large advantage over generic point-cloud motion segmentation, where structure must be estimated too).

### 2. E-step — responsibilities in closed form

Given body motions, the residual of point $i$ under body $k$ is

$$
r_{ik}^2 = \sum_{t=1}^{T} \big\lVert p_i(t) - R_k(t)\mu_i - \tau_k(t) \big\rVert^2 ,
\qquad
\gamma_{ik} = \frac{\pi_k \exp(-r_{ik}^2 / 2\sigma^2)}{\sum_{k'} \pi_{k'} \exp(-r_{ik'}^2 / 2\sigma^2)} .
$$

$r_{ik}^2$ is a sum of $3T$ squared residuals → by the $\chi^2$ concentration, $\mathrm{std}(r_{ik}^2)/\mathbb{E}[r_{ik}^2] \approx \sqrt{2/(3T)}$ — for $T=60$ the statistic is ~7× tighter than a per-frame test. This is the analytic replacement for MotNet's affinity.

### 3. M-step — weighted Kabsch per body per frame

Given responsibilities, each body's motion at each frame is a weighted absolute-orientation problem. With $w_i = \gamma_{ik}$, $\bar\mu = \sum_i w_i \mu_i / \sum_i w_i$, $\bar p(t) = \sum_i w_i p_i(t) / \sum_i w_i$:

$$
H_k(t) = \sum_i w_i\, (\mu_i - \bar\mu)(p_i(t) - \bar p(t))^\top = U \Sigma V^\top,
$$

$$
R_k(t) = V \,\mathrm{diag}(1,1,\det(VU^\top))\, U^\top,
\qquad
\tau_k(t) = \bar p(t) - R_k(t)\,\bar\mu .
$$

Closed form, $O(N)$ per body per frame, no iterations inside. This is precisely the "per-group Kabsch fitting" stage of MBS, promoted from post-processing to the inner loop.

### 4. Initialization and model selection

- **Init:** cluster a *trajectory feature* per Gaussian (e.g. FFT coefficients at the drive frequency + harmonics — phase and amplitude fingerprint, see proposal 06) with k-means++ for $K$ in a candidate range; or seed from proposal 04's spectral result.
- **$K$ selection:** BIC over the EM runs,
$$
\mathrm{BIC}(K) = \sum_{i,k} \gamma_{ik}\, r_{ik}^2 + \nu_K \log N, \quad \nu_K = 6TK + K,
$$
or run greedy splitting: after convergence, split the body with the largest within-residual (test by dip statistic on residual-projected coordinates) until no split improves BIC — this handles "107 parts" without knowing 107 in advance.
- **Spatial prior (optional but powerful):** multiply the E-step prior by a graph smoothness term or run one iteration of α-expansion on the canonical k-NN graph with unary $r_{ik}^2$ and Potts pairwise — rigid parts are contiguous (same argument as `rigidity_graph.py`), and this kills speckle assignments.

### 5. Complexity

Per iteration: E-step $O(NKT)$ (dominated by $r_{ik}$; use ROI from proposal 01 and an FPS subsample from proposal 06 to cut $N$), M-step $O(NKT)$ with tiny SVDs. 10–30 iterations typical. No GPU, no training data, no checkpoint, no OOD risk — the model *is* the physics.

### 6. Handling near-rigid / deforming parts

Bodies whose residual floor stays high after convergence (flexible couplings, belts) are flagged non-rigid ($\bar r_k > \kappa\sigma$) and can be sub-clustered with the subspace method of proposal 04 (which needs no rigidity), giving a rigid/non-rigid typed partition like MoGaF's grouping.

## Integration

- New module `motion-seg/motion_seg/segment_kabsch_em.py` — pure numpy (small per-body SVDs; scipy sparse for the optional graph prior).
- Same I/O contract as `segment_rigid.py`; register as `run_grid_seg.py --impl kabsch` so the full grid benchmark runs unchanged.
- Composes with 01 (ROI gate: run EM only inside $\mathcal{M}^+$, static = body 0 with $R(t)=I, \tau(t)=0$ pinned) and 06 (pre-smoothed trajectories, subsample + propagate).

## Risks

- EM local minima with 107 bodies — mitigate with spectral (04) or FFT-fingerprint initialization, restarts, and the greedy-split scheme.
- Kinematically identical parts (same motion) can never be separated by any motion method; they will (correctly) merge — evaluate whether pump01 GT contains such pairs to set the achievable ARI ceiling.
- $\sigma$ must be estimated per scene — use static points' jitter (06) as a calibrated noise model rather than a free knob.

## Validation plan

1. Self-test scene: ARI ≈ 1 expected, including recovering per-frame body motions.
2. Grid benchmark `run_grid_seg.py --impl kabsch`: ARI/IoU vs Option B and MBS rows already in `runs/`.
3. Diagnostic outputs: converged per-body residual floor vs $\sigma$ (shows how many bodies are actually resolvable at current reconstruction noise), BIC curve over $K$.
