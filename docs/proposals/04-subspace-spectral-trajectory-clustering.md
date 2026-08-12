# Proposal 04 — Subspace & spectral trajectory clustering (cheap MBS, classical line)

**Focus area:** 3 — cheaper methods inspired by MBS.

## Motivation

MBS's MotNet is a *learned* pairwise-affinity network trained on noisy scene flow at unit scale — out-of-distribution for our mm-scale 4DGS trajectories (measured ARI ≈ 0). But the problem MotNet approximates has a 25-year-old **closed-form** answer: the multibody motion segmentation literature (Vidal et al., GPCA/SSC/LSA, Hopkins-155 benchmark). In 4DGS we have the *best-case* version of that problem: full 3D trajectories (no projection ambiguity, no missing data, no correspondence problem). The classical result:

> **Points on one rigid body move in a low-dimensional linear subspace of trajectory space.**

No network, no training, no CUDA ops — an SVD and a sparse eigenproblem.

## Mathematical formulation

### 1. The motion subspace

Stack trajectory $i$ into $y_i = [\,p_i(1);\, p_i(2);\, \dots;\, p_i(T)\,] \in \mathbb{R}^{3T}$. For points $i$ on a rigid body undergoing $\{R(t), \tau(t)\}$:

$$
p_i(t) = R(t)\,\mu_i + \tau(t)
\;\Longrightarrow\;
y_i = \underbrace{\begin{bmatrix} R(1) \\ \vdots \\ R(T) \end{bmatrix}}_{3T \times 3}\, \mu_i + \underbrace{\begin{bmatrix} \tau(1) \\ \vdots \\ \tau(T) \end{bmatrix}}_{\text{translation}} .
$$

So all trajectories of one body lie in the **affine subspace** $\tau + \mathrm{span}(R_{(1)}, R_{(2)}, R_{(3)})$ of dimension $\le 3$ (linear subspace of dimension $\le 4$ after homogenizing, $\tilde y_i = [y_i;\, 1] \in \mathbb{R}^{3T+1}$). Degenerate motions lower the rank further (pure translation → dim 1; rotation about a fixed axis → the trajectory of each point is a circle → planar → dim 2), which GPCA-style methods handle natively.

The scene = a **union of $K$ low-dimensional subspaces** in $\mathbb{R}^{3T+1}$. Segmentation = subspace clustering.

### 2. Why this beats raw-distance rigidity in our noise regime

The rank structure is a **global** constraint: a trajectory agrees with a body iff its *entire* $3T$-vector lies in a 3-flat. Trajectory jitter $\epsilon_i(t)$ with $\mathbb{E}[\epsilon]=0$ adds a noise vector of norm $\approx \sigma\sqrt{3T}$, but it is spread over all $3T$ dimensions, while the signal of interest is confined to a 3-dimensional flat — the residual-to-flat distance concentrates:

$$
\mathbb{E}\, \mathrm{dist}(\tilde y_i + \epsilon_i,\, S_k)^2 \approx \sigma^2 (3T - 3),
\qquad
\text{but two different bodies' flats differ by } \sim A^2 T \text{ (amplitude-squared)}.
$$

The margin between "same subspace" and "different subspace" scales with $A^2 T$ while the within-subspace scatter scales with $\sigma^2 \cdot 3T$ — for $T = 60$ frames, subspace residuals average 60 frames of noise into the decision where Option B's per-edge std uses the same data far less efficiently (and then thresholds a heavy-tailed statistic).

### 3. Algorithm — Local Subspace Affinity (LSA) + spectral clustering

Full SSC (ℓ¹ self-expressiveness, $y_i = \sum_j c_{ij} y_j$) is $O(N^2 \cdot \text{ADMM})$ — too slow at $N > 10^5$. LSA keeps the theory and drops the cost:

1. **PCA denoise/reduce:** project $\{\tilde y_i\}$ onto the top-$D$ principal components ($D = 8$–$15$; union of $K$ rank-4 subspaces spans $\le 4K$ dims, so small $D$ suffices for $K \approx 10^2$ local structure). This also implements the frequency-domain denoising of proposal 06 implicitly (smooth trajectories = low-rank globally).
2. **Local fits:** for each point, fit a 4-flat to its $m$ nearest neighbors in projected space ($m \approx 12$–$20$) by SVD → local subspace estimate $S_i$.
3. **Affinity from subspace angles:** with principal angles $\theta_q(S_i, S_j)$,
$$
A_{ij} = \exp\!\Big(-\frac{\gamma}{2}\, \textstyle\sum_q \sin^2 \theta_q(S_i, S_j)\Big) \cdot \mathbb{1}[j \in \mathrm{kNN}(i)],
$$
sparsified by the canonical-space k-NN graph (rigid parts are contiguous — same argument as `rigidity_graph.py`).
4. **Spectral clustering:** normalized Laplacian $L = I - D^{-1/2} A D^{-1/2}$, take the $K$ smallest eigenvectors, embed, k-means. $K$ is chosen by the **eigengap** $\arg\max_K (\lambda_{K+1} - \lambda_K)$ — no Otsu-on-skewed-histogram failure mode.

Cost: k-NN $O(N \log N \cdot D)$, local SVDs $O(N m^2 D)$, one sparse eigendecomposition $O(\text{nnz} \cdot K)$ — all CPU, minutes at $N = 10^5$.

### 4. Variant — direct residual affinity ("EM-free Kabsch-lite")

Cheaper still, closer to MBS's per-pair affinity but analytic: for each graph edge $(i,j)$, fit the best shared rigid motion to the *pair* of trajectories (Kabsch on the concatenated 6-point track per frame, closed form) and set $A_{ij} = \exp(-\bar r_{ij}^2 / 2\sigma^2)$ from the mean residual $\bar r_{ij}$. Then the same spectral step. This is MotNet replaced by an exact geometric computation — see proposal 05 for the full EM version.

## Integration

- New module `motion-seg/motion_seg/segment_subspace.py` (numpy/scipy/sklearn-style, CPU; sklearn is not yet a dep — implement PCA/spectral with scipy to keep the package dependency-light).
- Input contract identical to `segment_rigid.py` (`trajectories.npz` in, `segmentation.npz` out) so `run.sh` and `run_grid_seg.py --impl subspace` work with a one-line addition.
- Combine with proposal 01's ROI gate via the existing `--roi` flag.

## Risks

- **Pure-translation degeneracy:** parts that only translate along the *same* axis with the same phase occupy the *same* 1-D subspace → merged. Mitigation: append canonical position to the feature ($[\tilde y_i;\, \beta \mu_i]$) making it an affine-clustering problem, and/or run inside the spatially contiguous graph so only adjacent same-motion parts can merge (they would also be merged by any rigidity method — they are kinematically indistinguishable).
- **Eigengap ambiguity** when parts move almost identically; report the eigengap spectrum per scene as a diagnostic.
- Harmonic (same-frequency different-phase) motions are handled fine — different phases = different flats.

## Validation plan

1. Self-test scene (7 bodies): expect ARI ≈ 1 (subspace methods are exact on noiseless rigid data).
2. Grid/sweep models via `run_grid_seg.py --impl subspace`: ARI/IoU vs Option B baseline; also with `--roi` (01) and SNR-denoised trajectories (06).
3. Report the eigenvalue spectrum per scene — tells us whether 107 clusters is even resolvable from the trajectory data at current reconstruction noise.
