# Proposal 03 — Seeded part-focused segmentation

**Focus area:** 2 — focus on a specific part instead of segmenting everything at once.

## Motivation

The current pipeline attempts a *global* partition into ~107 rigid parts in one shot, in a regime where the noise floor rivals the motion. Global partitioning forces a single threshold/resolution on parts of wildly different size and amplitude — the failure mode we observe (under/over-merging, ARI ≈ 0). The thesis pipeline doesn't actually need all 107 parts at once: motion amplification is applied per selected segment (`render_amp.py --amp_factors`), and evaluation cares about individual parts. A **seeded, one-part-at-a-time** formulation:

- turns a 107-way clustering into a binary *in-part / not-in-part* decision — statistically much easier;
- lets analysis resolution adapt to the part (denser graph, tighter thresholds only where needed);
- matches the interactive workflow (user clicks the part to amplify) and the GT workflow (seed = GT part id → direct per-part IoU).

## Mathematical formulation

### 1. Seed selection

A seed is either (a) a user click back-projected to the nearest visible Gaussian (through any camera, using the projection machinery of proposal 02), or (b) the GT part's Gaussians for validation. Let $s$ be the seed Gaussian set, $|s| \ge 1$.

### 2. Geodesic neighborhood on the canonical graph

On the canonical k-NN graph $G$ with edge lengths $\ell_{ij} = \lVert \mu_i - \mu_j \rVert$, compute graph geodesics $d_G(s, \cdot)$ (Dijkstra from the seed set). Restrict all further work to the ball

$$
B_r(s) = \{ i : d_G(s, i) \le r \}, \quad r \text{ adapted to the expected part size},
$$

which typically contains $10^2$–$10^3$ points instead of $10^5$ — enabling per-edge statistics that would be too expensive globally (e.g., full covariance of pairwise distance, multi-hypothesis thresholds).

### 3. Heat-diffusion (personalized PageRank) part score

The core is a diffusion of the seed mass over a **rigidity-weighted** graph — the graph analogue of "a part is the region reachable from the seed without crossing a non-rigid edge."

Let $W$ be the weighted adjacency on $B_r(s)$ with

$$
W_{ij} = \exp\!\Big( -\frac{\mathrm{std}_t\, d_{ij}(t)^2}{2\sigma_r^2} \Big),
\qquad d_{ij}(t) = \lVert p_i(t) - p_j(t) \rVert,
$$

i.e. edge conductance decays with rigidity violation ($\sigma_r$ = expected rigidity noise, estimable from static points — proposal 06). With lazy random-walk matrix $P = \tfrac{1}{2}(I + D^{-1}W)$ and seed distribution $e_s$ (uniform on $s$), the personalized PageRank vector is

$$
\pi = \alpha\, e_s + (1-\alpha)\, P^\top \pi
\quad\Longleftrightarrow\quad
\pi = \alpha \big(I - (1-\alpha) P^\top\big)^{-1} e_s ,
$$

solved by power iteration from the seed, $O(\text{nnz})$ per iteration, trivial at ball size. $\pi_i$ is the probability that a rigidity-biased random walk starting in the seed sits at $i$. Non-rigid boundary edges are bottlenecks with conductance $\to 0$, so $\pi$ decays sharply across part boundaries even when the boundary is only *slightly* less rigid than the interior — exactly the noise-regime behavior we need, because the diffusion integrates over **all paths**, not a single thresholded edge.

### 4. Decision and boundary refinement

Threshold $\pi_i > \tau_\pi$ (Otsu on $\log \pi$ within the ball — the distribution is strongly bimodal by construction). Then a refinement pass re-fits the boundary as a binary min-cut:

$$
\min_{x \in \{0,1\}^{|B|}} \sum_{(i,j)} W^{\text{cut}}_{ij} |x_i - x_j| + \lambda \sum_i (\pi_i - \tfrac12)(1 - 2x_i),
$$

solvable exactly at ball size with `scipy`/maxflow, or simply iterated conditional modes. The PageRank score supplies the unary terms; the rigidity conductance supplies the pairwise terms.

### 5. Iterate for full coverage (optional)

Repeatedly seed the largest unlabeled connected component → full partition, one part at a time, each with its own adaptive $\sigma_r$ and threshold. This "sequential extraction" also fixes the global-Otsu degeneracy documented in `docs/motion-segmentation.md`.

## Why this fits the noise regime

Single-edge thresholding fails when $\sigma_{\text{noise}} \approx$ motion. Diffusion succeeds there because a boundary crossing requires *many* independent noisy edges to simultaneously look rigid; the false-conductance probability multiplies along every cut path. Formally, the commute time $C(s, j) = 2m\, R_{\text{eff}}(s, j)$ is governed by effective resistance $R_{\text{eff}}$, which is dominated by the bottleneck cut — precisely the part boundary.

## Integration

- New module `motion-seg/motion_seg/seeded_part.py` (pure numpy/scipy: Dijkstra, power iteration, Otsu — all already-used dependencies).
- CLI: `--seed-gaussian-idx` / `--seed-gt-part N` (reads `gt_segmentation.npz`) / `--seed-click u v --camera v` (uses proposal 02's projection).
- Output: binary mask + optional iterated full partition; plugs into `evaluate_segmentation.py` unchanged.
- UI hook: `amp-ui/amp_ui/ampUI.py` already has a rendering view — the click-to-seed path is the natural UX for choosing what to amplify.

## Risks

- Articulated intra-part boundaries (pump shaft vs housing) may *correctly* split what GT calls one part — check against the GT part hierarchy; tune $\alpha$ (smaller $\alpha$ = longer-range diffusion = larger parts).
- Very small parts (few Gaussians) give weak seed mass; mitigate by seeding with the GT part or several clicks.

## Validation plan

1. Oracle mode: seed each of the 107 GT parts in turn on pump01, report per-part IoU distribution (median, p10).
2. Compare per-part IoU vs the global Option-B partition on the same models.
3. Interactive mode: 5–10 manual clicks, measure IoU vs GT.
