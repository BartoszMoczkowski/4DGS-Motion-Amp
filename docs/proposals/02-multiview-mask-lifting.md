# Proposal 02 — Multi-view mask lifting (restrict to the machine)

**Focus area:** 1 — limit analysis to the machine, exploiting the calibrated 10-camera capture.

## Motivation

Proposal 01 restricts the ROI using motion alone. This proposal uses **image-space** evidence instead — a stronger and complementary signal, and one uniquely cheap in this setup: unlike a casually captured video, our scenes have **10 fully calibrated cameras** (`multipleview` format stores per-frame camera matrices) and a **known static background plate** (the scene was authored in Omniverse; a clean-plate render or the first frame with the machine absent can be produced, or SAM can segment the machine from any frame).

The output is the same as 01 — a per-Gaussian foreground mask $\mathcal{M}$ — so the two compose (union or intersection) and share downstream tooling.

## Mathematical formulation

### 1. Per-Gaussian foreground votes

For camera $v$ with intrinsics $K_v$ and pose $[R_v \mid t_v]$, the projection of Gaussian mean $\mu_i$ at time $t$ is

$$
\pi_{v,t}(\mu_i) = \mathrm{proj}\!\big( K_v [R_v \mid t_v]\, [\mu_i(t);\,1] \big) \in \mathbb{R}^2 .
$$

Given a binary machine mask $M_{v,t} : \Omega \to \{0,1\}$ per view/frame, define the raw vote

$$
v_i = \frac{1}{|V|}\sum_{v \in V}\; M_{v,t_0}\!\big(\pi_{v,t_0}(\mu_i)\big),
$$

evaluated at a single reference time $t_0$ (votes could be pooled over $t$, but one well-chosen frame suffices and avoids the machine leaving its own mask under large deformation).

### 2. Occlusion-corrected voting

Naive projection votes wrongly mark background Gaussians *behind* the machine as foreground. Use the rasterizer's depth: Gaussian $i$ contributes to pixel $u$ in view $v$ only if its depth $z_i^{(v)}$ is within tolerance of the rendered depth $D_v(u)$:

$$
w_i^{(v)} = \mathbb{1}\!\big[\, |z_i^{(v)} - D_v(\pi_v(\mu_i))| < \tau_z \,\big],
\qquad
v_i = \frac{\sum_v w_i^{(v)} M_v(\pi_v(\mu_i))}{\sum_v w_i^{(v)} + \varepsilon}.
$$

This is exactly the visibility weighting the splatting $\alpha$-blending already computes, so $w_i^{(v)}$ can be harvested from a forward render with per-Gaussian contribution outputs (the rasterizer gives per-pixel $\alpha$ contributions; `core/motion_amp/renderer.py` already returns raw pre-rasterization parameters, and depth rendering is one extra channel). A cheaper approximation: keep the top-$\alpha$ contributors per masked pixel via the standard "inverse rendering" trick used by FlashSplat/Gaussian Grouping — accumulate $a_i = \sum_{v,u \in \text{mask}} \alpha_i^{(v)}(u)$ per Gaussian and threshold $a_i$; no explicit depth test needed.

### 3. Decision rule

$$
\mathcal{M} = \{ i : v_i > \tfrac{1}{2} \} \quad\text{(majority of visible views)},
$$

then the same graph-morphology closing as Proposal 01 §4 to fill holes (interior Gaussians of the machine that are occluded in every view get pulled in by dilation along the canonical k-NN graph — they are *inside* a foreground shell).

### 4. Where the mask $M_v$ comes from (cheapest first)

1. **Clean-plate differencing:** render/capture the scene without the machine (or an early frame where the machine region is known from the USD); $M_v(u) = \mathbb{1}[\lVert I_v(u) - I_v^{\text{plate}}(u)\rVert > \tau]$, morphologically cleaned. Zero ML, fully deterministic, and we control the scene generator.
2. **SAM / SAM-2** on one frame per view with a point/box prompt on the machine (one click per view, or promptable from the projected centroid of the known machine geometry).
3. **GT proxy (validation only):** project the GT-labeled init Gaussians to obtain an oracle mask — useful to measure the ceiling of this approach and to validate the lifting machinery independently of mask quality.

## Integration

- New module `motion-seg/motion_seg/mask_lift.py`: loads `cameras.json`/poses from the `multipleview` dataset (reuse `core/scene` camera classes, GPU env), rasterizes depth at $t_0$, computes $v_i$, writes `roi_mask` into `trajectories.npz` — identical contract to Proposal 01 so `segment_rigid.py --roi` works unchanged.
- Mask production is a small host-side script (clean-plate diff: pure OpenCV; SAM: `ultralytics`/HF, one-time per scene).

## Risks

- Projection votes are sensitive to calibration; our cameras come from Isaac Sim ground truth, so this is the best-case scenario for the method (a real-capture weakness, not ours).
- Transparent/reflective machine parts may diff poorly against the plate — SAM fallback covers it.
- Occluded interior Gaussians rely on the graph-closing step; verify interior recall against GT.

## Validation plan

1. Oracle-mask ceiling: lift GT-projected masks → measure achievable ARI with Option B inside that ROI.
2. Clean-plate masks → same metric. If oracle ≫ clean-plate, the bottleneck is the mask; if equal, segmentation itself is the bottleneck (proceed to proposals 05/06).
3. Report ROI precision/recall against GT part labels, plus ARI-within-machine as in 01.
