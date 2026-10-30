# W4 (T52) — Fixed-K soft learned motion components

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W3 (T51)  
**Experiment:** E3 (Best W3 configuration + fixed-$K$ soft motion components, split/merge disabled)  
**Gate before W5:** Zero-component initialization is an exact W3 native-4DGS no-op; densification/pruning/checkpoint/resume tests pass; offline render matches training render identically; synthetic single-mover ($K=1$) and two-mover ($K=2$) pilots beat the W3 baseline without binary masks or background freezing.

---

## 1. Objective

Discover shared low-dimensional rigid/periodic motion components during training **without binary ROI masks, without `pure_rigid`, and without `static_background`**, while every Gaussian retains its native HexPlane free-form residual pathway:

$$\Delta p_i(t) = h_i(t) + \sum_{k=1}^{K} q_{ik}\, g_{ik}(t), \qquad q_{ik} \ge 0, \quad q_{i0} = 1 - \sum_{k=1}^{K} q_{ik} \ge 0$$

where $h_i(t) = p_i^{\text{hex}}(t) - p_i^{\text{canon}}$ is the persistent native HexPlane residual, $g_{ik}(t) = \bigl(R_k(t)(p_i^{\text{canon}} - c_k) + c_k + T_k(t)\bigr) - p_i^{\text{canon}}$ is the 3D displacement of Gaussian $i$ under component $k$'s rigid transform, and $q_{i0}$ is the implicit null component. At $q_{i0} = 1$, Gaussian $i$ follows pure native HexPlane deformation—never frozen.

## 2. Design Requirements & Codebase Integration Points

1. **Design note first:**
   - Completed and committed in [`W4-design-note.md`](W4-design-note.md). Specifies:
     - Exact position and quaternion composition equations (evaluate candidate 3D displacements $g_{ik}(t)$ per component first and blend displacements linearly with $q_{ik}$; compose small-angle rotational increments in tangent/axis-angle space $\sum_{k=1}^K q_{ik} \theta_k(t) \hat{w}_k$ before quaternion conversion—never average rotation matrices or quaternions naively).
     - Warm-start initialization from a W3 native checkpoint by fitting $K$ candidate rigid trajectories + soft responsibilities $q_{ik} \in (0, 1)$ to extracted HexPlane residual trajectories $h_i(t)$.
     - Alternating optimization & EMA responsibility update schedule + gradual ramp-in $\alpha(t) \in [0, 1]$ of component displacement and detached responsibility-weighted residual penalty:
       $$\mathcal{L}_{\text{res}} = \lambda_{\text{res}}(t) \sum_{i=1}^{N} \left(1 - \operatorname{stopgrad}(q_{i0})\right) \|h_i(t)\|_2^2$$
       which gently encourages shared rigid motion to be explained by components while allowing $h_i(t)$ to absorb genuine non-rigid motion.
   - **Nyquist guardrail:** At $20.004\text{ fps}$, the real turbine pilot ($K=1$) must use `num_harmonics = 1` at $f_0 = 5.691\text{ Hz}$ ($h_2 = 11.382\text{ Hz}$ and $h_3 = 17.073\text{ Hz}$ exceed Nyquist). Also support a smooth low-dimensional B-spline/chebyshev trajectory basis for non-periodic synthetic movers.
2. **Densification & pruning state inheritance in [`core/scene/gaussian_model.py`](../../core/scene/gaussian_model.py#L351-L499):**
   - Soft responsibilities $q_{ik}$ (or logits $z_{ik}$) are per-Gaussian state of shape $[N, K]$.
   - Hook responsibility state into `densify_and_split` (repeat $N$ times from parent), `densify_and_clone` (copy from parent), `add_point_by_mask` (copy from masked parent), `densification_postfix` (concatenate), and `prune_points` (slice by `valid_points_mask`), recording parent provenance and never resetting densified points to null.
3. **Single deformation-dispatch API across all 4 call sites:**
   - Route [`core/gaussian_renderer/__init__.py:render()`](../../core/gaussian_renderer/__init__.py), [`core/utils/render_utils.py:get_state_at_time()`](../../core/utils/render_utils.py), [`core/motion_amp/renderer.py:render_mod()`](../../core/motion_amp/renderer.py), and both `train.py` files through one shared deformation-dispatch function so offline `render`, `seg_extract`, and `amp` always include component motion.

## 3. Verification & Tests

- **Zero-component / $\alpha=0$ equivalence test:** Matches W3 native HexPlane output to floating-point precision.
- **Simplex & gradient test:** $q_{ik} \ge 0$, $\sum_{k=1}^K q_{ik} \le 1$, finite gradients on component trajectory parameters and HexPlane residual.
- **Densification & pruning state test:** Verify `densify_and_split`, `densify_and_clone`, `add_point_by_mask`, and `prune` preserve responsibility tensor alignment `[N, K]` and parent values.
- **Training-vs-offline render equivalence test:** Saved checkpoint reloaded via `Scene(..., load_iteration=...)` produces identical `means3D_final`, `rotations_final`, and rendered RGB in `render()`, `get_state_at_time()`, and `render_mod()`.
- **Synthetic $K=1$ and $K=2$ smoke tests:** Recover one and two rigid movers without binary ROI masks.

## 4. Deliverables & Stop Conditions

- **Deliverables:** W4 design note, `core/scene/motion_components.py` module, `GaussianModel` densification hooks, unified deformation dispatch, checkpoint/resume support, component diagnostic plots, presets, and test suite (`orchestrator/tests/test_w4_soft_motion_components.py`).
- **Stop conditions:** Do not proceed if responsibilities collapse to null, if the HexPlane residual is disabled/frozen, if densification resets responsibilities, or if offline rendering omits component motion.

