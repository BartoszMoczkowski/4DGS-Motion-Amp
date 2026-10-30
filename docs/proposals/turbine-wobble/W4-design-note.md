# W4 (T52) Design Note: Fixed-K Soft Learned Motion Components

**Date:** 2026-10-30  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Task:** W4 / T52  
**Author:** AI Agent (Antigravity) & Bartosz Moczkowski  
**Specification:** [`W4-soft-motion-components.md`](W4-soft-motion-components.md)  

---

## 1. Mathematical Formulation

### 1.1 Motion Decomposition & Continuous Responsibilities

Let $p_i^{\text{canon}} \in \mathbb{R}^3$ be the canonical rest position of Gaussian $i \in \{1, \dots, N\}$.
At continuous normalized time $t \in [0, 1]$, the native HexPlane deformation field produces a deformed position $p_i^{\text{hex}}(t)$ and deformed rotation $r_i^{\text{hex}}(t)$.
The native HexPlane free-form displacement residual is defined as:
$$h_i(t) \triangleq p_i^{\text{hex}}(t) - p_i^{\text{canon}}$$

We introduce $K$ shared low-dimensional rigid motion components, indexed $k \in \{1, \dots, K\}$.
Each Gaussian $i$ has continuous soft responsibilities $q_i = [q_{i1}, \dots, q_{iK}]^T \in [0, 1]^K$, with an implicit null component responsibility $q_{i0}$:
$$q_{ik} \ge 0 \quad (\forall k \in \{1, \dots, K\}), \qquad q_{i0} = 1 - \sum_{k=1}^K q_{ik} \ge 0$$
Thus, $q_i$ lies on the standard unit simplex in $\mathbb{R}^{K+1}$.
When $q_{i0} = 1$ (all $q_{ik} = 0$), Gaussian $i$ follows pure native HexPlane deformation—never frozen, never pinned.

### 1.2 Exact Position & Quaternion Composition

#### 3D Position Displacement Composition
For each component $k$, its rigid transformation at time $t$ consists of:
- Rotation $R_k(t) \in \mathrm{SO}(3)$
- Translation $T_k(t) \in \mathbb{R}^3$
- Pivot / center of rotation $c_k \in \mathbb{R}^3$

The candidate 3D displacement of Gaussian $i$ under component $k$ is evaluated *per component first*:
$$g_{ik}(t) \triangleq \bigl( R_k(t)(p_i^{\text{canon}} - c_k) + c_k + T_k(t) \bigr) - p_i^{\text{canon}} = (R_k(t) - I)(p_i^{\text{canon}} - c_k) + T_k(t)$$

With ramp-in parameter $\alpha(t) \in [0, 1]$, candidate displacements are blended linearly with soft responsibilities $q_{ik}$:
$$p_i^{\text{final}}(t) = p_i^{\text{canon}} + h_i(t) + \alpha(t) \sum_{k=1}^K q_{ik} g_{ik}(t)$$

#### Rotation Composition in Tangent / Axis-Angle Space
Naively averaging rotation matrices or unit quaternions produces non-orthogonal matrices or unnormalized, distorted rotations. Instead, rotational increments are composed strictly in tangent (axis-angle) space:
- Each component $k$ defines an instantaneous rotation vector $\omega_k(t) = \theta_k(t) \hat{w}_k \in \mathbb{R}^3$, where $\hat{w}_k$ is a unit rotation axis and $\theta_k(t)$ is the rotation angle in radians.
- The blended component rotation vector for Gaussian $i$ is formed in tangent space:
  $$\omega_i^{\text{comp}}(t) = \alpha(t) \sum_{k=1}^K q_{ik} \omega_k(t) \in \mathbb{R}^3$$
- Let $\phi_i = \|\omega_i^{\text{comp}}(t)\|_2$. The tangent vector is converted to a unit quaternion $\Delta q_i^{\text{comp}}(t) \in \mathbb{H}$ using the stable exponential map:
  $$\Delta q_i^{\text{comp}}(t) = \begin{cases}
  \left[ 1 - \frac{\phi_i^2}{8}, \; \frac{1}{2} \left(1 - \frac{\phi_i^2}{24}\right) \omega_i^{\text{comp}}(t) \right], & \text{if } \phi_i < 10^{-5} \\
  \left[ \cos\left(\frac{\phi_i}{2}\right), \; \frac{\sin(\phi_i / 2)}{\phi_i} \omega_i^{\text{comp}}(t) \right], & \text{otherwise}
  \end{cases}$$
- The final Gaussian rotation quaternion is obtained via standard quaternion multiplication:
  $$r_i^{\text{final}}(t) = \Delta q_i^{\text{comp}}(t) \otimes r_i^{\text{hex}}(t)$$
  normalized to unit length.

When $\alpha(t) = 0$, $K = 0$, or $\sum_{k=1}^K q_{ik} = 0$, $\omega_i^{\text{comp}}(t) = 0 \implies \Delta q_i^{\text{comp}}(t) = [1, 0, 0, 0]$, yielding $p_i^{\text{final}}(t) = p_i^{\text{hex}}(t)$ and $r_i^{\text{final}}(t) = r_i^{\text{hex}}(t)$ identically.

---

## 2. Component Trajectory Parameterizations

### 2.1 Periodic Fourier Trajectory Basis & Nyquist Guardrail
For periodic motion (such as turbine wobble):
$$T_k(t) = \sum_{n=1}^{H_k} \left[ a_{k,n} \cos(2\pi n f_k t) + b_{k,n} \sin(2\pi n f_k t) \right] u_{k,n}$$
$$\omega_k(t) = \sum_{n=1}^{H_k} \left[ \tilde{a}_{k,n} \cos(2\pi n f_k t) + \tilde{b}_{k,n} \sin(2\pi n f_k t) \right] v_{k,n}$$

**Nyquist Guardrail:**
At capture frame rate $\mathrm{fps} = 20.004\text{ fps}$, the Nyquist limit is $f_{\text{Nyq}} \approx 10.002\text{ Hz}$.
For the real turbine wobble fundamental $f_0 = 5.691\text{ Hz}$:
- Harmonic $n = 1$: $f_1 = 5.691\text{ Hz} < f_{\text{Nyq}}$ (valid).
- Harmonic $n = 2$: $f_2 = 11.382\text{ Hz} > f_{\text{Nyq}}$ (violates Nyquist; strictly forbidden).
- Harmonic $n = 3$: $f_3 = 17.073\text{ Hz} > f_{\text{Nyq}}$ (violates Nyquist; strictly forbidden).

The real turbine pilot ($K=1$) must enforce `num_harmonics = 1`. If `num_harmonics > 1` is requested when $n \cdot f_k \ge \frac{1}{2}\mathrm{fps}$, the initialization raises an explicit `ValueError`.

### 2.2 Smooth Non-Periodic Chebyshev Trajectory Basis
For synthetic scenes with non-periodic motion (e.g. bouncing balls, rigid translations):
Time is mapped to normalized Chebyshev domain $\tau = 2t - 1 \in [-1, 1]$.
Trajectory coordinates are parameterized by Chebyshev polynomials $T_m(\tau)$:
$$T_k(t) = \sum_{m=0}^{D} c_{k,m} T_m(\tau), \qquad \omega_k(t) = \sum_{m=0}^{D} d_{k,m} T_m(\tau)$$
with recurrence $T_0(\tau)=1, T_1(\tau)=\tau, T_{m+1}(\tau)=2\tau T_m(\tau) - T_{m-1}(\tau)$. This guarantees stable, orthogonal, low-degree polynomials.

---

## 3. Warm-Start Initialization from Native HexPlane Residuals

Given a pre-trained native 4DGS (W3) checkpoint:
1. Extract HexPlane displacement trajectories across $M$ uniform time steps:
   $$h_i(t_m) = p_i^{\text{hex}}(t_m) - p_i^{\text{canon}}, \quad m = 1, \dots, M$$
2. Identify candidate motion groups:
   - For $K=1$ (single mover): compute dominant temporal frequency $f_0$ or fit Chebyshev trajectory to high-variance Gaussians.
   - For $K \ge 1$: fit $K$ candidate trajectories $(R_k, T_k, c_k)$ by k-means / SVD on candidate rigid displacement fields.
3. Compute initial residual errors:
   $$E_{ik} = \frac{1}{M} \sum_{m=1}^M \| h_i(t_m) - g_{ik}(t_m) \|_2^2, \qquad E_{i0} = \frac{1}{M} \sum_{m=1}^M \| h_i(t_m) \|_2^2$$
4. Set initial soft responsibilities using temperature-scaled softmax:
   $$q_{ik} = \frac{\exp(-E_{ik} / 2\sigma^2)}{\exp(-E_{i0} / 2\sigma^2) + \sum_{j=1}^K \exp(-E_{ij} / 2\sigma^2)}$$
   Gaussians with small displacement $\|h_i\|_2 \approx 0$ receive $E_{i0} \approx 0 \implies q_{i0} \approx 1$ ($q_{ik} \approx 0$).

---

## 4. Optimization Schedule, EMA Updates & Detached Residual Penalty

### 4.1 Responsibility Parameterization
Responsibilities are parameterized by unconstrained per-Gaussian logits $z_i \in \mathbb{R}^K$, with the null component logit pinned at $z_{i0} \equiv 0$:
$$q_{ik} = \frac{\exp(z_{ik})}{1 + \sum_{j=1}^K \exp(z_{ij})}, \qquad q_{i0} = \frac{1}{1 + \sum_{j=1}^K \exp(z_{ij})}$$
This unconditionally guarantees $q_{ik} \ge 0$ and $\sum_{k=1}^K q_{ik} \le 1$.

### 4.2 Alternating Optimization & EMA Updates
To avoid instability between the HexPlane residual and component trajectory parameters:
- **Trajectory optimization:** Component trajectories $(R_k(t), T_k(t))$ and responsibility logits $z_i$ are optimized via Adam with learning rate $\mathrm{lr}_{\text{comp}}$.
- **EMA updates:** Every $I_{\text{ema}}$ iterations, target responsibilities $q_{ik}^*$ are evaluated from current residuals $E_{ik}$ vs $E_{i0}$ and blended via EMA:
  $$q_{ik} \leftarrow (1 - \beta) q_{ik} + \beta q_{ik}^*$$
  with $\beta \in [0.05, 0.2]$.

### 4.3 Gradual Ramp-In & Detached Residual Penalty
To smoothly transition motion from HexPlane to components without causing transient photometric loss spikes:
- Ramp factor $\alpha(\tau) \in [0, 1]$ over fine iterations $\tau \in [\tau_{\text{ramp\_start}}, \tau_{\text{ramp\_end}}]$:
  $$\alpha(\tau) = \operatorname{clip}\left(\frac{\tau - \tau_{\text{ramp\_start}}}{\tau_{\text{ramp\_end}} - \tau_{\text{ramp\_start}}}, 0, 1\right)$$
- Detached responsibility-weighted residual penalty:
  $$\mathcal{L}_{\text{res}} = \lambda_{\text{res}}(\tau) \sum_{i=1}^N \left(1 - \operatorname{stopgrad}(q_{i0})\right) \|h_i(t)\|_2^2 = \lambda_{\text{res}}(\tau) \sum_{i=1}^N \left(\sum_{k=1}^K \operatorname{stopgrad}(q_{ik})\right) \|h_i(t)\|_2^2$$
  where $\lambda_{\text{res}}(\tau) = \lambda_{\text{res,max}} \cdot \alpha(\tau)$.
  **Crucial stopgrad:** Detaching $q_{i0}$ (or $q_{ik}$) prevents the optimizer from cheating $\mathcal{L}_{\text{res}}$ by driving responsibilities to zero. The penalty specifically forces the HexPlane residual to shed motion where components explain it.

---

## 5. Densification & Pruning Lifecycle

In `core/scene/gaussian_model.py`:
- `self._motion_responsibilities` has shape $[N, K]$.
- `densify_and_split`: newly created points inherit parent responsibilities directly (`repeat(N, 1)`), preserving responsibility mass.
- `densify_and_clone`: cloned points inherit parent responsibilities directly (`[selected_pts_mask]`).
- `add_point_by_mask`: masked points inherit parent responsibilities (`[selected_pts_mask][mask]`).
- `densification_postfix`: concatenates new responsibilities to optimizer / model state.
- `prune_points`: slices `self._motion_responsibilities[valid_points_mask]`.
- Under no circumstances does densification reset responsibilities to zero / null.

---

## 6. Single Deformation-Dispatch API

A unified function `dispatch_deformation`:
```python
dispatch_deformation(
    pc: GaussianModel,
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    time: torch.Tensor,
    stage: str = "fine",
    motion_components: Optional[MotionComponents] = None,
    fourier_head: Optional[Any] = None,
    pipe: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```
is the sole gateway for deforming Gaussians across:
1. `core/gaussian_renderer/__init__.py:render()`
2. `core/utils/render_utils.py:get_state_at_time()`
3. `core/motion_amp/renderer.py:render_mod()`
4. `core/train.py`
5. `orchestrator/pipeline/vendored/cuda/train.py`

This guarantees exact numerical parity across training, offline rendering, trajectory export, and motion amplification.

