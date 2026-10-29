# W7 (T55) — Optional validated per-track trajectory supervision

**Status:** deferred (contingency only — enable only if W4/W5 evidence shows a remaining amplitude/phase deficiency)  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W6 (T54)  
**Experiment:** E5 (Dedicated single-factor ablation on top of best W4/W5 configuration)  
**Gate:** Off by default (`lambda_trajectory: 0.0`); cannot supervise scene-wide mean displacement; invalid/insufficient optical tracks produce zero loss with loud `INCONCLUSIVE` status rather than a fallback global objective.

---

## 1. Objective

Retain 2D trajectory supervision strictly as an optional, late-stage, per-track instrument—never as a scene-wide mean displacement loss (`r9` defect) and never gated by a binary 3D ROI box.

## 2. Implementation Steps (Only if Triggered After W6)

1. **Remove scene-wide mean fallback in [`core/utils/trajectory_loss.py`](../../core/utils/trajectory_loss.py#L234-L268):**
   - Delete the `canonical_xyz[::step]` 500-point scene-wide subsampling fallback permanently.
   - Each supervision term must bind a specific quality-gated 2D track $m$ in camera $k$ to its specific 3D Gaussian neighborhood / soft component responsibility support along the camera ray at the track's anchor pixel $(u_{k,m}, v_{k,m})$, weighted by opacity/transmittance and soft component responsibility.
2. **Strict optical track quality gating:**
   - Before any track enters the loss, require minimum amplitude ($A_{\text{px}} \ge A_{\min}$), harmonic fit quality ($R^2 \ge R^2_{\min}$), temporal coverage, and absence of phase-correlation drift.
   - Exclude rejected tracks with logged reasons; if fewer than `min_valid_tracks` pass, set loss to `0.0` and log status `INCONCLUSIVE_INSUFFICIENT_TRACKS`.
3. **Normalized displacement units & confidence weighting:**
   - Supervise 2D displacement normalized by image dimensions (or focal length) at the camera's exact corrected timestamp $t_{k, i}$, weighted by normalized track confidence.

## 3. Verification & Tests

- **Locality isolation test:** Verify that a single active turbine track has zero gradient on Gaussians outside the track's ray/component support and can never induce global background motion.
- **Empty/invalid track safety test:** Verify that passing low-$R^2$ or static tracks produces exact `0.0` loss and an explicit `INCONCLUSIVE` diagnostic without falling back to arbitrary scene points.

## 4. Deliverables & Stop Conditions

- **Deliverables:** Refactored per-track quality-gated `TrajectoryReprojectionLoss`, unit tests (`orchestrator/tests/test_w7_validated_trajectory_loss.py`), and single-factor ablation report (`E5` vs best of `E3`/`E4`).
- **Stop conditions:** Disable immediately if static-region motion increases, if learnable camera offsets drift to absorb association error, or if held-out PSNR/SSIM regresses beyond tolerance.

