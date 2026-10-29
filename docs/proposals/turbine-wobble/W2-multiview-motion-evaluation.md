# W2 (T50) — Multi-view frequency-domain & trajectory motion evaluation

**Status:** done  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W1 (T49)  
**Gate before W3:** Evaluation tool recovers injected synthetic amplitude/phase within numerical tolerance, fails oscillators with wrong amplitude/phase despite matching frequency, and rejects low-quality optical tracks (`R²`, visibility, divergence) loudly instead of writing `"validated"`.

---

## 1. Objective

Build a maintained, quality-gated multi-view motion evaluation tool before interpreting `E1a`/`E1b` or adding motion supervision. A spectral peak at a configured frequency is never sufficient by itself; the evaluator must compare complex amplitude and phase $Z = A e^{i\phi}$ between real captures and rendered sequences on the exact corrected per-camera timebase, alongside ROI-free 3D trajectory diagnostics and held-out PSNR/SSIM guardrails.

## 2. Codebase Findings & Gaps Addressed in W2

1. **False `"validated"` status in [`scripts/track_wobble_phase.py`](../../scripts/track_wobble_phase.py):**
   - Hardened with explicit per-camera and cross-rig quality gating (`MIN_R2_THRESHOLD = 0.50`, `MIN_AMPLITUDE_PX = 0.05`, `MAX_ALLOWED_INDIVIDUAL_DISC_MS = 20.0`, `MAX_ALLOWED_MEAN_DISC_MS = 15.0`).
   - Any camera with $R^2 < 0.50$ or excessive discrepancy is flagged `REJECTED_LOW_R2` / `REJECTED_DISCREPANCY`.
   - Scenes like `turbine_r1_missing_6` and `lego_turbine_2_missing` now fail validation loudly with `rejected_low_r2` or `rejected_discrepancy` and recorded reasons, never silently writing `"validated"`.
2. **Single-camera FFT bin evaluation in [`scratch/eval_r11_cam01.py`](../../scratch/eval_r11_cam01.py):**
   - Replaced by full multi-camera evaluator [`scripts/eval_multiview_wobble.py`](../../scripts/eval_multiview_wobble.py) supporting streaming frame evaluation across all available cameras (cam01..cam10), per-camera exact timebases (offsets + drift), and least-squares harmonic complex phasor estimation at $f_0$.

## 3. Implementation Summary

1. **Corrected-timestamp rendering & sampling:**
   - Evaluates cameras at exact physical exposure instants $t_{k, i} = i/\text{fps} + \Delta t_k + \delta_k \cdot (i/\text{fps})$ from `offsets.json`.
   - Streams frames generator-by-generator from disk without full dataset accumulation to avoid 6 MP VRAM/RAM regressions.
2. **Least-squares harmonic complex projection at $f_0 = 5.691\text{ Hz}$:**
   - 1D principal axis projection aligns render tracks against the real reference eigenvector to prevent arbitrary PCA sign flips.
   - Fits $u(t) \approx c_0 + c_1 \cos(2\pi f_0 t) + c_2 \sin(2\pi f_0 t)$ to extract $Z = c_1 - i c_2 = A e^{i\phi}$ where $A = \sqrt{c_1^2 + c_2^2}$ and $\phi = \operatorname{atan2}(-c_2, c_1)$.
3. **Per-patch and per-camera metrics:**
   - Real & render amplitudes $A_{\text{real}}, A_{\text{render}}$, amplitude ratio $A_{\text{render}} / A_{\text{real}}$, absolute amplitude error.
   - Circular phase difference $\Delta \phi = \operatorname{atan2}(\sin(\phi_{\text{render}} - \phi_{\text{real}}), \cos(\phi_{\text{render}} - \phi_{\text{real}}))$, phase error in degrees and equivalent time error in milliseconds.
   - Complex vector error $\epsilon_Z = |Z_{\text{render}} - Z_{\text{real}}| / |Z_{\text{real}}|$.
   - Image PSNR, SSIM, and temporal standard-deviation (tstd) ratio maps.
   - Quality statuses: `PASS`, `FAIL_AMPLITUDE`, `FAIL_PHASE`, `FAIL_COMPLEX_ERR`, `REJECTED_LOW_R2`, `REJECTED_LOW_AMP`, `REJECTED_DIVERGENCE`.
4. **ROI-free 3D native-HexPlane trajectory diagnostics:**
   - Computes 3D harmonic displacement amplitudes across all Gaussians.
   - Quantiles: p50, p90, p99, and max displacement.
   - Gini concentration index $G \in [0, 1]$ and top-decile spatial spread ratio.
   - Phase coherence $R = |\frac{1}{M}\sum e^{i\phi_m}| \in [0, 1]$ among top moving Gaussians.
   - Background drift ratio $p50 / p90$ with automatic detection of whole-scene unconstrained drift (`global_drift_detected`).
5. **Strict stopping rules:**
   - Single-camera evaluation is barred from declaring a scene pass (`INCONCLUSIVE_SINGLE_CAMERA`).
   - FFT frequency match without complex amplitude/phase agreement fails gate.

## 4. Verification & Acceptance Tests

Unit test suite in [`orchestrator/tests/test_w2_multiview_eval.py`](../../orchestrator/tests/test_w2_multiview_eval.py) verifies:
- **Exact synthetic phasor recovery:** Recovers known $A$ and $\phi$ within $< 10^{-4}$ numerical tolerance on non-uniform drift/offset timestamps.
- **Adversarial false-positive rejection:** Detects and fails renders with $0.4\times$ amplitude or $60^\circ$ phase error even when oscillating at exact $5.691\text{ Hz}$.
- **Static & noisy track rejection:** Flags $A_{\text{real}} \approx 0$ as `REJECTED_LOW_AMP` and $R^2 < 0.50$ as `REJECTED_LOW_R2`.
- **ROI-free 3D diagnostics:** Distinguishes localized rotor wobble (Gini $> 0.80$, drift false) from scene-wide uniform drift (Gini $< 0.15$, drift true).
- **Stopping rules:** Prohibits single-camera passing and verifies multi-camera consensus gating.
- **End-to-end pipeline:** Multi-camera synthetic scene evaluation produces valid JSON report and passes consensus.

## 5. Exact Commands to Evaluate E1a vs E1b

```bash
# 1. Run multi-view evaluation on E1a (fixed sidecar timestamps):
uv run python scripts/eval_multiview_wobble.py \
    --scene data/multipleview/turbine_r1_missing_6 \
    --render-dir runs/turbine_r1_missing_6_w1_e1a/render_out \
    --trajectories runs/turbine_r1_missing_6_w1_e1a/trajectories.npz \
    --out-json runs/turbine_r1_missing_6_w1_e1a/eval_w2.json \
    --human

# 2. Run multi-view evaluation on E1b (guarded learnable offsets):
uv run python scripts/eval_multiview_wobble.py \
    --scene data/multipleview/turbine_r1_missing_6 \
    --render-dir runs/turbine_r1_missing_6_w1_e1b/render_out \
    --trajectories runs/turbine_r1_missing_6_w1_e1b/trajectories.npz \
    --out-json runs/turbine_r1_missing_6_w1_e1b/eval_w2.json \
    --human

# 3. Direct comparison between E1a and E1b metrics:
# Compare mean_amp_ratio, mean_phase_err_deg, mean_complex_vector_err, and 3D Gini index
# across eval_w2.json files.
```

