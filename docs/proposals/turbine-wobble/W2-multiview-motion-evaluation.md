# W2 (T50) — Multi-view frequency-domain & trajectory motion evaluation

**Status:** todo  
**Phase:** 14 (Segmentation-Free Wobble Recovery)  
**Depends on:** W1 (T49)  
**Gate before W3:** Evaluation tool recovers injected synthetic amplitude/phase within numerical tolerance, fails oscillators with wrong amplitude/phase despite matching frequency, and rejects low-quality optical tracks (`R²`, visibility, divergence) loudly instead of writing `"validated"`.

---

## 1. Objective

Build a maintained, quality-gated multi-view motion evaluation tool before interpreting `E1a`/`E1b` or adding motion supervision. A spectral peak at a configured frequency is never sufficient by itself; the evaluator must compare complex amplitude and phase $Z = A e^{i\phi}$ between real captures and rendered sequences on the exact corrected per-camera timebase, alongside ROI-free 3D trajectory diagnostics and held-out PSNR/SSIM guardrails.

## 2. Codebase Findings & Gaps to Address in W2

1. **False `"validated"` status in [`scripts/track_wobble_phase.py`](../../scripts/track_wobble_phase.py):**
   - Previously, `optical_offsets.json` recorded `"status": "validated"` despite 21.6–40.6 ms disagreement with sidecars (`lego_turbine_2_missing`) and $R^2 < 0.05$ on 6/10 cameras (`turbine_r1_missing_6`).
2. **Single-camera FFT bin evaluation in [`scratch/eval_r11_cam01.py`](../../scratch/eval_r11_cam01.py):**
   - Existing scratch evaluation only checked `cam01`, rendered at nominal uniform frame times, and took the nearest discrete FFT bin (`5.67 Hz`, bin width `0.077 Hz`) without checking phase residuals or cross-camera consistency.

## 3. Implementation Steps

1. **Corrected-timestamp rendering & sampling:**
   - Render each camera $k$ at its exact corrected physical/normalized exposure instants $t_{k, i}$ (streaming one frame at a time to avoid 6 MP frame-list OOM).
2. **Least-squares harmonic complex projection at $f_0 = 5.691\text{ Hz}$:**
   - For each patch track $u(t_{k,i})$ (after principal-axis projection with consistent sign convention against the real reference track), estimate complex phasor $Z = A e^{i\phi}$ via least-squares fit to $[1, \cos(2\pi f_0 t_{k,i}), \sin(2\pi f_0 t_{k,i})]$ on the actual (potentially non-uniform / drift-corrected) sample times $t_{k,i}$.
3. **Per-patch and per-camera metrics:**
   - Real amplitude $A_{\text{real}}$, render amplitude $A_{\text{render}}$, amplitude ratio $A_{\text{render}} / A_{\text{real}}$, absolute error $|A_{\text{render}} - A_{\text{real}}|$.
   - Circularly wrapped phase residual $\Delta \phi = \operatorname{atan2}(\sin(\phi_{\text{real}} - \phi_{\text{render}}), \cos(\phi_{\text{real}} - \phi_{\text{render}}))$.
   - Complex vector error $|Z_{\text{render}} - Z_{\text{real}}| / |Z_{\text{real}}|$.
   - Fit $R^2$, track correlation quality, sample count, and explicit status (`VALID`, `REJECTED_LOW_R2`, `REJECTED_LOW_AMP`, `REJECTED_TRACK_DIVERGENCE`, `INCONCLUSIVE`).
   - Per-camera held-out PSNR/SSIM and temporal standard-deviation ratio maps.
4. **ROI-free 3D native-HexPlane trajectory diagnostics:**
   - Without any binary ROI, summarize the 3D Gaussian trajectory field at $f_0 = 5.691\text{ Hz}$: spatial concentration (Gini / top-decile localization vs whole-scene bbox spread), p50/p90/p99 3D displacement amplitudes, phase coherence among top-moving Gaussians, and background/global drift check.

## 4. Verification & Acceptance Tests

- **Exact synthetic phasor recovery:** Known synthetic signals with non-uniform timestamps, drift, and per-camera offsets recover $A$ and $\phi$ within $< 10^{-4}$ numerical tolerance.
- **Adversarial false-positive rejection:** A synthetic render oscillating at the exact target frequency $5.691\text{ Hz}$ but with $0.5\times$ amplitude or $60^\circ$ phase error **fails** the complex real-vs-render gate even though its FFT peak is at $5.691\text{ Hz}$.
- **Static & noisy track rejection:** Static input ($A_{\text{real}} \approx 0$) or noisy tracks ($R^2 < R^2_{\min}$) produce `REJECTED` / `INCONCLUSIVE`, never `PASS` or `"validated"`.

## 5. Deliverables & Stop Conditions

- **Deliverables:** Maintained evaluation module and CLI (`scripts/eval_multiview_wobble.py`), unit test suite (`orchestrator/tests/test_w2_multiview_eval.py`), machine-readable JSON schema + summary table/plot generator, and exact commands to evaluate `E1a` vs `E1b`.
- **Stop conditions:** Never declare a scene passed from a single `cam01` patch or from an FFT frequency peak alone; never infer timing-sidecar error from rejected optical tracks.

