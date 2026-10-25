# Turbine Wobble Recovery — task board

> **Scope:** recover the measured 7.11 Hz, ±2.5–3 px rigid turbine wobble in the
> 4DGS reconstructions of the `lego_turbine_*` / `turbine_r1_missing_6` captures.
> Registered as docs proposals (not orchestrator tasks) because the work is in
> the data pipeline and the 4DGS core, not the orchestrator itself.
>
> **Source documents:**
> [`../../docs/turbine_wobble_codebase_brief.md`](../turbine_wobble_codebase_brief.md) —
> verified codebase evidence (file:line for every claim);
> [`../4DGS_Turbine_Wobble_Solutions.md`](../4DGS_Turbine_Wobble_Solutions.md) —
> solution architecture (layers L1–L5, remedies R1–R11);
> [`../turbine_wobble_fix_plan.md`](../turbine_wobble_fix_plan.md) — phased plan,
> gates, validation protocol, ablation discipline.

## Status board

| ID | Title | Phase | Depends on | Status |
|----|-------|-------|-----------|--------|
| T32 | Timing recovery: sidecar archaeology + `offsets.json` | 12 | — | done |
| T33 | `track_wobble_phase.py` optical offset fitter | 12 | — | done |
| T34 | Flash / common-event ambiguity resolution | 12 | T33 | todo |
| T35 | Regularizer schedule presets (R1) | 12 | — | done |
| T36 | Motion-mask weighted loss (R2) | 12 | — | done |
| T37 | Coarse-stage gating (R10) | 12 | — | done |
| T38 | Synthetic sync pre-flight scene | 12 | — | todo |
| T39 | Per-camera timestamps in loader (R6) | 12 | T32 or T33 | todo |
| T40 | Learnable offsets + trajectory reprojection loss (R7+R8) | 12 | T39, T36 | todo |
| T41 | Fourier-rigid deformation head (R9) | 12 | T39 | todo |
| T42 | Cross-scene validation & amplitude-ordering report | 12 | T40, T41 | todo |

## Dependency graph

```
T32 ─┐
     ├─► T39 ─┬─► T40 ─┐
T33 ─┘        │        ├─► T42
T33 ──► T34   └─► T41 ─┘
T35, T36, T37 (infra, land first; T36 also gates T40)
T38 (synthetic pre-flight) ─ gates the T39 retrain
```

## Execution phases (from the plan)

1. **Phase 0 — timing recovery (no GPU):** T32, T33, T34. Exit: `offsets.json`
   per scene (fps, exposure, per-camera offsets, drift), hashed into every
   downstream run manifest. Includes the 164/160/261 frame-count reconciliation.
2. **Phase 0.5 — synthetic pre-flight:** T38; gates the Phase-2 retrain.
3. **Phase 1 — config & loss quick wins (one retrain):** T35, T36, T37.
4. **Phase 2 — the timestamp fix (pivotal):** T39, then T40. Gate G2:
   ROI-averaged trajectory FFT peaks at 7.11 Hz.
5. **Phase 3 — structured motion:** T41, then T42 (amplitude must increase
   with missing-blade count across base / 2-missing / 5-missing).

## Global rules

- **Validation metrics (every retrain):** (1) spectral peak at 7.11 Hz;
  (2) amplitude fidelity vs optical tracks (±30 %); (3) cross-view phase
  coherence; (4) offset accuracy ≤ 2 ms where ground truth exists;
  (5) held-out PSNR/SSIM guardrail vs the static-collapse baseline.
- **Single-factor ablation ladder** after Phase 2: R6 → R2 → R1 → R10 → R7 →
  R8 → R9, one factor per run.
- **Every config = a named preset** under
  `orchestrator/pipeline/config/presets/lego_turbine_*.yaml`; every manifest
  records the hash of the `offsets.json` consumed.
- **Never enable learnable offsets (T40) unguarded:** cam01 anchor, ±40 ms
  clamp, and the trajectory loss must be in place first.
- Frame-count discrepancy, exposure times, and drift are Phase-0 deliverables,
  not afterthoughts.
