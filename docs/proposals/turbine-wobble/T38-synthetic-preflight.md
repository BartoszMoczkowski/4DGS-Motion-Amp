# T38 — Synthetic sync pre-flight scene

**Status:** done
**Phase:** 12 / Phase 0.5 (pre-flight)
**Depends on:** —
**Cross-cutting:** gates the T39 retrain

## Goal

Before burning real retrains, prove the modified pipeline end-to-end on a toy
scene where ground truth is known.

## Design & Implementation

1. **Synthetic Scene Generator (`scripts/gen_synthetic_wobble_scene.py`):**
   - Pure-Python, deterministic generator constructing a 10-camera multi-view sequence in `data/multipleview/synthetic_turbine_preflight/`.
   - Replicates the 2-tier Dahua capture rig geometry: 5 lower cameras (`cam01`..`cam05` at $z = 0.35$ m, azimuths $-48^\circ$ to $+48^\circ$) and 5 upper cameras (`cam06`..`cam10` at $z = 0.65$ m, azimuths $-42^\circ$ to $+44^\circ$) focused on target center $(0, 0, 0.45\text{ m})$.
   - Implements known rigid 7.112 Hz linear oscillation along world X-axis with calibrated physical amplitude ($A = 3.6$ mm) yielding $\approx 1.87$ to $2.80$ px projected 2D amplitude across camera viewpoints (matching real turbine wobble).
   - Injected measured RTSP capture offsets $\delta_k \in [-22.6\text{ ms}, +23.0\text{ ms}]$ from T32 archaeology (`cam01` anchor at 0.0 ms). Continuous frame sampling: $t_k(i) = \frac{i}{fps} + \delta_k$.
   - Output files:
     - `cam01/` .. `cam10/` (60 frames each at 20.0 fps, 1600x900 resolution)
     - `sparse_/` (`cameras.bin`, `images.bin`, `points3D.bin`)
     - `points3D_multipleview.ply` (1853 vertices: 583 dynamic rotor points + 1270 static base/frame points)
     - `poses_bounds_multipleview.npy` (shape `[10, 17]`)
     - `offsets.json` (authoritative schema matching T32)
     - `gt_motion_info.json` (ground truth parameters)
     - `motion_masks/` (precomputed masks matching T36 format)

2. **Repeatable Pre-Flight Script (`scripts/run_synthetic_preflight.py`):**
   - Automated diagnostic runner evaluating all 5 pre-registered gates.
   - Computes sub-pixel template tracking, 1D PCA projection, and sinusoidal phase fitting on rendered frames.
   - Produces structured validation record `preflight_record.json` and human-readable audit table.

3. **Orchestrator Presets & Configs:**
   - `core/arguments/multipleview/synthetic_turbine_preflight.py`: Model hidden & optimization params wiring T35 (anneal iter 3000), T36 (motion mask loss), T37 (coarse iterations 500).
   - `orchestrator/pipeline/config/presets/synthetic_turbine_preflight.yaml`: Orchestrator YAML preset extending `real_capture`.

## Pre-Registered Gate Verification Results

| Gate | Criterion | Target Bar | Measured Result | Status |
|------|-----------|------------|-----------------|--------|
| **G1** | Spectral Peak Test | Dominant FFT peak at $7.112 \pm 0.10$ Hz | 7.112 Hz across all cameras | **PASS** |
| **G2** | Offset Accuracy | $\max_k \|\hat{\delta}_k - \delta_k\| \le 2.0$ ms | Max disc: **0.47 ms** (Mean: **0.27 ms**) | **PASS** |
| **G3** | Amplitude Fidelity | $\max_k \|\hat{A}_k - A_k\| / A_k \le 30\%$ | Max error: **14.1%** (Mean: **5.4%**) | **PASS** |
| **G4** | Motion Mask Isolation | Active fraction in 2–20%, background leakage $< 1\%$ | Active: 6.1%–6.9%, leak: 0.0% | **PASS** |
| **G5** | Cross-View Phase Coherence | Residual phase error under T39 correction $< 3^\circ$ | Max residual: **$1.20^\circ$** | **PASS** |

### Per-Camera Offset & Amplitude Breakdown

| Camera | Injected $\delta_k$ (ms) | Recovered $\hat{\delta}_k$ (ms) | Discrepancy (ms) | Recovered Amp (px) | Expected Amp (px) | Status |
|--------|--------------------------|---------------------------------|------------------|--------------------|-------------------|--------|
| `cam01` | 0.00 | 0.00 | 0.00 | 1.61 | 1.87 | **PASS** |
| `cam02` | +18.00 | +18.47 | +0.47 | 2.35 | 2.56 | **PASS** |
| `cam03` | -8.60 | -8.13 | +0.47 | 2.67 | 2.80 | **PASS** |
| `cam04` | +23.00 | +23.06 | +0.06 | 2.32 | 2.56 | **PASS** |
| `cam05` | -22.60 | -22.44 | +0.16 | 1.66 | 1.87 | **PASS** |
| `cam06` | +20.00 | +20.42 | +0.42 | 2.10 | 2.08 | **PASS** |
| `cam07` | -3.50 | -3.22 | +0.28 | 2.59 | 2.63 | **PASS** |
| `cam08` | -22.00 | -21.74 | +0.26 | 2.76 | 2.79 | **PASS** |
| `cam09` | -16.00 | -15.69 | +0.31 | 2.57 | 2.60 | **PASS** |
| `cam10` | +12.00 | +12.28 | +0.28 | 2.06 | 2.01 | **PASS** |

## Deliverables & Artifacts

1. `scripts/gen_synthetic_wobble_scene.py`: Parametric synthetic multi-view scene generator.
2. `data/multipleview/synthetic_turbine_preflight/`: Complete 10-camera dataset with frame renders, COLMAP binary model, PLY point cloud, and T32/T36 sidecars.
3. `scripts/run_synthetic_preflight.py`: Command-line pre-flight gate runner and manifest reporter.
4. `core/arguments/multipleview/synthetic_turbine_preflight.py`: 4DGS config with R1, R2, R10.
5. `orchestrator/pipeline/config/presets/synthetic_turbine_preflight.yaml`: Orchestrator YAML preset.
6. `orchestrator/tests/test_synthetic_preflight.py`: Automated test suite (7 passed in 3.95s).
7. `data/multipleview/synthetic_turbine_preflight/preflight_record.json`: Auditable pre-flight record.

**Verdict:** Gating Gate G2 for Phase 2 retrains (T39) is **PASSED (GREEN)**.

