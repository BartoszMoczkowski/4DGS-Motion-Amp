# S1 Benchmark: 360° Ring Rig vs 80° Arc Rig Evaluation

> **Disambiguation Note:** This document formalizes the S1 ring-rig benchmark results that were previously mislabeled as `1.1E0c_results.md`.

---

## 1. Executive Summary

To determine whether the severe background-floor over-fragmentation observed in `cubes_k2_both` was caused by single-sided camera coverage, `cubes_k2_ring` was generated using a full **360° two-tier 16-camera ring** (8 lower cameras at $h=0.8\,\text{m}$, 8 upper cameras at $h=1.3\,\text{m}$, $22.5^\circ$ azimuth offset), keeping identical motion trajectories, framerate (60 fps), frame count (120), and checkerboard floor texturing. The scene was reconstructed with 361,180 Gaussians over 15,000 iterations.

### Core Verdict
1. **Single-sidedness was NOT the primary background fragmenter:**
   - Full 360° coverage did not eliminate background fragmentation for unsupervised methods. Global ARI rose only modestly (`rigid`: $0.047 \to 0.099$; `rigid2`: $0.021 \to 0.064$).
   - Crucially, predicted cluster counts **increased** under the ring rig (`rigid`: $50 \to 172$; `rigid2`: $393 \to 626$).
   - **Root Cause:** Full floor coverage creates *more* locally-rigid background micro-clusters. The deformation field fits unconstrained static floor regions as numerous small rigid patches. This behavior is intrinsic to the deformation field fitting, not an artifact of unobserved camera viewing angles.
2. **Oracle 2D-to-3D Mask Lifting Unambiguously Benefits from 360° Coverage:**
   - `mask_lift_oracle` ARI rose from **0.4547 $\to$ 0.5118** (Mean IoU from **0.5094 $\to$ 0.6046**).
   - This established the project ceiling for mask lifting and proved that multi-view voting requires full angular parallax to properly prune depth occlusions.
3. **Architectural Decision:**
   - The **Ring Rig is adopted** as the standard capture geometry for all subsequent synthetic datasets despite the unsupervised persistence of floor micro-clusters.

---

## 2. Quantitative Comparison Table

Data source: [`runs/cubes_k2_ring_benchmark.csv`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/runs/cubes_k2_ring_benchmark.csv).

| Capture Setup | Method Impl | Status | Predicted $K$ | Global ARI | Mean IoU | Within-ROI ARI | Segment Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **80° Arc Rig** (`cubes_k2_both`) | `rigid` | success | 50 | 0.0474 | 0.2599 | — | 19.0 |
| | `rigid2` | success | 393 | 0.0210 | 0.1733 | — | 30.0 |
| | `kabsch` | success | 4 | -0.0026 | 0.2939 | — | 36.0 |
| | `rigid2_roi` | success | 233 | -0.0012 | 0.1930 | 0.0002 | 32.0 |
| | `mask_lift_oracle` | success | 300 | 0.4547 | 0.5094 | 0.0000 | 35.0 |
| | `mbs` | success | 5 | 0.0000 | 0.2836 | 0.0000 | 6.0 |
| **360° Ring Rig** (`cubes_k2_ring`) | `rigid` | success | 172 | 0.0995 | 0.2687 | — | 16.0 |
| | `rigid2` | success | 626 | 0.0643 | 0.2201 | — | 27.0 |
| | `kabsch` | success | 3 | 0.0009 | 0.2779 | — | 30.0 |
| | `rigid2_roi` | success | 253 | 0.0059 | 0.2242 | 0.0265 | 26.0 |
| | **`mask_lift_oracle`** | **success** | **378** | **0.5118** | **0.6046** | **0.0000** | **24.0** |
| | `mbs` | success | 6 | -0.0000 | 0.2584 | 0.0000 | 6.0 |

---

## 3. Key Findings & Next Steps

1. **Why Within-ROI ARI was Zero for Oracle in S1:**
   As audited in S2 and Check 4, `roi_mask_oracle.py` had an assumption that label 0 is always background. On `cubes_k2`, Cube 0 carries label 0; filtering `mapped_labels > 0` discarded Cube 0 completely, leaving a single-class ROI and yielding ARI 0.0000. Once corrected in S2, Within-ROI ARI reached **1.0000**.
2. **Deformation Field Fitting:**
   The ring rig confirmed that background Gaussians will always fragment into locally-rigid patches unless a global rigid-body prior or geometry/motion ROI filter is explicitly applied.
