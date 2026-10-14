# Capture-Limitation Scene Series: Measurement Plan & Pre-Registered Gates

> **Document Status:** Active Protocol  
> **Target Series:** Series A (Motion Deviant), Series R (Capture Ladder), Series T (Material Ablation), Series F (Final Combined)  
> **Reference Model:** `pump01` (107 CAD parts, 106 movable bodies + static `frame_base`)  
> **Related Manifest:** [`docs/capture_limitation_scenes_manifest.csv`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/docs/capture_limitation_scenes_manifest.csv)  
> **Contact Sheet:** [`docs/validation_contact_sheet.png`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/docs/validation_contact_sheet.png)  

---

## 1. Executive Summary & Problem Formulation

Baseline motion segmentation on `pump01` suffered from severe boundary leakage and floater over-clustering. Crucially, the historical amplitude grid sweep (cells `M2`, `M4`, `M8` representing $2\times, 4\times, 8\times$ motion multiplier) yielded virtually flat trajectory separability:
$$\text{AUROC}(M2) = 0.576, \quad \text{AUROC}(M4) = 0.561, \quad \text{AUROC}(M8) = 0.597$$
A four-fold increase in physical motion amplitude bought no gain in separability. This establishes the prior hypothesis that **motion amplitude alone is not the primary binding constraint**. Instead, segmentation performance is governed by an interplay between:
1. **Capture Resolution & Target Pixel Density ($L_{\text{px}}$)**: Base physical displacement ($\sim 1\text{ mm}$) projects to only $\sim 1\text{ px}$ at $1600 \times 900$ at baseline distance ($7.5\text{ m}$). Discrete seed survival through the min-15 Gaussian filter requires higher Gaussian density on small parts.
2. **Material & Texture Realism**: Current appearance separability ($\text{AUROC} \approx 0.705$, color gating $0.496$) relies entirely on synthetic pastel `displayColor` boundaries. Real industrial machines exhibit uniform metallic alloy finishes where boundaries have zero color contrast, requiring high-frequency surface texture (Diamond et al. 2017) to provide local optical flow and photometric constraints.

This experimental series isolates these three orthogonal axes into 18 parametric cells:
* **Series A (`sA0`–`sA5`)**: Deviant-part motion detectability under controlled common-mode vibration.
* **Series R (`sR0`–`sR5`)**: Capture ladder separating sensor resolution ($1600 \to 3000\text{ px}$) from geometric proximity ($D \to D/2$).
* **Series T (`sT0`–`sT2`)**: Material realism ablation (synthetic pastel $\to$ uniform steel PBR $\to$ procedural sub-pixel texture).
* **Final Combined (`sF0`–`sF2`)**: Compound cells synthesizing deviant motion, metallic PBR, and close-range capture.

---

## 2. Quantitative Readout Metrics

Every cell in the series must report the standardized evaluation tuple:

| Metric | Code Symbol / Method | Purpose / Scope | Target Threshold |
|---|---|---|---|
| **Interface Edge AUROC** | `paired_interface_edge_auroc` | Phasor $f_0$ and raw trajectory AUROC on paired boundary Gaussian edges across part interfaces (A16 protocol). Evaluated separately on Cat A (contact $< 0.2\text{ mm}$) and Cat B (clearance gap). | $\ge 0.80$ (Separable) |
| **Deviant-Part IoU & F1** | `compute_part_iou_f1` | Precision, recall, $F_1$, and IoU for target part (`part_019`) against ground truth canonical mask. | $F_1 \ge 0.85$, $\text{IoU} \ge 0.75$ |
| **Discrete Seed Count & Size** | `extract_cluster_seeds` | Count of spatial CC / rigidity clusters surviving the discrete min-15 Gaussian filter (M1 metric), plus median seed Gaussian count. Directly monitors backbone extinction. | Surviving seeds $\ge 60$ |
| **Frozen Backbone ROI ARI** | `compute_within_roi_ari` | Adjusted Rand Index evaluated strictly within the dynamic ROI (excluding static `frame_base` background). | $\text{ARI} \ge 0.70$ |
| **Cat A Split Rate** | `compute_pairs_split_rate(cat_a)` | Fraction of the 39 baseline-merged contact part-pairs ($CAD < 0.2\text{ mm}$) correctly separated into distinct clusters. | Split Rate $\ge 0.50$ |
| **Cat B Split Rate** | `compute_pairs_split_rate(cat_b)` | Fraction of the 102 clearance-bridged part-pairs ($CAD \ge 0.2\text{ mm}$) correctly separated. | Split Rate $\ge 0.75$ |
| **Appearance Edge AUROC** | `appearance_edge_auroc` | Cosine feature / color distance AUROC across part interfaces, quantifying appearance boundary sharpness. | Monitored (T0 vs T1 vs T2) |

---

## 3. Pre-Registered Decision Gates

To prevent post-hoc rationalization, all hypotheses and passing criteria are pre-registered below:

```mermaid
flowchart TD
    subgraph Series A [Motion Deviant Ladder]
        A[Series A: sA0..sA5] --> Gate1{Gate 1: AUROC >= 0.80 at B*?}
        Gate1 -- Yes: B* <= 5A --> G1_Pass[Amplitude-Limited Failure Proven: SNR floor identified]
        Gate1 -- No: AUROC < 0.70 at 20A --> G1_Fail[Disproves Amplitude Hypothesis: Bound by Resolution/Texture]
    end

    subgraph Series R [Capture Proximity vs Sensor]
        R[Series R: sR0..sR5] --> Gate2{Gate 2: |AUROC R2 - AUROC R3| <= 0.03?}
        Gate2 -- Yes --> G2_Equiv[Equivalence Confirmed: Proximity substitutes 6 MP at 1/4 compute]
        Gate2 -- No: R2 >> R3 --> G2_Res[Sensor Megapixels Dominant: High-res required]
    end

    subgraph Series T [Material Realism & Texture]
        T[Series T: sT0..sT2] --> Gate3{Gate 3: T2 recovers >= 70% of T0-T1 drop?}
        Gate3 -- Yes --> G3_Text[Sub-pixel Texture Restores Tracking: Diamond hypothesis holds]
        Gate3 -- No --> G3_Fail[Texture Insufficient: PBR geometry ambiguity remains]
    end

    subgraph Final Combined [Real Deployment]
        G1_Pass & G2_Equiv & G3_Text --> F1[Run F1 & F2 Compound Honest Cells]
        F1 --> Gate4{ROI ARI >= 0.75 & Cat A Split >= 0.60?}
        Gate4 -- Yes --> Success[Production Pipeline Validated for Physical Machine]
        Gate4 -- No --> Refine[Requires Active Illumination / Structured Light]
    end
```

### Gate 1: Deviant-Part Separability Threshold $B^*$ (Series A)
* **Pre-Registered Hypothesis:** Under rigid whole-assembly common-mode vibration ($A = 1.0\text{ mm}$ world), increasing target part deviant amplitude $B$ introduces interface shear $|B - A|$. Deviant part edge AUROC will cross $\ge 0.80$ at a critical threshold $B^*$.
* **Passing Criterion:** $\text{AUROC}_{\text{deviant}}(B^*) \ge 0.80$.
* **Interpretation:**
  * If $B^* \le 5A$ ($5.0\text{ mm}$ world, $4\text{ mm}$ shear): The baseline failure on `pump01` was purely motion SNR-limited.
  * If $\text{AUROC} < 0.70$ even at $B = 20A$ ($20\text{ mm}$ world, $19\text{ mm}$ shear): Rejection of the amplitude hypothesis. Boundary leakage is fundamentally caused by discrete Gaussian blur or spatial connectivity bridging, not insufficient motion energy.

### Gate 2: Sensor Resolution vs Proximity Equivalence Test (Series R)
* **Pre-Registered Hypothesis:** Sensor resolution (R2: $3000 \times 2000$ at $D = 2.5$) and geometric proximity (R3: $1600 \times 900$ at $D = 1.25$) both provide $\approx 2.0\times$ pixel motion on target. They are hypothesized to achieve equivalent trajectory AUROC and segmentation ARI:
  $$|\text{AUROC}(R2) - \text{AUROC}(R3)| \le 0.03, \quad |\text{ARI}_{\text{ROI}}(R2) - \text{ARI}_{\text{ROI}}(R3)| \le 0.02$$
* **Passing Criterion:** Equivalence within the pre-registered tolerance bands.
* **Compute Implication:** R3 renders at $1600 \times 900$ (4.5 GPU min, 35 min train), while R2 renders at 6 MP (18.8 GPU min, 55 min train). If Gate 2 passes, physical robot / drone multi-camera captures should prioritize closer stand-off distance over expensive 6 MP sensors, saving $\sim 75\%$ capture and training compute.
* **Ceiling Check (R4):** R4 ($3000 \times 2000$ + half distance, $\sim 3.8\times$ pixel motion) establishes the experimental upper bound.

### Gate 3: Material Realism Tax & Sub-pixel Texture Recovery (Series T)
* **Pre-Registered Hypothesis:** Moving from synthetic pastel `displayColor` (T0) to uniform brushed steel alloy PBR (T1) without per-part color boundaries will cause a severe drop in appearance edge AUROC ($\Delta_{\text{appearance}} = \text{AUROC}(T0) - \text{AUROC}(T1) \ge 0.20$). Procedural surface texture (T2: lathe marks, rust patina, decals) will restore $\ge 70\%$ of this loss:
  $$\frac{\text{AUROC}(T2) - \text{AUROC}(T1)}{\text{AUROC}(T0) - \text{AUROC}(T1)} \ge 0.70$$
* **Passing Criterion:** Texture recovery fraction $\ge 0.70$.
* **Scientific Value:** Directly validates the Diamond et al. (2017) sub-pixel texture requirement for Eulerian/Lagrangian motion magnification on specular industrial CAD components.

### Gate 4: Honest Physical Deployment Feasibility (Series F)
* **Pre-Registered Hypothesis:** Cell F1 ($B = 10A \times \text{T1 metallic} \times \text{R4 capture}$) and Cell F2 ($B = 10A \times \text{T2 textured} \times \text{R4 capture}$) represent the true operating conditions of physical pump vibration diagnostics.
* **Passing Criterion:**
  $$\text{ARI}_{\text{within-ROI}} \ge 0.75, \quad \text{Cat A Split Rate} \ge 0.60, \quad \text{Deviant Part } F_1 \ge 0.85$$

---

## 4. Execution Staging & Compute Budget

Execution of the full series is gated by Stage 1 dry-run approval. When green-lit, execution follows strict cost-ordering:

```
Batch 1: Series A (6 cells, 1600x900, 60 frames) -> ~27 GPU min capture, ~210 GPU min train
Batch 2: Series T (3 cells, 1600x900, 60 frames) -> ~16 GPU min capture, ~105 GPU min train
Batch 3: Series R (6 cells, res/dist ladder)       -> ~96 GPU min capture, ~315 GPU min train
Batch 4: Series F (3 compound cells)              -> ~51 GPU min capture, ~145 GPU min train
-----------------------------------------------------------------------------------------
TOTAL BUDGET:                                       ~190 GPU min capture (~3.2 h)
                                                    ~775 GPU min train   (~12.9 h)
```

### Invariant Constraints
1. **Frame Rate & Length:** All cells maintain $60\text{ fps} \times 60\text{ frames}$ ($1.0\text{ s}$ clip) except R5 ($120\text{ frames}$, testing frequency bin resolution $\Delta f = 0.5\text{ Hz}$).
2. **Training Iterations:** Fixed at 30,000 iterations (coarse static + fine deformation) across all cells to prevent iteration-confounding.
3. **Ground-Truth Label 0:** `frame_base` is strictly assigned instance label 0 (static background).
4. **TimeCode Convention:** Point cloud sampling and ground-truth geometry export are evaluated strictly at `TimeCode(0)`.
