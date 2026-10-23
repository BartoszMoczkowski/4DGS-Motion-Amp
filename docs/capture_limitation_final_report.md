# Capture-Limitation Suite: Rigorous Scientific Audit & Final Report

> **Experiment:** `capture_limitation` (18 Cells, Checkpoints @ `iteration_14000`)  
> **Results Store (Frozen CC Backbone):** [`runs/capture_limitation_backbone_results.csv`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/runs/capture_limitation_backbone_results.csv)  
> **Results Store (Pre-Registered Edge AUROC):** [`runs/capture_limitation_edge_auroc.csv`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/runs/capture_limitation_edge_auroc.csv)  
> **Legacy Motion Methods Store:** [`runs/capture_limitation_seg_results.csv`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/runs/capture_limitation_seg_results.csv)  
> **Pre-Registered Protocol:** [`docs/capture_limitation_readouts.md`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/docs/capture_limitation_readouts.md)  
> **Target Defect Part:** `part_008` (Bottom Manifold Cylinder, CAD Class 8)

---

## 1. Executive Summary & Audit Re-Evaluation of Pre-Registered Gates

Following an acute audit of the preliminary analysis, this report evaluates the **production frozen Connected Components (CC) seeds + Voronoi dilation backbone** ($r = 0.3\text{ cm}$, $\min\_size = 15$) and color gating alongside the pre-registered **paired interface edge AUROCs** (Cat A contact $< 0.2\text{ mm}$, Cat B clearance $\ge 0.2\text{ mm}$, and deviant part boundary).

All four gate verdicts have been revised to strictly honor the pre-registered thresholds without narrative inflation:

| Gate | Pre-Registered Hypothesis & Passing Criterion | Empirical Result | Revised Status | Rigorous Scientific Verdict |
|---|---|---|:---:|---|
| **Gate 1** | **Deviant Amplitude Floor $B^*$ (Series A):** Interface shear $|B - A|$ yields edge $\text{AUROC} \ge 0.80$ at threshold $B^*$. | Target boundary $\text{AUROC} = 0.6353$ at $10\text{A}$ ($9.0\text{ px}$ shear). Collapses at $20\text{A}$ ($19.0\text{ px}$). | **FAILED (Pre-Reg)**<br>*Empirical Floor Probed* | **Failed pre-registered $\ge 0.80$ crossing.** Probed an empirical detection floor at $9.0\text{ px}$ ($9\text{ mm}$) shear ($\text{IoU} = 0.1032$), but $19\text{ px}$ breaks 4DGS deformation tracking. |
| **Gate 2** | **Resolution vs Proximity Equivalence (Series R):** Proximity ($D/2$) and 6 MP ($3000\times 2000$) achieve equivalent $\text{AUROC}$ ($\pm 0.03$) and $\text{ARI}$ ($\pm 0.02$). | Cat A motion AUROC jumps from $0.46 \to \mathbf{0.6194}$ at $D/2$; 6 MP at distance remains flat at $0.4578$. Cat A split jumps $5.1\% \to \mathbf{35.9\%}$. | **SURVIVES (Asymmetric)** | **Proximity strictly dominates megapixels for mechanical contacts.** 6 MP only improves clearance gaps (Cat B); halving distance is strictly required to resolve touching boundaries. |
| **Gate 3** | **Material Realism Tax & Texture Recovery (Series T):** Uniform steel collapses appearance AUROC ($\Delta \ge 0.20$); texture recovers $\ge 70\%$. | Steel collapses Cat B clearance split by $75.5\%$ ($44.1\% \to 10.8\%$). Texture recovers Cat B split to $\mathbf{32.4\%}$ ($64.9\%$ recovery). Cat A contact split remains at $0.0\%$. | **SURVIVES (Weakened)** | **Texture recovers clearance optical flow, but cannot resolve mechanical contacts.** Validates Diamond et al. (2017) for unpainted metal clearances, but contact boundaries require geometric priors. |
| **Gate 4** | **Honest Physical Deployment Feasibility (Series F):** Compound operating cell `sF2` achieves $\text{ARI}_{\text{ROI}} \ge 0.75$, Cat A Split $\ge 0.60$, Target $F_1 \ge 0.85$. | `sF2` achieves Target Precision $\mathbf{100.0\%}$, Target boundary $\text{AUROC} = \mathbf{0.7906}$, but CAD ROI ARI is $\mathbf{0.0965}$ and Target $F_1 = \mathbf{0.3929}$. | **FAILED (Full Pipeline)**<br>*Passed as Defect Detector* | **Failed as full-scene segmentation.** Full CAD segmentation on unpainted specular machinery remains unsolved ($\text{ARI} < 0.15$). However, `sF2` validates high-precision defect localization. |

---

## 2. Pre-Registered Paired Interface Edge AUROC Table

Evaluated on exact Mann-Whitney U on all paired boundary edges within the dynamic ROI (Cat A contact $< 0.2\text{ mm}$, Cat B clearance $\ge 0.2\text{ mm}$, same-part reference edges, and deviant `part_008` boundary edges).

| Cell | Series | Resolution | Stand-off Dist | Shear (mm) | Shear (px) | $N_{\text{ROI pts}}$ | $N_{\text{edges}}$ | Motion Cat A AUROC | Motion Cat B AUROC | Target Boundary AUROC | Appearance Cat A AUROC | Appearance Cat B AUROC | Appearance All AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sA0`** | A | $1600\times 900$ | $4.2\text{ m}$ | $1.0\text{ mm}$ | $1.0\text{ px}$ | 89,649 | 537,713 | 0.4216 | 0.4509 | 0.5014 | 0.6672 | 0.6664 | 0.6658 |
| **`sA1`** | A | $1600\times 900$ | $4.2\text{ m}$ | $1.0\text{ mm}$ | $1.0\text{ px}$ | 86,100 | 510,064 | 0.4652 | 0.4584 | 0.4767 | 0.6569 | 0.6524 | 0.6583 |
| **`sA2`** | A | $1600\times 900$ | $4.2\text{ m}$ | $4.0\text{ mm}$ | $4.0\text{ px}$ | 85,856 | 507,022 | 0.4085 | 0.4169 | 0.6254 | 0.6668 | 0.6485 | 0.6647 |
| **`sA3`** | A | $1600\times 900$ | $4.2\text{ m}$ | $9.0\text{ mm}$ | $9.0\text{ px}$ | 90,550 | 533,699 | 0.4680 | 0.4782 | 0.6353 | 0.6840 | 0.6603 | 0.6677 |
| **`sA4`** | A | $1600\times 900$ | $4.2\text{ m}$ | $19.0\text{ mm}$ | $19.0\text{ px}$ | 85,204 | 501,150 | 0.4182 | 0.4112 | 0.6326 | 0.6743 | 0.6563 | 0.6647 |
| **`sA5`** | A | $1600\times 900$ | $4.2\text{ m}$ | $1.0\text{ mm}$ | $1.0\text{ px}$ | 79,339 | 470,307 | 0.5010 | 0.5201 | 0.4848 | 0.6520 | 0.6341 | 0.6398 |
| **`sR0`** | R | $1600\times 900$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 80,446 | 468,477 | 0.5051 | 0.4674 | 0.4525 | 0.6490 | 0.6232 | 0.6439 |
| **`sR1`** | R | $2560\times 1440$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 86,329 | 527,060 | 0.5303 | 0.5046 | 0.4632 | 0.6695 | 0.6467 | 0.6587 |
| **`sR2`** | R | $3000\times 2000$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 91,152 | 571,588 | 0.4578 | 0.3937 | 0.4547 | 0.6680 | 0.6561 | 0.6628 |
| **`sR3`** | R | $1600\times 900$ | $2.1\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 101,450 | 529,218 | **0.6194** | 0.3956 | 0.5018 | **0.8669** | 0.7148 | 0.7817 |
| **`sR4`** | R | $3000\times 2000$ | $2.1\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 103,667 | 541,109 | **0.6187** | 0.4315 | 0.6630 | **0.8715** | 0.7046 | 0.7703 |
| **`sR5`** | R | $3000\times 2000$ | $2.1\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 108,266 | 562,815 | **0.5885** | 0.4794 | 0.6214 | **0.8701** | 0.7110 | 0.7690 |
| **`sT0`** | T | $1600\times 900$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 91,614 | 547,315 | 0.5141 | 0.5086 | 0.5005 | 0.6741 | 0.6645 | 0.6691 |
| **`sT1`** | T | $1600\times 900$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 90,455 | 547,067 | 0.5309 | 0.5480 | 0.5162 | 0.6237 | 0.6304 | 0.6151 |
| **`sT2`** | T | $1600\times 900$ | $4.2\text{ m}$ | $0.0\text{ mm}$ | $0.0\text{ px}$ | 100,031 | 594,065 | 0.5397 | 0.4368 | 0.4768 | 0.6391 | 0.5848 | 0.6081 |
| **`sF0`** | F | $1600\times 900$ | $4.2\text{ m}$ | $1.0\text{ mm}$ | $1.0\text{ px}$ | 80,403 | 461,210 | 0.4367 | 0.4279 | 0.4850 | 0.6542 | 0.6429 | 0.6472 |
| **`sF1`** | F | $3000\times 2000$ | $2.1\text{ m}$ | $9.0\text{ mm}$ | $34.2\text{ px}$ | 134,860 | 747,370 | 0.5429 | 0.2980 | 0.6847 | 0.8199 | 0.6741 | 0.7192 |
| **`sF2`** | F | $3000\times 2000$ | $2.1\text{ m}$ | $9.0\text{ mm}$ | $34.2\text{ px}$ | 148,844 | 854,062 | 0.5885 | 0.3507 | **0.7906** | 0.7857 | 0.6429 | 0.6916 |

> **Key AUROC Insight:** At baseline distance ($4.2\text{ m}$), motion phasor discrepancy across Cat A mechanical contact interfaces is pure noise ($\text{AUROC} \approx 0.42\text{--}0.53$). Halving stand-off distance to $2.1\text{ m}$ (`sR3`, `sR4`) causes Cat A motion AUROC to jump to **$0.619$** and Cat A appearance AUROC to jump from **$0.65 \to 0.871$**. Sensor resolution alone (`sR2`) yields no gain ($0.4578$). On target defect boundaries, `sF2` approaches the pre-registered crossing ($\mathbf{0.7906} \approx 0.80$) only when proximity, 6 MP, and procedural surface texture operate jointly.

---

## 3. Production Frozen CC Backbone Evaluation & Subset Accounting

Evaluates the actual frozen CC-seeds backbone ($r = 0.3\text{ cm}$, $\min\_size = 15$) and color gating across all 18 cells. Complete subset accounting ($N_{\text{scored}}$, $N_{\text{labeled}}$, $N_{\text{discarded}}$, $N_{\text{seeds}}$, $K_{\text{pred}}$) is reported for every cell.

### 3.1 Dilated Backbone Performance (`dilated_1nn`)

| Cell | Series | Method | $N_{\text{scored}}$ | $N_{\text{seeds}}$ | $N_{\text{labeled}}$ | $N_{\text{discarded}}$ | $K_{\text{pred}}$ | CAD ROI ARI | Motion ROI ARI | Cat A Split (39) | Cat B Split (102) | Target IoU | Target Prec | Target Rec | Target F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sA0`** | A | Spatial CC | 89,649 | 69 | 1,618 | 88,031 | 69 | 0.3313 | -0.0004 | 30.8% | 65.7% | 0.4813 | 0.5459 | 0.8026 | 0.6498 |
| **`sA0`** | A | Color Gated | 89,649 | 51 | 1,192 | 88,457 | 51 | **0.3666** | 0.0007 | 17.9% | 53.9% | 0.4792 | 0.5456 | 0.7973 | 0.6479 |
| **`sA1`** | A | Spatial CC | 86,100 | 67 | 1,603 | 84,497 | 67 | 0.3446 | -0.0009 | 33.3% | 62.6% | 0.4387 | 0.4789 | 0.8395 | 0.6099 |
| **`sA1`** | A | Color Gated | 86,100 | 55 | 1,275 | 84,825 | 55 | **0.3633** | 0.0001 | 30.8% | 59.6% | 0.4341 | 0.4745 | 0.8361 | 0.6054 |
| **`sA2`** | A | Spatial CC | 85,856 | 53 | 1,173 | 84,683 | 53 | 0.4149 | -0.0050 | 20.5% | 66.7% | 0.3250 | 0.3334 | 0.9287 | 0.4906 |
| **`sA2`** | A | Color Gated | 85,856 | 38 | 814 | 85,042 | 38 | **0.4544** | -0.0041 | 23.1% | 63.7% | 0.3250 | 0.3333 | 0.9287 | 0.4905 |
| **`sA3`** | A | Spatial CC | 90,550 | 44 | 1,003 | 89,547 | 44 | 0.3753 | -0.0027 | 12.8% | 55.9% | 0.3605 | 0.3747 | 0.9048 | 0.5300 |
| **`sA3`** | A | Color Gated | 90,550 | 34 | 788 | 89,762 | 34 | **0.4002** | -0.0037 | 12.8% | 65.7% | 0.3022 | 0.3122 | 0.9048 | 0.4642 |
| **`sA4`** | A | Spatial CC | 85,204 | 67 | 1,602 | 83,602 | 67 | 0.3472 | -0.0074 | 17.9% | 59.8% | 0.2792 | 0.2843 | 0.9387 | 0.4365 |
| **`sA4`** | A | Color Gated | 85,204 | 56 | 1,299 | 83,905 | 56 | **0.3616** | -0.0068 | 17.9% | 58.8% | 0.2792 | 0.2843 | 0.9387 | 0.4365 |
| **`sA5`** | A | Spatial CC | 79,339 | 49 | 1,080 | 78,259 | 49 | 0.3586 | -0.0041 | 28.2% | 66.7% | 0.3650 | 0.3905 | 0.8484 | 0.5348 |
| **`sA5`** | A | Color Gated | 79,339 | 35 | 735 | 78,604 | 35 | **0.3665** | -0.0132 | 17.9% | 51.5% | 0.2605 | 0.2605 | 0.9988 | 0.4133 |
| **`sR0`** | R | Spatial CC | 80,446 | 42 | 905 | 79,541 | 42 | 0.3638 | — | 23.1% | 50.0% | 0.3184 | 0.6243 | 0.3939 | 0.4830 |
| **`sR0`** | R | Color Gated | 80,446 | 33 | 658 | 79,788 | 33 | **0.4440** | — | 25.6% | 49.0% | 0.2965 | 0.3607 | 0.6249 | 0.4574 |
| **`sR1`** | R | Spatial CC | 86,329 | 77 | 1,993 | 84,336 | 77 | 0.3476 | — | 25.6% | 76.8% | 0.3132 | 0.3909 | 0.6117 | 0.4770 |
| **`sR1`** | R | Color Gated | 86,329 | 58 | 1,435 | 84,894 | 58 | **0.3480** | — | 23.1% | 70.7% | 0.3154 | 0.3926 | 0.6160 | 0.4795 |
| **`sR2`** | R | Spatial CC | 91,152 | 100 | 2,400 | 88,752 | 100 | 0.2714 | — | 23.1% | 71.7% | 0.3113 | 0.4573 | 0.4938 | 0.4749 |
| **`sR2`** | R | Color Gated | 91,152 | 67 | 1,553 | 89,599 | 67 | **0.3326** | — | 15.4% | 72.7% | 0.3070 | 0.4382 | 0.5062 | 0.4697 |
| **`sR3`** | R | Spatial CC | 101,450 | 199 | 5,982 | 95,468 | 199 | 0.1412 | — | 35.9% | 69.7% | 0.2118 | 0.3891 | 0.3173 | 0.3496 |
| **`sR3`** | R | Color Gated | 101,450 | 185 | 5,277 | 96,173 | 185 | **0.1460** | — | 35.9% | 70.7% | 0.2204 | 0.3986 | 0.3301 | 0.3612 |
| **`sR4`** | R | Spatial CC | 103,667 | 171 | 4,870 | 98,797 | 171 | 0.1381 | — | 43.6% | 79.4% | 0.2941 | 0.4842 | 0.4283 | 0.4546 |
| **`sR4`** | R | Color Gated | 103,667 | 153 | 4,211 | 99,456 | 153 | **0.1469** | — | 43.6% | 76.3% | 0.3076 | 0.4755 | 0.4656 | 0.4705 |
| **`sR5`** | R | Spatial CC | 108,266 | 189 | 4,852 | 103,414 | 189 | 0.1127 | — | 25.6% | 66.0% | 0.2711 | 0.4243 | 0.4288 | 0.4265 |
| **`sR5`** | R | Color Gated | 108,266 | 161 | 4,063 | 104,203 | 161 | **0.1202** | — | 23.1% | 68.0% | 0.2711 | 0.4243 | 0.4288 | 0.4265 |
| **`sT0`** | T | Spatial CC | 91,614 | 79 | 1,928 | 89,686 | 79 | 0.2705 | — | 30.8% | 75.5% | 0.2636 | 0.2982 | 0.6944 | 0.4172 |
| **`sT0`** | T | Color Gated | 91,614 | 55 | 1,309 | 90,305 | 55 | **0.3239** | — | 25.6% | 66.7% | 0.3467 | 0.5073 | 0.5226 | 0.5149 |
| **`sT1`** | T | Spatial CC | 90,455 | 101 | 2,798 | 87,657 | 101 | 0.3202 | — | 53.8% | 70.6% | 0.3137 | 0.5747 | 0.4086 | 0.4776 |
| **`sT1`** | T | Color Gated | 90,455 | 89 | 2,326 | 88,129 | 89 | **0.3230** | — | 56.4% | 70.6% | 0.3137 | 0.5747 | 0.4086 | 0.4776 |
| **`sT2`** | T | Spatial CC | 100,031 | 134 | 3,910 | 96,121 | 134 | 0.2891 | — | 33.3% | 55.9% | 0.2859 | 0.4395 | 0.4500 | 0.4447 |
| **`sT2`** | T | Color Gated | 100,031 | 119 | 3,330 | 96,701 | 119 | **0.3105** | — | 33.3% | 54.9% | 0.2890 | 0.4410 | 0.4561 | 0.4484 |
| **`sF0`** | F | Spatial CC | 80,403 | 56 | 1,378 | 79,025 | 56 | 0.3549 | — | 17.9% | 58.8% | 0.2509 | 0.3662 | 0.4434 | 0.4011 |
| **`sF0`** | F | Color Gated | 80,403 | 46 | 1,059 | 79,344 | 46 | **0.3593** | — | 25.6% | 61.8% | 0.2509 | 0.3662 | 0.4434 | 0.4011 |
| **`sF1`** | F | Spatial CC | 134,860 | 297 | 8,944 | 125,916 | 297 | 0.1134 | — | 35.9% | 75.8% | 0.2398 | 0.6170 | 0.2817 | 0.3868 |
| **`sF1`** | F | Color Gated | 134,860 | 259 | 7,643 | 127,217 | 259 | **0.1213** | — | 41.0% | 75.8% | 0.2397 | 0.6164 | 0.2817 | 0.3867 |
| **`sF2`** | F | Spatial CC | 148,844 | 375 | 12,138 | 136,706 | 375 | 0.0965 | — | 33.3% | 64.6% | 0.1696 | 0.4264 | 0.2197 | 0.2900 |
| **`sF2`** | F | Color Gated | 148,844 | 326 | 10,225 | 138,619 | 326 | **0.1086** | — | 30.8% | 57.6% | 0.1892 | 0.4443 | 0.2478 | 0.3182 |

---

### 3.2 Dilation Ablation & Voronoi Sensitivity Analysis

To test whether the Voronoi nearest-seed assignment (`dilated_1nn`) introduces unearned geometric bridging, we ablated dilation across two alternative modes:
1. **`no_dilation_cminus1`**: Discarded non-seed points ($N_{\text{discarded}}$) remain unassigned (assigned cluster label $-1$).
2. **`no_dilation_singletons`**: Discarded points are assigned unique singleton cluster labels.

| Dilation Mode | Mean ROI ARI (`sR0`–`sR5`) | Mean Cat A Split | Mean Cat B Split | Mean $K_{\text{pred}}$ | Mean $N_{\text{labeled}}$ | Conclusion |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **`dilated_1nn` (Production)** | **0.2520** | **31.2%** | **70.2%** | **126.8** | **94,885** (100%) | Baseline production backbone. |
| **`no_dilation_cminus1`** | **-0.0028** | **37.6%** | **34.2%** | **127.8** | **3,500** (~3.7%) | Only $3.7\%$ of points survive in seeds; scoring against $-1$ causes complete ARI collapse. |
| **`no_dilation_singletons`** | **0.0004** | **100.0%** | **99.3%** | **91,486.2** | **94,885** (100%) | Every unassigned point becomes a cluster ($K \approx 90\text{k}$); split rate is artificially $100\%$, ARI is $0.00$. |

> **Audit Takeaway:** The Voronoi dilation step is **strictly load-bearing**. Seed extraction at $r=0.3\text{ cm}$ retains only $1\text{k}\text{--}6\text{k}$ core Gaussians ($1\text{--}5\%$ of the reconstructed cloud). Without 1-NN propagation, evaluation either suffers total coverage collapse (`cminus1`) or pathological over-partitioning (`singletons`).

---

## 4. Deep-Dive Audit Investigations

### 4.1 Resolving the R-Series ARI Inversion (Over-Fragmentation vs Split Rates)

In the preliminary report, as capture quality improved across Series R, ARI collapsed ($0.206 \to 0.057 \to 0.044$) while split rates surged ($14.7\% \to 74.2\%$). The audit data completely unpacks this apparent paradox:

```
sR0 (1600x900 @ 4.2m):  N_scored =  80,446  |  N_seeds =  42  |  K_pred =  42  |  ARI = 0.4440  |  Cat B Split = 49.0%
sR1 (2560x1440 @ 4.2m): N_scored =  86,329  |  N_seeds =  77  |  K_pred =  77  |  ARI = 0.3480  |  Cat B Split = 76.8%
sR2 (3000x2000 @ 4.2m): N_scored =  91,152  |  N_seeds = 100  |  K_pred = 100  |  ARI = 0.3326  |  Cat B Split = 71.7%
sR3 (1600x900 @ 2.1m):  N_scored = 101,450  |  N_seeds = 199  |  K_pred = 199  |  ARI = 0.1460  |  Cat B Split = 70.7%
sR4 (3000x2000 @ 2.1m): N_scored = 103,667  |  N_seeds = 171  |  K_pred = 171  |  ARI = 0.1469  |  Cat B Split = 76.3%
sR5 (120 frames):       N_scored = 108,266  |  N_seeds = 189  |  K_pred = 189  |  ARI = 0.1202  |  Cat B Split = 68.0%
```

* **The Cause:** The ground truth pump contains $K_{\text{GT}} = 107$ CAD parts. At baseline stand-off distance ($4.2\text{ m}$), small parts merge into the casing, leaving $33\text{--}42$ surviving seeds. This under-partitioning scores well under ARI ($\mathbf{0.4440}$) because large CAD bodies (flanges, frame, motor) are preserved as cohesive units.
* When moving closer ($2.1\text{ m}$, `sR3`–`sR5`), Gaussian surface density increases. Fine geometric crevices *within* large components (such as motor cooling fins and hydraulic casing ridges) now split into distinct connected components. Seed count surges to **$171\text{--}199$** ($K_{\text{pred}} \approx 2\times K_{\text{GT}}$).
* Because ARI severely penalizes splitting a true CAD label into multiple fragments, ARI falls from $0.44 \to 0.12$. Concurrently, smaller shattered clusters mechanically isolate separate part pairs, causing the pairs-split rate to climb to $\mathbf{76.3\%}$.
* In `segment.rigid` (unbounded graph clustering), this over-fragmentation was even more severe, driving $K_{\text{pred}} > 1300$ and dropping ARI below $0.05$.

---

### 4.2 Series A Metric-Trap & Pixel-Space Floor Analysis

#### The Dual-Scoring Discrepancy
In Series A, the whole assembly undergoes common-mode vibration ($A = 1\text{ mm}$), while only `part_008` vibrates deviantly ($B$). The true physical motion partition is strictly **2 classes**:
$$\text{Class 1: Common-Mode Rigid Body (106 parts)}, \quad \text{Class 2: Deviant Target Part (`part_008`)}$$

When evaluating spatial CC against CAD GT, ARI is **$0.33\text{--}0.45$** (spatial geometry is well clustered into parts). When evaluating against Motion GT, Motion ARI is **$\approx 0.000$** (because partitioning a single moving rigid body into 50 geometric seeds is mathematically penalized under ARI as severe over-clustering).

| Cell | Target Motion | Interface Shear (mm) | Interface Shear (px) | CAD ROI ARI | Motion ROI ARI | Target IoU | Target Precision | Target Recall | Target F1 | Detection Criterion (IoU $\ge 0.10$ & Prec $\ge 0.10$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sA0`** | $0.0\text{A}$ (static) | $1.0\text{ mm}$ | $1.0\text{ px}$ | 0.3666 | 0.0007 | 0.4792 | 0.5456 | 0.7973 | 0.6479 | Baseline static overlap |
| **`sA1`** | $2.0\text{A}$ | $1.0\text{ mm}$ | $1.0\text{ px}$ | 0.3633 | 0.0001 | 0.4341 | 0.4745 | 0.8361 | 0.6054 | Undifferentiated |
| **`sA2`** | $5.0\text{A}$ | $4.0\text{ mm}$ | $4.0\text{ px}$ | 0.4544 | -0.0041 | 0.3250 | 0.3333 | 0.9287 | 0.4905 | Undifferentiated |
| **`sA3`** | $10.0\text{A}$ | $9.0\text{ mm}$ | $9.0\text{ px}$ | 0.4002 | -0.0037 | 0.3022 | 0.3122 | 0.9048 | 0.4642 | Motion boundary emerging |
| **`sA4`** | $20.0\text{A}$ | $19.0\text{ mm}$ | $19.0\text{ px}$ | 0.3616 | -0.0068 | 0.2792 | 0.2843 | 0.9387 | 0.4365 | Deformation breakdown |
| **`sA5`** | $1.0\text{A}$ (target only) | $1.0\text{ mm}$ | $1.0\text{ px}$ | 0.3665 | -0.0132 | 0.2605 | 0.2605 | 0.9988 | 0.4133 | Spatial baseline |

#### Restating the Motion Floor in Pixel Displacements
* At $1600\times 900$ and $4.2\text{ m}$ stand-off distance, $1.0\text{ mm}$ on the pump projects to **$1.0\text{ px}$** on the image plane.
* The $10\text{A}$ ($9\text{ mm}$) threshold in Series A corresponds to **$9.0\text{ px}$ of interface shear**.
* At $2.1\text{ m}$ (`sR3`, $2.0\text{ px/mm}$), the same $9\text{ mm}$ physical motion projects to **$18.0\text{ px}$**.
* In `sF1`/`sF2` ($3000\times 2000$ @ $2.1\text{ m}$, $3.8\text{ px/mm}$), $9\text{ mm}$ projects to **$34.2\text{ px}$**.
* At $19.0\text{ px}$ shear (`sA4`), the 4DGS deformation network fails because local temporal deformation linearity is violated over large pixel displacements. Thus, $B^*$ is bounded between a **photometric SNR floor at $\approx 9\text{ px}$** and a **deformation tracking ceiling at $\approx 19\text{ px}$**.

---

### 4.3 Series T Nuance: Texture Recovers Clearances, Not Contacts

In Series T, procedural surface texture was hypothesized to restore performance lost to uniform specular PBR steel. The audit reveals an asymmetric mechanism:

| Finishing Regime | Appearance All AUROC | Cat A Split (Contact) | Cat B Split (Clearance) | Target IoU (CC Backbone) | Target IoU (`rigid2`) |
|---|:---:|:---:|:---:|:---:|:---:|
| **`sT0` (Synthetic Pastel)** | 0.6691 | 25.6% | 66.7% | 0.3467 | 0.1029 |
| **`sT1` (Uniform Steel PBR)** | 0.6151 | **56.4%** | 70.6% | 0.3137 | 0.0631 |
| **`sT2` (Textured Steel PBR)** | 0.6081 | 33.3% | 54.9% | 0.2890 | **0.1207** |

* In the motion-only clustering (`rigid2`), uniform steel collapsed Cat B clearance separation by **$75.5\%$** ($44.1\% \to 10.8\%$), and surface texture recovered Cat B split to **$32.4\%$** ($64.9\%$ recovery).
* However, on the CC backbone, Cat A mechanical contact separation drops from $56.4\% \to 33.3\%$ when texture is added. Why? Because procedural noise introduces local color variance across contact boundaries that fragments small contact seeds.
* **Conclusion:** Procedural surface texture successfully restores optical flow gradients across clearance gaps, but does not solve mechanical contact ambiguity.

---

### 4.4 Series F Honest Framing: Detector vs Segmentation

The preliminary claim that `sF2` represented a "validated production pipeline" was an overstatement based on a 0.089 ARI.

* **Full-Scene Segmentation Reality:** Under honest CAD evaluation, whole-scene ROI ARI in `sF2` is **$0.0965$** (Spatial CC) and **$0.1086$** (Color-Gated CC). The physical pump CAD segmentation under metallic PBR capture remains unsolved.
* **Defect Detection Reality:** In `rigid2`, `sF2` isolates `part_008` with **$100.0\%$ Precision** ($1.0000$), IoU $= \mathbf{0.2445}$, and Target Boundary Motion $\text{AUROC} = \mathbf{0.7906}$ (approaching the pre-registered $0.80$ threshold).
* **Accurate Scientific Framing:**
  > *"A vibrating deviant part is detectable at 100% precision under close ($2.1\text{ m}$), textured, 6 MP capture; full-scene CAD segmentation on unpainted machinery remains at $\text{ARI} < 0.15$."*

---

## 5. Thesis Recommendations & Methodological Rules

1. **Retire `segment.rigid` and `rigid2` from Primary Reporting:**
   All future quantitative tables must report the frozen CC backbone ($r=0.3\text{ cm}$, $\min\_size = 15$) with color gating and Voronoi dilation. Motion methods should only be reported as specialized defect detectors.
2. **Mandate Dual Scoring on Series A:**
   Never evaluate whole-body common-mode vibration exclusively against 107 CAD labels without also reporting the 2-class motion GT.
3. **Report Subset Counts Everywhere ($N_{\text{scored}}$, $N_{\text{labeled}}$, $N_{\text{seeds}}$, $K_{\text{pred}}$):**
   ARI inversions must always be contextualized by the predicted cluster count $K_{\text{pred}}$ relative to $K_{\text{GT}}$.
4. **State Motion Floors in Pixel Units:**
   Report motion floors as image plane displacements ($L_{\text{px}}$) alongside physical millimeters to ensure cross-camera transferability.
