# 4DGS Motion Segmentation Visualizations & Interactive 3D Viewing Guide

This document compiles the current experimental state of the segmentation methods evaluated on synthetic industrial pump scenes, provides comparative benchmark dashboards, and details multiple ways to interactively view the Gaussians locally.

---

## 1. Quantitative Benchmark & Failure Analysis Dashboard

The benchmark results across the 7 evaluated grid/sweep scenes are summarized below:

![4DGS Segmentation Benchmark Dashboard](C:\Users\barte\.gemini\antigravity-ide\brain\8b6b2ec5-5357-40d9-8177-40b2799b8b44\benchmark_metrics_comparison.png)

### Key Metric Findings:
- **Adjusted Rand Index (ARI)**: All motion-only clustering methods (`rigid`, `rigid2`, `kabsch`, `mbs`) remain near zero ($\text{ARI} \approx -0.04 \text{ to } +0.02$). `Oracle Mask Lift` displays a misleadingly higher global ARI ($\approx 0.28$) solely because it correctly buckets the background class; within the moving machine region itself, its ARI remains near zero ($\text{ARI} \approx 0.00\text{--}0.10$).
- **Cluster Count Degradation ($K$)**: Baseline rigidity over-fragments due to pairwise edge noise ($K=89\text{--}350$), while MBS and Kabsch EM collapse to under-segmented clusters ($K=2\text{--}8$ vs $K_{\text{gt}}=107$).
- **Root-Cause Separability AUROC**: The per-edge rigidity separability AUROC is capped between **0.45 and 0.67** across all trained models (well below the $0.80$ viability threshold), confirming that reconstruction jitter masks physical millimeter-scale motion.

---

## 2. Point Cloud Segmentation Projections (Top, Front, Side)

Below is an orthographic projection comparison of **360,527 Gaussians** from the primary test scene (`grid-A20mm_M2`), showing the original 4DGS appearance alongside the color-coded segmentation outputs of each method:

![4DGS Orthographic Projections](C:\Users\barte\.gemini\antigravity-ide\brain\8b6b2ec5-5357-40d9-8177-40b2799b8b44\segmentation_projections_comparison.png)

---

## 3. Interactive Local 3D Gaussian & Segmentation Viewers

You have several ways to interact with the Gaussians and their segmentations locally:

### Option A: Standalone Interactive WebGL Viewer (Instant Browser Viewing)
A dedicated, standalone Three.js WebGL viewer has been generated and placed in the project root:
- **File**: [interactive_gaussian_viewer.html](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/interactive_gaussian_viewer.html)

**Features**:
- **Full 3D Navigation**: Left-click to orbit, right-click to pan, scroll wheel to zoom.
- **Dynamic Method Switcher**: Instantly switch between *True 4DGS RGB Color*, *Rigid Baseline*, *Rigid2 (Denoised)*, *Kabsch EM*, *MultiBodySync (MotNet)*, and *Oracle Mask Lifting*.
- **Adjustable Controls**: Slider for Gaussian point size, camera presets (Top, Front, Side, Perspective), and an auto-rotate showcase mode.
- **How to Open**: Double-click [interactive_gaussian_viewer.html](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/interactive_gaussian_viewer.html) or open it with any web browser (Chrome, Edge, Firefox).

---

### Option B: Local Python Open3D Viewer (High Performance Desktop GUI)
An interactive viewer script [view_gaussians_open3d.py](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/view_gaussians_open3d.py) allows you to inspect any generated `.ply` point cloud directly:

```bash
# View Kabsch EM segmentation:
uv run python view_gaussians_open3d.py --ply runs/grid-A20mm_M2/segmentation_colored_kabsch.ply

# View Rigid2 (Denoised) segmentation:
uv run python view_gaussians_open3d.py --ply runs/grid-A20mm_M2/segmentation_colored_rigid2.ply

# View the full canonical 4DGS point cloud:
uv run python view_gaussians_open3d.py --raw runs/grid-A20mm_M2/train_out/point_cloud/iteration_15000/point_cloud.ply
```

**Controls**:
- `Left Click + Drag`: Orbit / Rotate
- `Shift + Left Click` / `Right Click`: Pan
- `Scroll`: Zoom
- `+` / `-`: Increase / Decrease splat size
- `Q` / `ESC`: Exit

---

### Option C: SIBR Remote Real-Time Viewer (4DGS Upstream)
The repository includes SIBR viewer instructions for real-time interactive rendering over port 6017 (as described in [viewer_usage.md](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/docs/viewer_usage.md)):
```bash
./viewers/bin/SIBR_remoteGaussian_app.exe --port 6017
```

---

### Option D: Desktop Point Cloud Tools (CloudCompare / MeshLab)
You can directly open any of the output PLY files in standard desktop inspection tools like [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/):
- `runs/grid-A20mm_M2/segmentation_colored_kabsch.ply`
- `runs/grid-A20mm_M2/segmentation_colored_rigid2.ply`
- `runs/grid-A20mm_M2/segmentation_colored_mbs.ply`
- `runs/grid-A20mm_M2/segmentation_colored_mask_lift_oracle.ply`
- `runs/grid-A20mm_M2/train_out/point_cloud/iteration_15000/point_cloud.ply`

---

### Option E: Web Splat Viewers (SuperSplat / PlayCanvas)
For full ellipsoid / covariance splat rendering (rather than point approximations), you can drag any `.ply` or exported per-frame model into:
1. [SuperSplat Web Viewer](https://playcanvas.com/supersplat/editor)
2. Export time-resolved 3DGS frames using [export_perframe_3DGS.py](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/core/export_perframe_3DGS.py):
   ```bash
   uv run --package 4dgs-core python core/export_perframe_3DGS.py -m runs/grid-A20mm_M2/train_out --configs core/arguments/multipleview/default.py
   ```
