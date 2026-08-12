# T22 — Multi-view mask lifting (proposal 02) — implementation notes

2026-08-12

## What was done

All code for T22 landed today:

### 1. CUDA mask lifting script
`orchestrator/pipeline/vendored/cuda/mask_lift.py` — standalone argparse CLI that runs inside the `cuda` container (T09 shape).  Key algorithm:
- Loads trained 4DGS model + multipleview cameras via the existing `Scene`/`GaussianModel` loaders.
- Deforms Gaussians to `ref_time` using `get_state_at_time`.
- For each camera: renders depth via `gaussian_renderer.render()` (depth channel already supported by the rasterizer), projects each Gaussian mean to screen space using the camera's `full_proj_transform`, checks mask hit + depth match (`|z_view - D_v| < depth_tol`), accumulates votes.
- Vote score = (sum of mask hits where visible) / (sum of visible views), threshold at `vote_thresh`.
- k-NN graph dilation (`dilation_hops`) fills interior holes (Gaussians occluded in every view but inside the machine).
- Writes `roi_mask.npz` with `roi_mask` (bool[N]) and `snr` (float32[N] vote scores).

### 2. Host oracle stage
`orchestrator/pipeline/stages/roi_mask_oracle.py` — `roi.mask_oracle`, host env.  Reads `gt_segmentation.npz`, NN-aligns GT labels to trajectory canonical positions, produces a perfect ROI mask (`labels >= 0`).  This is the **ceiling benchmark** — it measures the best ARI achievable if the ROI were perfect.

### 3. CUDA stage wrapper
`orchestrator/pipeline/stages/roi_mask_lift.py` — `roi.mask_lift`, cuda env.  Builds CLI args, writes bridge file, calls `mask_lift.py` inside the container via `run_cuda_script`.

### 4. Registry / wiring
- Added `mask_lift` to `VENDORED_CUDA_SCRIPTS` in `cuda_common.py`.
- Added imports for `roi_mask_lift` and `roi_mask_oracle` in `stages/__init__.py`.

### 5. Presets
- `pump01_mask_oracle.yaml` — `roi.impl: mask_oracle`, `segment.impl: rigid2`
- `pump01_mask_lift.yaml` — `roi.impl: mask_lift`, `segment.impl: rigid2`

### 6. Benchmark harness
`scene-gen/run_grid_seg.py` extended with:
- `--impl mask_lift_oracle` → `roi.mask_oracle` + `segment.rigid2` + `seg_eval.default`
- `--impl mask_lift` → `roi.mask_lift` + `segment.rigid2` + `seg_eval.default` (requires `--masks-dir`)
- New `SEGMENT_STAGE` mapping replaces the fragile `impl.replace('_roi', '')` string hack.
- Idempotency for ROI-based impls keys off the ROI stage, not the segment stage.

### 7. Sandbox tests
- `tests/test_roi_mask_oracle.py` — 4 tests: GT label mapping, label-0 handling, NN alignment, artifact contract.
- `tests/test_mask_lift_helpers.py` — 6 tests: k-NN graph, graph dilation, mask path resolution, mask loading threshold/resize/missing.

## What is NOT done

- **Real GPU run** of the oracle ceiling on pump01 grid — needs Bartosz's machine + Docker Desktop + GPU.  The benchmark command is ready:
  ```bash
  .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl mask_lift_oracle
  ```
  This will produce `runs/grid_seg_mask_lift_oracle_results.csv`.

- **Clean-plate / SAM mask production** — `mask_lift` needs per-camera binary masks.  No masks exist yet for pump01.  The `masks_dir` contract supports three layouts (`camNN/frame_00001.png`, `camNN.png`, `mask_camNN.png`).

- **Depth sign convention verification** — the depth comparison in `mask_lift.py` uses `abs(z_view - d_render) < depth_tol`.  The rasterizer's depth output sign convention was not verified on real hardware.  If the sign is inverted, the depth test will fail (all Gaussians marked occluded) and the vote will fall back to frustum-only visibility.  This is safe (degrades gracefully) but should be confirmed.

## Decision rules for interpreting oracle results

Per `IMPLEMENTATION_PLAN.md` §3 and `docs/proposals/02-multiview-mask-lifting.md`:

1. Run oracle ceiling: `run_grid_seg.py --impl mask_lift_oracle`
2. Compare `ari_within_roi` (and global `ari`) against `rigid2_roi` (T19) results in `runs/grid_seg_rigid2_roi_results.csv`.
3. **If oracle ≫ rigid2_roi** (e.g., ARI-within-ROI jumps from ~0.1 to >0.5): the bottleneck is **ROI quality**, not clustering.  Invest in generating clean-plate/SAM masks and run `--impl mask_lift`.
4. **If oracle ≈ rigid2_roi** (both low): the bottleneck is **segmentation itself** (reconstruction quality limits).  T22 mask lifting won't help — proceed to T21 (subspace spectral) or conclude the thesis is reconstruction-quality-limited.

## Critical context from T19

T19's ROI motion gate did NOT improve ARI on real pump01 data.  The motion gate kept ~100% of points because reconstruction jitter after band-passing has energy comparable to true mm-scale motion.  This makes T22 (geometric mask lifting, immune to jitter) the **last remaining bet** before concluding the thesis is reconstruction-quality-limited.
