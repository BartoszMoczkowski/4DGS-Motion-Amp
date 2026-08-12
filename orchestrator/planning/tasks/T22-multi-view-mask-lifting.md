# T22 — Multi-view mask lifting (`roi.mask_lift` + `roi.mask_oracle`)

- Status: **done (2026-08-12)** — code landed, presets created, sandbox tests written; real GPU oracle-ceiling run pending Bartosz's machine
- Phase: 7 (segmentation rescue — `docs/proposals/IMPLEMENTATION_PLAN.md`)
- Depends on: T19 (ROI role plumbing), T09 (CUDA stage shape)
- Environment: cuda (`roi.mask_lift`), host (`roi.mask_oracle`)

## Goal

Implement proposal 02: lift per-view binary masks into a 3D Gaussian ROI mask via calibrated
camera projection + depth-rendered occlusion testing.  Two orchestrator impls:

- `roi.mask_lift` (cuda) — the real method; renders depth per view, projects Gaussians, checks
  mask hit + depth match, votes across views, dilates along k-NN graph.
- `roi.mask_oracle` (host) — validation ceiling; uses GT part labels directly as the ROI mask
  (nearest-neighbour aligned to trajectory points), measuring the best ARI achievable with
  perfect region-of-interest knowledge.

## In scope

- `pipeline/vendored/cuda/mask_lift.py` — standalone argparse CLI (T09 cuda-stage shape):
  loads trained 4DGS model + multipleview cameras, deforms to `ref_time`, renders depth via
  `gaussian_renderer.render()`, projects Gaussian means to screen, checks `|z_view - D_v| < τ_z`
  occlusion test, accumulates foreground votes, thresholds at `vote_thresh`, k-NN graph dilation
  to fill interior holes.  Mask directory layout auto-detects `camNN/frame_00001.png`,
  `camNN.png`, or `mask_camNN.png`.
- `pipeline/stages/roi_mask_lift.py` — `roi.mask_lift`, cuda env, inputs `model`, outputs
  `roi_mask`.  Adds `"mask_lift"` to `VENDORED_CUDA_SCRIPTS` in `cuda_common.py`.
- `pipeline/stages/roi_mask_oracle.py` — `roi.mask_oracle`, host env, inputs `trajectories` +
  `gt_segmentation`, outputs `roi_mask`.  NN-aligns GT labels to trajectory canonical positions.
- Config: `RoiMaskLiftConfig` already existed in `models.py` from the T19 planning pass;
  `RoiConfig.impl` Literal already included `"mask_lift"`.
- Presets: `pump01_mask_oracle.yaml`, `pump01_mask_lift.yaml`.
- `scene-gen/run_grid_seg.py` extended with `--impl mask_lift_oracle` and `--impl mask_lift`
  (+ `--masks-dir` CLI flag).  Idempotency keys off the ROI stage for ROI-based impls.
- `SEGMENT_STAGE` mapping introduced to cleanly resolve impl → segment stage name (replaces
  the fragile `impl.replace('_roi', '')` string hack that would break on `mask_lift_oracle`).

## Out of scope

- Clean-plate mask production (host-side differencing / SAM) — the `masks_dir` contract is
  defined, but actual mask generation is a pre-processing step outside the orchestrator.
- Real-hardware mask lift run — needs per-camera binary masks, which don't exist yet for pump01.

## Verification (sandbox, no GPU)

- `orchestrator/tests/test_roi_mask_oracle.py` — 4 tests: oracle matches GT labels (movers in,
  static out), label-0-as-background handling, NN alignment when GT/traj points differ, artifact
  contract (bool roi_mask + float32 snr).
- `orchestrator/tests/test_mask_lift_helpers.py` — 6 tests: k-NN graph shape/symmetry, graph
  dilation expands correctly, mask path resolution priority, mask loading grayscale threshold,
  resize, missing-file handling.

## Acceptance criteria (real hardware)

1. Oracle ceiling on pump01 grid: `run_grid_seg.py --impl mask_lift_oracle` produces
   `runs/grid_seg_mask_lift_oracle_results.csv`.  Compare `ari` and `ari_within_roi` against
   `rigid2_roi` (T19) — if oracle ≫ rigid2_roi, the bottleneck is ROI quality and mask lift
   is worth pursuing; if oracle ≈ rigid2_roi, the bottleneck is segmentation itself.
2. If (1) shows a large gap, generate clean-plate/SAM masks and run `--impl mask_lift`.

## Log

- 2026-08-12: All code landed.  The `mask_lift.py` CUDA script reuses `get_state_at_time` and
  `render()` from the existing 4DGS codebase (depth rendering already supported by the rasterizer).
  MiniCam construction mirrors the Camera class's matrix math exactly.  Depth comparison uses
  view-space z; exact sign convention to be verified on real hardware (if depth_tol needs tuning).
