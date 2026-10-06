# Omniverse → 4DGS synthetic-data pipeline

Compiled from `.claude_notes/NOTES_omniverse_pipeline.md` and `omniverse-pipeline/omniverse_pipeline/README.md` (full detail there).

## Why synthetic data

Real captures give no ground truth. Omniverse/Isaac Sim provides exact camera intrinsics/extrinsics (so COLMAP can be skipped entirely), per-pixel instance segmentation and per-object transforms (the only way to score segmentation quantitatively), and fully controllable motion (subtle, periodic, mm-scale — exactly the motion-amp target, with known true displacement).

## Architecture

```
prepared USD stage (parts + authored motion + semantics)
  → omni_capture.py   (headless Isaac Sim, Replicator BasicWriter; runs on the user's GPU)
      camNN/{rgb, instance_segmentation, camera_params}/…, per-frame object transforms
  → omni_to_4dgs.py   (pure Python, no Isaac dependency)
      data/multipleview/<scene>/: camNN/frame_XXXXX.jpg, sparse_/ (COLMAP bins written
      directly from GT poses), points3D_multipleview.ply, poses_bounds_multipleview.npy,
      gt_segmentation.npz, scene_scale.json
  → core/train.py (4DGS multipleview) → segmentation → core/render_amp.py
```

Key files: `omniverse-pipeline/omniverse_pipeline/{omni_capture.py, omni_to_4dgs.py, rig.py, split_mesh.py, add_motion.py, capture_config*.yaml}`. The camera rig is 8–12 configurable static cameras on a ring/dome looking at the subject's bbox center.

## The pump test asset

`CONJUNTO BOMBAS.usd` was a single fused mesh (not segmentable). Two prep tools fixed that:

- `split_mesh.py` — weld coincident vertices, split by connected components → **107 parts** (`frame_base` + 106 movers), each with `displayColor` and a semantics label → `CONJUNTO_BOMBAS_segmented.usd`. Geometry preserved exactly (208,906 faces).
- `add_motion.py` — per-part rigid SE(3) sinusoids pivoting about part centroids: translation 1–4 mm, rotation capped so surface displacement is 0.5–3 mm, integer cycle counts (2–5) so motion loops. Result: 60 frames @24 fps, peak surface displacement 1.75–6.7 mm → `CONJUNTO_BOMBAS_animated.usd` + `_animated_motion_groups.json` (GT part→segment map).

Both live in `Q:\Omniverse\assets\pump_radnom\`.

## Notable bugs found and fixed

- **Frame discovery found 0 frames** — Replicator nests annotator output in `camNN/rgb/` etc.; converter now prefers that subfolder.
- **NaN loss at the coarse→fine boundary** — root cause: raw Omniverse stage-unit camera translations gave `cameras_extent ≈ 4898`, which multiplies every learning rate (`spatial_lr_scale`); the first fine-stage grid step was ×~7.8 → NaN. Fixed by converting through `meters_per_unit` and rescaling the scene so the nerf++ camera radius lands at `--target-radius 4.0`; `scene_scale.json` records the total scale so internal distances can be reported back in physical mm. Secondary fixes: `SO_REUSEADDR` on the network-GUI socket (the NaN self-restart hit "Address already in use") and disabling `opacity_reset_interval` for pump01.
- **All-zero GT labels** — the point-cloud sampler labelled by the mesh prim name (always `"mesh"`); fixed to use the parent Xform name. Required a fresh capture.
- **Erratic test-camera video** — the LLFF `get_spiral` heuristic is wrong for inward-looking camera rings; added `get_orbit()` (constant-radius circular path) used by the multipleview video path only.
- **Dark patterned dome background** added to capture (stdlib-only PNG generator) for contrast and stable features; YAML fallback parser since Isaac's Python lacks PyYAML.

## 2026-10-05 review fixes (see `reviews/2026-10-05-omniverse-pipeline-review.md`)

- **Double init-color normalization (fixed).** `omni_to_4dgs.py` wrote init point-cloud colors as 0–1 floats, but the loader contract is 0–255 (`core/scene/dataset_readers.py` divides by 255 itself before `RGB2SH`), so all Gaussians initialized ~black. The stray `/255.0` was dropped in both `omni_to_4dgs.py` and the vendored `convert.py`; stored init colors are now 0–255 floats. (Low practical impact — init colors are random pseudo-colors and training recovers — but the intended initialization was silently discarded.)
- **`peak_surface_mm` ignored the 0.2 compose scale (fixed).** `gen_scenes.py` recorded per-part peak surface motion in subject-internal units while `compose_scene` applies `SCALE = 0.2`, so physical rendered-world motion was 5× smaller than recorded (e.g. `pump_A8mm_M2`'s recorded 22.98 mm ≈ 4.6 mm in the world). New runs record rendered-world millimetres and write `compose_scale` into each `*_motion.json`. **Historical `*_motion.json` / `grid_manifest.json` files — and `A20mm`-style cell names — are 5× too large**; scenes were not regenerated.
- **Grid seg trajectories undersampled (fixed; GPU re-run pending).** Grid scenes are 240 frames @ 60 fps with 40 motion cycles per clip, but seg trajectories were extracted at the default `n_times=60` — sampling 40 cycles at 60 points aliases past Nyquist (~20 cycles), so every frequency-calibrated stage operated on the wrong waveform. New preset `orchestrator/pipeline/config/presets/grid_seg.yaml` sets `seg_extract.n_times: 240` and `run_grid_seg.py` uses it; `seg_extract` also warns loudly when `n_times` is below the capture's frame count. The near-zero ARIs in `runs/grid_seg_results.csv` were measured on the aliased trajectories and await re-running.

## Who runs what

Isaac Sim capture cannot run in the assistant sandbox (no GPU) — `omni_capture.py` runs on the author's machine via the native Isaac Sim Python (`Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`; see the orchestrator's `capture.isaac` stage). `omni_to_4dgs.py` is pure Python and unit-tested in the sandbox. Training/rendering run in the CUDA container. A rendering-capable Isaac Docker container was attempted but abandoned — Vulkan (needed by Isaac's RTX renderer) is not supported under WSL2, an NVIDIA-stated hard limitation; see [orchestrator.md](orchestrator.md).
