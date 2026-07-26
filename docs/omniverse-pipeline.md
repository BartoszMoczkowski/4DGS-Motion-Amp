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

## Who runs what

Isaac Sim capture cannot run in the assistant sandbox (no GPU) — `omni_capture.py` runs on the author's machine via the native Isaac Sim Python (`Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`; see the orchestrator's `capture.isaac` stage). `omni_to_4dgs.py` is pure Python and unit-tested in the sandbox. Training/rendering run in the CUDA container. A rendering-capable Isaac Docker container was attempted but abandoned — Vulkan (needed by Isaac's RTX renderer) is not supported under WSL2, an NVIDIA-stated hard limitation; see [orchestrator.md](orchestrator.md).
