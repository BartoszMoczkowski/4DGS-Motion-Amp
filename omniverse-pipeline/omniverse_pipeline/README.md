# Omniverse → 4DGS testing pipeline

Generate controllable **synthetic multi-camera datasets** in NVIDIA Omniverse / Isaac Sim,
convert them into the format 4DGS's `multipleview` loader expects, and carry **ground-truth
camera poses + segmentation labels** through for evaluating motion segmentation (MBS etc.).

Stage in the larger project:
`Omniverse capture → 4DGS reconstruction → MBS segmentation → per-segment motion amplification`.
See `../.claude_notes/NOTES_omniverse_pipeline.md` for the full architecture and decisions,
and `../.claude_notes/NOTES_4dgs_motion_segmentation.md` for the MBS↔4DGS analysis.

## Files

| file | runs where | tested |
|---|---|---|
| `omni_capture.py` | **Isaac Sim 5.1 python** (GPU) | API-verified, not executed here |
| `rig.py`          | anywhere (numpy) | ✅ `python rig.py --selftest` |
| `omni_to_4dgs.py` | anywhere (numpy+Pillow) | ✅ `python omni_to_4dgs.py --selftest` |
| `split_mesh.py`   | anywhere (trimesh+usd-core) | ✅ ran on the pump (107 parts) |
| `add_motion.py`   | anywhere (usd-core+numpy) | ✅ ran on the pump (106 movers) |
| `capture_config.yaml` | config | — |

## 0. Scene prep for a fused CAD asset (the pump)

An STL-imported assembly is often a single fused mesh with no parts/motion. Two steps make it
a usable motion-segmentation test case (already done for `CONJUNTO BOMBAS.usd`):

```bash
# split fused mesh -> per-part labelled USD (connected components)
python split_mesh.py --in "Q:/Omniverse/assets/pump_radnom/CONJUNTO BOMBAS.usd" \
    --out "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd" \
    --group CONJUNTO_BOMBAS --preview seg.png            # -> 107 parts (frame_base + 106)

# author subtle periodic per-part motion -> animated USD + GT motion-segment map
python add_motion.py --in  "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd" \
    --out "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd" \
    --num-frames 60 --fps 24 --trans-amp-mm 1 4 --rot-surface-mm 0.5 3 --freq 2 5 --plot qa.png
```

Then point `capture_config.yaml` at the animated USD with
`semantic_roots: ["/World/CONJUNTO_BOMBAS"]`. `..._animated_motion_groups.json` is the
ground-truth part→segment map for scoring segmentation.

## 1. Capture (in Isaac Sim)

`omni_capture.py` opens a USD stage, rings it with a static camera rig (config-driven,
8–12 cams, ring or dome), adds lighting (a headless render is black otherwise), and captures
synchronized RGB + instance/semantic segmentation + GT camera params over the stage's
animation timeline. It also samples the meshes into an init point cloud whose points carry
per-instance labels (the GT for segmentation eval).

Two ready configs:
- `capture_config.yaml` — the animated factory `SceneAssembly.usd` (default).
- `capture_config_pump.yaml` — **the animated pump** (`CONJUNTO_BOMBAS_animated.usd`): 10-cam
  dome sized to the ~17 m pump, 60 frames matching the 0–59 timeline, `semantic_roots`
  labelling all 107 parts, dome+distant lighting. This is the ready end-to-end pump test.

```bat
:: Windows, using Isaac Sim's bundled python:
Q:\Omniverse\ISAAC_SIM\IsaacSim\tools\packman\python.bat ^
    omni_capture.py --config capture_config_pump.yaml
```

Lighting is controlled by the config `lighting:` block (dome + distant key light; set
`force: true` to add them even when the scene already has lights). Camera count/frames can be
overridden on the CLI (`--n-cameras`, `--frames`).

Output (`output.capture_dir`):

```
capture_scene01/
  cameras_gt.json                 # intrinsics + OpenCV cam->world per camera (authoritative)
  cam01/ .. camNN/
      rgb_XXXX.png
      instance_segmentation_XXXX.png (+ *_mapping.json)
      semantic_segmentation_XXXX.png (+ *_mapping.json)
      camera_params_XXXX.json     # Replicator's own pose export (cross-check)
  points3D_gt.ply                 # mesh-sampled init cloud (colored)
  points3D_labels.npy             # per-point instance label (GT segmentation)
  label_names.json
```

> **First-run notes.** Isaac Sim 5.1 is newer than the assistant's knowledge, so the
> Replicator/USD calls are written from the 5.1 docs but were not executed here. Watch for:
> the semantics helper import (three fallbacks are tried in `_apply_semantics`), the
> `orchestrator.step` signature, and whether `BasicWriter` writes each render product into a
> `camNN/` subfolder (it should, given the `name=` we pass). Run once headless with
> `--frames 2 --n-cameras 2` as a smoke test before a full capture.

## 2. Convert to 4DGS `multipleview`

`omni_to_4dgs.py` is pure Python (no Isaac dependency). It writes exact COLMAP **binary**
`sparse_` from the GT poses (no COLMAP feature-matching needed), renames frames to
`camNN/frame_XXXXX.jpg`, and builds `points3D_multipleview.ply`, `poses_bounds_multipleview.npy`,
and `gt_segmentation.npz`.

```bash
python omniverse_pipeline/omni_to_4dgs.py \
    --capture Q:/Omniverse/renders/capture_scene01 \
    --out .           `# 4DGS repo root` \
    --name scene01
# -> data/multipleview/scene01/{camNN, sparse_, *.ply, *.npy, gt_segmentation.npz}
```

Format details (verified against `scene/colmap_loader.py`, `scene/multipleview_dataset.py`,
`scene/dataset_readers.py`):
- image names in `images.bin` are `image{N}.jpg`; the loader maps them to `cam{N:02d}`.
- one shared PINHOLE camera (`cam_intrinsics[1]`); all views share intrinsics.
- COLMAP world-to-camera qvec/tvec; loader recovers `R = qvec2rotmat(q).T`, `T = tvec`.
- `poses_bounds` is LLFF `(N,17)`; used only for the spiral test-view path.

## 3. Train 4DGS (next step)

Runs in the **CUDA/PyTorch** env (the repo's existing `.devcontainer` or your GPU box), not
the Isaac Sim container. The multipleview path needs a per-dataset config at
`arguments/multipleview/<name>.py` (a `pump01.py` is provided), then:

```bash
# convenience wrapper (train + render):
bash omniverse_pipeline/train_pump.sh pump01

# equivalently, by hand (see README_4DGS.md):
python train.py  -s data/multipleview/pump01 --port 6017 \
    --expname "multipleview/pump01" --configs arguments/multipleview/pump01.py
python render.py --model_path "output/multipleview/pump01/" --skip_train \
    --configs arguments/multipleview/pump01.py
# -> trained Gaussians in output/multipleview/pump01/
```

## 4. Segment + amplify (after 4DGS)

Feed the trained Gaussians to segmentation (MBS / trajectory clustering — see
`../.claude_notes/NOTES_4dgs_motion_segmentation.md`), then run per-segment motion
amplification with `render_amp.py`. Score any segmentation method against the synthetic
ground truth: propagate `data/multipleview/pump01/gt_segmentation.npz` (per-init-point
instance labels) to the trained Gaussians by nearest init-point, and compare to the method's
labels (IoU/ARI). `..._animated_motion_groups.json` maps each part to its motion segment —
the whole reason for going synthetic.

## Known limitations / next steps (see notes)

- The **pump** asset (`CONJUNTO BOMBAS.usd`) is a **single merged mesh** — one instance, no
  moving parts. Use `SceneAssembly.usd` (animated CL6 line + drum fan) or author a controlled
  multi-part rig. To make the pump usable it must be split into parts and given motion.
- `SceneAssembly.usd` has unresolved refs (S3 drum-fan payload; two `metricsAssembler` unit
  sublayers under `N_02_PCB_Router`). Localize/fix these so headless open is clean.
- Motion should be **subtle + periodic** (few-pixel amplitude) to exercise motion amplification.
- `points3D` init currently uses mesh **vertices**; switch to Poisson-disk surface sampling if
  the vertex distribution is too uneven.
```
