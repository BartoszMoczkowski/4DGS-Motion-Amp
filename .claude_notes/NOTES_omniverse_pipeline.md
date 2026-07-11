# Omniverse → 4DGS → MBS → Motion-Amp testing pipeline

Working notes for the *testing pipeline* whose first goal is to generate controllable
synthetic multi-camera datasets from NVIDIA Omniverse / Isaac Sim, reconstruct them with
4DGS, segment them (MBS or alternatives), and run per-segment motion amplification.

Date: 2026-07-04. Companion to `NOTES_4dgs_motion_segmentation.md` (the MBS↔4DGS analysis).

---

## 0. Why synthetic data (the point of this pipeline)

Real captures give us **no ground truth**: no exact camera poses, no per-part motion
segmentation labels. From `NOTES_4dgs_motion_segmentation.md` open-Q #9 — GT for real scenes
is effectively impossible. Omniverse/Isaac Sim fixes this: it *knows* the exact camera
intrinsics/extrinsics and can emit per-pixel instance/semantic masks and per-object
transforms every frame. So a synthetic pipeline gives us:

1. **Exact camera poses** → we can skip (or validate) COLMAP, removing a lossy, failure-prone
   stage from the 4DGS front end.
2. **Ground-truth segmentation labels** → the only way to *quantitatively* evaluate MBS /
   trajectory-clustering (IoU / ARI) instead of eyeballing recolored renders.
3. **Controllable motion** → we can dial in subtle, periodic, small-amplitude machine motion
   (exactly the motion-amp target) and know the true displacement.

## 1. Locked decisions (from user Q&A, 2026-07-04)

- **Capture mode:** headless standalone Python (via Isaac Sim's `python.bat`) is the target.
  User has so far run things through the `Physical-AI-Learning-KAT` demo app GUI; wants to
  move to headless for repeatability.
- **Ground truth:** YES — export exact camera poses **and** instance/semantic segmentation +
  per-object transforms. Use GT poses as the primary path; COLMAP kept only as optional
  comparison.
- **Target scene:** the pump assembly (`assets/pump_radnom/CONJUNTO BOMBAS.usd`) is the
  nominal target, BUT the scenes are unfinished and we should improve them. In practice the
  richer `assets/physAI_start/SceneAssembly.usd` is the better first target (see §2).
- **Cameras:** 8–12 static cameras, count/arrangement **configurable** (override in config).
- **From `NOTES_4dgs_motion_segmentation.md`:** end goal = per-part motion amplification
  (small seg errors OK); **static** segmentation (one labeling per clip); targets are rigid
  machines; whole-clip, mostly periodic, small-amplitude motion; N > 1e5 Gaussians; retraining
  MBS is acceptable; segmentation feature = position.

## 2. Discovered state of `Q:\Omniverse` (2026-07-04)

- **Isaac Sim 5.1.0-rc.19** source at `ISAAC_SIM/IsaacSim` (has `tools/packman/python.bat`).
  **KAT app** `Physical-AI-Learning-KAT` (Kit 107.3) built under `_build/windows-x86_64`.
- **Assets:**
  - `pump_radnom/CONJUNTO BOMBAS.usd` — Y-up, cm units, bbox ~1195×2542×2039 cm.
    **Single merged `Mesh`** (`/World/node_/mesh_`), one material, no sub-parts, no animation.
    → **Not segmentable as-is**: motion segmentation needs multiple independently-moving rigid
    parts. This asset is one static blob. Needs splitting into parts + authored motion, or use
    it only as static background.
  - `physAI_start/SceneAssembly.usd` — **Z-up, cm units, ~30.9k prims / 8.7k meshes.**
    Contains: `Factory` (ref), `CL6_Line_Full` (a 7-station PCB assembly line, N_01…N_07,
    **animated**, timeline 0–500), `CONJUNTO_BOMBAS` (the pump, single mesh), a `DrumFan`
    driven by an OmniGraph `PushGraph` (rotating), **4 cameras already defined**
    (`/Cameras/Camera_01..04` + a `/Camera`), and an existing `Replicator/SDGPipeline`
    OmniGraph. Timeline 0–20 on the assembly layer.
    → This is the real working scene and the better first target: it already has animated,
    multi-part machinery and cameras.
  - `physAI_start/Factory/Factory.usd` (Z-up, timeline 1492–2430) and
    `Assemblies/CL6_Line_Full.usd` (Z-up, **mm units 0.001**, timeline 0–500).
- **Unfinished / broken references (the "not finished" part):**
  - `DrumFan_A03_01` payload points to an S3 URL (`omniverse-content-production…`) → needs
    internet at stage-open, or localize the asset.
  - Camera proxy meshes reference `…/_build/windows-x86_64/release/kit/resources/models/camera/camera.usd`
    (KAT build dir) — resolves on the user's machine, harmless (proxy geometry only).
  - Two `metricsAssembler` sublayers under `Machine_USD/N_02_PCB_Router/Source/Component/`
    (`gantry_picker.usd`, `gantry_cutting.usd`) fail to load → **unit mismatch fixups** that
    are missing. Likely a units/scale inconsistency (CL6 is mm, SceneAssembly is cm) to fix.
- **Existing capture code:** `scripts/rep_test.py` and `New Text Document.txt` — minimal
  single/two-camera `BasicWriter` RGB grabs, 20 frames, 2048×1024, default `rgb_XXXX.png`
  naming, output to `renders/replicator_test/` (loose files + `Replicator_02..05/rgb/`).
  → Not multi-cam-structured, no poses, no segmentation. Superseded by the new capture script.

## 3. Target 4DGS input format (what the converter must produce)

From `scene/multipleview_dataset.py`, `multipleviewprogress.sh`, `multicam.bat`, the 4DGS
`multipleview` loader expects, per scene `data/multipleview/<name>/`:

- `cam01/…camNN/` each with `frame_00001.jpg, frame_00002.jpg, …` (JPG, 1-indexed, zero-pad 5).
  **All cameras must have the same number of frames**, and frame *i* across cameras is the
  same timestamp. Time is assigned `t = i / image_length`.
- `sparse_/` — COLMAP model (`cameras.bin`/`.txt`, `images.bin`/`.txt`, `points3D.bin`/`.txt`).
  The loader reads camera intrinsics from `cam_intrinsics[1].params[0]` (focal) + width/height,
  and per-image extrinsics (`qvec`,`tvec`, name `cam<NN>.…`). Image `.name` basename is parsed
  `name[5:-4]` → the camera number (so image names like `cam01.jpg`).
- `points3D_multipleview.ply` — initial point cloud for Gaussian init.
- `poses_bounds_multipleview.npy` — LLFF array `(N_cams, 17)`: 15 = 3×5 pose
  `[R(3×3) | t(3×1) | hwf(3×1)]` flattened, + 2 = near/far. Used only for the spiral test
  camera path (`get_video_cam_infos`). LLFF convention: loader does
  `poses = concat([poses[...,1:2], -poses[...,:1], poses[...,2:4]], -1)` (down-right-back →
  right-up-back style remap). Converter must emit poses in LLFF's `[-u, r, -view]` camera-to-
  world layout with `hwf` = [height, width, focal].

**The key simplification:** since Omniverse gives exact poses, the converter can write
`sparse_` and `poses_bounds` **directly from GT** — no COLMAP feature-matching/mapping needed.
Only the initial `points3D` cloud needs a source (options in §5).

## 4. Pipeline architecture

```
 SceneAssembly.usd  ──(author: motion + semantics + camera rig)──►  prepared stage
        │
        ▼   omni_capture.py   (headless Isaac Sim 5.1, BasicWriter)
 capture/<scene>/
   camNN/rgb_XXXX.png                (synchronized multi-view RGB)
   camNN/instance_segmentation_XXXX.png + *_mapping.json   (GT masks)
   camNN/camera_params_XXXX.json      (GT intrinsics + view transform)
   transforms/frame_XXXX.json         (per-object world transforms — GT motion)
        │
        ▼   omni_to_4dgs.py   (pure Python, no Isaac dependency)
 data/multipleview/<scene>/
   camNN/frame_XXXXX.jpg
   sparse_/{cameras,images,points3D}.txt
   points3D_multipleview.ply
   poses_bounds_multipleview.npy
   gt_segmentation.npz   (per-3D-point / per-object labels for MBS eval)
        │
        ▼   train.py (4DGS multipleview)  →  output/<scene>/  (trained Gaussians)
        ▼   MBS / trajectory clustering    →  per-Gaussian segment labels
        ▼   render_amp.py per segment       →  amplified renders
```

## 5. Open design points / TODO for this pipeline

1. **points3D source.** Options: (a) sample points on the USD meshes (uniform surface
   sampling via pxr) with GT colors — cleanest; (b) triangulate from Replicator depth +
   GT poses; (c) fall back to COLMAP dense fusion. Start with (a) — a mesh surface sample is
   trivial to generate GT-labelled (each sampled point inherits its prim's instance id → this
   *is* the GT 3D segmentation for evaluation).
2. **GT 3D segmentation for eval.** Because 4DGS Gaussians are initialised from `points3D` and
   keep identity, if we label the init cloud by prim/instance, we can propagate labels to
   Gaussians (nearest-init-point) and compare MBS output against them. Store in `gt_segmentation.npz`.
3. **Scene authoring (the "improve the scenes" task).** For a first clean test we likely want
   a *controlled* scene: a few rigid parts with known periodic motion (small amplitude), known
   instance labels, good lighting, static cameras ringing the subject. Candidates:
   - Fix `SceneAssembly` (localize S3 drum-fan, resolve the two metricsAssembler unit fixups,
     confirm animation plays headless), OR
   - Build a **minimal parametric rig** USD from scratch (2–4 primitives on sinusoidal
     transforms) to validate the whole plumbing before fighting the big scene. Recommended as
     step 0 because it's fully controllable and needs no broken-ref fixes.
4. **Units / up-axis.** SceneAssembly is Z-up cm; CL6 is mm. 4DGS/COLMAP conventions are
   Y-down camera / world-scale agnostic but poses_bounds near/far must be sane. Converter must
   handle the world→LLFF axis remap explicitly and be told the stage up-axis + metersPerUnit.
5. **Camera rig.** 8–12 static cameras on a ring/dome around the subject, looking at its bbox
   center, radius ≈ 2–3× bbox radius. Generated in `omni_capture.py` from config; must be
   static across time (4DGS multipleview assumes fixed cameras, moving scene).
6. **Motion for amp.** Machines should have subtle periodic motion (few-pixel amplitude) so
   motion amplification has something to reveal. The CL6 line motion may be large/translational;
   may need a dedicated "vibration" scene.

## 5b. Pump de-fusing — DONE (2026-07-04)

The pump `CONJUNTO BOMBAS.usd` single fused mesh is now split into separate rigid parts.

- **Method:** weld coincident vertices, split by connected components. The STL-from-CAD
  assembly keeps each physical body as a topologically separate shell, so connectivity
  recovers the parts. Tool: `omniverse_pipeline/split_mesh.py` (trimesh + usd-core, pure
  python, no Isaac).
- **Result:** **107 connected components.** One large `frame_base` (104,308 faces = the
  genuinely-welded skid + manifold; kept as static background) + 106 separate parts
  (compressor bodies, motors, valves, fittings — many in identical pairs/triples, consistent
  with the "3× BITZER" units). Total faces preserved exactly (208,906); welded verts 103,897.
- **Output:** `Q:\Omniverse\assets\pump_radnom\CONJUNTO_BOMBAS_segmented.usd` (1.86 MB, Y-up,
  mpu 0.01, bbox unchanged). Each part is `/World/CONJUNTO_BOMBAS/<name>` (Xform → Mesh) with
  a per-part `displayColor` and a `UsdSemantics.LabelsAPI` `class` label (names: `frame_base`,
  `part_001`…`part_106`). Verified: 107 part Xforms, 107 meshes, 107 labels, geometry intact.
  Preview: `outputs/pump_segmentation_preview.png` (3 views, frame greyed, movers colored).
- **Regenerate:** `python omniverse_pipeline/split_mesh.py --in "<pump>.usd"
  --out CONJUNTO_BOMBAS_segmented.usd --group CONJUNTO_BOMBAS --min-faces 1 --preview p.png`.
  Use `--min-faces N` to fold sub-N-face fragments into the nearest part.
- **Env note:** the `Q:\Omniverse` mount has cross-process read/stdout caching flakiness in
  this assistant sandbox; generate+verify in ONE python process (or to /tmp) then `cp` to Q:.
- Semantics for capture come free via `semantic_roots: ["/World/CONJUNTO_BOMBAS"]`.

## 5c. Per-part motion — DONE (2026-07-04)

The segmented pump now has subtle periodic per-part motion → a controlled motion-seg / motion-
amp test case with exact GT labels. Tool: `omniverse_pipeline/add_motion.py`.

- **Model:** each movable part gets one rigid SE(3) sinusoid, `M(t)=Trans(c+off)·Rot(θ)·Trans(-c)`,
  pivoting about the part centroid `c` (so rotation doesn't fling far parts across the world).
  Authored as a time-sampled `xformOp:transform` per part. `frame_base` stays static.
- **Subtle + periodic:** translation 1–4 mm; rotation specified as a **surface** displacement
  (0.5–3 mm) converted to an angle per part via its radius (capped at 3°) so big and small
  parts both move only a few mm at the surface. Frequencies are **integer** cycles over the clip
  (2–5) so motion loops seamlessly. Units handled via stage metersPerUnit (pump = cm).
- **Result (default cfg, 60 frames @24fps):** 106 moving parts, peak surface displacement
  **1.75–6.7 mm** (mean 4.1). Verified on the written USD: biggest part (r≈5 m) real mesh-point
  displacement 6.75 mm; loop residual 2e-13; frame_base identity; geometry intact (208,906 faces).
- **Output:** `Q:\Omniverse\assets\pump_radnom\CONJUNTO_BOMBAS_animated.usd` (2.73 MB, timeline
  0–59 @24fps) + `..._animated_motion_groups.json` (per-part → motion-segment map = GT). QA
  image: `outputs/pump_motion_qa.png`.
- **GT segments:** default = each part is its own motion segment (label = part index). Use
  `--groups K` to instead cluster parts into K shared-motion groups (K+1 segments incl. frame);
  the json records the mapping so segmentation IoU/ARI can be scored.
- **Regenerate:** `python omniverse_pipeline/add_motion.py --in CONJUNTO_BOMBAS_segmented.usd
  --out CONJUNTO_BOMBAS_animated.usd --num-frames 60 --fps 24 --trans-amp-mm 1 4
  --rot-surface-mm 0.5 3 --freq 2 5 --plot qa.png`.
- **Env caveat (again):** the `Q:` mount + Python module/.pyc caching in this sandbox is
  unreliable across edits; generate+verify to /tmp in one process, then `cp` to Q:. A stale
  `.pyc` / stale-read served old code repeatedly despite the source being correct.
- **NEXT:** capture this USD (`usd_path: .../CONJUNTO_BOMBAS_animated.usd`,
  `semantic_roots: ["/World/CONJUNTO_BOMBAS"]`) → convert → train 4DGS → test segmentation
  against the motion_groups GT. Optionally add a light to the scene for capture.

## 5d. Isaac Sim + Claude Code dev container — DONE (2026-07-04)

To let a *locally-run* Claude Code agent execute the capture on the user's RTX GPU (the hosted
Cowork sandbox has no GPU/Docker), built a dev container combining Isaac Sim 5.1 + Claude Code.
Files in `omniverse_pipeline/devcontainer_isaacsim/` (the repo `.devcontainer/` is a protected
path in Cowork, so the user copies this folder to `.devcontainer/isaacsim/`):
- `Dockerfile` — FROM `nvcr.io/nvidia/isaac-sim:5.1.0` (rootless UID 1234, HOME=/isaac-sim),
  adds Node 20 + `@anthropic-ai/claude-code`, sets `CLAUDE_CONFIG_DIR=/isaac-sim/.claude`,
  `OMNI_KIT_ACCEPT_EULA=YES`.
- `devcontainer.json` — `runArgs:["--gpus","all","--network=host","--entrypoint","/bin/bash"]`,
  `containerEnv` ACCEPT_EULA/PRIVACY_CONSENT, `containerUser/remoteUser: "1234"`, mounts:
  workspace bind, `${localEnv:OMNIVERSE_DIR}`→`/omniverse` bind, named volumes for isaac cache
  (`.cache`,`.nv/ComputeCache`,`.local/share/ov/data`) + `claude-config`.
- `capture_config_pump.container.yaml` (in omniverse_pipeline/) — Linux-path twin of the pump
  config (`/omniverse/...`, out `/omniverse/renders/capture_pump`).
- `run_capture.sh` — `/isaac-sim/python.sh omni_capture.py --config <container cfg>`; smoke
  test with `--n-cameras 2 --frames 2`.
- `README.md` — NGC login, WSL2 GPU, Q:→/omniverse mount, open/run steps, caveats.
Validated: JSONC + YAML parse, bash syntax OK. NOT executed (no GPU/Docker here) — first
`Reopen in Container` is the real test. Key facts: image runs rootless UID 1234, needs
`docker login nvcr.io` + EULA; `/isaac-sim/python.sh` runs standalone scripts headless.

## 5e. Dark patterned background + 4DGS step — DONE (2026-07-05)

Capture works end-to-end on the animated pump (verified by user). Two additions:
- **Background:** `omni_capture._setup_lighting` now sets a DomeLight *environment texture* for a
  dark, patterned backdrop (better contrast + stable features for 4DGS). `lighting.dome_texture:
  generate` renders a near-black equirect grid (base/line/dot colors + `bg_step` configurable)
  to `<capture_dir>/bg_dome.png` at runtime via a stdlib-only PNG writer (`_write_png` /
  `_make_bg_texture`) — no Pillow needed. Enabled in both pump configs; distant key raised to
  4000 since the dark dome gives little fill. (The dome env is the visible background, so no
  backdrop-geometry/clip-plane issues.)
- **Config loader:** `_load_config` no longer hard-depends on PyYAML — falls back to a built-in
  parser (`_parse_simple_yaml`) for our config subset (verified identical to PyYAML incl. the
  new inline color lists). Fix for Isaac Sim python lacking pyyaml.
- **Semantics:** `_apply_semantics.tag()` rewritten to try Isaac's `add_update_semantics`
  (semantic_label/type_label) then raw USD schemas (UsdSemantics.LabelsAPI + legacy
  Semantics), each guarded so a wrong signature falls through (the old code only caught import
  errors, so a bad call signature aborted the run).
- **4DGS next step:** `arguments/multipleview/pump01.py` (config; time-res 150 for 60 frames),
  `omniverse_pipeline/train_pump.sh` (train + render), README §3/§4. Flow:
  `omni_capture -> omni_to_4dgs --name pump01 -> train_pump.sh pump01 -> output/multipleview/pump01`
  then segmentation scored vs `gt_segmentation.npz` / `*_motion_groups.json`. Training runs in
  the CUDA devcontainer, NOT the Isaac Sim one.

## 5f. Bug fix — omni_to_4dgs.py found 0 frames per camera (2026-07-06)

`convert()`'s `_load_frames_list` did `os.listdir(capture_dir/camNN)` looking for image files
directly in that folder. But `omni_capture.py` (Replicator `BasicWriter`) actually nests each
annotator's output in its own subfolder: `camNN/{rgb,camera_params,instance_segmentation,
semantic_segmentation}/`. So `os.listdir` only ever saw subdirectory names (no `.png`/`.jpg`
extension) → matched 0 files for every camera, for every capture (not scene-specific).
- **Fix:** in `convert()`, prefer `camNN/rgb/` as the source folder when it exists, else fall
  back to `camNN/` directly (kept for capture layouts without the subfolder nesting).
- **Verified:** re-ran frame discovery against the real `capture_pump` output (10 cams) — all
  10 now resolve 60/60 frames (`rgb_0000.png`…`rgb_0059.png`), matching the animated pump's
  60-frame clip. Full `convert()` (image re-encode + COLMAP bins) not re-run end-to-end here.
- **Next:** re-run `omni_to_4dgs.py --capture omniverse_pipeline/omniverse_out/capture_pump
  --out . --name pump01` for real to regenerate `data/multipleview/pump01/` with actual frames,
  then `train_pump.sh pump01`.

## 5g. Training crash — nan loss reexec hits "Address already in use" (2026-07-06)

First real `train_pump.sh pump01` attempt: loss went `nan` mid-training, which trips the
upstream 4D-Gaussians self-restart hack in `train.py` (`if torch.isnan(loss): ...
os.execv(sys.executable, [sys.executable]+sys.argv)` — restarts the whole process from
scratch on the same argv, no checkpoint). The re-exec'd process then crashed immediately in
`network_gui.init()` with `OSError: [Errno 98] Address already in use` on the `--port` from
`train_pump.sh` (default 6017).
- **Cause:** `gaussian_renderer/network_gui.py`'s module-level `listener` socket had no
  `SO_REUSEADDR`. If a viewer had ever connected (or just from TIME_WAIT/kernel timing), the
  port isn't immediately rebindable, so the freshly re-exec'd process's `init()` call fails.
- **Fix (both applied):** `network_gui.init()` now sets `SO_REUSEADDR` before `bind()`; and
  `train.py`'s nan branch now calls `network_gui.listener.close()` before `os.execv`.
- **First hypothesis (partial):** nan hit right after `reset opacity` printed at iteration
  3000 — `pump01.py` left `opacity_reset_interval` at the base default (3000), colliding with
  `coarse_iterations`. Fixed by adding `opacity_reset_interval = 60000` (disabled), matching
  `arguments/multipleview/default.py`. This removed the "reset opacity" print but **nan still
  happened** at the exact same coarse->fine boundary — so this wasn't the (whole) cause.
- **Real root cause (found 2026-07-06):** `Scene.cameras_extent` (`scene/dataset_readers.py`
  `getNerfppNorm` — max camera distance from mean center, *1.1) directly sets
  `spatial_lr_scale` in `GaussianModel.training_setup`, which **every** position/deformation/
  grid learning rate is multiplied by. `omni_to_4dgs.py` wrote raw Omniverse stage-unit camera
  translations straight through, ignoring the capture's own declared `meters_per_unit` (0.01
  for the pump = centimeters). Measured on `capture_pump`: camera centers ~4450 units from
  center -> `cameras_extent` ≈ **4898**. Since `deformation`/`grid` params get zero gradient
  for the entire coarse stage (deformation network isn't invoked until "fine"), the very first
  fine-stage optimizer step applies `grid_lr_init (0.0016) * 4898 ≈ 7.8` to grid feature
  params initialized near O(1) — an explosive first step -> nan, independent of opacity reset.
  This is why the nan reappeared at the identical iteration even after the opacity fix.
- **Fix:** `omni_to_4dgs.py` now converts camera translations / near-far / point cloud through
  `meters_per_unit`, then additionally rescales the whole scene so its nerf++ camera radius
  lands at `--target-radius` (default 4.0 — matches the scale 4DGS's hardcoded LRs assume; the
  other dataset configs never override per-scene LRs, so they rely on COLMAP naturally
  producing scenes in this range). Verified via the actual scale math against `capture_pump`:
  radius 4898 (raw) -> 48.98 (after mpu) -> 4.0 (after target-radius normalization); grid_lr
  effective drops from 7.83 to 0.0064. Writes `scene_scale.json` next to the scene
  (`total_scale`, so internal-unit distances/motion can be divided back to physical meters for
  reporting against the mm-scale GT motion in `*_animated_motion_groups.json`).
- **Regenerated** `data/multipleview/pump01/` with the corrected scale (2026-07-06): printed
  `camera radius after unit-conversion=48.98 -> target_radius=4.0 (total scale x0.0008167)`;
  all 10 cams have 60/60 frames; `sparse_/` + `scene_scale.json` written. Training not yet
  re-run to confirm it completes clean — that's the next real test of both fixes together.
- **`gt_segmentation.npz` all-zero labels — root cause found + fixed (2026-07-06):**
  `omni_capture.py`'s `_sample_pointcloud` labelled each point with `prim.GetName()` on the
  **mesh** prim. But `split_mesh.py` builds every part as `/World/{group}/{part_name}/mesh` —
  so the mesh prim's own name is literally `"mesh"` for all 107 parts, collapsing every point
  onto one label. `_apply_semantics` (used for the rendered instance-seg PNG masks) already
  does this correctly by tagging `root.GetChildren()` (the per-part **Xform**, not the mesh) —
  the point-cloud path just used the wrong prim.
  - **Fix:** `_sample_pointcloud` now uses the mesh's *parent* Xform name
    (`prim.GetParent().GetName()`), matching `_apply_semantics` and `split_mesh.py`'s layout.
    Falls back to the mesh's own name if it has no useful parent.
  - **Not yet re-verified against real data** — `omni_capture.py` only runs inside Isaac Sim
    (needs `pxr`/`Usd`, not available in this sandbox), so this is a code-review-level fix, not
    a validated one yet.
  - **Important: does not fix already-captured data.** `capture_pump/points3D_labels.npy` /
    `label_names.json` on disk now are still all-zero (produced by the old code) — there's no
    way to recover the correct per-point part id from the current ply + all-zero labels alone.
    **Needs a fresh Isaac Sim capture run** (`omni_capture.py` again on `CONJUNTO_BOMBAS_
    animated.usd`) to pick up the fix, then re-run `omni_to_4dgs.py --name pump01` again.

## 5h. Test-video camera "spinning rapidly" — root cause + fix (2026-07-06)

First successful pump01 train+render: the `video_rgb.mp4` test-path camera moved erratically,
too fast to see anything. Cause: `scene/multipleview_dataset.get_video_cam_infos` builds the
render path via `scene/neural_3D_dataset_NDC.get_spiral`, which is a NeRF/LLFF **spiral**
heuristic (avg-pose direction + near/far-derived focus depth + 90th-percentile translation as
spiral radius) — designed for narrow-baseline, forward-facing camera arrays (DyNeRF/Neural3DVideo
style). `omniverse_pipeline` captures are the opposite: a static 360-degree ring/dome of
cameras looking *inward* at the subject, so the "average camera pose" `get_spiral` computes is
close to degenerate, producing a wobbly, uneven path.
- **Verified numerically** against `pump01`'s actual `poses_bounds_multipleview.npy`: old
  spiral path had radius 4.47 ± 0.68 (≈15% wobble) and frame-to-frame step 0.166 ± 0.021
  (max 0.20) — vs. a real circular orbit's expected ~0 variance in both.
- **Fix:** added `get_orbit()` to `scene/neural_3D_dataset_NDC.py` — fits the actual camera
  ring's center/up-axis/radius/height and sweeps a constant-radius circular path (1 full
  rotation over 300 frames @ 30fps = 10s) always looking at the scene center.
  `multipleview_dataset.get_video_cam_infos` now calls this instead of `get_spiral`. Verified
  against real pump01 data: radius std ~3e-16 (exactly constant), frame-to-frame step
  0.0502 ± 3e-16 (exactly constant) — a smooth, calm, predictable orbit.
- Only `scene/multipleview_dataset.py`'s video path changed — `get_spiral` itself is untouched
  and still used as-is by the dynerf/hypernerf/dycheck dataset readers, where it's actually the
  right heuristic (real forward-facing captures).
- Not yet visually confirmed (no GPU/renderer in this sandbox) — next render should show a
  slow, steady one-rotation orbit instead of the erratic spin.

## 6. Constraints / who runs what

- Isaac Sim / Replicator **cannot be run in this assistant's sandbox** (no GPU, Windows-only
  app). `omni_capture.py` is written against the 5.1 API and **must be run by the user** with
  `Q:\Omniverse\ISAAC_SIM\IsaacSim\...\python.bat` (or the KAT python). API calls are verified
  against the 5.1 docs but not executed here — treat first run as a smoke test.
- `omni_to_4dgs.py` is pure Python (numpy/PIL/plyfile) and **is** unit-tested in the sandbox on
  synthetic camera data.

## 7. File pointers

- New pipeline code: `omniverse_pipeline/` (in this repo): `omni_capture.py`,
  `omni_to_4dgs.py`, `capture_config.yaml`, `README.md`.
- 4DGS ingestion: `scene/multipleview_dataset.py`, `scene/dataset_readers.py`,
  `scene/colmap_loader.py`, `multipleviewprogress.sh`, `multicam.bat`.
- Omniverse: `Q:\Omniverse\assets\physAI_start\SceneAssembly.usd`,
  `Q:\Omniverse\assets\pump_radnom\CONJUNTO BOMBAS.usd`, `Q:\Omniverse\scripts\rep_test.py`.
- Isaac Sim 5.1 replicator getting-started + writers docs (see README links).
```
