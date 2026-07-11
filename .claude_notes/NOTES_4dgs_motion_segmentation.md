# Adapting MultiBodySync motion segmentation to 4DGS data

Working notes, adaptation plan, and open questions.
Author aid: exploration of `submodules/multibody-sync-4dgs`, the 4DGS codebase, and the two papers in `papers/`.
Date: 2026-07-04

---

## 0. TL;DR

MultiBodySync (MBS) is built to solve a problem 4DGS *doesn't have*: recovering
correspondence and consistency across a set of **unordered, uncorresponded** point-cloud
scans. Its whole front half (pairwise scene-flow estimation + weighted permutation
synchronization) exists only to establish "which point in scan *k* is which point in
scan *l*".

In 4DGS the scene is a **single canonical set of `N` Gaussians** plus a **deformation
field** `D(x, t)`. The same Gaussian keeps its identity at every timestep, so we get
per-point trajectories `p_i(t)` for free, at arbitrary temporal resolution, plus extra
per-point signals (rotation, scale, opacity, SH color). That changes the task from
"jointly segment + correspond + register unordered scans" to essentially **"cluster
known per-point trajectories into rigid/coherent motion groups"**.

So the interesting, reusable part of MBS is its **motion-segmentation core** (grouping
points that share a rigid SE(3) motion, and the segmentation-synchronization idea). The
flow + permutation-sync machinery is largely redundant for us. This document lays out the
mismatch precisely and gives three adaptation options with trade-offs rather than
committing to one.

---

## 1. What MultiBodySync expects (source: `submodules/multibody-sync-4dgs`, CVPR 2021 paper)

### 1.1 Pipeline
Four stages, trained separately then combined (`train.py` dispatches on `--type` =
`flow | mot | conf | full`; models in `models/`):

1. **FlowNet** — pairwise scene flow `F_kl = φ_flow(X_k, X_l)` (PointPWC-style, 4-level
   pyramid 512→128→32→8). Establishes soft correspondence.
2. **ConfNet** — confidence on each flow/match (OANet-style `PointCN`).
3. **Weighted permutation synchronization** — closed-form spectral relaxation that makes
   the pairwise correspondences globally consistent across all `K` scans.
4. **MotNet** — per-point 12-D rigid transform (3×3 R + 3 t), + pairwise affinity/grouping,
   followed by **motion-segmentation synchronization** and per-segment Kabsch/SVD fitting.
   The number of bodies `S` is decided by an eigenvalue cutoff (`alpha`, e.g. 0.05).
The whole thing is **iterated** (`TestTimeFullNet`, `n_iter=4`) to refine.

### 1.2 Input format (this is the crux of the mismatch)
- On disk: `<dataset>/data/000000.npz`, keys `pc`, `segm`, `trans`.
- `pc`: **`(K, N, 3)`** — `K` scans/views (typically 2–4), `N` points, **xyz only**
  (no color, no normals, no time).
- `segm`: `(K, N)` integer per-point body labels (ground truth, for training).
- `trans`: dict of `(K, 4, 4)` SE(3) matrices per body + camera.
- **Points are unordered and uncorresponded across the `K` scans** — this is the entire
  reason FlowNet + permutation sync exist.
- Assumed **rigid** bodies (each segment moves by one SE(3) transform).

### 1.3 Output format
- `motion_absolute`: `(B, N, K, S)` per-point soft (train) / hard (test) segment labels.
- `raw_flow_dict`, `rigid_flows`: per view-pair flows.
- Per-segment SE(3) transforms via weighted Kabsch.

### 1.4 Key assumptions / constants
- `train_s = 6` (forced 6 segments in training), spectral eigenvalue cutoff picks `S` at test.
- 2–4 views; batch size 1 at test (`TestTimeFullNet` asserts it).
- Custom CUDA ops in `ext/` (FPS, ball-query, grouping, interpolation) — **must be compiled**
  (`torch.utils.cpp_extension.load`, needs CUDA toolkit + nvcc).
- Pretrained weights (articulated / solid) are Google-Drive downloads; `ckpt/` is empty
  except `.gitignore`.
- Deps: `pyquaternion, open3d==0.11.2, tensorboardx, pyyaml, tqdm` (+ torch, scipy, sklearn
  implied).
- **The fork `multibody-sync-4dgs` is currently unmodified vanilla MBS** — no 4DGS glue yet.
  It's a starting template.

---

## 2. What 4DGS provides (source: 4DGS codebase, 2310.08528v3)

### 2.1 Representation
Canonical `GaussianModel` (`scene/gaussian_model.py`), per-Gaussian attributes:

| attr | shape | meaning |
|---|---|---|
| `_xyz` | `(N, 3)` | canonical position |
| `_rotation` | `(N, 4)` | quaternion |
| `_scaling` | `(N, 3)` | log-scale |
| `_opacity` | `(N, 1)` | inverse-sigmoid opacity |
| `_features_dc` | `(N, 1, 3)` | SH DC (base color) |
| `_features_rest` | `(N, (deg+1)²−1, 3)` | SH rest |
| `_deformation_table` | `(N,)` bool | which Gaussians are deformable |

`N` is fixed for a **trained** model (densification changes `N` only during training).

### 2.2 Motion model — a deformation field, not per-frame clouds
Motion is a **canonical set + time-conditioned deformation** (HexPlane 6-plane
multi-resolution voxel encoder → tiny multi-head MLP), `scene/deformation.py` +
`scene/hexplane.py`. The network is **stateless**: given canonical attrs and a time
`t ∈ [0,1]`, it returns the full deformed state (not deltas at the API level):

```python
# utils/render_utils.py
def get_state_at_time(pc, viewpoint_camera):
    means3D, scales, rotations, opacity, shs = ...            # canonical
    time = torch.tensor(viewpoint_camera.time).repeat(N, 1)   # (N,1), t in [0,1]
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
        pc._deformation(means3D, scales, rotations, opacity, shs, time)
    return ...
```

Time is normalized `[0,1]` (`CameraInfo.time = idx / n_frames`); `scene.maxtime` holds the
range. You can sample **any** `t`, not just training keyframes.

### 2.3 Consequences that matter for segmentation
- **Correspondence is free.** Gaussian `i` at `t_a` is the same Gaussian at `t_b`; ordering
  is preserved. There is nothing to "match".
- **Dense trajectories.** For each Gaussian we can build `p_i(t)`, and also
  `q_i(t)` (rotation), `s_i(t)` (scale), `o_i(t)`, `SH_i(t)`.
- **Motion need not be rigid.** The deformation field is continuous/non-rigid, whereas MBS
  assumes piecewise-rigid bodies. Real 4DGS scenes may have deformable regions (cloth, faces,
  the DNeRF synthetic clips) — piecewise-rigid is an approximation.
- **Extra features exist** (opacity, scale, SH color) that MBS deliberately ignores. They
  can help or hurt segmentation.
- **Floaters / low-opacity Gaussians** are noise that has no analogue in clean MBS point clouds.

### 2.4 Existing extraction paths (reuse these)
- `export_perframe_3DGS.py` → `time_XXXXX.ply` per frame (full deformed Gaussians).
- `utils/render_utils.py::get_state_at_time` → deformed state tensors at a given `t`.
- User's motion-amp code (`render_amp.py`, `motion_amp/renderer.py`, `ampUI.py`) already
  stacks per-Gaussian per-frame values into `(N, d, T)` tensors and does temporal FFT
  filtering + amplification per channel (`pos3d, pos2d, rotation, scale, opacity, SHs, ...`).
  **This is exactly the trajectory tensor a segmentation stage would consume** — the plumbing
  to get `(N, 3, T)` position sequences already exists.

---

## 3. The core mismatch (why "different features imply a different approach")

| | MultiBodySync | 4DGS |
|---|---|---|
| Input | `K` unordered clouds, xyz only | 1 canonical cloud + deformation field |
| Correspondence | **Unknown** → must be learned (FlowNet) & synchronized | **Given** (persistent Gaussian identity) |
| Temporal structure | Non-sequential, sparse scans | Dense, arbitrary-`t` trajectories |
| Motion model | Piecewise **rigid** SE(3) | Continuous, possibly **non-rigid** |
| Per-point features | xyz | xyz + rot + scale + opacity + SH |
| Noise | clean synthetic / scans | floaters, low-opacity, redundant Gaussians |
| What's hard | correspondence + consistency | choosing/clustering the right motion signal |

The upshot: **the front two-thirds of MBS (flow + confidence + permutation sync) solve a
problem we've already solved.** What remains genuinely useful is the **motion-segmentation
idea**: group points by consistency of their SE(3) motion, decide the number of groups
spectrally, and fit per-group transforms. The open design question is whether to keep that
inside the MBS network or replace it with something simpler that exploits free correspondence.

---

## 4. Adaptation options (trade-offs — not yet a decision)

### Option A — Reuse MBS motion-segmentation head, drop flow/permutation-sync
**Idea:** Sample the deformation field at `K` timesteps to get `(K, N, 3)` with **known**
correspondence. Skip FlowNet entirely: feed ground-truth flow `F_kl = p(t_l) − p(t_k)`
directly into MotNet, and skip permutation synchronization (permutations are identity).
Keep MotNet + segmentation-synchronization + Kabsch.

- **Pros:** Reuses the trained, tested rigid-segmentation core and the spectral
  "how-many-bodies" logic. Least conceptual reinvention. MBS explicitly separates flow from
  motion, so injecting exact flow is natural.
- **Cons:** Still carries MBS's rigid-body assumption; still needs the `ext/` CUDA ops built;
  MotNet was trained on MBS flow statistics/scale — exact 4DGS flow may be out of distribution,
  so likely needs retraining/fine-tuning. `N` (100k–1M Gaussians) ≫ MBS's `N` (few thousand,
  subsampled to 256/512) → must subsample and propagate labels back.
- **Effort:** Medium. Mostly a data-adapter + retrain of the `mot`/`full` stages.

### Option B — Lightweight trajectory clustering (use MBS only as reference)
**Idea:** Build full trajectories `T_i = {p_i(t), q_i(t), s_i(t)}` and cluster directly:
motion-affinity from pairwise trajectory similarity (e.g. do points `i,j` admit a common
rigid transform over the window?), then spectral/graph clustering. No FlowNet, no MotNet.

- **Pros:** Exploits dense trajectories + free correspondence fully; no CUDA-op dependency;
  no large retraining; can be prototyped quickly in numpy/torch; naturally handles arbitrary
  temporal resolution; easy to fold in opacity/scale/color features.
- **Cons:** Loses MBS's learned robustness; need to design the affinity + choose cluster
  count ourselves; classic trajectory clustering can be sensitive to noise/floaters and to
  non-rigid regions.
- **Effort:** Low–Medium to prototype; Medium to make robust.

### Option C — Full MBS pipeline, adapted end-to-end
**Idea:** Keep the whole pipeline but retrain/finetune on 4DGS-derived data, possibly adding
extra input channels (opacity as confidence weight; scale/SH as features) and a non-rigid
term.

- **Pros:** Most faithful to the paper; end-to-end trainable; could handle harder scenes.
- **Cons:** Highest effort; retains machinery we don't need; requires building a labeled
  4DGS motion-segmentation dataset for supervision; longest path to first result.
- **Effort:** High.

**Recommendation for sequencing (not a commitment):** prototype **B** to get a fast baseline
and to *characterize the data* (how rigid is the motion? how bad are floaters?), while
standing up the data-adapter that **A** also needs. The findings from B directly inform
whether A/C's rigid assumption and retraining cost are justified.

---

## 5. Proposed plan (phased)

**Phase 0 — Environment & sanity (build/verify).**
- Build `submodules/*` CUDA ops (`simple-knn`, `depth-diff-gaussian-rasterization`, MBS `ext/`);
  confirm a trained 4DGS model loads and `get_state_at_time` runs.
- Download MBS pretrained weights; run `test_load.py` / `init_model.py` on `assets/laptop.npy`
  to confirm the vanilla model works before touching it.

**Phase 1 — Data adapter (needed by every option).**
- Write an extractor: trained 4DGS model → `(T, N, C)` trajectory tensor, reusing
  `get_state_at_time` / the motion-amp `generate_frame_data` plumbing.
- Add opacity/scale filtering to drop floaters; optional FPS subsampling to a manageable `N`
  with a label-propagation map back to all Gaussians.
- Define the on-disk format (mirror MBS `.npz` `pc (K,N,3)` for Option A; richer format for B).

**Phase 2 — Baseline segmentation (Option B).**
- Trajectory-affinity + spectral clustering baseline; visualize clusters by recoloring
  Gaussians and rendering (reuse the amp renderer). Sanity-check on DNeRF clips
  (`bouncingballs`, `jumpingjacks`, `standup`) where bodies are known.

**Phase 3 — MBS core integration (Option A).**
- Feed exact flow into MotNet + segmentation-sync + Kabsch, identity permutations.
- Fine-tune `mot`/`full` stages on 4DGS-derived flow if out-of-distribution.
- Compare against the Phase-2 baseline.

**Phase 4 — Evaluation & (optional) non-rigid handling.**
- Metrics: segmentation IoU/ARI vs. any available GT (DNeRF has per-object structure);
  motion-fit residual per segment; qualitative recolored renders.
- Decide whether non-rigid regions need a soft/deformable extension (Option C territory).

**Phase 5 — Write-up hooks for the thesis** (recolored renders, ablations on which
per-Gaussian features help).

Each phase has a **verification step** built in (run vanilla model first; visualize
clusters; quantitative comparison against baseline).

---

## 6. Open questions

**Scope / goal**
1. Is the end goal *segmentation* (label Gaussians into moving parts) as an end in itself,
   or a means to something (e.g. per-part motion amplification, editing, compression)? The
   answer changes how much rigid-motion fidelity we need.
2. Should segmentation be **static** (one labeling for the whole clip) or **time-varying**
   (parts can merge/split)? MBS produces one consistent labeling; 4DGS could support either.

- 1. The end goal is per-part motion amplification, to allow for spotting of subtle motions. As such small defects in the segmentation are not a massive problem. 
- 2. Static, as time-varying segmentation will add an extra dimension of complexity that we do not need

**Motion model**
3. How rigid are the target scenes? DNeRF synthetics (jumpingjacks, standup, mutant) are
   articulated-ish; `data/multipleview` (real, hand) may be more deformable. Do we commit to
   piecewise-rigid, or need a non-rigid/soft-assignment term?
4. Over what temporal window do we segment — the whole clip, or sliding windows? Fast local
   motion vs. slow global drift may need different windows.

- 3. Targets are going to be mostly machines which we can assume will be very rigid.
- 4. Whole clip, mostly periodic motion with potentially small amplitudes

**Data / scale**
5. Typical `N` for your trained models (10⁵–10⁶)? This decides subsampling strategy and whether
   MBS's 256/512-point core is viable without heavy label propagation.
6. How bad are floaters / low-opacity Gaussians in your outputs? Do we threshold by opacity,
   by `_deformation_table`, or by motion magnitude before clustering?
7. Which per-Gaussian channels should feed segmentation — position only, or also
   rotation/scale/opacity/SH? (Rotation trajectories are a strong rigid-motion cue MBS never had.)

- 5. Assume > 10^5 , 512 point subsamplling is most likely not going to work 
- 6. This should not be a major issue
- 7. position

**Method choice**
8. Preference on the A/B/C trade-off from §4 — fast baseline first, or go straight for
   reusing the MBS network?
9. Do we have (or can we synthesize) **ground-truth segmentation labels** for any 4DGS scene?
   Needed for Option C training and for quantitative evaluation of all options.
10. Is retraining MBS acceptable (GPU time, data prep), or must we stay training-free and
    lean on the pretrained weights / classical clustering?

- 8. We will most likely need to test multiple approaches
- 9. Highly unlikely ground truth for real scenes would be almost impossible to get, the first goal will be to create a pipeline for testing the algorithms based on Nvidia omniverse which might help
-  10. Yes retraining is ok.

**Infra**
11. Target GPU / CUDA version — do the MBS `ext/` ops and the 4DGS rasterizer build cleanly in
    the current `.venv` / Docker setup?
12. Do you want the segmentation output to plug back into the existing amp UI/renderer (recolor
    or amplify per segment), and if so in what format?

- 11. the devcontainer in docker can run the MBS
- 12. For now it does not matter, this will be of  concern later
---

## 6b. Phase 1+2 built: `motion_seg/` (2026-07-06)

First working implementation of the recommended sequencing (§4/§5: prototype Option B first).
New package `motion_seg/`:

- **`extract_trajectories.py`** (Phase 1 data adapter, **needs the training GPU env** — same
  as `train.py`/`render.py`). Loads a trained model via the same `Scene(dataset, gaussians,
  load_iteration=...)` + `get_combined_args` pattern as `render.py`/`export_perframe_3DGS.py`.
  Samples `utils/render_utils.get_state_at_time` at N evenly-spaced `t` (a lightweight
  `_TimeCam` stand-in is enough — that function only reads `.time` off its camera arg, no
  need for a real Camera). Writes `<model_path>/trajectories.npz`
  (`canonical_xyz (N,3), traj (N,T,3), opacity (N,), times (T,)`).
- **`rigidity_graph.py`** (Phase 2, Option B core — **pure numpy/scipy, no GPU/torch**): k-NN
  graph in canonical space, per-edge rigidity score = std-dev of pairwise distance over time
  (exactly 0 for a true rigid pair; a real signal because rigid transforms preserve pairwise
  distance and parts are spatially contiguous, so only part-boundary edges need cutting).
  Otsu-thresholds the (expected bimodal) edge-score distribution instead of a hand-picked
  absolute cutoff — the "right" cutoff depends on physical scale + the trained model's own
  position noise, which we can't calibrate without real GPU data. Connected components ->
  segments; tiny components get folded into their nearest big neighbor.
- **`segment_rigid.py`** — CLI wrapper (opacity-filters floaters first, label -1). Has
  `--selftest`: synthetic static-base + 6 independently-rotating parts (periodic, integer
  cycles, small-amplitude — mirrors `add_motion.py`'s real motion model), verifies recovered
  labels via ARI with **no GPU needed**. Verified: ARI 0.9988, 8 recovered segments vs 7 GT
  (one harmless extra boundary fragment) — confirms the core algorithm before ever touching
  real trained Gaussians.
- **`metrics.py`** — `adjusted_rand_index` + Hungarian `best_iou_matching`, no sklearn dep
  (repo only carries scipy) — kept small and dependency-light on purpose.
- **`evaluate_segmentation.py`** — scores predicted segmentation against
  `data/multipleview/<name>/gt_segmentation.npz` (now real, 107 instances, since the §5g/5h
  label-name bug was fixed). Point sets differ (GT = init cloud, predicted = trained/pruned
  Gaussians) but share the same coordinate frame (both went through `omni_to_4dgs.py`'s scale
  normalization) -> GT labels are nearest-neighbor-propagated onto predicted points before
  scoring. Also writes a color-per-segment PLY for a quick visual check in any mesh viewer.
  End-to-end verified on synthetic data (same fixtures as the selftest): ARI 0.9988, mean
  Hungarian IoU 0.982, PLY writer produces a valid binary PLY.
- **`run.sh <name>`** chains all three steps (mirrors `omniverse_pipeline/train_pump.sh`'s
  style); added `scipy>=1.11` to `pyproject.toml` (new dependency this package needs).
- **Not yet run against real pump01 data** — `extract_trajectories.py` needs the GPU
  devcontainer (this sandbox has no CUDA); the pure-numpy pieces are verified on synthetic
  data only so far. Next real step: `./motion_seg/run.sh pump01` after training completes,
  then look at `segmentation_preview.ply` and the ARI/IoU report — tune `--threshold-mult`
  /`--opacity-thresh`/`-k` if segments look under/over-merged (e.g. `frame_base` accidentally
  swallowing a barely-moving small part, or one physical part splitting into 2-3 segments).
- Option A (reuse MBS's MotNet core with injected exact flow) and Option C (full MBS retrain)
  are still open per §4/§8 if Option B's quality isn't sufficient once run for real.

## 6c. First real pump01 run — poor results, root cause, fixes (2026-07-06)

First real (GPU) run of `motion_seg/run.sh pump01`: extraction worked (119401 non-floater
points after opacity filtering), but segmentation was bad — only 4-5 segments recovered vs.
107 GT, ARI 0.05, mean IoU 0.13. Diagnosed via the printed `info` dict:
`n_kept_edges=910077/910144` — **99.99% of k-NN edges were kept**, i.e. the rigidity graph
was barely cut at all, so almost everything merged into one blob (`gt_label 0` = frame_base
absorbed nearly everything: pred_size 119313 out of 119401).

- **Root cause: `otsu_threshold` (linear-histogram Otsu) fails on the real edge-score
  distribution.** It's heavily right-skewed — most edges tiny (median 0.00125), a handful of
  outliers reaching ~0.5 (likely under-trained/noisy Gaussians with erratic per-frame
  positions) — so linear histogram bins put nearly all the real mass into 1-2 bins near zero
  and let the rare outliers stretch the axis; Otsu then finds "outliers vs. everything else"
  (threshold ≈0.38) instead of "rigid vs. non-rigid" (should be close to the ~0.0013 median).
  Verified the mechanism on a synthetic distribution shaped like the real one: linear Otsu
  picked 0.128 (keeps 99.997%), log-space Otsu picked 0.0013 (keeps ~49%) — same failure mode
  reproduced and fixed.
- **Fix:** added `otsu_threshold_log()` to `rigidity_graph.py` (Otsu in log-space, mapped back
  to linear) and switched `segment_by_rigidity` to use it by default. `--threshold-mult` is
  still there for manual retuning after looking at the (also new) preview PNG.
- **Caveat — this fixes the mechanical bug, may not fully fix quality.** The real median
  (0.00125) and p90 (0.0035) are only ~3x apart, not a dramatic bimodal gap — suggests the
  trained model's own frame-to-frame position noise may be comparable in scale to the true
  cross-part motion signal (consistent with the "improve the scene later" acknowledgment from
  the user — this is a first-pass, not-yet-refined scene/training run). If segments still look
  wrong after the fix, `--threshold-mult` (loosen/tighten) and `-k` (neighbor count) are the
  knobs; a genuinely low-SNR reconstruction may need better training before Option B improves
  further.
- **Also added (same session): PNG visualization.** User: PLY files are hard to reason about.
  New `motion_seg/visualize.py` (matplotlib, pure numpy, no GPU) — `render_segmentation_png`
  (3-view top/front/side scatter, colored per segment) and `render_comparison_png` (GT vs.
  predicted, 2 rows x 3 views). `segment_rigid.py` now writes `<out>_preview.png` by default;
  `evaluate_segmentation.py` writes `<pred>_vs_gt.png` by default (both still support the old
  PLY output too, opt-in). Added `matplotlib>=3.8` to `pyproject.toml`. Verified the rendering
  code directly (not just via Read) — produces valid, correctly-sized PNGs.
- `run.sh` updated: forwards extra args to `segment_rigid.py` (so `./motion_seg/run.sh pump01
  --threshold-mult 2` re-tunes without re-running the GPU extraction step) and supports
  `SKIP_EXTRACT=1` to reuse an existing `trajectories.npz`.

## 6d. Option A prototype: `motion_seg/mbs_infer.py` (2026-07-06, UNTESTED)

User asked to see how MBS inference itself would work on this data, alongside Option B.
Investigated `submodules/multibody-sync-4dgs` in depth (two research passes) before writing
anything — key findings that shaped the adapter:

- **No pretrained weights are vendored** (`ckpt/` is empty). `hubconf.py` points at a Google
  Drive-hosted checkpoint for the FULL pipeline (flow+conf+mot combined) — must be downloaded
  by the user; `torch.hub.load_state_dict_from_url` may choke on Google Drive's large-file
  redirect, in which case download manually and pass `--checkpoint`.
- **`MotNet` (`models/mot_net.py:202`)** — `forward(xyz, flow, sub_inds)`: `xyz (B,2,N,3)`
  (exactly 2 views per call), `flow (B,2,N,3)`, `sub_inds (B,2,256)` (a shared-per-view FPS
  subsample used internally). Returns `(pred_trans, group_matrix, res0, res1)` —
  `group_matrix` (test.py's `motion_ij`) is a `(B,256,256)` raw-logit pairwise "same rigid
  body" affinity; `.sigmoid()` (test.py:333) is the only activation needed. MotNet does NOT
  itself decide segment count/labels — that's `sync_motion_seg` (spectral, `utils/sync_util.py`).
- **No existing code path skips FlowNet/permutation-sync** — `test.py`'s `TestTimeFullNet` is
  the only inference driver and always calls FlowNet + perm-sync every iteration. Had to write
  new orchestration from scratch, modeled on `TestTimeFullNet.forward` (test.py:227-425) minus
  the flow-inference and `compose_dense`/`sync_perm` permutation-sync block (test.py:246-311).
- **Permutation-sync is genuinely infeasible at 4DGS scale anyway, not just skippable-for-
  convenience**: `symm_flow_to_perm` does an all-pairs `cdist` over the FULL per-view point
  count, and `sync_perm` eigendecomposes an `(n_view*N, n_view*N)` matrix — at N=100k this is
  ~40GB+ for one `cdist` alone. Confirms the design notes' N-scale concern (§6, open-Q #5):
  MBS's machinery (even setting aside the permutation-sync we're skipping) is built for
  N~256-1024, so `mbs_infer.py` opacity-filters + randomly subsamples to a working set (default
  4000) before ever calling MotNet.
- **Key simplification made possible by free correspondence**: since every "view" in our case
  is the *same* physical Gaussians at a different time (not different scans with different,
  uncorresponded points like vanilla MBS), we use ONE shared 256-point FPS subsample across all
  K views instead of MBS's per-view-independent subsample + permutation to align them — this
  *is* "skip permutation synchronization, it's the identity" made concrete.
- **`mbs_infer.py`** (new file): loads `trajectories.npz`, builds exact analytic flow per view
  pair (`flow[:,0]=pos_j-pos_i`, `flow[:,1]=pos_i-pos_j` — no FlowNet), runs MotNet per pair,
  combines via `compose_dense`+`sync_motion_seg` (both imported from the MBS source, not
  reimplemented, to minimize semantic-mismatch risk), averages the K per-view soft assignments
  for each of the 256 sub-sampled points (they're the same physical point observed K times, so
  we want one static label, not a per-view one — matches the design decision for static
  segmentation), then propagates labels to the full working set via `feature_propagation`
  (3-NN, imported from `models/full_net.py`) in **canonical** (rest-pose) space.
- **Explicitly NOT verified end-to-end** — no GPU, no compiled `ext/` CUDA ops, no downloaded
  checkpoint available in this sandbox. Only syntax-checked. The module docstring in
  `mbs_infer.py` has the full prerequisite/setup list (build `ext/`, get the checkpoint,
  possible fine-tuning need) and flags the known out-of-distribution risk: MotNet was trained
  on MBS's own noisy FlowNet flow at roughly unit-scale point clouds, not exact zero-noise
  4DGS flow at our `--target-radius=4.0`-normalized scale — may need fine-tuning to work well,
  exactly as anticipated in §4 Option A's original trade-off writeup.
- **Next step when actually run:** expect to debug real shape/behavior mismatches — this is a
  best-effort translation of intricate, undocumented tensor-shape conventions, not something
  verifiable without the actual GPU environment.

## 7. File pointers (quick reference)

- Option B implementation: `motion_seg/{extract_trajectories,rigidity_graph,segment_rigid,
  metrics,evaluate_segmentation,visualize}.py`, `motion_seg/run.sh <scene_name>`.
- Option A prototype (untested, see §6d): `motion_seg/mbs_infer.py`.
- MBS pipeline: `submodules/multibody-sync-4dgs/{train,test}.py`, `models/full_net.py`,
  `models/mot_net.py`, `dataset.py`, `utils/sync_util.py`.
- 4DGS motion: `scene/gaussian_model.py`, `scene/deformation.py`, `scene/hexplane.py`,
  `utils/render_utils.py::get_state_at_time`.
- Per-frame / trajectory extraction: `export_perframe_3DGS.py`, `merge_many_4dgs.py`,
  `motion_amp/renderer.py`, `render_amp.py` (`generate_frame_data`, `amplify_frame_data_eulerian`).
- Papers: `papers/2310.08528v3.pdf` (4D-GS), `papers/Huang_MultiBodySync_..._CVPR_2021.pdf`.
