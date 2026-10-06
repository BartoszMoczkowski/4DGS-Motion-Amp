# Correctness review — synthetic-data pipeline (omniverse-pipeline + scene-gen) (2026-10-05)

Scope: `omniverse-pipeline/omniverse_pipeline/` (`split_mesh.py`, `add_motion.py`, `compose_scene.py`, `omni_capture.py`, `omni_to_4dgs.py`, `rig.py`, capture YAMLs) and `scene-gen/` (`gen_scenes.py`, `run_grid_4dgs.py`, `run_grid_seg.py`, `frames_to_mp4.py`), cross-checked against downstream consumers and `runs/` artifacts. Both CPU selftests executed and pass.

## Confirmed bugs

### O1 — Init point-cloud colors normalized twice → all Gaussians initialize black (Low in practice)
`omni_to_4dgs.py:283` writes `rgb / 255.0` (0–1 floats), but the loader contract is 0–255: `core/scene/dataset_readers.py:128` divides by 255 again before `RGB2SH`. Verified numerically: intended 0.784 → loaded 0.003. The in-file comment shows the author knew the contract; the call site violates it. Vendored copy `orchestrator/pipeline/vendored/host/convert.py` carries the same bug. Low practical impact (init colors are random pseudo-colors; training recovers) but silently discards intended initialization.
**Fix: drop the `/255.0` in both copies.**

### O2 — Grid-scene trajectories extracted at 4× undersampling → 40-cycle motion aliases past Nyquist (Medium-High)
Grid scenes are 240 frames @ 60 fps with 40 motion cycles/clip (`data/scenes/grid/grid_manifest.json`). But `seg_extract.n_times` defaults to 60 (`orchestrator/pipeline/config/models.py:185`), no seg preset overrides it, `run_grid_seg.py` never sets it — actual run manifest shows `n_times: 60`. Sampling 40 cycles/unit-t at 60 points (Nyquist 30) aliases to ~20 cycles. Phase relationships survive, but every frequency-calibrated stage (rigid2 FFT denoising, kabsch FFT-fingerprint init, trajectory_denoise) operates on the wrong waveform. Plausible contributor to the near-zero grid ARIs (`runs/grid_seg_results.csv`: 0.002–0.009).
**Fix: set `seg_extract.n_times: 240` for grid runs and re-run seg.**

### O3 — Grid ground-truth amplitudes ignore the 0.2 compose scale → physical motion 5× smaller than recorded (Medium)
`gen_scenes.py:122` records `peak_surface_mm` in subject-internal units, but `compose_scene` applies `SCALE = 0.2` (`gen_scenes.py:54`) and the composed stage copies `metersPerUnit` (`compose_scene.py:81`). E.g. `pump_A8mm_M2`'s recorded amplified peak 22.98 mm is ~4.6 mm in the rendered world. Cell names (`A20mm`), `*_motion.json`, and recovered-vs-GT amplitude comparisons are off by exactly 5×.
**Fix: bake the 0.2 into `peak_surface_mm` or record the compose scale in `*_motion.json`.**

### O4 — Endpoint duplication + time-base mismatch between authoring, capture, and loader (Low-Medium, systematic)
`add_motion.py:152` authors `u = fi/(N-1)` → frame 0 and N−1 have identical poses; `omni_capture.py:252` samples `linspace(tl_start, tl_end, N)` (hits keyframes — good); but `multipleview_dataset.py:54` assigns training time `t = i/N`. Consequences: (a) one frame in N redundant; (b) integer cycles/clip become `f·N/(N−1)` cycles/unit-t — off DFT bins → spectral leakage in FFT-based amplification; (c) test-video/seg times reach past the last training time (extrapolation, worst for pump01: 0.983 vs 0.997). One accidental mercy: the duplicated endpoint makes the old `render_amp.py` wraparound delta zero.

### O5 — `run_grid_seg.py` reports stale metrics on failed runs (Medium-Low)
`run_grid_seg.py:227` reads `seg_eval_result.json` whenever it exists, regardless of run status; shared across impls, only overwritten on success. A run failing before seg_eval gets the previous impl's ARI/IoU next to `status=failed`.

### O6 — `count_gaussians` lexicographic sort (latent)
`run_grid_4dgs.py:104` `sorted(glob("iteration_*"))[-1]` — `iteration_7000` sorts after `iteration_30000`. Currently harmless (only 14000/15000 on disk), breaks on any schedule crossing a digit boundary.

## Suspicious / unconfirmed

- S1: Converter happily converts a failed/partial capture — `cameras_gt.json` written before rendering; `omni_to_4dgs.py:249-251` only warns on unequal frame counts; nothing checks `status.json`.
- S2: GT labels propagate onto background Gaussians (pump-only GT vs ~360 k trained Gaussians incl. factory background; `propagate_labels` assigns every Gaussian its nearest pump part) — plain-rigid grid ARI rows depressed independent of segmentation quality.
- S3: No post-hoc frame-sync verification for chunked capture (rests on USD keyframe determinism; writer numbering hiccup undetectable downstream).
- S4: Camera clipping range never authored in USD (relies on Isaac defaults; near/far only written to LLFF metadata).
- S5: `make_capture_variant` reuses partially-created variants (crash between mkdir and junction → silently reused forever).
- S6: `_parse_simple_yaml` fallback silently mis-parses block sequences (current configs comply).
- S7: `run_grid_4dgs.py` CSV append-only, duplicate run_ids (up to 3× in real data) — analysis must take last-row-wins; unenforced.
- S8: Doc drift — gen_scenes usage says `--base-amp-mm 1 4 16` (default is `[8,20,40]`); omni_capture docstring says Isaac Sim 5.1 (actual 6.0.1); frames_to_mp4 defaults 24 fps (grid is 60).

## Verified correct

- USD camera creation (Gf row-vector convention, basis rows) — independently verified by compose_scene selftest.
- COLMAP/LLFF conversion — selftests pass; images.bin round-trips to exact c2w (err ~1e-15).
- Intrinsics — rig.py fx=fy=(H/2)/tan(vfov/2) exactly matches USD aperture math; safe because converter always writes fx=fy.
- Unit/scale normalization — cameras, near/far, point cloud, GT points all scaled by same factor; `scene_scale.json` records it.
- Motion authoring math — pivot-about-centroid TRS, mm→stage-unit via metersPerUnit, transpose conventions all correct.
- Capture chunking sync — identical times array per chunk + purely keyframed animation → frame k aligns across cameras.
- Vendored convert fidelity — AST-level identical to reference except intentional drops.
- GT label provenance — point-cloud labels match instance-mask semantics; consistent subsampling indices everywhere.
- Grid comparability — identical base motions across cells via fresh `default_rng(seed)`; amplified part from separate stream.

## Top three actions

1. O2: set `seg_extract.n_times: 240` for grid runs, re-run seg (may explain near-zero grid ARIs).
2. O3: fix `peak_surface_mm` for the 0.2 compose scale.
3. O1: drop the `/255.0` in `omni_to_4dgs.py:283` and vendored `convert.py`.


---

## Omniverse/scene-gen fixes applied 2026-10-05

- **O1 (double color normalization)** — dropped the `/ 255.0` at `omni_to_4dgs.py`'s
  `write_ply(..., "points3D_multipleview.ply")` call site and identically in the vendored copy
  `orchestrator/pipeline/vendored/host/convert.py` (the two stay AST-identical apart from the
  intentional drops). Stored init colors are now 0–255 floats per the `fetchPly` loader
  contract (`core/scene/dataset_readers.py` divides by 255 itself). `run_grid_4dgs.py`'s
  `make_capture_variant` still writes 0–1 floats into the *capture* `points3D_gt.ply` — that's
  correct and untouched: `_read_ply_xyz_rgb` re-scales 0–1 sources to 0–255 on read.
- **O2 (grid seg undersampling)** — new preset `orchestrator/pipeline/config/presets/grid_seg.yaml`
  (`extends: pump01`, `seg_extract.n_times: 240`); `scene-gen/run_grid_seg.py`'s `rigid` impl
  (the only one that runs `seg_extract`) now uses it. `seg_extract.default` also gained a loud
  log warning when `n_times` is below the capture's actual frame count (counted from the
  `scene`/`capture` artifacts in the run manifest). Re-running seg on GPU with the new
  trajectories is still the owner's job.
- **O3 (GT amplitudes ignore compose scale)** — `gen_scenes.py`'s `generate_cell` now records
  `peak_surface_mm` (per-part) and `amplified_peak_mm` (manifest) in rendered-world millimetres
  (subject-internal × `SCALE` = 0.2) and writes `compose_scale` into each `*_motion.json`.
  Historical `*_motion.json`/`grid_manifest.json` files are 5× too large — noted in a comment
  at the write site. Scenes were NOT regenerated. `author_motion` itself is unchanged, so
  `--selftest` behavior is unaffected (verified: PASS).
- **O5 (stale metrics on failed runs)** — `run_grid_seg.py` only reads `seg_eval_result.json`
  when the run's `seg_eval.default` stage record is `success`/`skipped`; failed runs get empty
  metric cells.
- **O6 (lexicographic iteration sort)** — `run_grid_4dgs.py`'s `count_gaussians` sorts
  `iteration_*` by numeric suffix.

Verification: `ast.parse` on all edited files OK; `uv run --package pipeline pytest -q` from
`orchestrator/` — no new failures; `gen_scenes --selftest` PASS (with `usd-core`); `rig.py
--selftest` PASS; `segment_rigid --selftest` PASS (ARI=1.0000); `run_grid_4dgs.py` /
`run_grid_seg.py` import cleanly; `validate_config("grid_seg")` resolves with `n_times=240`.
