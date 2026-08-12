# T18 — Trajectory denoising + calibrated rigidity + `segment.rigid2`

- Status: done (2026-08-09) — implemented + sandbox-verified; real GPU/grid run pending Bartosz's machine
- Phase: 7 (segmentation rescue — `docs/proposals/IMPLEMENTATION_PLAN.md`)
- Depends on: T07
- Environment: host (pure numpy/scipy)

## Goal

Add the proposal-06 upgraded Option-B segmentation as a third impl under the `segment` role:
FFT band-pass denoising at the (auto-detected) drive frequency + harmonics, per-scene
calibrated rigidity z-scores (noise floor from static points), and a hard-gated spectral
graph partition — plus the z-score **separability diagnostic** (AUROC vs GT) that is the
go/no-go signal for per-edge methods per model (IMPLEMENTATION_PLAN §3). Attacks the measured
root blocker: reconstruction jitter ≈ true mm-scale motion (`docs/motion-segmentation.md`;
baseline ARI ≈ 0.002–0.009 in `runs/grid_seg_results.csv`).

## In scope

- `pipeline/vendored/host/trajectory_denoise.py` — `bandpass` (rfft over time, keep DC +
  drive-harmonic bins), `detect_drive_freq` (argmax of moving points' mean power spectrum),
  `trajectory_energy`, `motion_fingerprint` (kept for T20's EM init).
- `pipeline/vendored/host/rigidity_graph2.py` — `static_mask` (log-Otsu on trajectory energy,
  reusing the T07 log-Otsu), `calibrate_sigma_d` (median edge-score over static-static edges,
  1e-12 floor for noiseless synthetic data), `edge_zscores` (denoised score / σ_d),
  `adaptive_z_cut` (`min(z_thresh, median + 6·MAD)` — see Log below), `spectral_partition`
  (normalized-Laplacian eigsh + eigengap K + dependency-free k-means++, hard gate at `z_cut`),
  `components_partition` (baseline shape, calibrated cut), `fps_subsample` +
  `propagate_labels` (CPU q-NN majority vote), `separability_auroc` (Mann-Whitney ranks,
  no sklearn), `segment_by_rigidity2` (full pipeline incl. optional subsample recursion).
- `pipeline/vendored/host/segment_rigid2.py` — opacity-filter/floater wrapper mirroring
  `segment_rigid.py`'s structure; identical label conventions (−1 floater).
- `pipeline/stages/segment_rigid2.py` — `segment.rigid2`, host env, same I/O contract as
  `segment.rigid` (reads `trajectories.npz`, writes `segmentation.npz` `{points, labels}` —
  `seg_eval` untouched). Separability diagnostic via config `gt_segmentation_path` (per-run,
  like `seg_eval.recolored_ply`) → extra `separability` JSON artifact; GT is *not* a declared
  DAG input so GT-less inference runs still work (the §0.1 mechanism choice from the plan).
- Config: `SegmentRigid2Config` + `SegmentConfig.impl` Literal extended with `"rigid2"`.
- Preset `pump01_segB2.yaml` (`extends: pump01`, `segment.impl: rigid2`).
- `scene-gen/run_grid_seg.py --impl rigid2` → `runs/grid_seg_rigid2_results.csv`; reuses
  extracted trajectories (like `mbs`), sets `gt_segmentation_path` per run, preserves the
  previous `seg_eval_result.json` as `seg_eval_result_before_rigid2.json`.

## Out of scope

Leiden/igraph (spectral fallback only); changing `segment.rigid` (regression baseline);
ROI gating (T19), which further shrinks the graph the partition sees.

## Verification (sandbox, no GPU)

`orchestrator/tests/test_segment_rigid2.py` — T18-specific fixture (T07's scene has parts 0.9
apart → no cross-boundary k-NN edges, so the z-separability diagnostic would have nothing to
classify; and per-part free-running frequencies would be partially filtered by the single-f0
band-pass): 6 *adjacent* rigid parts sharing one drive frequency with different
axes/amplitudes/phases + a static base, white jitter at σ ≈ 0.3× motion amplitude (the real
models' measured failure ratio). Asserts: drive-freq auto-detection finds the true bin,
denoised z-score AUROC > 0.8 (go/no-go bar), full-pipeline ARI ≥ 0.95, components fallback,
subsample+propagate, and the stage-level DAG run (`segment.rigid2` + `seg_eval.default`,
eval ARI ≥ 0.95, `separability.json` artifact).

## Log (2026-08-09)

- First test run exposed a real design gap, fixed before landing: with all z-scores small in
  absolute terms (clean scene, calibrated σ), the soft `exp(-z²/2)` affinity never approaches
  zero — the graph stays fully connected, the eigengap is mushy, and spectral clustering
  fragments the uniform background cloud (ARI 0.11–0.27 on the fixture at any forced K).
  Fix: `adaptive_z_cut` hard-gates affinity at `min(z_thresh, median + 6·MAD)` before the
  spectral step (and is the cut in the components fallback). On the fixture this cuts exactly
  the boundary edges. The MAD rule scales with each scene's own same-part spread, so it
  complements rather than replaces the calibrated significance level on noisy real data.
- Second gap: `eigsh(which="SA")` on the normalized Laplacian mis-resolves the near-zero
  eigenvalue cluster (Lanczos misses degenerate multiplicities — dense solve confirmed 7×1.0
  where sparse found 2), producing garbage embeddings (ARI 0.08). Fixed twice over: (a) solve
  the equivalent, better-conditioned largest-eigenpairs-of-normalized-adjacency problem, and
  (b) restructure to a **hybrid** — connected components of the gated graph are coarse
  clusters for free; spectral sub-splitting runs only *within* components (where the top
  eigenvalue is simple and eigsh is reliable) and a sub-split is accepted only if its cut
  edges' mean z exceeds `split_z_ratio` (1.5) × the component mean (the "don't partition a
  uniform cloud" guard — geometric bottlenecks inside one rigid part have perfectly rigid
  z's). Final: both `partition="components"` (the default) and the guarded `"spectral"` reach
  ARI 1.0 on the noisy fixture; full sandbox suite 210 passed.
- Lesson recorded for T19+ (already in proposal 01's motivation): a large uniform static
  background is actively harmful to graph partitions — another reason the ROI gate matters.
