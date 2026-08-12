# Handoff — run the `segment.rigid2` grid benchmark and analyze results

> Audience: the agent (Kimi 2.6) picking this up. Everything below is self-contained; no prior
> conversation context is assumed. Repo root: `C:\Users\barte\Code\PythonScripts\4DGS-Motion-Amp`.
> Read `AGENTS.md` first if you haven't — it is the standing project briefing.

## Context in 60 seconds

This repo does per-part motion amplification of 4D Gaussian Splatting. The motion-segmentation
step (cluster ~10⁵ Gaussians into ~107 rigid parts on synthetic pump scenes) is failing on real
trained models: baseline Option B (`segment.rigid`) scores ARI ≈ 0.002–0.009 vs 107 GT parts
(`runs/grid_seg_results.csv`), Option A MBS scores ≈ 0 (`runs/grid_seg_mbs_results.csv`). Root
cause: reconstruction jitter ≈ true mm-scale motion.

**T18 is implemented and sandbox-verified** (6/6 new tests, 210/210 sandbox suite): a new
`segment.rigid2` orchestrator impl — FFT band-pass denoising at the auto-detected drive
frequency, per-scene calibrated rigidity z-scores (noise floor from static points), adaptive
cut (`min(z_thresh, median + 6·MAD)`), connected-components partition (default) or guarded
spectral, FPS subsample + q-NN propagation, and a z-score **separability AUROC** diagnostic vs
GT. See `orchestrator/planning/tasks/T18-segment-rigid2-denoise-calibrate.md` (log of the three
design gaps found and fixed) and `docs/proposals/06-multiscale-snr-multiscale.md` (math).

## Your task

Run the rigid2 benchmark over the 7 already-trained grid/sweep models and analyze the outcome.

### Step 1 — run it

From the repo root:

```
.venv\Scripts\python.exe scene-gen\run_grid_seg.py --impl rigid2
```

- CPU-only: reuses each run's already-extracted `trajectories.npz`; no GPU/Docker needed.
  Expect a few minutes per model (7 runs: `grid-A20mm_M2`, `grid-A20mm_M4`, `grid-A40mm_M8`,
  `sweep-A40mm_M8-g{10000,25000,50000,100000}`).
- Idempotent: reruns skip runs whose manifest already has a successful `segment.rigid2`.
  Safe to re-invoke after a timeout or interruption.
- Preset used: `pump01_segB2` (`segment.impl: rigid2`, all `SegmentRigid2Config` defaults).
- Per run it also sets `segment.rigid2.gt_segmentation_path` so each run emits
  `runs/<run_id>/separability.json`.

### Step 2 — collect and compare

- New results: `runs/grid_seg_rigid2_results.csv` (ari, mean_iou, n_pred, timings per run).
- Baselines to compare against, same models: `runs/grid_seg_results.csv` (Option B) and
  `runs/grid_seg_mbs_results.csv` (Option A).
- Per-run diagnostics: `runs/<run_id>/separability.json` —
  `denoised_z.auroc` vs `raw_score.auroc`, plus `sigma_d` and `drive_freq_used`. Also
  `runs/<run_id>/segmentation_colored_rigid2.ply` for visual inspection.

### Step 3 — interpret (decision rules from `docs/proposals/IMPLEMENTATION_PLAN.md` §3)

- **`denoised_z.auroc` < 0.8 on a model** → per-edge methods cap out there; that model's
  segmentation needs the Kabsch EM (T20, `docs/proposals/05-iterative-kabsch-em.md`), not more
  threshold tuning. Note the reconstruction-quality ceiling in the findings.
- **`denoised_z.auroc` ≥ 0.8 but ARI still low** → the partition/thresholding is the
  bottleneck: try the guarded spectral partition (`partition: "spectral"` in a preset variant)
  or check `n_pred` vs 107 (over/under-fragmentation) before touching anything else.
- **`raw_score.auroc` ≈ `denoised_z.auroc`** → denoising isn't helping on that model; check
  `drive_freq_used` in the stage log/metadata — auto-detection may have picked the wrong bin
  (set `drive_freq` explicitly in a preset variant; the grid cells' drive frequency is known
  from `scene-gen/gen_scenes.py`).
- Watch for the `static_mask` degenerate case: if a model is so noisy that log-Otsu on
  trajectory energy finds no static points, `sigma_d` falls back to the all-edge median —
  visible as an unusually large `sigma_d` in `separability.json`.

### Step 4 — report

Append a dated findings entry to `docs/motion-segmentation.md` (results section) and
`.claude_notes/`, including: the rigid2 vs rigid vs mbs ARI/IoU table, per-model AUROC, which
decision-rule branch each model falls into, and the recommendation (proceed to T19 ROI gating /
T20 Kabsch EM per `docs/proposals/IMPLEMENTATION_PLAN.md`). Do **not** overwrite or edit
existing rows in any `runs/*.csv` — they are the before/after evidence.

## Useful handles if something breaks

- Sandbox tests: `cd orchestrator && uv run --package pipeline --with pytest python -m pytest -q tests/test_segment_rigid2.py`
- Config schema: `orchestrator/pipeline/config/models.py` (`SegmentRigid2Config`)
- Stage: `orchestrator/pipeline/stages/segment_rigid2.py`; core logic:
  `orchestrator/pipeline/vendored/host/rigidity_graph2.py` + `trajectory_denoise.py`
- Stage logs per run: `runs/<run_id>/logs/segment.rigid2.log`
- Tuning knobs (preset YAML): `k`, `min_size`, `harmonics`, `drive_freq`, `z_thresh`,
  `partition`, `max_clusters`, `n_subsample` — create a new preset under
  `orchestrator/pipeline/config/presets/` (extends `pump01_segB2`), never edit shared presets
  in place for one experiment, and add a matching `--impl`/preset entry in
  `scene-gen/run_grid_seg.py` only if you need a separate results CSV; otherwise edit the
  `PRESET` mapping's target.

## Standing constraints (from AGENTS.md / orchestrator conventions)

- Logic lives in `orchestrator/pipeline/vendored/` — never import or shell out to
  `motion-seg/`, `core/`, or `omniverse-pipeline/` reference scripts.
- New experiments = new YAML presets; path translation only in `pipeline/paths.py`; numpy/scipy
  only in vendored host code (no torch/sklearn/igraph imports at module scope).
- Every change ends with a sandbox test run; GPU tests stay behind their env flags.
