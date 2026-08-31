# HANDOFF — T22 Oracle Ceiling Benchmark & Next Steps

> Audience: kimi code (coding agent with zero conversation context)
> Time budget: ~60 minutes to run benchmark + interpret results
> Output: Report back to normal kimi for analysis

## 60-second context

This is Bartosz Moczkowski's thesis project on motion amplification for 4D Gaussian Splatting.
T18–T20 implemented upgraded segmentation methods; T19 added ROI motion gating. **T19's motion
gate did NOT improve ARI on real pump01 data** — reconstruction jitter after band-passing has
energy comparable to true mm-scale motion, so the gate kept ~100% of points.

**T22 (multi-view mask lifting, proposal 02) is the last remaining bet** before concluding the
thesis is reconstruction-quality-limited. The oracle benchmark measures the ceiling: what ARI
would we get if the ROI mask were perfect (derived from GT labels)?

### Why this matters
- If oracle ceiling is high but motion-gate was low → bottleneck is ROI quality → mask lifting is worth pursuing
- If oracle ceiling is also low → bottleneck is segmentation itself → conclude reconstruction-quality-limited

## File map (what was just implemented)

| File | Role |
|------|------|
| `orchestrator/pipeline/vendored/cuda/mask_lift.py` | CUDA depth-rendered mask lifting script |
| `orchestrator/pipeline/stages/roi_mask_lift.py` | `roi.mask_lift` stage (CUDA env) |
| `orchestrator/pipeline/stages/roi_mask_oracle.py` | `roi.mask_oracle` stage (host env) — **this is what you run now** |
| `orchestrator/pipeline/config/presets/pump01_mask_oracle.yaml` | Preset for oracle mode |
| `scene-gen/run_grid_seg.py` | Benchmark harness — already extended with `--impl mask_lift_oracle` |
| `orchestrator/tests/test_roi_mask_oracle.py` | Sandbox tests (4 tests) |
| `orchestrator/tests/test_mask_lift_helpers.py` | Sandbox tests for mask lift helpers (6 tests) |

## Step 1 — Run the oracle ceiling benchmark

### Prerequisites
- Docker Desktop running, `cuda` image built (`4dgs-motion-amp-cuda:latest`)
- Prior grid runs have `trajectories.npz` extracted (from baseline `--impl rigid` pass)
- `gt_segmentation.npz` exists for each run (from `convert.default`)

### Command

```bash
cd C:\Users\barte\Code\PythonScripts\4DGS-Motion-Amp
.venv\Scripts\python.exe scene-gen\run_grid_seg.py --impl mask_lift_oracle
```

**Idempotency**: If `roi.mask_oracle` already succeeded for a run, it will be skipped. To force
re-run, manually delete the `roi.mask_oracle` stage record from the run's manifest, or add a
new run_id.

**Timeout**: Each run is fast (host-side, no GPU for `roi.mask_oracle`, just `segment.rigid2` +
`seg_eval.default`). Expect ~5–10 minutes total for all 7 runs.

### What it does per run
1. Seeds `gt_segmentation` artifact into manifest (if not already there)
2. Runs `roi.mask_oracle` — NN-aligns GT labels to trajectory points, writes perfect ROI mask
3. Runs `segment.rigid2` — clusters only points inside the oracle ROI (outside → label -2)
4. Runs `seg_eval.default` — computes ARI, ARI-within-ROI, mean IoU vs GT
5. Appends one row to `runs/grid_seg_mask_lift_oracle_results.csv`

## Step 2 — Collect results

### Primary output file
```
runs/grid_seg_mask_lift_oracle_results.csv
```

Key columns:
- `ari` — global ARI (may be low if ROI excludes many points)
- `ari_within_roi` — ARI computed only on points inside the oracle ROI (this is the critical metric)
- `n_roi_points` — how many points the oracle marked as "machine"
- `status` — should be `success` for all runs
- `error` — empty if clean

### Baseline files to compare against
Collect these existing CSVs from prior runs (should already exist in `runs/`):
```
runs/grid_seg_rigid2_results.csv          (T18)
runs/grid_seg_rigid2_roi_results.csv      (T19 motion gate)
runs/grid_seg_kabsch_results.csv          (T20)
runs/grid_seg_results.csv                 (baseline rigid)
```

### Per-run artifacts (optional but useful)
For each run, check:
```
runs/<run_id>/roi_mask.npz                # the oracle ROI mask
runs/<run_id>/seg_eval_result.json        # detailed eval summary
runs/<run_id>/segmentation_colored_mask_lift_oracle.ply  # colored PLY
```

## Step 3 — Interpret results (decision rules)

Read all CSVs, compare `ari_within_roi` (or `ari` if `ari_within_roi` is missing) per run:

### Decision 1: Oracle vs motion gate
Compare `runs/grid_seg_mask_lift_oracle_results.csv` vs `runs/grid_seg_rigid2_roi_results.csv`:

| Scenario | Oracle ARI-within-ROI | Motion-gate ARI-within-ROI | Interpretation |
|----------|----------------------|---------------------------|----------------|
| A | High (>0.5) | Low (<0.2) | **Bottleneck is ROI quality.** Mask lifting is worth pursuing. Proceed to clean-plate/SAM masks. |
| B | High (>0.5) | Also high (>0.4) | Motion gate was not the limiting factor; something else improved between T19 and now, or the runs are not comparable. Check config differences. |
| C | Low (<0.3) | Low (<0.3) | **Bottleneck is segmentation itself**, not ROI. T22 mask lifting won't help. Thesis is reconstruction-quality-limited. |
| D | Oracle low, but global `ari` is higher than motion-gate | — | The oracle excluded too many points (over-restrictive ROI). Check `n_roi_points` vs expected. |

### Decision 2: Oracle vs no-ROI rigid2
Compare oracle vs `runs/grid_seg_rigid2_results.csv`:
- If oracle `ari_within_roi` >> rigid2 `ari` → restricting to the machine region helps clustering
- If oracle `ari_within_roi` ≈ rigid2 `ari` → the full-point-cloud clustering was already doing fine on the machine region; the problem is elsewhere

### Decision 3: Denoised vs raw trajectories
If oracle ARI is still low, check `runs/<run_id>/separability.json` (from the rigid2 pass):
- `auroc` < 0.8 → per-edge methods are fundamentally capped on this model
- `auroc` ≥ 0.8 → the clustering algorithm is the bottleneck, not the edge quality

## Step 4 — If oracle shows promise (Scenario A)

Generate masks and run the real mask lift:

### Option A: Clean-plate differencing
1. Render/capture the pump01 scene **without the machine** (or use frame 0 if machine is absent)
2. For each camera, compute `|I - I_plate| > threshold`, morphologically clean
3. Save as `masks/cam01/frame_00001.png` (or `masks/cam01.png` for static masks)
4. Run:
   ```bash
   .venv\Scripts\python.exe scene-gen\run_grid_seg.py --impl mask_lift --masks-dir <path_to_masks>
   ```

### Option B: SAM / SAM-2
Use Segment Anything on one frame per view with a point/box prompt on the machine.

### Option C: GT-projected masks (validation only)
Project GT-labeled Gaussians to 2D to get oracle 2D masks, then lift them back — this validates
the lifting machinery but doesn't test mask quality.

## Step 5 — Report findings

Write a report to `.claude_notes/NOTES_T22_oracle_results_YYYY-MM-DD.md` with:

1. **Run summary**: Which runs succeeded/failed, any errors
2. **Table**: Run ID × ARI (oracle) × ARI-within-ROI (oracle) × ARI (rigid2 baseline) × ARI (rigid2_roi) × ARI (kabsch)
3. **Interpretation**: Which decision scenario (A/B/C/D) applies
4. **Recommendation**: Proceed to mask generation, or conclude reconstruction-limited
5. **Any code fixes needed**: If `mask_lift.py` had depth-sign issues or other runtime bugs, document them

Also update `docs/motion-segmentation.md` §Results with the new numbers.

**At the end, output a concise summary back to normal kimi** (the user) with:
- The decision scenario (A/B/C/D)
- The key numbers (oracle ARI-within-ROI vs motion-gate ARI-within-ROI)
- The recommended next step

## Known issues to watch for

1. **Depth sign convention in `mask_lift.py`**: The depth comparison uses `abs(z_view - d_render) < depth_tol`. If the rasterizer depth sign is inverted, all Gaussians will be marked occluded. This degrades gracefully to frustum-only visibility (no depth test), but may reduce accuracy. If you see `visible=0` for all cameras in the `mask_lift` log, this is the cause. Fix: change to `abs(z_view + d_render) < depth_tol` or similar.

2. **Mask directory not found**: `mask_lift` requires `masks_dir`. The oracle does NOT need masks.

3. **Missing trajectories**: If the baseline `rigid` pass was never run, `trajectories.npz` won't exist and `mask_lift_oracle` will fail. Run `--impl rigid` first to extract trajectories.

4. **Scheduler input resolution**: The T19 fix allows `ctx.inputs.get("roi_mask")` to work even when `roi_mask` is not a declared input. This should be stable but if `seg_eval` fails with "roi_mask not found", the scheduler reverted.

## Standing rules (from AGENTS.md / INSTRUCTIONS.md)

- **Never overwrite existing CSVs**. `run_grid_seg.py` appends rows; if you need to re-run, back up the CSV first.
- **Config is the single source of truth**. Do not edit `.sh` scripts or `core/arguments/*.py`.
- **Copy logic in, don't call original scripts**. If you fix `mask_lift.py`, edit the vendored file directly.
- **One task at a time**. Finish the oracle benchmark before starting mask generation.
