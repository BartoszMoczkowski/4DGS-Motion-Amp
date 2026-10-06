# Correctness review — motion-seg package (2026-10-05)

Scope: `motion-seg/motion_seg/` (`extract_trajectories.py`, `segment_rigid.py`, `mbs_infer.py`, `evaluate_segmentation.py`, `rigidity_graph.py`, `metrics.py`, `run.sh`), cross-checked against the MultiBodySync submodule, GT generation (`omni_capture.py`/`omni_to_4dgs.py`), training-side time conventions, and orchestrator vendored consumers. The `--selftest` was executed and the ARI implementation numerically validated against sklearn.

## Confirmed bugs

### M1 — The `--selftest` never exercises the rigidity edge-cutting it claims to validate (Medium-High)
`segment_rigid.py:59-121` + `rigidity_graph.py:24-36, 141-144`. Executed live: `n_edges=17280, n_kept=17280`, every edge score hit the `scale*1e-12` clamp floor → all edges kept → the 7 segments come entirely from the kNN graph already being disconnected (synthetic parts are spatially disjoint: min inter-part gap ≈0.55 vs k=10 neighborhoods). Edge scoring, log-Otsu thresholding, and edge removal — the discriminative core — are untested. A build where Otsu splits inside the float64 noise band yields the documented degenerate ARI 0.0. Also: trajectories are noiseless (reconstruction jitter, the documented real-data failure mode, is absent); the opacity-filter/floater path is untested; pass bar `ari > 0.9` is weaker than the documented "ARI ≈ 0.999". What IS validated: kNN construction, connected components, min-size merge, label mapping, ARI call.

### M2 — `mean_iou` inflated when pred/GT cluster counts differ (Medium)
`metrics.py:69-75` — `linear_sum_assignment` returns exactly `min(n_classes, n_clusters)` pairs; the mean covers only those pairs — unmatched GT classes count as nothing instead of 0. Verified numerically: 4 balanced GT classes vs 1 predicted cluster → mean_iou 0.25 (should be 0.0625). Not comparable across runs with different `n_pred` — exactly how it's used in the T18/T20 tables (rigid2 n_pred 350 vs rigid 89 vs 107 GT).

### M3 — Silent unenforced "GT label 0 = background" assumption in auto-ROI (Medium)
`evaluate_segmentation.py:67-69` — whenever GT has label 0 and any positive label (always true for `omni_capture.py` output, which labels in USD traversal order with no background semantics), `ari_within_roi` silently excludes GT class 0. Currently correct by coincidence (pump01 class 0 = static frame; cubes scenes class 0 = ground plane). The orchestrator's vendored `seg_eval.py:63-80` replicates the heuristic and records it into result JSON → T19/T22 CSVs. A scene where traversal puts a moving part at label 0 silently corrupts that column.

### M4 — Published MBS ARI ≈ 0 numbers substantially measure an evaluation artifact (Medium)
`mbs_infer.py:229-230` + preset `pump01_segA.yaml` (`n_points: 4000`) + `seg_eval` default `drop_floaters=False`: only 4 000 points get labeled; ≥98% of points on ~300 k-Gaussian grid models get −1, scored as one giant segment. The `mbs ARI` column in the T18 table is near-uninformative about MotNet's clustering quality. Settling: re-score existing `segmentation_mbs.npz` artifacts with `--drop-floaters`.

## Suspicious / unconfirmed

- No alignment sanity check between pred and GT npz (`evaluate_segmentation.py:28-32`) — wrong-scene GT produces confident garbage; log median/p99 NN distance.
- NaN/Inf trajectories silently swallowed (`rigidity_graph.py:39-43` no NaN guard; NaN scores silently cut → orphans folded into arbitrary neighbors by `merge_small_components`).
- `merge_small_components` silently no-ops when no component reaches `min_size` (`rigidity_graph.py:106`).
- Floater label −1 counted as a predicted segment (`evaluate_segmentation.py:64`); `run.sh` evaluates without `--drop-floaters`.
- `mbs_infer.py:172-173`: `n_views > len(times)` → duplicate view pairs with zero flow, silently fed to MotNet.
- `run.sh` can't pass `--n-times` to the extractor (passthrough goes only to `segment_rigid`); scenes trained on ≠60 frames get 60 samples regardless.

## Verified correct

- ARI implementation (`metrics.py:10-44`) matches sklearn's `adjusted_rand_score` over 600 randomized labelings including degenerate cases.
- Trajectory extraction time convention: training `t = i/N` matches `np.linspace(0, 1, 60, endpoint=False)` exactly; Gaussian ordering preserved end-to-end; float32 adequate.
- Rigidity metric math (std of pairwise distance = 0 for rigid pairs), kNN restriction, log-Otsu, noise-floor clamp all sound.
- MBS adapter fidelity verified line-by-line against upstream `test.py:312-365` — flow convention, rescale/clamp, compose_dense, sync_motion_seg, feature_propagation all faithful. Option A's poor quality is genuinely model/data, not the adapter (modulo M4's measurement caveat).

## Thesis guidance

ARI itself is trustworthy; treat `mean_iou` cross-run comparisons (M2), any `ari_within_roi` column (M3), and the `mbs ARI` column (M4) as measurement artifacts until re-scored; don't cite `--selftest` as validation of rigidity edge-cutting without adding adjacent/overlapping parts and trajectory noise to the fixture.

---

## Fixes applied 2026-10-05

All fixes are confined to `motion-seg/motion_seg/` (+ `run.sh`). No old artifacts were
re-scored; `drop_floaters` defaults unchanged (M4 remains a measurement caveat — re-score
existing `segmentation_mbs.npz` with `--drop-floaters` as the review recommends).

### M2 — `metrics.py` (`best_iou_matching`)
`mean_iou` is now the **mean over GT classes, unmatched GT classes counted as IoU 0**
(previously: mean over only the `min(n_gt, n_pred)` Hungarian-matched pairs). Convention
documented in the docstring and printed at runtime when unmatched GT classes exist.
Verified numerically: 4 balanced GT classes vs 1 predicted cluster → 0.0625 (old: 0.25).
**Numbers are not comparable with pre-2026-10-05 `mean_iou` values** (incl. the T18/T20
tables).

### M3 — `evaluate_segmentation.py`
- New `--bg-label LABEL|auto` CLI arg (also a `bg_label=` parameter on `evaluate()`).
  Default `None` = **no exclusion, no `ari_within_roi` computed** (behavior change: the
  old implicit "exclude GT 0" heuristic is gone). `auto` reproduces the legacy heuristic
  as an explicit opt-in. Whenever a label is excluded, the CLI prints loudly which label
  and how many points; the excluded label is also recorded in the result dict
  (`bg_label_excluded`).
- **DIVERGENCE WARNING:** `orchestrator/pipeline/vendored/host/seg_eval.py:63-80` still
  contains the OLD implicit heuristic (silent GT-0 exclusion) and the OLD `mean_iou`
  convention (via its own copy of the matching code — verify before trusting its
  `mean_iou`/`ari_within_roi` columns). Owner must sync the vendored copy manually;
  orchestrator files were not touched (owned by another agent).

### M1 — `segment_rigid.py` selftest fixture
- Case A (legacy noiseless/disjoint scene) kept as a sanity check, bar raised to ARI > 0.99.
- New primary Case B: 6 parts placed **adjacent** (cube faces touching, spacing 0.30 with
  half-size 0.15) so the kNN graph bridges part boundaries (verified: full graph has 2
  connected components for 7 GT groups), plus iid per-point-per-timestep Gaussian jitter
  (sigma = 0.002, ~1.3% of part half-size, comparable to 4DGS reconstruction jitter).
  Parts move with distinct frequencies/phases/axes **and sinusoidal translation**
  (trans_amp 0.08) — rotation alone left some adjacent-part boundary edges unseparable.
- Assertions: (i) fixture not vacuous (full kNN graph bridges parts), (ii) edges were
  actually cut (`n_kept_edges < n_edges`), (iii) ARI >= 0.99. Bar rationale: fixture
  scores ARI 1.0 on seeds 0–4; cutting nothing yields ARI ~0.84, so the bar is
  unreachable without correct edge cutting. Documented in the `_selftest` docstring.
- `--threshold-mult` now also applies in `--selftest` mode so the test's ability to FAIL
  can be demonstrated (`--selftest --threshold-mult 1e9` → FAIL, exit 1).

### NaN/Inf guard — `rigidity_graph.py` (`segment_by_rigidity`)
Points with non-finite canonical position or trajectory are counted, reported
(`info["n_nonfinite_points"]`, plus stderr warning), and handled deterministically: all
their graph edges are cut up front (isolated nodes), their position is replaced by the
finite-cloud median for graph/merge purposes, and `merge_small_components` folds them
into the nearest big component. All-non-finite input raises `ValueError`. FP warning in
`edge_rigidity_score` suppressed (NaN propagation is intentional now).

### `merge_small_components` silent no-op — `rigidity_graph.py`
Now returns `(labels, merge_info)`; when small components exist but none reach
`min_size`, it prints a warning and sets `info["merge_skipped_no_big_component"]=True`
(previously silent). `info["n_small_merged"]` also reported.

### Floater label −1 reporting — `evaluate_segmentation.py`
`predicted segments` printout now excludes −1 (`n_pred_nonfloater`); when floater points
are present and `--drop-floaters` is off, a NOTE explains that −1 is still scored as one
ordinary cluster in ARI. Scoring behavior unchanged — this is honest reporting only.

### `mbs_infer.py` view guard
`--n-views > len(times)` now raises a `ValueError` explaining the duplicate-view/zero-flow
corruption; a second check catches duplicate timestep indices from argmin collapse
(`n_views` close to `len(times)`).

### `run.sh` extractor passthrough
Extractor options (e.g. `--n-times 120`, `--iteration`) can now be passed via the
`EXTRACT_ARGS` env var (`EXTRACT_ARGS="--n-times 120" ./run.sh pump01`). Positional args
after the scene name still go to `segment_rigid.py` only — backward compatible.

### Verification runs
- `uv run --package motion-seg python -m motion_seg.segment_rigid --selftest` →
  **PASS** (A: ARI=1.0000; B: parts bridged=True, 517 of 17 183 edges cut, ARI=1.0000).
- `--selftest --threshold-mult 1e9` → **FAIL, exit 1** (0 edges cut, ARI=0.8398, vacuity
  guard tripped) — proves the test exercises thresholding.
- `mean_iou` numeric check: 4 balanced GT classes vs 1 cluster → **0.0625** ✓.
- NaN guard: 2 poisoned trajectories → `n_nonfinite_points=2`, deterministic merge ✓.
- `merge_small_components` no-op flag and `--bg-label` variants (None / 0 / auto /
  nonexistent label) exercised via CLI ✓.
- `ast.parse` on all edited files; `bash -n` on `run.sh` ✓.

## seg_eval sync applied 2026-10-06

The orchestrator's vendored scoring copies were re-synced to the fixed reference
conventions of 2026-10-05 (they had diverged — still mean-over-matched-pairs IoU and the
implicit label-0 exclusion heuristic):

- `orchestrator/pipeline/vendored/host/metrics.py` — `best_iou_matching` now computes
  `mean_iou` as the mean over **GT classes**, unmatched GT classes counting as IoU 0
  (item M2), byte-for-byte with `motion-seg/motion_seg/metrics.py`.
- `orchestrator/pipeline/vendored/host/seg_eval.py` — `evaluate()` gains the explicit
  `bg_label` parameter (None default = no exclusion; int = exclude that GT label;
  `"auto"` = opt-in legacy label-0 heuristic) and records `bg_label_excluded`; adds
  honest floater reporting fields `n_pred_nonfloater` / `n_floater_points` (−1 still
  scored as a cluster in ARI; `n_pred` unchanged).
- `orchestrator/pipeline/stages/seg_eval.py` — passes `bg_label` through from config and
  writes the new fields into `seg_eval_result.json` (all pre-existing field names kept;
  only additions). Note: `ari_within_roi` / `n_roi_points` are now absent from the JSON
  unless an ROI mask or `bg_label` is configured (previously auto-computed whenever GT
  label 0 coexisted with positive labels).
- `orchestrator/pipeline/config/models.py` — `SegEvalConfig.bg_label:
  Optional[int | Literal["auto"]] = None`.
- New tests `orchestrator/tests/test_seg_eval_conventions.py` pin: 4 balanced GT
  classes vs 1 cluster → `mean_iou` = 0.0625; default → no exclusion; explicit/auto
  `bg_label` → exclusion recorded; floater counts.
- Full suite from `orchestrator/`: 305 passed / 8 failed (failures confined to
  `test_segment_kabsch.py`, `test_roi_motion_gate.py` — being fixed separately — and the
  pre-existing environmental `test_mcp_server.py` GPU-presence assertion).
