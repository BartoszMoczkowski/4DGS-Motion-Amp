# Implementation plan — segmentation rescue proposals as orchestrator stages

> Audience: the coding agent (Kimi Code) implementing the proposals in `docs/proposals/01–06`.
> This plan respects the locked orchestrator conventions (`orchestrator/planning/INSTRUCTIONS.md`,
> `ARCHITECTURE.md`): copy logic into `pipeline/vendored/`, config is the single source of truth,
> path translation only in `pipeline/paths.py`, light imports at module scope, one task at a time,
> every task ends with verification. Nothing here imports `motion-seg/motion_seg/`, `core/`, or
> `omniverse-pipeline/` at runtime — those remain reference-only.

## 0. Design mapping: proposal → orchestrator concept

The orchestrator already has the exact extensibility shape we need: **roles with config-selectable
impls** (`segment.rigid` / `segment.mbs` switched by `segment.impl` in a preset). Every proposal
becomes either a new impl under an existing role, or one new role.

| Proposal | Orchestrator shape | Env | New/changed files |
|---|---|---|---|
| 06 (denoise, calibrated z-scores, Leiden partition, subsample+propagate) | `segment.rigid2` impl (host) + `separability` diagnostic artifact | host | `vendored/host/trajectory_denoise.py`, `vendored/host/rigidity_graph2.py`, `stages/segment_rigid2.py` |
| 01 (motion-gated ROI) | **new role `roi`**, impl `roi.motion_gate` (host), output artifact `roi_mask` (npz) | host | `vendored/host/motion_gate.py`, `stages/roi_motion_gate.py` |
| 02 (multi-view mask lifting) | `roi.mask_lift` impl — needs rasterizer depth → runs in `cuda` container | cuda | `vendored/cuda/mask_lift.py`, `stages/roi_mask_lift.py` |
| 03 (seeded part-focused) | `segment.seeded` impl (host); same output contract as `segment.rigid` | host | `vendored/host/seeded_part.py`, `stages/segment_seeded.py` |
| 04 (subspace/spectral) | `segment.subspace` impl (host) | host | `vendored/host/subspace_seg.py`, `stages/segment_subspace.py` |
| 05 (Kabsch EM) | `segment.kabsch` impl (host) | host | `vendored/host/kabsch_em.py`, `stages/segment_kabsch.py` |
| eval extensions | extend `seg_eval.default` (host) — ROI precision/recall + ARI-within-machine | host | `vendored/host/seg_eval.py`, `stages/seg_eval.py` (edit in place) |

### 0.1 The `roi_mask` artifact contract

New artifact kind `npz`, name `roi_mask`, arrays:

- `roi_mask` (bool[N]) — True = inside machine ROI
- `snr` (float32[N]) — per-point gating statistic (motion_gate) or vote (mask_lift), for diagnostics
- `labels` convention downstream: points outside ROI get label `-2` ("static"), floaters stay `-1`

All `segment.*` impls gain an **optional** input `roi_mask`. Follow the T07 pattern for external
inputs: when a run's DAG includes an `roi.*` stage, the scheduler wires it by artifact name; when
it doesn't, segment stages run on the full set (today's behavior). If the DAG/registry currently
requires all declared inputs, do not weaken that — instead declare `roi_mask` on the segment
stages' `inputs` only in the roi-enabled impls' wiring, i.e. keep `inputs = ("trajectories",)`
and read `ctx.inputs.get("roi_mask")` defensively. **First implementation step of T18 is to check
how `run_dag` resolves inputs and pick the minimal mechanism; document the choice in the task log.**

### 0.2 Config schema additions (`pipeline/config/models.py`)

```python
class SegmentRigid2Config(StrictModel):   # proposal 06
    k: int = 12
    min_size: int = 15
    opacity_thresh: float = 0.1
    denoise: bool = True            # FFT band-pass at drive freq + harmonics
    drive_freq: float | None = None # None => auto-detect from mean spectrum
    harmonics: int = 3
    calibrate_sigma: bool = True    # per-edge noise floor from static points
    z_thresh: float = 3.0           # replaces threshold_mult when calibrated
    threshold_mult: float = 1.0     # fallback path (calibrate_sigma=False)
    partition: Literal["components", "spectral"] = "spectral"  # Leiden optional, see note
    n_subsample: int = 0            # 0 => full set; else FPS + kNN propagate

class RoiMotionGateConfig(StrictModel):   # proposal 01
    drive_freq: float | None = None
    harmonics: int = 3
    dilation_hops: int = 1
    readmit_mult: float = 3.0       # rigidity-lock readmission, in units of sigma_noise

class RoiMaskLiftConfig(StrictModel):     # proposal 02
    masks_dir: str = ""             # external input; validated for traversal (paths.py)
    ref_time: int = 0
    depth_tol: float = 0.02
    vote_thresh: float = 0.5
    dilation_hops: int = 1

class SegmentSeededConfig(StrictModel):   # proposal 03
    seed_gt_part: int | None = None # seed from gt_segmentation.npz (validation mode)
    seed_indices: list[int] = []    # direct Gaussian indices
    radius_hops: int = 6
    alpha: float = 0.15             # PageRank restart
    iterate_full: bool = False      # sequential extraction to full partition

class SegmentSubspaceConfig(StrictModel): # proposal 04
    n_components: int = 12          # PCA projection dim D
    local_neighbors: int = 16       # m for local 4-flat fits
    k_neighbors: int = 12           # graph sparsification
    position_weight: float = 0.1    # beta: append canonical xyz to break translation degeneracy
    n_clusters: int = 0             # 0 => eigengap

class SegmentKabschConfig(StrictModel):   # proposal 05
    n_clusters: int = 0             # 0 => BIC over candidate range
    k_range: list[int] = [2, 200]
    init: Literal["fft", "spectral", "kmeans"] = "fft"
    max_iter: int = 50
    spatial_prior: bool = True
    greedy_split: bool = True

class SegmentConfig(StrictModel):
    impl: Literal["rigid", "mbs", "rigid2", "seeded", "subspace", "kabsch"] = "rigid"
    rigid: SegmentRigidConfig = ...
    mbs: SegmentMbsConfig = ...
    rigid2: SegmentRigid2Config = ...
    seeded: SegmentSeededConfig = ...
    subspace: SegmentSubspaceConfig = ...
    kabsch: SegmentKabschConfig = ...

class RoiConfig(StrictModel):             # new top-level section, off by default
    impl: Literal["none", "motion_gate", "mask_lift"] = "none"
    motion_gate: RoiMotionGateConfig = ...
    mask_lift: RoiMaskLiftConfig = ...
```

- `impl: "none"` means the DAG contains no `roi` stage — current presets are unaffected.
- `_check_impl_ready` validator: `segment.impl == "seeded"` requires exactly one of
  `seed_gt_part` / `seed_indices`; `roi.impl == "mask_lift"` requires `masks_dir`.
- No new third-party deps for the core path (numpy/scipy only). `python-igraph`/`leidenalg` are
  **optional**: if import fails, `partition="spectral"` (scipy eigsh) is the fallback. Do not add
  them to `pyproject.toml` until a real run shows spectral is insufficient.

## 1. Task breakdown (append to `orchestrator/planning/TASKS.md`)

Phase 7 — segmentation rescue. Write each task file in the established format (status header,
goal, in/out of scope, deliverables, acceptance criteria, reference-only files, log).

| ID | Title | Depends on | Env | Status |
|----|-------|-----------|-----|--------|
| T18 | Trajectory denoising + calibrated rigidity + `segment.rigid2` (proposal 06) | T07 | host | todo |
| T19 | ROI role: `roi.motion_gate` + roi-aware segment stages + eval extensions (01) | T18 | host | todo |
| T20 | `segment.kabsch` EM (05) | T18 | host | todo |
| T21 | `segment.subspace` spectral (04) | T18 | host | todo |
| T22 | `roi.mask_lift` multi-view mask lifting (02) | T19, T09 | cuda | todo |
| T23 | `segment.seeded` part-focused (03) | T19 | host | todo |

```
T07 ── T18 ──┬─ T19 ──┬─ T22
             │        └─ T23
             ├─ T20
             └─ T21
T09 ─────────┴─ T22 (cuda container exec shape)
```

Order rationale: T18 carries the noise-floor fix every other method consumes (denoised
trajectories, calibrated σ) plus the separability diagnostic that tells us whether per-edge
methods can work per model. T20 before T21 because Kabsch EM is the most principled bet; T21 is
the cross-check and provides `init: spectral` for T20. T22/T23 need the ROI plumbing from T19.

### T18 — `segment.rigid2` (proposal 06)

**In scope**
- `vendored/host/trajectory_denoise.py`: `bandpass(traj, drive_freq=None, harmonics=3) ->
  (traj_denoised, freq_used)`. Auto-detect drive frequency as argmax of the mean power spectrum
  over the top-decile energy points. Port the FFT conventions from `core/render_amp.py` (read it
  for the frequency-bin conventions; copy the logic, don't import).
- `vendored/host/rigidity_graph2.py`: upgraded port of `vendored/host/rigidity_graph.py`:
  denoise → per-edge z-scores with σ calibrated from static points (median edge std over edges
  whose both endpoints have band-limited energy below gate) → affinity graph → spectral
  partition with eigengap K (fallback: connected components for `partition="components"`) →
  optional FPS subsample + q-NN propagation (port propagation from
  `vendored/cuda/mbs_infer.py`'s label-propagation helper, CPU-ized).
- `stages/segment_rigid2.py` registered as `segment.rigid2`, same I/O contract as
  `segment.rigid` (reads `trajectories.npz`, writes `segmentation.npz` `{points, labels}`) —
  `seg_eval` unchanged for it.
- **Separability diagnostic** (deliverable of its own): per-run JSON artifact
  `separability.json` + histogram PNG — AUROC of same-part vs different-part classification from
  z-scores, computed against `gt_segmentation` when available (external input, same pattern as
  `seg_eval`'s `gt` input). This is the go/no-go signal for all per-edge methods.
- Presets: `pump01_segB2.yaml` (`extends: pump01`, `segment.impl: rigid2`).
- Extend `scene-gen/run_grid_seg.py` with `--impl rigid2` (it drives orchestrator stages; check
  how it selects impls today and mirror it).

**Out of scope:** Leiden/igraph dependency (spectral fallback only); changing `segment.rigid`
(keep for regression).

**Verification (sandbox, no GPU):** synthetic 7-body self-test with injected white noise at the
measured ratio (σ = 0.35 × motion amplitude): old rigid path should degrade, rigid2 should
recover ARI ≥ 0.95. Add `orchestrator/tests/test_segment_rigid2.py` mirroring the existing
vertical-slice test's structure. Real-hardware checklist: re-run grid benchmark into
`runs/grid_seg_results.csv` (new rows, `--impl rigid2`), confirm separability AUROC per model.

### T19 — ROI role + `roi.motion_gate` (proposal 01)

**In scope**
- `RoiConfig` schema; `roi` role registered like `segment` (role.impl convention — check
  `registry.py`/`models.py` for how the `role.impl` name resolves and mirror it).
- `vendored/host/motion_gate.py`: band-limited energy + SNR gate (log-Otsu, port the log-space
  Otsu from `vendored/host/rigidity_graph.py`) → k-NN graph dilation → rigidity-lock
  readmission. Writes `roi_mask.npz` per §0.1.
- `stages/roi_motion_gate.py` (`roi.motion_gate`), host env, input `trajectories`, output
  `roi_mask`.
- Wire optional `roi_mask` consumption into `segment.rigid`/`rigid2` (points outside ROI →
  label −2) per the mechanism chosen in §0.1.
- Extend `vendored/host/seg_eval.py` + stage: when both `roi_mask` (pred) and `gt` exist, report
  ROI precision/recall and **ARI-within-GT-machine** alongside global ARI; add these fields to
  the eval artifact metadata so `run_grid_seg.py` picks them up into the CSV (check how it
  collects fields today).
- Preset `pump01_roi_gate.yaml`: `extends: pump01`, `roi.impl: motion_gate`,
  `segment.impl: rigid2`.

**Verification:** sandbox test with synthetic scene + planted static background cloud (ROI recall
= 1.0 on movers, precision ≥ 0.9 after dilation). Real: pump01 + grid rerun; inspect the
before/after edge-score histograms (artifact PNG).

### T20 — `segment.kabsch` EM (proposal 05)

**In scope**
- `vendored/host/kabsch_em.py`: responsibilities E-step, weighted per-frame Kabsch M-step (small
  per-body SVDs via `numpy.linalg.svd`, batched), FFT-fingerprint init (reuse T18's denoise
  module for the kept-band coefficients), BIC over `k_range`, greedy split with dip statistic,
  optional Potts graph prior as one α-expansion-lite pass (iterated conditional modes on the
  k-NN graph is acceptable; document the simplification).
- `stages/segment_kabsch.py` (`segment.kabsch`), host env, same artifact contract; optional
  `roi_mask` input; consumes `sigma` from calibration (share the calibration helper from T18 —
  put it in `vendored/host/noise_calib.py` if T18 didn't already).
- Preset `pump01_kabsch.yaml`. `--impl kabsch` in `run_grid_seg.py`.

**Out of scope:** GPU acceleration; non-rigid sub-clustering of high-residual bodies (note as
follow-up; the typed rigid/non-rigid partition is a thesis-worthy extension).

**Verification:** sandbox self-test ARI ≈ 1 including motion recovery (compare fitted per-frame
R(t) to ground truth rotations of the synthetic bodies). Real: grid benchmark; per-body residual
floor vs σ in artifact metadata.

### T21 — `segment.subspace` spectral (proposal 04)

**In scope**
- `vendored/host/subspace_seg.py`: PCA projection (scipy), local 4-flat fits on m-NN, subspace-
  angle affinity sparsified by canonical k-NN graph, normalized-Laplacian spectral embedding,
  eigengap K, k-means (small k-means implementation or sklearn-if-present; do not add a hard
  sklearn dependency — implement 50 lines of k-means++ instead).
- Optional `position_weight` feature append for translation degeneracy.
- Stage + preset + `--impl subspace` in `run_grid_seg.py`.

**Verification:** self-test exact recovery; real grid run; eigenvalue spectrum saved as a
diagnostic artifact (JSON) — used to argue whether 107 clusters is resolvable at all.

### T22 — `roi.mask_lift` (proposal 02)

**In scope**
- `vendored/cuda/mask_lift.py` with own argparse CLI (T09 cuda-stage shape: stage builds CLI,
  `ctx.containers.exec_in_container` runs it; **never import torch in the host process**). Port
  camera loading from `core/scene` conventions and depth rendering from
  `core/motion_amp/renderer.py`'s pre-rasterization path (reference only).
- Host-side mask production helper: `pipeline/vendored/host/clean_plate_diff.py` (pure
  numpy/OpenCV-free — use PIL/numpy already available; clean-plate differencing +
  morphological cleanup) so the whole flow is orchestratable; SAM-produced masks are an
  alternative `masks_dir` content, documented in the preset comment.
- Oracle mode: derive masks by projecting GT labels — measures the ceiling, validates the
  lifting machinery independent of mask quality.
- `stages/roi_mask_lift.py` (`roi.mask_lift`), cuda env; GT/oracle masks path runs on host
  (pure projection) — two impls `roi.mask_lift` (cuda) and `roi.mask_oracle` (host) is cleaner
  than env-switching inside one impl.

**Verification:** oracle-mask ceiling run on pump01 first (ARI-within-machine); then clean-plate
masks; compare. GPU-gated test per the `PIPELINE_TEST_GPU` convention.

### T23 — `segment.seeded` (proposal 03)

**In scope**
- `vendored/host/seeded_part.py`: geodesic ball (Dijkstra on k-NN graph), rigidity-weighted
  personalized PageRank (power iteration), Otsu on log π, ICM boundary refinement, optional
  sequential extraction to full partition.
- Seed sources: `seed_gt_part` (reads `gt_segmentation.npz` — external input artifact like
  `seg_eval`'s `gt`), `seed_indices`. Click-to-seed UI integration is **out of scope** (note as
  follow-up for the Layer 3 UI / amp UI).
- Stage + preset `pump01_seeded_oracle.yaml`.

**Verification:** oracle mode on pump01 — per-part IoU distribution over all 107 GT parts
(report median/p10 in artifact metadata). Interactive mode deferred.

## 2. Cross-cutting requirements (every task)

1. **Copy, don't call.** All logic lands in `pipeline/vendored/{host,cuda}/`. Reference scripts
   (`motion-seg/motion_seg/*`, `core/render_amp.py`, `core/motion_amp/renderer.py`) are read-only
   references; cite the ported function in the vendored module docstring, as existing vendored
   modules do.
2. **Config-only experiments.** New comparisons are presets under
   `pipeline/config/presets/` (`extends:` pump01 or the grid cell's preset), never new `.sh`
   files or edited `core/arguments/*.py`.
3. **No new hard dependencies.** numpy/scipy only in vendored host code. `torch`, `igraph`,
   `sklearn`, `leidenalg` must not appear at module scope anywhere in `pipeline/`; if an optional
   accel is used, import inside the function with a documented fallback.
4. **Artifact contract stability.** Every `segment.*` impl writes `segmentation.npz`
   `{points, labels}` with the same label conventions (−1 floater, −2 static/outside-ROI) so
   `seg_eval` and the UI stay backend-agnostic. Any new diagnostic goes in a separate artifact
   (`separability.json`, histogram PNGs) attached to the same stage record.
5. **Path handling.** `masks_dir`, `gt_segmentation`, and any new external inputs resolve
   through `pipeline/paths.py`; validate traversal per the security note in AGENTS.md §8.
6. **Tests.** Sandbox tests in `orchestrator/tests/` for every host stage (synthetic scenes, no
   GPU). GPU/Isaac tests gated behind the existing env flags. Update the task's Status header
   and `TASKS.md` board when done, with a Log section as in T10.
7. **Benchmark continuity.** All grid reruns append new rows to `runs/grid_seg_results.csv`
   via `scene-gen/run_grid_seg.py --impl <name>` — never overwrite existing rows; the CSV is the
   before/after evidence.
8. **Docs.** When a task lands, update `docs/motion-segmentation.md` results section and the
   DAG table in `ARCHITECTURE.md` (new `roi` role row), and drop a dated note in
   `.claude_notes/`.

## 3. Go/no-go decision points

- **After T18:** check `separability.json` AUROC per grid model. AUROC < 0.8 on a model means
  per-edge methods cap out there — skip straight to T20 (EM) for those models and note the
  reconstruction-quality ceiling in the thesis notes.
- **After T19+T20 on pump01:** if ARI-within-machine ≥ 0.5, proceed to grid-wide reruns; if not,
  run T21 (subspace) as the cross-check before investing in T22.
- **T22** is justified only if oracle masks show a large gap over the motion gate (i.e. the
  bottleneck is ROI quality, not clustering).
