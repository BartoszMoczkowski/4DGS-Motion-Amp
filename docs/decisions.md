# Decisions, invalidated conclusions, and open gaps

Distilled from the chronological working notes in `.claude_notes/`, the task board
(`orchestrator/planning/TASKS.md`), and the 2026-10-05/06 correctness reviews in `reviews/`.

**Precedence rule used throughout:** where the 2026-10-05/06 reviews contradict an older note,
the review wins. Where two notes conflict without a review to arbitrate, both are cited and the
conflict is stated explicitly.

**As-of date:** 2026-10-06.

---

## 1. Current state summary

### Works / validated

- **Full synthetic pipeline end-to-end on real hardware.** On 2026-07-19 the entire chain
  `prep_split → prep_motion → capture.isaac → convert → train → render → seg_extract →
  segment.rigid → amp` completed for real, producing an amplified video (milestones M1–M5 all
  reached; real-hardware caveat noted per task). Source: `NOTES_pipeline_orchestration.md`
  ("Ninth bug... actual milestone" entry), `TASKS.md` milestones section.
- **Motion amplification core is corrected and unit-verified.** All four `render_amp.py` methods
  are now textbook mean-anchored Eulerian displacement amplification (`out[t] = mean +
  a·filter(v[t] − mean)`), with output sanitization (opacity clamped to [0,1], scales ≥ 1e-8,
  quaternions re-normalized). Verified by 77/77 CPU checks on synthetic data, mirrored byte-for-byte
  into the vendored `orchestrator/pipeline/vendored/cuda/amp.py`. Source:
  `reviews/2026-10-05-motion-amp-correctness-review.md` ("core fixes applied").
- **amp-ui is trustworthy again.** In-place re-amplification, swapped channel labels, broken paths,
  unsynchronized GPU timing, and the `cameras.py` timebase problems are all fixed (2026-10-05).
  Source: `reviews/2026-10-05-motion-amp-correctness-review.md` ("amp-ui fixes applied").
- **Metrics are trustworthy going forward.** ARI implementation validated against sklearn over 600
  randomized labelings; `mean_iou` now means "mean over GT classes, unmatched = 0"; background
  exclusion is explicit (`--bg-label`); vendored orchestrator scoring re-synced to the same
  conventions on 2026-10-06. Source: `reviews/2026-10-05-motion-seg-review.md`.
- **Option-B segmentation (rigidity graph) is genuinely self-tested.** The new selftest Case B
  (adjacent parts + jitter, edges actually cut) passes at ARI 1.0; the vacuous old fixture is kept
  only as sanity Case A. Source: `reviews/2026-10-05-motion-seg-review.md` (M1 fix).
- **Kabsch EM is mathematically sound in sandbox.** After the 2026-10-06 fixes (no sigma
  annealing, proper Gaussian BIC, `init="spectral"`), it reaches ARI 0.9988–1.0 on the 7-body
  fixture, and BIC has a genuine interior minimum at the true K. Source:
  `reviews/orchestrator-correctness-review.md` (round 3), `NOTES_T20_kabsch_em_2026-08-11.md`
  (addendum).
- **Orchestrator sandbox suite is green:** 312 passed / 1 failed, the one failure being
  `test_gpu_status_over_real_http_with_valid_token`, which fails on any GPU-equipped machine by
  design. Source: `reviews/orchestrator-correctness-review.md` (round 3).
- **Container strategy settled and verified:** `cuda` image builds and trains for real;
  `capture.isaac` runs native Windows Isaac Sim (Vulkan is unsupported under WSL2/Docker —
  NVIDIA-confirmed platform limitation); stale images/containers are auto-detected via build-hash
  labels. Source: `NOTES_pipeline_orchestration.md` (T11 fixups, 2026-07-18 entries), `TASKS.md`.

### Known-broken, unmeasured, or provisional

- **Motion-only segmentation on real reconstructions is unresolved — again.** All measured grid
  ARIs are ≈ 0, but (a) trajectories were extracted at 4× undersampling relative to the 240-frame /
  40-cycle captures (aliasing past Nyquist, review bug O2), and (b) the Kabsch-EM BIC/annealing
  bugs invalidate the "107 parts are not resolvable" ceiling claim. Re-runs are pending before any
  thesis-level conclusion is safe. Sources: `reviews/2026-10-05-omniverse-pipeline-review.md` (O2),
  `NOTES_T20_kabsch_em_2026-08-11.md` (addendum).
- **Per-part (per-segment) motion amplification does not exist in `render_amp.py`.** `amp_factors`
  is per *parameter channel*; the "segmented" method variants are memory chunking, not segmentation
  masks. Per-part amplification currently exists only conceptually via the orchestrator seg
  pipeline. Source: `reviews/2026-10-05-motion-amp-correctness-review.md` (scope finding).
- **MBS (Option A) runs on real data but its published ARI ≈ 0 numbers are a measurement artifact**
  (only 4,000 of ~300k points labeled; the rest scored as one giant segment). Adapter fidelity to
  upstream MBS is verified, so genuine out-of-distribution quality is likely poor, but the true
  number is unmeasured. Source: `reviews/2026-10-05-motion-seg-review.md` (M4).
- **Pre-2026-10-05 benchmark artifacts are not comparable to new runs:** archived `results.csv`
  timings (no GPU sync), all `mean_iou` columns, all `ari_within_roi` columns, and every amp video
  rendered with the old `eulerian` semantics. See §3.
- **T22 real mask lifting was never run** (oracle ceiling only; no clean-plate/SAM masks exist;
  depth sign convention in `mask_lift.py` unverified). Source:
  `NOTES_T22_mask_lifting_2026-08-12.md`, `NOTES_T22_oracle_results_2026-08-12.md`.
- **T17 (cancel/concurrency hardening) is todo** with the design locked (stop the whole container).
  Source: `TASKS.md`, `NOTES_pipeline_orchestration.md` (T17 entry).
- **Endpoint/time-base mismatch (review bug O4) is confirmed but NOT fixed:** `add_motion.py`
  authors `u = fi/(N−1)` (duplicated endpoints), the loader assigns `t = i/N` — integer cycles
  land off DFT bins (spectral leakage) and test/seg times extrapolate past the last training time.
  Source: `reviews/2026-10-05-omniverse-pipeline-review.md` (O4; absent from its fixes-applied
  list).

---

## 2. Decision log

Chronological. Status: **active** / **superseded** / **invalidated** (see §3 for the invalidated
details).

| Date | Decision | Rationale | Status | Source |
|---|---|---|---|---|
| 2026-07-04 | Synthetic Omniverse data is the evaluation enabler: GT camera poses + per-part labels + controllable mm-scale motion | Real captures have no GT; quantitative ARI/IoU evaluation is otherwise impossible | active | `NOTES_omniverse_pipeline.md` §0 |
| 2026-07-04 | Segmentation scope: static per-clip labeling, position-only features, whole-clip window, rigid machine targets; end goal is per-part amplification so small seg errors are tolerable | Owner answers to open questions 1–7 | active | `NOTES_4dgs_motion_segmentation.md` §6 |
| 2026-07-04 | Method sequencing: prototype Option B (trajectory clustering) first; Option A (MBS MotNet with injected exact flow) and C (full retrain) held in reserve | B is cheap, exploits free correspondence, characterizes the data | active (B default; A later implemented via T10 and performs poorly — see §3.2; C never pursued) | `NOTES_4dgs_motion_segmentation.md` §4 |
| 2026-07-04 | Pump de-fused by weld + connected components → 107 parts; `frame_base` kept static | STL-from-CAD keeps each body a separate shell | active | `NOTES_omniverse_pipeline.md` §5b |
| 2026-07-04 | Motion model: one rigid sinusoid per part about its centroid; integer cycles per clip; mm-scale surface displacement | Seamless looping; big and small parts move comparably at the surface | active | `NOTES_omniverse_pipeline.md` §5c |
| 2026-07-05 | Dark patterned dome texture as capture background | Contrast + stable features for 4DGS | active | `NOTES_omniverse_pipeline.md` §5e |
| 2026-07-06 | GT poses written directly into `sparse_`/`poses_bounds`; COLMAP skipped (optional comparison only) | Omniverse poses are exact; COLMAP is lossy and failure-prone | active | `NOTES_omniverse_pipeline.md` §1, §3 |
| 2026-07-06 | Scenes normalized so nerf++ camera radius = 4.0 (`--target-radius`), with `scene_scale.json` recording the factor | 4DGS's hardcoded LRs assume this scale; raw cm-unit scenes explode the grid LR → NaN | active | `NOTES_omniverse_pipeline.md` §5g |
| 2026-07-05/06 | Real-capture (IP camera) path: keep Viseron NVR; wallclock-arrival timestamps + host/camera NTP sync; RTCP camera-clock timestamping rejected | Camera provably sends no RTCP Sender Reports; wallclock has jitter but no long-run drift | active (IP-camera path; USB path is `cameras.py` — see §3.18) | `NOTES_viseron_setup.md` |
| 2026-07-11 | Orchestrator = custom lightweight DAG + stage registry + config presets; three layers (execution / MCP / UI) | Fixes manual pipeline running, gives Claude GPU access, unifies scattered config | active | `NOTES_pipeline_orchestration.md` header |
| 2026-07-11 | Runtime host = WSL2 driving Docker Desktop | Locked decision | **superseded** 2026-07-14 by native Windows | `NOTES_pipeline_orchestration.md` header, "Runtime host moved off WSL2" |
| 2026-07-12 | Config is the single source of truth: one pydantic schema, `extends:` preset layering, `extra="forbid"` | Experiments are YAML, not new `.sh` files; typos fail in seconds | active | `NOTES_pipeline_orchestration.md` T02 |
| 2026-07-13 | Cross-run caching; cache key = config + input content hashes + git SHA + stage-source hash; fast fingerprint (size+mtime+first/last 1MiB) instead of full hashes | Full SHA-256 of multi-GB artifacts per manifest touch is too slow | active (hardened 2026-10-05: directory hashing, existence revalidation, output verification) | `NOTES_pipeline_orchestration.md` T03/T05; `reviews/orchestrator-correctness-review.md` |
| 2026-07-14 | "Wrap, don't rewrite" → **"copy the logic in, don't call the original script"** (vendoring policy) | Reference scripts are unversioned testing material; importing them is a live dependency on drifting code | active | `NOTES_pipeline_orchestration.md` (policy change) |
| 2026-07-14 | Runtime host = native Windows; WSL2 bundling deferred (T16) | Docker Desktop is reachable from Windows Python directly; WSL2 setup burden not worth it now | active | `NOTES_pipeline_orchestration.md` ("Runtime host moved off WSL2"), `TASKS.md` |
| 2026-07-15 | CUDA stages exec vendored CLI scripts as separate processes inside the container; bridge file serializes un-CLI-able hyperparams | torch/scene/arguments only exist in the container; dict/list hyperparams can't round-trip through argparse | active | `NOTES_pipeline_orchestration.md` T09 |
| 2026-07-16 | `capture.isaac` runs Isaac Sim as a **native Windows subprocess**, not in Docker; CPU-only prep stages stay in the `isaac` container | Vulkan unsupported under WSL2 (NVIDIA-confirmed); no Docker-side fix exists | active | `NOTES_pipeline_orchestration.md` (T11 fixups, Vulkan root-cause entry), `TASKS.md` |
| 2026-07-18 | Stale `cuda` image/container auto-detection via build-hash labels; Docker build moved to `/opt/build` | Three incidents of "reuse-by-design defeats a one-time fix"; `/workspace` is bind-mounted and shadows anything built there | active | `NOTES_pipeline_orchestration.md` (2026-07-18 entries) |
| 2026-07-19 | MCP server = streamable HTTP + bearer token (no default, fail fast), plain-ASGI auth middleware | `BaseHTTPMiddleware` buffers and breaks SSE streaming; a default token is one accident from an open endpoint | active | `NOTES_pipeline_orchestration.md` T13/T14 |
| 2026-07-19 | UI (Streamlit) imports Layer 1 in-process rather than going through MCP HTTP | UI runs on the same machine as the pipeline; no reason to cross a network+auth boundary | active | `NOTES_pipeline_orchestration.md` T15 |
| 2026-07-19 | Cancellation mechanism = stop the whole container (accept dropped warm state) | Bartosz tested container stop; fast enough. Finer per-exec cancel noted as later improvement | design locked, **code todo (T17)** | `TASKS.md`, `NOTES_pipeline_orchestration.md` (T17) |
| 2026-07-27 | Warehouse scene composed by pure USD references; orchestrator `prep_compose` stage deliberately deferred | Owner decision: capture can point at the composed scene manually | active | `NOTES_warehouse_scene.md` |
| 2026-08-05 | Segmentation PNGs stay raw 16-bit label IDs (black-looking); colorization rejected for GT use | Colorized PNGs lose exact IDs; display issue is cosmetic | active | `known_issues/black_segmentation_masks.md` |
| 2026-08-11 | Separability AUROC (0.8 bar) as the go/no-go diagnostic for per-edge methods; all 7 models below bar → proceed to Kabsch EM (T20) | The diagnostic correctly predicted per-edge failure; per-body EM aggregates evidence across all points of a body | active as framework; **the measured numbers are provisional** (see §3.11) | `NOTES_T18_rigid2_benchmark_2026-08-11.md` |
| 2026-08-11 | Kabsch EM with FFT-fingerprint init + BIC model selection adopted as the per-body method (proposal 05) | Per-body rigid fitting should be more robust to jitter than per-edge separability | **partially superseded**: spectral init added 2026-10-06 and recommended for reruns; BIC conclusion invalidated (§3.1) | `NOTES_T20_kabsch_em_2026-08-11.md` + addendum; `reviews/orchestrator-correctness-review.md` round 3 |
| 2026-08-12 | ROI motion gate (band-pass energy → log-Otsu → dilation → readmission) deployed as `roi.motion_gate` preprocessing | Hypothesis: static background poisons the graph | **result negative on real data** (gate keeps ~100% of points; jitter ≈ motion after band-pass); code active | `T19_roi_motion_gate_results.md` |
| 2026-08-12 | Oracle-ceiling decision: abandon clean-plate/SAM mask generation **for segmentation purposes**; `roi.mask_oracle` kept as a diagnostic ceiling | Perfect ROI does not rescue rigid2 (ARI-within-ROI ≤ 0.105) | **contested** — see §3.19; the measurement rides on provisional evidence | `NOTES_T22_oracle_results_2026-08-12.md` |
| 2026-10-05 | All four amp methods redefined as mean-anchored displacement amplification; outputs sanitized before rasterize | Old `eulerian*` amplified velocity, not displacement; `abs*` anchored to the last frame | active | `reviews/2026-10-05-motion-amp-correctness-review.md` |
| 2026-10-05/06 | Metric conventions: `mean_iou` = mean over GT classes (unmatched = 0); background exclusion only via explicit `bg_label` | Old conventions inflated IoU and silently excluded GT label 0 | active; historical numbers flagged (§3.9, §3.10) | `reviews/2026-10-05-motion-seg-review.md` |
| 2026-10-05 | Orchestrator security hardening: `run_id`/preset-name/external-artifact path validation; `stop_container` refuses unmanaged containers; `roi.impl: "none"` skipped in planning | Path traversal / arbitrary file read / clobber bugs found in review | active | `reviews/orchestrator-correctness-review.md` round 2 |
| 2026-10-06 | Kabsch EM: sigma annealing dropped (fixed calibrated σ), proper Gaussian BIC, `init="spectral"` added | Annealing start destroyed every initialization; old BIC was monotonic by construction | active | `reviews/orchestrator-correctness-review.md` round 3 |

---

## 3. Invalidated conclusions

The most important section. Each entry: the original claim, where it lives, why it's dead, and
what replaces it. **Do not cite any of the following from the notes without also citing the
invalidation.**

### 3.1 "BIC says 107 parts are not resolvable" (the reconstruction-quality ceiling)

- **Original claim:** BIC increases monotonically with K (61494 → 461206 for K = 20 → 150 on
  grid-A20mm_M2), so "the data itself says only ~20 motion groups are statistically justified" and
  "107 GT parts are beyond the reconstruction-quality ceiling."
- **Source:** `NOTES_T20_kabsch_em_2026-08-11.md` ("Key finding").
- **Why it's dead:** two implementation bugs, not a property of the data. Bug A: EM sigma annealing
  started at σ = 1.0, flattening all responsibilities to 1/K — a degenerate fixed point (GT init +
  σ=1.0 → ARI −0.003; data-scaled σ → ARI 0.9988). Bug B: the BIC formula omitted the Gaussian
  1/σ² likelihood factor, so BIC was monotonic in K *by construction*.
- **Replacement:** the resolvability question is **open again**. Re-measure with the fixed code
  (no annealing, proper Gaussian BIC with genuine interior minimum) and `init="spectral"`.
- **Evidence:** `NOTES_T20_kabsch_em_2026-08-11.md` (2026-10-06 addendum);
  `reviews/orchestrator-correctness-review.md` round 3.

### 3.2 "MBS (Option A) segments real data with ARI ≈ 0"

- **Original claim:** the `mbs ARI` column in the T18 table (≈ 0 on all 7 models) and the summary
  "Option A ... segments poorly on these scenes (ARI ≈ 0 ...)".
- **Source:** `NOTES_T18_rigid2_benchmark_2026-08-11.md` (results table); also propagated into
  `AGENTS.md` §9.
- **Why it's dead (as a number):** the MBS preset labels only 4,000 points; ≥ 98% of points on
  ~300k-Gaussian grid models got label −1 and were scored as one giant segment
  (`drop_floaters=False`). The published ARI is "near-uninformative about MotNet's clustering
  quality" (review M4).
- **What survives:** the MBS adapter was verified line-by-line faithful to upstream, and MotNet is
  genuinely out-of-distribution for mm-scale 4DGS trajectories — so quality is *likely* poor, but
  the true score is unmeasured.
- **Replacement:** re-score existing `segmentation_mbs.npz` artifacts with `--drop-floaters`
  (not yet done).
- **Evidence:** `reviews/2026-10-05-motion-seg-review.md` (M4 + "Verified correct").

### 3.3 Old `eulerian` / `eulerian_mod` amplified displacement

- **Original claim (implicit in every pre-2026-10-05 amp render):** `--method eulerian` with
  factor `a` amplifies displacement by `a`.
- **Why it's dead:** the implementation amplified the first difference (velocity): gain ≈
  `1 + (a−1)·ω`; at pump01's 60-frame one-period regime, `a=10` yields only ~2× displacement gain
  plus a phase shift. Additionally the rolled difference injected a wrap-around jump
  (`v[0]−v[T−1]`) that spread broadband energy no band-pass could remove. Identity at `a=1` passed,
  which is why it went unnoticed.
- **Replacement:** all four methods reimplemented as `out[t] = mean + a·filter(v[t] − mean)`;
  measured displacement gain at `a=5` is exactly 5.000.
- **Consequence:** every amp video/benchmark produced before 2026-10-05 embodies the old
  semantics and is not directly comparable to new output.
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bugs 1, 2; core fixes).

### 3.4 `abs` variants anchor to the "initial element"

- **Original claim:** code comments said the `eulerian_abs*` reference is the initial element.
- **Why it's dead:** `torch.narrow(values_tensor, -1, -1, 1)` takes the **last** frame; and
  `abs` with `a=1` full-band did not reproduce the input.
- **Replacement:** per-Gaussian temporal mean as reference for all four methods; `a=1` full-band
  is now an exact identity for all four.
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bug 13; core fixes).

### 3.5 `run_renders_auto.py` VRAM-mode timing comparison

- **Original claim (implicit):** the archived `results.csv` `time` column compares low-VRAM vs
  normal modes fairly.
- **Why it's dead:** no `torch.cuda.synchronize()` before stopping the timer; the two arms were
  not measuring the same thing.
- **Replacement:** sync added in both UI and harness; archived timing rows are not publishable
  without re-running.
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bug 4; amp-ui fixes).

### 3.6 ampUI interactive tuning results after the first click

- **Original claim (implicit):** each render click in the Streamlit UI shows the effect of the
  current parameters on the raw model.
- **Why it's dead:** amplify functions mutate the frame data in place; every click after the first
  re-amplified the previous output. All interactive comparisons after click 1 were corrupted.
- **Replacement:** `_clone_values()` deep-copy at render time; `self.values` retains raw data.
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bug 3; amp-ui fixes).

### 3.7 UI channel labels "rotation" / "scale"

- **Original claim:** the UI's channel list was `["pos3d","pos2d","rotation","scale",...]`.
- **Why it's dead:** `values_list` order is `[means3D, means2D, scales, rotations, ...]` — indices
  2 and 3 were swapped. A user amplifying "scale" was amplifying rotations (and hitting the
  un-normalized-quaternion bug).
- **Replacement:** labels corrected to match the actual order.
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bug 6; amp-ui fixes).

### 3.8 `segment_rigid --selftest` validates rigidity edge-cutting ("ARI ≈ 0.999")

- **Original claim:** the selftest "verifies on a synthetic 7-body scene" with "expected ARI ≈
  0.999" (also quoted as 0.9988 in §6b of the segmentation notes and in `AGENTS.md` §6).
- **Why it's dead:** the fixture's parts are spatially disjoint, so the k-NN graph is already
  disconnected — executed live by the reviewer: `n_kept_edges == n_edges` (17,280/17,280), every
  edge score at the clamp floor, so no edge scoring/thresholding/cutting is ever exercised. Worse,
  the same selftest produced a **degenerate ARI 0.0** in the T07 sandbox run where log-Otsu split
  inside the float64 noise band — so the notes contain mutually inconsistent "validation" outcomes
  (0.9988/0.999 claimed in §6b and `AGENTS.md`, 0.0 observed in the T07 entry) for a test that
  never exercised its discriminative core either way.
- **Replacement:** new primary Case B — adjacent parts (k-NN graph bridges boundaries, verified
  non-vacuous) + iid jitter; asserts edges were actually cut; bar ARI ≥ 0.99.
- **Evidence:** `reviews/2026-10-05-motion-seg-review.md` (M1);
  `NOTES_pipeline_orchestration.md` (T07 "real bug found, not fixed" entry);
  `NOTES_4dgs_motion_segmentation.md` §6b.

### 3.9 `mean_iou` values in the T18/T20 benchmark tables

- **Original claim:** `mean_iou` columns support cross-run/cross-method comparison (e.g. "rigid2
  IoU 0.20–0.23 vs rigid 0.12–0.26" in the T18 note).
- **Why it's dead:** `linear_sum_assignment` returns `min(n_gt, n_pred)` pairs and the mean covered
  only those — unmatched GT classes counted as nothing instead of 0. With n_pred varying from 2 to
  350 against 107 GT classes, the numbers are not comparable across rows. (The T18 note itself
  already flagged n_pred=2 mean IoU as "misleading"; the review shows the metric was wrong, not
  just misleading.)
- **Replacement:** `mean_iou` = mean over GT classes, unmatched = 0 (fixed in both the reference
  package and the vendored copy, 2026-10-05/06). **Numbers are not comparable with pre-2026-10-05
  values.**
- **Evidence:** `reviews/2026-10-05-motion-seg-review.md` (M2).

### 3.10 `ari_within_roi` columns (T19/T22 CSVs) via the implicit GT-0 heuristic

- **Original claim (implicit):** `ari_within_roi` is a neutral, well-defined metric.
- **Why it's dead as a convention:** it silently assumed "GT label 0 = background" — correct for
  pump01 (label 0 = static frame) and cubes scenes (label 0 = ground) **by coincidence**, but a
  scene where traversal order puts a moving part at label 0 would silently corrupt the column. The
  old implicit heuristic is gone from both the reference and (as of 2026-10-06) the vendored
  evaluator; exclusion now requires an explicit `bg_label`.
- **Replacement:** `--bg-label LABEL|auto`; the pump01 T19/T22 values happen to be computed with
  the correct exclusion, but should be cited as legacy-convention numbers.
- **Evidence:** `reviews/2026-10-05-motion-seg-review.md` (M3 + seg_eval sync applied 2026-10-06).

### 3.11 Grid ARI ≈ 0 as clean evidence that the methods fail

- **Original claim (implicit in T18/T19/T20/T22 interpretation sections):** near-zero ARIs on the
  7 grid/sweep models measure method quality on the true motion signal.
- **Why it's compromised:** grid scenes are 240 frames @ 60 fps with 40 motion cycles/clip, but
  `seg_extract.n_times` defaulted to 60 and nothing overrode it — sampling 40 cycles/unit-t at 60
  points aliases past Nyquist to ~20 cycles. Every frequency-calibrated stage (rigid2 FFT
  denoising, kabsch FFT-fingerprint init, trajectory denoising) operated on the wrong waveform. The
  review calls this a "plausible contributor to the near-zero grid ARIs."
- **Replacement:** preset `grid_seg.yaml` (`seg_extract.n_times: 240`) + a loud under-sampling
  warning in `seg_extract.default`; **GPU re-run is the owner's pending job**. All grid-based
  segmentation conclusions are provisional until then.
- **Evidence:** `reviews/2026-10-05-omniverse-pipeline-review.md` (O2 + fixes applied).

### 3.12 Grid ground-truth amplitudes (`peak_surface_mm`, cell names `A20mm` etc.)

- **Original claim (implicit):** `*_motion.json` amplitudes are physical rendered-world
  millimetres.
- **Why it's dead:** `gen_scenes.py` recorded subject-internal units but `compose_scene` applies
  SCALE = 0.2 — recorded amplitudes are **5× too large** in the rendered world (e.g. recorded
  22.98 mm ≈ 4.6 mm real).
- **Replacement:** `gen_scenes.py` now records rendered-world mm and writes `compose_scale`;
  **historical `*_motion.json`/`grid_manifest.json` were NOT regenerated** and remain 5× off.
- **Evidence:** `reviews/2026-10-05-omniverse-pipeline-review.md` (O3 + fixes applied).

### 3.13 Init point-cloud colors were correct

- **Original claim (implicit):** `omni_to_4dgs.py` writes GT colors for Gaussian init.
- **Why it's dead:** it wrote 0–1 floats, but the loader contract is 0–255 and divides by 255
  again — all Gaussians initialized effectively black (0.784 intended → 0.003 loaded). Low
  practical impact (training recovers), but the intended initialization was silently discarded in
  every scene converted before the fix.
- **Replacement:** `/255.0` dropped in both the reference and the vendored copy.
- **Evidence:** `reviews/2026-10-05-omniverse-pipeline-review.md` (O1 + fixes applied).

### 3.14 "Opacity reset at iteration 3000 caused the pump01 NaN"

- **Original claim:** `opacity_reset_interval = 3000` colliding with `coarse_iterations` was the
  NaN cause.
- **Why it's dead:** the NaN recurred at the identical iteration after the opacity fix. The real
  root cause was `cameras_extent` ≈ 4898 (raw cm units) multiplying every grid LR → explosive first
  fine-stage step. The note self-corrects this; recorded here because the wrong hypothesis appears
  first in the same section.
- **Replacement:** unit conversion + `--target-radius 4.0` normalization in `omni_to_4dgs.py`
  (decision log, 2026-07-06).
- **Evidence:** `NOTES_omniverse_pipeline.md` §5g.

### 3.15 "CUDA 12.4 toolkit vs torch-cu126 wheel mismatch broke the image build"

- **Original claim (hypothesis):** toolkit/wheel version mismatch caused `uv sync --frozen` to
  fail during the extension builds.
- **Why it's dead:** red herring. The real cause was `TORCH_CUDA_ARCH_LIST` unset under GPU-less
  `docker build` → empty arch list → `IndexError` in torch's arch-flag detection. Fixed with
  `ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"`.
- **Evidence:** `NOTES_pipeline_orchestration.md` (2026-07-18 "Root cause found" entry).

### 3.16 "Exit code 0 from a container stage means the stage worked"

- **Original claim (implicit in early T11 runs):** `capture.isaac` reporting `success` meant a
  capture existed.
- **Why it's dead — twice:** (a) Kit's auto-shutdown returned exit 0 after a fatal
  `WriterRegistryError`, and the bogus success poisoned the cross-run cache; (b) even after adding
  a `cameras_gt.json` check, that file is written from pure USD geometry — it exists when zero
  frames rendered. The same class ("trained but never checkpointed") hit `train.py` via the
  `save_iterations` ordering bug.
- **Replacement:** stage-level output verification (`cameras_gt.json` + camNN count for capture;
  `point_cloud/` existence for train) and, as of 2026-10-05, **scheduler-level** verification that
  every declared output exists before recording success/caching.
- **Evidence:** `NOTES_pipeline_orchestration.md` (T11 third/fourth bug entries; eighth bug
  entry); `reviews/orchestrator-correctness-review.md` (bug 1.8 + scheduler fixes).

### 3.17 "RTCP camera-clock timestamps are strictly better than wallclock stamping" (Viseron)

- **Original claim:** dropping `-use_wallclock_as_timestamps` would give camera-clock-based
  timestamps via RTCP Sender Reports.
- **Why it's dead:** a full 60 s debug capture showed the camera sends **zero** RTCP Sender
  Reports; without re-anchoring, RTP-delta timestamps accumulate clock-rate error uncorrected — the
  "clean recording" observed was worse for absolute timing, not better. The ONVIF and vendor-API
  leads were also chased and closed (no compliant stream; API not implemented on this firmware).
- **Replacement:** reverted to Viseron default wallclock stamping + host/camera NTP sync.
- **Evidence:** `NOTES_viseron_setup.md` (RTCP test + "RESOLVED" entries).

### 3.18 `cameras.py` fixed-20 fps writer as an adequate timebase (USB real-capture path)

- **Original claim (implicit):** the USB multi-camera recorder's output videos have a trustworthy
  frame rate and frame-aligned cameras.
- **Why it's dead:** frames were paced by `sleep(0.05)` + variable read latency, stamped at a fixed
  20 fps with no timestamps recorded — corrupting the frequency axis for FFT analysis undetectably;
  the sync-event pulse let busy workers miss frames, so frame k in camera1 ≠ frame k in camera2; an
  unverified resolution request could make OpenCV silently discard every frame.
- **Replacement:** per-camera sidecar timestamp CSVs (authoritative timebase), `threading.Barrier`
  sync, verified resolution, shared error flag. **Trustworthy now, but the end-to-end real-capture
  path through this recorder is untested.**
- **Evidence:** `reviews/2026-10-05-motion-amp-correctness-review.md` (bug 16; amp-ui fixes).

### 3.19 "The thesis conclusion stands: motion-only segmentation is reconstruction-quality-limited"

- **Original claim:** after T22's oracle ceiling, "the bottleneck is segmentation itself, not ROI
  quality ... the thesis conclusion stands" (Scenario C).
- **Source:** `NOTES_T22_oracle_results_2026-08-12.md` §3–4; anticipated in
  `NOTES_T20_kabsch_em_2026-08-11.md` "Next steps" item 3.
- **Why it is contested (not fully dead):** the conclusion rests on (a) rigid2/kabsch/AUROC
  numbers measured on **aliased trajectories** (§3.11), (b) a kabsch run whose model selection was
  broken (§3.1), and (c) legacy metric conventions (§3.9, §3.10). The oracle note itself carries
  the caveat that it measures the ceiling of *rigid2*-with-perfect-ROI only. The 2026-10-06 review
  round explicitly re-opens the resolvability question "before any 'reconstruction-quality ceiling'
  claim is made."
- **Replacement:** treat Scenario C as the current best hypothesis, not a settled conclusion;
  settle it after the n_times=240 re-run + fixed-kabsch re-measurement.
- **Evidence:** `NOTES_T22_oracle_results_2026-08-12.md` (its own caveat);
  `reviews/2026-10-05-omniverse-pipeline-review.md` (O2);
  `reviews/orchestrator-correctness-review.md` round 3 ("Thesis-relevant" note).

### 3.20 "`roi.impl: \"none\"` means the DAG contains no roi stage — current presets are unaffected"

- **Original claim:** `RoiConfig`'s docstring promised the default was a no-op.
- **Why it's dead:** `_auto_stage_plan` appended an unregistered `roi.none` stage, making
  `run_pipeline("base")` and every default preset raise `StageNotFoundError` — unrunnable through
  the API, MCP, and UI, with two red tests on master. T19/T22 added the roi impls without teaching
  the planner to drop "none", and the suite was not re-run.
- **Replacement:** fixed 2026-10-05 — planner skips any multi-impl role whose selector is "none".
- **Evidence:** `reviews/orchestrator-correctness-review.md` (bug 1.1 + round-2 fix).

---

## 4. Open questions / known gaps

### Segmentation (highest stakes for the thesis)

1. **Grid seg re-run at `n_times=240`** (fix for O2 is in place via `grid_seg.yaml`; the GPU
   re-run itself is pending). Until this lands, all grid ARI conclusions are provisional (§3.11).
2. **Kabsch EM re-measurement on real data** with fixed BIC, no annealing, and `init="spectral"`
   — decides whether the 107-part resolvability question reopens (§3.1). Note the FFT-fingerprint
   init still caps at ARI ≈ 0.91 on the all-rotations sandbox fixture (Gap C: fingerprints vary
   within parts); `init="spectral"` is the recommended seed.
3. **MBS re-scoring with `--drop-floaters`** on existing `segmentation_mbs.npz` artifacts (§3.2).
4. **T21 (subspace spectral) and T23 (seeded part-focused) remain todo** on the task board.
5. **Scenario C (reconstruction-quality-limited) is a hypothesis, not a conclusion** (§3.19) —
   settle after items 1–2.
6. **`mask_lift.py` depth sign convention unverified** on real hardware; degrades gracefully to
   frustum-only voting if wrong. Mask production (clean-plate/SAM) was never done and is
   deprioritized for segmentation per the oracle result — but would matter again if item 2
   revives segmentation hopes.
7. **Drive-frequency auto-detection suspicion** from T18 (all grid models report
   `drive_freq_used = 1`; true value likely 10 cycles/clip) — never resolved; raw-vs-denoised
   AUROC was nearly identical, so impact is probably small.

### Motion amplification

8. **No per-motion-segment amplification exists in `render_amp.py`** — `amp_factors` is
   per-channel; "segmented" variants are memory chunking. If per-part amplification is a thesis
   claim, this gap must be closed or explicitly scoped.
9. **Frequency bounds are fractions of the Nyquist of the frame count**, not real time —
   non-portable across scenes with different frame counts/fps; needs at least a sentence in the
   thesis docs.
10. **Empirical loose ends from the amp review:** timing-bias magnitude vs archived CSVs;
    `_record_memory_history` accumulation across UI repeats; `frozen_cam` asymmetry between
    harness and UI.

### Real capture

11. **`cameras.py` end-to-end path untested** — the recorder now has a trustworthy timebase
    (§3.18) but no real capture has been run through reconstruction/amp with it.
12. **Viseron/IP-camera path:** NTP sync plan decided (host chrony + per-camera web-UI NTP config,
    ~5–10 min poll), but the empirical frame-accuracy verification (LED flash / clap test against
    the ~16.7 ms budget) was never run.

### Pipeline / infra

13. **Endpoint duplication + time-base mismatch (O4) confirmed, unfixed:** authoring writes
    `u = fi/(N−1)` (endpoints duplicated), loader uses `t = i/N` — integer cycles land off DFT
    bins (spectral leakage into FFT-based stages) and seg/test times extrapolate past the last
    training time.
14. **T17 todo:** real cancellation (design locked: stop whole container), concurrent-job guard,
    typed video preview return.
15. **Isaac cache-permission fixup only runs on fresh container creation** — the "reuse defeats
    the fixup" gap was worked around manually (`docker rm -f pipeline-isaac`); the notes do not
    record the run-on-every-start code fix ever being applied. Status unconfirmed.
16. **Vendored/reference divergence is now intentional in places** (vendored `amp.py` documents
    its divergence from the "verbatim copy" policy; `metrics.py`/`seg_eval.py` re-synced
    2026-10-06; the Otsu plateau fix was mirrored back into the reference `rigidity_graph.py`) —
    there is no automated drift check between reference scripts and vendored copies.
17. **Unconfirmed suspicious items from the reviews** (evidence gathered, bugs not confirmed):
    converter accepts failed/partial captures; GT labels propagate onto background Gaussians
    (depressing plain-rigid grid ARI independent of segmentation quality); no post-hoc frame-sync
    verification for chunked capture; `run_grid_4dgs.py` CSV has duplicate run_ids
    (last-row-wins); doc drift in `gen_scenes`/`omni_capture`/`frames_to_mp4` defaults. Full lists:
    `reviews/2026-10-05-omniverse-pipeline-review.md` (S1–S8),
    `reviews/2026-10-05-motion-seg-review.md` ("Suspicious / unconfirmed"),
    `reviews/orchestrator-correctness-review.md` §2.
18. **`test_gpu_status_over_real_http_with_valid_token` fails on any GPU-equipped machine by
    design** — it encodes the sandbox, not the contract. The one standing red test.

### Scene authoring

19. **Warehouse scene:** pump orientation in the factory and exposure under the factory's 18
    sphere lights still need a visual check at first capture; the composed scene is references-only
    and breaks if `Q:` is unmounted (deliberate).
20. **Phase 8–11 streamlining tasks (T24–T31)** — amp CLI, declarative experiments + MLflow,
    metrics registry, MCP consolidation, docs alignment — are all todo; explicitly rejected options
    (W&B/cloud tracking, Hydra, pluggy discovery, self-hosted servers) should not be re-proposed.

---

## Source files

- `.claude_notes/NOTES_4dgs_motion_segmentation.md` — MBS↔4DGS analysis, options A/B/C, Option B
  implementation, first real pump01 run, MBS adapter notes.
- `.claude_notes/NOTES_omniverse_pipeline.md` — synthetic-data pipeline, pump de-fusing, motion
  authoring, scale-normalization NaN root cause, orbit camera fix.
- `.claude_notes/NOTES_pipeline_orchestration.md` — orchestrator design + T01–T17 implementation
  log, real-hardware debugging saga, policy changes.
- `.claude_notes/NOTES_T18_rigid2_benchmark_2026-08-11.md` — rigid2 grid benchmark + separability
  AUROC go/no-go.
- `.claude_notes/NOTES_T20_kabsch_em_2026-08-11.md` — Kabsch EM implementation + grid benchmark +
  2026-10-06 invalidation addendum.
- `.claude_notes/T19_roi_motion_gate_results.md` — ROI motion gate implementation + negative
  benchmark.
- `.claude_notes/NOTES_T22_mask_lifting_2026-08-12.md` — mask lifting implementation.
- `.claude_notes/NOTES_T22_oracle_results_2026-08-12.md` — oracle ceiling benchmark + Scenario C.
- `.claude_notes/NOTES_warehouse_scene.md` — warehouse scene composition.
- `.claude_notes/NOTES_viseron_setup.md` — Viseron NVR + camera time-sync investigation (IP-camera
  real-capture path; otherwise off-thesis).
- `.claude_notes/known_issues/black_segmentation_masks.md` — 16-bit label PNG display issue.
- `.claude_notes/NOTES_docs.md` — docs compilation meta-note.
- `orchestrator/planning/TASKS.md` — task board T01–T31, milestones M1–M5.
- `reviews/2026-10-05-motion-amp-correctness-review.md`,
  `reviews/2026-10-05-motion-seg-review.md`,
  `reviews/2026-10-05-omniverse-pipeline-review.md`,
  `reviews/orchestrator-correctness-review.md` — correctness reviews + applied fixes (precedence
  over older notes).
