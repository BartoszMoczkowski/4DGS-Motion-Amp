# Orchestrator correctness review (pipeline/ + mcp_server/ + ui/)

Scope: deep correctness review of `orchestrator/` against `planning/ARCHITECTURE.md`,
`planning/INSTRUCTIONS.md`, and `planning/TASKS.md`. Verified by reading code and by executing
`run_pipeline("base")` / `_auto_stage_plan` plus the sandbox test suite on this machine
(250 passed, 12 failed, 9 skipped).

## 1. Confirmed bugs

### 1.1 CRITICAL — `roi.impl: "none"` breaks every default preset's `run_pipeline`
`pipeline/api.py:31-63` (`_auto_stage_plan`) appends `roi.none` to the stage plan whenever the
`roi` role's selector is `"none"` — the default (`pipeline/config/models.py:357`). No `roi.none`
stage is registered, so `pipeline.dag.graph.resolve_nodes` raises
`StageNotFoundError: no stage registered as 'roi.none'`.
`RoiConfig`'s own docstring (models.py:354) promises "`impl: "none"` means the DAG contains no
`roi` stage — current presets are unaffected"; the code does the opposite.
Verified live: `api.run_pipeline("base")` raises `StageNotFoundError`. `base.yaml`, `pump01.yaml`,
and every preset not explicitly setting `roi.impl` are unrunnable through the public API, the MCP
`run_pipeline` tool, and the UI.
This also breaks two existing tests on master:
`tests/test_stages_isaac.py::test_api_run_pipeline_seeds_external_artifacts_before_run_dag` and
`tests/test_mcp_tools.py::test_run_pipeline_returns_run_id_immediately` (the latter expects a
`MissingDependencyError` for `raw_mesh`; it now gets `StageNotFoundError` first). T19/T22 added the
`roi.*` impls without teaching `_auto_stage_plan` to drop `"none"`, and the suite was apparently
not re-run.

### 1.2 HIGH — CUDA image staleness check silently disabled on real daemons
`pipeline/containers/manager.py:284-293` (`_cuda_image_up_to_date`): when the stored build-hash
label does NOT match the current on-disk hash, the function returns `True` (reuse stale image)
unless `PIPELINE_REBUILD_CUDA_IMAGE=1` is set — on any real Docker client. This directly
contradicts `ensure_image`'s docstring ("a changed Dockerfile/pyproject.toml/uv.lock change since
the last build triggers an automatic rebuild instead of silently reusing a stale image",
manager.py:169-182) and reopens the exact bug `_cuda_build_hash` was added to close
(manager.py:55-77, T11's broken-venv incident). Introduced in commit 4617542 ("fix(capture)..."),
undocumented in `.claude_notes`. Bonus smell: manager.py:285 branches on
`type(self.client).__name__.startswith("_Fake")` — production behavior keyed to a test double's
class name.

### 1.3 HIGH — Forced upstream rerun does not invalidate downstream stages for directory artifacts
`pipeline/dag/scheduler.py:207`: input hashes are `art.content_hash or ""`. Directory artifacts
(`capture`, `model`, `scene`, `renders` — kinds `dataset`/`model`) are never hashed
(`hash_path` raises on non-files; scheduler.py:280-285 only hashes `p.is_file()`), so their
contribution to every downstream cache key is a constant `""`. Force-rerunning
`capture.isaac` (or `train.default`) via `run_stage(..., force=True)` and then resuming the run
leaves `convert`/`render`/`seg_extract` cache keys unchanged — they are skipped and silently
consume/regurgitate stale results. The `""` placeholder is documented as deliberate in
`cache.py:66-87`, but the consequence for force/rerun workflows is not handled anywhere.

### 1.4 HIGH — Cross-run cache skip never revalidates that artifacts still exist
`pipeline/dag/scheduler.py:213-223`: on a `get_cached` hit the stage is recorded `skipped` with
the cached artifact records verbatim — no existence check, no re-hash. If the producing run's
directory was deleted or the file modified externally, the new run references dead/stale paths;
downstream stages then use the stale recorded `content_hash` as well, so the corruption propagates
without any stage ever executing. `cache.py:110-118` deliberately treats a corrupt index as cold,
but a *valid* index pointing at deleted files is not covered. Same hole in the same-run path:
`_already_recorded` (scheduler.py:119-129) trusts the manifest record, never the filesystem.

### 1.5 MEDIUM-HIGH — MCP: no confinement on artifact paths — arbitrary file read
`mcp_server/server.py:152-183` accepts `external_artifacts: dict[str, Artifact]` with an
unconstrained `path`; `pipeline/api.py:174-178` seeds them into the manifest unchecked.
`artifact_resource` (server.py:339-356) then serves `Path(artifact.path).read_bytes()`, and
`get_preview` (server.py:286-313) serves `Image(path=...)` for any artifact in any run's manifest.
An authenticated client can seed an artifact with `path="C:/Users/barte/<anything>"` and read it
back. Host stages compound it: `seg_eval` (stages/seg_eval.py:52-53) `np.load`s the seeded
`gt_segmentation` path natively (no `to_container` root check to catch it, accidentally or
otherwise). The project instructions explicitly say external artifact paths must not be passed in
without validation; nothing validates them.

### 1.6 MEDIUM-HIGH — MCP: `run_id` and preset name are path-traversal-prone
No validation of `run_id` anywhere: `pipeline/artifacts/paths.py:37-63` interpolates it straight
into `runs_root / run_id`. MCP tools `get_run_status`/`tail_logs`/`read_artifact`/`cancel_run`/
`run_stage` take client-supplied `run_id` (server.py; the `log`/`manifest`/`artifact` resources
too). `run_id="../../x"` reads `.../x/manifest.json` / `.../x/logs/<stage>.log` outside the runs
root. On the write side, `pipeline/api.py:121` (`new_run_id`) embeds the raw preset name into the
run id and `create_run` (manifest.py:91-107) `mkdir(parents=True)`s it — a preset name containing
`../` (also accepted by `resolve_preset` -> `_preset_path`, config/resolver.py:36-43, which will
happily load `../../foo.yaml`) turns `run_pipeline` into an out-of-tree directory write. The
bearer token limits this to authenticated clients, but the design claims "whitelisted ops only".

### 1.7 MEDIUM — `stop_container` can stop ANY container on the host
`pipeline/containers/manager.py:465-473` (`stop_by_id`): `containers.get(container_id)` +
`stop()` with no `pipeline.managed` label check. `list_containers` filters by label, but the stop
path doesn't — the MCP `stop_container` tool (server.py:105-111) can kill the unrelated Viseron
NVR container or anything else Docker Desktop runs. Trivially fixed by checking labels after
`get()`.

### 1.8 MEDIUM — Stages can report success without their declared output existing
Only `train.default` (stages/train.py:93-100) and `capture.isaac` (stages/capture_isaac.py:144-162)
verify outputs after a 0 exit. `amp.default` (stages/amp.py:103-115), `render.default`
(stages/render.py:61-68), `segment.mbs` (stages/segment_mbs.py:103-110), `prep_split.default`
(stages/prep_split.py:71-78), `seg_extract`/`prep_motion` return artifacts unconditionally; the
scheduler's hash step (scheduler.py:280-285) silently skips non-existent files (`if p.is_file()`),
so a script that exits 0 without writing produces a `success` record AND a cross-run cache entry
(`put_cached`, scheduler.py:299) that poisons every future run with the same key — the exact
failure class the train/capture checks were added for (their comments say so). The scheduler, not
each stage, is the right place for a "declared outputs must exist" check.

### 1.9 MEDIUM — OOM-fallback results are cached under the non-fallback key
`run_with_oom_retry` (resources/oom_retry.py:95-134) swaps `ctx.config` to the reduced-memory
variant on retry, but the scheduler records and caches the result under the cache key computed
from the ORIGINAL config (scheduler.py:208 -> 287-299). A later identical run skips the stage and
reuses artifacts produced with degraded settings (halved `n_points`, `rt_subframes`, ...) without
ever OOMing or logging why. `StageRecord.oom_fallback` at least records it in the manifest.

### 1.10 LOW-MEDIUM — `run_pipeline(run_id=<existing>)` silently clobbers the old run
`pipeline/api.py:173` calls `create_run` unconditionally; `create_run` (manifest.py:91-107)
overwrites `manifest.json`/`config_snapshot.json` with an all-pending manifest. Any caller passing
an existing `run_id` (the parameter is public) destroys that run's recorded history before
`run_dag`'s resume logic ever sees it. `run_dag` itself handles existing manifests carefully —
`run_pipeline` bypasses that.

## 2. Suspicious spots (evidence, not confirmed user-visible bugs)

- **Resume-after-failure with stale partial files.** Stages write into per-run dirs
  (`ctx.run_dir/capture`, `train_out`, ...) with `mkdir(exist_ok=True)` and never clean them. On a
  rerun after failure, `capture.isaac`'s cam-dir count check (capture_isaac.py:146) and
  `train.default`'s `point_cloud` existence check (train.py:93-94) can be satisfied by stale
  leftovers from the failed attempt. Exit-0 + stale leftovers = false success. Narrow but real.
- **`get_cached` schema drift crashes the scheduler.** cache.py:127 `Artifact.model_validate(data)`
  is outside `_load_index`'s corruption guard; one hand-edited/old-schema index entry raises
  `ValidationError` out of `run_dag` instead of cold-missing.
- **`tail_logs`/`log_resource` read whole log files per poll** (server.py:227, 337). Multi-hour
  train logs make each MCP poll an O(file) read; fine at current scale, will not be at 100 MB.
- **`_lock_for` keyed by `str(path)` without normalization** (manifest.py:188) while its docstring
  claims differently-spelled equal paths share a lock. In practice all callers derive paths the
  same way, so the race window is theoretical.
- **`job_error` returns full tracebacks to MCP clients** (jobs.py:72) — internal paths/code leak.
  Acceptable for an authenticated single-user tool; worth knowing.
- **OOM retry leaves `ctx.config` mutated after a successful retry** (oom_retry.py:128-134) —
  harmless today (ctx is discarded) but a future per-stage ctx reuse would inherit fallback config.
- **Concurrent jobs on the same `run_id` are unguarded** (jobs.py:25-30 documents it; `_spawn`
  overwrites `_jobs[run_id]`). Known-open under T17 (todo) — not reported as new.

## 3. Looked risky but is correct

- **Topo sort / cycle detection** (dag/graph.py:92-119): Kahn over internal edges, deterministic
  tie-break, clear `CycleError` naming the stuck set. Correct.
- **Failed stage stops the run, downstream stay `pending`** — verified by
  `test_failed_stage_stops_scheduling_descendants_stay_pending`; resume falls out of the cache
  check cleanly.
- **Config `extends:` layering** (config/resolver.py:25-69): true deep merge, per-chain cycle
  detection, diamonds handled, lists replace (documented behavior), `extra="forbid"` catches
  dropped/typo'd keys at validation. Solid.
- **Bearer auth** (mcp_server/auth.py): every HTTP request gated, constant-time compare, lifespan
  pass-through is correct and safe, no default token (config.py:49-58). Real-loopback tests exist
  and pass.
- **Vendoring rule**: no stage imports or subprocesses the reference scripts; `cuda`/`isaac` stages
  exec only `pipeline/vendored/**` inside containers. Grep-verified.
- **Heavy module-scope imports**: none outside `vendored/cuda/*` (never imported on the host).
- **Manifest writes**: atomic temp+rename + per-path lock + Windows retry loop — thoroughly done.
- **Container exec**: non-zero exit surfaces as `CudaStageError`/`IsaacStageError` with the log
  path; combined stdout/stderr streamed to disk. No silent crashes.
- **`_strip_host_prefix`** boundary matching is case-insensitive, separator-tolerant, and
  prefix-safe (`Q:/Omniverse2` doesn't match `Q:/Omniverse`).

## 4. Convention violations (separate list)

- **`pipeline/stages/cuda_common.py:47`**: `CUDA_EXTRA_ENV = {"PYTHONPATH": "/workspace/core"}` —
  hardcodes `/workspace` outside `pipeline/paths.py`, against INSTRUCTIONS.md's "Never hardcode
  `Q:\` / `/omniverse` / `/workspace` anywhere else". Should derive from
  `paths.REPO_ROOT_CONTAINER`.
- **`pipeline/stages/isaac_common.py:102`**: `DEFAULT_NATIVE_ISAAC_PYTHON = r"Q:\Omniverse\..."`.
  Softened by the fact INSTRUCTIONS.md's own Environments section names this exact default, but it
  is still a `Q:\` literal outside `paths.py`.
- **`pipeline/containers/manager.py:285`**: production logic branches on a test double's class
  name (`_Fake...`), and (worse) uses that to decide correctness behavior — see bug 1.2.
- **`test_import.py::test_no_heavy_imports_at_module_scope`** only imports the top-level
  `pipeline` package (whose `__init__` imports nothing), so it would NOT catch a future
  module-scope `import torch/docker/...` in `pipeline.api`, `pipeline.dag.scheduler`, or
  `pipeline.stages.*`. The rule currently holds (grep-verified), but the test doesn't enforce it.

## 5. Test-suite state and coverage gaps

**The sandbox suite is red on master: 12 failed / 250 passed / 9 skipped** (ran here with
`uv run --package pipeline --with pytest python -m pytest tests/ -q`):
- 2 failures are bug 1.1 (`roi.none`).
- `test_mcp_server.py::test_gpu_status_over_real_http_with_valid_token` asserts
  `payload["gpu"] is None` — environment-dependent; fails on any machine WITH a GPU (including
  this one: it read the real RTX 3090). The test encodes the sandbox, not the contract.
- 3 `test_roi_motion_gate.py` + 6 `test_segment_kabsch.py` failures (numeric/assertion failures in
  vendored host algorithms — e.g. `assert roi1[-50:].sum() == 0` got 48). Not diagnosed here
  (outside orchestrator-core scope); possibly platform/seed sensitivity, possibly real regressions.
  Either way, master does not pass its own suite on Windows.

Gaps relative to the risks above:
- No test runs `run_pipeline` with the REAL registry end-to-end plan (the two that come closest
  are the ones currently failing); `_auto_stage_plan` is monkeypatched away in test_dag.py.
- No test that a cache hit revalidates artifact existence (1.4), that directory-artifact changes
  invalidate downstream (1.3), or that editing a vendored `cuda/`/`host/` module busts the cache —
  `stage_source_hash` covers only the stage class file (cache.py:40-57), so vendored-code edits in
  a dirty tree do NOT invalidate; the git SHA only moves on commit. Untested and undocumented.
- No negative-authz/traversal tests for `run_id`, preset name, or `external_artifacts` paths
  (1.5/1.6); no test that `stop_container` refuses an unmanaged id (1.7).
- Container-manager tests use a thorough fake client (good), but nothing covers the real-daemon
  branch of `_cuda_image_up_to_date` (1.2) — precisely the branch that silently reuses stale
  images.

## Known-open items NOT reported as new (per TASKS.md)

- `cancel` unimplemented / no concurrent-job guard -> T17 (todo), surfaced honestly by the MCP
  `cancel_run` tool.
- T16 (WSL2 bundling) deferred; T21/T23 todo; T22 real-GPU run pending.

## Round-2 fixes applied 2026-10-05

Fixed by the api/artifacts/config/mcp/containers agent (scheduler/oom_retry/output-verification
bugs 1.3/1.4/1.8/1.9 handled by a parallel agent):

- **1.1 (`roi.impl: "none"`)** — `pipeline/api.py::_auto_stage_plan` now skips any multi-impl
  role whose config selector is `"none"` instead of emitting an unregistered `roi.none` stage.
  `run_pipeline("base")` gets past stage resolution (fails later on the documented missing
  `raw_mesh` external input, as intended). Both previously-red tests
  (`test_stages_isaac.py::test_api_run_pipeline_seeds_external_artifacts_before_run_dag`,
  `test_mcp_tools.py::test_run_pipeline_returns_run_id_immediately`) pass again.
- **1.2 (stale CUDA image silently reused)** — `pipeline/containers/manager.py::
  _cuda_image_up_to_date` no longer branches on the fake test client's class name and no longer
  honors `PIPELINE_REBUILD_CUDA_IMAGE`; a build-hash mismatch now ALWAYS means "not up to date"
  (rebuild), per `ensure_image`'s docstring and the T11-incident intent. No opt-out retained:
  any silent-reuse flag re-opens the exact bug the hash label was added to close.
- **1.7 (`stop_container` could stop any host container)** — `manager.py::stop_by_id` now raises
  `ContainerError` unless the container carries the `pipeline.managed` label.
- **1.6 (path traversal)** — single choke-point validation: `pipeline/artifacts/paths.py::
  validate_run_id` (charset `[A-Za-z0-9_.-]`, no `..`, plus a resolve+`is_relative_to`
  confinement check inside `run_dir`, so every manifest/log/run-dir consumer is covered), and
  `pipeline/config/resolver.py::validate_preset_name` (same charset + confinement under
  `presets/`) called by `_preset_path`. `store.list_runs` skips invalid-id directories.
- **1.5 (MCP arbitrary file read)** — one policy in `pipeline/artifacts/paths.py`:
  `validate_external_artifact_path` (seeding-time gate in `api.run_pipeline`: absolute, resolves
  under runs/repo/assets roots via `pipeline.paths.get_roots`, exists, dir-vs-file shape matches
  the declared kind; dict-shaped values from MCP clients are coerced through `Artifact` first)
  and `resolve_servable_artifact_path` (serving-time gate applied in `mcp_server/server.py`'s
  `artifact_resource` + `get_preview` and `mcp_server/artifact_view.py`'s
  `read_artifact_summary`). The roots match the repo's T06 doctrine ("every path the pipeline
  cares about falls under the repo or assets root"); `PIPELINE_ASSETS_ROOT`/
  `PIPELINE_REPO_ROOT` overrides apply.
- **1.10 (`run_pipeline(run_id=<existing>)` clobber)** — `api.run_pipeline` raises
  `FileExistsError` if the run's manifest already exists, and validates the `run_id` charset
  before any directory is created.

Tests: new `tests/test_security_guards.py` (traversal rejection, roi.none plan, external-artifact
gate, dict coercion, clobber guard); `tests/test_containers.py` gained
`test_stop_by_id_refuses_an_unmanaged_container` and
`test_stale_cuda_image_is_rebuilt_with_no_env_opt_out`; `tests/test_mcp_tools.py` gained
`test_artifact_resource_refuses_a_path_outside_the_allowed_roots` and
`test_run_id_traversal_is_a_tool_error` (its `_seed_completed_run` now sets
`PIPELINE_ASSETS_ROOT` so the synthetic artifacts sit inside an allowed root);
`tests/test_stages_isaac.py`'s `run_pipeline` test now creates the external input files on disk
(seeding requires existence).

Suite (`uv run --package pipeline --extra mcp pytest -q`): before 12 failed / 250 passed;
after 10 failed / 289 passed. The 2 roi.none failures are fixed; the remaining 10 are the
out-of-scope 3 `test_roi_motion_gate` + 6 `test_segment_kabsch` numeric failures (owned by the
parallel agent) plus `test_gpu_status_over_real_http_with_valid_token`, which fails on this
GPU-equipped machine by design. Note: plain `uv run --package pipeline pytest -q` (no `mcp`
extra) still cannot collect `test_mcp_server.py`/`test_mcp_tools.py` (`anyio` missing) —
pre-existing environment behavior, unchanged.


---

## Scheduler fixes applied 2026-10-05 (bugs 1.3 / 1.4 / 1.8 / 1.9)

Fixed by the scheduler/scene-gen agent (this section only covers `pipeline/dag/scheduler.py`,
`pipeline/artifacts/hashing.py`, `pipeline/stages/render.py`, `pipeline/stages/seg_extract.py`
and their tests; bugs 1.1/1.2/1.5/1.6/1.7 were owned by a parallel agent — see their section
above).

- **1.3 (directory artifacts never hashed)** — `pipeline/artifacts/hashing.py` gained
  `hash_directory()`: a SHA-256 over sorted relative file paths with per-file size+mtime, plus
  full content hashes for files ≤ 4 MiB (capture metadata) and size+mtime only for large files
  (frames/point clouds), mirroring `_fast_fingerprint`'s documented tradeoff. The scheduler's
  post-success hash step now fingerprints directories too, and `_input_hash()` hashes
  hashless (externally seeded) inputs on the fly, so a forced `capture.isaac`/`train.default`
  rerun changes every downstream cache key. `tests/test_scheduler_hardening.py` covers both the
  fingerprint itself and the force-rerun→resume scenario.
- **1.4 (cache skip never revalidated)** — new `_artifacts_intact()` gate on both freshness
  paths (`_already_recorded` and the cross-run `get_cached` hit): recorded artifact paths must
  still exist, and *file* hashes must still match. Directories are existence-only by design —
  `render`/`amp` write into their input model dir in place, so a recorded directory hash
  legitimately goes stale. A stale entry is treated as a cache miss and re-run (logged to the
  stage log). Three regression tests: deleted output, modified output, same-run resume.
- **1.8 (missing-output tolerance)** — the scheduler now verifies, inside the same try/except
  that records failures, that every declared output is returned and every returned artifact
  path exists (non-empty for directory kinds) before recording `success`/caching
  (`StageOutputError`). `render.default` additionally checks the expected
  `<model_path>/<split>/ours_<it>/` dirs for each non-skipped split, since its `renders`
  artifact re-registers the pre-existing model dir (same pattern as train/capture's checks).
- **1.9 (OOM-fallback cached under original key)** — on an `oom_fallback` success the scheduler
  recomputes the cache key from the *effective* config (`ctx.config`, left swapped to the
  fallback by `run_with_oom_retry`) and records/caches under that key only. Deliberate
  consequence, documented in the code: a resume of a fallback stage re-runs it (and OOM-retries
  again) rather than reusing degraded outputs for the original config.

Test updates forced by the fixes: `test_stages_isaac.py`'s chain test previously asserted
`convert.default` stays `skipped` after an upstream group change — that assertion *encoded*
bug 1.3 and now correctly expects a rerun; `test_stages_cuda.py`'s fake execs stub render
split dirs; `test_dag.py`'s OOM toy stage writes its declared output. New file
`tests/test_scheduler_hardening.py` (9 tests). Suite: no new failures — remaining reds are the
pre-existing out-of-scope set (3 `test_roi_motion_gate` + 6 `test_segment_kabsch` +
`test_gpu_status` by design).


---

## Round-3 vendored-algorithm fixes applied 2026-10-06

Fixed by the vendored-algorithm agent (this section covers
`pipeline/vendored/host/kabsch_em.py`, `pipeline/vendored/host/rigidity_graph.py`,
`pipeline/vendored/host/motion_gate.py`, `pipeline/config/models.py`,
`tests/test_segment_kabsch.py`, `tests/test_roi_motion_gate.py`, plus the same Otsu fix
mirrored into `motion-seg/motion_seg/rigidity_graph.py`). All fixes follow the verified
root-cause diagnosis; the 9 tests committed red in `5f8394c` are now green.

- **Bug A — EM sigma-annealing start destroyed initialisation**
  (`vendored/host/kabsch_em.py`, `_em_single`, was `sigma_current = max(sigma, 1.0)` at old
  line 281). Residuals are sums of squares over 3T≈180 dims with per-coord σ≈0.008–0.01, so
  starting at σ=1.0 flattened all responsibilities to 1/K — a degenerate fixed point
  (verified: GT init → ARI −0.003; data-scaled σ → ARI 0.9988). Proposal 05 specifies a
  fixed per-scene-calibrated σ and no annealing, so the annealing (start + adaptive update)
  was dropped entirely per the spec rather than rescaled. Side effect documented in the
  test: from the FFT seed, EM now needs ~43 iterations to converge, so
  `test_em_single_converges_and_improves_likelihood` was bumped `max_iter` 30→60.
- **Bug B — BIC missed the Gaussian 1/σ² factor** (`kabsch_em.py:_bic`, old lines 238–243).
  `weighted_r + ν_K·log(N)` compared an O(10–80) residual term against an O(8k–42k)
  penalty → BIC monotonically increasing in K by construction, model selection always
  collapsed to the smallest K, and greedy splits were always rejected. Replaced with the
  proper Gaussian BIC `3TN·log(σ̂²) + (6TK+K)·log(3TN)`, σ̂² = Σγr²/(3TN).
  **Thesis-relevant:** the real-data conclusion in
  `.claude_notes/NOTES_T20_kabsch_em_2026-08-11.md` ("BIC increases monotonically with K →
  107 parts not resolvable") is an artifact of this bug; a dated addendum was appended to
  that notes file (append-only) invalidating it pending re-measurement.
- **Bug D — Otsu plateau tie-breaking** (`vendored/host/rigidity_graph.py:72` and the same
  code in `motion-seg/motion_seg/rigidity_graph.py:74`). `centers[np.argmax(between)]`
  returns the FIRST bin of a max plateau, hugging the noise cluster on cleanly bimodal
  distributions (0.1% margin in the dilation fixture → far points leaked into the ROI).
  Replaced with the plateau midpoint:
  `idx = np.flatnonzero(between == between.max()); 0.5*(centers[idx[0]]+centers[idx[-1]])`.
- **motion_gate precision fixes** (`vendored/host/motion_gate.py`):
  (a) default `dilation_hops` 1→0 (line 32) — dilation admits overlapping statics; tests
  that need it pass `dilation_hops=1` explicitly (`test_dilation_captures_neighbors`
  already did). Note: the *config* layer (`RoiMotionGateConfig.dilation_hops = 1`,
  forwarded verbatim by the `roi.motion_gate` stage) was intentionally left unchanged —
  stage behavior is only affected if the preset opts in.
  (b) Readmission candidates now anchor on the energy-gated `moving` set instead of the
  post-dilation `roi` (old lines 109–111) — matches the documented intent ("points rigidly
  connected to the moving region") and eliminates the readmission cascade through
  static–static edges (measured: 61 → 0 readmitted FPs at hops=1).
  (c) No-signal detector added to the degenerate guard (line 72 area):
  `energy.max() < min_signal_ratio * q25(energy)`, `min_signal_ratio=10.0` — implements the
  rule the T19 notes documented but never coded. Deviation from the diagnosis: it proposed
  the *median* as noise-floor proxy, but the median sits in the mover cluster when ≥50% of
  the scene moves (exactly the `test_snr_shape_and_range` fixture, which regressed with the
  median form), so the 25th percentile is used instead; all-static ratio ≈ 2–3, moving
  scenes ≫ 10.
- **Gap C — implemented, not xfailed.** FFT-fingerprint init caps at ARI ≈ 0.91 on the
  all-rotations fixture (fingerprints vary within parts). Proposal 05 §4 already prescribes
  the remedy ("seed from proposal 04's spectral result") and the algorithm was already in
  the tree — the calibrated rigidity-affinity spectral partition
  (`rigidity_graph2.segment_by_rigidity2(partition="spectral")`). Added `init="spectral"`
  to `kabsch_em.py` (`_init_spectral`, with an overflow-merge step mapping the spectral
  cluster count onto exactly K bodies) and to `SegmentKabschConfig.init`'s Literal (the
  config comment already anticipated it). Measured on the T20 fixture: spectral seed alone
  ARI 1.0; after EM refinement ARI 0.9988, converged in 9 iterations. The three ARI-gated
  tests (`test_segment_by_kabsch_recovers_parts`, `test_fps_subsample_path`,
  `test_segment_kabsch_stage_runs_end_to_end`) and `test_bic_prefers_correct_k` now use
  `init="spectral"`; `init="fft"` remains the default and keeps coverage via
  `test_em_single_converges_and_improves_likelihood` and `test_greedy_split_can_improve_bic`.

Suite (`uv run --package pipeline pytest -q`, from `orchestrator/`): before 299 passed /
10 failed; after **312 passed / 1 failed**, the one failure being
`test_gpu_status_over_real_http_with_valid_token`, which fails on this GPU-equipped machine
by design. No new failures; the 4 previously-passing tests in the two target files still
pass (13/13 in `test_segment_kabsch.py` + `test_roi_motion_gate.py`).
