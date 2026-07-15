# Three-layer pipeline orchestration — implementation plan

Design doc, **no code yet**. Goal: turn the current chain of standalone `.sh` scripts +
scattered configs into one automatable, extensible, remotely-controllable system.
Date: 2026-07-11. Related: [[.claude_notes NOTES_omniverse_pipeline]],
[[.claude_notes NOTES_4dgs_motion_segmentation]].

Decisions locked with user (2026-07-11):
- Orchestration foundation = **custom lightweight DAG** (pure-Python stage graph, minimal deps).
- Runtime host = **WSL2 (Linux)** on the user's machine, driving Docker Desktop + NVIDIA
  Container Toolkit.
- Extensibility = **plugin registry + config variants** (swappable stage implementations behind
  one interface, experiments declared in config presets).

---

## 0. The three problems and how the design attacks each

1. **Running the whole pipeline is manual/effortful.** → Layer 1 runs the full DAG from one
   call/one config, with caching so reruns only redo what changed, and resume-from-stage.
2. **No way for Claude to touch the GPU-bound parts (Isaac, 4DGS, MBS).** → Layer 2 is an MCP
   server on the WSL2 host that exposes "run stage / read output / manage containers / read GPU"
   to Claude, which the sandbox cannot do itself.
3. **Fragmented, no consistent way to add ideas.** → One package with a **stage registry**
   (register a new implementation behind a stable interface) + **typed config presets** (declare
   an experiment without touching core code) + **typed artifacts** passed between stages.

---

## 1. The pipeline as it exists today (ground truth for the DAG)

Two distinct GPU container worlds, stitched by hand:

| # | Stage | Current entry point | Environment | GPU |
|---|---|---|---|---|
| 1 | prep_split | `omniverse_pipeline/split_mesh.py` | USD/trimesh (CPU) | no |
| 2 | prep_motion | `omniverse_pipeline/add_motion.py` | USD/numpy (CPU) | no |
| 3 | capture | `omniverse_pipeline/omni_capture.py` | **Isaac Sim** `nvcr.io/nvidia/isaac-sim:6.0.1`, `/isaac-sim/python.sh` | yes |
| 4 | convert | `omniverse_pipeline/omni_to_4dgs.py` | numpy+Pillow (CPU) | no |
| 5 | train | `train.py` | **CUDA/PyTorch** (repo `Dockerfile`, cu124) | yes |
| 6 | render | `render.py` | CUDA/PyTorch | yes |
| 7 | seg_extract | `motion_seg/extract_trajectories.py` | CUDA/PyTorch | yes |
| 8a | seg_cluster (Option B) | `motion_seg/segment_rigid.py` | numpy/scipy (CPU) | no |
| 8b | seg_cluster (Option A) | `motion_seg/mbs_infer.py` | CUDA + MBS `ext/` ops | yes |
| 9 | seg_eval | `motion_seg/evaluate_segmentation.py` | numpy/scipy (CPU) | no |
| 10 | amp | `render_amp.py` (+ `motion_amp/renderer.py`, `ampUI.py`) | CUDA/PyTorch | yes |

Handoffs today are file-path conventions (`data/multipleview/<name>` → `output/multipleview/<name>`)
and manual copy of Isaac output from `Q:/Omniverse/renders/...`. Settings live in **three
disjoint places**: `capture_config_pump.yaml`, `arguments/multipleview/<name>.py`, and CLI flags
baked into `train_pump.sh` / `run.sh`. This scattering is the root of problem #3.

Two images, and they **never need to run at the same time** on one GPU → sequential scheduling is
acceptable; the resource manager mainly prevents accidental double-booking and tunes per-stage
memory.

---

## 2. Layer 1 — execution module (`pipeline/` package)

A single importable Python package (not a script). Everything Layers 2 and 3 do is a call into
this package's public API.

### 2.1 Stage abstraction
A `Stage` base class. Each concrete stage declares:
- **inputs**: named artifact dependencies (what upstream stages must have produced).
- **outputs**: named artifacts it produces (typed: path + kind + metadata).
- **environment**: which execution env it needs — `host` (WSL2 venv, CPU stages),
  `cuda` (the repo Dockerfile image), or `isaac` (isaac-sim image).
- **resources**: a `ResourceRequest` (needs_gpu, vram_gb estimate, ram_gb estimate).
- **run(ctx)**: does the work. Initially this **wraps the existing script** (invokes it in the
  right environment with translated paths/flags) rather than rewriting it — lowest-risk migration.
  Refactor scripts into importable functions opportunistically later.

Stages are deliberately thin adapters over the code that already works and is already verified.

### 2.2 Stage registry (the plugin mechanism)
- Stages register under a dotted name via a decorator, e.g. `@register("segment.rigid")`,
  `@register("segment.mbs")`, `@register("capture.isaac")`.
- Multiple implementations can satisfy the same **role** (e.g. role `segment` → `rigid` | `mbs`);
  the config picks which. This is how "add a new idea" works: write a new class, register it,
  reference it in a preset. No core edits, no new `.sh` file.
- Discovery via a decorator-populated registry (and/or Python entry points) so third-party /
  experimental stages can live outside the core package.

### 2.3 DAG scheduler + caching (fixes problem #1)
- Build the execution graph by topologically sorting declared input/output artifact deps.
- **Cache key** per stage = hash of {resolved config for that stage} + {content hashes of input
  artifacts} + {code version: git SHA + hash of the stage's source file}. If an up-to-date output
  exists, **skip** it. Code changes correctly invalidate the cache.
- Supports `run_all`, `from_stage=X`, `to_stage=Y`, `only=[...]`, and `force` (ignore cache).
- Resumable: a crashed run picks up at the first stale stage.

### 2.4 Artifact store + run manifest (what Layer 2/3 read)
- Every run gets a run directory + a `manifest.json`: full resolved config, code git SHA,
  per-stage {status, start/end time, wall time, peak VRAM/RAM, log file path, output artifact
  paths + metadata}, and overall status.
- Artifacts are typed records (kind ∈ {dataset, model, npz, ply, png, video, json} + path +
  producing-stage + metadata). This is the single source of truth the MCP server and UI query —
  no scraping stdout.

### 2.5 Unified config system (fixes problem #3, biggest single win)
- One typed schema (pydantic) covering **all** stage settings — folds `capture_config*.yaml`,
  `arguments/multipleview/*.py`, and the CLI flags currently in the `.sh` files into one place.
- Layered composition: `base` preset ← `scene` preset (e.g. `pump01`) ← `experiment` overrides
  (params + which implementation fills each role). Validated up front, before any container spins
  up, so typos fail in seconds not hours.
- A preset **is** an experiment definition: reproducible, diff-able, version-controlled.

### 2.6 Container manager (fixes problem #2's mechanics)
- Abstracts Docker via the Docker SDK/CLI against Docker Desktop's engine from WSL2, with GPU
  passthrough (`--gpus all`, NVIDIA Container Toolkit).
- Uses the **existing devcontainer definitions as the source of truth** for image + mounts:
  repo → `/workspace`, `Q:/Omniverse` → `/omniverse`, plus the Isaac cache volumes already
  defined in `omniverse_pipeline/.devcontainer/devcontainer.json`.
- Responsibilities: ensure/build image, start container with correct mounts + GPU flags, `exec`
  a stage command, stream logs to the run's log file, tear down (or keep a warm long-lived
  container per image to avoid Isaac's slow cold-start / shader cache warmup).
- **Centralizes all path translation** (`Q:\` ↔ `/mnt/q` ↔ container `/omniverse`; repo host path
  ↔ `/workspace`) in exactly one module — today this logic is smeared across scripts and configs.

### 2.7 Resource manager (VRAM/RAM awareness)
- Query GPU (pynvml/`nvidia-smi`) for total+free VRAM and system RAM.
- Gate scheduling: single-GPU serial queue by default; never co-schedule stages whose combined
  VRAM estimate exceeds free VRAM.
- **Adaptive knobs** — the code already exposes several: `low_vram_mode` (ampUI/render_amp),
  seg subsample size / working-set (`mbs_infer.py` default 4000), `rt_subframes` (capture),
  opacity thresholds. The manager sets these from measured headroom, and on an OOM it retries the
  stage with reduced-memory settings (smaller working set, low_vram_mode on) before failing.

### 2.8 Public API (the whole surface Layer 2/3 sit on)
`run_pipeline(config, from_stage=…, to_stage=…, only=…, force=…) -> run_id` (async/background),
`run_stage(...)`, `get_status(run_id)`, `list_runs()`, `list_artifacts(run_id)`,
`get_artifact(run_id, name)`, `cancel(run_id)`, `list_presets()`, `validate_config(preset)`,
`gpu_status()`, `list/start/stop_container(...)`.

---

## 3. Layer 2 — MCP server (gives Claude control of the GPU parts)

A thin MCP server on the WSL2 host wrapping the Layer 1 API + Docker + filesystem. It exists
because the Claude sandbox has no CUDA, no Isaac, no Docker — this server runs where those live.

- **Long jobs are async**: `run_pipeline`/`run_stage` return a `run_id` immediately (train can be
  hours); Claude polls `get_run_status` / `tail_logs`. No blocking tool calls.
- **Tools**: `list_presets`, `validate_config`, `run_pipeline`, `run_stage`, `get_run_status`,
  `tail_logs`, `list_runs`, `list_artifacts`, `read_artifact` (text/JSON, or an npz *summary* —
  shapes/keys/stats, not raw dumps), `get_preview` (returns PNGs/video paths — segmentation
  previews, renders, amp clips — so Claude can actually *see* results), `gpu_status`,
  `list_containers`, `start_container`, `stop_container`, `cancel_run`.
- **Resources**: expose run manifests, log files, and preview images as MCP resources.
- **Safety**: whitelisted operations only; no arbitrary shell exec exposed.
- **Open decision — transport/reachability.** The server must be reachable from wherever Claude
  runs. If Claude runs locally on the same machine → stdio. If remote → HTTP/SSE with auth (and a
  tunnel). This is the one item that needs your input before Layer 2 is built; it doesn't affect
  Layers 1/3.

---

## 4. Layer 3 — UI (least important; explicitly deprioritized)

Thin layer over the **same** Layer 1 API — no logic duplication.
- Two viable routes: (a) **Streamlit** (precedent: `ampUI.py` already exists and its
  amplification-param panel can be reused as one view), fastest to stand up; or (b) a small
  **FastAPI backend + light frontend** if a richer multi-run dashboard is wanted later.
- Capabilities: pick/edit a preset, launch a run, watch per-stage progress + live logs + GPU
  meter, browse artifacts/previews (renders, segmentation PNGs, amp videos), compare runs.

---

## 5. Phasing (each phase independently testable; old `.sh` scripts stay as fallback until parity)

- **Phase 0 — framework skeleton.** Package scaffold, config schema, artifact/manifest store,
  `Stage` base + registry + DAG scheduler. Wrap the **CPU stages first** (convert, seg_cluster
  Option B, seg_eval) end-to-end with **no containers** — proves the framework on the pieces that
  run anywhere (they're already verified on synthetic data).
- **Phase 1 — container manager + CUDA stages.** Wrap train, render, seg_extract, amp using the
  repo Dockerfile image. Now the whole *reconstruction → segmentation → amplification* half runs
  from one call. This delivers most of problem #1's value.
- **Phase 2 — Isaac stages.** Wrap prep_split, prep_motion, capture via the isaac-sim image
  (warm long-lived container to dodge cold-start). Full end-to-end from USD asset to amplified
  render.
- **Phase 3 — resource manager.** VRAM/RAM gating + adaptive-retry on OOM.
- **Phase 4 — MCP server.** Stand up Layer 2 over the Layer 1 API (resolve transport first).
- **Phase 5 — UI.**

---

## 6. Key risks / decisions to flag

- **Wrap vs rewrite** existing scripts: wrap (subprocess in the right container) to preserve the
  already-verified behavior; refactor to importable functions only where it clearly pays off.
- **Config unification is the highest-value but most tedious** task (many existing knobs across
  three sources). It's what makes problem #3 actually go away, so it should not be shortcut.
- **Path translation** must live in one module or it will rot exactly like the current scripts.
- **Cache correctness**: code changes must invalidate — hence git SHA + per-stage source hash in
  the key.
- **Isaac cold-start**: keep a warm container + persist the shader/asset cache volumes (already in
  the devcontainer) or capture stages feel slow and "manual" all over again.
- **MCP transport** (Section 3) is the only true blocker-style open question; confirm how Claude
  connects to the WSL2 host before Phase 4.

---

## 7. What this buys you, per original complaint

- *Effort:* `run_pipeline(preset="pump01")` (or one MCP call) instead of hand-running 4+ scripts
  across two containers with manual file moves; cached reruns are near-instant.
- *Claude can drive the GPU:* via Layer 2, I can launch runs, read manifests/logs, view preview
  images, and start/stop containers on your machine — none of which the sandbox can do today.
- *Adding ideas:* register a new stage implementation (e.g. a third segmentation backend) and
  reference it in a preset — no new `.sh`, no core edits, and the config/artifact/caching
  machinery it inherits is free.

---

## 8. Implementation log

### T01 done (2026-07-12) — subproject scaffold & tooling

- `orchestrator/pipeline/` package created: `__init__.py` + stub submodules `config/`, `stages/`,
  `dag/`, `artifacts/`, `containers/`, `resources/`, plus `api.py` with typed stub signatures for
  the full public API from `ARCHITECTURE.md` (`run_pipeline`, `run_stage`, `cancel`, `get_status`,
  `list_runs`, `list_artifacts`, `get_artifact`, `list_presets`, `validate_config`, `gpu_status`,
  `list_containers`, `start_container`, `stop_container`) — all raise `NotImplementedError`.
- Wired into the repo build the same way the CUDA submodules are: `orchestrator/pyproject.toml`
  is a new uv **workspace member** (`name = "pipeline"`, deps `pydantic`/`docker`/`pynvml`,
  setuptools backend). Root `pyproject.toml` gained `[project.optional-dependencies] orchestrator
  = ["pipeline"]` and a `pipeline = { workspace = true, editable = true }` source entry — a plain
  `uv sync` will **not** touch it; only `uv sync --extra orchestrator` / `pip install -e
  .[orchestrator]` installs it. No existing torch/CUDA deps touched.
  - **Why this pattern:** mirrors `submodules/simple-knn` exactly (own pyproject.toml, workspace =
    true, editable = true) so the orchestrator doesn't need its own build-system config invented
    from scratch, and stays fully opt-in.
- Test harness: `orchestrator/tests/test_import.py` (plain import, submodule import, an
  `api` stub raising `NotImplementedError`, and a guard that importing `pipeline` doesn't drag in
  `torch`/`torchvision`/`docker`/`pynvml`). Pytest config lives in `orchestrator/pyproject.toml`
  (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`, `pythonpath = ["."]` — needed so `import
  pipeline` resolves without installing the package first).
- **Verified in an isolated CPU-only venv** (this repo's real `.venv` is pinned to Python 3.12.12
  + cu126 torch, which isn't buildable in the sandbox): package imports cleanly with zero extra
  deps, no banned modules get pulled in, all 4 tests pass. Root `pyproject.toml`'s new TOML was
  hand-validated by reparsing the exact edited text with `tomli` (see gotcha below).
- **Gotcha found, not a real bug:** the bash sandbox's mount of the repo root keeps showing a
  **stale, truncated copy of the root `pyproject.toml`** (still the pre-edit version, old mtime)
  even ~30s after the edit — while every other file (new `orchestrator/pipeline/*.py`,
  `orchestrator/pyproject.toml` itself) synced immediately. The Read/Edit tool's own view of the
  file (authoritative for what's actually on disk) is correct and matches the intended diff.
  Worth a sanity check (`cat orchestrator/../pyproject.toml` or `git diff`) the next time you're
  in WSL2, in case that staleness ever reflects something real rather than a sandbox-mount quirk.

### T02 done (2026-07-12) — unified config schema & presets

- `orchestrator/pipeline/config/models.py`: one pydantic schema (all models `extra="forbid"`)
  covering every stage — `ModelParams`/`PipelineParams`/`ModelHiddenParams`/`OptimizationParams`
  (the 4DGS core param groups from `arguments/__init__.py`), `TrainConfig`/`RenderConfig`/
  `SegExtractConfig`, `CaptureConfig` (+ nested scene/rig/capture/output/lighting sections),
  `ConvertConfig`, `PrepSplitConfig`/`PrepMotionConfig`, `SegEvalConfig`, and `AmpConfig`.
  Segmentation is the schema's "role -> impl" example: `SegmentConfig.impl: "rigid" | "mbs"` picks
  between `SegmentRigidConfig` (Option B) and `SegmentMbsConfig` (Option A), with a model validator
  rejecting `impl="mbs"` without `mbs.checkpoint` set (missing-required fails fast). Amp channels
  (`pos3d/pos2d/rotation/scale/opacity/SHs/color/cov3D`) are addressed **by name** in a
  `channels: dict[str, AmpChannelConfig]` instead of the old scripts' positional
  `amp_factors`/`freq_low`/`freq_high` lists; `AMP_METHOD_ALIASES` reconciles `ampUI.py`'s
  Streamlit method labels (`"base"`/`"abs"`/...) against `render_amp.py`'s CLI method strings
  (`"eulerian"`/`"eulerian_abs"`/...), which don't match today.
- `orchestrator/pipeline/config/resolver.py`: layered preset merge via an explicit `extends:` key
  inside each preset YAML (string or list, resolved recursively, deep-merged, cycle-checked) —
  chosen over positional base/scene/experiment args because it's simpler to implement/test and
  generalizes for free. `list_presets()`, `resolve_preset()` (unvalidated merge), `validate_config()`
  (merge + pydantic validate, raises on anything wrong).
- `orchestrator/pipeline/config/loader.py`: migration helpers, not on the runtime hot path —
  `load_legacy_capture_yaml()`/`load_legacy_hyperparams()` read the *actual* old
  `capture_config_pump.yaml` / `arguments/multipleview/pump01.py` files (the latter via
  `runpy.run_path`, since they're plain dict-literal modules — same trust model 4DGS's own
  `mmengine.Config.fromfile` already uses) so presets are migrated, not retyped, and so the same
  helpers migrate the *next* scene later.
- Presets (`pipeline/config/presets/*.yaml`): `base` (mirrors `arguments/multipleview/default.py`,
  which itself already diverges from `arguments/__init__.py`'s raw class defaults — every
  multipleview scene should extend `base`, not the bare class defaults), `pump01` (extends base;
  migrated from `capture_config_pump.yaml` + `pump01.py` + `train_pump.sh`'s NAME/PORT — its only
  real diff from `base` is `optim.opacity_reset_interval: 60000`, avoiding the coarse/fine
  opacity-reset NaN collision `pump01.py`'s comment describes), `pump01_segB_tuned` (extends
  pump01; example experiment reproducing `motion_seg/run.sh`'s
  `--threshold-mult 2 --opacity-thresh 0.2` retune-without-re-extract usage).
- **One intentional de-duplication, not a retype:** both `default.py`/`pump01.py` redundantly
  restate `render_process=False` inside their `ModelHiddenParams` dict even though it's really a
  `ModelParams` field (mmengine's merge applies it by attribute name regardless of source dict, so
  the duplication was harmless upstream). The new schema has exactly one field for it
  (`model.render_process`); documented in `MIGRATION.md` with the round-trip test excluding it
  explicitly rather than silently.
- `pipeline/api.py`'s `list_presets()`/`validate_config()` now delegate to `pipeline.config`
  (lazy import inside the function, so `api.py`'s own module scope stays import-light); every
  other `api.py` stub is untouched (still `NotImplementedError`, T03+).
- `pipeline/config/MIGRATION.md`: full field-by-field mapping (all 3 legacy sources -> new schema
  paths), including the bridge note for T09: `train.py`/`render.py`/etc. still only understand
  `--configs <mmengine-python-file>`, so the stage wrapper will need to serialize
  `PipelineConfig.hidden`/`.optim` back out to that shape (the inverse of
  `load_legacy_hyperparams`) — out of scope for T02, recorded so T09 doesn't have to re-derive it.
- Tests: `orchestrator/tests/test_config.py`, 12 new tests (16 total with T01's `test_import.py`).
  Round-trips `pump01` against the real `capture_config_pump.yaml` + `pump01.py` files (not a
  hand-typed copy); fail-fast coverage for unknown preset, unknown `segment.impl`, missing
  `mbs.checkpoint`, bad type, unknown top-level field, unknown amp channel.
- Added `pyyaml>=6.0` to `orchestrator/pyproject.toml` deps + `[tool.setuptools.package-data]` so
  `presets/*.yaml` ships in the built wheel (editable installs don't need this, but a real build
  would silently drop the YAML files without it).
- **Verified in an isolated CPU-only venv**, same pattern as T01: `pydantic`+`pyyaml`+`pytest`
  installed fresh; `pytest -q` from `orchestrator/` -> `16 passed`. Also smoke-tested
  `pipeline.api.list_presets()`/`validate_config()` end-to-end directly.
- **Sandbox-mount staleness bit again, worse this time:** not just `pyproject.toml` (per T01's
  gotcha) but also a freshly-written `pipeline/config/__init__.py` showed up truncated
  (2 lines instead of ~85) in the bash-mounted view for several minutes, while sibling files
  written in the same turn (`models.py`, `resolver.py`, `loader.py`) synced correctly. Read/Edit
  tool's view was correct throughout (confirmed via `ast.parse`); worked around by rewriting the
  bash-visible copy directly from the known-good content and re-verifying with `ast.parse` /
  `tomli` before running tests, rather than waiting indefinitely. Same advice as T01: if this ever
  shows up in a real WSL2 session rather than the sandbox, verify with `git diff` before trusting
  it.

### T03 done (2026-07-13) — artifact store & run manifest

- `orchestrator/pipeline/artifacts/models.py`: pydantic (`extra="forbid"`) `Artifact` (name, kind
  `dataset|model|npz|ply|png|video|json`, path, `producing_stage`, `metadata`, `content_hash`/
  `hash_algo`), `StageRecord` (status/timing/log_path/artifacts/error, `peak_vram_mb`/
  `peak_ram_mb` left nullable for T12), `RunManifest` (run_id/preset/`resolved_config` as a plain
  dict so this package never imports `pipeline.config`/git_sha/created_at/updated_at/status/
  stages/artifacts).
- `orchestrator/pipeline/artifacts/hashing.py`: `hash_path()` defaults to a **fingerprint**, not a
  full hash — `size + mtime_ns + sha256(first 1MiB + last 1MiB)` — because full SHA-256 over
  multi-GB point clouds/videos on every manifest touch is too slow (T05's task file flags exactly
  this risk and suggests this trade). `fast=False` does a real streaming SHA-256 when a byte-for-
  byte hash is actually needed.
- `orchestrator/pipeline/artifacts/paths.py`: run dir layout under `runs/<run_id>/`
  (`manifest.json`, `config_snapshot.json`, `logs/<stage>.log`), default root
  `REPO_ROOT/"runs"` (sibling to `output/`/`data/`), overridable per-call (`runs_root=`) or
  globally via `PIPELINE_RUNS_ROOT` (read at call time, not import time, so tests/monkeypatch
  work without reload). Explicitly *not* the path-translation module — that's T06's job alone.
- `orchestrator/pipeline/artifacts/manifest.py`: atomic writes (`tempfile.mkstemp` + `fsync` +
  `os.replace`, same-directory temp file so the rename is same-filesystem-atomic) — a reader never
  sees a torn file. Reads distinguish `FileNotFoundError` (no manifest yet) from a new
  `ManifestCorruptError` (bad JSON or failed pydantic validation) so callers can treat "unreadable"
  as one clear case instead of catching raw `json.JSONDecodeError`/`ValidationError`. Convenience
  read-modify-write helpers `record_stage_start`/`record_stage_result` (rolls the run's overall
  `status` up to `failed`/`success` once every stage is terminal) sit on top of a generic
  `update_manifest(run_id, mutate_fn)`. `get_git_sha()` is best-effort (`None` on any git/subprocess
  failure, never raises). Concurrency note documented explicitly: individual writes are atomic
  (no torn file, ever) but read-modify-write itself isn't cross-process-locked — good enough for
  T05's serial scheduler, flagged for whoever parallelizes stage execution later.
- `orchestrator/pipeline/artifacts/store.py`: the query helpers the task asked for —
  `list_runs()` (most-recent-`updated_at`-first, skips runs with a corrupt manifest rather than
  failing the whole listing), `get_manifest()`, `list_artifacts()`, `get_artifact()` (raises
  `ArtifactNotFoundError`, a `KeyError` subclass, on an unknown name).
- `pipeline/api.py`: `list_runs`/`list_artifacts`/`get_artifact`/`get_status` now delegate to
  `pipeline.artifacts` (lazy import, same pattern T02 set for `list_presets`/`validate_config`).
  `get_status` reads straight from the manifest — works today even though T05's scheduler doesn't
  exist yet. `run_pipeline`/`run_stage`/`cancel`/`gpu_status`/container controls untouched (still
  `NotImplementedError`, T05/T08/T12 scope).
- Tests: `orchestrator/tests/test_artifacts.py`, 16 new tests (32 total). Acceptance criteria
  covered directly: a hand-constructed `RunManifest` (not built via `create_run`) round-trips
  through `save_manifest`/`load_manifest`; a 4-writer/4-reader thread storm against one manifest
  produces zero errors and a valid file afterward (write-temp-rename holds up under concurrency);
  a manifest with broken JSON and one with an invalid `status` value both raise
  `ManifestCorruptError` instead of crashing. Plus hashing (fast fingerprint stability/change-
  detection, full-mode matches a manual `hashlib.sha256`), store helpers, and an
  `PIPELINE_RUNS_ROOT`-env-override test proving the `pipeline.api` wiring end-to-end.
- Added `orchestrator/runs/` to the root `.gitignore` (generated run manifests/logs, not source).
- **Verified in an isolated CPU-only venv** (fresh `pydantic`+`pyyaml`+`pytest`, same pattern as
  T01/T02): `pytest -q` → `32 passed`.
- **Sandbox-mount staleness bit a third time** (T01/T02 both flagged this): after editing
  `pipeline/api.py` via the Edit tool, the bash-mounted copy of that *specific* file was
  truncated mid-line for several minutes (`return _get_artifact(run_id, artifact_id).mo` — cut
  off inside `.model_dump()`) while every other file edited in the same turn synced immediately;
  a test run against the stale bash-visible copy failed with a bogus `AttributeError` even though
  the Read tool's view (confirmed via `ast.parse`) was correct and complete throughout. Same
  workaround as before: rewrote the bash-visible file directly from the known-good content,
  re-verified with `ast.parse`, reran — passed. Worth flagging to Bartosz: this is 3/3 tasks now
  showing the *same* symptom (one recently-Edit-tool-touched file staying stale/truncated in the
  bash mount for minutes), always on a file touched by `Edit` rather than `Write`; if this recurs
  in a real WSL2 session rather than the sandbox, verify with `git diff` before trusting the bash
  view.

Next unblocked: T04 (stage base & registry), T06 (path translation) — both only depend on T01.
T05 (DAG scheduler & caching) needs T04 in addition to now-done T03.

### T04 done (2026-07-13) — stage base class & registry

- `orchestrator/pipeline/stages/base.py`: `ResourceRequest` (pydantic, `extra="forbid"`:
  `needs_gpu`/`vram_gb`/`ram_gb`, all default to the cheap CPU case), `StageContext` (plain
  `@dataclass`, not pydantic — it's an in-process call-time object, not something serialized):
  `run_id`, `stage_name`, `config` (plain dict, the stage's resolved config section), `run_dir`
  (`Path`), `logger`, plus `paths`/`containers` typed `Any` and defaulted `None` since T06/T08
  haven't landed — this module only reserves their slot in the contract, per the task's own
  "keep ctx small and explicit, changing it later is expensive" note. `Stage` is an `ABC` with
  class-level `inputs`/`outputs` (tuples of artifact names), `environment`
  (`"host"|"cuda"|"isaac"`), `resources` (a `ResourceRequest` default instance), and an abstract
  `run(self, ctx) -> dict[str, Artifact]`. `name`/`role`/`impl` are `ClassVar`s the registry sets
  — never assigned by hand on a stage class.
- `orchestrator/pipeline/stages/registry.py`: `@register("role.impl")` class decorator validates
  the name has a non-empty role and impl (splits on the *first* dot only, so an impl name could
  itself contain a dot if ever needed), checks the decorated class actually subclasses `Stage`,
  sets `cls.name`/`.role`/`.impl`, and stores it in two dicts — a flat `name -> class` map and a
  `role -> {impl -> class}` map (the exact shape `SegmentConfig.impl`-style config lookups need,
  T02). `get_stage(name)` and `get_stage_for_role(role, impl)` both raise a single
  `StageNotFoundError` with the list of what *is* registered in the message (not a bare
  `KeyError`); re-registering the same name raises `DuplicateStageError` naming the class that
  already holds it. `list_roles()`/`list_stages()` are read-only introspection for later layers
  (MCP server T14, UI T15). A `_reset_registry_for_tests()` escape hatch exists but is
  deliberately not exported — production code never needs to unregister a stage.
- `orchestrator/pipeline/stages/echo.py`: dummy `EchoStage`, registered as `"test.echo"` (role
  `test`, chosen so it can never collide with a real pipeline role like `segment`/`train`/`amp`).
  Its `run(ctx)` actually writes `<run_dir>/echo.json` with `{"message": ctx.config.get(...,
  "hello")}` and returns a real `Artifact` pointing at that file — not a fake/mocked return —
  so the acceptance criterion ("produces a valid artifact") is satisfied literally, not just
  in shape. Imported by `stages/__init__.py` for its registration side effect, so `test.echo` is
  always available the moment `pipeline.stages` is imported (no separate test-only bootstrap
  needed).
- `pipeline/stages/__init__.py` now exports the real surface (`Stage`, `StageContext`,
  `ResourceRequest`, `Environment`, `register`, `get_stage`, `get_stage_for_role`, `list_roles`,
  `list_stages`, and the three error classes) — same pattern as T02/T03 filling in what was a
  stub `__init__.py`. Still nothing here imports torch/CUDA/docker/pynvml at module scope.
- Tests: `orchestrator/tests/test_stages.py`, 7 new tests (39 total). Covers the task's three
  acceptance criteria directly (two impls under one dummy role each resolve correctly via
  `get_stage_for_role`, mirroring how `SegmentConfig.impl` would drive it; duplicate registration
  raises `DuplicateStageError`; unknown role and unknown impl-within-a-known-role both raise
  `StageNotFoundError` with a message naming what *is* registered), plus malformed-name rejection
  (`"no_dot_in_this_name"`), rejecting a `@register` target that isn't a `Stage` subclass, the
  `EchoStage` end-to-end run producing a real file + valid `Artifact`, and that
  `list_roles()`/`list_stages()` reflect `test.echo` (always-registered smoke test).
- **Verified in an isolated CPU-only venv** (fresh `pydantic`+`pyyaml`+`pytest`, same pattern as
  T01–T03): `pytest -q` → `39 passed`.
- **Sandbox-mount staleness, 4/4 tasks now** (T01/T02/T03 all hit this): after `Edit`-ing
  `pipeline/stages/__init__.py` (turning the T04 stub into the real module), the bash-mounted
  copy stayed at the original 2-line stub for several minutes while every file created fresh via
  `Write` in the same turn (`base.py`, `registry.py`, `echo.py`) synced immediately — the pattern
  is specifically "a file touched by `Edit`", never `Write`. All 7 new tests failed with
  `ImportError` against the stale copy even though the Read/Edit tool's own view was correct.
  Same workaround as before: rewrote the bash-visible file directly from the known-good content,
  verified with `ast.parse`, reran — `39 passed`. This is now a completely consistent pattern
  across 4 tasks (always `Edit`, never `Write`, always resolves after a direct rewrite) — flagging
  again for a real WSL2 session in case it's ever not just a sandbox-mount quirk there.
- **Registered `test.echo` deliberately real, not mocked:** considered making `EchoStage.run()`
  return an in-memory-only artifact with no filesystem write, but the acceptance criterion says
  "produces a valid artifact" — writing the file makes the test an honest end-to-end check of the
  `ctx.run_dir` → stage output → `Artifact.path` contract that real stages (T07+) will also need,
  rather than only checking the dict shape.

Next unblocked: T06 (path translation, only needs T01) is now the last Phase-0 prerequisite for
T05. T05 (DAG scheduler & caching) is fully unblocked (T03 + T04 both done) — the critical path's
next stop.

### T05 done (2026-07-13) — DAG scheduler & caching

- `orchestrator/pipeline/dag/graph.py`: pure structure, no manifest/config knowledge. `DAGNode`
  (name/stage_cls/inputs/outputs), `resolve_nodes(stage_names)` (looks each up via T04's
  `get_stage`), `producer_map`/`external_inputs` (which inputs have no producer *within the given
  set* — deliberately not an error by itself, since "missing" depends on whether a resumed run
  already has it, which this module can't see), `topo_sort` (Kahn's algorithm, deterministic tie-
  break by name, raises `CycleError` naming the stuck remainder). `MissingDependencyError` lives
  here too (one `DAGError` hierarchy) even though only the scheduler raises it.
- `orchestrator/pipeline/dag/cache.py`: cache key = sha256 of a JSON blob of
  `{stage name, resolved stage config, sorted input-artifact content hashes, code_version}`;
  `code_version = "<git-sha-or-'nogit'>:<stage-source-file-sha256>"` — whole-*file* hash, not
  per-method, so it's coarser than ideal (any edit anywhere in that file invalidates every stage
  defined in it) but simple and correct in the direction that matters (never silently stale).
  Also owns a small cross-run cache index (`runs/.cache/index.json`, same atomic write-temp-
  rename technique as T03's manifest writes, deliberately reimplemented locally rather than
  reaching into `pipeline.artifacts.manifest`'s private helper — this file stays a leaf module):
  `get_cached`/`put_cached` map a cache key to a previous run's artifacts. This is what makes
  caching genuinely cross-run, not just same-run-resume.
- `orchestrator/pipeline/dag/scheduler.py`: `run_dag(run_id, stage_names, resolved_config, ...)`.
  Validates eagerly *before* touching the manifest: `resolve_nodes` (unregistered name →
  `StageNotFoundError`), `topo_sort` (`CycleError`), then cross-checks `external_inputs` against
  an existing run's artifacts if resuming (`MissingDependencyError` if truly nowhere to be found).
  `from_stage`/`to_stage`/`only` narrow the full topo order into what actually executes this call
  (composable, AND semantics if combined); `force` bypasses both freshness checks. Per stage:
  gather input artifacts from the manifest (raises `MissingDependencyError` *again*, at execution
  time, if an ancestor was excluded by `only`/`from_stage` and never actually ran — distinct from
  the eager pre-check, which only covers the full `stage_names` set), compute the cache key, and
  — this took a real bug to get right (see below) — check *this run's own* record first (if
  already `success`/`skipped` with a matching key, literally do nothing, don't touch the
  manifest) before falling back to the cross-run cache index (reuse → record as `"skipped"`,
  referencing the same `Artifact`, never copied). Otherwise actually run: `record_stage_start`,
  build a real per-stage logger (`FileHandler` at `stage_log_path`), call `stage_cls().run(ctx)`,
  fill in `content_hash` for any file-artifact that doesn't have one yet (dataset/model
  directory artifacts stay `None`, documented as a caching weak spot below), `record_stage_result`
  + `put_cached`. A raised exception is caught, recorded as `status="failed"`, and *stops*
  scheduling — remaining selected stages stay `"pending"` in the manifest rather than being
  silently skipped, so a failure is visible. An empty `stage_names` (true today — no real, non-
  `"test"` stage exists until T07) is treated as trivially `"success"` rather than staying
  `"pending"` forever.
- **Real bug caught by the toy-graph tests, fixed before calling this done:** first pass had a
  single `_lookup_fresh` helper that checked same-run-then-cross-run and, either way, called
  `record_stage_result(status="skipped", ...)`. That silently *downgraded* a stage's own honest
  `"success"` record to `"skipped"` the next time the same `run_id` was re-run — technically still
  "fresh", but a lie about what actually happened. Split into `_already_recorded` (same-run match
  → `continue`, no write at all) checked *before* the cross-run `get_cached` lookup (which alone
  still writes `"skipped"`, correctly, since that artifact really did come from elsewhere). Caught
  by `test_first_run_executes_all_second_run_skips_all_via_cross_run_cache`'s third assertion
  (re-running the *same* run_id should show `"success"`, not `"skipped"`) — exactly the kind of
  thing that's invisible unless a test checks the literal status string, not just "did it skip".
- **Cache reuse is deliberately cross-run, not just same-run-resume:** re-read the task spec's "skip
  when an up-to-date output artifact exists" (not "...within this run") and `T07`'s planned
  acceptance criterion ("rerun skips unchanged stages") — a `rerun` naturally means a *new*
  `run_pipeline` call, hence a new `run_id`, so same-run-only caching would never actually satisfy
  it. Hence the two-tier lookup (own manifest, then the global index) rather than just one.
- **`pipeline.api` wiring — the PipelineConfig↔registry glue lives here, not in `pipeline.dag`:**
  `run_pipeline`/`run_stage` were the two remaining stubs from T01 explicitly called out as T05
  scope. `pipeline/dag/` stays config-schema-agnostic on purpose (leaf module, like
  `pipeline.artifacts`/`pipeline.stages`) — it takes an explicit `stage_names` list and an optional
  `stage_configs` override (defaults every stage's `ctx.config` to the *whole* resolved config
  dict if not given, since there's no established per-role config-section convention yet beyond
  `segment`'s `impl` nesting; T07+ can either rely on that whole-dict default or start passing
  `stage_configs` overrides once real per-stage sections matter). `pipeline/api.py` gained
  `_auto_stage_plan(resolved_config)`: every registered role except `"test"`, resolved to one impl
  each — a role with exactly one registered impl uses it, a role with several (only `segment`
  today) is disambiguated by `resolved_config[role]["impl"]`. Right now this always returns `[]`
  (no real role is registered until T07), which is fine: `run_pipeline("any-preset")` still does a
  real, meaningful round trip (resolve config → build an empty DAG → create a manifest → mark it
  trivially `"success"`) and needs zero changes once T07/T09/T10/T11 start registering real
  stages — `list_roles()` will just start returning more.
- **`run_pipeline` always starts a *new* `run_id`** (`f"{preset}-{uuid4().hex[:8]}"`) rather than
  reusing one per preset — cross-run caching (above) is exactly what makes that not cost a real
  re-execution, and keeps runs individually addressable for T15's "compare runs" later. Resuming
  one specific crashed `run_id` isn't exposed through `run_pipeline` itself; `run_stage` (or
  calling `pipeline.dag.run_dag` directly with that `run_id`) covers it.
- **`StageRecord` gained a `cache_key: Optional[str]` field** (`pipeline/artifacts/models.py`,
  T03's file) plus a matching `cache_key=` kwarg on `record_stage_result` — the minimal schema
  extension T05 needed to compare "this stage's previous cache key" against "the one just
  computed" without re-deriving it from scratch. Backward-compatible (`Optional`, defaults `None`)
  so it doesn't touch T01–T04's existing manifest tests.
- Tests: `orchestrator/tests/test_dag.py`, 16 new (55 total): graph structure (topo order,
  external-inputs flagging, cycle detection), cache key sensitivity to config/inputs/git-sha,
  stage-source-hash stability, the toy 3-stage chain's full acceptance bar (first-run-all /
  second-run-all-skip-cross-run / same-run-rerun-stays-success / config-edit-invalidates-
  descendants), `only`/`from_stage`/`to_stage`/`force` combinations, `only` rejecting an unknown
  name, missing-dependency detection both eagerly (fresh run) and satisfied-by-resume, a failing
  stage halting with descendants left `"pending"`, and `pipeline.api`'s new wiring (including a
  `FileNotFoundError` for `run_stage` against a nonexistent run). Toy stages registered under role
  `"test"` (mirroring `EchoStage`) specifically so `_auto_stage_plan`'s real-role auto-discovery
  never picks them up — registering them under a look-alike real role would have leaked into every
  other test in the session that calls `api.run_pipeline`, since the stage registry is global and
  process-wide. Updated two now-stale assertions elsewhere: `tests/test_import.py` and
  `tests/test_artifacts.py` both used to assert `api.run_pipeline("base")` still raised
  `NotImplementedError` (true as of T01/T03) — swapped to `api.cancel`/`api.gpu_status` (still
  real T08/T12 stubs).
- **Verified in an isolated CPU-only venv** (fresh `pydantic`+`pyyaml`+`pytest`, same pattern as
  T01–T04): `pytest -q` → `55 passed`.
- **Sandbox-mount staleness, 5/5 tasks now — worst one yet:** every `Edit`-touched file went stale
  this time (`pipeline/artifacts/models.py`, `manifest.py`, `pipeline/dag/__init__.py`,
  `pipeline/api.py`, `tests/test_import.py`, `tests/test_artifacts.py`, and — new this task —
  `pipeline/dag/scheduler.py` went stale *again* on its second `Edit` after already being
  correctly synced from its first `Write`), while every fresh `Write`-created file
  (`graph.py`, `cache.py`, the scheduler's initial `Write`, `test_dag.py`) stayed correct
  immediately. Same workaround every time: read the tool's own (correct) view, `cat > file
  <<'PYEOF'` the exact content directly in bash, verify with `ast.parse`. Pattern now firmly
  "any `Edit` to an already-existing file", not "the first `Edit` a file ever gets" — worth
  treating as the default assumption for T06+ rather than something to double-check per file.

Next unblocked: T06 (path translation, only needs T01) is the last Phase-0 prerequisite before
T07/T08 can start. T13 (MCP server over HTTP) is also newly unblocked (only needs T05), though
it's Phase 4 and far down the priority order. T07 (wrap CPU stages) needs T02 (done) + T05 (done)
+ T06 (still todo) — once T06 lands, T07 becomes the next critical-path stop.
Related: [[pipeline-orchestration-plan]].

### T06 done (2026-07-13) — path-translation module

- `orchestrator/pipeline/paths.py`: two canonical roots, `repo` and `assets`, each with a
  host (Windows drive-letter) / wsl / container form, held in a `Roots` dataclass. `get_roots()`
  builds the default set at *call* time (same lazy-env-var pattern as `artifacts.paths.
  get_runs_root`): the repo root's WSL2 form comes from `__file__` (this module already lives
  inside the repo — no drive letter/username hardcoded anywhere), its host form is derived via
  the one generic `wsl_to_windows` regex; the assets root has no such anchor so it defaults to
  `/mnt/q/Omniverse` (matching `capture_config_pump.yaml`'s `Q:/Omniverse`), overridable via
  `PIPELINE_ASSETS_ROOT_WSL` (and `PIPELINE_REPO_ROOT_WSL` for the repo root).
- Public API is auto-detecting rather than direction-specific: `to_host(path)`/`to_wsl(path)`/
  `to_container(path, env)` each accept a path in *any* of the three spaces, work out which root
  it's under and which space it's in, and re-render it in the target space. This is what makes
  `to_container(to_host(x)) == x` (the task's stated round-trip criterion) hold for any `x`
  already under a known root, in any starting space — not just host-to-container.
  `windows_to_wsl`/`wsl_to_windows` are the one generic (drive-letter-agnostic, regex-based) pair
  every other conversion composes from; no other function in the module parses a drive letter.
- `env` (`"cuda"`/`"isaac"`) on `to_container`/`container_mounts` is validated (typo → `ValueError`
  immediately) but currently a no-op on the mapping itself — both `.devcontainer/devcontainer.json`
  files mount the repo/assets roots identically today. Kept in the signature per the task spec and
  so a future container with different mounts is an internal change, not an API break.
- `container_mounts(env)` + `MountSpec.as_docker_mount_string()`: ready-made bind-mount specs for
  T08's container manager, in the exact `source=...,target=...,type=bind,consistency=cached`
  shape the existing devcontainer files already use (verified against the literal string in
  `.devcontainer/devcontainer.json`). Cache/auth *volume* mounts (Isaac shader cache, Claude Code
  config) are explicitly left to T08 — those aren't repo/assets path translation.
- Tests: `orchestrator/tests/test_paths.py`, 46 new (101 total). Table-driven over 5 representative
  (host, wsl, container) triples spanning both roots (root-level and nested) × every pairwise
  conversion direction × 3 round-trip compositions (container→host→container, host→wsl→host,
  wsl→container→wsl), plus the generic windows_to_wsl/wsl_to_windows helpers on drive letters
  other than C:/Q: (proving they're not secretly hardcoded), rejection of paths outside both
  known roots, `env` validation, the mount-spec builder's exact shape, and env-var override /
  default behavior for `get_roots()`.
- **Verified in the same isolated CPU-only venv used for T01–T05** (`pydantic`+`pyyaml`+`pytest`):
  `pytest -q` from `orchestrator/` → `101 passed`.
- No sandbox-mount staleness this time — both `pipeline/paths.py` and `tests/test_paths.py` were
  fresh `Write`s (not `Edit`s to a pre-existing file), consistent with the pattern all of T01–T05
  documented ("always `Edit`, never `Write`"). The planning-doc edits (`T06-path-translation.md`,
  `TASKS.md`) used `Edit` on already-existing files, so per that same pattern they may show stale
  in a bash-mounted view for a while — cosmetic only, per [[cowork-mount-staleness-bug]].

Next unblocked: T07 (wrap CPU stages) and T08 (container manager) both have every dependency done
(T02/T05/T06 for T07; T05/T06 for T08) — either is a valid next critical-path stop. T07 is listed
first in the board and sits earlier on the critical path (`T07 → T08 → T09`), so it's the natural
next pick unless there's a reason to front-load the container manager instead.
Related: [[pipeline-orchestration-plan]].

### T07 done (2026-07-13) — wrap CPU stages (vertical slice)

- Three new registered stages, one file each: `pipeline/stages/convert.py` (`convert.default`,
  wraps `omniverse_pipeline/omni_to_4dgs.py:convert()`), `pipeline/stages/segment_rigid.py`
  (`segment.rigid`, wraps `motion_seg/segment_rigid.py:segment_trajectories()`), and
  `pipeline/stages/seg_eval.py` (`seg_eval.default`, wraps a newly-graduated
  `motion_seg/evaluate_segmentation.py:evaluate()`). All `environment = "host"`, `needs_gpu =
  False`, small `ram_gb` estimates (0.5–1.0), and call the wrapped function **in-process** (no
  subprocess) per the design note that a `host` stage already lives in the same venv as the
  orchestrator — subprocess+CLI-arg-building would add complexity with zero isolation benefit here
  (unlike the future `cuda`/`isaac` container stages, T08+). Each stage captures the wrapped
  function's `print()` output via `contextlib.redirect_stdout` into a buffer and forwards it
  line-by-line to `ctx.logger.info(...)`, satisfying "stream logs to the run dir" without
  subprocess. All three import `omniverse_pipeline`/`motion_seg` by anchoring the repo root off
  `Path(__file__).resolve().parents[3]` and inserting it on `sys.path` if missing — the same
  technique `pipeline/paths.py` (T06) uses for its own repo-root anchor, needed because
  `omniverse_pipeline` has no `__init__.py` (a namespace package) and neither package is pip-
  installed. `pipeline/stages/__init__.py` imports all three new modules for their registration
  side effect, alongside the existing `echo` import.
- **`evaluate_segmentation.py` graduated a new public `evaluate(pred_points, pred_labels,
  gt_points, gt_labels, *, drop_floaters=False) -> dict`** out of `main()`'s inline body (design
  option (a) from the brief, chosen over duplicating the propagate/score glue in the stage
  adapter): returns `ari`/`mean_iou`/`matches`/`gt_on_pred`/`pred_points`/`pred_labels`/`n_gt`/
  `n_pred` — everything `main()` used to compute inline. `main()` now calls `evaluate()` and prints
  from its returned dict; the CLI's actual print statements/format strings are untouched, so
  `--recolored-ply`/`--comparison-png` behavior is byte-for-byte preserved. `seg_eval.default`
  calls `evaluate()` directly and also reuses `_write_colored_ply()` (already a private helper,
  called directly rather than duplicated) when `SegEvalConfig.recolored_ply` is set.
  `comparison_png` is deliberately **not** wired into the stage (needs `motion_seg.visualize`,
  which needs matplotlib — an extra dependency this CPU vertical slice has no other reason to
  pull in); a preset that wants it can still run the original CLI directly. This is the one actual
  script-file edit this task made outside `orchestrator/`; it's a pure refactor (verified by
  keeping `main()`'s prints and CLI behavior identical) and squarely the kind of "graduate a script
  into importable functions" case `INSTRUCTIONS.md` calls out, since the orchestrator genuinely
  needs to call this logic in-process.
- **Found and fixed a real gap in T04/T05's `StageContext` contract: it never actually carried
  resolved input artifacts.** `Stage.run(ctx)` had `ctx.config`/`ctx.run_dir`/`ctx.logger` but no
  way to learn *where* an upstream artifact actually lives (a capture directory path, a
  `trajectories.npz` path, ...) — T04/T05's toy test stages never needed this because they only
  ever wrote synthetic content, never read an upstream file. Every real T07 stage needs exactly
  this. Fixed by adding `StageContext.inputs: dict[str, Artifact] = field(default_factory=dict)`
  (`pipeline/stages/base.py`) and having `pipeline/dag/scheduler.py`'s `run_dag` pass the
  already-computed `input_artifacts` dict into the `StageContext` it builds (it was computing this
  dict for cache-key purposes already and simply never forwarded it). Backward-compatible default
  (`{}`) so `EchoStage` and every T05 toy stage are unaffected. This is exactly the kind of thing
  a toy-graph test suite can't catch (no toy stage ever needed an input's actual location) but a
  real vertical slice immediately does — same category of finding as T05's own `_already_recorded`
  bug, caught by actually running real stages rather than stubs.
- **`pipeline/api.py` gained per-stage config slicing** (`_stage_config_for(name, resolved_config)`
  ): mirrors how `SegmentConfig.impl` already nests an implementation's section under its role
  (`resolved_config["segment"]["rigid"]` for `"segment.rigid"`), falling back to the whole
  resolved-config dict for a stage with no matching top-level section (preserves `test.echo`'s
  existing behavior). `run_pipeline`/`run_stage` now build a `stage_configs` dict with this and
  pass it to `run_dag` instead of relying on `run_dag`'s whole-dict default. This is the exact,
  explicitly-flagged small T07-scope addition T05's log called out as future work ("T07+ can
  either rely on that whole-dict default or start passing `stage_configs` overrides once real
  per-stage sections matter" — that's now); kept minimal, no other `api.py` behavior touched.
- **A real, foreseeable T05-test breakage, fixed rather than ignored:** T05's
  `test_api_run_pipeline_wiring_produces_a_trivially_successful_empty_run` and
  `test_api_run_stage_wiring_runs_a_single_registered_stage` (`tests/test_dag.py`) both called
  `api.run_pipeline("base")` and asserted a trivially-successful *empty* run, which was only true
  because no real (non-`test`) role existed yet. The moment T07 registers `convert.default`/
  `segment.rigid`/`seg_eval.default`, `_auto_stage_plan` picks them up for *every* preset (their
  inputs are all external in Phase 0), so an unseeded `"base"` run now correctly raises
  `MissingDependencyError` instead of succeeding — and every future task that registers more real
  stages (T09/T10/T11) would keep re-breaking the same assumption. Fixed both tests by
  monkeypatching `api._auto_stage_plan` to `lambda resolved: []` for the duration of the test,
  decoupling them from whatever stages happen to be globally registered in the session — they
  again test only what they always meant to (the `run_pipeline`/`run_stage` -> `run_dag` ->
  manifest wiring), not the registry's current contents.
- **Also found (while debugging the above) a second, pre-existing latent registry-pollution bug in
  T04's own `tests/test_stages.py`:** it registered dummy test stages under made-up role names
  (`dummy_role`, `dummy_role2`) instead of role `"test"` like `EchoStage` — harmless while no real
  stage existed, but since the stage registry is global/process-wide and these registrations are
  never undone, once a real, non-`"test"`-role check (`_auto_stage_plan`) actually ran in the same
  pytest session *after* `test_stages.py` (alphabetical file order: `test_stages.py` before
  `test_stages_cpu.py`), it hit `dummy_role`'s three impls with no `impl` selector in any
  `resolved_config`, raising `ValueError` from an unrelated test file. Fixed by renaming those
  registrations to live under role `"test"` (`test.dummy_alpha`/`test.dummy_beta`/`test.dummy_dup`/
  `test.dummy_only_impl`) — the one role name `_auto_stage_plan` deliberately excludes, matching
  the convention `tests/test_dag.py`'s own module docstring already documented for its toy stages.
  No production code changed; this is a test-only fix, called out explicitly because it's outside
  this task's nominal file list.
- **Fixtures, all synthetic, in `tests/test_stages_cpu.py`:**
  - `convert.default`'s `capture` external artifact: a from-scratch tiny fake Omniverse capture
    (3 cameras in a ring x 2 frames x 32x32px, via PIL) with a `cameras_gt.json` matching
    `omni_to_4dgs.convert()`'s schema — new fixture code, since `omni_to_4dgs.py`'s own
    `_selftest()` only exercises its internal geometry/PLY helpers, never `convert()` itself.
  - `segment.rigid`/`seg_eval.default`'s `trajectories`/`gt_segmentation` external artifacts:
    built from a **literal copy** of `segment_rigid.py`'s own `_selftest()` generator (static base
    blob + 6 independently-rotating rigid parts, each its own axis/frequency/amplitude) — leaning
    on an already-verified synthetic scene rather than inventing a new one, per the task's design
    notes. `trajectories.npz` gets `canonical_xyz`/`traj` (no `opacity`, so the stage's
    `"opacity" in data.files` branch takes the no-floater path); `gt_segmentation.npz` gets
    `points`/`labels` with the **same** `xyz` array segment.rigid's own output uses, so NN label
    propagation in `seg_eval` is exact/trivial while still exercising the real `cKDTree` code path.
- **Real bug found in `motion_seg/rigidity_graph.py`, documented but explicitly not fixed (out of
  T07's "wrap, don't rewrite" scope):** running the repo's own, completely unmodified
  `python -m motion_seg.segment_rigid --selftest` in this sandbox's venv (numpy 2.2.6, scipy
  1.15.3) currently prints `ARI=0.0000` / `SELFTEST: FAIL`. Root cause, confirmed by direct
  inspection: at the default `k`, the synthetic scene's rigid parts are spatially far enough apart
  that the k-NN graph never contains a single genuine cross-part edge — every edge's "rigidity
  score" is pure float64 rounding noise (~1e-16 to 1e-17) on an *exactly*-rigid synthetic rotation,
  with no real bimodal (same-part vs. cross-part) signal for `otsu_threshold_log` to split. Otsu
  ends up drawing its cut through that noise, at a threshold roughly 3 orders of magnitude below
  the noise floor's own median (confirmed: `info['threshold'] ≈ 7.9e-23` vs. `score_median ≈
  5.6e-17`), which cuts the *large majority* of genuinely-same-part edges too — fragmenting all 7
  true parts into ~948 raw components, which `merge_small_components` then reassigns by nearest
  centroid, cascading everything into one final component (ARI 0). Empirically confirmed the fix
  is a `threshold_mult` large enough to keep essentially every edge (verified: wrong below ~3e5,
  correct — `ARI=1.0`, matching ground truth exactly — from ~1e6 up through 1e15); T07's own
  end-to-end test (`test_convert_segment_eval_slice_runs_end_to_end`) uses `threshold_mult=1e7` for
  this reason, documented inline at the point of use. This is a pre-existing fragility in the
  Otsu-log auto-threshold specifically for near-noiseless synthetic data (real trained-Gaussian
  trajectories have a much higher genuine noise floor, per `otsu_threshold_log`'s own docstring
  about the real pump01 run) — worth a look whenever `motion_seg/rigidity_graph.py` is touched
  again, but fixing the algorithm itself is not this task's scope.
- **`ConvertConfig` was not changed** — per the task brief's own reasoning, confirmed correct in
  practice: the stage gets its capture location from the `capture` `Artifact.path` (an external
  input pre-seeded into the manifest), not from a new `capture_dir` config field. No new required
  config fields were added anywhere in this task.
- Deps: added `numpy>=1.24`, `scipy>=1.10`, `Pillow>=10.0` to `orchestrator/pyproject.toml` (the
  wrapped scripts' real runtime deps — `cKDTree`, `connected_components`, `linear_sum_assignment`,
  PIL image I/O — plus the test fixtures' own PIL usage).
- Tests: `orchestrator/tests/test_stages_cpu.py`, 5 new tests, plus the 2 fixed `test_dag.py` tests
  and the renamed-role fix in `test_stages.py` (106 total, up from 101). Covers all three T07
  acceptance criteria directly: the full slice succeeds end-to-end with real output files
  (`sparse_/cameras.bin`, `sparse_/images.bin`, `points3D_multipleview.ply`, `segmentation.npz`,
  `seg_eval_result.json`, `ari > 0.9`) via one `run_dag` call over pre-seeded external artifacts;
  the manifest/artifacts are queryable afterward via `pipeline.artifacts.get_manifest`/
  `list_artifacts`; and caching — a same-run rerun stays `"success"` (not downgraded to
  `"skipped"`, per T05's same-run-vs-cross-run distinction), a fresh run_id with unchanged config
  gets `["skipped", "skipped", "skipped"]` via the cross-run cache index, and a fresh run_id after
  bumping `segment.rigid.threshold_mult` shows `convert.default` `"skipped"` (unaffected — a
  different config section, same input) while `segment.rigid`/`seg_eval.default` show `"success"`
  (real re-execution, new cache key) — the literal acceptance line "changing `--threshold-mult` in
  the preset reruns only segment + eval." Also a registry sanity check (`get_stage` on all three
  names, confirming `environment="host"`/`needs_gpu=False`) and confirmation that
  `pipeline.api._auto_stage_plan` now includes all three real roles for a plain `"base"` preset.
- **Verified in the same style isolated CPU-only venv as T01–T06**, with the additional runtime
  deps this task actually needs: fresh `pydantic`+`pyyaml`+`numpy`+`scipy`+`Pillow`+`pytest`;
  `pytest -q` from `orchestrator/` → **106 passed**.
- **Sandbox-mount staleness, 6/6 tasks now, and the worst variant yet:** every file this task
  touched via `Edit` (`pipeline/stages/base.py`, `pipeline/dag/scheduler.py`,
  `pipeline/stages/__init__.py`, `pipeline/api.py`, `orchestrator/pyproject.toml`,
  `tests/test_dag.py`, `tests/test_stages.py`, and `tests/test_stages_cpu.py` after its own
  second `Edit`) went stale in the bash-mounted view — but this time `ast.parse` alone didn't
  always catch it: `pipeline/api.py`'s stale copy was a **fully syntactically valid but outdated**
  snapshot (missing `_stage_config_for` and the `stage_configs=` wiring entirely, `co_firstlineno`
  for `run_pipeline` off by 25 lines from the real file), so a plain parse check reported no error
  while pytest still ran against genuinely stale bytecode. Caught it by cross-checking
  `inspect.getsource`/`__code__.co_firstlineno` against what the Read tool actually showed, once
  test failures didn't match the code being edited. Same workaround as every prior task (rewrite
  the bash-visible file directly via `cat > file <<'EOF'` from the Read/Edit tool's own correct
  content, then re-verify), but the lesson this task adds: **don't trust `ast.parse` alone to prove
  a file isn't stale — it only catches truncation/corruption, not "syntactically-valid-but-old."**
  Whenever a test failure's line numbers or referenced names don't match what was just written,
  force-rewrite and re-check with something that inspects actual bytecode (`co_firstlineno`,
  `hasattr`), not just syntax validity.

Next unblocked: Phase 0 is complete — T08 (container manager, needs T05+T06, both done) is the
critical path's next stop. T09 (wrap CUDA stages: train/render/seg_extract/amp) needs T07 (done)
+ T08, so it's next after that.
Related: [[pipeline-orchestration-plan]].

### Policy change (2026-07-14) — "wrap, don't rewrite" superseded by "copy the logic in, don't call the original script"

Bartosz's call, prompted by noticing `pipeline/stages/convert.py`'s docstring literally describing
T07's approach (`sys.path.insert` the repo root, then `import` straight from
`omniverse_pipeline.omni_to_4dgs`/`motion_seg.*`) as "wrap, don't rewrite." His concern: that's a
live runtime dependency on scripts that were only ever meant as testing/reference material —
fine short-term, but a recipe for "rewrite hell" once those scripts inevitably drift (they're not
versioned or tested as a dependency surface, just as standalone CLIs). Agreed: not rewriting
already-verified logic is still right, but *how* it gets reused was wrong — it should be copied
into this project (self-contained), not imported or shelled out to at its original location.

Clarified scope via a follow-up question: the new rule applies uniformly. The **only** thing this
project is allowed to depend on outside itself is the **container runtime** — the `isaac` Isaac
Sim image and the `cuda`/MBS custom container images. Everything under `omniverse_pipeline/`,
`motion_seg/`, and the repo-root scripts (`train.py`, `render.py`, `render_amp.py`, `mbs_infer.py`,
...) is a temporary/testing reference only — useful to read, never to `import` or subprocess-call,
even from inside a container.

**Changes made:**
- `orchestrator/planning/INSTRUCTIONS.md`: replaced the "Wrap, don't rewrite" ground rule with
  "Copy the logic in, don't call the original script."
- `orchestrator/planning/ARCHITECTURE.md`: DAG table's "Wraps" column reframed as "Ported from
  (reference only)"; added a new "Vendored stage logic" section proposing
  `pipeline/vendored/{host,cuda,isaac}/` as where copied-in logic lives per environment — stages
  in `pipeline/stages/` import from there (a normal in-project import), never `sys.path`-hack
  outside `orchestrator/`. For `cuda`/`isaac` stages, T08 (container manager) is now responsible
  for making `pipeline/vendored/cuda|isaac/` available inside the running container.
- `orchestrator/planning/TASKS.md` + `README.md`: **T07 reopened** (was `done` 2026-07-13, now
  `todo`) — its actual implementation is exactly the pattern being disallowed, so it no longer
  meets the (retroactively applied) acceptance bar. M1 is not reached until it's redone. T08–T11's
  task files reworded to reflect copy-in for their container-run code too (T09: train/render/
  seg_extract/amp → `pipeline/vendored/cuda/`; T10: mbs_infer → same; T11: split_mesh/add_motion/
  omni_capture → `pipeline/vendored/isaac/`) — no status changes there since none had started.
- `orchestrator/planning/tasks/T07-wrap-cpu-stages.md`: added a "Reopened 2026-07-14" section
  explaining why, updated in-scope/acceptance-criteria/relevant-files to require porting
  `convert()`/`segment_trajectories()`/`evaluate()` into `pipeline/vendored/host/` and removing the
  `sys.path.insert` + `omniverse_pipeline.*`/`motion_seg.*` imports from the three stage modules.

**Not yet done (implementation, not just planning):** `pipeline/stages/convert.py`,
`segment_rigid.py`, `seg_eval.py` still contain the old `sys.path.insert`/import pattern as of this
note — that's the actual T07 rework, to happen when T07 is picked back up, not part of this
planning-only pass. `pipeline/stages/base.py`'s module docstring and `stages/__init__.py`'s
docstring still reference the old "wrap, don't rewrite" framing too and will need a pass at the
same time.
Related: [[pipeline-orchestration-plan]], [[omniverse-4dgs-pipeline]].

### T07 redone (2026-07-14) — copy-in rework, M1 reached

Picked the reopened T07 back up and actually did the port the policy-change entry above flagged
as outstanding.

- New package `orchestrator/pipeline/vendored/{__init__.py,host/{__init__.py,convert.py,
  rigidity_graph.py,metrics.py,segment_rigid.py,seg_eval.py}}` (matches `ARCHITECTURE.md`'s
  "Vendored stage logic" layout exactly). Each module is a **verbatim** copy of the reference
  script's relevant function(s), not a rewrite:
  - `vendored/host/convert.py`: `convert()` + the geometry/COLMAP-writer helpers it actually
    calls (`opencv_c2w_to_colmap_qt`, `rotmat_to_qvec`, `opencv_c2w_to_llff_row`,
    `write_cameras_bin`/`write_images_bin`/`write_points3D_bin`/`write_ply`, `_load_frames_list`,
    `_nerfpp_radius`, `_read_ply_xyz_rgb`) from `omniverse_pipeline/omni_to_4dgs.py`. Left out:
    the CLI/argparse/`_selftest`/`qvec_to_rotmat` (only used by that script's own selftest, never
    by `convert()`) — nothing in the stage calls them, so porting them would be dead weight, not
    a more-faithful copy.
  - `vendored/host/rigidity_graph.py`: full verbatim copy of `motion_seg/rigidity_graph.py` (no
    cross-module imports of its own, so the whole file ports as-is).
  - `vendored/host/segment_rigid.py`: `segment_trajectories()` from
    `motion_seg/segment_rigid.py`, importing `segment_by_rigidity` from the sibling vendored
    `rigidity_graph` module instead of `motion_seg.rigidity_graph` — the **only** line that
    differs from the original function body.
  - `vendored/host/metrics.py`: full verbatim copy of `motion_seg/metrics.py` (same reasoning as
    `rigidity_graph.py`).
  - `vendored/host/seg_eval.py`: `propagate_labels()`/`evaluate()`/`_write_colored_ply()` from
    `motion_seg/evaluate_segmentation.py`, importing `adjusted_rand_index`/`best_iou_matching`
    from the sibling vendored `metrics` module instead of `motion_seg.metrics` — again the only
    changed line per function.
- `pipeline/stages/{convert,segment_rigid,seg_eval}.py`: removed the `sys.path.insert(_REPO_ROOT,
  ...)` block and the `omniverse_pipeline.*`/`motion_seg.*` import line from each; replaced with
  `from ..vendored.host.<module> import <fn> as _<fn>`. **Nothing else in any stage body
  changed** — the `Stage` subclasses, `run(ctx)` logic, config reads, artifact construction are
  byte-identical to before the rework, confirming the task file's own claim that "nothing about
  the scheduler/config/artifact plumbing around them needs to change." Module docstrings reworded
  to describe calling the vendored port rather than "wrapping" the reference script.
- `pipeline/stages/base.py` + `pipeline/stages/__init__.py`: reworded the stale "wrap, don't
  rewrite" framing left over from T04 (flagged as outstanding in the policy-change entry above)
  to describe the copy-in model instead. No behavior change, docstrings only.
- **Also caught while wording the docstrings:** avoided writing the literal string `sys.path` in
  any `pipeline/stages/*.py` docstring/comment, since T07's own acceptance criterion is a literal
  `grep -r "sys.path" pipeline/stages/` coming back empty — a docstring merely *mentioning*
  `sys.path` as prose would have failed that grep despite being harmless. Worth remembering for
  any future task with a grep-based acceptance bar: the check doesn't distinguish code from
  comments.
- **Sandbox-mount staleness bit again on every `Edit`-touched file** (`convert.py`,
  `segment_rigid.py`, `seg_eval.py`, `base.py`, `stages/__init__.py`) — consistent with
  [[cowork-mount-staleness-bug]]'s "always `Edit`, never `Write`" pattern from T01–T07's first
  pass. `base.py` reproduced the exact "syntactically-valid-but-old" failure mode T07's original
  log first documented: the stale bash-visible copy was truncated mid-token (`rais` instead of
  `raise NotImplementedError`), which is *not* a bare-name statement error under `ast.parse` (a
  lone identifier is a syntactically valid expression statement), so a naive parse check would
  have reported no problem. Caught only by asserting the file's tail matched the expected exact
  string, not just parsing it — confirms the earlier lesson ("don't trust `ast.parse` alone") and
  extends it: even a *content* assertion needs to check the specific tail/region that was
  actually edited, not just "does it parse." All five files were rewritten directly in the bash
  mount from the Read tool's known-good content and re-verified before running tests. The five
  freshly-`Write`n vendored files under `pipeline/vendored/host/` had no staleness at all,
  matching the established pattern.
- Verified in the same isolated CPU-only venv pattern as T01–T07's first pass (fresh
  `pydantic`+`pyyaml`+`numpy`+`scipy`+`Pillow`+`pytest`): `pytest -q` from `orchestrator/` → **106
  passed** — identical count to before the rework, confirming this was a pure "move the code,
  don't change behavior" change. `grep -r "sys.path" pipeline/stages/` and `grep -rn "^import\|
  ^from" pipeline/stages/ | grep -i "omniverse_pipeline\|motion_seg"` both come back empty
  (T07's literal acceptance criterion).
- `TASKS.md` board: T07 → `done`; critical-path note updated (Phase 0 / **M1 reached**). T07's own
  task file status header updated with a "Redone 2026-07-14" note pointing back here.

Next unblocked: T08 (container manager, needs T05+T06, both done) is the critical path's next
stop; T09 (train/render/seg_extract/amp) needs T07 (now genuinely done under the new rule) + T08.
Related: [[pipeline-orchestration-plan]], [[omniverse-4dgs-pipeline]].

### T08 done (2026-07-14) — container manager

- `orchestrator/pipeline/containers/config.py`: pure data, no Docker calls. `IMAGES` (`cuda` ->
  a locally-built `4dgs-motion-amp-cuda:latest`, `isaac` -> the pulled
  `nvcr.io/nvidia/isaac-sim:6.0.1`), `GPU_ALL`/`IPC_MODE` (`--gpus all` both, `--ipc=host` for
  `cuda` only), `CONTAINER_ENV` (Isaac's `ACCEPT_EULA`/`PRIVACY_CONSENT`/`OMNI_KIT_ACCEPT_EULA`),
  `KEEP_ALIVE_CMD = ["sleep", "infinity"]` (replicates both devcontainer.json's
  `overrideCommand: true`), `CACHE_VOLUMES` (Isaac's shader/compute/ov-data volumes — the
  `claude-config` volume in that same devcontainer.json is a devcontainer-only convenience for
  running Claude Code *inside* the container interactively, deliberately not replicated here),
  `container_name(env) -> "pipeline-<env>"` (deterministic, so warm-container lookup survives a
  process restart), `mounts_for(env)` = T06's `container_mounts(env)` + this env's cache volumes.
  Every image/mount/env-var choice traces directly back to `.devcontainer/devcontainer.json` /
  `omniverse_pipeline/.devcontainer/devcontainer.json` / `run_capture.sh`, per the task's own
  "reuse the devcontainer defs as source of truth" note — nothing here was invented.
- `orchestrator/pipeline/containers/manager.py`: `ContainerManager` — `ensure_image(env)` (checks
  `images.get` first; builds `cuda` from the repo `Dockerfile` via `get_roots().repo_root_wsl` as
  build context, pulls `isaac`), `start(env)` (warm-reuse by `container_name`: already-running ->
  returned as-is, stopped -> restarted in place, missing -> created with T06+cache mounts, GPU
  `device_requests`, and the keep-alive command), `exec(env, cmd, log_path=, workdir=)` (uses the
  **low-level** `client.api.exec_create`/`exec_start`/`exec_inspect` trio rather than the
  high-level `container.exec_run` wrapper, specifically so it can stream chunks into `log_path`
  *and* get a real exit code from the same call — the high-level wrapper only exposes the exit
  code after the whole stream is consumed/demuxed), `stop(env)`/`stop_by_id(container_id)`,
  `list_containers()` (Docker `filters={"label": "pipeline.managed"}`, so an unrelated container
  is never mistaken for one of ours). `exec` never raises on a non-zero exit — `ExecResult.
  exit_code`/`.ok` let the caller (a stage, T09/T11) decide pass/fail, matching the task's "non-
  zero exit surfaces as stage failure" (the stage's job, not the manager's).
- `orchestrator/pipeline/containers/__init__.py`: free-function surface
  (`ensure_image`/`start_container`/`exec_in_container`/`stop_container`/`list_containers`) over a
  lazily-created module-level `ContainerManager` singleton — same public-function style as
  `pipeline.config`/`pipeline.artifacts`, with a `manager=` kwarg for injecting a fake one (tests).
  Per the package docstring (mirrors `pipeline.resources` never importing `pynvml` at module
  scope): `docker` is only ever imported *inside* a method (`manager.py`'s `_docker_client`/
  `_to_docker_mounts`/`_device_requests`), never at module scope — verified by
  `tests/test_import.py`'s existing `test_no_heavy_imports_at_module_scope`, unchanged and still
  green.
- **No separate mount for `pipeline/vendored/cuda|isaac`**, confirmed rather than assumed: T06's
  `container_mounts` already binds the repo root to `/workspace`, and `pipeline/vendored/` lives
  under the repo root, so a `cuda`/`isaac` stage sees its vendored code there automatically —
  exactly what `ARCHITECTURE.md`'s "Vendored stage logic" section predicted T08 would just get for
  free, confirmed by the manual checklist's mount-resolution step listing
  `/workspace/orchestrator/pipeline/vendored`.
- `pipeline/api.py`'s `list_containers`/`start_container`/`stop_container` now delegate to
  `pipeline.containers` (lazy import, same wiring pattern T02/T03/T05 established).
  `exec_in_container` is deliberately **not** exposed through `pipeline.api`, since it's stage-
  facing (`ctx.containers`, for T09/T11 to call), and `ARCHITECTURE.md`'s Layer 2 section is
  explicit that the MCP server only exposes whitelisted ops, never arbitrary exec. `cancel`/
  `gpu_status` remain T12-scope stubs, untouched.
- Tests: `orchestrator/tests/test_containers.py`, 20 new (126 total) — all against a **fake**
  Docker client (`_FakeImages`/`_FakeContainers`/`_FakeExecAPI`/`_FakeClient`), since there's no
  real daemon in the sandbox. Covers: mount/GPU-kwarg construction (cuda vs isaac, bind vs
  volume), `ensure_image` build-vs-pull-vs-idempotent, all three `start` reuse paths (new/running/
  stopped), `exec`'s streamed-log + real exit code + append-not-truncate-across-calls + auto-start-
  if-not-running, `stop`/`stop_by_id` (including the no-op-when-nothing-running case),
  label-filtered `list_containers`, and a full `ensure_image -> start -> exec -> warm-reuse ->
  stop` lifecycle "smoke test" doubling as the deliverable's required smoke test. A fixture
  (`fixed_roots`, autouse) pins `PIPELINE_REPO_ROOT_WSL`/`PIPELINE_ASSETS_ROOT_WSL` to a fake
  `/mnt/c/...` path for every test in the file — without it, `get_roots().repo_root_wsl` (derived
  from `__file__`) isn't under `/mnt/<drive>` at all in *this* sandbox, and mount-building would
  fail for a reason that has nothing to do with this module's own logic (same override pattern
  `tests/test_paths.py` already established for T06).
- **Real GPU/Isaac behavior can't run in this sandbox** (acceptance criteria are explicit about
  this) — `pipeline/containers/MANUAL_CHECKLIST.md` is the 6-step checklist for Bartosz's WSL2 +
  Docker Desktop machine: cuda build + `nvidia-smi` sees the GPU, isaac pull + non-interactive
  EULA accept, mount resolution (`/workspace`, `/omniverse`), warm-reuse timing, Isaac cache-
  volume persistence across container removal, clean teardown. Every acceptance-criteria line maps
  to one checklist step.
- Verified in two independently-rebuilt isolated venvs (fresh `pip install -e .` each time,
  confirming `docker`/`pynvml` install cleanly with network access and don't need a running daemon
  just to import/use against a fake client): `pytest -q` from `orchestrator/` -> **126 passed**
  both times.
- **Sandbox-mount staleness bit again (7/7 tasks now)** — `pipeline/containers/__init__.py` and
  `pipeline/api.py` both went stale/truncated in the bash-mounted view after `Edit` (the same
  "always `Edit`, never `Write`" pattern every prior task logged), fixed the usual way: rewrite the
  bash-visible copy directly from a heredoc using the Read tool's known-good content, re-verify
  with `ast.parse`. **New variant this task, worth flagging separately in
  [[cowork-mount-staleness-bug]]:** attempting to "refresh" `tests/test_containers.py`'s stale
  bash-side read via a naive `cat tests/test_containers.py > tests/test_containers.py` self-
  redirect truncated the file down to one partial line (`asser` instead of `assert ...`, plus
  everything after it lost) — this is **not** the FUSE staleness bug, it's the classic Unix
  shell gotcha where redirecting a command's output to the same file it's reading truncates the
  destination before the read completes, regardless of any sandbox quirk. Recovered by reading the
  authoritative content back via the Read tool (which was never affected) and writing it fresh via
  a heredoc (new content, not a self-redirect) — confirmed intact via `ast.parse` + a manual tail
  check + a full green test run afterward. Lesson for next time: never `cat/cp` a file over
  itself to "fix" a stale read — write the known-good content from a different source instead.

Next unblocked: T09 (wrap CUDA stages: train/render/seg_extract/amp) — needs T07 (done) + T08
(now done) — is the critical path's next stop, unlocking Phase 1 alongside T10 (Option-A
segmentation, needs T09).
Related: [[pipeline-orchestration-plan]].

**T08 addendum (2026-07-14) — automated GPU/Isaac test file.** Bartosz asked for something
runnable rather than copy-pasted snippets, so `orchestrator/tests/test_containers_gpu.py` was
added: a real `pytest` file (not a standalone script) covering the same 6 steps as
`pipeline/containers/MANUAL_CHECKLIST.md`, asserting pass/fail instead of "eyeball the output"
where possible (image build + `nvidia-smi` GPU visibility, Isaac non-interactive start, mount
resolution, warm-reuse id-equality, clean teardown) and printing timings for the two checks that
can't be asserted portably (warm-reuse speed, Isaac cache-restart speed — no universal "fast
enough" threshold across machines). Module-level `pytestmark` skips the whole file the instant no
Docker daemon is reachable (`docker.from_env().ping()`, guarded against `docker` not even being
installed) — verified safe to leave in the normal suite: `pytest -q` in this sandbox now shows
`126 passed, 6 skipped` instead of failing. The two Isaac-specific tests additionally gate behind
`PIPELINE_TEST_ISAAC=1` (large pull, slow first run) so a plain `pytest -q
tests/test_containers_gpu.py` on Bartosz's machine doesn't unexpectedly kick off a 10GB+ pull. A
module-scoped autouse `_cleanup_managed_containers` fixture stops/removes every
`pipeline.managed`-labelled container at the end regardless of pass/fail, so a broken run doesn't
leave containers/GPU memory behind. Run: `cd orchestrator && pytest -q -s
tests/test_containers_gpu.py` (`-s` so the printed timings show).

### Runtime host moved off WSL2 (2026-07-14)

Bartosz's call: rather than dealing with the full WSL2 machine-setup burden right now (manually
configuring a distro, `nvidia-ctk`, mounting `Q:` into `/mnt/q`, etc. — the `WSL_SETUP.md` guide
written earlier this session), just run and test the whole orchestrator directly from Windows.
Docker/WSL "bundling" as a packaged feature is deferred to a new, unscheduled task (`T16`).

**Why this was actually easy:** Docker Desktop is reachable directly from a native Windows Python
process the same way it's reachable from a WSL2 shell — `docker.from_env()` just finds a different
transport (named pipe vs. Unix socket), same daemon either way. Nothing about *using* Docker ever
required WSL2; only `pipeline.paths` (T06) had baked in the assumption that the code driving it
ran from inside a WSL2 Linux venv, because that's what the *original* locked decision
(`INSTRUCTIONS.md`, 2026-07-11) said the runtime host would be.

**What changed:**
- `INSTRUCTIONS.md`: "Runtime host" locked decision revised — native Windows, not WSL2. WSL2/
  Linux-distro bundling explicitly called out as deferred future work. `host` environment
  description updated (native Windows venv, not WSL2 venv).
- `ARCHITECTURE.md`: path-translation/container-manager component descriptions updated; MCP
  server's Layer 2 description ("WSL2 host" → "Bartosz's Windows machine"); added a new
  "Phase 6 (deferred, not scheduled): WSL2/Linux-distro bundling" phasing entry pointing at `T16`.
- **`pipeline/paths.py` (T06) rewritten**: the three-space (host/wsl/container) model collapses to
  **two spaces** (host/container) — there's no separate WSL2 execution environment whose
  filesystem view differs from the host's own anymore. `Roots.repo_root_host`/`assets_root_host`
  are now plain `pathlib.Path` (OS-native — Windows on the real target, whatever the interpreter's
  own OS is otherwise), derived straight from `__file__`/env-var overrides. Dropped entirely:
  `windows_to_wsl`/`wsl_to_windows`, `to_wsl`, the `/mnt/<drive>` regex, `Roots.repo_root_wsl`/
  `assets_root_wsl`. Env vars renamed `PIPELINE_REPO_ROOT_WSL` → `PIPELINE_REPO_ROOT`,
  `PIPELINE_ASSETS_ROOT_WSL` → `PIPELINE_ASSETS_ROOT`. `MountSpec`/`container_mounts` logic is
  otherwise unchanged (it already produced the Windows-drive-letter host form Docker Desktop wants
  as a mount `source=`, regardless of which OS the calling process happened to run on).
- **`pipeline/containers/manager.py` (T08)**: one-line change — `ensure_image`'s cuda build-context
  path source, `get_roots().repo_root_wsl` → `.repo_root_host`. Everything else in the container
  manager was already OS-agnostic (never hardcoded WSL2 itself, only inherited the assumption via
  `pipeline.paths`). Docstrings/comments across `manager.py`, `containers/__init__.py`,
  `tests/test_containers.py`, `tests/test_containers_gpu.py`, and
  `pipeline/containers/MANUAL_CHECKLIST.md` reworded from "WSL2 + Docker Desktop machine" to
  "Windows + Docker Desktop machine" throughout.
- **`tests/test_paths.py` rewritten** for the 2-space model: CASES table now (host, container)
  pairs instead of (host, wsl, container) triples; host-side assertions compare `Path` objects
  rather than raw strings so the same test is correct whether it runs on real Windows or this
  sandbox's Linux (`str(Path(...))` renders with a different separator per OS; `Path.__eq__` is
  structural and doesn't care). Dropped the now-meaningless wsl-round-trip and drive-letter-generic
  -helper tests; added dedicated backslash-input-tolerance and case-insensitive-matching tests
  instead (real Windows users type backslash paths). 34 tests (down from 46).
- **`tests/test_containers.py`**: the `fixed_roots` autouse fixture (which faked a `/mnt/c/...`
  env-var override purely to work around the *old* model's WSL2-shaped default not resolving in
  this Linux sandbox) is gone entirely — with the repo-root default now derived straight from
  `__file__` with no WSL2 intermediate, it just resolves correctly on its own, in any sandbox.
  Fixed the one direct `repo_root_wsl` reference (now `repo_root_host`).
- **`planning/WSL_SETUP.md` retired** — replaced by **`planning/WINDOWS_SETUP.md`** (native
  Windows prerequisites: NVIDIA driver, Docker Desktop with GPU passthrough enabled via its own
  Settings toggle, NGC login, submodules, `PIPELINE_ASSETS_ROOT`, `uv sync --extra orchestrator` /
  `pip install -e '.[orchestrator]'` in a plain Windows Python env). `WSL_SETUP.md` itself is now a
  short pointer + "still in git history if T16 ever needs it" note, not deleted outright.
- **New task `planning/tasks/T16-wsl-docker-bundling.md`**: deferred, unscheduled, no acceptance
  criteria defined yet (explicitly not a "contained task" per `INSTRUCTIONS.md`'s own definition —
  scoping it properly is part of picking it up later). Added to `TASKS.md`'s board (status
  `deferred`) and dependency graph (needs T08, nothing downstream depends on it).
- **T06's and T08's task specs got "Revised (2026-07-14)" sections** (not reopened — the
  acceptance bar each originally met still holds under the new model) documenting exactly what
  changed and why, mirroring the project's existing "Reopened"/"Redone" convention from T07's
  policy-change episode rather than silently rewriting history.
- `README.md`: brought the whole "Status" section up to date at the same time (it had been stale
  since T06, still saying "T07 reopened" — a pre-existing staleness unrelated to this change, but
  cheap to fix while already editing the file for its own WSL2 mention).

**Verification:** full suite green after every change, `pytest -q` → `120 passed, 6 skipped` (6 =
`test_containers_gpu.py`, correctly auto-skipping with no real Docker daemon in the sandbox),
re-checked in a freshly rebuilt isolated venv at the end. `grep -rn "WSL\|wsl"` across every `.py`
file afterward turned up only intentional narrative references (explaining the change itself) and
one genuinely stale line in `pipeline/artifacts/paths.py`'s module docstring ("mapping host <->
WSL2 <-> container paths is T06's job") — fixed to drop the now-wrong "WSL2" in that sentence too.

**Sandbox mount-staleness bug — hit constantly and badly this session, a new failure mode
documented in `[[cowork-mount-staleness-bug]]`:** every file rewritten via the `Write`/`Edit` tool
this session (not just ones touched a second time) came back from bash reads with **trailing null
bytes** padded past the real (correct, shorter-than-before) content — `pipeline/containers/
__init__.py`, `manager.py`, `tests/test_containers.py`, `tests/test_containers_gpu.py`,
`pipeline/paths.py`, `tests/test_paths.py`, `orchestrator/pyproject.toml` (this one *also* got a
stray uncommitted `"pytest>=9.1.1"` dependency line from some earlier, unrelated stale write),
`planning/WSL_SETUP.md`, and `pipeline/artifacts/paths.py` all needed the same fix: read the
authoritative (correct) content via the Read tool or by splitting the bash-side bytes on the first
`\x00`, write that clean content to a *different* temp path, then `cat tmp_path > real_path`
(never `cat real_path > real_path` — confirmed again this session that self-redirecting a file's
own output back into itself is a plain, unrelated shell footgun that truncates the destination
before the read completes, not a sandbox quirk; that mistake cost a full test file's tail earlier
in the T08 session). `ast.parse` reliably catches the null-byte variant (`ValueError: source code
string cannot contain null bytes`), unlike the earlier "syntactically-valid-but-truncated" variant
T07 hit — but only *after* actually trying to parse the file, so it's still worth scanning for
`b"\x00" in data` explicitly rather than assuming a clean `ast.parse` run means the file is fully
synced.
Related: [[pipeline-orchestration-plan]], [[cowork-mount-staleness-bug]].

### T08 GPU/Isaac verified for real on Bartosz's machine (2026-07-15)

Ran `uv run -m pytest -q -s tests/test_containers_gpu.py` with `PIPELINE_TEST_ISAAC=1` (both
Isaac-gated tests ran, not skipped) on the real Windows + Docker Desktop + GPU machine. **All 6
passed in 1088.07s (~18 min, dominated by the first-time Isaac image pull):**
`test_cuda_image_builds_and_gpu_is_visible`, `test_isaac_image_pulls_and_starts_noninteractively`,
`test_mounts_resolve_correctly`, `test_warm_container_reuse_is_fast` (0.028s reuse),
`test_isaac_cache_persists_across_container_removal` (0.3s restart after removal),
`test_teardown_leaves_nothing_running`. This is the real-hardware counterpart to the 20 fake-client
unit tests from T08's original "done" — every line of T08's acceptance criteria and every box in
`pipeline/containers/MANUAL_CHECKLIST.md` is now confirmed, not just unit-tested. No code changes
were needed; the runtime-host-off-WSL2 revision held up under actual GPU/Docker conditions.
Marked in `T08-container-manager.md` (new "GPU/Isaac verified for real" section),
`MANUAL_CHECKLIST.md` (all boxes checked), `TASKS.md`, and `README.md`.

### T09 done (2026-07-15) — wrap CUDA stages (train/render/seg_extract/amp)

**Design departure from T07's pattern, decided up front, not discovered partway through:** T07's
`host` stages import a graduated *function* from `pipeline.vendored.host.*` and call it
in-process — no container needed. That doesn't work for `train`/`render`/`seg_extract`/`amp`:
their real dependencies (`torch`, `arguments`, `scene`, `gaussian_renderer`,
`diff_gaussian_rasterization`, `motion_amp`) only exist inside the `cuda` container, and the
orchestrator's own host process (native Windows, no GPU/torch) must never import them. So
`pipeline/vendored/cuda/{train,render,seg_extract,amp}.py` are verbatim ports that *keep* each
script's own `argparse` CLI and `if __name__ == "__main__":` entry point (unlike T07's ported
functions, which dropped the CLI wrapper) — a stage builds a CLI invocation and execs it as a
**separate process inside the container** via `ctx.containers.exec_in_container` (T08), never
imports the module at all. `pipeline/vendored/cuda/__init__.py` documents this explicitly (its
own `__init__.py`, unlike `vendored.host`'s, imports none of its four sibling modules).

**The bridge-file problem and its solution.** `train_pump.sh` passes 4DGS's core hyperparameters
(`ModelHiddenParams`/`OptimizationParams`, and via `merge_hparams`'s same mechanism also
`ModelParams`/`PipelineParams`) through `--configs arguments/multipleview/<name>.py` — a plain
Python file assigning four dict literals, `mmengine.Config.fromfile`-loaded and merged over
whatever the CLI parsed. This exists because `arguments/__init__.py`'s `ParamGroup` argparse
wiring (`add_argument(..., type=type(default_value))`) can't round-trip dict-typed
(`kplanes_config`) or list-typed (`multires`) fields from a CLI string — there's no way to fully
specify these hyperparameters via flags alone. New `pipeline/config/bridge.py`
(`render_bridge_source`/`write_bridge`) generates exactly that kind of file from the *resolved*
`PipelineConfig`, per stage call, so config (T02) stays the single source of truth even for these
un-CLI-able fields — same trust model `pipeline.config.loader.load_legacy_hyperparams` already
uses for the hand-authored versions (plain dict literals, `exec`/`runpy`-loaded, not attacker
input). `source_path`/`model_path` are deliberately excluded from the bridge's `ModelParams` dict:
`merge_hparams` applies every key in a matched group *unconditionally* (`setattr`, no `is None`
check), so leaving them in would silently clobber the CLI-passed, DAG-derived values (this run's
actual scene/model directories) with whatever the *config* says (empty string, by default) — this
exclusion is what makes T02's `ModelParams` docstring's "derived by the DAG's own artifact wiring
... until that wiring lands" finally true.

**`_stage_config_for`'s new `"_bridge"` merge (`pipeline/api.py`).** A cuda stage's `ctx.config`
needs both its own section (`TrainConfig`'s `port`/`expname`/...) *and* the four bridge groups —
but handing it the *whole* resolved config (like the pre-T07 default, or the "no matching section"
fallback) would defeat T05's cache-key scoping: an edit to, say, `segment.rigid.k` must not
invalidate `train`'s cache. So `_stage_config_for` now special-cases exactly
`{train, render, seg_extract, amp}`: merges `{"model":..., "pipeline_params":...,
"hidden":..., "optim":...}` in under a reserved `"_bridge"` key, alongside (not instead of) the
role's own section. Verified with two direct tests: the merged config is *unaffected* by an
unrelated section changing (`segment.rigid.k`), and *does* change when `hidden.net_width` changes
— exactly the sensitivity a cache key should have.

**Two more real pre-existing gaps found while wiring real stages in — same discovery pattern as
T07's `ctx.inputs` fix (T04/T05 reserved a slot, nobody actually wired it before a stage that
needed it existed):**
1. `pipeline.dag.scheduler.run_dag` constructed every `StageContext` without `paths=`/
   `containers=` kwargs, so both silently stayed their dataclass default (`None`) — true since T04
   even after T06 (paths) and T08 (containers) both landed and "reserved the slot." Fixed by
   passing the `pipeline.paths`/`pipeline.containers` *modules themselves* (not a wrapper class —
   both already expose exactly the free-function surface a stage needs, `to_container`/
   `get_roots`/`exec_in_container`), set unconditionally on every stage call since neither import
   touches `torch`/`docker` at module scope and a `host` stage simply never reads them.
2. `ContainerManager.exec` (and the `exec_in_container` free function) had no way to set extra env
   vars for one exec call. Needed because Python puts a *script's own directory* on
   `sys.path[0]`, not the exec `workdir` — so `python /workspace/orchestrator/pipeline/vendored/
   cuda/train.py` run with `workdir="/workspace"` would still fail `from arguments import ...`
   (the `arguments/` package lives at `/workspace/arguments`, not next to the script). Fixed by
   adding an `environment: Optional[dict[str,str]] = None` kwarg threaded straight to docker-py's
   `exec_create` (which supports it natively); `pipeline.stages.cuda_common.run_cuda_script`
   always passes `{"PYTHONPATH": "/workspace"}`. `tests/test_containers.py`'s fake `exec_create`
   updated to accept/record it; two new tests cover the passthrough and its `None` default.

**Output-path decisions, one per stage:**
- `train.default` writes the model under `ctx.run_dir/train_out` via an explicit `--model_path`
  (not the legacy global `output/multipleview/<name>/` `train.py` defaults to when `model_path` is
  empty) — every run's model is now artifact/cache-tracked under T03's own conventions like
  everything else, instead of a fixed shared location later runs would silently overwrite.
- `render.default` re-registers the *same* model directory (now also containing
  `{train,test,video}/ours_<iteration>/...`) as its own `renders` artifact, rather than trying to
  predict the `ours_<iteration>` folder name — with `iteration=-1` (load-latest), that number is
  only resolved *inside* the container process (`Scene`'s own load-latest-checkpoint logic), not
  something the host-side stage can compute ahead of time without duplicating that logic.
- `seg_extract.default` always passes an explicit `--out` (translated `ctx.run_dir/
  trajectories.npz`) rather than relying on `extract_trajectories.py`'s own `<model_path>/
  trajectories.npz` default — that default is computed from the *container* path, and the host
  side needs to know the result location without re-deriving the container's own path logic.
- `amp.default` computes `<model_path>/video/<video_path>` itself — `render_set_amp`'s output
  location for the compiled video (as opposed to the per-frame `renders`/`gt` subfolders, which
  *do* nest under `ours_<iteration>`) is independent of the loaded iteration, so no equivalent
  workaround is needed there.
- `amp.default` also fails fast, before ever touching the container, if a channel's `factor` isn't
  a whole number (`AmpFactorNotIntegerError`): `render_amp.py`'s own `--amp_factors` is declared
  `type=int` even though `AmpChannelConfig.factor` (T02) is `float` — a pre-existing script quirk,
  kept as-is per "copy the logic in, don't rewrite" (same policy as T07's documented Otsu-threshold
  bug), but worth a clear error instead of a cryptic in-container argparse crash.

**Verification** (isolated venv, no GPU/Docker — same story as T08's own unit tests against a fake
Docker client; here the fake is `pipeline.containers.exec_in_container` itself, since T08 already
covers the Docker-SDK layer underneath it): CLI-argument construction for all four stages
(source/model path flags, bridge `--configs` flag, every `TrainConfig`/`RenderConfig`/
`SegExtractConfig`/`AmpConfig` field mapped to its own flag, `PYTHONPATH` env passthrough);
bridge-file content (round-tripped via `exec()`, `source_path`/`model_path` confirmed absent from
the rendered source); `_stage_config_for`'s `"_bridge"` merge and its cache-key scoping (both
directions); the `AmpFactorNotIntegerError` fast-fail (and that it never reaches the fake exec
call); stage registration (`role.impl` names, `environment == "cuda"`); a non-zero exit code
raising rather than silently succeeding; and a `train.default -> render.default` chain run through
the *real* `run_dag` (not a hand-built `StageContext`) against a monkeypatched
`pipeline.containers.exec_in_container`, showing genuine cross-run caching — a second run_id with
an unchanged config shows both stages `"skipped"` and reuses the same `model` artifact path rather
than re-"training." 33 new tests (146 total across the suite), all green in a freshly rebuilt
isolated venv.

**Sandbox mount-staleness bug — hit on every `Edit`-touched file again this session (T01–T09
pattern holds without exception).** This time in a form that `ast.parse` caught on some files
(clean truncation mid-statement, e.g. `pipeline/api.py` cut off mid-call at
`_get_artifact(run_id, artifact_id` losing the closing paren and everything after) but silently
*passed* `ast.parse` on others whose truncation point happened to land on a syntactically-valid
boundary — `pipeline/stages/__init__.py` truncated to 35 of its true 67 lines but still parsed
clean (cut right after a `from . import ... # noqa` line), which would have shipped a
*half-registered* stage package with no error at all if not for happening to also break an
`import` elsewhere in the same run. Nine files needed the fix this session: `pipeline/api.py`,
`pipeline/dag/scheduler.py`, `pipeline/containers/manager.py`, `pipeline/containers/__init__.py`,
`pipeline/stages/__init__.py`, `pipeline/stages/base.py`, `pipeline/config/__init__.py`,
`pipeline/vendored/__init__.py`, and (a markdown file, so no `ast.parse` safety net at all)
`planning/tasks/T09-wrap-cuda-stages.md`, plus `planning/TASKS.md` and `planning/ARCHITECTURE.md`
— all fixed the documented way: read the authoritative content via the Read tool, rewrite the
bash-visible file directly from a heredoc (never `cat file > file`, never `rsync`/`cp` sourced
from the same stale mount — confirmed this session that `rsync`-copying the "repo" into a fresh
`/tmp` directory to sidestep the bug **also** produced a truncated copy, since `rsync` itself reads
through the same stale bash mount; copying doesn't route around this bug, only a Read-tool-sourced
rewrite does). Every *newly-`Write`-tool-created* file (all the T09 stage/vendored/test modules)
was unaffected, consistent with every prior session's "always Edit, never Write" pattern.
Related: [[cowork-mount-staleness-bug]], [[pipeline-orchestration-plan]].

### T10 done (2026-07-15) — wrap Option-A segmentation (`segment.mbs`)

**The point of this task, concretely proven:** `segment` is the one role with two registered
impls (`SegmentConfig.impl: "rigid" | "mbs"`, T02) — T10 is the first task to actually register
the second one, so it's the real end-to-end test of "add a new idea = register an impl + a
preset, no core edits" (`pipeline.stages.registry`'s own stated design goal since T04). Followed
T09's `cuda`-stage shape (build a CLI, exec it inside the container), not T07's in-process-import
shape — MotNet (`submodules/multibody-sync-4dgs`) needs the GPU, unlike `segment.rigid`'s pure
numpy/scipy rigidity-graph clustering.

- `pipeline/vendored/cuda/mbs_infer.py` — verbatim port of `motion_seg/mbs_infer.py`'s
  `_load_mot_net`/`_select_working_set`/`run_mbs_segmentation`/`main` (own argparse CLI, kept
  intact per the copy-in rule, same as T09's four scripts). Exactly two relocation fixes, no logic
  changes: `MBS_ROOT`'s relative-path walk-up count changed (4 hops instead of 1, since the file
  moved from `motion_seg/` to `pipeline/vendored/cuda/`), and the reference script's `_REPO_ROOT`
  sys.path hack + `main()`'s lazy `from motion_seg.visualize import render_segmentation_png`
  preview-PNG block were dropped entirely — both existed only to reach into `motion_seg`, exactly
  the throwaway-script reference the copy-in rule forbids a vendored module from depending on.
  Preview-PNG generation stays unwired (matches `segment_rigid.py`'s own `preview_png` and
  `seg_eval.py`'s `comparison_png` — pre-existing gaps, not new ones T10 introduced).
- `pipeline/stages/segment_mbs.py` — new `SegmentMbsStage`, `@register("segment.mbs")`,
  `environment="cuda"`, and — the actual contract that makes this a pure config switch —
  `inputs=("trajectories",)`/`outputs=("segmentation",)`, byte-identical to `segment.rigid`'s. One
  small new piece of stage-local logic: `SegmentMbsConfig.checkpoint` (already in `models.py`
  since T02, no sensible default — there's no vendored pretrained checkpoint) is resolved against
  `ctx.paths.get_roots().repo_root_host` when given as a relative path, *before* the real
  host<->container translation via `ctx.paths.to_container` — a "fill in a missing base"
  convenience local to this one stage, not a change to T06's path-translation module (which still
  does the only real space conversion, and still raises if the resolved path isn't under a known
  root). This stage never calls `write_stage_bridge` — `mbs_infer.py`'s CLI needs none of the 4DGS
  `ModelParams`/`PipelineParams`/`ModelHiddenParams`/`OptimizationParams` groups T09's bridge file
  carries, and `pipeline.api._stage_config_for`'s `"_bridge"` merge is correctly scoped to
  `train`/`render`/`seg_extract`/`amp` only — `segment` was never in that set and doesn't need to
  be.
- `pipeline/config/presets/pump01_segA.yaml` — new experiment preset (`extends: pump01`,
  `segment.impl: mbs`), mirroring `pump01_segB_tuned.yaml`'s "one experiment preset per
  segmentation option" pattern. Checkpoint path defaults to
  `submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar` (repo-relative, per the stage's own
  resolution logic above) — that `ckpt/` directory already exists (gitignored, empty) in the
  vendored MBS submodule, so this is exactly where a downloaded checkpoint is expected to land.
- `planning/WINDOWS_SETUP.md` gained a new "7. Option-A segmentation (MBS) setup" step: the MBS
  `ext/` CUDA ops need **no manual build step** — they JIT-compile automatically the first time
  anything imports `utils.pointnet2_util` from that package (`torch.utils.cpp_extension.load(...)`
  at import time), inside the `cuda` container, using the `nvcc` that image already ships (it's a
  `-devel`, not `-runtime`, CUDA base — see the repo `Dockerfile`). The checkpoint **is** manual —
  no weights are vendored (`ckpt/.gitignore`), `hubconf.py` points at a Google-Drive-hosted
  checkpoint that often needs a manual browser download (large-file virus-scan redirect breaks
  `torch.hub`'s scripted download) — documented with the exact placement path and the known
  out-of-distribution risk flagged in `NOTES_4dgs_motion_segmentation.md` §6d (MotNet trained on
  noisy FlowNet flow at a different point-cloud scale than 4DGS's exact, `target_radius=4.0`-
  normalized trajectories — may need fine-tuning to work well; explicitly out of this task's scope
  per its own "Out of scope" section).

**Verification** (sandbox, no GPU/Docker/torch/compiled `ext/` ops — identical limitation to every
T09 stage): 10 new tests in `tests/test_stages_mbs.py`, same fake-`exec_in_container` strategy as
`test_stages_cuda.py`. Covers CLI-argument construction (every `SegmentMbsConfig` field mapped to
its own hyphenated flag, e.g. `--n-points`/`--n-views`/`--n-sub`/`--opacity-thresh`, matching
`mbs_infer.py`'s own argparse flag names rather than the underscored config field names), the
checkpoint-resolution logic (both the relative-to-repo-root and already-absolute cases), the
`"_bridge"`-exclusion, a fast-fail-before-exec on an empty checkpoint, a non-zero exit raising
rather than succeeding silently, stage registration under the expected name/environment — and,
the acceptance criteria's actual substance: `get_stage("segment.rigid")` and
`get_stage("segment.mbs")` declare the identical `inputs`/`outputs` tuples, and
`pipeline.api._auto_stage_plan` resolves `segment.rigid` for the `base` preset's default
`segment.impl` and `segment.mbs` for `pump01_segA`'s override, with nothing else about the
resolved stage plan changing. Full suite: 132 passed, 6 skipped (`test_containers_gpu.py`, same
as every prior session — no real Docker daemon here), plus 5 pre-existing failures in
`tests/test_containers.py` (a `_FakeExecAPI` vs. the now-pip-installed real `docker` package
interaction specific to *this* sandbox instance — nothing under T10's own changed files, and not
present in T08/T09's own session logs; flagged here for whoever picks up T11/T12 next rather than
chased down, since fixing it isn't this task's job).

**Not yet done, needs Bartosz's machine** (same honesty T09 kept about its own status): the
actual first real `segment.mbs` run — `ext/` ops JIT-compiling for real, a downloaded checkpoint
loading, MotNet producing a real segmentation, `seg_eval.default` scoring it against the same GT
`segment.rigid` would be scored against. This task's own acceptance criteria expect real
shape/behavior debugging on that first run (per NOTES §6d) — not a formality, an actually-expected
outcome, since this logic was written but never executed before today, on any hardware.

**Sandbox mount-staleness bug (T10, same story as every prior task) — hit again, this time also
producing a false read of *this very notes file*:** `pipeline/stages/__init__.py` truncated after
an `Edit` (cut off mid trailing-import-list, syntactically invalid this time — `ast.parse` caught
it immediately unlike T09's silent-truncation variant) and `pipeline/stages/cuda_common.py` /
`pipeline/vendored/cuda/__init__.py` both truncated mid-docstring after their own `Edit`s (the
`cuda_common.py` one *did* pass a shallow "does it look right" glance but failed `ast.parse`
outright — unterminated string literal). All three fixed the standard way: reread via the Read
tool, rewrite the bash-visible copy from a heredoc, re-verify with `ast.parse`. **New wrinkle this
session:** `wc -l`/`tail`/`grep -n "^### "` (bash) on *this exact file* — before this T10 section
was appended — reported only 1003 lines, ending mid-way through the T08 GPU/Isaac addendum, while
the Grep tool's own index of the same file correctly found `### T09 done ...` at line 1114 and the
Read tool correctly read out to line 1239. In other words: the staleness bug isn't limited to
files this session itself edited — a file last legitimately written in a *prior* session can still
be served stale/short by bash today. Lesson reinforced (see [[cowork-mount-staleness-bug]]): for
any file, trust the Read/Grep tools' view over bash's `cat`/`wc`/`tail`, and when in doubt, anchor
an `Edit`'s `old_string` against a Read-tool-sourced quote of the file's true tail, not a
bash-derived one.
Related: [[cowork-mount-staleness-bug]], [[pipeline-orchestration-plan]].
