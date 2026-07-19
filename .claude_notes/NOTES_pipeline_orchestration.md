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

### T11 done (2026-07-16) — wrap Isaac stages (prep/capture front end)

**Goal reached (Milestone M3, per ARCHITECTURE.md's phasing):** a preset's auto-planned DAG now
runs truly end to end from a raw USD asset through amp — `prep_split.default -> prep_motion.default
-> capture.isaac -> convert.default -> ... -> amp.default`. `capture.isaac` produces the `capture`
artifact `convert.default` (T07) has declared as its external input since Phase 0, closing that
loop with zero changes to `convert.default` itself.

- `pipeline/vendored/isaac/{rig,split_mesh,add_motion,omni_capture}.py` — verbatim ports of
  `omniverse_pipeline/{rig,split_mesh,add_motion,omni_capture}.py`, same copy-in rule as T09/T10's
  `pipeline/vendored/cuda/*`. `rig.py` is pure numpy (no Isaac Sim dependency) so it's also
  exercised directly in the sandbox test suite (no container needed) — the one vendored-isaac
  module that's actually import-safe outside Isaac Sim.
- `pipeline/stages/isaac_common.py` — mirrors `cuda_common.py`'s shared CLI-exec plumbing, but
  targets `/isaac-sim/python.sh` (Isaac Sim's own bundled interpreter — the only place
  `pxr`/`omni.*` are importable) instead of the container's plain `python`. Re-exports
  `cuda_common`'s generic argparse flag builders rather than duplicating them (env-agnostic), and
  adds one new one, `star_list_flag`, for `add_motion.py`'s `--exclude` (`nargs="*"`, where an
  explicit zero-value invocation is meaningfully different from omitting the flag — `list_flag`
  (`nargs="+"`) can't express that). No `PYTHONPATH` fix needed here (unlike `cuda_common`'s
  `PYTHONPATH=/workspace`): `omni_capture.py`'s `import rig` is a same-directory import, already
  resolved via the script's own directory on `sys.path[0]`.
- `pipeline/stages/{prep_split,prep_motion,capture_isaac}.py` — three new stages, `environment=
  "isaac"`. `prep_split.default`/`prep_motion.default` are CPU-only (`needs_gpu=False`) despite
  running in the `isaac` container — no separate small-CPU image exists yet, adding one was ruled
  out of this "contained task"'s scope (new `Env` literal, new Dockerfile, new container config)
  for what would only save container-startup weight; see `pipeline.vendored.isaac`'s package
  docstring. `capture.isaac` genuinely needs the GPU (`needs_gpu=True`) for headless Isaac Sim
  rendering.

**Real design gap found and fixed while wiring this in (same "found while integrating" pattern as
T07's `ctx.inputs`, T09's `ctx.paths`/`ctx.containers`):** `ARCHITECTURE.md`'s original stage table
sketched these as `prep.split`/`prep.motion` — but `pipeline.stages.registry.register`'s
`"role.impl"` split takes everything before the *first* dot as the role, so those two names would
have collided into one ambiguous `"prep"` role with two impls (`split`/`motion`), which
`pipeline.api._auto_stage_plan` would then treat as needing a `resolved_config["prep"]["impl"]`
selector that doesn't exist — and `_stage_config_for` would fail to slice either stage's own
config section (there's no top-level `resolved_config["prep"]`, only `prep_split`/`prep_motion`).
Renamed to `prep_split.default`/`prep_motion.default` instead — each its own single-impl role,
matching its own top-level `PipelineConfig` section 1:1 (`capture.isaac` needed no such fix: role
`capture` already has its own section and no second impl to collide with). Updated
`ARCHITECTURE.md`'s stage table and every docstring that had already assumed the sketched names.

**`capture.isaac`'s config bridge:** unlike the `cuda` stages' single bridge-config-file pattern
(T09's `write_stage_bridge`), this stage writes a full `--config` YAML from `CaptureConfig` (T02)
— re-nesting `headless` back under `app` (the one place the pydantic schema flattens the original
YAML's structure) — then overrides `scene.usd_path`/`output.capture_dir` via `omni_capture.py`'s
own `--usd`/`--out` CLI flags (already built into that script) with the DAG's real
`animated_mesh` input / this run's own capture directory, the same "derive from the DAG's artifact
wiring, not the static config value" pattern `train.default` uses for `source_path`/`model_path`.
One subtlety: `CaptureFrameConfig.near`/`.far` are `Optional[float] = None`, and
`omni_capture.py`'s own `cap_cfg.get("near", radius*0.05)` would have that computed fallback
*shadowed* by an explicit `null` in the YAML (`dict.get` returns the stored `None`, not the
fallback, when the key is present) — so `_write_capture_config` drops those two keys entirely
when unset, rather than writing them as `null`.

**`ArtifactKind` gained `"usd"`** (`pipeline/artifacts/models.py`) — `prep_split.default`/
`prep_motion.default`'s `segmented_mesh`/`animated_mesh` outputs are single USD mesh files, and
none of the seven existing kinds fit a bare single-file mesh format.

**`pipeline.api.run_pipeline` gained an `external_artifacts` parameter** — a second real gap found
while wiring this in: previously `run_pipeline` had no way at all to satisfy a fresh auto-planned
run's *external* inputs (T05's `MissingDependencyError` check) before calling `run_dag`; every
caller that needed one had to bypass `run_pipeline` and call `run_dag` directly against a
hand-seeded manifest (`tests/test_stages_cpu.py`'s `_seed_run` helper). Harmless while
`convert`/`seg_eval` were the DAG's only external-input consumers and only tests ever exercised
this path — but now that `prep_split.default` needs an external `raw_mesh` (a real CAD asset with
no in-repo producer) for a preset's *auto-planned* full run to work at all, this became a real
caller-facing gap. Fixed by promoting the seed-then-`run_dag` sequence `_seed_run` already proved
out from a test helper into `run_pipeline` itself: `create_run` a fresh manifest, merge in
`external_artifacts` via `update_manifest` if given, then call `run_dag` as before.

**Verification** (sandbox, no Isaac Sim/GPU/Docker — identical limitation to every T09/T10 GPU
stage): `tests/test_stages_isaac.py`, 12 new tests, same fake-`exec_in_container` strategy as
`test_stages_cuda.py`/`test_stages_mbs.py`, plus two things those didn't need: (1) direct,
no-container unit tests of `pipeline.vendored.isaac.rig`'s pure-numpy camera math (orthonormal
c2w matrices, ring/dome dispatch); (2) a real `prep_split.default -> prep_motion.default ->
capture.isaac -> convert.default` chain through `run_dag`, where the fake `exec_in_container`
plays each vendored Isaac script's part just well enough (writing a placeholder `.usd` file for
split/motion, a full synthetic Omniverse-capture-shaped directory for `omni_capture.py`, reusing
`test_stages_cpu.py`'s own fixture shape) that `convert.default` — real, unmocked, in-process host
logic, unchanged since T07 — actually runs against the result and produces a real
`multipleview` scene directory. Confirmed cross-run caching end to end, and confirmed changing
`prep_split.group` re-runs `prep_split`/`prep_motion`/`capture.isaac` (their `usd`-kind,
single-file artifacts get real content hashes) but leaves `convert.default` cached — a real,
pre-existing T03/T05 cache-granularity limitation this chain exposes for the first time (nothing
upstream of `convert.default` was ever a *directory*-kind artifact before T11):
`pipeline.artifacts.hashing.hash_path` only ever hashes files, never a directory tree (documented
in that module's own docstring as "the caller's decision"), so `capture`'s `content_hash` stays
`None` regardless of what actually changed inside it, and `convert.default`'s cache key never sees
a difference. Not fixed here (out of scope, same "found, not fixed" precedent as T07/T09's own
notes) — documented in the test with a full explanation for whoever picks up T12+ next. Also
verified `pipeline.api.run_pipeline`'s new `external_artifacts` parameter end to end (seeding both
`raw_mesh` and `gt_segmentation` — the latter still required by `run_dag`'s external-input check
even when `only=` narrows *execution* to just the three Isaac stages, since that check runs over
the whole auto-planned DAG before `only` gets a chance to narrow anything).

Full suite independently re-verified in a fresh isolated venv (`pip install -e '.[orchestrator]'`
equivalent + `pytest -q`): **151 passed, 6 skipped** (`test_containers_gpu.py`, no real Docker
daemon here, as always) — 139 passed pre-T11, 12 new. Hit the sandbox mount-staleness bug on every
`Edit`-touched file again (6/6 this session: `pipeline/artifacts/models.py`,
`pipeline/config/models.py`, `pipeline/vendored/__init__.py`, `pipeline/vendored/isaac/__init__.py`,
and — a new variant — the freshly-`Write`-created `tests/test_stages_isaac.py` after a subsequent
`Edit` to fix two test bugs came back on bash with the edit entirely missing, not just truncated,
with a *matching* line count to the pre-edit version) — same rewrite-from-Read-tool-content-via-
heredoc workaround each time, `ast.parse`-verified after every rewrite. See
[[cowork-mount-staleness-bug]].

**Not yet done, needs Bartosz's machine** (same honest status every GPU-touching task since T08
has kept): the actual first real run — Isaac Sim opening a real USD, `split_mesh.py`/
`add_motion.py` needing `trimesh` manually `pip install`ed into the `isaac` container (not
preinstalled, unlike `pxr`/usd-core, which Isaac Sim's bundled interpreter already ships),
`omni_capture.py`'s full headless capture, and — the task's own acceptance criteria — reproducing
`run_capture.sh`'s `--n-cameras 2 --frames 2` smoke test via `run_stage`, then a full
`run_pipeline(preset="pump01")` from raw asset through amp. `planning/WINDOWS_SETUP.md` gained a
new "8. Isaac prep/capture stages setup" step for the `trimesh` install + a note that
`pump01.yaml`'s `capture.scene.usd_path`/`capture.output.capture_dir` are now just fallback values
— `capture.isaac` always overrides both from the DAG's real artifact wiring at runtime — plus
where to obtain the raw fused-mesh asset (`CONJUNTO_BOMBAS.usd`) to pre-seed as `raw_mesh`.

Next unblocked: T12 (resource manager, needs T09, already unblocked since T09) and T13 (MCP server,
needs T05, already unblocked since T05) are the only two `todo` tasks with no remaining
dependency; T14 needs T13+T09, T15 needs T09 — both also reachable now. T11 had no downstream
dependents in the graph itself (`TASKS.md`'s dependency graph shows T11 as a leaf off T08/T09), so
nothing else was unblocked by finishing it, but it does complete the "runs end-to-end from a raw
asset" milestone (M3) the whole subproject was ultimately for (problem #1 in
[[pipeline-orchestration-plan]]).
Related: [[cowork-mount-staleness-bug]], [[pipeline-orchestration-plan]].

### T11 addendum (2026-07-16) — real-hardware acceptance test added

Bartosz confirmed `test_containers_gpu.py` (T08) passes for real on his machine
(`PIPELINE_TEST_ISAAC=1`, 6 passed in 5.91s — fast because the `isaac`/`cuda` images and cache
volumes were already warm from T08's own 2026-07-15 real-hardware verification) and asked for
T11's own two acceptance criteria to get the same "real test behind a flag" treatment.

Added `tests/test_stages_isaac_gpu.py`, mirroring `test_containers_gpu.py`'s exact pattern (skip
if no reachable Docker daemon, skip unless `PIPELINE_TEST_ISAAC=1`, since every test here needs
the `isaac` image regardless): a `trimesh`-importability sanity check (fails fast with a pointer
to `WINDOWS_SETUP.md` step 8.1 instead of a confusing mid-script `ImportError`), criterion 1
(`run_capture.sh`'s `--n-cameras 2 --frames 2` smoke test reproduced via `run_stage` against the
already-existing animated pump asset), and criterion 2 (a full `prep_split.default ->
prep_motion.default -> capture.isaac -> convert.default -> train.default -> render.default ->
seg_extract.default -> segment.rigid -> amp.default` run from the raw fused mesh, deliberately
excluding `seg_eval.default` since it needs an unrelated external `gt_segmentation` artifact not
part of T11's own criteria, and deliberately trimming `n_cameras`/`num_frames`/`iterations`/
`coarse_iterations`/`n_times` down to a fast smoke pass rather than a real multi-hour
reconstruction). Both real-asset tests skip independently with a clear reason if their asset
(`PIPELINE_TEST_ANIMATED_MESH`/`PIPELINE_TEST_RAW_MESH`, each defaulting to the documented
`Q:/Omniverse/assets/pump_radnom/...` convention) isn't found on disk.

`planning/WINDOWS_SETUP.md`'s step 8 point 4 now points at this file instead of describing the
manual override inline. Sandbox-verified only as "collects and skips cleanly" (3 skipped, no
Docker/`PIPELINE_TEST_ISAAC` here) — full suite still 151 passed, 6 skipped as before (9 skipped
when `docker` happens to be pip-installed in the sandbox venv, since that satisfies the first
skip condition and falls through to the second). Not yet run for real; the trimmed
iteration/frame counts in criterion 2 are first-guess values, not validated against a real
training run.

### T11 real-hardware fixup (2026-07-16)

First real run of `tests/test_stages_isaac_gpu.py` on Bartosz's Windows + Docker Desktop + GPU
machine (via the direct file-mount, I read `capture.log`/`prep_split.log` straight out of the
run directories under the connected repo folder — no need for pasted output). Two real bugs found:

**1. `prep_split.default` failed for real: `ModuleNotFoundError: No module named 'pxr'`.**
`split_mesh.py`'s `load_geometry()` does a bare `from pxr import Usd, UsdGeom` with no
`SimulationApp` launch. `isaac_common.py`'s original docstring claimed `pxr` is "wired onto" Isaac
Sim's bundled interpreter's own `sys.path` — true for some earlier Isaac Sim release this
assumption was probably written against, false for the actual `nvcr.io/nvidia/isaac-sim:6.0.1`
image this project pulls. Proof it's a Kit-bootstrap issue, not a broken exec/mount/PYTHONPATH
problem: the same exec plumbing (`/isaac-sim/python.sh` via `docker exec`) *does* successfully
import `trimesh` (a real pip-installed package) in the sandbox's own
`test_trimesh_is_importable_in_the_isaac_container` check, and `omni_capture.py` — which launches
`isaacsim.SimulationApp` before ever touching `pxr` — successfully opened and traversed the real
USD stage in the very same log (`[capture] opening .../CONJUNTO_BOMBAS_animated.usd`). So `pxr`
really is only made importable by Kit's own extension loader running inside a live `SimulationApp`,
not a static PYTHONPATH `python.sh` sets up for free.

Fix: added `pipeline/stages/_isaac_kit_bootstrap.py` — new orchestrator glue, explicitly *not* a
vendored/ported copy (the copy-in rule is about `pipeline/vendored/isaac/*.py` staying untouched,
which it does). It launches a headless, do-nothing `SimulationApp`, then hands off to the real
target script's own `main()` via `runpy.run_path(..., run_name="__main__")`, with `sys.argv`
rewritten so the target sees exactly what it would if invoked directly. `isaac_common.py`'s
`run_isaac_script` now has a `NEEDS_KIT_BOOTSTRAP = frozenset({"split_mesh", "add_motion"})` set
and inserts the bootstrap into the `cmd` list ahead of the real script for those two keys only —
`omni_capture` is left alone since it already does this itself and must not be double-wrapped.

**2. `capture.isaac` reported manifest `"success"` but never wrote `cameras_gt.json`.** The real
`capture.log` showed, right at Kit startup: `PermissionError: [Errno 13] Permission denied:
'/isaac-sim/.cache/warp'` (from `omni.warp.core`'s kernel-cache init), which cascaded into
`omni.replicator.core-1.13.27` failing `startup_extension` entirely (`AttributeError: 'NoneType'
object has no attribute '_register_status_callback'` in the dependent `replicator_yaml`
extension), and later `omni_capture.py`'s own `rep.writers.get("BasicWriter")` raised
`WriterRegistryError: No writer with name 'BasicWriter' was found in registry` — a real, fatal
Python exception inside the script. But Kit's own shutdown path (`SimulationApp.close()` wasn't
called explicitly, "Shutting down automatically") apparently still yields exit code 0 from the
container's perspective, so `run_isaac_script`'s only signal (the process exit code) reported
success. This is the same "container exit 0 ≠ actually worked" caveat already flagged in this
task's original write-up, now confirmed for real.

Root cause of the permission error: Docker initializes a brand-new *named volume*'s content by
copying whatever the image already has at that mount path (`/isaac-sim/.cache`) — including its
ownership. `nvcr.io/nvidia/isaac-sim`'s own image-baked `/isaac-sim/.cache` ends up owned by a UID
that doesn't match whatever `docker exec`'s default user actually is on this real machine (no
`--user` override is set anywhere in `pipeline.containers`), so every write into that persisted
cache volume silently failed the first time it was ever exec'd into.

Fix: `ContainerManager.start()` now calls a new `_fixup_isaac_cache_permissions()` right after
creating a **fresh** `isaac` container only (not on reuse/restart) — a best-effort
`chmod -R 0777` (as `user="root"` on the exec) across the three cache-volume mount points
(`/isaac-sim/.cache`, `/isaac-sim/.nv/ComputeCache`, `/isaac-sim/.local/share/ov/data`). This
touches the volume's actual on-disk permissions, so it only strictly needs to happen once per
volume's lifetime — re-running it on every fresh-container creation is just cheap and simple,
not something that needs extra state to dedupe. Never raises (a failure here just means the
pre-existing cold-cache-every-time behavior, not a broken pipeline).

**Both fixes are backed by sandbox tests** (`test_containers.py`'s three new
`test_start_*_isaac_cache_permissions*`/`test_start_never_chmods_for_cuda`/
`test_start_does_not_rechmod_*` tests; `test_stages_isaac.py`'s cmd-shape assertions updated for
the bootstrap-wrapped `cmd` list), full suite still green (151 passed, 9 skipped — the 9 being
`test_containers_gpu.py`'s 6 + `test_stages_isaac_gpu.py`'s 3, both Docker/ISAAC-flag-gated).
**Neither fix has been verified against real Isaac Sim/GPU hardware** — I have no way to run Isaac
Sim myself; next step is Bartosz re-running `tests/test_stages_isaac_gpu.py` for real.

**Process note:** the repo folder connected to this session is the same folder the user's local
`pytest` run writes `runs/<run_id>/logs/*.log` into — I can (and should, going forward) read those
log files directly via the `Read` tool rather than asking the user to paste them.

### T11 cache-permission fixup didn't actually take (2026-07-16, later same day)

Bartosz re-ran `test_stages_isaac_gpu.py` twice more (`t11-capture-smoke-1784224663`/
`t11-full-smoke-1784224686` around 19:58, then `t11-capture-smoke-1784225823`/
`t11-full-smoke-1784225823` around 20:17-20:18) — **same `PermissionError: [Errno 13] Permission
denied: '/isaac-sim/.cache/warp'` and downstream `WriterRegistryError: No writer with name
'BasicWriter'` in both `capture.log`s**, i.e. yesterday's `_fixup_isaac_cache_permissions()` fix
(`pipeline/containers/manager.py`) has not actually fixed anything on the real machine yet.

Root cause, from re-reading `manager.py`'s own `start()`: the chmod fixup only runs in the
"create a brand-new container" branch (`if container is not None: ... return container.id` short-
circuits before it for a reused/restarted one). `containers/config.py`'s `container_name()` is
**deterministic** (`f"pipeline-{env}"` — always `pipeline-isaac`, no run-id in it), and the cache
volumes (`isaac-cache`/`isaac-compute`/`isaac-ovdata`) are named Docker volumes that outlive
container recreation by design. So: the `pipeline-isaac` container (and its cache volumes) already
existed on Bartosz's machine from *before* this fix was written, `start()` has been finding and
reusing that same container on every single pytest invocation since, and the fixup branch has
never once executed — the bad on-disk volume permissions from the very first cold run are still
there and will stay there indefinitely under a reused container.

**This is a real gap in the fix, not a "hasn't propagated yet" thing** — reuse-by-design (the
whole point of a warm container) directly defeats a fixup that only fires on fresh-create. Two
ways to unstick it, independent of each other:

1. **No code change, immediate**: on the host, `docker rm -f pipeline-isaac` (and optionally
   `docker volume rm isaac-cache isaac-compute isaac-ovdata` to also drop the cold-cache warmup,
   though that's not required — the chmod fixup doesn't need an empty volume, just a fresh
   container exec'd into it) so the next `start()` call takes the "create new" branch and the
   existing fixup actually runs.
2. **Code fix**: make `_fixup_isaac_cache_permissions` run on every `start()` call, not just
   fresh-create — it's already `chmod -R 0777`, idempotent, and the docstring itself already
   argues this is "cheap and simple" vs. tracking one-time state. This also protects against the
   silent-failure case (the `except Exception: pass` in the fixup swallows any exec error with no
   log), which may be what happened the first time it ran.

Not yet applied either fix — flagging for Bartosz/next session to pick one (or both: manual
`docker rm` now to unblock today's testing, code fix afterward so this can't recur silently).

### T11 third bug: cross-run cache poisoned by the bogus "success" (2026-07-16, same day)

Bartosz did the manual `docker rm -f pipeline-isaac` (option 1 above), re-ran the suite, and it
failed *fast* in the exact same shape (`convert.default`: `cameras_gt.json` not found) — meaning
Docker was never touched again. Checked the four newest runs' manifests
(`t11-full-smoke-1784228311/332/389/413`): `prep_split.default`/`prep_motion.default`/
`capture.isaac` all show `status: "skipped"`, still pointing at artifact paths under the very
first broken run, `t11-full-smoke-1784225823`.

Cause: `pipeline.dag.cache`'s cross-run index (`runs/.cache/index.json`) records a stage's outputs
under a `cache_key` (config + input hashes + code version) purely from the scheduler seeing
`status="success"` come back from `Stage.run()` — it has no way to know that `capture.isaac`'s
original "success" was actually a swallowed Kit crash (the bug just above). Once written, that
entry makes every future run with the same resolved config treat `capture.isaac` as fresh forever,
via `pipeline/dag/scheduler.py`'s `get_cached(cache_key, ...)` check — completely independent of
whatever's fixed in the container itself. The literal poisoned entries were sitting in
`runs/.cache/index.json`, one for the full-chain config (`n_cameras=3`/`num_frames=4`) and one for
the capture-smoke config (`n_cameras=2`/`num_frames=2`), each still pointing at
`t11-full-smoke-1784225823`/`t11-capture-smoke-1784224663`'s own (empty-except-`bg_dome.png`)
`capture/` directory.

**Both fixes applied this session** (Bartosz confirmed "yes do both"):

1. Deleted `runs/.cache/index.json` outright (needed `mcp__cowork__allow_cowork_file_delete` —
   the connected-folder delete-protection kicked in first). Safe: it's a pure cache, rebuilt from
   scratch as stages genuinely succeed; nothing reads it as a source of truth.
2. `pipeline/stages/capture_isaac.py`'s `CaptureIsaacStage.run()` now checks
   `capture_dir_host / "cameras_gt.json"` exists right after `run_isaac_script(...)` returns, and
   raises `IsaacStageError` (imported from `isaac_common`) if not — the same file
   `test_stages_isaac_gpu.py`'s own criterion-1 test already asserts on, and the last thing
   `omni_capture.py`'s `main()` writes, so its absence is the cheapest reliable proxy for "Kit
   silently died after startup." Raising here (rather than returning as if nothing happened) makes
   `pipeline/dag/scheduler.py`'s `except Exception` path mark the stage `"failed"` and return
   immediately — `put_cached` is only ever called *after* a stage's `run()` returns normally
   (see `run_dag`'s loop body), so a stage that fails this check can never poison the cross-run
   cache again. This directly plugs the hole the previous fixup-permissions fix didn't cover (that
   one only stopped the *permission* error from recurring; this one stops a *reported* success
   from ever being wrong regardless of root cause).
   - Two existing sandbox unit tests (`test_capture_isaac_stage_writes_config_yaml_and_overrides_
     usd_and_out`, `test_capture_isaac_stage_keeps_explicit_near_far_when_set` in
     `tests/test_stages_isaac.py`) used a bare `_FakeContainers.exec_in_container` that returned
     `exit_code=0` with no filesystem side effect at all — would now correctly fail this new check.
     Updated that shared fake to drop a stub `cameras_gt.json` under `--out` whenever the command
     is an `omni_capture.py` invocation, mirroring the same file's own (already-existing, more
     elaborate) `_fake_isaac_exec` helper used by the full-chain `run_dag` test.
   - Full suite verified green after the change: 151 passed, 9 skipped (unchanged) —
     `/tmp/t11venv/bin/pytest -q` from `orchestrator/`.

**Not yet done**: `_fixup_isaac_cache_permissions` still only runs on fresh container creation
(option 2 from the previous entry) — the manual `docker rm` unblocks *this* session, but the
underlying "reuse bypasses the fixup" gap in `pipeline/containers/manager.py` is still open if the
container/volume ever gets recreated again. Next real-hardware run (post `docker rm` + cache
delete + this code fix) is the one that actually proves `capture.isaac` for real — still unverified
against live Isaac Sim/GPU as of this note.

### T11 fourth bug: `cameras_gt.json` alone wasn't a strong enough success check (2026-07-16, same day)

Bartosz re-ran after the `docker rm` + cache-index delete + `cameras_gt.json` check. Real progress
this time — **the permission bug is genuinely gone**: `capture.log` now shows `Warp 1.13.0
initialized` cleanly (no more `/isaac-sim/.cache/warp` `PermissionError`), `omni.replicator.core`
starts without the `replicator_yaml`/`BasicWriter` cascade, and `omni_capture.py`'s `main()` runs
all the way through its own timeline-stepping loop and prints `[capture] done -> .../capture` with
no exception. But both tests still failed — a *new*, different failure:

- `capture.log` also shows `[Error] [omni.hydratexture.plugin] IHydraTexture refResource had no
  GPU foundation` and four (= `num_frames`) `[Warning] [omni.replicator.core.scripts.orchestrator]
  Timed out while waiting for pending Replicator writer schedules to drain` — the RTX render
  products never actually produced frame data, so `BasicWriter` never wrote a single `camNN/`
  directory. Looks like a renderer/GPU-passthrough problem (Docker Desktop + WSL2 GPU access, a
  Vulkan/EGL headless-rendering config issue, or similar) rather than anything in this repo's
  Python — flagging for Bartosz to check on the Docker/driver side; nothing here can diagnose that
  further without hardware access.
- Because `cameras_gt.json` and the point-cloud files (`points3D_gt.ply`/`points3D_labels.npy`/
  `label_names.json`) are written from pure USD stage geometry, not from rendering, they exist
  fine even when zero frames actually got rendered — so the previous fix's "does `cameras_gt.json`
  exist" check wasn't a strong enough proxy after all. `capture.isaac` got reported `"success"`
  again, `convert.default` then failed downstream with a `[WinError 3]` on the missing
  `.../capture/cam01` directory.

**Fix**: `CaptureIsaacStage.run()` (`pipeline/stages/capture_isaac.py`) now *also* checks that the
number of `camNN`-prefixed directories under the capture dir matches `rig.n_cameras` — the same
thing `convert.default` needs and `test_stages_isaac_gpu.py`'s own criterion-1 test already
asserts (`len(cam_dirs) == n_cameras`) — raising `IsaacStageError` if not, alongside the existing
`cameras_gt.json` check. Updated `tests/test_stages_isaac.py`'s shared `_FakeContainers` fake to
also create the right number of empty `camNN/` directories (reads `rig.n_cameras` back out of the
`--config` YAML the stage itself writes, so it stays correct regardless of what a given test sets
`n_cameras` to) — full suite re-verified green, 151 passed / 9 skipped. Also deleted
`runs/.cache/index.json` again (it had re-poisoned itself with this run's bogus "success").

**Still open / not fixable from here**: the actual RTX-rendering failure (`IHydraTexture ... no
GPU foundation` + writer-drain timeouts). Next real-hardware run will at least *fail loudly and
correctly* now instead of reporting a false success, which should make root-causing the renderer
issue itself easier.

### Root cause of the RTX-rendering failure: WSL2 doesn't support Vulkan (2026-07-16, same day)

Bartosz asked why GPU passthrough works fine for the `cuda` container but not `isaac`, given both
request GPU access identically in `pipeline/containers/config.py`/`manager.py` (same
`DeviceRequest(count=-1, capabilities=[["gpu"]])`, no isaac-specific capability being dropped
anywhere in this repo's code). Answer, confirmed via NVIDIA's own developer forum (searched
2026-07-16): **Vulkan is not supported under WSL2**, full stop — this is an NVIDIA-stated platform
limitation, not a Docker Desktop flag/config gap:

> "We currently do not support WSL2 and Xvfb. We require Vulkan on Linux." — NVIDIA staff, Isaac
> Sim forum ([thread](https://forums.developer.nvidia.com/t/isaac-sim-x86-64-headless-docker-wsl2-support/278252))
>
> "Vulkan is used by the Hydra Engine for RTX rendering in our Kit SDK. Currently, Vulkan is not
> supported on WSL." — same thread, follow-up

CUDA compute (`libcuda.so`, what the `cuda` container/PyTorch training needs) passes through
WSL2's GPU paravirtualization fine — that's the whole reason `test_containers_gpu.py`'s `cuda`
tests and this project's training/render/amp stages are expected to work. Isaac Sim's Hydra/RTX
renderer needs actual Vulkan on Linux for render-product/texture creation, which WSL2 doesn't
provide (or provides only partially/unreliably) — matches this run's specific symptom
(`IHydraTexture ... no GPU foundation`, RTX render products never producing frames) and a near-
identical harder failure from a few months back
([github.com/robotmcp/ros-mcp-server#289](https://github.com/robotmcp/ros-mcp-server/issues/289):
`VkResult: ERROR_INCOMPATIBLE_DRIVER` / `vkCreateInstance failed. Vulkan 1.1 is not supported` /
`GPU Foundation is not initialized` — Bartosz's run got further than this before failing, Kit/
extensions all started fine, but the underlying cause looks like the same WSL2-Vulkan gap).

**This is a hard architectural constraint, not a bug to keep chasing in `pipeline/containers` or
`capture_isaac.py`.** Docker Desktop on Windows runs Linux containers via a WSL2-backed VM by
default regardless of whether the orchestrating Python process itself runs natively on Windows or
inside WSL2 (see `manager.py`'s own "Revised 2026-07-14" docstring note) — so the container
running Isaac Sim is always subject to this limitation as currently set up. Options going forward,
not yet decided/actioned:

1. Run the `isaac` container against a real Linux host (bare metal, dual-boot, or a Linux VM with
   real — not WSL2-paravirtualized — GPU passthrough) — NVIDIA's actually-supported path.
2. Check whether Isaac Sim's native Windows install (outside Docker/WSL2 entirely) supports the
   render path `omni_capture.py` needs — would sidestep this whole layer for just `capture.isaac`.
3. Watch for NVIDIA adding WSL2 Vulkan support later — several forum threads on this are from
   2026, still open as of this note.

### Native-isaac fix confirmed working; new bug found one stage later — `cuda` image never had a real Python (2026-07-16/17)

Bartosz re-ran `tests/test_stages_isaac_gpu.py` after the native-execution change (option 2 above,
implemented same session — see the "adjust the project plan" entry). **Real validation: 2 of 3
tests passed**, including `test_capture_isaac_smoke_reproduces_run_capture_sh` — `capture.isaac`
genuinely worked against the native Isaac Sim install, and the full-chain test's manifest confirms
`capture.isaac: success` and `convert.default: success` for real (not skipped/cached). The
Vulkan/WSL2 fix is validated on real hardware.

The one failure was new and unrelated: `train.default` (a `cuda`-container stage, T09) failed with
`train exited with code 127`, log: `OCI runtime exec failed: ... exec: "python": executable file
not found in $PATH`. Root cause, found in the repo-root `Dockerfile`: the venv-creation step was
commented out —
```
# RUN uv venv .venv && \
#     uv sync --frozen
```
— right above the `ENV PATH="/workspace/.venv/bin:$PATH"` line that assumes it exists. So the
`4dgs-motion-amp-cuda:latest` image this project builds (via `ContainerManager.ensure_image`) has
never actually had a working Python interpreter on `PATH` at all — nothing installed `torch`/etc.,
and there's no bare `python`/`python3` symlink either (the base `nvidia/cuda:...-devel` image ships
no Python, and the Dockerfile only installs `python3-dev`, headers only). This is why `train`/
`render`/`seg_extract`/`amp`/`segment.mbs` (T09/T10) were all honestly logged as "not yet run for
real" — this is the *first* real execution of any `cuda`-container script, and it immediately
surfaced this. `pipeline.stages.cuda_common.run_cuda_script`'s hardcoded `cmd = ["python", ...]`
is correct *given a working venv* — not a bug in the orchestrator code, the Dockerfile just never
finished setting one up.

**Fix:** uncommented the two `RUN` lines in `Dockerfile`. `uv sync --frozen` (no `--extra` needed)
installs the plain `dependencies` group — `torch`/`torchvision`/`diff-gaussian-rasterization`/
`simple-knn`/etc. — exactly what the vendored `cuda` scripts need; the `orchestrator` extra is
deliberately never installed inside this container (it's pure-Python DAG code, meant to run on the
host, not the GPU image). `requires-python = "==3.12.12"` is an exact pin the Ubuntu-22.04 base
image doesn't ship; `uv venv .venv` (no explicit `--python`) will fetch a matching standalone
Python via uv's own toolchain management during the build, same as it would locally.

**Bartosz still needs to do a one-time manual step before re-running:** `ContainerManager.
ensure_image` only builds `cuda` if the image tag doesn't already exist locally
(`_image_present` check gates the whole `images.build(...)` call) — the exact same "reuse defeats a
fix" shape as the isaac cache-permission bug above. Since this test run already built and cached
the broken image *and* created a `pipeline-cuda` container from it, both need removing so the next
`ensure_image`/`start()` call actually rebuilds from the fixed Dockerfile:
```
docker rm -f pipeline-cuda
docker rmi 4dgs-motion-amp-cuda:latest
```
Not fixed in code this session (same as the isaac fixup's "runs only on fresh container creation"
gap) — worth a follow-up task if this class of bug (a fixed image/Dockerfile not actually getting
picked up without manual `docker rm`/`docker rmi`) keeps recurring.

### Follow-up: `ensure_image` now auto-detects a stale `cuda` image, no more manual `docker rm`/`rmi` (2026-07-18)

Bartosz asked for the follow-up fix flagged above. `pipeline/containers/manager.py`: added
`_cuda_build_hash(repo_root)` (sha256 of `Dockerfile` + `pyproject.toml` + `uv.lock`, in that
order, `b"<missing>"` fallback per file) and a `pipeline.cuda_build_hash` Docker label storing it.
`ensure_image("cuda")` now compares the *current* hash against the label on whatever image is
already present (`_cuda_image_up_to_date`) instead of just checking the tag exists, and rebuilds
automatically on any mismatch — the exact gap the previous entry's manual `docker rm -f
pipeline-cuda` / `docker rmi 4dgs-motion-amp-cuda:latest` was working around. `isaac` deliberately
keeps the old simple presence check — it's pulled by pinned tag from NGC, never built locally, so
there's no local content to hash.

Rebuilding the image alone isn't enough either, though — a *running* `pipeline-cuda` container
would still be backed by the old image's filesystem underneath it even after the tag gets
rebuilt. So `start()` also gained `_container_is_stale` (compares the found container's
`image.id` against the current image's id) and `_recreate_stale_container` (stop + remove); a
stale container is torn down and recreated fresh instead of being warm-reused, closing the second
half of the gap in one pass.

This is the third instance this task's seen of the same shape — reuse-by-design (warm containers,
cached images) silently defeating a one-time fixup — after the isaac cache-permission bug and the
cross-run cache-poisoning bug above. Unlike those two, this one's now handled generically enough
(hash-on-disk vs. hash-on-image, not a fixed one-off chmod) that it shouldn't need a fourth
instance to notice the pattern again.

Verified in the sandbox only (5 new tests in `tests/test_containers.py`, 160 passed/9 skipped
total, fake Docker client extended to track per-build image ids/labels) — not yet re-verified
against a real Docker daemon on Bartosz's machine, since the manual `docker rm`/`rmi` workaround it
replaces was only ever exercised there once, already worked, and there's no pending real-hardware
run blocked on this specifically. See `planning/tasks/T08-container-manager.md`'s matching
"Revised (2026-07-18)" entry.

### The staleness fix worked, but the rebuild it triggered failed with no diagnostic info (2026-07-18, same day)

Bartosz re-ran `test_stages_isaac_gpu.py` for real. Confirms the fix above is working exactly as
intended: `ensure_image` detected the stale label and rebuilt automatically, no manual `docker rm`/
`rmi` needed this time. But the rebuild itself failed: `train.default` errored with `"failed to
build '4dgs-motion-amp-cuda:latest': The command '/bin/sh -c uv venv .venv &&     uv sync
--frozen' returned a non-zero code: 1"`, after 1351s (~22.5 min) — a long enough runtime that it
plausibly got well into resolving/compiling something (likely `diff-gaussian-rasterization`/
`simple-knn`'s CUDA-extension builds, the slowest step in `uv sync` here) before failing. No
`runs/<id>/logs/` entry has the actual `uv sync` output — `docker.errors.BuildError`'s `str()` is
just that one generic line, and `ensure_image` wasn't capturing the fuller `.build_log` docker-py
attaches to the exception.

**Fixed in code:** added `ContainerManager._persist_cuda_build_log`, which pulls `exc.build_log`
(present on a real `BuildError`) and writes it to `runs/.cache/cuda_build.log`, with the path
appended to the raised `ImageNotAvailableError`'s message. This makes the *next* failure
diagnosable without a second 22-minute wait — it does **not** explain *today's* failure, since the
log from this run was never captured in the first place (the fix landed after the fact). 2 new
tests (162 total, 9 skipped) using a fake `BuildError`-alike with `.build_log` verify the log gets
written and referenced; a `build_log=None` case confirms the helper never masks the real error if
docker-py didn't attach one.

**Recommended next step for Bartosz, to get today's actual error without waiting again:** run
`docker build -f Dockerfile -t 4dgs-motion-amp-cuda:latest .` directly (from the repo root, in a
terminal on the real machine) — this streams the live `uv sync` output to the terminal
immediately, same build, much faster feedback than going through another full DAG test run.
One hypothesis worth checking in that output: the Dockerfile's base image is
`nvidia/cuda:12.4.1-devel-ubuntu22.04` (CUDA 12.4 toolkit/nvcc), but `pyproject.toml` pins torch to
the `pytorch-cu126` wheel index (CUDA 12.6) — `diff-gaussian-rasterization`/`simple-knn` are
`no-build-isolation-package`s compiled from source against whatever nvcc + torch combination ends
up installed, and a toolkit/wheel CUDA-version mismatch at that step is a plausible (not
confirmed) cause of a `uv sync --frozen` failure at exactly this stage. Not fixed — needs the real
build log to confirm before touching the pin.

### Root cause found (2026-07-18): `docker build` has no GPU, so torch's arch-detection crashes on an empty list — not the CUDA-version mismatch hypothesis

Bartosz ran the recommended manual `docker build -f Dockerfile -t 4dgs-motion-amp-cuda:latest .`
and got the real error immediately: `simple-knn`'s (and, by the same mechanism,
`diff-gaussian-rasterization`'s) editable-wheel build fails inside
`torch.utils.cpp_extension._get_cuda_arch_flags`:

```
IndexError: list index out of range
  arch_list[-1] += '+PTX'
```

preceded by the warning `The detected CUDA version (12.4) has a minor version mismatch with the
version that was used to compile PyTorch (12.6)` — which turned out to be a red herring, not the
actual cause (torch explicitly says "Most likely this shouldn't be a problem," and it wasn't). The
real mechanism: `docker build` (unlike `docker run --gpus all`) never has GPU passthrough — no
`nvidia-smi`, no visible CUDA device — so when `TORCH_CUDA_ARCH_LIST` isn't set,
`_get_cuda_arch_flags` falls back to querying `torch.cuda.device_count()`, gets `0`, builds an
empty `arch_list`, and then unconditionally does `arch_list[-1] += '+PTX'` on it — a crash on
*any* machine building these extensions inside a plain `docker build`, independent of the
12.4-vs-12.6 toolkit/wheel pairing. This had never been hit before because `train.default` (T09)
was the *first* real execution of anything needing these compiled extensions — T09/T10 were always
honestly logged as "not yet run for real" up to this point (see the T09/T10 "done" entries above).

**Fixed:** added `ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"` to the repo-root `Dockerfile`, right before
the `uv sync --frozen` step — `8.6` is the compute capability of Bartosz's actual GPU (confirmed
from Isaac Sim capture logs: `NVIDIA GeForce RTX 3090`, 24 GB, `sm_86`), `+PTX` keeps a little
forward-compatibility margin (JIT-compilable on a newer-but-compatible architecture) at negligible
extra build cost for two small extensions. This is a one-line, environment-only fix — no changes
needed to `diff-gaussian-rasterization`/`simple-knn` source or to `pipeline.stages.cuda_common`.

This Dockerfile edit is exactly the scenario the same-day `ensure_image` staleness fix
(`[[pipeline-orchestration-plan]]`'s "Auto-detect stale cuda image" entry) exists for: the next
`ensure_image("cuda")` call will see a changed `_cuda_build_hash` and rebuild automatically, no
manual `docker rm`/`rmi` needed this time. Not yet re-verified end-to-end — Bartosz still needs to
re-run `test_stages_isaac_gpu.py` (or the manual `docker build`) to confirm this specific fix
actually gets past the `simple-knn`/`diff-gaussian-rasterization` build step; `train`/`render`/
`seg_extract`/`amp`'s own logic is still genuinely untested against real hardware beyond this.

### Same error came back after the TORCH_CUDA_ARCH_LIST fix -- train.default failed with the exact "python not found" OCI error again (2026-07-18, same day)

Bartosz re-ran and got `OCI runtime exec failed: exec failed: unable to start container process:
exec: "python": executable file not found in $PATH` again -- the identical symptom as the very
first cuda bug (venv commented out), but this time reported *after* the image build itself (no
`"failed to build"` error surfaced), meaning `ensure_image`/`start()` returned normally and the
failure happened at `exec` time inside a running container. Two competing explanations, not yet
distinguished:

1. **Stale-container reuse**: this build attempt (with the `TORCH_CUDA_ARCH_LIST` fix) succeeded
   and produced a new, working image, but the warm `pipeline-cuda` container from an earlier,
   still-broken build never actually got recreated -- i.e. a bug in the same-day
   `_container_is_stale`/`_recreate_stale_container` fix meant to prevent exactly this.
2. **Still-broken venv**: the build itself completed (exit 0) but still didn't produce a working
   `python` binary/symlink in `.venv/bin` for some other reason (e.g. a `uv venv`/toolchain quirk),
   independent of container reuse entirely.

Checked real docker-py's source: `Container.image` is derived from the container's creation-time
`attrs['ImageID']` (fixed for the container's lifetime, re-read fresh via `container.reload()`)
rather than the current tag -- so `_container_is_stale`'s comparison logic checks out on
inspection. That's not the same as confirming it actually worked on the real daemon, though.

**Code hardening (not yet a root-cause fix):** `ensure_image` now persists the build log on a
*successful* build too, not just a failed one (previously only `_persist_cuda_build_log`'s
exception path wrote anything) -- a build exiting 0 doesn't guarantee the resulting image actually
works, and until now a "successful" build's log was invisible entirely. 162 tests total (no
existing test needed to change -- the fake `images.build()` already returned an empty log
generator on success; added an autouse fixture to `tests/test_containers.py` so this doesn't leave
a stray `runs/.cache/cuda_build.log` in the real working tree after every test run).

**Asked Bartosz to run, before any further rebuild** (cheap, seconds, no waiting):
```
docker images 4dgs-motion-amp-cuda:latest --format "{{.ID}} {{.CreatedAt}}"
docker inspect pipeline-cuda --format "{{.Image}}"
docker exec pipeline-cuda ls -la /workspace/.venv/bin/
```
If the image ID and the container's `.Image` don't match -> hypothesis 1 (our staleness-detection
code has a bug, needs fixing). If they match and `python`/`python3` genuinely aren't in
`.venv/bin/` -> hypothesis 2 (still a Dockerfile/uv problem, need `runs/.cache/cuda_build.log` from
the successful build to see what `uv sync` actually did). Not resolved as of this entry.

### Root cause found and fixed (2026-07-18, later same day): the venv was built inside `/workspace`, which gets bind-mounted (and shadowed) by the live host repo at container runtime

Bartosz's diagnostic output ruled out hypothesis 1 cleanly: `docker images` and
`docker inspect pipeline-cuda --format "{{.Image}}"` both reported the exact same id
(`853cc64aa964...`) -- the container really was running the just-rebuilt image, `[[pipeline-
orchestration-plan]]`'s `_container_is_stale` fix worked correctly. But `docker exec pipeline-cuda
ls -la /workspace/.venv/bin/` came back `No such file or directory` -- not "python missing", the
whole `.venv` directory doesn't exist inside the running container, despite the build completing
successfully.

The mechanism: `pipeline/containers/config.py`'s `mounts_for("cuda")` (T06's `container_mounts`)
bind-mounts the *entire* live host repo directory over `/workspace` at container **runtime** --
this is deliberate and load-bearing (it's how a stage sees the current `pipeline/vendored/cuda/
*.py` without an image rebuild every time source changes, per `ARCHITECTURE.md`'s "Vendored stage
logic"). But a Docker bind mount doesn't *merge* with the underlying image layer at that path --
it completely replaces it. The Dockerfile's `RUN uv venv .venv && uv sync --frozen` ran inside
`WORKDIR /workspace`, baking the venv into the image at `/workspace/.venv` -- which the runtime
bind mount then hides entirely, the instant the container starts. This explains the whole day's
saga in one shot: even a Dockerfile that builds perfectly was *always* going to produce a
container with no working `python`, because the build's own output was structurally unreachable
at runtime. The `diff-gaussian-rasterization`/`simple-knn` CUDA extensions have the exact same
exposure -- `[tool.uv.sources]` marks them `editable = true`, and an editable install of a
compiled extension builds its `.so` file in place inside the source tree (here,
`submodules/{depth-diff-gaussian-rasterization,simple-knn}/...`, also under `/workspace`) -- so
those would have been silently shadowed too, one stage further into the pipeline than we'd gotten
to yet.

**Fixed:** moved the entire build -- `WORKDIR`, the `COPY`s of `pyproject.toml`/`uv.lock`/
`submodules/`, `uv venv`/`uv sync --frozen`, and the resulting `PATH` -- to `/opt/build` instead of
`/workspace`, then `WORKDIR /workspace` again afterward (unaffected: `pipeline.stages.
cuda_common`'s `PYTHONPATH=/workspace`, and every `exec()` call's explicit `workdir=` -- those
correctly refer to the *live* repo mount, which is what they're supposed to see). `/opt/build`
is never touched by any bind mount, so the venv and the editable-installed extensions' compiled
`.so` files (whose absolute-path finder was baked in against `/opt/build/submodules/...` at build
time) both survive into the running container untouched. One-line-conceptually but
multi-line-in-practice change, confined entirely to the repo-root `Dockerfile` -- no changes to
`pipeline/containers/config.py`'s mounts (moving *those* off `/workspace` instead was considered
and rejected, since the live-mount behavior for `pipeline/vendored/cuda/*.py` is exactly what makes
code changes not require an image rebuild -- the bug was specifically in *what else* got built
into the same shadowed path, not in the mount itself).

This also means the Dockerfile fixed here is the *fourth* distinct real bug found across today's
single real-hardware run (venv commented out -> `TORCH_CUDA_ARCH_LIST` missing -> this
mount-shadowing structural issue), each one only surfacing once the previous one was fixed and the
build/run got one step further. Sandbox suite still green throughout (162 passed, 9 skipped) --
this fix is Dockerfile-only, nothing in `orchestrator/pipeline/` changed, so no new tests were
needed; the `ensure_image` staleness-hash mechanism (`[[pipeline-orchestration-plan]]`) will detect
this Dockerfile change and rebuild automatically on the next run, same as it did for the previous
two fixes today. Not yet verified for real -- Bartosz needs to re-run to confirm `train.default`
(and everything downstream: `render`/`seg_extract`/`amp`) actually completes now that both the venv
and the compiled extensions should genuinely be reachable at runtime.

### Fifth bug, same real-hardware attempt: `write_bridge` wrote the bridge file in Windows' cp1252 encoding, and `mmengine` choked reading it as UTF-8 inside the Linux container (2026-07-18)

With the venv/mount-shadowing fix in place, `train.default` finally got as far as actually running
`train.py` inside the container — and immediately hit a new, unrelated crash:
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97 in position 64`, raised from
`mmengine.Config.fromfile` reading the `--configs` bridge file `pipeline.config.bridge.write_bridge`
generates.

Root cause: `write_bridge` (`pipeline/config/bridge.py`) called `out_path.write_text(...)` with no
explicit `encoding=`. On Bartosz's native Windows process, `Path.write_text`'s default falls back
to `locale.getpreferredencoding(False)` — `cp1252` on his machine, not UTF-8. The bridge file's own
`_HEADER` constant had an em dash (`—`, U+2014) in its first comment line; cp1252 encodes that as
the single byte `0x97`. Position 64 in the file is exactly where that em dash sits in `_HEADER`'s
first line — confirmed by counting characters up to it. `mmengine.Config.fromfile` then reads the
file *inside the Linux `cuda` container*, where Python's default is UTF-8, and `0x97` alone is not
a valid UTF-8 start byte — hence the crash. This would have hit on the very first `train.default`
run regardless of any of today's other four bugs; it only remained hidden because nothing had
gotten far enough to actually read this file until now.

**Fixed:** `write_bridge` now writes with explicit `encoding="utf-8"`; `_HEADER`'s em dash was
also swapped for a plain ASCII `--` (belt-and-suspenders — the encoding fix alone is sufficient,
but there's no reason for a generated artifact to depend on a fancy dash rendering correctly).

**Audited the rest of `orchestrator/pipeline/` for the same bug class** (any `write_text`/
`read_text`/`open()`/`os.fdopen` call without an explicit `encoding=`, on a file written on Windows
and potentially read elsewhere, or vice versa) and fixed every hit found, since each one is a
"works until someone's config/scene name has a non-ASCII character, or until any comment in the
generated file does" landmine, exactly like this one:
- `pipeline/artifacts/manifest.py` — `_atomic_write_json` (writes `manifest.json`/
  `config_snapshot.json`) and `load_manifest`'s read, both now explicit UTF-8.
- `pipeline/dag/cache.py` — `_atomic_write_json` (writes the cross-run cache index) and
  `_load_index`'s read, both now explicit UTF-8.
- `pipeline/config/loader.py` — `load_legacy_capture_yaml`'s read, now explicit UTF-8.
  (`load_legacy_hyperparams` uses `runpy.run_path`, which always assumes UTF-8 for `.py` source
  per PEP 3120 regardless of OS locale — not vulnerable, left as-is.)
- `pipeline/config/resolver.py` — `_load_yaml` (reads preset YAML files), now explicit UTF-8.
- `pipeline/stages/capture_isaac.py` — the Isaac capture-config YAML write (lower risk in
  practice, since both the native-Windows writer and the native-Windows Isaac Sim reader share one
  machine/locale today, but fixed for consistency and in case a scene name ever has a non-ASCII
  character).
- `pipeline/stages/echo.py`/`seg_eval.py` — low-risk (currently all-ASCII content) but fixed for
  consistency with the rest.

Not touched (already binary mode or already had explicit encoding): `containers/manager.py`'s log
file handles, `isaac_common.py`'s log file handle, `artifacts/hashing.py`'s content-hash reads.

Verified: full suite still green (162 passed, 9 skipped) — none of these changes affect behavior
in the sandbox (all sandbox-side reads/writes happen to be pure-ASCII content today, which is
exactly why this bug class went undetected until real, non-ASCII content hit a real Windows
process). This is the *fifth* distinct real bug found across this one real-hardware attempt at
`train.default` (venv commented out -> `TORCH_CUDA_ARCH_LIST` missing -> `/workspace`
mount-shadowing -> this encoding bug), each only surfacing once the previous one was fixed and the
run got one step further. Not yet verified for real — Bartosz needs to re-run once more to confirm
`train.default` finally completes.

### Sixth bug, discovered on re-run: the native Isaac Sim python.bat default path was simply wrong (2026-07-18)

Bartosz re-ran the full suite and got two failures, both `capture.isaac`-related: `native Isaac
Sim python launcher not found at Q:\Omniverse\ISAAC_SIM\IsaacSim\tools\packman\python.bat`. This
is the default `pipeline.stages.isaac_common.DEFAULT_NATIVE_ISAAC_PYTHON` chosen back on
2026-07-16 (via `AskUserQuestion`, matching `omni_capture.py`'s own pre-orchestrator docstring
convention) — it simply doesn't exist on Bartosz's actual machine. He supplied the correct path
directly: `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat` (a versioned
standalone-package install, not the `ISAAC_SIM\IsaacSim\...` layout the original docstring
assumed).

**Fixed:** `DEFAULT_NATIVE_ISAAC_PYTHON` updated to the corrected path in
`pipeline/stages/isaac_common.py`; docs referencing the old path updated for consistency
(`INSTRUCTIONS.md` x2, `TASKS.md`, `WINDOWS_SETUP.md`, `T11-wrap-isaac-stages.md`). No test
depended on the literal default-path string, so the full suite stayed green (162 passed, 9
skipped) without any test changes. `PIPELINE_ISAAC_NATIVE_PYTHON` remains the escape hatch for a
differently-laid-out install, unchanged.

This is unrelated to the day's other five bugs (Dockerfile venv, `TORCH_CUDA_ARCH_LIST`,
`/workspace` mount-shadowing, the bridge-file encoding bug) — those were all `cuda`-container
issues; this is `capture.isaac`'s native-execution path, and it's simply a wrong hardcoded default,
not a design flaw. Not yet re-verified — Bartosz needs to re-run once more.

### Seventh bug, further re-run: `amp.py` still used the removed `mmcv.Config` API (2026-07-18)

With the Isaac path fixed, the `cuda`-container chain ran much further this time —
`train.default` completed for real (`train_out/cfg_args` written, model trained), `render.default`
started rendering — before crashing in `amp.default`: `AttributeError: module 'mmcv' has no
attribute 'Config'` at `pipeline/vendored/cuda/amp.py`'s `mmcv.Config.fromfile(args.configs)`.

Root cause: `train.py`/`render.py`/`seg_extract.py` all call `mmengine.Config.fromfile(...)` for
`--configs`, but `amp.py` (vendored from `render_amp.py`) still called `mmcv.Config.fromfile(...)`
— an inconsistency the module's own docstring had actually already *documented*, back when this
was vendored, as a "kept as-is, existing inconsistency in the reference scripts" quirk (grouped
with the genuinely-benign `--amp_factors: type=int` quirk from the same T09 pass). That
classification was wrong: this project's pinned `mmcv==2.2.0` (`pyproject.toml`) removed the
`Config` class entirely (relocated upstream to `mmengine.config`), so `mmcv.Config.fromfile` isn't
a preservable behavioral difference — it's a hard crash on every machine, unconditionally.
`amp.py` is the *last* stage in the `train → render → seg_extract → segment.rigid → seg_eval →
amp` chain, so it had simply never been executed for real until this run got far enough.

**Fixed:** both occurrences in `pipeline/vendored/cuda/amp.py` (lines ~673/705, one per
`if __name__ == "__main__":`-adjacent code path) swapped `mmcv.Config.fromfile` →
`mmengine.Config.fromfile`, matching the other three vendored scripts; module docstring corrected
to stop calling this a preserved quirk. 162 tests, all still passing (no sandbox test exercises
this code path directly — it's inside the vendored script's own runtime logic, only exercised via
a real `cuda`-container exec). This is the seventh distinct real bug surfaced by this same
real-hardware attempt across two consecutive sessions (Dockerfile venv → `TORCH_CUDA_ARCH_LIST` →
`/workspace` mount-shadowing → bridge-file encoding → wrong Isaac python path → this
`mmcv`/`mmengine` API mismatch), each only surfacing once the run got one stage further than
before — genuinely the deepest into the full pipeline chain it's gotten yet. Not yet re-verified —
Bartosz needs to re-run once more to see whether `amp.default` (and the full chain) completes.

### Eighth bug, same re-run: `train.py` never actually saved a checkpoint, because of an argument-merge ordering bug (2026-07-18)

Past the `mmcv`/`mmengine` fix, `amp.default` crashed differently: `mmengine.Config.fromfile`
parsed fine this time (`feature_dim: 32` printed), but then `Scene(...)`'s `searchForMaxIteration`
raised `FileNotFoundError: ... train_out/point_cloud` — no `point_cloud/` directory existed at
all. `train.default` itself had reported `exit_code=0` (success) and run for only ~32 seconds — far
too fast for real training, and with no error logged anywhere.

Traced via the run's own artifacts (`train_default_arguments_bridge.py`,
`arguments/__init__.py`): the bridge config set `OptimizationParams.iterations = 100` (a smoke-test
override), but the vendored `train.py`'s `__main__` block did
``args.save_iterations.append(args.iterations)`` *before* `merge_hparams(args, config)` applied
that override — at that point `args.iterations` was still argparse's own default, `30_000`
(`arguments/__init__.py`'s `OptimizationParams.__init__`). So `save_iterations` ended up
containing the reference script's stock milestones (`[14000, 20000, 30000, 45000, 60000]`) plus a
stray `30000` — never `100`, the run's *actual* final iteration once the config merge took effect.
Training genuinely ran its full 100 iterations (governed correctly by the merged `args.iterations`
everywhere else), but `if iteration in saving_iterations` never matched, so `scene.save(iteration)`
(which writes `point_cloud/iteration_<N>/point_cloud.ply`) was never called — a completely silent
failure mode, since `train.py` itself never treats "trained but never checkpointed" as an error.

This wasn't a bug in this project's own code at all — it's in the *reference* `train.py`'s own
logic, and it only ever mattered for this project's specific usage pattern: the reference script
assumes `--iterations` is passed directly on the CLI (already final at argparse-parse time,
matching the interactive `train_pump.sh` workflow this was ported from), whereas this project
routinely overrides `iterations` *after* parsing, via the `--configs` bridge file
(`pipeline.config.bridge`) — exactly the mechanism that makes config the single source of truth
(`INSTRUCTIONS.md`). The original script's assumption simply doesn't hold under that usage.

**Fixed:** moved `args.save_iterations.append(args.iterations)` to *after* the `--configs`
merge block in `pipeline/vendored/cuda/train.py`, so it always uses the true final iteration
count. 162 tests still pass (nothing in the sandbox exercises this vendored script's own runtime
logic — only reachable via a real `cuda`-container exec). This is the eighth distinct real bug
surfaced by this same real-hardware attempt, and arguably the most consequential one: a
silent-success failure mode that would have made `train.default` look done while quietly producing
an unusable model, on *every* run that overrides `iterations` via config (i.e. every normal use of
this orchestrator) — not just Bartosz's specific smoke-test preset. Not yet re-verified — pending
another re-run to confirm a real checkpoint gets written and `amp.default` completes.

### Ninth "bug," and the actual milestone: the full pipeline completed end to end for real (2026-07-19)

Bartosz closed some memory-heavy programs (freeing host RAM, consistent with the SIGKILL/137
diagnosis above being a host-RAM OOM, not VRAM) and re-ran. Result: `1 failed, 2 passed` —
but the "failure" turned out to be a bug in the *test's own assertion*, not the pipeline. Checked
the actual run's manifest (`runs/t11-full-smoke-1784412475/manifest.json`) directly: `amp.default`,
`render.default`, `seg_extract.default`, and `segment.rigid` all `"success"`;
`prep_split.default`/`prep_motion.default`/`capture.isaac`/`train.default` all `"skipped"` (served
correctly from the cross-run cache, since their inputs/config hadn't changed since an earlier
successful run — exactly what T05's caching is for). Overall `manifest.status == "success"`, and
`amp_video`'s artifact path (`.../train_out/video/render.mp4`) is a real file on disk, confirmed via
`find`. **This is the first time the entire capture → prep → train → render → seg_extract →
segment → amp chain has completed for real, producing a genuine amplified video output** — the
actual milestone this whole day-plus of real-hardware debugging (eight real bugs: Dockerfile venv,
`TORCH_CUDA_ARCH_LIST`, `/workspace` mount-shadowing, bridge-file encoding, wrong Isaac python
path, `mmcv`/`mmengine` mismatch, `save_iterations` ordering, plus this SIGKILL/memory episode)
was actually building toward.

The test itself only *looked* like it failed because `tests/test_stages_isaac_gpu.py`'s
`test_pump01_prep_through_amp_completes` asserted every stage's status `== "success"` literally,
which rejects `"skipped"` — even though the very next line up (`manifest.status == "success"`)
already treats a run with skipped-but-cached early stages as a successful run. This assertion had
simply never been exercised against a cache-hit scenario before, since every prior real-hardware
attempt failed at some earlier stage before ever getting far enough, on a fresh-enough run, to hit
this exact combination (later stages genuinely new work, earlier stages correctly cache-skipped).
Fixed by accepting both `"success"` and `"skipped"` in that per-stage loop. 163 tests still green
(this file's own tests auto-skip in the sandbox, `PIPELINE_TEST_ISAAC=1`-gated; syntax-checked and
the rest of the suite re-verified).

**Also hardened `train.default` itself against this failure mode recurring**, same principle as
`capture.isaac`'s `cameras_gt.json`/`camNN` check earlier this same task: `pipeline/stages/
train.py`'s `run()` now checks that `model_host / "point_cloud"` actually exists and is non-empty
after `run_cuda_script` returns, raising `CudaStageError` (not letting a bare exit-0 report success
and get cross-run cached) if not. Updated `tests/test_stages_cuda.py`'s two exec-fakes
(`_FakeContainers`, and the module-level `fake_exec` in the `train -> render` `run_dag` test) to
stub a `point_cloud/iteration_1/point_cloud.ply` on a simulated successful train call, and added
`test_train_stage_raises_if_exit_zero_but_no_checkpoint_written` as an explicit regression test.
163 tests total (up from 162), all green.

### T12 done (2026-07-19) — resource manager (VRAM/RAM + adaptive retry)

`orchestrator/pipeline/resources/` — the last Phase-3 task, unblocked since T09 — filled in for
real: `query.py` (VRAM via `pynvml` first, `nvidia-smi` CLI fallback; system RAM via `psutil`'s
`available` figure — all three imported lazily *inside* functions, never at module scope, same
"stays importable with no GPU/psutil installed" convention as `pipeline.containers.manager`'s
`docker` import; `tests/test_import.py`'s `test_no_heavy_imports_at_module_scope` now covers
`psutil` alongside `torch`/`docker`/`pynvml`), `gating.py` (`check_headroom`/
`InsufficientResourcesError` — the one gating hook T05's scheduler docstring always said T12 would
slot into, right before a stage runs; fails **open** whenever a dimension can't be measured, and
is a total no-op for a stage with the `ResourceRequest()` default), `adaptive.py` (pure "given this
much free memory, what should `low_vram_mode`/segmentation working-set/`rt_subframes` be" linear-
ramp calculations, floored/capped, never scaling *up* past what a preset already asked for),
`monitor.py` (`ResourceMonitor` — a background-thread poller measuring peak VRAM/RAM *above its
own start-time baseline* across one stage's execution, filling the `StageRecord.peak_vram_mb`/
`peak_ram_mb` fields T03 left nullable specifically for this task), and `oom_retry.py`
(`run_with_oom_retry` — catches a stage failure, checks `is_oom_error` by scanning the *log file*
the failing exception's own `log_path` attribute points at for CUDA-OOM marker text, and retries
exactly once with `reduced_memory_config`'s stage-specific reduced-memory fallback if one exists
for that stage name — `amp.*` forces `low_vram_mode`, `segment.mbs` halves its working-set/FPS-
subsample size, `capture.isaac` halves `rt_subframes`; `train`/`render`/`seg_extract` have no known
safe knob yet, so a real OOM there still re-raises immediately rather than guessing).

Wiring: `CudaStageError`/`IsaacStageError` (`pipeline.stages.cuda_common`/`isaac_common`) gained a
real `log_path` constructor attribute (previously only embedded in the message string) so
`is_oom_error` can read a failing stage's captured output directly, including `capture.isaac`'s own
two "exited 0 but didn't really work" checks (T11). `StageRecord` (`pipeline.artifacts.models`)
gained `oom_fallback: Optional[dict]` (backward-compatible, same pattern as T05's `cache_key`
addition); `record_stage_result` (`pipeline.artifacts.manifest`) gained matching
`peak_vram_mb`/`peak_ram_mb`/`oom_fallback` kwargs, each only written when not `None` (the
`cache_key` "only set if given" pattern, so a caller unaware of these fields never clobbers a
previous value). `pipeline.dag.scheduler.run_dag`'s per-stage loop — exactly where its own T05-era
docstring said this would land — now: (1) calls `check_headroom` right before a real (non-cached)
stage, recording a clean `"failed"` manifest entry with a clear message on
`InsufficientResourcesError` rather than crashing; (2) wraps execution in a `ResourceMonitor`;
(3) calls `run_with_oom_retry` instead of a bare `stage_cls().run(ctx)`, threading `oom_fallback`
through to `record_stage_result` on success. `pipeline.api.gpu_status()` now delegates to
`pipeline.resources.gpu_status()` (the one `api.py` stub actually in T12's scope — `cancel` stays
a stub, cancellation is explicitly out of scope for this task).

New dependency: `psutil>=5.9` added to `orchestrator/pyproject.toml` (cross-platform RAM query;
`pynvml`/`docker` were already T01-era dependencies, never actually installed/exercised for real
until now).

**A real sandbox-testing wrinkle, not a code bug**: this sandbox is a genuine (if small, ~4GB) VM
with its own incidental system RAM that has nothing to do with what a stage needs on Bartosz's
real machine — every T09/T10/T11 integration test that runs a `cuda`/`isaac` stage through
`run_dag` does so against a *fake* `exec_in_container` (no real GPU/Docker work ever happens), so
gating those tests against *this* sandbox's real ~3.5GB-free RAM would fail them for a reason with
zero bearing on the scheduler/stage logic under test — several did, the first time the full suite
ran with real gating wired in. Fixed with a new `tests/conftest.py` autouse fixture that forces
`pipeline.resources.query.query_gpu_memory`/`query_ram` to return `None` (the same "can't measure,
don't block" value real telemetry returns on a GPU-less machine) for every test by default;
`tests/test_resources.py` monkeypatches its own canned values back in per-test to actually exercise
gating/monitoring. This only worked cleanly because `gating.py`/`monitor.py` import
`pipeline.resources.query` *as a module* (`from . import query as _query`), mirroring
`pipeline.dag.scheduler`'s own `from .. import containers as _containers` convention, rather than
importing the two functions by name — a name-import would have bound a stale reference the
fixture's `monkeypatch.setattr` couldn't reach.

47 new tests (`tests/test_resources.py`'s 36 unit tests covering query/gating/adaptive/monitor/
oom_retry directly, plus 2 new integration tests appended to `tests/test_dag.py` proving the
gating-fails-cleanly and peak-mem+oom_fallback-recorded behaviors through a real `run_dag` call) —
210 total collected, 178 passed/skipped clean (169 passed, 9 skipped — the pre-existing real-GPU-
gated tests, unchanged). The other 32 (`tests/test_containers.py`, entirely pre-existing, untouched
by this task) error at fixture setup on a `PermissionError` unlinking the *real* repo's
`runs/.cache/cuda_build.log` — a genuine leftover file from Bartosz's actual 2026-07-16→19
real-hardware runs (confirmed via `runs/`'s own directory listing: dozens of real `t11-*-smoke-*`
run directories), not creatable/deletable from this sandbox session for the same reason
[[cowork-mount-staleness-bug]] documents elsewhere — unrelated to `pipeline.resources` or anything
this task touched; `test_containers.py`'s own `_clean_cuda_build_log` fixture (from T11,
2026-07-18) already deliberately targets the real repo root by its own docstring's design.

Real VRAM/RAM gating, peak-mem accuracy, and the OOM-retry's actual memory-reduction effectiveness
all still need verification on Bartosz's real machine — nothing here can be confirmed against a
genuine CUDA OOM or real GPU headroom from this sandbox. **T12 marked done** — resource-manager
scope (query/gate/adapt/monitor/retry, filling T03's nullable peak-mem fields) is complete and
sandbox-verified; only T13 (MCP server) and T15 (UI) remain un-started, plus the deferred T16.

### T03 bug found running the full suite for real on Windows (2026-07-19): `os.replace` can transiently deny a rename under concurrent readers

Bartosz ran the whole suite for real (33 min total — mostly `test_containers_gpu.py`'s real `cuda`
image build/GPU checks, expected and unrelated) and hit one genuine failure:
`test_concurrent_writes_never_produce_a_torn_file` (T03, `tests/test_artifacts.py` — many threads
hammering `save_manifest`/`load_manifest` against the same `manifest.json`) raised a raw
`PermissionError(13, 'Access is denied')` from a writer thread. Root cause: `_atomic_write_json`'s
`os.replace(tmp_name, path)` is atomic on both platforms, but Windows enforces mandatory file
locking — `MoveFileEx` (what `os.replace` uses there) can be transiently *denied* while another
thread has `path` open for reading, unlike POSIX, where a rename succeeds regardless of open file
handles. Invisible in the sandbox's Linux runs (every task through T11 exercised this code path
without ever hitting it) — only surfaced the first time this specific test ran on real Windows
hardware, same pattern as every other "sandbox-clean, Windows-only" bug this project has hit
(cp1252 encoding, `/workspace` mount-shadowing, etc.).

Fixed with `pipeline.artifacts.manifest._replace_with_retry`: retries `os.replace` up to 10 times
with a 20ms backoff on `PermissionError`, re-raising the last one if every attempt fails (a
genuinely stuck lock — antivirus, a leaked handle — should still surface as an error, not hang or
silently drop a write). POSIX essentially never raises `PermissionError` here, so the loop exits
on the first attempt there — a no-op in practice on Linux/the sandbox. Two new unit tests
(`test_replace_with_retry_recovers_from_transient_permission_error`/
`test_replace_with_retry_reraises_after_exhausting_attempts`, faking `os.replace` itself since the
sandbox can't reproduce the real Windows race) plus the original concurrency test, all green.

**Follow-up, same day: the retry alone wasn't enough — replaced with a per-path lock.** Bartosz
re-ran and the test still failed, now with a *mix* of raw `PermissionError`s (contention outlasting
the retry budget) and `ManifestCorruptError`s (the *reader* side — `load_manifest`'s existing
`except OSError` wraps a transient `PermissionError` from `path.read_text()` the same way, which
`_replace_with_retry` never covered at all) — under this test's real load (4 writers x 50
iterations racing 4 continuously-looping readers), a fixed retry count is inherently racy: there's
always some contention level high enough to exhaust it. Replaced the probabilistic fix with a
deterministic one: `_lock_for(path)` (a process-wide `dict[str, threading.Lock]`, one lock per
resolved manifest path) now guards `save_manifest`'s full write and `load_manifest`'s full read —
serializing this *process's own* concurrent access means no two of this module's own threads ever
have the same manifest path open at the same instant, so the Windows sharing-violation race simply
can't occur between them, regardless of load. `_replace_with_retry` stays (bumped to 25 attempts/
50ms) as a fallback for a handle held by something *outside* this process (antivirus, a second
orchestrator instance) that a Python-level lock can't see. New
`test_lock_for_is_identical_per_path_and_distinct_across_paths` unit-tests the registry itself
(same lock object for the same path, distinct objects across paths). 172 tests green in the
sandbox (up from 171). Still pending Bartosz's next full-suite run to confirm this actually holds
under the real Windows race that found both bugs.

Separately, the ~20-minute test around #30 is `test_containers_gpu.py::test_cuda_image_builds_and_
gpu_is_visible` doing a real `ensure_image("cuda")` build — expected and already documented (T11's
"~22-minute `uv sync --frozen`" finding from 2026-07-18); T12 didn't touch the repo-root
`Dockerfile`/`pyproject.toml`/`uv.lock` the build-hash check keys off of (only `orchestrator/
pyproject.toml`, a different file), so this wasn't a new rebuild trigger — just the known slow
first build (or a rebuild from some other real change on his machine since the last run).

**Follow-up, same day: `test_containers_gpu.py` gated behind an explicit `PIPELINE_TEST_GPU=1` opt-in, not just Docker reachability.** Bartosz asked for the slow real-build/GPU checks to be
kept out of a normal full-suite run. Root cause of why they weren't already: this file's only
gate was "is a Docker daemon reachable" — fine in the sandbox (never true there) but *always* true
on Bartosz's own machine, so a plain `pytest -q` over the whole repo silently ran real Docker/GPU
checks (including a potential ~20-minute `cuda` rebuild) as part of what should've been a fast,
everyday pass — exactly what T11's own `test_stages_isaac_gpu.py` already avoided by requiring
`PIPELINE_TEST_ISAAC=1` explicitly, a pattern this file hadn't adopted for itself. Fixed by adding
`_RUN_GPU = os.environ.get("PIPELINE_TEST_GPU") == "1"`, combined with the Docker-reachability
check into one short-circuited `_skip_reason()` function (not two independently-evaluated
`skipif` marks, which would still call `_docker_reachable()`'s real socket probe unconditionally
regardless of the flag) so a bare `pytest -q` never even attempts to reach Docker. Updated
`planning/WINDOWS_SETUP.md`'s step 6 and `pipeline/containers/MANUAL_CHECKLIST.md`'s invocation
instructions to include the new flag. Verified both skip paths in the sandbox: no flag -> instant
skip with "set PIPELINE_TEST_GPU=1..."; flag set but no reachable Docker (still true in the
sandbox) -> falls through to the pre-existing "no reachable Docker daemon" reason. Full suite
still 171 passed/9 skipped. `test_stages_isaac_gpu.py` needed no change — it already required
`PIPELINE_TEST_ISAAC=1` on its own.
