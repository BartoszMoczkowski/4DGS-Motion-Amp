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
