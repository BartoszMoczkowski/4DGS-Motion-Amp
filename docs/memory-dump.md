# Assistant memory dump (2026-07-19)

Verbatim dump of the assistant's persistent project memory. Excluded: `cowork-mount-staleness-bug` (a Claude Cowork sandbox tooling issue, not project content). Inline mentions of that sandbox bug inside the entries below are kept as historical context; `[[…]]` is memory cross-link syntax.

---

## Memory: omniverse-4dgs-pipeline

*State + design of the Omniverse→4DGS→MBS→motion-amp testing pipeline*

Thesis project "4DGS motion amp segmentation". First goal = a testing pipeline:
Omniverse/Isaac-Sim synthetic multi-cam datasets -> 4DGS -> MBS segmentation -> per-segment
motion amplification. Rationale: synthetic data gives exact camera poses AND ground-truth
segmentation labels (impossible for real scenes).

Built `omniverse_pipeline/` in the repo: `omni_capture.py` (Isaac Sim 5.1 headless multi-cam
Replicator capture; runs on user's GPU, not the sandbox), `rig.py` (tested camera-rig math),
`omni_to_4dgs.py` (tested pure-Python converter -> 4DGS `multipleview` format incl. COLMAP
binary sparse_ from GT poses), `capture_config.yaml`, `README.md`.

Pump de-fused + animated (2026-07-04): `split_mesh.py` split `CONJUNTO BOMBAS.usd` (single
fused mesh) into 107 connected-component parts (frame_base + 106) ->
`CONJUNTO_BOMBAS_segmented.usd` (labelled, colored). `add_motion.py` authored subtle periodic
per-part rigid motion (1-4mm trans, 0.5-3mm rot surface, integer cycles, pivots about part
centroid, frame_base static) -> `CONJUNTO_BOMBAS_animated.usd` (60 frames @24fps, peak surface
disp 1.75-6.7mm) + `_animated_motion_groups.json` (GT part->segment map). Both in
Q:\Omniverse\assets\pump_radnom\. Env gotcha: the Q: mount + Python .pyc caching serve stale
code/reads across edits — generate+verify to /tmp in one process then cp to Q:.

Key facts: Q:\Omniverse has Isaac Sim 5.1.0-rc.19 + KAT app (Kit 107.3). Target scene =
`assets/physAI_start/SceneAssembly.usd` (Z-up cm, animated CL6 PCB line + drum fan, 4 cams,
existing Replicator pipeline). The pump `CONJUNTO BOMBAS.usd` is a SINGLE merged mesh — not
segmentable as-is, needs splitting + authored motion. SceneAssembly has unresolved refs (S3
drum-fan payload, two metricsAssembler unit sublayers under N_02_PCB_Router).
4DGS multipleview loader: image names `image{N}.jpg` -> folder `cam{N:02d}`, shared
`cam_intrinsics[1]`, reads `sparse_/*.bin` (binary COLMAP).

Design decisions locked with user: headless capture; export GT poses+segmentation; 8-12
configurable cameras; static segmentation; rigid-machine targets; whole-clip periodic motion.
Full detail in repo `.claude_notes/NOTES_omniverse_pipeline.md`. Related: [[.claude_notes NOTES_4dgs_motion_segmentation]].

---

## Memory: pipeline-orchestration-plan

*Three-layer orchestration system (execution module + MCP server + UI) to fix pipeline effort/GPU-access/fragmentation — all 17 tasks done/handled as of 2026-07-19 except deferred T16 and open T17*

Plan (2026-07-11) to replace the fragmented chain of `.sh` scripts + scattered configs with a
three-layer system solving three long-standing problems: (1) running the whole pipeline is
manual, (2) Claude can't touch GPU-bound parts (Isaac/4DGS/MBS), (3) no consistent way to add
new ideas.

Decisions locked with user: Layer 1 = **custom lightweight DAG** package (`pipeline/`, stage
registry + typed artifacts + unified pydantic config presets + Docker container manager +
VRAM/RAM resource manager); runtime host = **WSL2** driving Docker Desktop; extensibility =
**plugin registry + config variants**. Layer 2 = MCP server on the WSL2 host wrapping Layer 1's
API (async runs, read manifests/logs/preview images, start/stop containers) — the one open
question is MCP transport/reachability from the Claude sandbox. Layer 3 = thin UI (Streamlit,
reusing `ampUI.py`), deprioritized.

Two container worlds it must orchestrate: isaac-sim:6.0.1 (capture + USD prep) and the repo
CUDA/PyTorch Dockerfile (train/render/seg_extract/amp).

MCP transport decided (2026-07-12): **HTTP** (streamable HTTP/SSE + auth), not stdio, for local-
or-remote flexibility. Subproject scaffolded at repo `orchestrator/`: `README.md`, `planning/`
{`INSTRUCTIONS.md` (locked decisions + working rules), `ARCHITECTURE.md` (source of truth),
`TASKS.md` (board + dep graph), `tasks/T01..T15-*.md` (15 contained task specs)}. Future code goes
in `orchestrator/{pipeline,mcp_server,ui}/`. Critical path T01→T04/T03→T05→T07→T08→T09.
Original plan: `.claude_notes/NOTES_pipeline_orchestration.md`.

**T01 done (2026-07-12):** `orchestrator/pipeline/` package scaffolded (stub submodules
config/stages/dag/artifacts/containers/resources + `api.py` typed stubs). Wired into build as a
new uv workspace member (own `orchestrator/pyproject.toml`, name `pipeline`), root `pyproject.toml`
exposes it via `[project.optional-dependencies] orchestrator = ["pipeline"]` — opt-in only, no
existing torch/CUDA deps touched. Test harness + pytest config under `orchestrator/tests/`.
Verified import-clean (no torch/docker/pynvml pulled in) in an isolated venv.

**T02 done (2026-07-12):** `orchestrator/pipeline/config/` — one pydantic schema (`models.py`,
all `extra="forbid"`) covering every stage (capture/convert/prep_split/prep_motion, the 4DGS core
param groups, train/render/seg_extract, segmentation, seg_eval, amp). Segmentation is the
role→impl example: `SegmentConfig.impl: "rigid"|"mbs"`, missing `mbs.checkpoint` fails fast. Amp
channels addressed by name (`channels: dict[str, AmpChannelConfig]`) instead of the old scripts'
positional lists; `AMP_METHOD_ALIASES` reconciles ampUI.py's Streamlit labels vs render_amp.py's
CLI method strings (they don't match today). Presets (`config/presets/*.yaml`) use an `extends:`
chain resolved by `resolver.py`: `base` (mirrors `arguments/multipleview/default.py`), `pump01`
(extends base; migrated from `capture_config_pump.yaml`+`pump01.py`+`train_pump.sh`, only real
diff from base is `optim.opacity_reset_interval: 60000`), `pump01_segB_tuned` (example
experiment). `loader.py` has reusable migration helpers so future scenes get migrated the same
way, not hand-typed. Full field-by-field mapping in `pipeline/config/MIGRATION.md`.

**T03 done (2026-07-13):** `orchestrator/pipeline/artifacts/` — pydantic `Artifact`
(name/kind[dataset|model|npz|ply|png|video|json]/path/producing_stage/metadata/content_hash) and
`RunManifest` (resolved_config as plain dict, git_sha, per-stage `StageRecord`, overall status).
Run layout `runs/<run_id>/{manifest.json, config_snapshot.json, logs/}`, default root
`REPO_ROOT/runs` overridable via `runs_root=` or `PIPELINE_RUNS_ROOT`. Manifest writes are atomic
(temp+fsync+`os.replace`); reads raise `ManifestCorruptError` on bad JSON/schema. Content hashing
defaults to a fast fingerprint (size+mtime+partial-sha256); `fast=False` opt-in for a real hash.

**T04 done (2026-07-13):** `orchestrator/pipeline/stages/` — `base.py` (`Stage` ABC with
`inputs`/`outputs`/`environment`(host|cuda|isaac)/`resources` + abstract `run(ctx)`),
`registry.py` (`@register("role.impl")` decorator + clear `StageNotFoundError`/
`DuplicateStageError`), `echo.py` (dummy stage that writes a real file).

**T05 done (2026-07-13):** `orchestrator/pipeline/dag/` — `graph.py` (Kahn topo-sort with
`CycleError`, `external_inputs`), `cache.py` (cache key = sha256 of resolved stage config +
sorted input-artifact hashes + code version; cross-run cache index at `runs/.cache/index.json`),
`scheduler.py` (`run_dag` with `from_stage`/`to_stage`/`only`/`force`; a failed stage stops
scheduling, descendants stay `"pending"`). `api.py` gained `_auto_stage_plan` (auto-discovers
registered roles, disambiguates multi-impl roles via `resolved_config[role]["impl"]`).

**T06 done (2026-07-13):** `orchestrator/pipeline/paths.py` — host↔container path mapping for two
roots (`repo`, `assets`); `container_mounts(env)` gives ready-made bind-mount specs matching the
existing devcontainer convention.

**T07 done → reopened → redone (2026-07-13/14), M1 reached.** Three real stages registered:
`convert.default`, `segment.rigid`, `seg_eval.default`. Policy change mid-way ("wrap, don't
rewrite" → **"copy the logic in, don't call the original script"**): `omniverse_pipeline/`,
`motion_seg/`, and repo-root scripts are reference-only — logic is vendored verbatim into
`pipeline/vendored/{host,cuda,isaac}/`, never imported/subprocessed; the only external dependency
allowed is the container runtime. Also fixed two real pre-existing gaps found while wiring:
`StageContext` never carried resolved input artifacts (added `ctx.inputs`), and `api.py` passed
every stage the whole config instead of its own section. Acceptance verified: full
convert→segment→eval slice end-to-end; unchanged rerun all-`"skipped"` via cross-run cache;
param bump re-runs only downstream stages.

**Runtime host revised (2026-07-14): dropped WSL2, runs natively from Windows.** Docker Desktop
is reachable identically either way; `paths.py` collapsed from 3 path spaces to 2;
`WINDOWS_SETUP.md` replaced `WSL_SETUP.md`; deferred task T16 (WSL/Docker bundling) added.

**T08 done (2026-07-14), GPU/Isaac-verified on real hardware (2026-07-15).** `pipeline/
containers/` — `config.py` (images, mounts, Isaac cache volumes, GPU/EULA env) + `manager.py`'s
`ContainerManager` (`ensure_image`/`start`/`exec`/`stop`; warm reuse by deterministic name
`pipeline-<env>`; exec streams logs and returns real exit codes). All 6 real-hardware checks
passed (cuda build + GPU passthrough, Isaac pull + EULA, mounts, warm reuse, cache persistence,
teardown).

**T09 done (2026-07-15):** `pipeline/vendored/cuda/{train,render,seg_extract,amp}.py` — verbatim
CLI ports executed inside the `cuda` container via `exec_in_container`. New `config/bridge.py`
writes a temp `arguments/multipleview/`-style file from the resolved config each stage call
(config stays the single source of truth); `source_path`/`model_path` passed as CLI flags.
Fixed two more gaps: `run_dag` never set `ctx.paths`/`ctx.containers`; `exec` gained
`environment=` to carry `PYTHONPATH=/workspace`.

**T10 done (2026-07-15):** `segment.mbs` — second impl behind the `segment` role, first real
proof of "add a new idea = register an impl + a preset, no core edits". Vendored port of
`motion_seg/mbs_infer.py`; new `pump01_segA.yaml` preset; checkpoint download is manual
(Google-Drive-hosted); known out-of-distribution risk documented.

**T11 done (2026-07-16), revised same day:** `prep_split.default`/`prep_motion.default`/
`capture.isaac`. Real runs surfaced: Kit-bootstrap wrapper needed for bare `pxr` imports; an
Isaac cache-volume permission bug; cross-run cache poisoned by a bogus "success" (fixed +
stronger post-hoc output checks); and finally the root cause of no-rendered-frames — **Vulkan
is not supported under WSL2** (NVIDIA-stated hard limitation), so `capture.isaac` was repointed
at the native Windows Isaac Sim install (`run_native_isaac_script`, subprocess against
`PIPELINE_ISAAC_NATIVE_PYTHON`). CPU-only prep stages stay containerized. Native fix confirmed
on real hardware 2026-07-16/17.

**The cuda-container bug chain (2026-07-17/18),** each surfacing only after the previous fix:
Dockerfile venv build commented out; `ensure_image` only checked tag existence (fixed:
build-hash label + auto-rebuild + stale-container recreation); failed builds had no diagnostics
(fixed: persist build log to `runs/.cache/cuda_build.log`); `TORCH_CUDA_ARCH_LIST` unset —
`docker build` has no GPU so torch arch-detection crashes on an empty list (fixed:
`ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"`, RTX 3090); the venv was baked into `/workspace`, which the
runtime bind mount shadows (fixed: build moved to `/opt/build`); `write_bridge` wrote cp1252 that
the Linux container's UTF-8 reader choked on (fixed + encoding audit across the package); wrong
hardcoded native Isaac python path (real path:
`Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat`); vendored `amp.py` still
used the removed `mmcv.Config` API (→ `mmengine.Config`); and `train.py` silently never saved a
checkpoint — `save_iterations.append(args.iterations)` ran before the `--configs` merge, so
overridden `iterations` never checkpointed (fixed + `TrainStage` now verifies a non-empty
`point_cloud/` before reporting success).

**MILESTONE (2026-07-19): the full pipeline completed end-to-end on real hardware for the first
time** — `prep_split → prep_motion → capture.isaac → convert → train → render → seg_extract →
segment.rigid → amp`, first four stages correctly cache-served on the rerun, real amplified
`render.mp4` on disk. (An interim `amp.default` exit-137 SIGKILL was diagnosed as host-RAM OOM,
resolved by freeing memory.) Eleven distinct real bugs across the 2026-07-16→07-19 saga, each
found only because the previous fix let the run get one step further.

**T13 done (2026-07-19):** `orchestrator/mcp_server/` — HTTP MCP server skeleton: bearer-token
auth (no default token, fails fast), plain-ASGI auth gate (not `BaseHTTPMiddleware`, which breaks
SSE streaming), `gpu_status` as connectivity proof. Tested against a real uvicorn server + real
MCP client. `CONNECTING.md` documents bind options.

**T14 done (2026-07-19), M4:** full MCP surface — 15 tools + 3 `run://` resources. `jobs.py`
turns `run_pipeline`/`run_stage` into background-thread jobs returning a `run_id` immediately;
pre-stage background failures surface via a `job_error` field. `artifact_view.py` shapes results
per artifact kind (npz summaries, inline png previews, path-pointers for video). `TOOLS.md` is
the Claude-facing usage doc.

**T15 done (2026-07-19), M5:** `orchestrator/ui/` — single-file Streamlit app (five tabs) over a
thin adapter importing Layer 1 in-process (transport decision: not HTTP — UI always runs on the
same machine); reuses `mcp_server.jobs`/`artifact_view` (both mcp-package-free) so UI and
Claude-over-MCP see identical shapes. Folded in `ampUI.py`'s amp-parameter panel; "save as new
preset" writes YAML. Streamlit gotcha: widget keys must be namespaced by preset name or stale
values persist across preset switches. Suite: 232 passed / 9 skipped.

**T17 opened (2026-07-19, no code yet):** real `cancel_run` (mechanism decided: stop the whole
container via `ContainerManager.stop`, accept dropped warm state; per-exec cancellation noted as
possible later improvement), a concurrency guard for `run_id` races in `jobs.py`, typed return
for `get_preview`'s video branch. Depends only on T14, fully unblocked.
