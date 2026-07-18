# Architecture

Single source of truth for how the three layers fit together. Expands the original plan in
`../../.claude_notes/NOTES_pipeline_orchestration.md`; when the design changes, update *this* file.

## Problems being solved

1. Running the whole pipeline is manual/effortful.
2. Claude can't touch the GPU-bound parts (Isaac, 4DGS, MBS).
3. The pipeline is fragmented with no consistent way to add modifications/new ideas.

## The pipeline as a DAG (ground truth)

| # | Stage (role.impl) | Ported from (reference only) | Env | GPU |
|---|---|---|---|---|
| 1 | prep_split.default | `omniverse_pipeline/split_mesh.py` | isaac* | no |
| 2 | prep_motion.default | `omniverse_pipeline/add_motion.py` | isaac* | no |
| 3 | capture.isaac | `omniverse_pipeline/omni_capture.py` | isaac† | yes |
| 4 | convert | `omniverse_pipeline/omni_to_4dgs.py` | host | no |
| 5 | train | `train.py` | cuda | yes |
| 6 | render | `render.py` | cuda | yes |
| 7 | seg_extract | `motion_seg/extract_trajectories.py` | cuda | yes |
| 8 | segment.rigid (B) / segment.mbs (A) | `motion_seg/segment_rigid.py` / `mbs_infer.py` | host / cuda | no / yes |
| 9 | seg_eval | `motion_seg/evaluate_segmentation.py` | host | no |
| 10 | amp | `render_amp.py` (+ `motion_amp/renderer.py`) | cuda | yes |

\* `prep_split`/`prep_motion` are plain USD/trimesh CPU work with no real Isaac Sim runtime
dependency — decided in T11 to run in the existing `isaac` container anyway (it already has
`pxr`/usd-core) rather than stand up a separate small CPU image just for these two, which was
judged out of scope for a single contained task. Named `prep_split.default`/`prep_motion.default`,
not `prep.split`/`prep.motion` as originally sketched here — see T11's log
(`.claude_notes/NOTES_pipeline_orchestration.md`) for why the dot-form would have collided both
into one ambiguous `"prep"` role under the registry's `role.impl` convention.

† **Revised 2026-07-16:** `capture.isaac` keeps `environment = "isaac"` (still the same GPU
resource class, still mutually exclusive with `cuda` — see below), but no longer actually execs
inside the `isaac` Docker container. NVIDIA confirmed Vulkan (what Isaac Sim's Hydra/RTX renderer
needs) isn't supported under WSL2, which backs Docker Desktop's Linux containers on Windows; a
real-hardware run showed `omni_capture.py` completing with no exception while its RTX render
products never produced a frame. `capture.isaac` now execs `omni_capture.py` as a **native Windows
subprocess** against Bartosz's real Isaac Sim install instead
(`pipeline.stages.isaac_common.run_native_isaac_script`, `PIPELINE_ISAAC_NATIVE_PYTHON` env var) —
see `INSTRUCTIONS.md`'s locked decision and `.claude_notes/NOTES_pipeline_orchestration.md`'s
"adjust the project plan" entry. `prep_split`/`prep_motion` are unaffected — CPU-only, no
rendering — and still exec inside the `isaac` container as before.

The two GPU images never need to run simultaneously → single-GPU **serial** scheduling is fine.

**"Ported from" means exactly that, not "imports at runtime."** `omniverse_pipeline/`, `motion_seg/`,
and the repo-root scripts are throwaway/testing scripts — a reference for already-verified logic,
not a dependency. No stage may `sys.path`-hack into them or shell out to them (superseded
"wrap, don't rewrite"; see `INSTRUCTIONS.md` and `.claude_notes/NOTES_pipeline_orchestration.md`,
2026-07-14). The only thing genuinely external to this project is the **container runtime**
(`isaac`/`cuda` images) — never the script files. See "Vendored stage logic" below for where the
copied-in code lives.

### Vendored stage logic

Each stage's `run()` calls orchestrator-owned code, not the original script:

```
orchestrator/
  pipeline/
    stages/            <- Stage subclasses: orchestration only (config, artifacts, logging, ctx)
    vendored/          <- copied-in logic, ported from the reference scripts above
      host/            <- convert, segment.rigid, seg_eval logic (plain CPU, runs in `host` venv)
      cuda/             <- train/render/seg_extract/amp/segment.mbs logic (runs inside the `cuda`
                            container; the container is external, this code is not)
      isaac/            <- prep_split.default/prep_motion.default/capture.isaac logic. Still one
                            package (same "copy, don't reimplement" rule) even though, since
                            2026-07-16, capture.isaac's script runs as a native Windows subprocess
                            rather than inside the `isaac` container -- see the DAG table's dagger
                            note above and pipeline/stages/isaac_common.py's module docstring.
```

A `host`-environment stage (T07: `convert`/`segment.rigid`/`seg_eval`) imports from
`pipeline.vendored.host.<module>` directly (a normal in-project import, not a `sys.path` reach
outside `orchestrator/`) and calls the ported function in-process — no container involved, no
GPU/torch dependency.

A `cuda`/`isaac`-environment stage (T09: `train`/`render`/`seg_extract`/`amp`) never imports its
vendored module at all — `pipeline/vendored/cuda/*.py`'s real dependencies (`torch`, `arguments`,
`scene`, `gaussian_renderer`, ...) only exist inside the container, not in the orchestrator's own
host process. Instead the stage builds a CLI invocation (`python pipeline/vendored/cuda/<name>.py
<args>`) from the resolved config and hands it to `ctx.containers.exec_in_container(...)` (T08) —
see `pipeline/stages/cuda_common.py`. T08's repo bind-mount (`/workspace`) is what makes
`pipeline/vendored/cuda|isaac` visible inside the running container, so the container executes
the orchestrator's copy, not whatever happens to be sitting in
`omniverse_pipeline/`/`motion_seg`/repo root at the time.

## Layer 1 — `pipeline/` execution module

Components (each maps to a task):

- **Config** (T02): one pydantic schema for *all* settings; layered presets
  `base ← scene ← experiment`. Replaces `capture_config*.yaml`, `arguments/multipleview/*.py`, and
  `.sh` flags. Validated before anything runs.
- **Artifacts + run manifest** (T03): typed artifact records; per-run `manifest.json` with
  resolved config, git SHA, per-stage status/timing/logs/outputs/peak-mem. The read surface for
  Layers 2/3.
- **Stage base + registry** (T04): `Stage` declares inputs/outputs/environment/resources and a
  `run(ctx)`. Registered under `role.impl` names; multiple impls per role are config-selectable.
- **DAG scheduler + cache** (T05): topo-sort by artifact deps; skip fresh stages. Cache key =
  resolved-config + input-artifact hashes + code version (git SHA + stage source hash).
  `from_stage`/`to_stage`/`only`/`force`, resume-on-crash.
- **Path translation** (T06): the *only* place that maps host ↔ container paths. **Revised
  2026-07-14:** two spaces, not three — the runtime host is native Windows (see
  `INSTRUCTIONS.md`'s locked decision), so there's no separate WSL2 execution environment whose
  filesystem view differs from the host's own. WSL2/Linux-distro support, if it ever comes back,
  re-enters as a third space the same way `container` already works.
- **Container manager** (T08): Docker SDK/CLI over Docker Desktop, driven directly from Windows;
  reuses the existing devcontainer defs for image+mounts; GPU passthrough; warm long-lived
  containers; log streaming; also mounts/bakes in `pipeline/vendored/cuda|isaac` so container
  stages run the orchestrator's own copied-in code, not the reference scripts (see "Vendored stage
  logic" above). **Revised 2026-07-16:** no longer the execution path for every `isaac`-labelled
  stage — `capture.isaac` runs as a native Windows subprocess instead (WSL2 doesn't support Vulkan,
  see the DAG table's dagger note); the container manager still owns `prep_split.default`/
  `prep_motion.default`'s execution and all of `cuda`'s, unchanged.
- **Resource manager** (T12): pynvml/`nvidia-smi` VRAM+RAM query; serial gating so combined VRAM
  never exceeds free; adaptive knobs (`low_vram_mode`, seg working-set, `rt_subframes`) and
  OOM-retry with reduced memory.
- **Public API**: `run_pipeline`, `run_stage`, `get_status`, `list_runs`, `list_artifacts`,
  `get_artifact`, `cancel`, `list_presets`, `validate_config`, `gpu_status`, container controls.
  Everything Layers 2/3 need is a call here.

## Layer 2 — MCP server (HTTP)

Thin server on Bartosz's Windows machine (running natively, driving Docker Desktop directly — see
`INSTRUCTIONS.md`'s locked decision) wrapping the Layer 1 API + Docker + filesystem. Exists because
the Claude sandbox has no CUDA/Isaac/Docker.

- **Transport: HTTP** (streamable HTTP / SSE) with auth, so Claude can be local or remote (T13).
- **Async jobs**: `run_pipeline`/`run_stage` return a `run_id` immediately; Claude polls status /
  tails logs (train can be hours).
- **Tools/resources** (T14): list/validate presets, run pipeline/stage, get status, tail logs,
  list runs/artifacts, read artifact (text/JSON/npz *summary*), get preview (PNG/video so Claude
  can *see* results), gpu status, list/start/stop container, cancel run. Manifests, logs, previews
  exposed as MCP resources. Whitelisted ops only; no arbitrary shell.

## Layer 3 — UI (deprioritized)

Thin layer over the **same** Layer 1 API (T15). Streamlit first (reuse `ampUI.py`'s amp-param
panel): pick/edit preset, launch run, watch stage progress + logs + GPU, browse artifacts/previews,
compare runs.

## Phasing → tasks

- Phase 0 (framework, CPU-only slice): T01–T07
- Phase 1 (container manager + CUDA stages): T08–T10
- Phase 2 (Isaac stages): T11
- Phase 3 (resources): T12
- Phase 4 (MCP over HTTP): T13–T14
- Phase 5 (UI): T15
- **Phase 6 (deferred, not scheduled): WSL2/Linux-distro bundling.** Packaging a proper WSL2 +
  Docker setup (e.g. so the whole orchestrator ships as a one-command WSL2 environment rather than
  assuming Docker Desktop is already installed on Windows) — see T16 in `TASKS.md`. Explicitly
  deferred 2026-07-14 in favor of running natively on Windows first; revisit if/when running from
  Linux/WSL2 actually matters again (e.g. a non-Windows machine, or CI).

## Cross-cutting risks

Config unification is highest-value but tedious (T02). Cache correctness needs code-version in the
key (T05). Path translation must stay centralized (T06). Isaac cold-start needs warm container +
persisted cache volumes (T08/T11). MCP auth + reachability from the sandbox to Bartosz's Windows
machine is the main integration unknown (T13).
