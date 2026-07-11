# Architecture

Single source of truth for how the three layers fit together. Expands the original plan in
`../../.claude_notes/NOTES_pipeline_orchestration.md`; when the design changes, update *this* file.

## Problems being solved

1. Running the whole pipeline is manual/effortful.
2. Claude can't touch the GPU-bound parts (Isaac, 4DGS, MBS).
3. The pipeline is fragmented with no consistent way to add modifications/new ideas.

## The pipeline as a DAG (ground truth)

| # | Stage (role.impl) | Wraps | Env | GPU |
|---|---|---|---|---|
| 1 | prep.split | `omniverse_pipeline/split_mesh.py` | isaac/host* | no |
| 2 | prep.motion | `omniverse_pipeline/add_motion.py` | isaac/host* | no |
| 3 | capture.isaac | `omniverse_pipeline/omni_capture.py` | isaac | yes |
| 4 | convert | `omniverse_pipeline/omni_to_4dgs.py` | host | no |
| 5 | train | `train.py` | cuda | yes |
| 6 | render | `render.py` | cuda | yes |
| 7 | seg_extract | `motion_seg/extract_trajectories.py` | cuda | yes |
| 8 | segment.rigid (B) / segment.mbs (A) | `motion_seg/segment_rigid.py` / `mbs_infer.py` | host / cuda | no / yes |
| 9 | seg_eval | `motion_seg/evaluate_segmentation.py` | host | no |
| 10 | amp | `render_amp.py` (+ `motion_amp/renderer.py`) | cuda | yes |

\* split/motion are USD/trimesh CPU work; run in `isaac` (has USD) or a small CPU image — decided in T11.

The two GPU images never need to run simultaneously → single-GPU **serial** scheduling is fine.

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
- **Path translation** (T06): the *only* place that maps host ↔ WSL2 ↔ container paths.
- **Container manager** (T08): Docker SDK/CLI over Docker Desktop from WSL2; reuses the existing
  devcontainer defs for image+mounts; GPU passthrough; warm long-lived containers; log streaming.
- **Resource manager** (T12): pynvml/`nvidia-smi` VRAM+RAM query; serial gating so combined VRAM
  never exceeds free; adaptive knobs (`low_vram_mode`, seg working-set, `rt_subframes`) and
  OOM-retry with reduced memory.
- **Public API**: `run_pipeline`, `run_stage`, `get_status`, `list_runs`, `list_artifacts`,
  `get_artifact`, `cancel`, `list_presets`, `validate_config`, `gpu_status`, container controls.
  Everything Layers 2/3 need is a call here.

## Layer 2 — MCP server (HTTP)

Thin server on the WSL2 host wrapping the Layer 1 API + Docker + filesystem. Exists because the
Claude sandbox has no CUDA/Isaac/Docker.

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

## Cross-cutting risks

Config unification is highest-value but tedious (T02). Cache correctness needs code-version in the
key (T05). Path translation must stay centralized (T06). Isaac cold-start needs warm container +
persisted cache volumes (T08/T11). MCP auth + reachability from the sandbox to the WSL2 host is the
main integration unknown (T13).
