# Orchestrator: the three-layer pipeline system

Compiled from `orchestrator/planning/` and `.claude_notes/NOTES_pipeline_orchestration.md` (full detail there — `planning/ARCHITECTURE.md` is the design source of truth, `planning/TASKS.md` the board).

## Why it exists

Three long-standing problems with the original chain of `.sh` scripts + scattered configs: running the whole pipeline was manual; Claude couldn't touch GPU-bound stages (Isaac/4DGS/MBS); there was no consistent way to add new ideas.

## The three layers

- **Layer 1 — `orchestrator/pipeline/`**: custom lightweight DAG execution package. Stage registry (`@register("role.impl")`), typed pydantic artifacts + atomic run manifests (`runs/<run_id>/`), one unified pydantic config schema with `extends:`-chained YAML presets (`base`, `pump01`, `pump01_segA`, …), cross-run content-hash caching, Docker container manager (`cuda` + `isaac` images, warm reuse, stale-image auto-rebuild via a build-hash label), host↔container path translation. Runs natively on Windows driving Docker Desktop.
- **Layer 2 — `orchestrator/mcp_server/`**: HTTP MCP server (streamable HTTP/SSE, bearer-token auth) wrapping Layer 1: 15 tools (presets, runs, artifacts, background `run_pipeline`/`run_stage` jobs, log tailing, previews, GPU/container control) + 3 `run://` resources. `mcp_server/TOOLS.md` is the Claude-facing usage doc; `CONNECTING.md` covers bind options.
- **Layer 3 — `orchestrator/ui/`**: single-file Streamlit app (Presets, Launch & Monitor, Runs & Artifacts, Compare Runs, GPU/Containers), importing Layer 1 in-process and reusing the MCP server's job/artifact-view modules so UI and Claude see identical shapes. Supersedes `ampUI.py`'s folder-scanning workflow.

## Key rules and decisions

- **"Copy the logic in, don't call the original script."** `omniverse_pipeline/`, `motion_seg/`, and repo-root scripts are reference-only; stage logic is vendored verbatim into `pipeline/vendored/{host,cuda,isaac}/`. The only external dependency allowed is the container runtime.
- CUDA stages (`train/render/seg_extract/amp`, plus `segment.mbs`) exec as separate processes inside the `cuda` container; a generated "bridge" config file keeps the pydantic config the single source of truth for 4DGS's `merge_hparams` mechanism.
- `capture.isaac` runs against the **native Windows Isaac Sim install** (subprocess via `python.bat`), not a container — Vulkan rendering is unsupported under WSL2 (hard NVIDIA limitation). CPU-only USD prep stages stay in the `isaac` container.
- Extensibility proof: `segment.mbs` was added as a second impl behind the `segment` role with zero core edits — just a stage registration + a preset.

## Task history

T01 scaffold → T02 config → T03 artifacts/manifests → T04 stage base/registry → T05 DAG scheduler + caching → T06 path translation → T07 CPU stages (redone under the copy-in rule; M1) → T08 container manager (GPU-verified on real hardware) → T09 CUDA stages → T10 `segment.mbs` → T11 Isaac stages (revised to native capture; M3) → T12 resources → T13 MCP skeleton → T14 full MCP tool set (M4) → T15 UI (M5). All done as of 2026-07-19. Open: **T17** (real `cancel_run` — decided mechanism: stop the whole container; job concurrency guard; typed preview return). Deferred: **T16** (WSL2/Docker bundling). Test suite: 232 passed / 9 skipped (GPU-only tests auto-skip in the sandbox).

## The real-hardware debugging saga (2026-07-16 → 07-19)

Eleven distinct real bugs, each only surfacing after the previous fix let the run get one stage further: Isaac cache-volume permissions; cross-run cache poisoned by a bogus "success" (fixed + stronger post-hoc output checks); weak `capture.isaac` success check; the Vulkan/WSL2 limitation (→ native capture); the `cuda` Dockerfile's venv build commented out; `TORCH_CUDA_ARCH_LIST` unset (`docker build` has no GPU, torch arch-detection crashes on an empty list — fixed with `8.6+PTX` for the RTX 3090); the venv baked into `/workspace` being shadowed by the runtime bind mount (build moved to `/opt/build`); `write_bridge` emitting cp1252 (encoding audit across the package); wrong hardcoded native Isaac Python path; vendored `amp.py` still on the removed `mmcv.Config` API; and a **silent-success** bug where `train.py` appended `save_iterations` before the config merge, so overridden `iterations` never checkpointed (fixed + `TrainStage` now verifies a non-empty `point_cloud/` before reporting success).

**Milestone (2026-07-19): the full `prep_split → prep_motion → capture.isaac → convert → train → render → seg_extract → segment.rigid → amp` chain completed end-to-end on real hardware**, with the first four stages correctly served from cross-run cache on the final rerun and a real amplified `render.mp4` on disk. (One `amp.default` exit-137 SIGKILL along the way was diagnosed as host-RAM OOM, resolved by freeing memory, not a code change.)

## Setup

First-time machine setup: `orchestrator/planning/WINDOWS_SETUP.md` (Docker Desktop, NGC login/EULA, MBS checkpoint download, native Isaac path, MCP server bind/token).
