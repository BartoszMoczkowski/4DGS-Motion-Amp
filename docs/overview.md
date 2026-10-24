# Project overview

## Goal

Thesis project: **per-part motion amplification of 4D Gaussian Splatting reconstructions**, to reveal subtle (mm-scale, mostly periodic) motion in rigid machinery. The chain is:

```
synthetic multi-cam capture (Omniverse / Isaac Sim)
  → 4DGS reconstruction (canonical Gaussians + deformation field)
  → motion segmentation (cluster Gaussians into rigid motion groups)
  → per-segment Eulerian motion amplification (render_amp.py)
```

(2026-10-05 scope note: `render_amp.py`'s `amp_factors` are per *parameter channel* (8 slots), not per motion part — the "segmented" method variants are memory chunking, not segmentation masks. Per-part amplification currently flows through the orchestrator's seg pipeline (`seg_extract → segment.* → amp`), not through `render_amp.py` alone.)

Synthetic data is the enabler: real captures give no ground-truth camera poses and no per-part segmentation labels, so quantitative evaluation (ARI / IoU) is only possible on scenes we author ourselves.

Key design decisions (locked with the author): static segmentation (one labeling per clip), rigid-machine targets, whole-clip periodic small-amplitude motion, position as the segmentation feature, N > 10⁵ Gaussians, retraining MBS is acceptable.

## Repo map

| Path | What it is |
|---|---|
| `src/amp/` | Unified root `amp` CLI entry point (`amp capture`, `amp train`, `amp render`, `amp seg`, `amp exp`, `amp orch`, `amp bench`, `amp mlflow`, `amp explore`) |
| `core/` (package `4dgs-core`: `train.py`, `render.py`, `render_amp.py`, `scene/`, …) | Upstream [4DGaussians](https://github.com/hustvl/4DGaussians) codebase (base of this fork) plus the author's motion-amplification code (FFT-based Eulerian amplification over per-Gaussian trajectory tensors) |
| `amp-ui/` (package `amp-ui`: `amp_ui/ampUI.py`, `cameras.py`, `run_renders_auto.py`) | Standalone Streamlit UI, USB camera recorder, benchmark harness |
| `omniverse-pipeline/` | Isaac Sim capture + USD prep + converter to 4DGS `multipleview` format — see [omniverse-pipeline.md](omniverse-pipeline.md) |
| `motion-seg/` | Motion segmentation: rigidity-graph clustering (Option B) + MBS inference adapter (Option A) + evaluation — see [motion-segmentation.md](motion-segmentation.md) |
| `orchestrator/` | Pipeline orchestration: DAG execution package (`pipeline/`), HTTP MCP server (`mcp_server/`), Streamlit UI (`ui/`) — see [orchestrator.md](orchestrator.md) |
| `experiments/` | Declarative experiment specifications (`pump_grid_v2.yaml`, `spinning_cubes.yaml`) run via `amp exp run` |
| `runs/` | Canonical orchestrator run tree (`runs/<run_id>/` with `manifest.json`, `config_snapshot.json`, logs, and stage directories) |
| `output/` | Legacy model output directory (from direct-script runs; superseded by `runs/`, migration guide in [migration_legacy_output.md](migration_legacy_output.md)) |
| `scratch/` | Unsupported one-off scripts, diagnostic harnesses, and retired ad-hoc code (see [scratch/README.md](../scratch/README.md)) |
| `submodules/multibody-sync-4dgs` | Fork of MultiBodySync (CVPR 2021), used as segmentation reference |
| `.claude_notes/` | Chronological working notes (the primary detailed record) |
| `data/` | Datasets (DNeRF synthetics + `multipleview` scenes incl. the generated `pump01`) |

## Status & Milestones

- **M5 (2026-07-19)**: The full 9-stage pipeline (`prep_split → prep_motion → capture.isaac → convert → train → render → seg_extract → segment.rigid → amp`) completed end-to-end on real hardware, with cross-run caching working. The MCP server (15 tools, 3 resources) and Streamlit UI (`amp orch ui`) are fully operational.
- **M6 (Workspace & CLI Streamlining, 2026-10 — T24–T26)**: Single root workspace install via `uv sync`; unified `amp` CLI replaces all raw script invocations; legacy `motion_seg/run.sh` deleted; one-off analysis scripts moved to `scratch/`.
- **M7 (Experiment Standardization & Exploration, 2026-10 — T27–T30)**: Declarative experiment specs (`experiments/*.yaml`) executed via `amp exp run`; unified run ID convention (`<experiment>-<cell>-<8hex>`); results store `runs/experiment_results.csv`; local MLflow tracking (`amp mlflow serve` / `amp mlflow status`) with degradation-safe logging; unified inspection navigator `amp explore`; job concurrency guard and container cancellation hardening (T17).
- Segmentation diagnostics (T18–T22): Trajectory denoising (T18), Kabsch EM (T20), ROI motion gating (T19), and Oracle mask lifting (T22) confirm that on mm-scale periodic motions with ~10⁵ Gaussians, motion-only segmentation is reconstruction-quality-limited (jitter ≈ motion). Practical contributions center on the motion amplification pipeline and synthetic evaluation framework.
