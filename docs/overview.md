# Project overview

## Goal

Thesis project: **per-part motion amplification of 4D Gaussian Splatting reconstructions**, to reveal subtle (mm-scale, mostly periodic) motion in rigid machinery. The chain is:

```
synthetic multi-cam capture (Omniverse / Isaac Sim)
  → 4DGS reconstruction (canonical Gaussians + deformation field)
  → motion segmentation (cluster Gaussians into rigid motion groups)
  → per-segment Eulerian motion amplification (render_amp.py)
```

Synthetic data is the enabler: real captures give no ground-truth camera poses and no per-part segmentation labels, so quantitative evaluation (ARI / IoU) is only possible on scenes we author ourselves.

Key design decisions (locked with the author): static segmentation (one labeling per clip), rigid-machine targets, whole-clip periodic small-amplitude motion, position as the segmentation feature, N > 10⁵ Gaussians, retraining MBS is acceptable.

## Repo map

| Path | What it is |
|---|---|
| root (`train.py`, `render.py`, `scene/`, …) | Upstream [4DGaussians](https://github.com/hustvl/4DGaussians) codebase (base of this fork) |
| `render_amp.py`, `motion_amp/`, `ampUI.py`, `cameras.py`, `run_renders_auto.py` | Author's motion-amplification code (FFT-based Eulerian amplification over per-Gaussian trajectory tensors, Streamlit UI) |
| `omniverse_pipeline/` | Isaac Sim capture + USD prep + converter to 4DGS `multipleview` format — see [omniverse-pipeline.md](omniverse-pipeline.md) |
| `motion_seg/` | Motion segmentation: rigidity-graph clustering (Option B) + MBS inference adapter (Option A) + evaluation — see [motion-segmentation.md](motion-segmentation.md) |
| `orchestrator/` | Pipeline orchestration: DAG execution package, HTTP MCP server, Streamlit UI — see [orchestrator.md](orchestrator.md) |
| `submodules/multibody-sync-4dgs` | Fork of MultiBodySync (CVPR 2021), used as segmentation reference |
| `.claude_notes/` | Chronological working notes (the primary detailed record) |
| `data/`, `output/` | Datasets (DNeRF synthetics + `multipleview` scenes incl. the generated `pump01`) and trained models |

## Status (2026-07-19)

- Omniverse → 4DGS pipeline works end-to-end; the "pump" test asset (107 rigid parts, authored mm-scale periodic motion, exact GT labels) is captured, converted, and trainable.
- Option B segmentation (`segment.rigid`) is implemented and verified on synthetic data (ARI ≈ 0.999); first real pump run was poor (low reconstruction SNR), log-space Otsu fix applied, quality still gated by training quality. Option A (`segment.mbs`) is wired into the orchestrator but not yet run on real GPU data.
- The orchestrator is complete through milestone M5: **the full 9-stage pipeline (`prep_split → prep_motion → capture.isaac → convert → train → render → seg_extract → segment.rigid → amp`) completed end-to-end on real hardware on 2026-07-19**, with cross-run caching working. The MCP server (15 tools, 3 resources) and Streamlit UI are done. Remaining: T17 (cancel/job hardening, open) and T16 (WSL bundling, deferred).
