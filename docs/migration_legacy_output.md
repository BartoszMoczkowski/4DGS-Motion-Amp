# Migration Guide: Legacy Output Directories (`output/` → `runs/`)

This guide explains the migration path from legacy direct-script output directories (`output/multipleview/<scene>/`) to the unified orchestrator layout (`runs/<run_id>/`).

---

## 1. Background & Context

In earlier iterations of the codebase (and upstream [4DGaussians](https://github.com/hustvl/4DGaussians)), running scripts directly dumped outputs into:
```
output/multipleview/<scene>/
    point_cloud/iteration_14000/
    trajectories.npz
    segmentation.npz
    segmentation_mbs.npz
    cfg_args
```

While functional for one-off manual executions, this layout caused several pain points:
- **No provenance or metadata**: Missing exact git SHA, configuration overrides, execution timestamps, or stage execution statuses.
- **No stage isolation**: Upstream training and downstream segmentation outputs were intermingled in one folder.
- **No caching or DAG tracking**: The DAG engine, content-hash caching, and local MLflow tracking could not track or reuse these unmanifested outputs.
- **Tool incompatibility**: Modern commands (`amp explore`, `amp render`, `amp seg eval`, MCP server tools, and Streamlit UI) require a valid `manifest.json`.

---

## 2. Canonical Orchestrator Layout (`runs/<run_id>/`)

The canonical, single source of truth for all pipeline executions is `runs/<run_id>/`:

```
runs/<run_id>/
├── manifest.json              # Atomic record: status, git SHA, stage records, artifacts
├── config_snapshot.json       # Exact resolved configuration used for the run
├── logs/                      # Per-stage execution logs
│   ├── train_default.log
│   ├── seg_extract_default.log
│   └── segment_rigid.log
├── train_out/                 # 4DGS training output (point_cloud/, deformation.pth)
├── seg_extract_out/           # Extracted 3D trajectories (trajectories.npz)
├── segment_<impl>_out/        # Segmentation masks & previews (segmentation.npz)
├── seg_eval_<impl>_out/       # Evaluation metrics (seg_eval_result.json)
└── amp_default_out/           # Amplified video renders (amplified.mp4)
```

### Run-ID Naming Convention
Run IDs strictly conform to:
1. **Experiment specs (`amp exp run`)**: `<experiment>-<cell>-<8hex>` (e.g., `pump_grid_v2-pump_A20mm_M2-9104a031`).
2. **Standalone preset runs (`amp train` / `run_pipeline`)**: `<preset>-<8hex>` (e.g., `pump01-15e286ff`).
3. **Charset**: `[A-Za-z0-9_.-]+` (no path separators, no `..`, strictly confined under `runs/`).

---

## 3. How to Migrate an Existing Legacy Directory

A migration utility is provided at [`scripts/migrate_legacy_output.py`](file:///c:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp/scripts/migrate_legacy_output.py).

### Automated Migration

To wrap an existing model (such as `output/multipleview/pump01`) into a modern run directory:

```bash
uv run python scripts/migrate_legacy_output.py \
    --legacy-dir output/multipleview/pump01 \
    --run-id legacy-pump01-00000001 \
    --preset pump01
```

Options:
- `--dry-run`: View what files will be copied/structured without touching the disk.
- `--link`: Use hardlinks instead of copying large point cloud files (saves disk space on Windows NTFS).

### What the Migration Does
1. Creates `runs/<run_id>/` and `logs/`.
2. Organizes `point_cloud/` and `cfg_args` into `runs/<run_id>/train_out/`.
3. Moves or links `trajectories.npz` into `runs/<run_id>/seg_extract_out/`.
4. Moves or links `segmentation.npz` (and previews) into `runs/<run_id>/segment_rigid_out/`.
5. Moves or links `segmentation_mbs.npz` into `runs/<run_id>/segment_mbs_out/` if present.
6. Writes a verified `manifest.json` marking stages as `success` and registering artifacts (`model`, `trajectories`, `segmentation`).
7. Writes a valid `config_snapshot.json`.

---

## 4. Exercising the Migrated Run

Once migrated, the run is immediately recognizable by all standard tools:

1. **Inspect with unified explorer**:
   ```bash
   uv run amp explore legacy-pump01-00000001
   ```

2. **Render motion amplification**:
   ```bash
   uv run amp render legacy-pump01-00000001 \
       --amp-factors "2 -1 -1 -1 -1 -1 -1 -1" \
       --method eulerian \
       --video out.mp4
   ```

3. **Evaluate or re-run segmentation**:
   ```bash
   uv run amp seg eval legacy-pump01-00000001
   uv run amp seg rigid legacy-pump01-00000001 --k 16
   ```

4. **UI & MCP Access**:
   The run appears automatically in `uv run amp orch ui` under the "Runs & Artifacts" tab and is queryable over MCP via `get_run(run_id="legacy-pump01-00000001")`.
