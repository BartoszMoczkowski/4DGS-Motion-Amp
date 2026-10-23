# Tools & resources (T30 consolidated) — a usage guide for Claude

The T30 MCP server consolidates intent-level operations into 12 tools, minimizing LLM tool-calling round-trips while providing comprehensive inspection and execution capabilities.

Read `CONNECTING.md` first for transport and authentication setup.

---

## Preferred workflow

1. **Inspect First (`explore_experiment`)**:
   Instead of sequentially calling `list_runs`, `list_artifacts`, `view_file`, and checking folders, call `explore_experiment()`:
   - Without args: returns workspace overview (experiments, recent runs, detected datasets).
   - With experiment name (`target="pump_grid_v2"`): returns cells, variants, execution matrix, and pivoted evaluation metrics (ARI, IoU).
   - With run ID (`target="pump01-15e286ff"`): returns preset, status, stage timeline with peak VRAM, config diffs against base, registered artifacts, and un-manifested files.
   - With capture/dataset path (`target="data/bouncingballs"`): returns camera count, frame count, bounds, and COLMAP status.

2. **Launch Runs (`run_experiment`)**:
   - `run_experiment(target="pump_grid_v2")` to run an experiment spec.
   - `run_experiment(target="pump01")` to run a preset. Auto-discovers and seeds required external artifacts (`raw_mesh`, `capture`, `scene`, `gt_segmentation`) from known paths. Fails fast with clear actionable diagnostics if an asset is missing. Returns `run_id` immediately (async).

3. **Monitor Progress (`get_run_status`)**:
   Poll `get_run_status(run_id="...", include_logs=True)`. Returns per-stage progress, peak VRAM/RAM, MLflow URLs, and recent log lines in one call.

4. **Inspect & Preview Results**:
   - `get_run_artifact(run_id, artifact_name)` for JSON, NPZ summary (never raw giant arrays), or PLY vertex counts.
   - `get_run_preview(run_id, artifact_name)` for inline PNG images or typed video pointers.
   - `get_run_report(run_id)` for the T29/T30 report bundle, summary markdown, and MLflow links.

---

## Complete tool reference (12 tools)

| Tool | Parameters | Description |
|---|---|---|
| `explore_experiment` | `target?`, `view="default"`, `include_unregistered=False` | **Single unified navigator**. Replaces 10–30 separate tool calls. Traverses workspace overview, experiment specs, run timelines, config diffs, artifacts, and captures in ONE shot. |
| `run_experiment` | `target`, `stages?`, `overrides?`, `external_artifacts?`, `force=False` | Launch an experiment spec (e.g. `pump_grid_v2`) or preset (e.g. `pump01`). Auto-seeds external artifacts. Returns `run_id` immediately. |
| `run_stage` | `run_id`, `stage`, `force=False`, `overrides?` | Run a single stage (e.g. `render.default`, `segment.rigid2`) within an existing run. |
| `get_run_status` | `run_id`, `include_logs=False`, `stage_log?`, `max_log_lines=100` | Per-stage status, execution progress, peak VRAM/RAM, MLflow URL, and optional log tailing. |
| `list_runs` | `limit=25` | Quick listing of recent runs with preset, status, and timestamps. |
| `cancel_run` | `run_id` | Safely cancel an in-flight run. Stops running containers, marks manifest cancelled, and updates MLflow. |
| `get_run_artifact` | `run_id`, `artifact_name` | JSON-safe summary of an artifact (JSON content, NPZ array stats, PLY vertex counts, dataset listings). |
| `get_run_preview` | `run_id`, `artifact_name` | Visual preview: inline `Image` for PNG or typed `VideoPreview` pointer for video clips. |
| `get_run_report` | `run_id` | Path to generated report bundle (`runs/<id>/report/`), index markdown, and MLflow URL. |
| `manage_preset` | `action="list"\|"validate"\|"save"`, `preset?`, `name?`, `base_preset?`, `overrides?`, `overwrite=False` | Inspect available presets, validate resolved config, or save a new variant extending a base preset. |
| `gpu_status` | — | Current VRAM and RAM allocation, total limits, and free headroom on the machine. |
| `manage_container` | `action="list"\|"start"\|"stop"`, `env?`, `container_id?` | List, start, or stop managed Docker containers (`cuda` or `isaac`). |

---

## Resources

Reachable via MCP resource URI:
- `run://{run_id}/manifest`: raw JSON manifest.
- `run://{run_id}/log/{stage}`: full raw untruncated stage log text.
- `run://{run_id}/artifact/{artifact_name}`: raw bytes of a file-kind artifact.

---

## Result shaping & error handling

- **No giant dumps**: NPZ files return array statistics (`shape`, `dtype`, `min`, `max`, `mean`); point clouds return header counts; directory artifacts return entry listings.
- **Structured errors**: Unknown presets, missing external assets, and concurrency conflicts return actionable, typed error descriptions rather than raw stack traces.
