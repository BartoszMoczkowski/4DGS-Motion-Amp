"""Report generation layer for single-run bundles and multi-run comparisons (T29).

Produces standardized report bundles under `runs/<run_id>/report/` with canonical filenames
(`summary.png`, `training_curves.png`, `index.md`) and comparative summaries with MLflow
deep-links.

Light-import rule: no heavy module imports at module scope.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pipeline.artifacts import (
    get_runs_root,
    load_manifest,
    run_dir,
)
from pipeline.reporting.metrics import (
    METRIC_CATALOG,
    extract_manifest_metrics,
    format_metric,
    get_metric_def,
)

logger = logging.getLogger(__name__)


def _load_training_curves_from_disk(r_dir: Path) -> list[dict[str, Any]]:
    """Fallback: read training curves from on-disk JSON file."""
    curves_file = r_dir / "train_out" / "training_curves.json"
    if not curves_file.is_file():
        return []
    try:
        data = json.loads(curves_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("curves", [])
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning("Failed loading training curves from %s: %r", curves_file, exc)
    return []


def _load_training_curves_data(run_id: str, r_dir: Path, manifest: Any) -> Any:
    """Fetch training curve points from MLflow history or disk fallback."""
    from pipeline.reporting.charts import TrainingCurvePoint, TrainingCurvesData
    from pipeline import tracking

    points: list[TrainingCurvePoint] = []

    # 1. Try MLflow metric history if reachable
    if tracking.is_enabled():
        try:
            history = tracking.get_metric_history(run_id, "stage.train.default.loss")
            if not history:
                # Try generic loss key
                history = tracking.get_metric_history(run_id, "loss")
            if history:
                l1_hist = {h["step"]: h["value"] for h in tracking.get_metric_history(run_id, "stage.train.default.l1")}
                psnr_hist = {h["step"]: h["value"] for h in tracking.get_metric_history(run_id, "stage.train.default.test_psnr")}
                train_psnr_hist = {h["step"]: h["value"] for h in tracking.get_metric_history(run_id, "stage.train.default.train_psnr")}
                for h in sorted(history, key=lambda x: x["step"]):
                    step = h["step"]
                    points.append(
                        TrainingCurvePoint(
                            iteration=step,
                            loss=h["value"],
                            l1=l1_hist.get(step),
                            test_psnr=psnr_hist.get(step),
                            train_psnr=train_psnr_hist.get(step),
                        )
                    )
        except Exception as exc:
            logger.debug("MLflow history retrieval fallback for %s: %r", run_id, exc)

    # 2. Disk fallback if MLflow returned no points
    if not points:
        disk_curves = _load_training_curves_from_disk(r_dir)
        for c in disk_curves:
            it = c.get("iteration") or c.get("step")
            if it is not None and "loss" in c:
                points.append(
                    TrainingCurvePoint(
                        iteration=int(it),
                        loss=float(c["loss"]),
                        l1=float(c["l1"]) if "l1" in c and c["l1"] is not None else None,
                        elapsed=float(c["elapsed"]) if "elapsed" in c and c["elapsed"] is not None else None,
                        test_psnr=float(c["test_psnr"]) if "test_psnr" in c and c["test_psnr"] is not None else None,
                        train_psnr=float(c["train_psnr"]) if "train_psnr" in c and c["train_psnr"] is not None else None,
                        points=int(c["points"]) if "points" in c and c["points"] is not None else None,
                    )
                )

    return TrainingCurvesData(run_id=run_id, points=points, label=manifest.preset)


def build_run_report(
    run_id: str,
    force: bool = False,
    runs_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Generate the canonical single-run report bundle (charts + index.md).

    Writes outputs to `runs/<run_id>/report/` and logs to MLflow when tracking is enabled.
    """
    root = runs_root or get_runs_root()
    r_dir = run_dir(run_id, runs_root=root)
    rep_dir = r_dir / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_id, runs_root=root)
    mlflow_meta = manifest.metadata.get("mlflow", {})
    mlflow_url = mlflow_meta.get("url")

    from pipeline.reporting.charts import (
        BenchmarkBarPoint,
        plot_benchmark_bars,
        plot_training_curves,
        save_report_figure,
    )

    charts_generated: list[str] = []

    # 1. Benchmark & resource summary figure
    summary_png = rep_dir / "summary.png"
    if force or not summary_png.is_file():
        bench_pts: list[BenchmarkBarPoint] = []
        for s_name, rec in manifest.stages.items():
            if rec.wall_time_s is not None or rec.peak_vram_mb is not None:
                bench_pts.append(
                    BenchmarkBarPoint(
                        name=s_name,
                        wall_time_s=rec.wall_time_s or 0.0,
                        peak_vram_mb=rec.peak_vram_mb or 0.0,
                    )
                )
        if bench_pts:
            fig_sum = plot_benchmark_bars(bench_pts)
            save_report_figure(fig_sum, summary_png)
            charts_generated.append("summary.png")

    # 2. Training curves figure
    training_png = rep_dir / "training_curves.png"
    if force or not training_png.is_file():
        tdata = _load_training_curves_data(run_id, r_dir, manifest)
        if tdata.points:
            fig_train = plot_training_curves(tdata)
            save_report_figure(fig_train, training_png)
            charts_generated.append("training_curves.png")

    # 3. Categorized metrics
    metrics_by_phase = extract_manifest_metrics(manifest, run_dir_path=r_dir)

    # 4. Generate index.md
    index_md_path = rep_dir / "index.md"
    lines: list[str] = [
        f"# Run Report: `{run_id}`",
        "",
        "## Summary Overview",
        "",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| **Preset** | `{manifest.preset}` |",
        f"| **Status** | `{manifest.status}` |",
        f"| **Git SHA** | `{manifest.git_sha or 'unknown'}` |",
        f"| **Created** | {manifest.created_at} |",
        f"| **Updated** | {manifest.updated_at} |",
    ]

    total_time = metrics_by_phase["bench"].get("total_wall_time_s")
    if total_time:
        lines.append(f"| **Total Wall Time** | {format_metric('wall_time_s', total_time)} |")
    max_vram = metrics_by_phase["bench"].get("peak_vram_mb")
    if max_vram:
        lines.append(f"| **Peak GPU VRAM** | {format_metric('peak_vram_mb', max_vram)} |")

    if mlflow_url:
        lines.append(f"| **MLflow Run** | [View in MLflow UI]({mlflow_url}) |")
    lines.append("")

    # Metrics tables by phase
    lines.append("## Metric Results")
    lines.append("")

    for phase in ("segment", "train", "render", "research", "amp", "bench"):
        p_metrics = metrics_by_phase.get(phase, {})
        if not p_metrics:
            continue
        lines.append(f"### {phase.title()} Metrics")
        lines.append("")
        lines.append("| Metric | Value | Unit | Description |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for m_name, m_val in sorted(p_metrics.items()):
            mdef = get_metric_def(m_name)
            desc = mdef.description if mdef else ""
            unit = mdef.unit if mdef else ""
            formatted = format_metric(m_name, m_val)
            lines.append(f"| `{m_name}` | **{formatted}** | {unit} | {desc} |")
        lines.append("")

    # Figures section
    all_pngs = sorted([p.name for p in rep_dir.glob("*.png")])
    if all_pngs:
        lines.append("## Figures & Visualizations")
        lines.append("")
        for chart_name in all_pngs:
            title = chart_name.replace("_", " ").replace(".png", "").title()
            lines.append(f"### {title}")
            lines.append(f"![{title}]({chart_name})")
            lines.append("")

    # Artifacts section
    if manifest.artifacts:
        lines.append("## Registered Artifacts")
        lines.append("")
        lines.append("| Name | Kind | Producing Stage | Path |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for art_name, art in sorted(manifest.artifacts.items()):
            lines.append(f"| `{art_name}` | `{art.kind}` | `{art.producing_stage}` | `{art.path}` |")
        lines.append("")

    index_text = "\n".join(lines)
    index_md_path.write_text(index_text, encoding="utf-8")

    # 5. Log report directory artifacts to MLflow if enabled
    from pipeline import tracking

    if tracking.is_enabled() and mlflow_meta.get("run_id"):
        try:
            tracking._mlflow()
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            mlflow_run_id = mlflow_meta["run_id"]
            client.log_artifact(mlflow_run_id, str(index_md_path), artifact_path="report")
            for png in rep_dir.glob("*.png"):
                client.log_artifact(mlflow_run_id, str(png), artifact_path="report")
        except Exception as exc:
            logger.debug("Failed logging report artifacts to MLflow: %r", exc)

    return {
        "run_id": run_id,
        "report_dir": str(rep_dir),
        "exists": rep_dir.is_dir(),
        "has_index": index_md_path.is_file(),
        "index_markdown": index_text,
        "charts": all_pngs,
        "mlflow_url": mlflow_url,
    }


def build_comparison_report(
    experiment: str,
    impls: Optional[list[str]] = None,
    metric: str = "ari",
    out_dir: Optional[Path] = None,
    runs_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Generate comparative summary bundle across variants for an experiment.

    Produces overlaid step curves, pivot table, and deep-links to MLflow compare UI.
    """
    root = runs_root or get_runs_root()
    target_dir = out_dir or (root / "reports" / experiment)
    target_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.reporting.charts import (
        GridCellEval,
        TrainingCurvesData,
        plot_multi_run_training_curves,
        plot_segmentation_grid_scatter_heatmap,
        save_report_figure,
    )

    matching_runs: list[tuple[str, Any]] = []
    mlflow_run_ids: list[str] = []

    for child in root.iterdir():
        if not child.is_dir() or not (child / "manifest.json").is_file():
            continue
        try:
            m = load_manifest(child.name, runs_root=root)
            exp_name = m.metadata.get("experiment") or m.preset
            if exp_name == experiment or child.name.startswith(f"{experiment}-"):
                matching_runs.append((child.name, m))
                mf = m.metadata.get("mlflow", {})
                if mf.get("run_id"):
                    mlflow_run_ids.append(mf["run_id"])
        except Exception:
            continue

    if not matching_runs:
        return {
            "experiment": experiment,
            "report_dir": str(target_dir),
            "run_count": 0,
            "error": f"No completed runs found matching experiment {experiment!r}",
        }

    # 1. Gathers training curves for multi-run overlay
    runs_curves: list[TrainingCurvesData] = []
    for r_id, m in matching_runs[:10]:  # limit overlay to 10 curves for readability
        tdata = _load_training_curves_data(r_id, root / r_id, m)
        if tdata.points:
            variant_name = m.metadata.get("variant") or m.metadata.get("cell") or r_id
            tdata.label = variant_name
            runs_curves.append(tdata)

    charts: list[str] = []
    if runs_curves:
        fig_curves = plot_multi_run_training_curves(runs_curves, metric_key="loss")
        save_report_figure(fig_curves, target_dir / "compare_training_curves.png")
        charts.append("compare_training_curves.png")

    # 2. Collect grid cell eval records
    grid_records: list[GridCellEval] = []
    variant_impl_pivot: dict[str, dict[str, float]] = {}

    for r_id, m in matching_runs:
        cell_name = m.metadata.get("cell") or m.metadata.get("variant") or r_id
        # Parse base_amp and multiplier from cell name if formatted as pump_A<X>mm_M<Y>
        base_amp = 20.0
        multiplier = 2.0
        if "A" in cell_name and "mm" in cell_name:
            import re
            m_amp = re.search(r"A(\d+)mm", cell_name)
            if m_amp:
                base_amp = float(m_amp.group(1))
            m_mul = re.search(r"M(\d+)", cell_name)
            if m_mul:
                multiplier = float(m_mul.group(1))

        # Check seg_eval artifacts
        metrics_by_phase = extract_manifest_metrics(m, run_dir_path=root / r_id)
        seg_metrics = metrics_by_phase.get("segment", {})

        # Extract per-impl results
        for art_name, art in m.artifacts.items():
            if art_name.startswith("seg_eval_result"):
                impl_name = art_name.replace("seg_eval_result_", "")
                if impl_name == "seg_eval_result":
                    impl_name = "default"
                if impls and impl_name not in impls:
                    continue
                meta = art.metadata or {}
                val = float(meta.get(metric, seg_metrics.get(metric, 0.0)))
                iou_val = float(meta.get("mean_iou", seg_metrics.get("mean_iou", 0.0)))

                grid_records.append(
                    GridCellEval(
                        cell=cell_name,
                        base_amp_mm=base_amp,
                        multiplier=multiplier,
                        impl=impl_name,
                        ari=val if metric == "ari" else float(meta.get("ari", 0.0)),
                        mean_iou=iou_val,
                    )
                )
                variant_impl_pivot.setdefault(cell_name, {})[impl_name] = val

    if grid_records:
        fig_grid = plot_segmentation_grid_scatter_heatmap(grid_records, metric=metric)
        save_report_figure(fig_grid, target_dir / "compare_metrics.png")
        charts.append("compare_metrics.png")

    # 3. Construct MLflow compare URL
    from pipeline import tracking

    mlflow_uri = tracking.default_tracking_uri()
    mlflow_compare_url = None
    if mlflow_run_ids:
        ids_json = json.dumps(mlflow_run_ids)
        mlflow_compare_url = f"{mlflow_uri}/#/compare-runs?runs={ids_json}"

    # 4. Write comparison markdown report
    report_file = target_dir / "compare_report.md"
    lines: list[str] = [
        f"# Experiment Comparison: `{experiment}`",
        "",
        f"Evaluated **{len(matching_runs)}** runs across metric **`{metric.upper()}`**.",
        "",
    ]
    if mlflow_compare_url:
        lines.append(f"[Open in MLflow Multi-Run Compare UI]({mlflow_compare_url})")
        lines.append("")

    if variant_impl_pivot:
        all_impls = sorted(list({imp for row in variant_impl_pivot.values() for imp in row}))
        lines.append("## Accuracy Pivot Table")
        lines.append("")
        headers = ["Variant / Cell"] + all_impls
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join([":---"] + [":---:" for _ in all_impls]) + " |")
        for v_name, row in sorted(variant_impl_pivot.items()):
            cells = [f"`{v_name}`"]
            for imp in all_impls:
                if imp in row:
                    cells.append(f"**{row[imp]:.4f}**")
                else:
                    cells.append("-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if charts:
        lines.append("## Comparative Visualizations")
        lines.append("")
        for chart_name in charts:
            title = chart_name.replace("_", " ").replace(".png", "").title()
            lines.append(f"### {title}")
            lines.append(f"![{title}]({chart_name})")
            lines.append("")

    report_text = "\n".join(lines)
    report_file.write_text(report_text, encoding="utf-8")

    # Also write index.md pointing to same content for uniform HTTP browsing
    (target_dir / "index.md").write_text(report_text, encoding="utf-8")

    return {
        "experiment": experiment,
        "report_dir": str(target_dir),
        "report_file": str(report_file),
        "run_count": len(matching_runs),
        "charts": charts,
        "mlflow_compare_url": mlflow_compare_url,
    }
