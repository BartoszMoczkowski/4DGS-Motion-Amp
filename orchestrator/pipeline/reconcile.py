"""Reconciliation utility (T28b) for manifests, MLflow tracking, and experiment results.

Fixes runs stuck in 'RUNNING' status, registers untracked physical captures in MLflow,
ingests external CSV metrics, uploads diagnostic plots, and refreshes experiment_results.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pipeline.artifacts import (
    get_runs_root,
    load_manifest,
    update_manifest,
    RunManifest,
)
from pipeline.artifacts.paths import manifest_path, run_dir

logger = logging.getLogger(__name__)


def reconcile_manifests(dry_run: bool = False, runs_root: Optional[Path] = None) -> dict[str, Any]:
    """Inspect all run manifests and fix manifest.status for runs whose executed stages completed."""
    root = runs_root or get_runs_root()
    stats = {"scanned": 0, "fixed_success": 0, "fixed_failed": 0, "unchanged": 0}

    for child in root.iterdir():
        if not child.is_dir() or not (child / "manifest.json").exists():
            continue
        stats["scanned"] += 1
        run_id = child.name
        try:
            m = load_manifest(run_id, runs_root=root)
        except Exception as exc:
            logger.warning("skipping corrupt manifest %s: %r", run_id, exc)
            continue

        if m.status in ("success", "failed"):
            stats["unchanged"] += 1
            continue

        executed_stages = [
            rec for rec in m.stages.values()
            if rec.status not in ("pending", None)
        ]
        if not executed_stages:
            stats["unchanged"] += 1
            continue

        has_failed = any(rec.status == "failed" for rec in executed_stages)
        all_done = all(rec.status in ("success", "skipped") for rec in executed_stages)

        new_status = None
        if has_failed:
            new_status = "failed"
            stats["fixed_failed"] += 1
        elif all_done:
            new_status = "success"
            stats["fixed_success"] += 1

        if new_status and not dry_run:
            def _set_status(manifest: RunManifest) -> None:
                manifest.status = new_status

            update_manifest(run_id, _set_status, runs_root=root)

    return stats


def reconcile_mlflow_runs(dry_run: bool = False, runs_root: Optional[Path] = None) -> dict[str, Any]:
    """Reconcile MLflow SQLite store with manifest statuses, backfill metrics and plots."""
    from pipeline import tracking

    root = runs_root or get_runs_root()
    stats = {
        "mlflow_terminated": 0,
        "mlflow_registered": 0,
        "metrics_ingested": 0,
        "plots_uploaded": 0,
    }

    if not tracking.is_enabled():
        return stats

    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow.tracking import MlflowClient

    client = None
    try:
        tracking._mlflow()
        test_client = MlflowClient()
        test_client.search_experiments()
        client = test_client
    except Exception:
        db_file = root / "mlflow" / "mlflow.db"
        if db_file.is_file():
            sqlite_uri = f"sqlite:///{db_file.resolve().as_posix()}"
            import os
            os.environ[tracking.TRACKING_URI_ENV] = sqlite_uri
            mlflow.set_tracking_uri(sqlite_uri)
            client = MlflowClient()

    if client is None:
        logger.warning("MLflow tracking store unreachable (neither HTTP server nor sqlite db)")
        return stats

    # 1. Update existing MLflow runs that are stuck in RUNNING
    from mlflow.entities import Metric

    mapping_file = root / "capture_limitation_trained_runs.json"
    cell_to_run: dict[str, str] = {}
    if mapping_file.is_file():
        try:
            cell_to_run = json.loads(mapping_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    for child in root.iterdir():
        if not child.is_dir() or not (child / "manifest.json").exists():
            continue
        run_id = child.name
        try:
            m = load_manifest(run_id, runs_root=root)
        except Exception:
            continue

        # Determine effective status (handles dry-run or manifests about to be fixed)
        effective_status = m.status
        if effective_status not in ("success", "failed"):
            executed = [
                rec for rec in m.stages.values()
                if rec.status not in ("pending", None)
            ]
            if executed:
                if any(rec.status == "failed" for rec in executed):
                    effective_status = "failed"
                elif all(rec.status in ("success", "skipped") for rec in executed):
                    effective_status = "success"

        meta = m.metadata or {}
        exp_name = meta.get("experiment") or m.preset
        exp = client.get_experiment_by_name(exp_name)

        existing_run = None
        if exp is not None:
            existing_run = tracking._find_run(client, exp.experiment_id, run_id)

        # If not tracked yet, register turbine runs
        if existing_run is None and ("turbine" in run_id or "lego_turbine" in m.preset):
            if not dry_run:
                if "experiment" not in meta:
                    def _add_meta(manifest: RunManifest) -> None:
                        manifest.metadata["experiment"] = "lego_turbine"
                    m = update_manifest(run_id, _add_meta, runs_root=root)

                mlflow_run_id = tracking.start_run(run_id, m.preset, m.resolved_config, m.metadata)
                if mlflow_run_id:
                    tracking.record_backlink(run_id, mlflow_run_id)
                    tracking.log_stage_metrics(run_id, m)
                    tracking.log_artifacts(run_id, m)
                    tracking.finish_run(run_id, m, effective_status)
                    stats["mlflow_registered"] += 1
            else:
                stats["mlflow_registered"] += 1
            continue

        if existing_run is not None:
            # Terminate stuck RUNNING runs
            if existing_run.info.status == "RUNNING" and effective_status in ("success", "failed"):
                if not dry_run:
                    term_status = "FINISHED" if effective_status == "success" else "FAILED"
                    client.set_tag(existing_run.info.run_id, tracking.TAG_STATUS, effective_status)
                    client.set_terminated(existing_run.info.run_id, status=term_status)
                    stats["mlflow_terminated"] += 1
                else:
                    stats["mlflow_terminated"] += 1

            # Upload segmentation renders / plots if present
            renders_dir = child / "segmentation_renders"
            if renders_dir.is_dir():
                for img in renders_dir.glob("*.png"):
                    if not dry_run:
                        try:
                            if img.stat().st_size < tracking.ARTIFACT_MAX_BYTES:
                                client.log_artifact(existing_run.info.run_id, str(img), artifact_path="segmentation_renders")
                                stats["plots_uploaded"] += 1
                        except Exception:
                            pass
                    else:
                        stats["plots_uploaded"] += 1

            # Upload evaluation diagnostic figures for capture limitation series
            figs_dir = root.parent / "docs" / "figures"
            run_to_cell = {v: k for k, v in cell_to_run.items()}
            cell_id = run_to_cell.get(run_id)
            if cell_id and figs_dir.is_dir():
                series_figs = ["composite_all_runs_camera_views.png"]
                if cell_id.startswith("sA"):
                    series_figs.append("fig4_series_a_motion_saturation.png")
                elif cell_id.startswith("sR"):
                    series_figs.extend(["fig2_radius_control_tradeoff.png", "fig3_resolution_vs_distance_ladder.png"])
                elif cell_id.startswith("sT") or cell_id.startswith("sF"):
                    series_figs.extend(["fig1_appearance_at_contacts_inversion.png", "fig5_material_tax_comparison.png"])

                for f_name in series_figs:
                    fig_path = figs_dir / f_name
                    if fig_path.is_file():
                        if not dry_run:
                            try:
                                client.log_artifact(existing_run.info.run_id, str(fig_path), artifact_path="evaluation_figures")
                                stats["plots_uploaded"] += 1
                            except Exception:
                                pass
                        else:
                            stats["plots_uploaded"] += 1

    # 2. Ingest capture_limitation_backbone_results.csv metrics into corresponding MLflow runs
    csv_path = root / "capture_limitation_backbone_results.csv"
    if csv_path.is_file():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("cell")
                    rid = cell_to_run.get(cid)
                    if not rid:
                        continue
                    m = load_manifest(rid, runs_root=root)
                    exp_name = m.metadata.get("experiment") or m.preset
                    exp = client.get_experiment_by_name(exp_name)
                    if not exp:
                        continue
                    run = tracking._find_run(client, exp.experiment_id, rid)
                    if not run:
                        continue

                    # Ingest numeric scalar metrics
                    metric_entities = []
                    for k, val in row.items():
                        if k in ("cell", "series", "res", "dist", "method", "dilation_mode"):
                            continue
                        try:
                            fval = float(val)
                            metric_entities.append(Metric(f"eval.{k}", fval, timestamp=0, step=0))
                        except (ValueError, TypeError):
                            pass

                    if metric_entities:
                        if not dry_run:
                            client.log_batch(run.info.run_id, metrics=metric_entities)
                        stats["metrics_ingested"] += len(metric_entities)
        except Exception as exc:
            logger.warning("could not ingest capture limitation csv: %r", exc)

    return stats


def reconcile_all(dry_run: bool = False, runs_root: Optional[Path] = None) -> dict[str, Any]:
    """Execute complete reconciliation: manifests, MLflow database, and experiment_results.csv."""
    m_stats = reconcile_manifests(dry_run=dry_run, runs_root=runs_root)
    ml_stats = reconcile_mlflow_runs(dry_run=dry_run, runs_root=runs_root)

    # Re-export experiment_results.csv
    export_stats = 0
    if not dry_run:
        try:
            from pipeline.experiments.results import collect_results, export_csv
            rows = collect_results(runs_root=runs_root)
            out_file = export_csv(rows, runs_root=runs_root)
            export_stats = 1 if out_file.is_file() else 0
        except Exception as exc:
            logger.warning("export_csv failed during reconciliation: %r", exc)

    return {
        "manifests": m_stats,
        "mlflow": ml_stats,
        "results_csv_refreshed": bool(export_stats),
    }
