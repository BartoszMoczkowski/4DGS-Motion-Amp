"""Unified experiment navigator (T30): single high-density inspection entry point.

Traverses experiments, run configs, stage progress, registered artifacts, un-manifested files,
datasets/captures, and metrics in ONE shot — eliminating 10-30 sequential tool calls.

Default output is structured JSON (for MCP agents and programmatic tools); CLI provides
`--human` flag for formatted terminal tables/markdown trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pipeline.artifacts import (
    RunManifest,
    get_manifest,
    get_runs_root,
    list_runs,
    run_dir,
)
from pipeline.config import list_presets, resolve_preset
from pipeline.paths import get_roots


def _find_repo_root() -> Path:
    try:
        return get_roots().repo_root_host
    except Exception:
        return Path(__file__).resolve().parents[2]


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _diff_configs(base: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    f_base = _flatten_dict(base)
    f_curr = _flatten_dict(current)
    diffs = []
    for k in sorted(set(f_base) | set(f_curr)):
        v_base = f_base.get(k, "<unset>")
        v_curr = f_curr.get(k, "<unset>")
        if v_base != v_curr:
            diffs.append({"key": k, "base": v_base, "current": v_curr})
    return diffs


def _gather_metrics(r_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for p in sorted(r_dir.glob("seg_eval_result*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            impl = p.stem.replace("seg_eval_result_", "").replace("seg_eval_result", "default")
            metrics[f"segmentation_{impl}"] = {
                "ari": data.get("ari"),
                "mean_iou": data.get("mean_iou"),
                "n_clusters": data.get("n_clusters") or data.get("k_pred"),
                "n_points": data.get("n_points"),
            }
        except Exception:
            pass

    for p in sorted(r_dir.glob("*_out/metrics.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            stage_name = p.parent.name.replace("_out", "")
            metrics[stage_name] = data.get("metrics", data)
        except Exception:
            pass

    return metrics


def _scan_unregistered_files(
    r_dir: Path, manifest: RunManifest, runs_root: Path
) -> list[dict[str, Any]]:
    if not r_dir.is_dir():
        return []

    registered_paths: set[Path] = set()
    for art in manifest.artifacts.values():
        try:
            p = Path(art.path)
            if not p.is_absolute():
                p = (runs_root / p).resolve()
            registered_paths.add(p.resolve())
            if p.is_dir():
                for sub in p.rglob("*"):
                    registered_paths.add(sub.resolve())
        except Exception:
            pass

    # Standard manifest and stage log paths
    registered_paths.add((r_dir / "manifest.json").resolve())
    for stage_name in manifest.stages:
        registered_paths.add((r_dir / "logs" / f"{stage_name}.log").resolve())

    unregistered: list[dict[str, Any]] = []
    for item in sorted(r_dir.rglob("*")):
        if item.is_file():
            resolved = item.resolve()
            if resolved not in registered_paths:
                try:
                    rel = item.relative_to(r_dir).as_posix()
                except ValueError:
                    rel = str(item)
                unregistered.append({
                    "path": rel,
                    "size_bytes": item.stat().st_size,
                })
    return unregistered


def _explore_run(
    run_id: str,
    view: str = "default",
    include_unregistered: bool = False,
    runs_root: Optional[Path] = None,
) -> dict[str, Any]:
    r_root = runs_root or get_runs_root()
    manifest = get_manifest(run_id, runs_root=r_root)
    r_dir = run_dir(run_id, runs_root=r_root)

    # Config diff against base preset
    base_cfg: dict[str, Any] = {}
    try:
        base_cfg = resolve_preset("base")
    except Exception:
        pass
    config_diff = _diff_configs(base_cfg, manifest.resolved_config)

    # Timeline of stages
    timeline = []
    total_time = 0.0
    for name, rec in manifest.stages.items():
        duration = rec.wall_time_s
        if duration is not None:
            total_time += duration
        entry: dict[str, Any] = {
            "stage": name,
            "status": rec.status,
            "start_time": rec.start_time,
            "end_time": rec.end_time,
            "duration_s": round(duration, 2) if duration is not None else None,
            "peak_vram_mb": rec.peak_vram_mb,
            "peak_ram_mb": rec.peak_ram_mb,
            "error": rec.error,
            "artifacts": rec.artifacts,
        }
        timeline.append(entry)

    # Registered artifacts
    artifacts = []
    for art in manifest.artifacts.values():
        p = Path(art.path)
        artifacts.append({
            "name": art.name,
            "kind": art.kind,
            "path": art.path,
            "producing_stage": art.producing_stage,
            "content_hash": art.content_hash,
            "size_bytes": p.stat().st_size if p.is_file() else None,
        })

    result: dict[str, Any] = {
        "kind": "run",
        "run_id": manifest.run_id,
        "preset": manifest.preset,
        "status": manifest.status,
        "git_sha": manifest.git_sha,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "metadata": manifest.metadata,
        "mlflow_url": manifest.metadata.get("mlflow", {}).get("url"),
        "total_duration_s": round(total_time, 2),
        "timeline": timeline,
        "artifacts": artifacts,
        "metrics": _gather_metrics(r_dir),
    }

    if view in ("full", "diff") or config_diff:
        result["config_diff"] = config_diff

    if include_unregistered or view == "full":
        result["unregistered_files"] = _scan_unregistered_files(r_dir, manifest, r_root)

    return result


def _explore_experiment(
    exp_name: str,
    view: str = "default",
    experiments_dir: Optional[Path] = None,
    runs_root: Optional[Path] = None,
) -> dict[str, Any]:
    repo_root = _find_repo_root()
    exp_dir = experiments_dir or (repo_root / "experiments")
    r_root = runs_root or get_runs_root()

    spec_path = exp_dir / f"{exp_name}.yaml"
    if not spec_path.is_file() and exp_name.endswith(".yaml"):
        spec_path = Path(exp_name)
    spec_data: Optional[dict[str, Any]] = None
    if spec_path.is_file():
        import yaml

        try:
            spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Find matching runs
    all_runs = list_runs(runs_root=r_root)
    linked_runs = []
    for r in all_runs:
        rid = r["run_id"]
        try:
            m = get_manifest(rid, runs_root=r_root)
            exp_tag = m.metadata.get("experiment")
            if exp_tag == exp_name or rid.startswith(f"{exp_name}-"):
                linked_runs.append({
                    "run_id": rid,
                    "preset": m.preset,
                    "status": m.status,
                    "cell": m.metadata.get("cell"),
                    "variant": m.metadata.get("variant"),
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                    "stages": {name: rec.status for name, rec in m.stages.items()},
                    "mlflow_url": m.metadata.get("mlflow", {}).get("url"),
                    "metrics": _gather_metrics(run_dir(rid, runs_root=r_root)),
                })
        except Exception:
            continue

    # Results pivot
    results_csv = r_root / "experiment_results.csv"
    pivoted_results: list[dict[str, Any]] = []
    if results_csv.is_file():
        import csv

        try:
            with results_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("experiment") == exp_name or any(
                        r["run_id"] == row.get("run_id") for r in linked_runs
                    ):
                        pivoted_results.append(row)
        except Exception:
            pass

    result: dict[str, Any] = {
        "kind": "experiment",
        "name": exp_name,
        "spec_file": str(spec_path) if spec_path.is_file() else None,
        "spec": spec_data,
        "linked_runs_count": len(linked_runs),
        "linked_runs": linked_runs,
        "results_table": pivoted_results,
    }
    return result


def _explore_capture(cap_dir: Path) -> dict[str, Any]:
    resolved = cap_dir.resolve()
    # Check for images / frames
    frame_count = 0
    img_dirs = [resolved / "images", resolved / "input", resolved]
    for d in img_dirs:
        if d.is_dir():
            count = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
            if count > frame_count:
                frame_count = count

    # Camera metadata
    cameras: list[str] = []
    cam_json = resolved / "cameras_gt.json"
    if not cam_json.is_file():
        cam_json = resolved / "cameras.json"
    transforms_json = list(resolved.glob("transforms*.json"))

    if cam_json.is_file():
        try:
            cdata = json.loads(cam_json.read_text(encoding="utf-8"))
            if isinstance(cdata, list):
                cameras = [f"cam_{i}" for i in range(len(cdata))]
        except Exception:
            pass
    elif transforms_json:
        try:
            tdata = json.loads(transforms_json[0].read_text(encoding="utf-8"))
            frames = tdata.get("frames", [])
            cameras = [f.get("file_path", f"frame_{i}") for i, f in enumerate(frames)]
        except Exception:
            pass

    # Poses & bounds
    pb_stats: Optional[dict[str, Any]] = None
    pb_file = resolved / "poses_bounds.npy"
    if pb_file.is_file():
        try:
            import numpy as np

            pb = np.load(pb_file)
            pb_stats = {
                "shape": list(pb.shape),
                "near_min": float(np.min(pb[:, -2])),
                "far_max": float(np.max(pb[:, -1])),
            }
        except Exception:
            pass

    # COLMAP sparse point cloud
    colmap_stats: Optional[dict[str, Any]] = None
    sparse_dir = resolved / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = resolved / "colmap" / "sparse" / "0"
    if sparse_dir.is_dir():
        colmap_stats = {
            "sparse_dir": str(sparse_dir),
            "has_cameras": (sparse_dir / "cameras.bin").is_file()
            or (sparse_dir / "cameras.txt").is_file(),
            "has_images": (sparse_dir / "images.bin").is_file()
            or (sparse_dir / "images.txt").is_file(),
            "has_points": (sparse_dir / "points3D.bin").is_file()
            or (sparse_dir / "points3D.txt").is_file()
            or (sparse_dir / "points3D.ply").is_file(),
        }

    # Ground truth segmentation
    gt_stats: Optional[dict[str, Any]] = None
    gt_file = resolved / "gt_segmentation.npz"
    if not gt_file.is_file():
        gt_file = resolved / "points3D_labels.npy"
    if gt_file.is_file():
        try:
            import numpy as np

            if gt_file.suffix == ".npz":
                with np.load(gt_file) as data:
                    seg = data.get("segmentation") or data.get("labels") or data.get("arr_0")
                    if seg is not None:
                        unique = np.unique(seg)
                        gt_stats = {"parts_count": len(unique), "point_count": int(seg.size)}
            else:
                arr = np.load(gt_file)
                gt_stats = {"parts_count": len(np.unique(arr)), "point_count": int(arr.size)}
        except Exception:
            pass

    return {
        "kind": "capture",
        "name": resolved.name,
        "path": str(resolved),
        "frame_count": frame_count,
        "camera_count": len(cameras),
        "poses_bounds": pb_stats,
        "colmap": colmap_stats,
        "ground_truth": gt_stats,
    }


def _explore_overview(
    experiments_dir: Optional[Path] = None, runs_root: Optional[Path] = None
) -> dict[str, Any]:
    repo_root = _find_repo_root()
    exp_dir = experiments_dir or (repo_root / "experiments")
    r_root = runs_root or get_runs_root()

    # List experiments
    experiments: list[dict[str, Any]] = []
    if exp_dir.is_dir():
        for p in sorted(exp_dir.glob("*.yaml")):
            try:
                import yaml

                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                experiments.append({
                    "name": p.stem,
                    "spec_file": str(p),
                    "pipeline_preset": doc.get("pipeline", {}).get("preset"),
                    "cells": doc.get("capture", {}).get("cells", []),
                    "segment_impls": [e.get("impl") for e in doc.get("segment", []) if isinstance(e, dict)],
                })
            except Exception:
                experiments.append({"name": p.stem, "spec_file": str(p)})

    # List recent runs
    raw_runs = list_runs(runs_root=r_root)
    recent_runs: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for r in raw_runs[:15]:
        rid = r["run_id"]
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        m_url = None
        try:
            m = get_manifest(rid, runs_root=r_root)
            m_url = m.metadata.get("mlflow", {}).get("url")
        except Exception:
            pass
        recent_runs.append({
            "run_id": rid,
            "preset": r.get("preset"),
            "status": status,
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "mlflow_url": m_url,
        })

    for r in raw_runs[15:]:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Detected captures/datasets
    detected_captures: list[dict[str, str]] = []
    candidates = [
        repo_root / "data" / "multipleview",
        repo_root / "data",
        repo_root / "omniverse-pipeline" / "data" / "scenes" / "grid",
    ]
    for parent in candidates:
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    # Check if likely a capture
                    if (
                        (child / "images").is_dir()
                        or (child / "cameras_gt.json").is_file()
                        or (child / "transforms_train.json").is_file()
                        or (child / "poses_bounds.npy").is_file()
                    ):
                        detected_captures.append({
                            "name": child.name,
                            "path": str(child),
                        })

    return {
        "kind": "overview",
        "total_runs": len(raw_runs),
        "status_breakdown": status_counts,
        "experiments": experiments,
        "recent_runs": recent_runs,
        "detected_captures": detected_captures,
    }


def explore(
    target: Optional[str] = None,
    view: str = "default",
    include_unregistered: bool = False,
    *,
    runs_root: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Unified inspection navigator.

    - Target is None or "": workspace overview (experiments, runs, captures).
    - Target is a run ID: returns full provenance, config diffs, timeline, artifacts, unregistered files.
    - Target is an experiment spec/name: returns variants, linked run IDs, execution statuses, metrics pivot.
    - Target is a capture path: returns frame counts, camera list, sync residuals, COLMAP/GT stats.
    """
    if not target or target.strip() in ("", "overview", "all"):
        return _explore_overview(experiments_dir=experiments_dir, runs_root=runs_root)

    t = target.strip()
    r_root = runs_root or get_runs_root()
    repo_root = _find_repo_root()
    exp_dir = experiments_dir or (repo_root / "experiments")

    # 1. Is it a run ID?
    try:
        if (r_root / t / "manifest.json").is_file():
            return _explore_run(
                t, view=view, include_unregistered=include_unregistered, runs_root=r_root
            )
    except Exception:
        pass

    # 2. Is it an experiment spec or experiment name?
    if (exp_dir / f"{t}.yaml").is_file() or (t.endswith(".yaml") and Path(t).is_file()):
        exp_name = t[:-5] if t.endswith(".yaml") else t
        return _explore_experiment(
            exp_name, view=view, experiments_dir=exp_dir, runs_root=r_root
        )

    # Check if any run has metadata.experiment == t
    for r in list_runs(runs_root=r_root):
        try:
            m = get_manifest(r["run_id"], runs_root=r_root)
            if m.metadata.get("experiment") == t:
                return _explore_experiment(
                    t, view=view, experiments_dir=exp_dir, runs_root=r_root
                )
        except Exception:
            continue

    # 3. Is it a capture directory or dataset name?
    cand_path = Path(t)
    if not cand_path.is_dir():
        for base in [
            repo_root / "data" / "multipleview" / t,
            repo_root / "data" / t,
            repo_root / "omniverse-pipeline" / "data" / "scenes" / "grid" / t,
        ]:
            if base.is_dir():
                cand_path = base
                break

    if cand_path.is_dir():
        return _explore_capture(cand_path)

    # Fallback: couldn't find exact match, return suggestions
    return {
        "kind": "not_found",
        "target": target,
        "error": f"target {target!r} not found as run ID, experiment spec, or dataset path",
        "overview": _explore_overview(experiments_dir=exp_dir, runs_root=r_root),
    }


def format_explore_human(data: dict[str, Any]) -> str:
    """Format exploration payload as readable terminal markdown tables and trees."""
    kind = data.get("kind", "unknown")
    lines: list[str] = []

    if kind == "overview":
        lines.append("# Workspace Overview\n")
        lines.append(f"Total Runs: {data.get('total_runs', 0)}")
        breakdown = data.get("status_breakdown", {})
        lines.append(f"Statuses: {', '.join(f'{k}: {v}' for k, v in breakdown.items())}\n")

        lines.append("## Experiments")
        for exp in data.get("experiments", []):
            cells = ", ".join(exp.get("cells", [])) or "custom"
            impls = ", ".join(exp.get("segment_impls", [])) or "standard"
            lines.append(f"- **{exp['name']}** (preset: {exp.get('pipeline_preset', 'base')})")
            lines.append(f"  cells: [{cells}] | impls: [{impls}]")

        lines.append("\n## Recent Runs")
        lines.append("| Run ID | Preset | Status | Created At | MLflow |")
        lines.append("|---|---|---|---|---|")
        for r in data.get("recent_runs", []):
            mlink = f"[link]({r['mlflow_url']})" if r.get("mlflow_url") else "-"
            lines.append(
                f"| `{r['run_id']}` | {r.get('preset')} | **{r.get('status')}** | {r.get('created_at', '')[:19]} | {mlink} |"
            )

        lines.append("\n## Detected Captures / Datasets")
        for cap in data.get("detected_captures", []):
            lines.append(f"- `{cap['name']}` ({cap['path']})")

    elif kind == "run":
        lines.append(f"# Run: {data['run_id']}\n")
        lines.append(f"- **Preset**: `{data.get('preset')}`")
        lines.append(f"- **Status**: `{data.get('status')}`")
        lines.append(f"- **Git SHA**: `{data.get('git_sha')}`")
        lines.append(f"- **Duration**: {data.get('total_duration_s')}s")
        if data.get("mlflow_url"):
            lines.append(f"- **MLflow Run**: {data['mlflow_url']}")

        lines.append("\n## Stage Timeline")
        lines.append("| Stage | Status | Duration (s) | Peak VRAM | Error |")
        lines.append("|---|---|---|---|---|")
        for s in data.get("timeline", []):
            dur = s.get("duration_s", "-")
            vram = f"{s['peak_vram_mb']}MB" if s.get("peak_vram_mb") else "-"
            err = s.get("error") or "-"
            lines.append(f"| `{s['stage']}` | **{s['status']}** | {dur} | {vram} | {err} |")

        if data.get("metrics"):
            lines.append("\n## Metrics")
            for m_group, vals in data["metrics"].items():
                lines.append(f"### {m_group}")
                if isinstance(vals, dict):
                    lines.append("| Metric | Value |")
                    lines.append("|---|---|")
                    for k, v in vals.items():
                        lines.append(f"| {k} | {v} |")

        if data.get("config_diff"):
            lines.append("\n## Config Diff vs Base")
            lines.append("| Setting | Base | Run Value |")
            lines.append("|---|---|---|")
            for d in data["config_diff"]:
                lines.append(f"| `{d['key']}` | `{d['base']}` | `{d['current']}` |")

        lines.append("\n## Registered Artifacts")
        for a in data.get("artifacts", []):
            sz = f" ({a['size_bytes']} bytes)" if a.get("size_bytes") else ""
            lines.append(f"- **{a['name']}** [{a['kind']}] -> `{a['path']}`{sz}")

        if data.get("unregistered_files"):
            lines.append("\n## Unregistered Files in Run Dir")
            for u in data["unregistered_files"]:
                lines.append(f"- `{u['path']}` ({u['size_bytes']} bytes)")

    elif kind == "experiment":
        lines.append(f"# Experiment: {data['name']}\n")
        if data.get("spec_file"):
            lines.append(f"- Spec File: `{data['spec_file']}`")
        lines.append(f"- Linked Runs: {data.get('linked_runs_count')}")

        lines.append("\n## Linked Runs")
        lines.append("| Run ID | Cell | Variant | Status | MLflow |")
        lines.append("|---|---|---|---|---|")
        for r in data.get("linked_runs", []):
            mlink = f"[link]({r['mlflow_url']})" if r.get("mlflow_url") else "-"
            lines.append(
                f"| `{r['run_id']}` | {r.get('cell') or '-'} | {r.get('variant') or '-'} | **{r['status']}** | {mlink} |"
            )

        if data.get("results_table"):
            lines.append("\n## Results Summary")
            rows = data["results_table"]
            headers = list(rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    elif kind == "capture":
        lines.append(f"# Capture / Dataset: {data['name']}\n")
        lines.append(f"- Path: `{data['path']}`")
        lines.append(f"- Frames: {data.get('frame_count')}")
        lines.append(f"- Cameras: {data.get('camera_count')}")
        if data.get("poses_bounds"):
            pb = data["poses_bounds"]
            lines.append(f"- Bounds: near={pb.get('near_min')}, far={pb.get('far_max')}")
        if data.get("colmap"):
            cm = data["colmap"]
            lines.append(f"- COLMAP: cameras={cm.get('has_cameras')}, points={cm.get('has_points')}")
        if data.get("ground_truth"):
            gt = data["ground_truth"]
            lines.append(f"- Ground Truth: {gt.get('parts_count')} parts, {gt.get('point_count')} labeled points")

    else:
        lines.append(f"# Target Not Found: {data.get('target')}")
        lines.append(data.get("error", "No details"))

    return "\n".join(lines)
