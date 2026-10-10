"""Unified experiment results store (T27): ``runs/experiment_results.csv``.

One wide, nullable schema replaces the ~13 per-script CSV variants. **Manifests are the source
of truth** — :func:`collect_results` re-derives every row from on-disk run manifests (run
``metadata`` for experiment/cell/variant identity, ``StageRecord``s for timings/peak VRAM,
per-impl ``seg_eval_result_<impl>.json`` files / artifact metadata for metrics, and
``count_gaussians`` for the trained Gaussian count); :func:`export_csv` (re)generates the store
keyed ``(run_id, impl)``, latest-wins. The runner additionally appends rows as phases finish
(:func:`append_result`), but an ``amp exp export`` always rebuilds the same content.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ..artifacts import get_runs_root, load_manifest
from ..artifacts.manifest import ManifestError
from .variants import count_gaussians

COLUMNS = [
    "run_id", "experiment", "cell", "variant", "impl", "status",
    "ari", "mean_iou", "n_gt", "n_pred",
    "ari_within_roi", "n_roi_points", "separability_auroc",
    "target_gaussians", "actual_gaussians",
    "convert_s", "train_s", "render_s", "seg_extract_s", "segment_s",
    "train_peak_vram_mb", "error",
]

#: Default store location (under the runs root so PIPELINE_RUNS_ROOT overrides apply).
STORE_FILENAME = "experiment_results.csv"


def _stage_time(manifest: Any, stage: str) -> Any:
    rec = manifest.stages.get(stage)
    return rec.wall_time_s if rec is not None and rec.wall_time_s is not None else ""


def _eval_summary(manifest: Any, artifact_name: str) -> dict[str, Any]:
    """The seg_eval summary for one per-impl artifact: prefer the JSON file, fall back to the
    artifact metadata recorded at stage time (ari/mean_iou only)."""
    art = manifest.artifacts.get(artifact_name)
    if art is None:
        return {}
    path = Path(art.path)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return dict(art.metadata)


def rows_for_manifest(manifest: Any, run_dir: Path) -> list[dict[str, Any]]:
    """All rows (pipeline-phase + per-impl seg rows) for one manifest; ``[]`` when the run
    carries no ``metadata.experiment``."""
    meta = manifest.metadata
    if not meta.get("experiment"):
        return []
    rows: list[dict[str, Any]] = []
    base = {
        "run_id": manifest.run_id,
        "experiment": meta.get("experiment", ""),
        "cell": meta.get("cell", ""),
        "variant": meta.get("variant", ""),
        "impl": "",
        "status": manifest.status,
        "target_gaussians": meta.get("target_gaussians", ""),
        "actual_gaussians": count_gaussians(run_dir) or "",
        "convert_s": _stage_time(manifest, "convert.default"),
        "train_s": _stage_time(manifest, "train.default"),
        "render_s": _stage_time(manifest, "render.default"),
        "seg_extract_s": _stage_time(manifest, "seg_extract.default"),
        "train_peak_vram_mb": (
            manifest.stages["train.default"].peak_vram_mb
            if "train.default" in manifest.stages
            and manifest.stages["train.default"].peak_vram_mb is not None
            else ""
        ),
        "error": next(
            (f"{n}: {r.error}" for n, r in manifest.stages.items() if r.status == "failed"),
            "",
        ),
    }
    rows.append({**{c: "" for c in COLUMNS}, **base})

    for art_name in sorted(manifest.artifacts):
        if not art_name.startswith("seg_eval_result"):
            continue
        impl = art_name.removeprefix("seg_eval_result_") if art_name != "seg_eval_result" else ""
        summary = _eval_summary(manifest, art_name)
        seg_rec = manifest.stages.get(f"segment.{impl}")
        if seg_rec is None:  # entry name != stage impl (e.g. rigid2_roi -> segment.rigid2)
            seg_rec = next(
                (r for n, r in manifest.stages.items()
                 if n.startswith("segment.") and r.status in ("success", "skipped")),
                None,
            )
        sep = manifest.artifacts.get("separability")
        sep_auroc = (sep.metadata.get("auroc") or sep.metadata.get("edge_auroc") or "") if sep else ""
        row = dict(base)
        row.update({
            "impl": impl,
            "ari": summary.get("ari", ""),
            "mean_iou": summary.get("mean_iou", ""),
            "n_gt": summary.get("n_gt", ""),
            "n_pred": summary.get("n_pred", ""),
            "ari_within_roi": summary.get("ari_within_roi", ""),
            "n_roi_points": summary.get("n_roi_points", ""),
            "separability_auroc": sep_auroc,
            "segment_s": seg_rec.wall_time_s if seg_rec and seg_rec.wall_time_s is not None else "",
        })
        rows.append({**{c: "" for c in COLUMNS}, **row})
    return rows


def collect_results(runs_root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Re-derive result rows from every run manifest carrying ``metadata.experiment``.

    Corrupt manifests are skipped; ad-hoc/debug runs (no experiment metadata) are excluded.
    """
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            manifest = load_manifest(run_dir.name, runs_root=root)
        except (FileNotFoundError, ManifestError):
            continue
        rows.extend(rows_for_manifest(manifest, run_dir))
    return rows


def export_csv(rows: Iterable[dict[str, Any]], path: Optional[Path] = None, *,
               runs_root: Optional[Path] = None) -> Path:
    """Write the store keyed ``(run_id, impl)``, latest-wins; sorted for stable diffs."""
    out = Path(path) if path is not None else (
        (Path(runs_root) if runs_root is not None else get_runs_root()) / STORE_FILENAME
    )
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(str(row.get("run_id", "")), str(row.get("impl", "")))] = row
    ordered = sorted(
        keyed.values(),
        key=lambda r: (str(r.get("experiment", "")), str(r.get("cell", "")),
                       str(r.get("variant", "")), str(r.get("run_id", "")), str(r.get("impl", ""))),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in ordered:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    return out


def append_result(row: dict[str, Any], path: Optional[Path] = None, *,
                  runs_root: Optional[Path] = None) -> Path:
    """Insert-or-replace one row (keyed ``(run_id, impl)``) in the store."""
    out = Path(path) if path is not None else (
        (Path(runs_root) if runs_root is not None else get_runs_root()) / STORE_FILENAME
    )
    existing: list[dict[str, Any]] = []
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    existing.append(row)
    return export_csv(existing, out)
