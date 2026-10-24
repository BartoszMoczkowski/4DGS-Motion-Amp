"""Canonical metric registry and catalog for 4DGS Motion-Amp (T29).

Defines standard metric declarations across all pipeline phases (train, render, segment,
research, bench, amp) ensuring a single house style and uniform representation across
CLI, MLflow, Streamlit, MCP, and thesis figures.

Light-import rule: stdlib only (typing, dataclasses, pathlib, json, re, logging).
No torch, no tensorboard, no matplotlib.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricDef:
    """Canonical declaration for one pipeline or research metric."""

    name: str
    phase: str  # train | render | segment | research | bench | amp
    unit: str
    description: str
    source: str
    higher_is_better: Optional[bool] = None
    default_format: str = ".4f"

    def format_value(self, val: Any) -> str:
        """Format value according to metric unit and default precision."""
        if val is None:
            return "N/A"
        if isinstance(val, (int, bool)):
            return str(val)
        if isinstance(val, float):
            if self.unit in ("s", "sec"):
                if val >= 60.0:
                    mins = int(val // 60)
                    secs = val % 60
                    return f"{mins}m {secs:.1f}s"
                return f"{val:.2f}s"
            if self.unit in ("MB", "MiB"):
                return f"{val:.1f} MB"
            if self.unit in ("GB", "GiB"):
                return f"{val:.2f} GB"
            if self.unit in ("dB",):
                return f"{val:.2f} dB"
            if self.unit in ("Hz",):
                return f"{val:.2f} Hz"
            if self.unit in ("rpm", "RPM"):
                return f"{val:.1f} RPM"
            if self.unit in ("%", "ratio"):
                return f"{val * 100.0:.2f}%" if val <= 1.0 else f"{val:.2f}%"
            try:
                return f"{val:{self.default_format}}"
            except Exception:
                return str(val)
        return str(val)


METRIC_CATALOG: dict[str, MetricDef] = {
    # --- Train step & convergence metrics ---
    "total_loss": MetricDef(
        name="total_loss",
        phase="train",
        unit="",
        description="Total training objective loss (L1 + DSSIM + regularizers)",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".5f",
    ),
    "loss": MetricDef(
        name="loss",
        phase="train",
        unit="",
        description="Total training objective loss (alias for total_loss)",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".5f",
    ),
    "l1_loss": MetricDef(
        name="l1_loss",
        phase="train",
        unit="",
        description="L1 photometric reconstruction loss",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".5f",
    ),
    "l1": MetricDef(
        name="l1",
        phase="train",
        unit="",
        description="L1 photometric loss (alias for l1_loss)",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".5f",
    ),
    "iter_time_s": MetricDef(
        name="iter_time_s",
        phase="train",
        unit="s",
        description="Elapsed time for iteration step or evaluation window",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".3f",
    ),
    "elapsed": MetricDef(
        name="elapsed",
        phase="train",
        unit="s",
        description="Cumulative elapsed training time",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".1f",
    ),
    "train_psnr": MetricDef(
        name="train_psnr",
        phase="train",
        unit="dB",
        description="Peak Signal-to-Noise Ratio on training views",
        source="training_curves.json",
        higher_is_better=True,
        default_format=".2f",
    ),
    "test_psnr": MetricDef(
        name="test_psnr",
        phase="train",
        unit="dB",
        description="Peak Signal-to-Noise Ratio on held-out test views",
        source="training_curves.json",
        higher_is_better=True,
        default_format=".2f",
    ),
    "test_ssim": MetricDef(
        name="test_ssim",
        phase="train",
        unit="",
        description="Structural Similarity Index on held-out test views",
        source="training_curves.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "test_lpips": MetricDef(
        name="test_lpips",
        phase="train",
        unit="",
        description="Learned Perceptual Image Patch Similarity on test views",
        source="training_curves.json",
        higher_is_better=False,
        default_format=".4f",
    ),
    "total_points": MetricDef(
        name="total_points",
        phase="train",
        unit="count",
        description="Active 3D Gaussian count in reconstruction",
        source="training_curves.json",
        higher_is_better=None,
        default_format="d",
    ),
    "deformation_rate": MetricDef(
        name="deformation_rate",
        phase="train",
        unit="ratio",
        description="Fraction of Gaussians undergoing non-rigid deformation",
        source="training_curves.json",
        higher_is_better=None,
        default_format=".3f",
    ),

    # --- Render metrics ---
    "psnr": MetricDef(
        name="psnr",
        phase="render",
        unit="dB",
        description="Mean novel-view PSNR",
        source="render_out/metrics.json",
        higher_is_better=True,
        default_format=".2f",
    ),
    "ssim": MetricDef(
        name="ssim",
        phase="render",
        unit="",
        description="Mean novel-view SSIM",
        source="render_out/metrics.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "lpips": MetricDef(
        name="lpips",
        phase="render",
        unit="",
        description="Mean novel-view LPIPS",
        source="render_out/metrics.json",
        higher_is_better=False,
        default_format=".4f",
    ),

    # --- Segmentation metrics ---
    "ari": MetricDef(
        name="ari",
        phase="segment",
        unit="",
        description="Adjusted Rand Index vs ground truth labels",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "mean_iou": MetricDef(
        name="mean_iou",
        phase="segment",
        unit="",
        description="Mean best-match Intersection-over-Union across ground-truth parts",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "n_pred": MetricDef(
        name="n_pred",
        phase="segment",
        unit="count",
        description="Number of distinct predicted motion segments",
        source="seg_eval_result.json",
        higher_is_better=None,
        default_format="d",
    ),
    "n_gt": MetricDef(
        name="n_gt",
        phase="segment",
        unit="count",
        description="Number of ground-truth parts evaluated",
        source="seg_eval_result.json",
        higher_is_better=None,
        default_format="d",
    ),
    "ari_within_roi": MetricDef(
        name="ari_within_roi",
        phase="segment",
        unit="",
        description="Adjusted Rand Index evaluated exclusively inside Region of Interest",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "n_roi_points": MetricDef(
        name="n_roi_points",
        phase="segment",
        unit="count",
        description="Number of Gaussians within Region of Interest mask",
        source="seg_eval_result.json",
        higher_is_better=None,
        default_format="d",
    ),
    "n_floater_points": MetricDef(
        name="n_floater_points",
        phase="segment",
        unit="count",
        description="Number of discarded low-opacity / floater Gaussians",
        source="seg_eval_result.json",
        higher_is_better=False,
        default_format="d",
    ),
    "f1_macro": MetricDef(
        name="f1_macro",
        phase="segment",
        unit="",
        description="Macro-averaged F1 score over matched parts",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "mean_part_iou": MetricDef(
        name="mean_part_iou",
        phase="segment",
        unit="",
        description="Mean Intersection-over-Union across segmented parts",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "k_alignment": MetricDef(
        name="k_alignment",
        phase="segment",
        unit="ratio",
        description="Ratio of predicted cluster count to ground-truth part count (n_pred / n_gt)",
        source="seg_eval_result.json",
        higher_is_better=None,
        default_format=".3f",
    ),
    "k_ratio": MetricDef(
        name="k_ratio",
        phase="segment",
        unit="ratio",
        description="Alternative cluster alignment ratio",
        source="seg_eval_result.json",
        higher_is_better=None,
        default_format=".3f",
    ),

    # --- Research / domain evaluation (Capture Limitation & Turbine suites) ---
    "cat_a_split_rate": MetricDef(
        name="cat_a_split_rate",
        phase="research",
        unit="ratio",
        description="Interface split rate for adjacent part Category A",
        source="seg_eval_result.json",
        higher_is_better=False,
        default_format=".4f",
    ),
    "cat_b_split_rate": MetricDef(
        name="cat_b_split_rate",
        phase="research",
        unit="ratio",
        description="Interface split rate for adjacent part Category B",
        source="seg_eval_result.json",
        higher_is_better=False,
        default_format=".4f",
    ),
    "pairs_split_rate": MetricDef(
        name="pairs_split_rate",
        phase="research",
        unit="ratio",
        description="Fraction of adjacent boundary point pairs wrongly split or merged",
        source="seg_eval_result.json",
        higher_is_better=False,
        default_format=".4f",
    ),
    "interface_auroc": MetricDef(
        name="interface_auroc",
        phase="research",
        unit="",
        description="Area under ROC for boundary edge cut decision",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "target_iou": MetricDef(
        name="target_iou",
        phase="research",
        unit="",
        description="Intersection-over-Union on deviant target part (e.g. missing blade)",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "target_f1": MetricDef(
        name="target_f1",
        phase="research",
        unit="",
        description="F1 score on deviant target part",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "target_prec": MetricDef(
        name="target_prec",
        phase="research",
        unit="",
        description="Precision on deviant target part",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "target_rec": MetricDef(
        name="target_rec",
        phase="research",
        unit="",
        description="Recall on deviant target part",
        source="seg_eval_result.json",
        higher_is_better=True,
        default_format=".4f",
    ),
    "mean_displacement": MetricDef(
        name="mean_displacement",
        phase="research",
        unit="mm",
        description="Mean non-rigid displacement magnitude across Gaussians",
        source="seg_extract_out/metrics.json",
        higher_is_better=None,
        default_format=".3f",
    ),
    "p95_displacement": MetricDef(
        name="p95_displacement",
        phase="research",
        unit="mm",
        description="95th-percentile displacement amplitude",
        source="seg_extract_out/metrics.json",
        higher_is_better=None,
        default_format=".3f",
    ),
    "max_displacement": MetricDef(
        name="max_displacement",
        phase="research",
        unit="mm",
        description="Maximum observed Gaussian displacement amplitude",
        source="seg_extract_out/metrics.json",
        higher_is_better=None,
        default_format=".3f",
    ),
    "fft_dominant_freq_hz": MetricDef(
        name="fft_dominant_freq_hz",
        phase="research",
        unit="Hz",
        description="Dominant vibration or rotation frequency detected via FFT",
        source="seg_extract_out/metrics.json",
        higher_is_better=None,
        default_format=".2f",
    ),
    "estimated_rpm": MetricDef(
        name="estimated_rpm",
        phase="research",
        unit="RPM",
        description="Estimated shaft or impeller speed in revolutions per minute",
        source="seg_extract_out/metrics.json",
        higher_is_better=None,
        default_format=".1f",
    ),
    "n_seeds": MetricDef(
        name="n_seeds",
        phase="research",
        unit="count",
        description="Initial seed Gaussians selected for clustering",
        source="seg_out/metrics.json",
        higher_is_better=None,
        default_format="d",
    ),
    "n_labeled": MetricDef(
        name="n_labeled",
        phase="research",
        unit="count",
        description="Gaussians assigned to valid motion clusters",
        source="seg_out/metrics.json",
        higher_is_better=True,
        default_format="d",
    ),
    "n_discarded": MetricDef(
        name="n_discarded",
        phase="research",
        unit="count",
        description="Gaussians rejected as unclusterable noise",
        source="seg_out/metrics.json",
        higher_is_better=False,
        default_format="d",
    ),
    "k_pred": MetricDef(
        name="k_pred",
        phase="research",
        unit="count",
        description="Predicted rigid part cluster count",
        source="seg_out/metrics.json",
        higher_is_better=None,
        default_format="d",
    ),

    # --- Benchmark & resource metrics ---
    "wall_time_s": MetricDef(
        name="wall_time_s",
        phase="bench",
        unit="s",
        description="Stage wall-clock execution time",
        source="manifest.StageRecord",
        higher_is_better=False,
        default_format=".2f",
    ),
    "peak_vram_mb": MetricDef(
        name="peak_vram_mb",
        phase="bench",
        unit="MB",
        description="Peak GPU VRAM consumed during stage execution",
        source="manifest.StageRecord",
        higher_is_better=False,
        default_format=".1f",
    ),
    "peak_ram_mb": MetricDef(
        name="peak_ram_mb",
        phase="bench",
        unit="MB",
        description="Peak host system RAM consumed during stage execution",
        source="manifest.StageRecord",
        higher_is_better=False,
        default_format=".1f",
    ),

    # --- Motion amp metrics ---
    "runtime_s": MetricDef(
        name="runtime_s",
        phase="amp",
        unit="s",
        description="Motion-amplified video rendering time",
        source="amp_out/metrics.json",
        higher_is_better=False,
        default_format=".2f",
    ),
    "video_path": MetricDef(
        name="video_path",
        phase="amp",
        unit="path",
        description="Output amplified video file path",
        source="manifest.artifacts",
        higher_is_better=None,
        default_format="s",
    ),
}


def get_metric_def(name: str) -> Optional[MetricDef]:
    """Look up a metric definition in the canonical catalog."""
    return METRIC_CATALOG.get(name)


def format_metric(name: str, value: Any) -> str:
    """Format any metric value according to its catalog definition."""
    mdef = get_metric_def(name)
    if mdef is not None:
        return mdef.format_value(value)
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def extract_manifest_metrics(
    manifest: Any,
    run_dir_path: Optional[Path] = None,
) -> dict[str, dict[str, Any]]:
    """Extract and categorize all scalar metrics for a run into standard phases.

    Returns dict mapping phase name ('train', 'segment', 'render', 'bench', 'research', 'amp')
    to a dict of {metric_name: value}.
    """
    out: dict[str, dict[str, Any]] = {
        "train": {},
        "segment": {},
        "render": {},
        "research": {},
        "bench": {},
        "amp": {},
    }

    # 1. Benchmark metrics from StageRecords
    total_wall_time = 0.0
    max_peak_vram = 0.0
    for stage_name, rec in getattr(manifest, "stages", {}).items():
        if getattr(rec, "wall_time_s", None) is not None:
            out["bench"][f"{stage_name}.wall_time_s"] = rec.wall_time_s
            total_wall_time += rec.wall_time_s
        if getattr(rec, "peak_vram_mb", None) is not None:
            out["bench"][f"{stage_name}.peak_vram_mb"] = rec.peak_vram_mb
            max_peak_vram = max(max_peak_vram, rec.peak_vram_mb)
        if getattr(rec, "peak_ram_mb", None) is not None:
            out["bench"][f"{stage_name}.peak_ram_mb"] = rec.peak_ram_mb

    if total_wall_time > 0:
        out["bench"]["total_wall_time_s"] = total_wall_time
    if max_peak_vram > 0:
        out["bench"]["peak_vram_mb"] = max_peak_vram

    # 2. Artifact metadata metrics (seg_eval, amp, etc.)
    for art_name, art in getattr(manifest, "artifacts", {}).items():
        meta = getattr(art, "metadata", {})
        if not meta:
            continue
        # Segmentation eval metrics
        if art_name.startswith("seg_eval_result"):
            for k in ("ari", "mean_iou", "ari_within_roi", "f1_macro", "mean_part_iou", "k_alignment", "pairs_split_rate"):
                if k in meta and meta[k] is not None:
                    out["segment"][k] = meta[k]
        # Amp render metrics
        if art_name == "amplified_render" or getattr(art, "kind", "") == "video":
            if "runtime_s" in meta:
                out["amp"]["runtime_s"] = meta["runtime_s"]
            out["amp"]["video_path"] = art.path

    # 3. Disk-based metrics from run directory if provided
    if run_dir_path is not None and run_dir_path.is_dir():
        # Check training curves
        curves_file = run_dir_path / "train_out" / "training_curves.json"
        if curves_file.is_file():
            try:
                data = json.loads(curves_file.read_text(encoding="utf-8"))
                curves = data.get("curves", data) if isinstance(data, dict) else data
                if isinstance(curves, list) and curves:
                    last_pt = curves[-1]
                    for k in ("loss", "l1", "elapsed", "test_psnr", "train_psnr", "points"):
                        if k in last_pt and last_pt[k] is not None:
                            target_k = "total_loss" if k == "loss" else ("l1_loss" if k == "l1" else k)
                            out["train"][target_k] = last_pt[k]
            except Exception as exc:
                logger.debug("Failed parsing training_curves.json: %r", exc)

        # Check seg_eval_result*.json
        for json_path in run_dir_path.glob("seg_eval_result*.json"):
            try:
                res = json.loads(json_path.read_text(encoding="utf-8"))
                for k in ("ari", "mean_iou", "n_pred", "n_gt", "ari_within_roi", "f1_macro", "pairs_split_rate"):
                    if k in res and res[k] is not None:
                        out["segment"][k] = res[k]
                for rk in ("cat_a_split_rate", "cat_b_split_rate", "interface_auroc", "target_iou"):
                    if rk in res and res[rk] is not None:
                        out["research"][rk] = res[rk]
            except Exception as exc:
                logger.debug("Failed parsing %s: %r", json_path, exc)

        # Check generic stage metrics.json files
        for stage_dir in run_dir_path.iterdir():
            if stage_dir.is_dir() and stage_dir.name.endswith("_out"):
                m_file = stage_dir / "metrics.json"
                if m_file.is_file():
                    try:
                        m_data = json.loads(m_file.read_text(encoding="utf-8"))
                        metrics_map = m_data.get("metrics", m_data) if isinstance(m_data, dict) else {}
                        for mk, mv in metrics_map.items():
                            mdef = get_metric_def(str(mk))
                            phase = mdef.phase if mdef else "research"
                            out.setdefault(phase, {})[str(mk)] = mv
                    except Exception as exc:
                        logger.debug("Failed parsing %s: %r", m_file, exc)

    return out
