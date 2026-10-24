"""Committed chart generation functions in house style (T29).

Provides prebuilt publication-quality plotting functions behind a committed matplotlib
style sheet (`house_style.mplstyle`) and a locked `impl→color` map.

Light-import rule (Layer 1 discipline):
NEVER import matplotlib or numpy at module scope! All plotting imports happen inside
functions or context manager to keep `pipeline` importable in sandbox tests and CLI startup light.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

#: Committed house colors per segmentation implementation
IMPL_COLORS: dict[str, str] = {
    "rigid": "#1f77b4",        # Blue
    "rigid2": "#2ca02c",       # Green
    "kabsch": "#ff7f0e",       # Orange
    "multicut": "#9467bd",     # Purple
    "mbs": "#d62728",          # Crimson
    "rigid2_roi": "#17becf",   # Cyan
    "mask_lift": "#8c564b",    # Brown
    "mask_lift_oracle": "#e377c2",  # Pink
}
DEFAULT_IMPL_COLOR = "#7f7f7f"


def get_impl_color(impl: str) -> str:
    """Return the committed house color for a segmentation implementation."""
    return IMPL_COLORS.get(impl.lower(), DEFAULT_IMPL_COLOR)


@contextlib.contextmanager
def house_style():
    """Context manager applying the committed house_style.mplstyle and non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path(__file__).parent / "house_style.mplstyle"
    if style_path.is_file():
        with plt.style.context(str(style_path)):
            yield
    else:
        yield


# --- Typed data contracts -------------------------------------------------------------------


@dataclass
class TrainingCurvePoint:
    iteration: int
    loss: float
    l1: Optional[float] = None
    elapsed: Optional[float] = None
    train_psnr: Optional[float] = None
    test_psnr: Optional[float] = None
    points: Optional[int] = None


@dataclass
class TrainingCurvesData:
    run_id: str
    points: list[TrainingCurvePoint] = field(default_factory=list)
    label: Optional[str] = None


@dataclass
class GridCellEval:
    cell: str
    base_amp_mm: float
    multiplier: float
    impl: str
    ari: float
    mean_iou: float


@dataclass
class DeformationSpectralData:
    times: Sequence[float]
    displacement_mean: Sequence[float]
    displacement_p95: Sequence[float]
    freqs_hz: Sequence[float]
    psd: Sequence[float]
    dominant_freq_hz: Optional[float] = None


@dataclass
class GaussianSweepPoint:
    target_gaussians: int
    actual_gaussians: int
    psnr: float
    peak_vram_mb: float
    train_time_s: float


@dataclass
class BenchmarkBarPoint:
    name: str
    wall_time_s: float
    peak_vram_mb: float
    category: Optional[str] = None


@dataclass
class SegmentationPreviewData:
    pred_points: Any  # shape (N, 3)
    pred_labels: Any  # shape (N,)
    gt_points: Optional[Any] = None  # shape (M, 3)
    gt_labels: Optional[Any] = None  # shape (M,)
    title: Optional[str] = None


@dataclass
class FrameStripData:
    frames: list[Any]  # list of HxWx3 uint8 numpy arrays or PIL Images
    timestamps: list[float]
    title: Optional[str] = None


# --- Chart functions -------------------------------------------------------------------------


def plot_training_curves(data: TrainingCurvesData) -> Any:
    """Plot single-run training loss, L1, and validation PSNR curves over global iterations."""
    import matplotlib.pyplot as plt

    with house_style():
        fig, (ax_loss, ax_psnr) = plt.subplots(1, 2, figsize=(11.0, 4.5))

        if not data.points:
            ax_loss.text(0.5, 0.5, "No training curves available", ha="center", va="center")
            ax_psnr.text(0.5, 0.5, "No PSNR data available", ha="center", va="center")
            return fig

        iters = [p.iteration for p in data.points]
        losses = [p.loss for p in data.points]
        l1s = [p.l1 for p in data.points if p.l1 is not None]
        l1_iters = [p.iteration for p in data.points if p.l1 is not None]

        ax_loss.plot(iters, losses, label="Total Loss", color="#1f77b4", linewidth=1.8)
        if l1s and len(l1s) == len(l1_iters):
            ax_loss.plot(l1_iters, l1s, label="L1 Loss", color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.85)

        ax_loss.set_title(f"Training Loss — {data.label or data.run_id}")
        ax_loss.set_xlabel("Global Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True)

        # PSNR subplot
        test_pts = [(p.iteration, p.test_psnr) for p in data.points if p.test_psnr is not None]
        train_pts = [(p.iteration, p.train_psnr) for p in data.points if p.train_psnr is not None]

        has_psnr = False
        if test_pts:
            has_psnr = True
            ax_psnr.plot([t[0] for t in test_pts], [t[1] for t in test_pts], label="Test PSNR", color="#2ca02c", marker="o", markersize=4, linewidth=1.8)
        if train_pts:
            has_psnr = True
            ax_psnr.plot([t[0] for t in train_pts], [t[1] for t in train_pts], label="Train PSNR", color="#17becf", marker="s", markersize=4, linestyle=":", linewidth=1.5)

        if not has_psnr:
            ax_psnr.text(0.5, 0.5, "No validation PSNR records", ha="center", va="center", color="#777777")
        else:
            ax_psnr.legend(loc="lower right")

        ax_psnr.set_title("Validation PSNR (dB)")
        ax_psnr.set_xlabel("Global Iteration")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.grid(True)

        fig.tight_layout()
        return fig


def plot_multi_run_training_curves(
    runs_data: list[TrainingCurvesData],
    metric_key: str = "loss",
) -> Any:
    """Overlaid training curves comparing multiple runs/variants on a single figure."""
    import matplotlib.pyplot as plt

    with house_style():
        fig, ax = plt.subplots(figsize=(9.0, 5.0))

        if not runs_data:
            ax.text(0.5, 0.5, "No runs to compare", ha="center", va="center")
            return fig

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for idx, rdata in enumerate(runs_data):
            color = colors[idx % len(colors)]
            label = rdata.label or rdata.run_id

            if metric_key == "loss":
                x = [p.iteration for p in rdata.points]
                y = [p.loss for p in rdata.points]
            elif metric_key in ("test_psnr", "psnr"):
                pts = [(p.iteration, p.test_psnr) for p in rdata.points if p.test_psnr is not None]
                x = [t[0] for t in pts]
                y = [t[1] for t in pts]
            else:
                x = [p.iteration for p in rdata.points]
                y = [getattr(p, metric_key, None) or 0.0 for p in rdata.points]

            if x and y:
                ax.plot(x, y, label=label, color=color, linewidth=1.8, alpha=0.9)

        ylabel = "PSNR (dB)" if "psnr" in metric_key.lower() else ("Loss" if metric_key == "loss" else metric_key)
        ax.set_title(f"Comparative Training Curves — {ylabel}")
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True)
        fig.tight_layout()
        return fig


def plot_segmentation_grid_scatter_heatmap(
    records: list[GridCellEval],
    metric: str = "ari",
) -> Any:
    """Faceted scatter plot and heatmap of segmentation accuracy vs motion amplitude & multiplier."""
    import matplotlib.pyplot as plt
    import numpy as np

    with house_style():
        fig, (ax_scatter, ax_heat) = plt.subplots(1, 2, figsize=(11.5, 4.8))

        if not records:
            ax_scatter.text(0.5, 0.5, "No evaluation records", ha="center", va="center")
            ax_heat.text(0.5, 0.5, "No evaluation records", ha="center", va="center")
            return fig

        # Left: Scatter of Metric vs total displacement (base_amp_mm * multiplier) faceted by impl
        by_impl: dict[str, list[GridCellEval]] = {}
        for r in records:
            by_impl.setdefault(r.impl, []).append(r)

        for impl, recs in by_impl.items():
            color = get_impl_color(impl)
            x = [r.base_amp_mm * r.multiplier for r in recs]
            y = [getattr(r, metric, 0.0) for r in recs]
            ax_scatter.scatter(x, y, label=impl, color=color, s=50, alpha=0.9, edgecolors="none")

        metric_label = metric.upper() if metric in ("ari", "iou") else metric.replace("_", " ").title()
        ax_scatter.set_title(f"{metric_label} vs Effective Motion (mm)")
        ax_scatter.set_xlabel("Effective Amplitude (base_amp_mm × multiplier)")
        ax_scatter.set_ylabel(metric_label)
        ax_scatter.set_ylim(-0.05, 1.05)
        ax_scatter.legend(title="Implementation", loc="lower right", fontsize=8)
        ax_scatter.grid(True)

        # Right: Heatmap across base_amp_mm x multiplier (average over impls)
        base_amps = sorted(list({r.base_amp_mm for r in records}))
        multipliers = sorted(list({r.multiplier for r in records}))

        if len(base_amps) > 1 and len(multipliers) > 1:
            matrix = np.full((len(base_amps), len(multipliers)), np.nan)
            for r in records:
                bi = base_amps.index(r.base_amp_mm)
                mi = multipliers.index(r.multiplier)
                val = getattr(r, metric, np.nan)
                if np.isnan(matrix[bi, mi]):
                    matrix[bi, mi] = val
                else:
                    matrix[bi, mi] = 0.5 * (matrix[bi, mi] + val)

            im = ax_heat.imshow(matrix, cmap="viridis", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
            ax_heat.set_xticks(range(len(multipliers)))
            ax_heat.set_xticklabels([f"×{m}" for m in multipliers])
            ax_heat.set_yticks(range(len(base_amps)))
            ax_heat.set_yticklabels([f"{a:.0f}mm" for a in base_amps])
            ax_heat.set_xlabel("Multiplier")
            ax_heat.set_ylabel("Base Amplitude")
            ax_heat.set_title(f"Mean {metric_label} Heatmap")
            fig.colorbar(im, ax=ax_heat, label=metric_label, shrink=0.8)
        else:
            # Bar chart fallback when grid is single-dimensional
            cells = sorted(list({r.cell for r in records}))
            x_pos = np.arange(len(cells))
            mean_vals = [np.mean([getattr(r, metric, 0.0) for r in records if r.cell == c]) for c in cells]
            ax_heat.bar(x_pos, mean_vals, color="#1f77b4", alpha=0.85, width=0.6)
            ax_heat.set_xticks(x_pos)
            ax_heat.set_xticklabels(cells, rotation=30, ha="right")
            ax_heat.set_ylim(0, 1.05)
            ax_heat.set_title(f"Mean {metric_label} by Cell")
            ax_heat.set_ylabel(metric_label)

        fig.tight_layout()
        return fig


def plot_deformation_spectral(data: DeformationSpectralData) -> Any:
    """Plot non-rigid displacement amplitude time-series and FFT spectral density."""
    import matplotlib.pyplot as plt

    with house_style():
        fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9.0, 6.0))

        # Top: time-series
        ax_time.plot(data.times, data.displacement_mean, label="Mean displacement", color="#1f77b4", linewidth=1.8)
        ax_time.plot(data.times, data.displacement_p95, label="P95 displacement", color="#ff7f0e", linestyle="--", linewidth=1.5)
        ax_time.set_title("Deformation Trajectory Over Time")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Displacement (mm)")
        ax_time.legend(loc="upper right")
        ax_time.grid(True)

        # Bottom: FFT Power Spectral Density
        ax_freq.plot(data.freqs_hz, data.psd, color="#2ca02c", linewidth=1.8, label="PSD")
        if data.dominant_freq_hz is not None:
            ax_freq.axvline(data.dominant_freq_hz, color="#d62728", linestyle=":", linewidth=2, label=f"Dominant: {data.dominant_freq_hz:.2f} Hz")
        ax_freq.set_title("FFT Displacement Spectral Density")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Power Spectral Density")
        ax_freq.legend(loc="upper right")
        ax_freq.grid(True)

        fig.tight_layout()
        return fig


def plot_gaussian_sweep(points: list[GaussianSweepPoint]) -> Any:
    """Plot PSNR and Peak VRAM / Training Time against Gaussian count sweep."""
    import matplotlib.pyplot as plt

    with house_style():
        fig, ax_psnr = plt.subplots(figsize=(8.5, 4.8))
        ax_vram = ax_psnr.twinx()

        if not points:
            ax_psnr.text(0.5, 0.5, "No Gaussian sweep data", ha="center", va="center")
            return fig

        pts = sorted(points, key=lambda p: p.target_gaussians)
        x = [p.target_gaussians for p in pts]
        psnr_vals = [p.psnr for p in pts]
        vram_vals = [p.peak_vram_mb for p in pts]

        l1 = ax_psnr.plot(x, psnr_vals, marker="o", color="#1f77b4", linewidth=2.0, label="PSNR (dB)")
        l2 = ax_vram.plot(x, vram_vals, marker="s", color="#ff7f0e", linestyle="--", linewidth=1.8, label="Peak VRAM (MB)")

        ax_psnr.set_title("Gaussian Count Sweep Trade-off")
        ax_psnr.set_xlabel("Initial / Target Gaussian Count")
        ax_psnr.set_ylabel("PSNR (dB)", color="#1f77b4")
        ax_vram.set_ylabel("Peak VRAM (MB)", color="#ff7f0e")

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax_psnr.legend(lines, labels, loc="lower right")
        ax_psnr.grid(True)

        fig.tight_layout()
        return fig


def plot_benchmark_bars(points: list[BenchmarkBarPoint]) -> Any:
    """Plot wall time (s) and peak VRAM (MB) across stages or model variants."""
    import matplotlib.pyplot as plt
    import numpy as np

    with house_style():
        fig, (ax_time, ax_vram) = plt.subplots(1, 2, figsize=(10.5, 4.5))

        if not points:
            ax_time.text(0.5, 0.5, "No benchmark data", ha="center", va="center")
            ax_vram.text(0.5, 0.5, "No benchmark data", ha="center", va="center")
            return fig

        names = [p.name for p in points]
        y_pos = np.arange(len(names))
        times = [p.wall_time_s for p in points]
        vrams = [p.peak_vram_mb for p in points]

        ax_time.barh(y_pos, times, color="#1f77b4", alpha=0.85)
        ax_time.set_yticks(y_pos)
        ax_time.set_yticklabels(names)
        ax_time.invert_yaxis()
        ax_time.set_xlabel("Wall Time (s)")
        ax_time.set_title("Execution Wall Time")
        ax_time.grid(axis="x")

        ax_vram.barh(y_pos, vrams, color="#ff7f0e", alpha=0.85)
        ax_vram.set_yticks(y_pos)
        ax_vram.set_yticklabels([])
        ax_vram.invert_yaxis()
        ax_vram.set_xlabel("Peak VRAM (MB)")
        ax_vram.set_title("Peak GPU Memory")
        ax_vram.grid(axis="x")

        fig.tight_layout()
        return fig


def plot_segmentation_preview(data: SegmentationPreviewData) -> Any:
    """Multi-view 2D projection preview of predicted vs ground-truth motion clusters."""
    import matplotlib.pyplot as plt
    import numpy as np

    with house_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
        (ax_pred, ax_gt) = axes

        pred_pts = np.asarray(data.pred_points)
        pred_lbls = np.asarray(data.pred_labels)

        if len(pred_pts) > 0 and len(pred_lbls) == len(pred_pts):
            # Downsample if too dense for crisp rendering
            if len(pred_pts) > 5000:
                idx = np.random.choice(len(pred_pts), size=5000, replace=False)
                sub_pts, sub_lbls = pred_pts[idx], pred_lbls[idx]
            else:
                sub_pts, sub_lbls = pred_pts, pred_lbls

            cmap = plt.get_cmap("tab20")
            ax_pred.scatter(sub_pts[:, 0], sub_pts[:, 1], c=sub_lbls, cmap=cmap, s=3, alpha=0.7)
            ax_pred.set_title("Predicted Motion Clusters")
        else:
            ax_pred.text(0.5, 0.5, "No predicted point cloud", ha="center", va="center")

        ax_pred.set_xlabel("X (m)")
        ax_pred.set_ylabel("Y (m)")
        ax_pred.grid(True)
        ax_pred.set_aspect("equal", adjustable="datalim")

        if data.gt_points is not None and data.gt_labels is not None:
            gt_pts = np.asarray(data.gt_points)
            gt_lbls = np.asarray(data.gt_labels)
            if len(gt_pts) > 5000:
                idx = np.random.choice(len(gt_pts), size=5000, replace=False)
                gt_pts, gt_lbls = gt_pts[idx], gt_lbls[idx]

            cmap_gt = plt.get_cmap("tab20")
            ax_gt.scatter(gt_pts[:, 0], gt_pts[:, 1], c=gt_lbls, cmap=cmap_gt, s=3, alpha=0.7)
            ax_gt.set_title("Ground-Truth Parts")
        else:
            ax_gt.text(0.5, 0.5, "No ground-truth point cloud", ha="center", va="center")

        ax_gt.set_xlabel("X (m)")
        ax_gt.set_ylabel("Y (m)")
        ax_gt.grid(True)
        ax_gt.set_aspect("equal", adjustable="datalim")

        if data.title:
            fig.suptitle(data.title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def plot_amp_frame_strip(data: FrameStripData) -> Any:
    """Horizontal contact sheet of video frames across time."""
    import matplotlib.pyplot as plt
    import numpy as np

    with house_style():
        n = len(data.frames)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No frames provided", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(1, n, figsize=(min(3.0 * n, 15.0), 3.0))
        if n == 1:
            axes = [axes]

        for idx, (frame, t) in enumerate(zip(data.frames, data.timestamps)):
            arr = np.asarray(frame)
            axes[idx].imshow(arr)
            axes[idx].set_title(f"t = {t:.2f}s", fontsize=9)
            axes[idx].axis("off")

        if data.title:
            fig.suptitle(data.title, fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig


def save_report_figure(fig: Any, output_path: Path | str, dpi: int = 200) -> Path:
    """Save a matplotlib Figure to disk, close it, and return its Path."""
    import matplotlib.pyplot as plt

    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi)
    plt.close(fig)
    return p
