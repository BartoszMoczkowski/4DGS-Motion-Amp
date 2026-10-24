"""Unit tests for pipeline.reporting.charts (T29)."""

import ast
from pathlib import Path

import pytest
from pipeline.reporting.charts import (
    IMPL_COLORS,
    BenchmarkBarPoint,
    DeformationSpectralData,
    FrameStripData,
    GaussianSweepPoint,
    GridCellEval,
    SegmentationPreviewData,
    TrainingCurvePoint,
    TrainingCurvesData,
    get_impl_color,
    plot_amp_frame_strip,
    plot_benchmark_bars,
    plot_deformation_spectral,
    plot_gaussian_sweep,
    plot_multi_run_training_curves,
    plot_segmentation_grid_scatter_heatmap,
    plot_segmentation_preview,
    plot_training_curves,
    save_report_figure,
)


def test_impl_colors_mapping():
    """Verify locked impl color mapping."""
    assert get_impl_color("rigid") == "#1f77b4"
    assert get_impl_color("rigid2") == "#2ca02c"
    assert get_impl_color("kabsch") == "#ff7f0e"
    assert get_impl_color("multicut") == "#9467bd"
    assert get_impl_color("mbs") == "#d62728"
    assert get_impl_color("unknown_impl") == "#7f7f7f"


def test_plot_training_curves():
    """plot_training_curves creates figure with loss and PSNR subplots."""
    pts = [
        TrainingCurvePoint(iteration=10, loss=0.5, l1=0.45, elapsed=1.0),
        TrainingCurvePoint(iteration=20, loss=0.2, l1=0.18, elapsed=2.0),
        TrainingCurvePoint(iteration=30, loss=0.05, l1=0.04, elapsed=3.0, test_psnr=28.5, train_psnr=29.1),
    ]
    data = TrainingCurvesData(run_id="run_test", points=pts)
    fig = plot_training_curves(data)

    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_multi_run_training_curves():
    """plot_multi_run_training_curves overlays curves for multiple runs."""
    r1 = TrainingCurvesData(run_id="run_1", points=[TrainingCurvePoint(iteration=10, loss=0.5), TrainingCurvePoint(iteration=20, loss=0.1)])
    r2 = TrainingCurvesData(run_id="run_2", points=[TrainingCurvePoint(iteration=10, loss=0.4), TrainingCurvePoint(iteration=20, loss=0.08)])

    fig = plot_multi_run_training_curves([r1, r2], metric_key="loss")
    assert fig is not None
    assert len(fig.axes) == 1
    # Legend should have entries for both runs
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2


def test_plot_segmentation_grid_scatter_heatmap():
    """plot_segmentation_grid_scatter_heatmap produces scatter and heatmap subplots."""
    records = [
        GridCellEval(cell="A20_M2", base_amp_mm=20.0, multiplier=2.0, impl="rigid", ari=0.92, mean_iou=0.85),
        GridCellEval(cell="A20_M2", base_amp_mm=20.0, multiplier=2.0, impl="kabsch", ari=0.95, mean_iou=0.88),
        GridCellEval(cell="A40_M4", base_amp_mm=40.0, multiplier=4.0, impl="rigid", ari=0.88, mean_iou=0.80),
        GridCellEval(cell="A40_M4", base_amp_mm=40.0, multiplier=4.0, impl="kabsch", ari=0.91, mean_iou=0.84),
    ]
    fig = plot_segmentation_grid_scatter_heatmap(records, metric="ari")
    assert fig is not None
    assert len(fig.axes) >= 2


def test_plot_deformation_spectral():
    """plot_deformation_spectral plots displacement and PSD."""
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    d_mean = [0.0, 1.2, 2.5, 1.8, 0.2]
    d_p95 = [0.0, 2.0, 4.0, 3.1, 0.5]
    freqs = [0.0, 5.0, 10.0, 15.0, 20.0]
    psd = [0.1, 0.5, 4.2, 0.8, 0.2]

    data = DeformationSpectralData(
        times=times,
        displacement_mean=d_mean,
        displacement_p95=d_p95,
        freqs_hz=freqs,
        psd=psd,
        dominant_freq_hz=10.0,
    )
    fig = plot_deformation_spectral(data)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_gaussian_sweep():
    """plot_gaussian_sweep plots trade-off curve."""
    pts = [
        GaussianSweepPoint(target_gaussians=10000, actual_gaussians=9800, psnr=26.5, peak_vram_mb=4000.0, train_time_s=60.0),
        GaussianSweepPoint(target_gaussians=25000, actual_gaussians=24500, psnr=28.2, peak_vram_mb=6000.0, train_time_s=120.0),
    ]
    fig = plot_gaussian_sweep(pts)
    assert fig is not None
    assert len(fig.axes) == 2  # twin axes


def test_plot_benchmark_bars():
    """plot_benchmark_bars plots time and VRAM bars."""
    pts = [
        BenchmarkBarPoint(name="train.default", wall_time_s=120.0, peak_vram_mb=6000.0),
        BenchmarkBarPoint(name="segment.rigid", wall_time_s=5.0, peak_vram_mb=0.0),
    ]
    fig = plot_benchmark_bars(pts)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_segmentation_preview():
    """plot_segmentation_preview plots 2D projections."""
    import numpy as np

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    lbls = np.array([0, 1, 0])
    data = SegmentationPreviewData(pred_points=pts, pred_labels=lbls, gt_points=pts, gt_labels=lbls)
    fig = plot_segmentation_preview(data)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_amp_frame_strip():
    """plot_amp_frame_strip plots contact sheet of frames."""
    import numpy as np

    f1 = np.zeros((32, 32, 3), dtype=np.uint8)
    f2 = np.ones((32, 32, 3), dtype=np.uint8) * 128
    data = FrameStripData(frames=[f1, f2], timestamps=[0.0, 0.5])
    fig = plot_amp_frame_strip(data)
    assert fig is not None
    assert len(fig.axes) == 2


def test_save_report_figure(tmp_path: Path):
    """save_report_figure closes figure and writes output file."""
    pts = [BenchmarkBarPoint(name="stageA", wall_time_s=1.0, peak_vram_mb=10.0)]
    fig = plot_benchmark_bars(pts)
    out_file = tmp_path / "test_bars.png"

    saved = save_report_figure(fig, out_file)
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_ast_scan_charts_pure_cpu():
    """Ensure reporting/charts.py has NO torch or tensorboard, and no module-scope heavy imports."""
    file_path = Path(__file__).resolve().parents[1] / "pipeline" / "reporting" / "charts.py"
    assert file_path.is_file()

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    forbidden = {"torch", "tensorboard"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden

    # Module-scope imports: check top-level body only
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                assert alias.name not in {"matplotlib", "numpy"}, f"Module-scope import {alias.name} in charts.py violates light-import rule"
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module:
                root_mod = stmt.module.split(".")[0]
                assert root_mod not in {"matplotlib", "numpy"}, f"Module-scope from-import {stmt.module} in charts.py violates light-import rule"
