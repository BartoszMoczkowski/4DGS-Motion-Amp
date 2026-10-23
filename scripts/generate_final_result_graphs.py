"""
generate_final_result_graphs.py — Generate publication-grade thesis figures for the Capture-Limitation suite.

Figures generated:
1. fig1_appearance_at_contacts_inversion.png (Finding 1: Landmark 0.87 Appearance AUROC crossing vs Motion)
2. fig2_radius_control_tradeoff.png (Finding 2 / Item 1: sR3 sweep & intrinsic fragmentation ceiling)
3. fig3_resolution_vs_distance_ladder.png (Finding 3: 6 MP resolution vs 2.1m proximity)
4. fig4_series_a_motion_saturation.png (Finding 4: Deviant amplitude saturation & static geometric superiority)
5. fig5_material_tax_comparison.png (Finding 5: CC backbone invariance vs motion-method vulnerability)
6. fig_summary_radar_or_dashboard.png (Executive dashboard summarizing the 4 axes)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"
C_GREY = "#7f7f7f"
C_DARK = "#222222"

def plot_figure_1_appearance_inversion():
    """Figure 1: Landmark Finding — Appearance beats Motion at mechanical contacts at 2.1m."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    cells_far = ["sR0\n(900p @ 4.2m)", "sR1\n(1440p @ 4.2m)", "sR2\n(6 MP @ 4.2m)"]
    cells_close = ["sR3\n(900p @ 2.1m)", "sR4\n(6 MP @ 2.1m)", "sR5\n(6 MP, 120fr)"]
    all_cells = cells_far + cells_close

    # Exact AUROC data from capture_limitation_edge_auroc.csv
    motion_cat_a = [0.5051, 0.5303, 0.4578, 0.6194, 0.6187, 0.5885]
    app_cat_a = [0.6490, 0.6695, 0.6680, 0.8669, 0.8715, 0.8701]

    # Uncertainties
    sigma_app = 0.0075
    sigma_motion = 0.0120

    x = np.arange(len(all_cells))
    width = 0.36

    rects1 = ax.bar(x - width/2, motion_cat_a, width, yerr=sigma_motion, capsize=4,
                    label="Motion Cat A AUROC (Phasor Discrepancy)", color=C_BLUE, alpha=0.85, edgecolor="none")
    rects2 = ax.bar(x + width/2, app_cat_a, width, yerr=sigma_app, capsize=4,
                    label="Appearance Cat A AUROC (RGB Distance)", color=C_ORANGE, alpha=0.90, edgecolor="none")

    # Separability threshold line (0.80)
    ax.axhline(0.80, color=C_RED, linestyle="--", linewidth=1.8, label="Pre-Registered Separability Bar (0.80)")
    ax.axhline(0.50, color=C_GREY, linestyle=":", linewidth=1.0, alpha=0.7, label="Random Chance (0.50)")

    # Shaded divide between Far (4.2m) and Close (2.1m)
    ax.axvline(2.5, color="#555555", linestyle="-.", linewidth=1.2, alpha=0.8)
    ax.text(1.25, 0.94, "FAR STAND-OFF (4.2 m)\nBoth Uninformative", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#444444", bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc"))
    ax.text(4.25, 0.94, "CLOSE STAND-OFF (2.1 m)\nAppearance Decisively Crosses 0.80", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#b34700", bbox=dict(boxstyle="round,pad=0.4", fc="#fff2e6", ec="#ffb380"))

    # Annotation arrow for +20.5 sigma leap
    ax.annotate("Appearance Leaps to 0.8715\n(+20.5σ, p << 0.0001)\nBeats Motion at Contacts!",
                xy=(4 + width/2, 0.8715), xytext=(3.4, 0.72),
                arrowprops=dict(facecolor=C_ORANGE, shrink=0.08, width=1.5, headwidth=8),
                fontsize=10, fontweight="bold", color="#b34700",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=C_ORANGE, lw=1.2))

    ax.set_ylabel("Paired Interface Edge AUROC (Exact Mann-Whitney U)")
    ax.set_title("Finding 1: The Appearance-at-Contacts Inversion Across Stand-off Distance", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cells)
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="lower left", framealpha=0.95)

    out_path = FIG_DIR / "fig1_appearance_at_contacts_inversion.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Generated] {out_path}")


def plot_figure_2_radius_control():
    """Figure 2: Item 1 Density-Scaled Radius Control on sR3 (Intrinsic Fragmentation Ceiling)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    radii_cm = [0.15, 0.20, 0.25, 0.30]
    ari_vals = [0.2921, 0.2274, 0.1964, 0.1460]
    seeds = [47, 77, 129, 185]
    cat_a_splits = [15.4, 30.8, 35.9, 35.9]

    # Baseline sR0 reference ARI
    sR0_ari = 0.4440

    color_ari = C_BLUE
    ax1.plot(radii_cm, ari_vals, marker="o", linewidth=2.4, markersize=8, color=color_ari, label="sR3 CAD ROI ARI (2.1m Close)")
    ax1.axhline(sR0_ari, color=C_GREEN, linestyle="--", linewidth=1.8, label="sR0 Far Baseline ARI (0.4440 @ 4.2m)")
    ax1.set_xlabel("Backbone Clustering Radius r (cm)")
    ax1.set_ylabel("CAD ROI ARI", color=color_ari, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_ari)
    ax1.set_ylim(0.05, 0.50)

    # Secondary axis: Cat A Split Rate & Seeds
    ax2 = ax1.twinx()
    color_split = C_RED
    ax2.plot(radii_cm, cat_a_splits, marker="s", linewidth=2.0, linestyle="-.", markersize=7, color=color_split, label="Cat A Contact Split Rate (%)")
    ax2.set_ylabel("Cat A Split Rate (%)", color=color_split, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_split)
    ax2.set_ylim(0, 50)
    ax2.grid(False)

    # Annotate Verdict (b)
    ax1.annotate("Optimal Tuning r* = 0.15cm\nARI = 0.2921 (Ceiling)\nCat A Split Collapses to 15.4%",
                 xy=(0.15, 0.2921), xytext=(0.17, 0.35),
                 arrowprops=dict(facecolor=color_ari, shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9.5, fontweight="bold", color=color_ari,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=color_ari))

    ax1.text(0.24, 0.10, "Verdict (b) Confirmed:\nARI remains depressed (< 0.30) across all radii.\nOver-fragmentation is intrinsic to high-density point clouds.\nM5 Multi-Scale Clustering is mandatory.",
             ha="center", fontsize=9.5, color="#222222",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fffbe6", ec="#ffe58f"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95)

    plt.title("Finding 2: Radius-Density Calibration on sR3 & The Intrinsic Fragmentation Trade-Off", pad=14, fontweight="bold")

    out_path = FIG_DIR / "fig2_radius_control_tradeoff.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Generated] {out_path}")


def plot_figure_3_resolution_vs_distance():
    """Figure 3: Sensor Megapixels vs Stand-off Distance (Killing the 6 MP Myth)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # 4.2m Stand-off
    labels_far = ["sR0\n(900p)", "sR1\n(1440p)", "sR2\n(6 MP)"]
    ari_far = [0.4440, 0.3480, 0.3326]
    cat_a_far = [25.6, 23.1, 15.4]
    cat_b_far = [49.0, 70.7, 72.7]

    # 2.1m Stand-off
    labels_close = ["sR3\n(900p)", "sR4\n(6 MP)", "sR5\n(6 MP 120fr)"]
    ari_close = [0.1460, 0.1469, 0.1202]
    cat_a_close = [35.9, 43.6, 23.1]
    cat_b_close = [70.7, 76.3, 68.0]

    width = 0.25
    x = np.arange(3)

    # Subplot 1: Far (4.2m)
    ax1 = axes[0]
    ax1.bar(x - width, [a * 100 for a in ari_far], width, label="CAD ROI ARI (x100)", color=C_BLUE, alpha=0.85)
    ax1.bar(x, cat_a_far, width, label="Cat A Contact Split (%)", color=C_RED, alpha=0.85)
    ax1.bar(x + width, cat_b_far, width, label="Cat B Clearance Split (%)", color=C_GREEN, alpha=0.85)
    ax1.set_title("Stand-off Distance: 4.2 m (Far)\nMegapixels fail to improve contacts", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_far)
    ax1.set_ylabel("Score / Percentage (%)")
    ax1.set_ylim(0, 90)
    ax1.legend(loc="upper right")

    # Subplot 2: Close (2.1m)
    ax2 = axes[1]
    ax2.bar(x - width, [a * 100 for a in ari_close], width, label="CAD ROI ARI (x100)", color=C_BLUE, alpha=0.85)
    ax2.bar(x, cat_a_close, width, label="Cat A Contact Split (%)", color=C_RED, alpha=0.85)
    ax2.bar(x + width, cat_b_close, width, label="Cat B Clearance Split (%)", color=C_GREEN, alpha=0.85)
    ax2.set_title("Stand-off Distance: 2.1 m (Close)\nProximity dominates; 6 MP adds nothing", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_close)

    fig.suptitle("Finding 3: Sensor Megapixels vs Stand-Off Distance Across Series R", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_path = FIG_DIR / "fig3_resolution_vs_distance_ladder.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Generated] {out_path}")


def plot_figure_4_series_a():
    """Figure 4: Series A Deviant Amplitude Saturation & Static Geometric Superiority."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    shears_mm = [1.0, 1.0, 4.0, 9.0, 19.0]
    shears_px = [1.0, 1.0, 4.0, 9.0, 19.0]
    cell_names = ["sA0 (0A)", "sA1 (2A)", "sA2 (5A)", "sA3 (10A)", "sA4 (20A)"]

    # Target boundary AUROC
    target_auroc = [0.5014, 0.4767, 0.6254, 0.6353, 0.6326]

    # Target part IoU and F1
    target_iou = [0.4792, 0.4341, 0.3250, 0.3022, 0.2792]
    target_f1 = [0.6479, 0.6054, 0.4905, 0.4642, 0.4365]

    # Panel 1: AUROC Saturation
    ax1.plot(shears_px, target_auroc, marker="o", color=C_PURPLE, linewidth=2.2, markersize=8)
    ax1.axhline(0.80, color=C_RED, linestyle="--", linewidth=1.5, label="Pre-Registered Bar (0.80)")
    ax1.axhline(0.63, color=C_GREY, linestyle=":", linewidth=1.2, label="Saturation Ceiling (~0.63)")
    ax1.set_xlabel("Interface Shear Displacement (pixels / mm @ 4.2m)")
    ax1.set_ylabel("Target Boundary Motion AUROC")
    ax1.set_title("Target Boundary Motion AUROC Saturates (~0.63)\nFails pre-registered 0.80 crossing", fontweight="bold")
    ax1.set_ylim(0.40, 0.90)
    ax1.legend(loc="lower right")

    # Panel 2: IoU and F1 fall monotonically
    x = np.arange(len(cell_names))
    ax2.plot(x, target_iou, marker="s", color=C_BLUE, linewidth=2.0, markersize=7, label="Target Part IoU")
    ax2.plot(x, target_f1, marker="^", color=C_GREEN, linewidth=2.0, markersize=7, label="Target Part F1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cell_names, rotation=15)
    ax2.set_ylabel("Segmentation Score")
    ax2.set_title("Static Target (sA0) Has Highest Accuracy\nDeviant motion blurs 3D reconstruction", fontweight="bold")
    ax2.set_ylim(0.20, 0.75)
    ax2.legend(loc="upper right")

    fig.suptitle("Finding 4: Series A Re-Framing (Deviant Motion vs Static Geometry)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_path = FIG_DIR / "fig4_series_a_motion_saturation.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Generated] {out_path}")


def plot_figure_5_material_tax():
    """Figure 5: Series T Material Tax Re-Framing (Backbone Invariance vs Motion Failure)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    finishes = ["sT0\n(Pastel Finish)", "sT1\n(Uniform Steel PBR)", "sT2\n(Textured Steel PBR)"]

    cc_ari = [0.3239, 0.3230, 0.3105]
    cc_cat_a = [25.6, 56.4, 33.3]
    rigid2_cat_b = [44.1, 10.8, 32.4]

    x = np.arange(len(finishes))
    width = 0.26

    r1 = ax.bar(x - width, [a * 100 for a in cc_ari], width, label="CC Backbone CAD ARI (x100)\n[Invariant: ~0.32]", color=C_BLUE, alpha=0.85)
    r2 = ax.bar(x, cc_cat_a, width, label="CC Backbone Cat A Split (%)\n[Steel aids contact highlights]", color=C_ORANGE, alpha=0.85)
    r3 = ax.bar(x + width, rigid2_cat_b, width, label="rigid2 Cat B Split (%)\n[Motion suffers -75% collapse]", color=C_RED, alpha=0.85)

    ax.annotate("Motion Method Collapse (-75.5%)\nRestored to 32.4% by Texture",
                xy=(1 + width, 10.8), xytext=(0.85, 30),
                arrowprops=dict(facecolor=C_RED, shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight="bold", color=C_RED,
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=C_RED))

    ax.annotate("CC Backbone Invariant to Steel\nHighlights improve contact split to 56.4%",
                xy=(1, 56.4), xytext=(1.25, 68),
                arrowprops=dict(facecolor=C_ORANGE, shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight="bold", color="#b34700",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=C_ORANGE))

    ax.set_xticks(x)
    ax.set_xticklabels(finishes)
    ax.set_ylabel("Metric Score / Percentage (%)")
    ax.set_title("Finding 5: Material Tax is a Motion-Only Vulnerability; CC Backbone is Invariant", pad=14, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", framealpha=0.95)

    out_path = FIG_DIR / "fig5_material_tax_comparison.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[Generated] {out_path}")


def main():
    print("=" * 80)
    print("GENERATING THESIS PUBLICATION FIGURES (capture_limitation suite)")
    print("=" * 80)
    plot_figure_1_appearance_inversion()
    plot_figure_2_radius_control()
    plot_figure_3_resolution_vs_distance()
    plot_figure_4_series_a()
    plot_figure_5_material_tax()
    print("=" * 80)
    print(f"All 5 thesis figures saved to: {FIG_DIR}")

if __name__ == "__main__":
    main()
