"""
compose_runs_camera_grid.py — Compose all 18 runs into a single unified composite image
showing each run from the same camera (cam01):
- Version 1: Normal Render (cam01/frame_00001.jpg)
- Version 2: Gaussians Colored to Labels (projected into cam01 view)

Layout:
- 6 rows x 3 double-columns (Series A, Series R, Series T + F).
- Ultra-high resolution (2400 x 1650 px).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MAPPING_FILE = REPO_ROOT / "runs" / "capture_limitation_trained_runs.json"
OUT_IMG = REPO_ROOT / "docs" / "figures" / "composite_all_runs_camera_views.png"
OUT_IMG.parent.mkdir(parents=True, exist_ok=True)

from scripts.evaluate_frozen_backbone_capture_limitation import read_point_cloud_ply
from scripts.run_radius_control import load_cell_data, run_cc_clustering

CELL_META = [
    # Series A
    {"id": "sA0", "col": 0, "row": 0, "title": "sA0: Target Static (1mm Shear)", "subtitle": "CAD ARI: 0.367 | IoU: 0.479"},
    {"id": "sA1", "col": 0, "row": 1, "title": "sA1: Target 2A (1mm Shear)", "subtitle": "CAD ARI: 0.363 | IoU: 0.434"},
    {"id": "sA2", "col": 0, "row": 2, "title": "sA2: Target 5A (4mm Shear)", "subtitle": "CAD ARI: 0.454 | IoU: 0.325"},
    {"id": "sA3", "col": 0, "row": 3, "title": "sA3: Target 10A (9mm Shear)", "subtitle": "CAD ARI: 0.400 | IoU: 0.302"},
    {"id": "sA4", "col": 0, "row": 4, "title": "sA4: Target 20A (19mm Shear)", "subtitle": "CAD ARI: 0.362 | IoU: 0.279"},
    {"id": "sA5", "col": 0, "row": 5, "title": "sA5: Target 1A Only (Static Body)", "subtitle": "CAD ARI: 0.367 | IoU: 0.261"},

    # Series R
    {"id": "sR0", "col": 1, "row": 0, "title": "sR0: 900p @ 4.2m (Baseline)", "subtitle": "CAD ARI: 0.444 | Cat A: 25.6%"},
    {"id": "sR1", "col": 1, "row": 1, "title": "sR1: 1440p @ 4.2m", "subtitle": "CAD ARI: 0.348 | Cat A: 23.1%"},
    {"id": "sR2", "col": 1, "row": 2, "title": "sR2: 6 MP @ 4.2m", "subtitle": "CAD ARI: 0.333 | Cat A: 15.4%"},
    {"id": "sR3", "col": 1, "row": 3, "title": "sR3: 900p @ 2.1m (Close)", "subtitle": "App AUROC: 0.867 | Cat A: 35.9%"},
    {"id": "sR4", "col": 1, "row": 4, "title": "sR4: 6 MP @ 2.1m", "subtitle": "App AUROC: 0.872 | Cat A: 43.6%"},
    {"id": "sR5", "col": 1, "row": 5, "title": "sR5: 6 MP @ 2.1m (120fr)", "subtitle": "App AUROC: 0.870 | Cat A: 28.2%"},

    # Series T & F
    {"id": "sT0", "col": 2, "row": 0, "title": "sT0: Synthetic Pastel Finish", "subtitle": "CAD ARI: 0.324 | Cat A: 25.6%"},
    {"id": "sT1", "col": 2, "row": 1, "title": "sT1: Uniform Brushed Steel PBR", "subtitle": "CAD ARI: 0.323 | Cat A: 56.4%"},
    {"id": "sT2", "col": 2, "row": 2, "title": "sT2: Procedural Textured Steel", "subtitle": "CAD ARI: 0.311 | Cat A: 33.3%"},
    {"id": "sF0", "col": 2, "row": 3, "title": "sF0: Baseline Control (Far Pastel)", "subtitle": "CAD ARI: 0.359 | Cat A: 25.6%"},
    {"id": "sF1", "col": 2, "row": 4, "title": "sF1: Honest Steel (Close 6 MP)", "subtitle": "CAD ARI: 0.121 | Cat A: 41.0%"},
    {"id": "sF2", "col": 2, "row": 5, "title": "sF2: Optimal Real (Textured 6 MP)", "subtitle": "Target Prec: 100% | Bound AUROC: 0.791"},
]

def render_segmented_gaussians(cid: str, run_id: str, trained_runs: dict, thumb_w: int, thumb_h: int) -> Image.Image:
    """Project 3D Gaussians into cam01 view, colored by production CC segmentation labels."""
    dataset_dir = REPO_ROOT / "data" / "multipleview" / f"pump01_{cid}"
    poses_arr = np.load(dataset_dir / "poses_bounds_multipleview.npy")
    pose = poses_arr[0, :-2].reshape(3, 5)
    H_cam, W_cam, f = int(pose[0, 4]), int(pose[1, 4]), float(pose[2, 4])
    c0, c1, c2, C = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]

    # Load PLY points
    run_dir = REPO_ROOT / "runs" / run_id
    ply_file = sorted((run_dir / "train_out/point_cloud").glob("*/*.ply"))[-1]
    ply_data = read_point_cloud_ply(ply_file)
    xyz_all = ply_data["xyz"]

    # Load dynamic ROI & segmentation labels
    c_data = load_cell_data(cid, trained_runs)
    r_val = 0.0015 if cid in ("sR3", "sR4", "sR5") else 0.0030
    res = run_cc_clustering(c_data["sub_xyz"], c_data["sub_rgb"], r_val=r_val, color_gated=True, c_th=0.15, min_size=15)
    sub_labels = res["dilated_1nn"]

    # Assign labels: dynamic ROI gets sub_labels (1..K), non-ROI gets 0
    gt_path = dataset_dir / "gt_segmentation.npz"
    gt_data = np.load(gt_path)
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_data["points"])
    _, nn = tree_gt.query(xyz_all, k=1)
    gt_labels = gt_data["labels"][nn]
    oracle_roi = (gt_labels > 0)

    full_labels = np.zeros(len(xyz_all), dtype=np.int32)
    full_labels[oracle_roi] = sub_labels + 1 # shift by 1 so label 0 is background

    # Projection
    diff = xyz_all - C
    zc = - (diff @ c2)
    xc = diff @ c1
    yc = diff @ c0

    valid = (zc > 0.1)
    u = W_cam / 2.0 + f * (xc / np.maximum(zc, 1e-4))
    v = H_cam / 2.0 + f * (yc / np.maximum(zc, 1e-4))
    in_frame = valid & (u >= 0) & (u < W_cam) & (v >= 0) & (v < H_cam)

    order = np.argsort(zc[in_frame])[::-1]
    u_s = u[in_frame][order]
    v_s = v[in_frame][order]
    l_s = full_labels[in_frame][order]

    # Distinct palette with 40 colors
    cmap1 = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    palette = [cmap1(i) for i in range(20)] + [cmap2(i) for i in range(20)]

    unique_l = np.unique(sub_labels + 1)
    color_map = {0: (0.12, 0.12, 0.14, 0.25)} # subtle dark slate for casing/background
    for i, lab in enumerate(unique_l):
        color_map[lab] = palette[i % len(palette)]

    colors = np.array([color_map.get(l, (0.5, 0.5, 0.5, 0.5)) for l in l_s])

    fig, ax = plt.subplots(figsize=(W_cam / 200, H_cam / 200), dpi=100, facecolor="#0a0a0c")
    ax.set_facecolor("#0a0a0c")
    pt_size = 1.0 if W_cam <= 1600 else 1.8
    ax.scatter(u_s, v_s, c=colors, s=pt_size, linewidths=0, alpha=0.95)
    ax.set_xlim(0, W_cam)
    ax.set_ylim(H_cam, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    temp_path = REPO_ROOT / f"scratch_temp_{cid}.png"
    fig.savefig(temp_path, facecolor="#0a0a0c")
    plt.close(fig)

    img = Image.open(temp_path).convert("RGB")
    img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    if temp_path.exists():
        temp_path.unlink()
    return img


def get_normal_render(cid: str, thumb_w: int, thumb_h: int) -> Image.Image:
    """Load cam01/frame_00001.jpg normal render."""
    img_path = REPO_ROOT / "data" / "multipleview" / f"pump01_{cid}" / "cam01" / "frame_00001.jpg"
    img = Image.open(img_path).convert("RGB")
    return img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)


def main():
    print("=" * 80)
    print("COMPOSING ALL 18 RUNS FROM SAME CAMERA VIEW (cam01)")
    print("=" * 80)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        trained_runs = json.load(f)

    # Grid Dimensions
    thumb_w = 340
    thumb_h = 191
    pair_w = thumb_w * 2 + 6
    pair_h = thumb_h + 38

    col_gap = 24
    row_gap = 14

    margin_x = 35
    margin_top = 110
    margin_bottom = 35

    canvas_w = margin_x * 2 + 3 * pair_w + 2 * col_gap
    canvas_h = margin_top + 6 * pair_h + 5 * row_gap + margin_bottom

    print(f"Canvas size: {canvas_w} x {canvas_h} px")
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(14, 15, 18))
    draw = ImageDraw.Draw(canvas)

    # Master Header
    draw.rectangle([(0, 0), (canvas_w, 95)], fill=(20, 22, 28))
    draw.text((canvas_w // 2, 28), "Capture-Limitation Suite: All 18 Runs Under Unified Camera Perspective (cam01)",
              fill=(255, 255, 255), anchor="mm")
    draw.text((canvas_w // 2, 65), "Left: Photometric Normal Render (Capture Image)  |  Right: 3D Gaussians Projected & Colored by Production CC Segmentation",
              fill=(180, 190, 205), anchor="mm")

    # Column Banners
    col_titles = [
        ("SERIES A: DEVIANT MOTION LADDER (0A to 20A)", 0),
        ("SERIES R: SENSOR RESOLUTION & DISTANCE LADDER", 1),
        ("SERIES T (TOP) & SERIES F (BOTTOM): MATERIAL FINISHES", 2),
    ]
    for title, col_idx in col_titles:
        x_left = margin_x + col_idx * (pair_w + col_gap)
        x_center = x_left + pair_w // 2
        draw.rectangle([(x_left, 100), (x_left + pair_w, 120)], fill=(30, 34, 44))
        draw.text((x_center, 110), title, fill=(220, 230, 245), anchor="mm")

    # Process all 18 cells
    for idx, item in enumerate(CELL_META):
        cid = item["id"]
        c_idx = item["col"]
        r_idx = item["row"]
        run_id = trained_runs[cid]

        print(f"[{idx+1}/18] Rendering pair for {cid} ({item['title']})...")

        # Compute bounding box
        x_start = margin_x + c_idx * (pair_w + col_gap)
        y_start = margin_top + 25 + r_idx * (pair_h + row_gap)

        # Header for this cell
        draw.rectangle([(x_start, y_start), (x_start + pair_w, y_start + 32)], fill=(24, 27, 34))
        draw.text((x_start + 10, y_start + 10), item["title"], fill=(240, 240, 245))
        draw.text((x_start + pair_w - 10, y_start + 10), item["subtitle"], fill=(120, 200, 150), anchor="ra")

        # Sub-images
        y_img = y_start + 34
        norm_img = get_normal_render(cid, thumb_w, thumb_h)
        canvas.paste(norm_img, (x_start, y_img))

        # Label badge on normal render
        draw.rectangle([(x_start + 6, y_img + 6), (x_start + 105, y_img + 22)], fill=(0, 0, 0, 180))
        draw.text((x_start + 10, y_img + 9), "Normal Render", fill=(255, 255, 255))

        # Gaussians colored by labels
        x_seg = x_start + thumb_w + 6
        seg_img = render_segmented_gaussians(cid, run_id, trained_runs, thumb_w, thumb_h)
        canvas.paste(seg_img, (x_seg, y_img))

        # Label badge on segmented render
        draw.rectangle([(x_seg + 6, y_img + 6), (x_seg + 120, y_img + 22)], fill=(0, 0, 0, 180))
        draw.text((x_seg + 10, y_img + 9), "CC Seg Gaussians", fill=(100, 220, 255))

    # Save
    canvas.save(OUT_IMG, quality=95)
    print("=" * 80)
    print(f"Composite image successfully created and saved to: {OUT_IMG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
