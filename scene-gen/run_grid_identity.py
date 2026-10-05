"""scene-gen/run_grid_identity.py — Benchmark harness for E2 Per-Gaussian Identity Encodings.

Executes Gaussian Grouping identity feature learning and evaluation across:
1. Baseline testbeds: cubes_k2_both and cubes_k2_ring
2. 7 pump models: grid-A20mm_M2, grid-A20mm_M4, grid-A40mm_M8, and sweep-A40mm_M8-g10000..g100000
3. Original pastel pump01

Emits:
- runs/cubes_seg_identity_results.csv
- runs/grid_seg_identity_results.csv
- Colored PLY files: <run_dir>/segmentation_colored_identity.ply
- Segmentation npz files: <run_dir>/segmentation_identity.npz
"""

from __future__ import annotations

import argparse
from argparse import ArgumentParser, Namespace
import csv
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "scene-gen"))

from identity.mask_provider import OracleMaskProvider, create_sam_mask_provider
from identity.train import train_identity_encodings
from identity.cluster import evaluate_identity_segmentation, write_colored_ply

CUBES_CSV = REPO_ROOT / "runs" / "cubes_seg_identity_results.csv"
GRID_CSV = REPO_ROOT / "runs" / "grid_seg_identity_results.csv"

CSV_HEADER = [
    "timestamp",
    "run_id",
    "mode",
    "n_points",
    "n_gt",
    "n_pred",
    "ari_global",
    "ari_within_roi",
    "mean_iou",
    "ari_motion",
    "mean_iou_motion",
]


def append_csv_row(csv_path: Path, row: Dict[str, Any]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[csv] Appended row to {csv_path}")


def run_identity_on_model(
    run_id: str,
    model_path: str,
    render_dir: str,
    gt_seg_path: str,
    gt_motion_path: Optional[str] = None,
    mode: str = "oracle",
    iterations: int = 1500,
    lr: float = 0.005,
    lambda_3d: float = 0.1,
    csv_path: Path = GRID_CSV,
) -> Dict[str, Any]:
    """Execute Identity Encodings training and evaluation for a single model."""
    print(f"\n=======================================================")
    print(f"Running Identity Encodings on {run_id} (mode={mode})")
    print(f"=======================================================")

    # Normalize paths for host vs Docker container
    if not os.path.exists(render_dir):
        alt_render = render_dir.replace("Q:/Omniverse", "/omniverse").replace("Q:\\Omniverse", "/omniverse")
        if os.path.exists(alt_render):
            render_dir = alt_render

    if not os.path.exists(model_path):
        alt_model = model_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace")
        if os.path.exists(alt_model):
            model_path = alt_model

    if not os.path.exists(gt_seg_path):
        alt_gt = gt_seg_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace")
        if os.path.exists(alt_gt):
            gt_seg_path = alt_gt

    if gt_motion_path and not os.path.exists(gt_motion_path):
        alt_motion = gt_motion_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace")
        if os.path.exists(alt_motion):
            gt_motion_path = alt_motion

    run_dir = Path(model_path).parent if "train_out" in model_path else Path(model_path)

    # 1. Load masks (Oracle or SAM)
    # Determine camera names and frame count
    cam_names = [f"cam{i:02d}" for i in range(1, 17)]
    # Filter to existing camera directories in render_dir
    cam_names = [c for c in cam_names if os.path.isdir(os.path.join(render_dir, c))]
    if not cam_names:
        raise ValueError(f"No valid camera directories found in {render_dir}")

    # Determine frame count from first camera
    first_cam_dir = os.path.join(render_dir, cam_names[0], "rgb")
    if not os.path.isdir(first_cam_dir):
        first_cam_dir = os.path.join(render_dir, cam_names[0])
    frames = [f for f in os.listdir(first_cam_dir) if f.endswith((".jpg", ".png"))]
    n_frames = len(frames)
    print(f"[{run_id}] Detected {len(cam_names)} cameras, {n_frames} frames per camera")

    if mode == "oracle":
        mask_provider = OracleMaskProvider(render_dir, cam_names, n_frames)
    elif mode == "sam":
        sam_ckpt = str(REPO_ROOT / "weights" / "sam_vit_b_01ec64.pth")
        cache_dir = str(run_dir / "sam_masks_cache")
        # Load xyz points for cross-camera projection
        ply_file = os.path.join(model_path, "point_cloud", "iteration_15000", "point_cloud.ply")
        if not os.path.isfile(ply_file):
            ply_file = os.path.join(model_path, "point_cloud", "iteration_14000", "point_cloud.ply")
        from plyfile import PlyData
        pdata = PlyData.read(ply_file)
        pc_xyz = np.stack([pdata.elements[0]["x"], pdata.elements[0]["y"], pdata.elements[0]["z"]], axis=1)

        # Build cam_dirs
        cam_dirs = {c: os.path.join(render_dir, c, "rgb") if os.path.isdir(os.path.join(render_dir, c, "rgb")) else os.path.join(render_dir, c) for c in cam_names}

        # Need cameras list from scene
        from arguments import ModelParams
        cfg_file = os.path.join(model_path, "cfg_args")
        with open(cfg_file, "r") as f:
            args = eval(f.read())
        if hasattr(args, "source_path") and not os.path.exists(args.source_path):
            alt_src = args.source_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace").replace("C:\\Users\\barte\\Code\\PythonScripts\\4DGS-Motion-Amp", "/workspace")
            if os.path.exists(alt_src):
                args.source_path = alt_src
        if hasattr(args, "model_path") and not os.path.exists(args.model_path):
            alt_m = args.model_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace").replace("C:\\Users\\barte\\Code\\PythonScripts\\4DGS-Motion-Amp", "/workspace")
            if os.path.exists(alt_m):
                args.model_path = alt_m
        from scene.gaussian_model import GaussianModel
        from scene import Scene
        dummy_pc = GaussianModel(args.sh_degree, args)
        scene = Scene(args, dummy_pc, load_iteration=-1, shuffle=False)
        cams = scene.getTrainCameras()

        mask_provider = create_sam_mask_provider(
            cam_dirs=cam_dirs,
            sam_checkpoint=sam_ckpt,
            output_cache_dir=cache_dir,
            pc_xyz=pc_xyz,
            cameras=cams,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Train identity features
    grouping_model, pred_labels, norm_features = train_identity_encodings(
        model_path=model_path,
        mask_provider=mask_provider,
        iterations=iterations,
        lr=lr,
        lambda_3d=lambda_3d,
    )

    # 3. Save artifacts
    out_seg_path = run_dir / f"segmentation_identity_{mode}.npz"
    out_ply_path = run_dir / f"segmentation_colored_identity_{mode}.ply"
    features_path = run_dir / f"identity_features_{mode}.npz"

    pred_pts = grouping_model.gaussians.get_xyz.detach().cpu().numpy()
    np.savez_compressed(out_seg_path, points=pred_pts, labels=pred_labels)
    np.savez_compressed(features_path, features=norm_features)
    write_colored_ply(pred_pts, pred_labels, str(out_ply_path))

    # Also save standard segmentation.npz if not present or requested
    np.savez_compressed(run_dir / "segmentation.npz", points=pred_pts, labels=pred_labels)

    # 4. Evaluate against GT
    eval_res = evaluate_identity_segmentation(
        pred_points=pred_pts,
        pred_labels=pred_labels,
        gt_segmentation_path=gt_seg_path,
        gt_motion_path=gt_motion_path,
    )

    print(f"\n[{run_id} Results - Mode: {mode}]")
    print(f"  Gaussians: {eval_res['n_points']}")
    print(f"  GT Parts:  {eval_res['n_gt_parts']}")
    print(f"  Pred K:    {eval_res['n_pred_clusters']}")
    print(f"  Global ARI:       {eval_res['ari_global']:.4f}")
    print(f"  Within-ROI ARI:   {eval_res['ari_within_roi']:.4f}")
    print(f"  Mean IoU:         {eval_res['mean_iou']:.4f}")
    if eval_res['ari_motion'] is not None:
        print(f"  Motion-class ARI: {eval_res['ari_motion']:.4f}")
        print(f"  Motion Mean IoU:  {eval_res['mean_iou_motion']:.4f}")

    # 5. Append to CSV
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": run_id,
        "mode": mode,
        "n_points": eval_res["n_points"],
        "n_gt": eval_res["n_gt_parts"],
        "n_pred": eval_res["n_pred_clusters"],
        "ari_global": f"{eval_res['ari_global']:.6f}",
        "ari_within_roi": f"{eval_res['ari_within_roi']:.6f}",
        "mean_iou": f"{eval_res['mean_iou']:.6f}",
        "ari_motion": f"{eval_res['ari_motion']:.6f}" if eval_res["ari_motion"] is not None else "",
        "mean_iou_motion": f"{eval_res['mean_iou_motion']:.6f}" if eval_res["mean_iou_motion"] is not None else "",
    }
    append_csv_row(csv_path, row)

    return eval_res


def main():
    parser = argparse.ArgumentParser(description="Run Identity Encodings benchmark (E2)")
    grid_model_names = [
        "grid-A20mm_M2", "grid-A20mm_M4", "grid-A40mm_M8",
        "sweep-A40mm_M8-g10000", "sweep-A40mm_M8-g25000",
        "sweep-A40mm_M8-g50000", "sweep-A40mm_M8-g100000"
    ]
    parser.add_argument("--scene", type=str, default="cubes_k2_both",
                        choices=["cubes_k2_both", "cubes_k2_ring", "pump_grid", "pump01", "all"] + grid_model_names)
    parser.add_argument("--mode", type=str, default="oracle", choices=["oracle", "sam", "both"])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lambda_3d", type=float, default=0.1)
    args = parser.parse_args()

    modes = ["oracle", "sam"] if args.mode == "both" else [args.mode]

    # Handle scenes
    if args.scene in ("cubes_k2_both", "all"):
        for m in modes:
            run_identity_on_model(
                run_id="cubes_k2_both",
                model_path=str(REPO_ROOT / "runs" / "cubes-cubes_k2_both" / "train_out"),
                render_dir="Q:/Omniverse/renders/capture_cubes_k2_both",
                gt_seg_path=str(REPO_ROOT / "data" / "multipleview" / "cubes_k2_both" / "gt_segmentation.npz"),
                mode=m,
                iterations=args.iterations,
                lr=args.lr,
                lambda_3d=args.lambda_3d,
                csv_path=CUBES_CSV,
            )

    if args.scene in ("cubes_k2_ring", "all"):
        for m in modes:
            run_identity_on_model(
                run_id="cubes_k2_ring",
                model_path=str(REPO_ROOT / "runs" / "cubes-cubes_k2_ring" / "train_out"),
                render_dir="Q:/Omniverse/renders/capture_cubes_k2_ring",
                gt_seg_path=str(REPO_ROOT / "data" / "multipleview" / "cubes_k2_ring" / "gt_segmentation.npz"),
                mode=m,
                iterations=args.iterations,
                lr=args.lr,
                lambda_3d=args.lambda_3d,
                csv_path=CUBES_CSV,
            )

    all_grid_models = [
        ("grid-A20mm_M2", "Q:/Omniverse/renders/capture_pump_A20mm_M2"),
        ("grid-A20mm_M4", "Q:/Omniverse/renders/capture_pump_A20mm_M4"),
        ("grid-A40mm_M8", "Q:/Omniverse/renders/capture_pump_A40mm_M8"),
        ("sweep-A40mm_M8-g10000", "Q:/Omniverse/renders/capture_pump_A40mm_M8"),
        ("sweep-A40mm_M8-g25000", "Q:/Omniverse/renders/capture_pump_A40mm_M8"),
        ("sweep-A40mm_M8-g50000", "Q:/Omniverse/renders/capture_pump_A40mm_M8"),
        ("sweep-A40mm_M8-g100000", "Q:/Omniverse/renders/capture_pump_A40mm_M8"),
    ]
    if args.scene in ("pump_grid", "all") or args.scene in grid_model_names:
        grid_models = all_grid_models if args.scene in ("pump_grid", "all") else [m for m in all_grid_models if m[0] == args.scene]
        gt_motion = str(REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_motion_classes.npy")

        for r_id, r_dir in grid_models:
            gt_matches = list((REPO_ROOT / "runs" / r_id / "convert_out").rglob("gt_segmentation.npz"))
            if gt_matches:
                gt_path = str(gt_matches[0])
            else:
                gt_path = str(REPO_ROOT / "runs" / "grid-A40mm_M8" / "convert_out" / "data" / "multipleview" / "capture_pump_A40mm_M8" / "gt_segmentation.npz")

            for m in modes:
                run_identity_on_model(
                    run_id=r_id,
                    model_path=str(REPO_ROOT / "runs" / r_id / "train_out"),
                    render_dir=r_dir,
                    gt_seg_path=gt_path,
                    gt_motion_path=gt_motion,
                    mode=m,
                    iterations=args.iterations,
                    lr=args.lr,
                    lambda_3d=args.lambda_3d,
                    csv_path=GRID_CSV,
                )

    if args.scene in ("pump01", "all"):
        gt_motion = str(REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_motion_classes.npy")
        for m in modes:
            run_identity_on_model(
                run_id="pump01",
                model_path=str(REPO_ROOT / "output" / "multipleview" / "pump01"),
                render_dir="Q:/Omniverse/renders/capture_pump",
                gt_seg_path=str(REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_segmentation.npz"),
                gt_motion_path=gt_motion,
                mode=m,
                iterations=args.iterations,
                lr=args.lr,
                lambda_3d=args.lambda_3d,
                csv_path=GRID_CSV,
            )


if __name__ == "__main__":
    main()
