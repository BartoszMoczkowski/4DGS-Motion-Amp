"""
generate_3way_comparison.py — Generate 3-way side-by-side video comparison:
[Original Capture | 4DGS RGB Reconstruction | Motion Clusters]

Matches the format of runs/turbine_r1_missing_6-17907710/segmentation_renders/comparison_3way_cam01.mp4:
- Resolution: 4800 x 900 (3 panels of 1600 x 900 each)
- FPS: 25.0
- Codec: H.264 (yuv420p)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def assemble_comparison(run_id: str, camera: str = "cam01", fps: int = 25):
    run_dir = REPO_ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Identify scene directory from manifest or heuristic
    manifest_path = run_dir / "manifest.json"
    scene_dir = None
    if manifest_path.is_file():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "artifacts" in manifest and "scene" in manifest["artifacts"]:
            scene_dir = Path(manifest["artifacts"]["scene"]["path"])
    
    if scene_dir is None or not scene_dir.is_dir():
        # Fallback to lego_turbine_2_missing
        scene_dir = REPO_ROOT / "data" / "multipleview" / "lego_turbine_2_missing"

    cam_dir = scene_dir / camera
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Camera frame directory not found: {cam_dir}")

    frame_files = sorted(list(cam_dir.glob("frame_*.jpg")) + list(cam_dir.glob("frame_*.png")))
    if not frame_files:
        raise RuntimeError(f"No frames found in {cam_dir}")

    n_frames = len(frame_files)
    print(f"Found {n_frames} source frames in {cam_dir}")

    out_renders_dir = run_dir / "segmentation_renders"
    out_renders_dir.mkdir(parents=True, exist_ok=True)

    cluster_video_src = run_dir / "cluster_video.mp4"
    if not cluster_video_src.is_file():
        # Check if already in segmentation_renders
        cluster_video_src = out_renders_dir / f"cluster_video_{camera}.mp4"
        if not cluster_video_src.is_file():
            raise FileNotFoundError(f"Missing cluster video: {run_dir / 'cluster_video.mp4'}")

    target_cluster_video = out_renders_dir / f"cluster_video_{camera}.mp4"
    if cluster_video_src.resolve() != target_cluster_video.resolve():
        print(f"Copying {cluster_video_src.name} to {target_cluster_video.name}...")
        shutil.copy2(cluster_video_src, target_cluster_video)

    # 1. Encode original camera video if not already present
    original_video_path = out_renders_dir / f"original_{camera}.mp4"
    if not original_video_path.is_file():
        print(f"Encoding original video to {original_video_path}...")
        first_img = cv2.imread(str(frame_files[0]))
        h_orig, w_orig = first_img.shape[:2]
        
        orig_writer = imageio.get_writer(
            str(original_video_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-crf", "18"],
        )
        for fpath in tqdm(frame_files, desc=f"Original {camera}"):
            bgr = cv2.imread(str(fpath))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            orig_writer.append_data(rgb)
        orig_writer.close()
        print(f"Original video saved ({w_orig}x{h_orig})")
    else:
        print(f"Original video already exists at {original_video_path}")

    # 2. Assemble 3-way side-by-side comparison video
    comparison_video_path = out_renders_dir / f"comparison_3way_{camera}.mp4"
    print(f"Assembling 3-way comparison video: {comparison_video_path}...")

    cluster_reader = imageio.get_reader(str(target_cluster_video))
    cluster_meta = cluster_reader.get_meta_data()
    cluster_fps = cluster_meta.get("fps", fps)

    comp_writer = imageio.get_writer(
        str(comparison_video_path),
        fps=cluster_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "18"],
    )

    for idx, fpath in enumerate(tqdm(frame_files, desc="Composing 3-way")):
        try:
            cluster_frame = cluster_reader.get_data(idx)
        except (IndexError, RuntimeError):
            print(f"Cluster reader reached EOF at index {idx}")
            break

        # cluster_frame contains RGB Reconstruction + Clusters side by side
        h_c, w_c = cluster_frame.shape[:2]
        target_h = h_c
        target_panel_w = w_c // 2  # Each panel in cluster_frame is exactly half the total width

        # Load and resize original frame
        bgr = cv2.imread(str(fpath))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_resized = cv2.resize(rgb, (target_panel_w, target_h), interpolation=cv2.INTER_AREA)

        # Concatenate: [Original (1600x900) | 4DGS RGB (1600x900) | Motion Clusters (1600x900)]
        composite_frame = np.concatenate([orig_resized, cluster_frame], axis=1)

        comp_writer.append_data(composite_frame)

    cluster_reader.close()
    comp_writer.close()
    print(f"\n[Success] 3-way comparison video generated at:\n  {comparison_video_path}")
    print(f"File size: {comparison_video_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3-way comparison video")
    parser.add_argument("run_id", type=str, default="lego_turbine_2_missing_r9-415299ad", nargs="?")
    parser.add_argument("--camera", type=str, default="cam01")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    assemble_comparison(args.run_id, args.camera, args.fps)
