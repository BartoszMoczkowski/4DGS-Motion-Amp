"""
Renders a video from the perspective of one of the cameras with Gaussians
colored in accordance with their motion segmentation cluster labels.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
import mmengine
from utils.params_utils import merge_hparams
from utils.general_utils import safe_state

def build_cluster_colors(labels: np.ndarray, num_clusters: int | None = None) -> np.ndarray:
    """Build an (N, 3) RGB color tensor for each Gaussian based on its cluster ID."""
    N = len(labels)
    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    K = len(uniq) if num_clusters is None else num_clusters
    
    cmap = plt.colormaps["tab10"] if K <= 10 else plt.colormaps["tab20"]
    colors_table = {lab: np.array(cmap(i % cmap.N)[:3], dtype=np.float32) for i, lab in enumerate(uniq)}
    colors_table[-1] = np.array([0.35, 0.35, 0.35], dtype=np.float32)  # Floaters in dim grey

    rgb_array = np.zeros((N, 3), dtype=np.float32)
    for lab, color in colors_table.items():
        rgb_array[labels == lab] = color
    return rgb_array

def main():
    parser = ArgumentParser(description="Render 4DGS with cluster-colored Gaussians")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument("--segmentation_path", type=str, required=True)
    parser.add_argument("--camera_index", default=0, type=int, help="0-indexed camera view (e.g. 0 for cam01, 4 for cam05)")
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--resolution_scale", default=1.0, type=float)
    parser.add_argument("--side_by_side", action="store_true", help="Render RGB + Cluster colored side by side")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if args.configs:
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)

    print(f"Loading model from {args.model_path}...")
    dataset = model.extract(args)
    hyper = hyperparam.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        # Load segmentation
        print(f"Loading segmentation from {args.segmentation_path}...")
        seg_data = np.load(args.segmentation_path)
        labels = seg_data["labels"]
        n_clusters = len(set(labels[labels != -1]))
        print(f"Gaussians: {len(labels)}, Clusters: {n_clusters}, Floaters: {(labels == -1).sum()}")

        cluster_colors_np = build_cluster_colors(labels, num_clusters=n_clusters)
        cluster_colors_gpu = torch.tensor(cluster_colors_np, dtype=torch.float32, device="cuda")

        bg_color = [0, 0, 0] if not dataset.white_background else [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Determine camera view indices
        cam_folder = dataset.source_path
        cam_subdirs = sorted([d for d in os.listdir(cam_folder) if d.startswith("cam") and os.path.isdir(os.path.join(cam_folder, d))])
        if not cam_subdirs:
            raise RuntimeError(f"No camXX folders found in {cam_folder}")
        
        chosen_cam = cam_subdirs[args.camera_index % len(cam_subdirs)]
        image_length = len(os.listdir(os.path.join(cam_folder, chosen_cam)))
        print(f"Rendering view from camera '{chosen_cam}' (index {args.camera_index}) across {image_length} frames...")

        start_idx = (args.camera_index % len(cam_subdirs)) * image_length
        end_idx = start_idx + image_length

        rendered_frames = []
        for f_idx in tqdm(range(start_idx, end_idx), desc=f"Rendering {chosen_cam}"):
            view = scene.train_camera[f_idx]
            
            # Render with cluster color override
            res_cluster = render(view, gaussians, pipe, background, override_color=cluster_colors_gpu, stage="fine")
            img_cluster = (255 * np.clip(res_cluster["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)

            if args.side_by_side:
                # Render natural RGB appearance as well
                res_rgb = render(view, gaussians, pipe, background, override_color=None, stage="fine")
                img_rgb = (255 * np.clip(res_rgb["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
                # Concatenate side by side (RGB | Cluster)
                frame = np.concatenate([img_rgb, img_cluster], axis=1)
            else:
                frame = img_cluster

            rendered_frames.append(frame)

        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        print(f"Writing {len(rendered_frames)} frames to {args.output_video} @ {args.fps} FPS...")
        imageio.mimwrite(args.output_video, rendered_frames, fps=args.fps, quality=9)
        print(f"[Done] Output video saved: {args.output_video}")

if __name__ == "__main__":
    main()
