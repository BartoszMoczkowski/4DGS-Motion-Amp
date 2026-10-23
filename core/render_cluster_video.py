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
    parser.add_argument("--camera", default="0", type=str, help="Camera name (e.g. cam01, cam05) or index (0-9)")
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--resolution_scale", default=0.5, type=float, help="Scale factor for video resolution (e.g. 0.5 for half-res)")
    parser.add_argument("--side_by_side", action="store_true", help="Render RGB + Cluster colored side by side")
    parser.add_argument("--add_labels", action="store_true", help="Overlay text labels on side-by-side video")
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

        # Determine camera view indices by inspecting dataset image paths
        image_paths = scene.train_camera.dataset.image_paths
        cam_to_indices = {}
        for idx, p in enumerate(image_paths):
            c_name = Path(p).parent.name
            cam_to_indices.setdefault(c_name, []).append(idx)

        all_cams = sorted(cam_to_indices.keys())
        if not all_cams:
            raise RuntimeError("No camera folders found in dataset image paths")

        if args.camera in cam_to_indices:
            chosen_cam = args.camera
        else:
            try:
                c_idx = int(args.camera) % len(all_cams)
                chosen_cam = all_cams[c_idx]
            except ValueError:
                chosen_cam = all_cams[0]

        target_indices = cam_to_indices[chosen_cam]
        print(f"Rendering view from camera '{chosen_cam}' across {len(target_indices)} frames...")

        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        import cv2

        writer = imageio.get_writer(args.output_video, fps=args.fps, quality=8, macro_block_size=1)

        for f_idx in tqdm(target_indices, desc=f"Rendering {chosen_cam}"):
            view = scene.train_camera[f_idx]
            
            # Render with cluster color override
            res_cluster = render(view, gaussians, pipe, background, override_color=cluster_colors_gpu, stage="fine")
            img_cluster = np.ascontiguousarray((255 * np.clip(res_cluster["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0))

            if args.side_by_side:
                # Render natural RGB appearance as well
                res_rgb = render(view, gaussians, pipe, background, override_color=None, stage="fine")
                img_rgb = np.ascontiguousarray((255 * np.clip(res_rgb["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0))

                if args.add_labels:
                    # Draw clean banner labels
                    cv2.putText(img_rgb, "4DGS RGB Reconstruction", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(img_cluster, f"Motion Clusters ({n_clusters} parts)", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

                frame = np.concatenate([img_rgb, img_cluster], axis=1)
            else:
                frame = img_cluster

            if args.resolution_scale != 1.0:
                new_w = int(frame.shape[1] * args.resolution_scale)
                new_h = int(frame.shape[0] * args.resolution_scale)
                new_w -= new_w % 2
                new_h -= new_h % 2
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            writer.append_data(frame)

        writer.close()
        print(f"[Done] Output video saved: {args.output_video}")

if __name__ == "__main__":
    main()
