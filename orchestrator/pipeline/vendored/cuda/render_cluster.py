"""Vendored copy of ``core/render_cluster_video.py`` (T32, per
``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule).

Executes as a separate process inside the ``cuda`` container via ``ctx.containers``;
never imported directly on the host. Renders video with Gaussians dynamically colored
by motion segmentation cluster labels.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def build_cluster_colors(labels: np.ndarray, num_clusters: int | None = None) -> np.ndarray:
    """Build an (N, 3) RGB color array in [0, 1] for each Gaussian based on its cluster ID."""
    N = len(labels)
    uniq = sorted(int(l) for l in np.unique(labels) if l >= 0)
    K = len(uniq) if num_clusters is None else num_clusters

    cmap = plt.colormaps["tab10"] if K <= 10 else plt.colormaps["tab20"]
    colors_table = {lab: np.array(cmap(i % cmap.N)[:3], dtype=np.float32) for i, lab in enumerate(uniq)}
    # Floaters / unassigned / background
    colors_table[-1] = np.array([0.35, 0.35, 0.35], dtype=np.float32)  # Floaters in dim grey
    colors_table[-2] = np.array([0.20, 0.20, 0.20], dtype=np.float32)  # Static background in darker grey

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
    parser.add_argument("--configs", type=str, required=False)
    parser.add_argument("--segmentation_path", type=str, required=True)
    parser.add_argument("--camera", default="cam01", type=str, help="Camera name (e.g. cam01, cam05) or index (0-9)")
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--resolution_scale", default=0.5, type=float, help="Scale factor for video resolution")
    parser.add_argument("--side_by_side", action="store_true", help="Render RGB + Cluster colored side by side")
    parser.add_argument("--add_labels", action="store_true", help="Overlay text labels on side-by-side video")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if args.configs:
        try:
            import mmengine
            config = mmengine.Config.fromfile(args.configs)
        except ImportError:
            import mmcv
            config = mmcv.Config.fromfile(args.configs)
        from utils.params_utils import merge_hparams
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
        valid_labels = labels[labels >= 0]
        n_clusters = len(set(valid_labels)) if len(valid_labels) > 0 else 0
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

        # Fuzzy match requested camera name
        chosen_cam = None
        if args.camera in cam_to_indices:
            chosen_cam = args.camera
        else:
            cam_digits = "".join(filter(str.isdigit, args.camera))
            if cam_digits:
                c_idx = int(cam_digits)
                candidates = [f"cam{c_idx:02d}", f"cam{c_idx}", f"camera_{c_idx}"]
                for cand in candidates:
                    if cand in cam_to_indices:
                        chosen_cam = cand
                        break
                if chosen_cam is None and 0 <= c_idx < len(all_cams):
                    chosen_cam = all_cams[c_idx]

            if chosen_cam is None:
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
            img_cluster = np.ascontiguousarray(
                (255 * np.clip(res_cluster["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
            )

            if args.side_by_side:
                # Render natural RGB appearance as well
                res_rgb = render(view, gaussians, pipe, background, override_color=None, stage="fine")
                img_rgb = np.ascontiguousarray(
                    (255 * np.clip(res_rgb["render"].cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
                )

                if args.add_labels:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    scale = max(0.6, 1.2 * (img_rgb.shape[1] / 1600.0))
                    thick = max(1, int(2 * (img_rgb.shape[1] / 1600.0)))
                    y_pos = int(50 * (img_rgb.shape[0] / 900.0))
                    cv2.putText(img_rgb, "4DGS RGB Reconstruction", (30, y_pos), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                    cv2.putText(img_cluster, f"Motion Clusters ({n_clusters} parts)", (30, y_pos), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

                frame = np.concatenate([img_rgb, img_cluster], axis=1)
            else:
                frame = img_cluster

            if args.resolution_scale != 1.0:
                new_w = int(frame.shape[1] * args.resolution_scale)
                new_h = int(frame.shape[0] * args.resolution_scale)
                new_w -= new_w % 2
                new_h -= new_h % 2
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # Ensure dimensions are even for H.264
                if frame.shape[1] % 2 != 0 or frame.shape[0] % 2 != 0:
                    new_w = frame.shape[1] - (frame.shape[1] % 2)
                    new_h = frame.shape[0] - (frame.shape[0] % 2)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            writer.append_data(frame)

        writer.close()
        print(f"[Done] Output video saved: {args.output_video}")


if __name__ == "__main__":
    main()
