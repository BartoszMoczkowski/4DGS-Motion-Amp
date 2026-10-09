"""scene-gen/identity/train.py — Identity Encodings training loop (Gaussian Grouping).

Supervises 16D per-Gaussian identity features using on-demand 2D instance masks while keeping
all 3DGS geometry, opacities, and deformation field frozen.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from arguments import ModelHiddenParams, ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel

from .mask_provider import MaskProvider
from .model import GaussianGroupingModule, build_knn_edges


def train_identity_encodings(
    model_path: str,
    mask_provider: MaskProvider,
    iterations: int = 1500,
    lr: float = 0.005,
    lambda_3d: float = 0.1,
    feature_dim: int = 16,
    log_interval: int = 100,
) -> Tuple[GaussianGroupingModule, np.ndarray, np.ndarray]:
    """Train 16D identity features on a trained 4DGS model.

    Args:
        model_path: path to directory containing point_cloud/iteration_15000 and cfg_args.
        mask_provider: MaskProvider supplying on-demand 2D instance masks.
        iterations: number of training steps.
        lr: learning rate for identity parameters and linear classifier.
        lambda_3d: weight for 3D spatial consistency loss.
        feature_dim: dimensionality of identity feature (default 16).

    Returns:
        grouping_model: trained GaussianGroupingModule.
        pred_labels: (N,) int32 discrete class labels.
        norm_features: (N, 16) float32 normalized identity embeddings.
    """
    print(f"\n[train_identity] Loading 4DGS model from {model_path}...")
    cfg_file = os.path.join(model_path, "cfg_args")
    with open(cfg_file, "r") as f:
        args = eval(f.read())

    # Adjust source_path and model_path for container vs host
    if hasattr(args, "source_path") and not os.path.exists(args.source_path):
        alt_src = args.source_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace").replace("C:\\Users\\barte\\Code\\PythonScripts\\4DGS-Motion-Amp", "/workspace")
        if os.path.exists(alt_src):
            args.source_path = alt_src
    if hasattr(args, "model_path") and not os.path.exists(args.model_path):
        alt_m = args.model_path.replace("C:/Users/barte/Code/PythonScripts/4DGS-Motion-Amp", "/workspace").replace("C:\\Users\\barte\\Code\\PythonScripts\\4DGS-Motion-Amp", "/workspace")
        if os.path.exists(alt_m):
            args.model_path = alt_m

    gaussians = GaussianModel(sh_degree=args.sh_degree, args=args)
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    gaussians._deformation.deformation_net.set_aabb(xyz.max(axis=0), xyz.min(axis=0))
    gaussians._deformation.to("cuda")
    gaussians._deformation.eval()

    train_cams = scene.getTrainCameras()
    print(f"[train_identity] Loaded {len(train_cams)} training viewpoints")

    num_classes = mask_provider.num_classes
    print(f"[train_identity] Supervising {num_classes} instance classes (including background)")

    # Initialize grouping module
    grouping_model = GaussianGroupingModule(
        gaussians=gaussians,
        num_classes=num_classes,
        feature_dim=feature_dim,
        lambda_3d=lambda_3d,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [grouping_model.identity_params], "lr": lr},
            {"params": grouping_model.classifier.parameters(), "lr": lr},
        ]
    )

    xyz_cuda = gaussians.get_xyz.detach().clone()

    # Prepare lightweight index mapping for training cameras
    print("[train_identity] Indexing training camera masks...")
    cam_items: List[Tuple[int, str, int]] = []
    cam_names = mask_provider.cam_names
    n_frames = mask_provider.n_frames
    n_cams = len(train_cams)

    for idx in range(n_cams):
        if hasattr(train_cams, "dataset") and hasattr(train_cams.dataset, "image_paths"):
            img_path = str(train_cams.dataset.image_paths[idx])
            cam_key = next((k for k in cam_names if k in img_path), Path(img_path).parent.name)
            t_val = float(train_cams.dataset.image_times[idx])
        elif hasattr(train_cams, "dataset") and hasattr(train_cams.dataset[idx], "time"):
            info = train_cams.dataset[idx]
            c_name = str(info.image_name) if hasattr(info, "image_name") else f"cam_{idx}"
            cam_key = next((k for k in cam_names if k in c_name), cam_names[0])
            t_val = float(info.time)
        else:
            c = train_cams[idx]
            c_name = str(c.image_name)
            cam_key = next((k for k in cam_names if k in c_name), cam_names[0])
            t_val = float(c.time)
            c.original_image = None
            del c

        fi = int(round(t_val * (n_frames - 1))) if n_frames > 1 else 0
        fi = max(0, min(fi, n_frames - 1))
        cam_items.append((idx, cam_key, fi))

    print(f"[train_identity] Starting {iterations} optimization steps...")
    t0 = time.time()

    for step in range(1, iterations + 1):
        idx, cam_key, fi = random.choice(cam_items)
        cam = train_cams[idx]
        cam.original_image = None
        H, W = cam.image_height, cam.image_width
        mask_np = mask_provider.get_mask(cam_key, fi, target_size=(W, H))

        mask_tensor = torch.from_numpy(mask_np).to(device="cuda", dtype=torch.int64)

        optimizer.zero_grad()
        loss, l_2d, l_3d = grouping_model.forward_loss(cam, mask_tensor, xyz_cuda)
        loss.backward()
        optimizer.step()
        del cam

        if step % log_interval == 0 or step == iterations:
            elapsed = time.time() - t0
            steps_per_sec = step / max(elapsed, 1e-4)
            print(
                f"[step {step:4d}/{iterations}] loss: {loss.item():.4f} "
                f"(2D CE: {l_2d:.4f}, 3D cos: {l_3d:.4f}) | {steps_per_sec:.1f} it/s",
                flush=True
            )

    elapsed_total = time.time() - t0
    print(f"[train_identity] Fine-tuning completed in {elapsed_total:.1f}s ({iterations / elapsed_total:.1f} it/s)")

    # Extract discrete labels and features
    pred_labels = grouping_model.get_gaussian_labels()
    norm_features = grouping_model.get_normalized_features()

    n_clusters = len(np.unique(pred_labels))
    print(f"[train_identity] Assigned {len(pred_labels)} Gaussians into {n_clusters} discrete classes")

    return grouping_model, pred_labels, norm_features
