#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    if np.array(resized_image_PIL).max()!=1:
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def get_regularization_weights(iteration, hyper):
    """Compute active regularization weights for the current fine-stage iteration.

    Returns:
        (time_smoothness_weight, l1_time_planes_weight, plane_tv_weight)

    Schedule behavior (R1):
        - If regularizer_anneal_iter <= 0: schedule is inactive; returns initial weights.
        - If iteration < regularizer_anneal_iter: returns initial weights while geometry stabilizes.
        - If iteration >= regularizer_anneal_iter:
            - If regularizer_anneal_type == "linear" and regularizer_anneal_end_iter > regularizer_anneal_iter:
                linearly anneals from initial to final weights until regularizer_anneal_end_iter.
            - Otherwise ("step" / default):
                immediately switches to the final weights.

    Unset or negative (< 0) `_final` values preserve the initial weight for that channel.
    """
    def _get(key, default=None):
        if isinstance(hyper, dict):
            return hyper.get(key, default)
        return getattr(hyper, key, default)

    ts_init = float(_get("time_smoothness_weight", 0.0) or 0.0)
    l1_init = float(_get("l1_time_planes", 0.0) or 0.0)
    ptv_init = float(_get("plane_tv_weight", 0.0) or 0.0)

    anneal_iter = _get("regularizer_anneal_iter", 0)
    if anneal_iter is None:
        anneal_iter = 0
    anneal_iter = int(anneal_iter)

    if anneal_iter <= 0 or iteration < anneal_iter:
        return ts_init, l1_init, ptv_init

    ts_final_val = _get("time_smoothness_weight_final", None)
    l1_final_val = _get("l1_time_planes_final", None)
    ptv_final_val = _get("plane_tv_weight_final", None)

    ts_target = float(ts_final_val) if (ts_final_val is not None and float(ts_final_val) >= 0) else ts_init
    l1_target = float(l1_final_val) if (l1_final_val is not None and float(l1_final_val) >= 0) else l1_init
    ptv_target = float(ptv_final_val) if (ptv_final_val is not None and float(ptv_final_val) >= 0) else ptv_init

    anneal_type = str(_get("regularizer_anneal_type", "step") or "step").lower()
    end_iter = _get("regularizer_anneal_end_iter", 0)
    if end_iter is None:
        end_iter = 0
    end_iter = int(end_iter)

    if anneal_type == "linear" and end_iter > anneal_iter:
        if iteration >= end_iter:
            return ts_target, l1_target, ptv_target
        alpha = (iteration - anneal_iter) / float(end_iter - anneal_iter)
        ts_weight = (1.0 - alpha) * ts_init + alpha * ts_target
        l1_weight = (1.0 - alpha) * l1_init + alpha * l1_target
        ptv_weight = (1.0 - alpha) * ptv_init + alpha * ptv_target
        return ts_weight, l1_weight, ptv_weight

    # Default: "step" iteration-gated weight switch
    return ts_target, l1_target, ptv_target

