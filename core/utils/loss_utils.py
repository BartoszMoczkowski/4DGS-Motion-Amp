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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def normalize_loss_weights(
    weights: torch.Tensor,
    mask_max: float = 15.0,
    floor: float = 1.0,
    normalize_mean: bool = True,
) -> torch.Tensor:
    """Prepares continuous, floor-bounded, mean-normalized soft loss weights (W3 / T51).

    1. If weights are normalized in [0, 1], maps to [floor, mask_max]:
       w = floor + (mask_max - floor) * weights
    2. Enforces floor >= floor (default 1.0).
    3. Normalizes by spatial mean so E[w] = 1.0, preserving the unweighted L1 loss scale.
    4. Detaches from autograd graph (loss weights do not receive gradients).
    """
    w = weights.detach().float()
    if w.max() <= 1.01:
        w = floor + (mask_max - floor) * w
    w = torch.clamp(w, min=floor)
    if normalize_mean:
        if w.dim() >= 3:
            # Spatial dimensions are the last two (-2, -1)
            mean_val = w.mean(dim=(-2, -1), keepdim=True)
            w = w / torch.clamp(mean_val, min=1e-8)
        else:
            w = w / torch.clamp(w.mean(), min=1e-8)
    return w.detach()


def l1_loss(network_output, gt, mask=None):
    diff = torch.abs(network_output - gt)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            if mask.shape[0] == diff.shape[0]:
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(0)
        diff = diff * mask
    return diff.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

