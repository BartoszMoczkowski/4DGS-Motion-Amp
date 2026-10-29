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

from typing import Any, Optional
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time


def _resolve_roi_mask(pc, pipe, means3D):
    """ROI fallback chain (T41): per-point mask → voxel grid → bbox → None.

    The voxel grid (pc._roi_grid, see scene/roi_grid.py) is the authoritative
    form: positional and therefore densification-safe.
    """
    roi_mask = getattr(pc, "_roi_mask", None)
    if roi_mask is not None:
        return roi_mask
    roi_grid = getattr(pc, "_roi_grid", None)
    if roi_grid is not None:
        from scene.roi_grid import voxel_roi_mask
        return voxel_roi_mask(means3D, roi_grid)
    b = getattr(pipe, "roi_bbox", None)
    if b is not None:
        return (
            (means3D[:, 0] >= b[0])
            & (means3D[:, 0] <= b[3])
            & (means3D[:, 1] >= b[1])
            & (means3D[:, 1] <= b[4])
            & (means3D[:, 2] >= b[2])
            & (means3D[:, 2] <= b[5])
        )
    return None


def render(
    viewpoint_camera,
    pc : GaussianModel,
    pipe,
    bg_color : torch.Tensor,
    scaling_modifier = 1.0,
    override_color = None,
    stage="fine",
    cam_type=None,
    custom_time: Optional[torch.Tensor] = None,
    fourier_head: Optional[Any] = None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time_val = custom_time if custom_time is not None else viewpoint_camera.time
        if isinstance(time_val, torch.Tensor):
            time = time_val.to(means3D.device).view(1, 1).repeat(means3D.shape[0], 1)
        else:
            time = torch.tensor(time_val, dtype=torch.float32, device=means3D.device).view(1, 1).repeat(means3D.shape[0], 1)
    else:
        raster_settings = viewpoint_camera['camera']
        time_val = custom_time if custom_time is not None else viewpoint_camera['time']
        if isinstance(time_val, torch.Tensor):
            time = time_val.to(means3D.device).view(1, 1).repeat(means3D.shape[0], 1)
        else:
            time = torch.tensor(time_val, dtype=torch.float32, device=means3D.device).view(1, 1).repeat(means3D.shape[0], 1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    active_fourier_head = fourier_head or getattr(pc, "fourier_head", None)
    bypass_hexplane = (
        active_fourier_head is not None
        and getattr(active_fourier_head, "pure_rigid", False)
        and getattr(active_fourier_head, "static_background", False)
    )
    deformation_point = pc._deformation_table
    if bypass_hexplane:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            means3D,
            scales,
            rotations,
            opacity,
            shs,
        )
    elif "coarse" in stage:
        if getattr(pipe, "gated_coarse", False):
            # Option B (R10): Gated coarse stage.
            # ROI Gaussians query the deformation field from iteration 0 so the canonical
            # pose anchors at a reference instant rather than the temporal mean.
            roi_mask = _resolve_roi_mask(pc, pipe, means3D)
            if roi_mask is not None:
                means3D_final = means3D.clone()
                scales_final = scales.clone()
                rotations_final = rotations.clone()
                opacity_final = opacity.clone()
                shs_final = shs.clone()
                if roi_mask.any():
                    t_in = time[roi_mask] if time.shape[0] == means3D.shape[0] else time
                    m_def, s_def, r_def, o_def, sh_def = pc._deformation(
                        means3D[roi_mask],
                        scales[roi_mask],
                        rotations[roi_mask],
                        opacity[roi_mask],
                        shs[roi_mask],
                        t_in,
                    )
                    means3D_final[roi_mask] = m_def
                    scales_final[roi_mask] = s_def
                    rotations_final[roi_mask] = r_def
                    opacity_final[roi_mask] = o_def
                    shs_final[roi_mask] = sh_def
            else:
                means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
                    pc._deformation(
                        means3D, scales, rotations, opacity, shs, time
                    )
                )
        else:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
                means3D,
                scales,
                rotations,
                opacity,
                shs,
            )
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            pc._deformation(means3D, scales, rotations, opacity, shs, time)
        )
    else:
        raise NotImplementedError

    if active_fourier_head is not None:
        roi_mask = _resolve_roi_mask(pc, pipe, means3D)
        means3D_final, rotations_final = active_fourier_head(
            means3D_canon=means3D,
            means3D_deformed=means3D_final,
            rotations_canon=rotations,
            rotations_deformed=rotations_final,
            time=time,
            roi_mask=roi_mask,
        )

    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color
        shs_final = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "means3D_final": means3D_final}

