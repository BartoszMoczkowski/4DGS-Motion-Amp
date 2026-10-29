import torch
@torch.no_grad()
def get_state_at_time(pc,viewpoint_camera):    
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None
    active_fourier_head = getattr(pc, "fourier_head", None)
    bypass_hexplane = (
        active_fourier_head is not None
        and getattr(active_fourier_head, "pure_rigid", False)
        and getattr(active_fourier_head, "static_background", False)
    )
    if bypass_hexplane:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            means3D,
            scales,
            rotations,
            opacity,
            shs,
        )
    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    if active_fourier_head is not None:
        from gaussian_renderer import _resolve_roi_mask
        roi_mask = _resolve_roi_mask(pc, None, means3D)
        means3D_final, rotations_final = active_fourier_head(
            means3D_canon=means3D,
            means3D_deformed=means3D_final,
            rotations_canon=rotations,
            rotations_deformed=rotations_final,
            time=time,
            roi_mask=roi_mask,
        )

    return means3D_final, scales_final, rotations_final, opacity, shs_final