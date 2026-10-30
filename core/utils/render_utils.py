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
    from scene.deformation_dispatch import dispatch_deformation
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = dispatch_deformation(
        pc=pc,
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        opacity=opacity,
        shs=shs,
        time=time,
        stage="fine",
    )

    return means3D_final, scales_final, rotations_final, opacity, shs_final