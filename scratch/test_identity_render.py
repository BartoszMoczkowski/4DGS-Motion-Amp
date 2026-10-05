import torch
import torch.nn as nn
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

def test():
    print("Testing chunked 16D identity feature rendering...")
    model_path = "/workspace/runs/cubes-cubes_k2_both/train_out"
    from argparse import Namespace
    cfg_file = f"{model_path}/cfg_args"
    with open(cfg_file, "r") as f:
        cfg_str = f.read()
    args = eval(cfg_str)
    pc = GaussianModel(sh_degree=args.sh_degree, args=args)
    ply_path = f"{model_path}/point_cloud/iteration_15000/point_cloud.ply"
    pc.load_ply(ply_path)
    model_iter_dir = f"{model_path}/point_cloud/iteration_15000"
    pc.load_model(model_iter_dir)
    xyz = pc.get_xyz.detach().cpu().numpy()
    pc._deformation.deformation_net.set_aabb(xyz.max(axis=0), xyz.min(axis=0))
    pc._deformation.to("cuda")
    pc._deformation.eval()
    for p in pc._deformation.parameters():
        p.requires_grad = False
    N = pc.get_xyz.shape[0]
    print(f"Loaded {N} Gaussians and deformation network")

    pc._xyz.requires_grad = False
    pc._scaling.requires_grad = False
    pc._rotation.requires_grad = False
    pc._opacity.requires_grad = False
    pc._features_dc.requires_grad = False
    pc._features_rest.requires_grad = False

    identity = nn.Parameter(torch.randn(N, 16, device="cuda", requires_grad=True))
    optimizer = torch.optim.Adam([identity], lr=0.01)

    from scene.cameras import MiniCam
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
    import numpy as np

    W, H = 800, 450
    fovx = focal2fov(1000.0, W)
    fovy = focal2fov(1000.0, H)
    w2v = torch.eye(4, device="cuda")
    w2v[2, 3] = 3.0
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full_proj = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    cam = MiniCam(
        width=W, height=H, fovy=fovy, fovx=fovx,
        znear=0.01, zfar=100.0,
        world_view_transform=w2v,
        full_proj_transform=full_proj,
        time=0.0
    )

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    pipe = Pipe()

    bg_zero = torch.zeros(3, device="cuda")
    chunks = []
    for c in range(0, 16, 3):
        chunk_feat = identity[:, c:min(c+3, 16)]
        if chunk_feat.shape[1] < 3:
            pad = torch.zeros(N, 3 - chunk_feat.shape[1], device="cuda")
            chunk_feat = torch.cat([chunk_feat, pad], dim=1)
        res = render(cam, pc, pipe, bg_zero, override_color=chunk_feat, stage="fine")
        chunks.append(res["render"][:min(3, 16 - c)])

    rendered_feat = torch.cat(chunks, dim=0)
    print(f"Rendered feature map shape: {rendered_feat.shape}")
    assert rendered_feat.shape == (16, H, W), f"Expected (16, {H}, {W}), got {rendered_feat.shape}"

    loss = (rendered_feat ** 2).sum()
    loss.backward()
    assert identity.grad is not None, "Gradients should flow to identity parameter"
    print(f"Loss: {loss.item():.4f}, identity grad norm: {identity.grad.norm().item():.4f}")
    optimizer.step()
    print("SUCCESS: Chunked 16D feature rendering and backward pass verified!")

if __name__ == "__main__":
    test()
