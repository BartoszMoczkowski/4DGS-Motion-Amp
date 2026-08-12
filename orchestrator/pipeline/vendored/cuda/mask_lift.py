"""Multi-view mask lifting for T22 (``roi.mask_lift``, proposal 02 in
``docs/proposals/02-multiview-mask-lifting.md``).

Loads a trained 4DGS model, deforms Gaussians to a reference time, renders a depth map
from every calibrated camera, and lifts per-view binary masks into a 3D Gaussian ROI mask
via occlusion-corrected projection voting.

Runs inside the ``cuda`` container (needs torch + the rasterizer).  This is a standalone
argparse CLI, never imported by the orchestrator host process.

Mask directory layout (any of the following, checked in order):
  1. ``<masks_dir>/camNN/frame_00001.png``  — one mask per frame, matching image layout
  2. ``<masks_dir>/camNN.png``               — one static mask per camera
  3. ``<masks_dir>/mask_camNN.png``          — alternate static naming

Mask images are read as grayscale and thresholded at 128 (white = foreground).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

# NOTE: torch and the 4DGS core imports (arguments / gaussian_renderer / scene / utils)
# are deferred into the functions that use them so this module stays importable on the
# host (orchestrator sandbox tests import the pure-numpy helpers).  They rely on
# PYTHONPATH=/workspace/core inside the cuda container.


class _TimeCam:
    """Dummy camera with only ``.time`` — sufficient for ``get_state_at_time``."""
    def __init__(self, t: float):
        self.time = t


def _load_mask(mask_path: str, target_size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Load a binary mask image.  Returns bool[H,W] or None if missing."""
    if not os.path.isfile(mask_path):
        return None
    im = Image.open(mask_path).convert("L")
    if target_size is not None and im.size != target_size:
        im = im.resize(target_size, Image.NEAREST)
    return np.array(im) > 128


def _find_mask_path(masks_dir: str, cam_name: str, frame_name: str | None = None) -> str | None:
    """Resolve mask path for a camera / frame, or None if not found."""
    candidates = []
    if frame_name is not None:
        candidates.append(os.path.join(masks_dir, cam_name, frame_name))
        # Try same name with .png extension
        base, _ = os.path.splitext(frame_name)
        candidates.append(os.path.join(masks_dir, cam_name, base + ".png"))
    # Static mask per camera
    candidates.append(os.path.join(masks_dir, f"{cam_name}.png"))
    candidates.append(os.path.join(masks_dir, f"mask_{cam_name}.png"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _build_knn_edges(xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """Return (E,2) int64 edge array for an undirected k-NN graph."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    src = np.repeat(np.arange(len(xyz)), k)
    dst = idx[:, 1:].reshape(-1)
    edges = np.stack([src, dst], axis=1)
    # Symmetrise
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return edges


def _dilate_graph(mask: np.ndarray, edges: np.ndarray, hops: int = 1) -> np.ndarray:
    """Dilate a bool mask along an undirected edge graph."""
    roi = mask.copy()
    for _ in range(hops):
        border = roi[edges[:, 0]] | roi[edges[:, 1]]
        roi[edges[border].reshape(-1)] = True
    return roi


def _make_minicam(R: np.ndarray, T: np.ndarray, W: int, H: int, fx: float, fy: float,
                  znear: float = 0.01, zfar: float = 100.0, time: float = 0.0) -> MiniCam:
    """Build a MiniCam from COLMAP-style R,T and pinhole intrinsics."""
    import torch
    from scene.cameras import MiniCam
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)
    w2v = getWorld2View2(R, T, translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
    world_view_transform = torch.tensor(w2v, dtype=torch.float32).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    return MiniCam(
        width=W, height=H, fovy=FoVy, fovx=FoVx,
        znear=znear, zfar=zfar,
        world_view_transform=world_view_transform.cuda(),
        full_proj_transform=full_proj_transform.cuda(),
        time=time,
    )


def lift_masks(
    gaussians: GaussianModel,
    pipeline_params: PipelineParams,
    cameras: list[MiniCam],
    masks: list[np.ndarray | None],
    ref_time: float = 0.0,
    depth_tol: float = 0.02,
    vote_thresh: float = 0.5,
    dilation_hops: int = 1,
    k: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lift per-view binary masks to a per-Gaussian ROI mask.

    Args:
        gaussians: trained GaussianModel (on CUDA).
        pipeline_params: PipelineParams for rendering.
        cameras: list of MiniCam, one per view.
        masks: parallel list of bool[H,W] or None.
        ref_time: deformation time.
        depth_tol: view-space depth tolerance for occlusion test.
        vote_thresh: fraction of visible views that must vote foreground.
        dilation_hops: k-NN graph dilation hops to fill interior holes.
        k: k for the k-NN graph used in dilation.

    Returns:
        roi_mask: bool[N] — True = inside machine ROI.
        votes: float32[N] — raw vote score (numerator/denominator).
        info: dict with diagnostics.
    """
    import torch
    from gaussian_renderer import render
    from utils.render_utils import get_state_at_time

    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # Deform to ref_time
    with torch.no_grad():
        means3D_final, *_ = get_state_at_time(gaussians, _TimeCam(float(ref_time)))

    # Project all means to all cameras in one batch for speed
    means_world = means3D_final  # (N, 3)

    vote_num = np.zeros(n, dtype=np.float32)
    vote_den = np.zeros(n, dtype=np.float32)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    pipe = pipeline_params

    for ci, (cam, mask) in enumerate(zip(cameras, masks)):
        if mask is None:
            print(f"[mask_lift] camera {ci}: no mask, skipping")
            continue

        # Render depth
        with torch.no_grad():
            out = render(cam, gaussians, pipe, bg_color, stage="fine")
        depth_map = out["depth"].squeeze().cpu().numpy()  # (H, W)
        H, W = depth_map.shape

        # Project means to this camera
        means_homo = torch.cat([means_world, torch.ones(n, 1, device=device)], dim=1)  # (N, 4)
        proj = (cam.full_proj_transform @ means_homo.T).T  # (N, 4)
        proj = proj.cpu().numpy()

        # NDC -> screen
        w = proj[:, 3]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)
        ndc = proj[:, :3] / w[:, None]
        u = (ndc[:, 0] + 1.0) * 0.5 * W
        v = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * H  # flip Y for image coords
        pix_u = np.rint(u).astype(np.int32)
        pix_v = np.rint(v).astype(np.int32)

        # View-space z for depth comparison
        view_homo = (cam.world_view_transform.cpu() @ means_homo.T).T.cpu().numpy()
        z_view = view_homo[:, 2]

        # Frustum bounds
        in_frustum = (
            (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
            (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0) &
            (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
        )

        # Pixel bounds
        in_image = (pix_u >= 0) & (pix_u < W) & (pix_v >= 0) & (pix_v < H)

        # Mask hit
        in_mask = np.zeros(n, dtype=bool)
        valid = in_frustum & in_image
        in_mask[valid] = mask[pix_v[valid], pix_u[valid]]

        # Depth match (occlusion test)
        depth_match = np.zeros(n, dtype=bool)
        if valid.any():
            d_render = depth_map[pix_v[valid], pix_u[valid]]
            # The rasterizer returns view-space z; compare with our computed z_view.
            # Use absolute tolerance.  A Gaussian is considered visible at this pixel if
            # its view-space z is within depth_tol of the rendered depth there.
            depth_match[valid] = np.abs(z_view[valid] - d_render) < depth_tol

        # Count votes
        visible = depth_match  # only count if not occluded
        vote_num += in_mask & visible
        vote_den += visible

        print(f"[mask_lift] camera {ci}: visible={int(visible.sum())} mask_hits={int((in_mask & visible).sum())}")

    # Vote score with epsilon to avoid div-by-zero
    eps = 1e-8
    votes = vote_num / np.maximum(vote_den, eps)
    roi = votes > vote_thresh

    # Graph dilation to fill interior holes
    if dilation_hops > 0:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        edges = _build_knn_edges(xyz, k=k)
        roi = _dilate_graph(roi, edges, hops=dilation_hops)

    n_roi = int(roi.sum())
    info = {
        "n_points": n,
        "n_cameras": len(cameras),
        "n_roi": n_roi,
        "n_roi_ratio": n_roi / n if n else 0.0,
        "vote_thresh": vote_thresh,
        "depth_tol": depth_tol,
        "dilation_hops": dilation_hops,
    }
    return roi, votes.astype(np.float32), info


def main():
    import torch  # noqa: F401  (ensures CUDA env before core imports)
    from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from scene import Scene
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description="Lift 2D masks to 3D Gaussian ROI mask")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="directory containing per-camera mask images")
    parser.add_argument("--ref-time", type=float, default=0.0,
                        help="deformation time for reference frame (0..1)")
    parser.add_argument("--depth-tol", type=float, default=0.02)
    parser.add_argument("--vote-thresh", type=float, default=0.5)
    parser.add_argument("--dilation-hops", type=int, default=1)
    parser.add_argument("--k", type=int, default=12,
                        help="k for k-NN graph dilation")
    parser.add_argument("--out", type=str, default="",
                        help="output .npz path (default: <model_path>/roi_mask.npz)")
    args = get_combined_args(parser)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(True)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    print(f"[mask_lift] loaded {gaussians.get_xyz.shape[0]} Gaussians from "
          f"{dataset.model_path} (iteration {scene.loaded_iter})")

    # Load cameras from the multipleview dataset.
    # We need one camera per view (not per frame).  For multipleview, the dataset's
    # train_camera has one entry per (camera, frame).  The first frame of each camera
    # gives us the static camera pose.
    train_ds = scene.getTrainCameras().dataset  # unwrap FourDGSdataset -> multipleview_dataset
    n_frames = len(os.listdir(os.path.join(dataset.source_path, "cam01")))
    n_cams = len(train_ds.image_paths) // n_frames
    print(f"[mask_lift] scene has {n_cams} cameras, {n_frames} frames per camera")

    # Determine image resolution from the first actual image (image_paths are strings).
    im0 = Image.open(train_ds.image_paths[0])
    W, H = im0.size
    fx = fy = train_ds.focal[0]

    # For each camera, grab the first-frame pose and build a MiniCam.
    cameras: list[MiniCam] = []
    masks: list[np.ndarray | None] = []
    for ci in range(n_cams):
        idx = ci * n_frames  # first frame of this camera
        _, (R, T), time = train_ds[idx]
        cam = _make_minicam(R, T, W, H, fx, fy, time=time)
        cameras.append(cam)

        cam_name = f"cam{ci + 1:02d}"
        mask_path = _find_mask_path(args.masks_dir, cam_name, frame_name="frame_00001.png")
        mask = _load_mask(mask_path, target_size=(W, H)) if mask_path else None
        masks.append(mask)
        print(f"[mask_lift] camera {ci} ({cam_name}): mask={mask_path is not None}")

    pipe = pipeline.extract(args)
    roi_mask, votes, info = lift_masks(
        gaussians, pipe, cameras, masks,
        ref_time=args.ref_time,
        depth_tol=args.depth_tol,
        vote_thresh=args.vote_thresh,
        dilation_hops=args.dilation_hops,
        k=args.k,
    )

    out_path = args.out or os.path.join(dataset.model_path, "roi_mask.npz")
    np.savez(out_path, roi_mask=roi_mask, snr=votes)
    print(f"[ok] wrote {out_path}  roi={int(roi_mask.sum())}/{len(roi_mask)} "
          f"ratio={info['n_roi_ratio']:.3f}")


if __name__ == "__main__":
    main()
