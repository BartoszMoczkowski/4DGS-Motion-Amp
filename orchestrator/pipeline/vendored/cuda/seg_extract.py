"""Vendored, verbatim copy of ``motion-seg/motion_seg/extract_trajectories.py`` (T09 copy-in, per
``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule; see
``pipeline.vendored.cuda``'s package docstring). Body is byte-for-byte the reference script's,
including its own argparse ``main()`` entry point — this file is executed as a separate process
inside the ``cuda`` container (``pipeline/stages/seg_extract.py`` builds the CLI invocation),
never imported by the orchestrator's own host process.

Original docstring:

    Phase 1 data adapter (.claude_notes/NOTES_4dgs_motion_segmentation.md): extract a dense
    per-Gaussian trajectory tensor from a TRAINED 4DGS multipleview model, for motion
    segmentation (see motion-seg/motion_seg/segment_rigid.py).

    Needs the training GPU environment (same devcontainer/venv as train.py / render.py) — the
    GaussianModel, its deformation network, and `.get_xyz`/`.get_opacity` all live on CUDA.
    Cannot run in a CPU-only sandbox; run this the same way you run render.py.

    Usage (mirrors render.py's argument style):
        uv run python -m motion_seg.extract_trajectories --model_path output/multipleview/pump01 \\
            --configs arguments/multipleview/pump01.py --n-times 60
"""
from __future__ import annotations

import os
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.render_utils import get_state_at_time


class _TimeCam:
    """get_state_at_time (utils/render_utils.py) only reads `.time` off its
    `viewpoint_camera` argument — no need to build a real Camera (intrinsics/image/etc.) just
    to sample the deformation field at an arbitrary timestep."""

    def __init__(self, t: float):
        self.time = t


def extract(gaussians: GaussianModel, n_times: int):
    """Sample the trained deformation field at `n_times` evenly spaced points over [0,1)
    (same convention as training: `CameraInfo.time = i / n_frames`, see
    scene/multipleview_dataset.py). Returns (canonical_xyz, traj, opacity, times) as numpy
    arrays — traj is (N, n_times, 3)."""
    times = np.linspace(0.0, 1.0, n_times, endpoint=False)
    n = gaussians.get_xyz.shape[0]
    traj = np.empty((n, len(times), 3), dtype=np.float32)
    with torch.no_grad():
        for ti, t in enumerate(times):
            means3D_final, *_ = get_state_at_time(gaussians, _TimeCam(float(t)))
            traj[:, ti, :] = means3D_final.detach().cpu().numpy()
        canonical_xyz = gaussians.get_xyz.detach().cpu().numpy()
        opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    return canonical_xyz, traj, opacity, times.astype(np.float32)


def main():
    parser = ArgumentParser(description="Extract per-Gaussian trajectories for motion segmentation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)  # unused directly, kept so get_combined_args/cfg_args match render.py
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--n-times", type=int, default=60,
                         help="evenly spaced timesteps to sample over [0,1) — match the "
                              "capture's frame count for the finest trajectory you trained on")
    # NOTE: default must NOT be None. get_combined_args (arguments/__init__.py) starts from
    # the training-time cfg_args namespace and only overlays cmdline args whose parsed value
    # is `!= None`; since `--out` was never one of train.py's own arguments, a None default
    # here means the attribute is simply absent (not None) from the merged Namespace when
    # you don't pass --out explicitly -> AttributeError instead of a clean fallback.
    parser.add_argument("--out", type=str, default="",
                         help="output .npz path (default: <model_path>/trajectories.npz)")
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
    print(f"[extract] loaded {gaussians.get_xyz.shape[0]} Gaussians from "
          f"{dataset.model_path} (iteration {scene.loaded_iter})")

    canonical_xyz, traj, opacity, times = extract(gaussians, args.n_times)

    out_path = args.out or os.path.join(dataset.model_path, "trajectories.npz")
    np.savez(out_path, traj=traj, canonical_xyz=canonical_xyz, opacity=opacity, times=times)
    print(f"[ok] wrote {out_path}  traj{traj.shape} canonical_xyz{canonical_xyz.shape} "
          f"opacity{opacity.shape}")


if __name__ == "__main__":
    main()
