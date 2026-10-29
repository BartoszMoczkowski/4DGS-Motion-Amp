"""Minimal memory-light render: one camera's full frame series, saved per-frame.

Runs INSIDE the cuda container (core/ on sys.path via /workspace bind-mount).
Renders the trained model (HexPlane deformation; no Fourier head — mirrors
vendored render.py) for a single named camera across all 60 frames, writing
each PNG immediately (no list accumulation, no video encode) to avoid the
OOM that killed render.default on the 600-frame train split.

Usage (container):
  python /workspace/scratch/render_cam_series.py \
      --model_path /workspace/runs/<run_id>/train_out \
      --configs /workspace/runs/<run_id>/render_default_arguments_bridge.py \
      --cam cam01 --out /workspace/runs/<run_id>/cam01_series
"""
import os
import sys

sys.path.insert(0, "/workspace/core")

import torch
import torchvision
from argparse import ArgumentParser

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--cam", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fourier", action="store_true",
                        help="Load fourier_head.pth + roi_voxels.npz and apply the head "
                             "(training-time output; default off mirrors vendored render.py)")
    parser.add_argument("--only-frame", type=int, default=-1,
                        help="Render only the N-th frame of this camera's series (0-based) "
                             "and stop; for repairing single corrupt frames")
    args = get_combined_args(parser)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)

    dataset = model.extract(args)
    hyper = hyperparam.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        fourier_head = None
        if args.fourier:
            import json
            from scene.fourier_rigid import FourierRigidHead
            from scene.roi_grid import load_roi_grid
            state = json.load(open(os.path.join(args.model_path, "fourier_head_state.json")))
            fps = scene.train_camera.fps if hasattr(scene, "train_camera") else 20.0
            img_len = getattr(scene.train_camera.dataset, "image_length", 60) if hasattr(scene, "train_camera") else 60
            fourier_head = FourierRigidHead(
                base_frequency_hz=state["frequency_hz"],
                num_harmonics=state["num_harmonics"],
                fps=fps,
                image_length=img_len,
                pure_rigid=state.get("pure_rigid", False),
                static_background=True,  # r14 runs set this; harmless if head was trained without it
            ).cuda()
            fourier_head.load_state_dict(torch.load(os.path.join(args.model_path, "fourier_head.pth")))
            fourier_head.eval()
            roi_npz = os.path.join(dataset.source_path, "roi_voxels.npz")
            gaussians._roi_grid = load_roi_grid(roi_npz, device="cuda")
            print(f"loaded fourier head ({state['frequency_hz']:.3f} Hz) + ROI grid")
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        print(f"total train views: {len(views)}", flush=True)
        # Positional pre-slice (dataset is camera-major). NEVER hold more than
        # one Camera at a time: each Camera carries its 6MP image on cuda
        # (~75MB); a 261-view list is ~20GB VRAM -> OOM-kill (exit 137).
        n_per_cam = len(views) // 10
        cam_idx = int(args.cam.replace("cam", "")) - 1
        start, end = cam_idx * n_per_cam, (cam_idx + 1) * n_per_cam
        v0 = views[start]
        name0 = str(getattr(v0, "cam_name", "") or getattr(v0, "image_name", ""))
        positional_ok = args.cam in name0
        del v0
        import gc; gc.collect(); torch.cuda.empty_cache()
        if not positional_ok:
            print(f"positional slice mismatch ({name0}); falling back to name scan", flush=True)

        os.makedirs(args.out, exist_ok=True)
        n_done = 0
        for idx in range(len(views)):
            if positional_ok and not (start <= idx < end):
                continue
            v = views[idx]
            if not positional_ok:
                nm = str(getattr(v, "cam_name", "") or getattr(v, "image_name", ""))
                if args.cam not in nm:
                    del v
                    continue
            if 0 <= args.only_frame < n_done:
                del v
                break
            if args.only_frame >= 0 and n_done != args.only_frame:
                del v
                n_done += 1
                continue
            r = render(v, gaussians, pipe, background, cam_type=scene.dataset_type,
                       fourier_head=fourier_head)["render"]
            torchvision.utils.save_image(r, os.path.join(args.out, f"{n_done:05d}.png"))
            del v, r
            n_done += 1
            if args.only_frame >= 0:
                break
            if n_done % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()
                print(f"  {n_done} frames", flush=True)
        print(f"rendered {n_done} views of {args.cam} -> {args.out}")


if __name__ == "__main__":
    main()
