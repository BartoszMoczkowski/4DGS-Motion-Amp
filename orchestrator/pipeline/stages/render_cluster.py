"""``render.cluster`` — renders 4DGS cluster-colored video inside the ``cuda`` container (T32).

Runs the vendored ``render_cluster.py`` script (``pipeline/vendored/cuda/render_cluster.py``)
to render views with Gaussians dynamically colored according to their motion segmentation
cluster labels, optionally with RGB reconstruction side-by-side.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import (
    CudaStageError,
    bool_flag,
    flag,
    run_cuda_script,
    write_stage_bridge,
)
from .registry import register


@register("render.cluster")
class RenderClusterStage(Stage):
    """Renders novel view video with Gaussians colored by motion segmentation cluster labels.

    Inputs:
        model: Trained 4DGS model directory.
        segmentation: Motion segmentation result (.npz) containing labels.
        scene (optional): 4DGS multipleview dataset directory (for cameras/bounds).

    Outputs:
        cluster_video: Rendered cluster video (.mp4).
    """

    inputs = ("model", "segmentation")
    outputs = ("cluster_video",)
    environment = "cuda"
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=0.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_container = ctx.paths.to_container(model.path, env="cuda")

        cfg = ctx.config
        seg_path_cfg = cfg.get("segmentation_path")
        if seg_path_cfg:
            seg_host = Path(seg_path_cfg)
        else:
            seg = ctx.inputs["segmentation"]
            seg_host = Path(seg.path)
        seg_container = ctx.paths.to_container(seg_host, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        output_video_name = cfg.get("output_video", "cluster_video.mp4")
        out_host = ctx.run_dir / output_video_name
        out_container = ctx.paths.to_container(out_host, env="cuda")

        args = [
            *flag("model_path", str(model_container)),
            *flag("segmentation_path", str(seg_container)),
            *flag("output_video", str(out_container)),
            *flag("configs", str(bridge_container)),
            *flag("camera", cfg.get("camera", "cam01")),
            *flag("fps", cfg.get("fps", 25)),
            *flag("resolution_scale", cfg.get("resolution_scale", 0.5)),
            *flag("iteration", cfg.get("iteration", -1)),
            *bool_flag("side_by_side", cfg.get("side_by_side", True)),
            *bool_flag("add_labels", cfg.get("add_labels", True)),
            *bool_flag("quiet", cfg.get("quiet", False)),
        ]

        scene = ctx.inputs.get("scene")
        if scene is not None:
            scene_container = ctx.paths.to_container(scene.path, env="cuda")
            args.extend(flag("source_path", str(scene_container)))

        run_cuda_script(ctx, "render_cluster", args, log_name="render_cluster")

        if not out_host.is_file() or out_host.stat().st_size == 0:
            raise CudaStageError(
                f"render.cluster exited 0 but wrote no video at {out_host} — see logs/render_cluster.log"
            )

        return {
            "cluster_video": Artifact(
                name="cluster_video",
                kind="video",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={
                    "camera": cfg.get("camera", "cam01"),
                    "fps": cfg.get("fps", 25),
                    "resolution_scale": cfg.get("resolution_scale", 0.5),
                    "side_by_side": cfg.get("side_by_side", True),
                    "add_labels": cfg.get("add_labels", True),
                },
            )
        }
