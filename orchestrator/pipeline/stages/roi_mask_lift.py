"""``roi.mask_lift`` — T22's multi-view mask lifting stage (proposal 02,
``docs/proposals/02-multiview-mask-lifting.md``).

Runs ``pipeline/vendored/cuda/mask_lift.py`` inside the ``cuda`` container.
Produces the ``roi_mask`` artifact (``roi_mask.npz``) consumed optionally by downstream
``segment.*`` stages.
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import flag, run_cuda_script, write_stage_bridge
from .registry import register


@register("roi.mask_lift")
class RoiMaskLiftStage(Stage):
    inputs = ("model",)
    outputs = ("roi_mask",)
    environment = "cuda"
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=4.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_container = ctx.paths.to_container(model.path, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        out_host = ctx.run_dir / "roi_mask.npz"
        out_container = ctx.paths.to_container(out_host, env="cuda")

        cfg = ctx.config  # RoiMaskLiftConfig fields
        masks_dir = cfg.get("masks_dir", "")
        if not masks_dir:
            raise ValueError("roi.mask_lift requires roi.mask_lift.masks_dir to be set")
        masks_dir_container = ctx.paths.to_container(masks_dir, env="cuda")

        args = [
            *flag("model_path", str(model_container)),
            *flag("configs", str(bridge_container)),
            *flag("masks_dir", str(masks_dir_container)),
            *flag("ref_time", cfg.get("ref_time", 0)),
            *flag("depth_tol", cfg.get("depth_tol", 0.02)),
            *flag("vote_thresh", cfg.get("vote_thresh", 0.5)),
            *flag("dilation_hops", cfg.get("dilation_hops", 1)),
            *flag("k", cfg.get("k", 12)),
            *flag("out", str(out_container)),
        ]

        run_cuda_script(ctx, "mask_lift", args, log_name="mask_lift")

        return {
            "roi_mask": Artifact(
                name="roi_mask",
                kind="npz",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={},
            )
        }
