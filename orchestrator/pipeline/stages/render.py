"""``render.default`` — runs the vendored, ported copy of ``render.py``'s CLI
(``pipeline/vendored/cuda/render.py``) inside the ``cuda`` container (T08/T09).

See ``pipeline.stages.train``'s and ``pipeline.stages.cuda_common``'s module docstrings for the
general CLI-invocation-in-a-container design this follows.
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import bool_flag, flag, run_cuda_script, write_stage_bridge
from .registry import register


@register("render.default")
class RenderStage(Stage):
    """Renders train/test/video views for a trained model (train/test PSNR sanity + a preview
    video), mirroring ``train_pump.sh``'s ``render.py --model_path ... --skip_train --configs
    ...`` step.

    ``inputs["model"]``'s directory already has ``cfg_args`` (written by ``train.default``) with
    the training-time ``source_path``, so — matching the reference script's own usage — this
    stage does *not* pass ``--source_path``; ``render.py``'s ``ModelParams(parser, sentinel=True)``
    + ``get_combined_args`` reads it back out of that file. Writes renders in place under the same
    model directory (``<model_path>/{train,test,video}/ours_<iteration>/...``, ``render.py``'s own
    convention) — ``outputs["renders"]`` re-registers that same directory as this stage's own
    artifact for the manifest, rather than trying to predict the ``ours_<iteration>`` folder name
    an ``iteration=-1`` (load-latest) run resolves to only *inside* the container process.
    """

    inputs = ("model",)
    outputs = ("renders",)
    environment = "cuda"
    # ram_gb 8->0 (2026-08-06): same reasoning as stages/train.py -- this stage's RAM lives
    # inside the WSL2 VM, which is hard-capped via ~/.wslconfig (memory=20GB); the host-free-RAM
    # gate only deadlocked batches against vmmemWSL's held cache. VRAM gating stays.
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=0.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_host = model.path
        model_container = ctx.paths.to_container(model_host, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        cfg = ctx.config  # RenderConfig's own fields; `configs` (legacy field) is ignored — this
        # stage always generates its own bridge file from the resolved config instead.
        args = [
            *flag("model_path", str(model_container)),
            *flag("iteration", cfg.get("iteration", -1)),
            *flag("configs", str(bridge_container)),
            *bool_flag("skip_train", cfg.get("skip_train", False)),
            *bool_flag("skip_test", cfg.get("skip_test", False)),
            *bool_flag("skip_video", cfg.get("skip_video", False)),
            *bool_flag("quiet", cfg.get("quiet", False)),
        ]

        run_cuda_script(ctx, "render", args, log_name="render")

        return {
            "renders": Artifact(
                name="renders",
                kind="dataset",
                path=str(model_host),
                producing_stage=ctx.stage_name,
                metadata={"iteration": cfg.get("iteration", -1)},
            )
        }
