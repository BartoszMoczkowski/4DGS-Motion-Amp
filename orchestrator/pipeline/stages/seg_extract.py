"""``seg_extract.default`` — runs the vendored, ported copy of
``motion_seg/extract_trajectories.py``'s CLI (``pipeline/vendored/cuda/seg_extract.py``) inside
the ``cuda`` container (T08/T09).

See ``pipeline.stages.train``'s and ``pipeline.stages.cuda_common``'s module docstrings for the
general CLI-invocation-in-a-container design this follows. Needs the same GPU environment as
``train``/``render`` (loads the trained deformation network) — see the reference script's own
module docstring.
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import flag, run_cuda_script, write_stage_bridge
from .registry import register


@register("seg_extract.default")
class SegExtractStage(Stage):
    """Samples the trained deformation field at ``n_times`` evenly spaced timesteps into a dense
    per-Gaussian trajectory tensor (``trajectories.npz``) — the data adapter feeding
    ``segment.rigid``/``segment.mbs`` (T07/T10).

    ``inputs["model"]`` is ``train.default``'s trained model directory. The output path is always
    computed and passed explicitly via ``--out`` (rather than relying on the reference script's
    own ``<model_path>/trajectories.npz`` default, which it would compute from the *container*
    path) so this stage knows exactly where to find the result afterward without re-deriving the
    container's path-resolution logic on the host side.
    """

    inputs = ("model",)
    outputs = ("trajectories",)
    environment = "cuda"
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=8.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_container = ctx.paths.to_container(model.path, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        out_host = ctx.run_dir / "trajectories.npz"
        out_container = ctx.paths.to_container(out_host, env="cuda")

        cfg = ctx.config  # SegExtractConfig's own fields (iteration/n_times); `configs` ignored,
        # same as `render.default` — this stage always generates its own bridge file.
        n_times = int(cfg.get("n_times", 60))
        args = [
            *flag("model_path", str(model_container)),
            *flag("iteration", cfg.get("iteration", -1)),
            *flag("configs", str(bridge_container)),
            "--n-times", str(n_times),
            *flag("out", str(out_container)),
        ]

        run_cuda_script(ctx, "seg_extract", args, log_name="seg_extract")

        return {
            "trajectories": Artifact(
                name="trajectories",
                kind="npz",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={"n_times": n_times},
            )
        }
