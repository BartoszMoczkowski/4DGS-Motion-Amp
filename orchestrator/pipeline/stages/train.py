"""``train.default`` — runs the vendored, ported copy of ``train.py``'s CLI
(``pipeline/vendored/cuda/train.py``) inside the ``cuda`` container (T08/T09).

Unlike the T07 ``host``-environment stages (which import a graduated *function* and call it
in-process), this stage never imports the vendored script — it builds a CLI invocation and execs
it as a separate process inside the container via ``ctx.containers`` (see
``pipeline.stages.cuda_common``'s module docstring for why: the reference script's real
dependencies, ``torch``/``arguments``/``scene``/``gaussian_renderer``, only exist there). Per the
"copy the logic in, don't call the original script" rule, this still never invokes the *original*
repo-root ``train.py`` — only the ported, in-project copy.
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import (
    CudaStageError,
    bool_flag,
    flag,
    list_flag,
    opt_flag,
    run_cuda_script,
    write_stage_bridge,
)
from .registry import register


@register("train.default")
class TrainStage(Stage):
    """4DGS coarse+fine reconstruction training.

    ``inputs["scene"]`` is the 4DGS ``multipleview`` scene directory (``convert.default``'s
    ``scene`` output). The trained model is written under ``ctx.run_dir/train_out`` — not the
    legacy global ``output/multipleview/<name>/`` location train.py defaults to — by passing an
    explicit ``--model_path``, so every run's model is self-contained and artifact/cache-tracked
    like everything else T03 manages (see ``pipeline.config.bridge``'s docstring for why
    ``source_path``/``model_path`` are the two ``ModelParams`` fields *not* carried by the bridge
    file: they're derived here, per-run, instead of coming from config).
    """

    inputs = ("scene",)
    outputs = ("model",)
    environment = "cuda"
    # Rough estimates for T12 (resource manager, not yet built) — 4DGS training on a short
    # multipleview capture comfortably fits an 8-12GB card in practice; padded here since nothing
    # measures real headroom yet and a stale underestimate is worse than an overestimate.
    resources = ResourceRequest(needs_gpu=True, vram_gb=16.0, ram_gb=16.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        scene = ctx.inputs["scene"]
        source_container = ctx.paths.to_container(scene.path, env="cuda")

        model_host = ctx.run_dir / "train_out"
        model_host.mkdir(parents=True, exist_ok=True)
        model_container = ctx.paths.to_container(model_host, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        cfg = ctx.config  # TrainConfig's own fields (ip/port/.../checkpoint_iterations); see
        # `pipeline.api._stage_config_for`'s T09 note for where `_bridge` (consumed above) came
        # from.
        args = [
            *flag("source_path", str(source_container)),
            *flag("model_path", str(model_container)),
            *flag("configs", str(bridge_container)),
            *flag("ip", cfg.get("ip", "127.0.0.1")),
            *flag("port", cfg.get("port", 6009)),
            *flag("expname", cfg.get("expname") or ctx.run_id),
            *flag("debug_from", cfg.get("debug_from", -1)),
            *bool_flag("detect_anomaly", cfg.get("detect_anomaly", False)),
            *list_flag("test_iterations", cfg.get("test_iterations")),
            *list_flag("save_iterations", cfg.get("save_iterations")),
            *bool_flag("quiet", cfg.get("quiet", False)),
            *list_flag("checkpoint_iterations", cfg.get("checkpoint_iterations")),
            *opt_flag("start_checkpoint", cfg.get("start_checkpoint")),
        ]

        run_cuda_script(ctx, "train", args, log_name="train")

        # A zero exit code alone doesn't mean a checkpoint was actually written -- found on T11's
        # real-hardware run (2026-07-18): a `save_iterations` ordering bug in the vendored
        # `train.py` (see its module docstring) meant a run could train its full iteration count,
        # exit 0, and still never call `scene.save()`, leaving `point_cloud/` empty/missing. Same
        # principle as `capture.isaac`'s cameras_gt.json/camNN check -- a stage must not report
        # success (and get cross-run cached, see `pipeline.dag.cache`) without its declared
        # artifact actually existing.
        point_cloud_dir = model_host / "point_cloud"
        if not point_cloud_dir.is_dir() or not any(point_cloud_dir.iterdir()):
            raise CudaStageError(
                f"train.default exited 0 but wrote no checkpoint under {point_cloud_dir} (no "
                f"iteration_<N>/point_cloud.ply) -- see logs/train.log. Often means the trained "
                f"run's final iteration never landed in the vendored script's own "
                f"save_iterations list; see pipeline/vendored/cuda/train.py's module docstring."
            )

        return {
            "model": Artifact(
                name="model",
                kind="model",
                path=str(model_host),
                producing_stage=ctx.stage_name,
                metadata={"port": cfg.get("port", 6009), "expname": cfg.get("expname") or ctx.run_id},
            )
        }
