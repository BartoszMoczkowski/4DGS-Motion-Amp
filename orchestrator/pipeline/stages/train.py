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

import hashlib
import json
from pathlib import Path
from typing import Any

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


def get_regularization_weights(
    iteration: int, hyper: Any
) -> tuple[float, float, float]:
    """Compute active regularization weights for the current fine-stage iteration.

    Returns:
        (time_smoothness_weight, l1_time_planes_weight, plane_tv_weight)

    Schedule behavior (R1):
        - If regularizer_anneal_iter <= 0: schedule is inactive; returns initial weights.
        - If iteration < regularizer_anneal_iter: returns initial weights while geometry stabilizes.
        - If iteration >= regularizer_anneal_iter:
            - If regularizer_anneal_type == "linear" and regularizer_anneal_end_iter > regularizer_anneal_iter:
                linearly anneals from initial to final weights until regularizer_anneal_end_iter.
            - Otherwise ("step" / default):
                immediately switches to the final weights.

    Unset or negative (< 0) `_final` values preserve the initial weight for that channel.
    """
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(hyper, dict):
            return hyper.get(key, default)
        return getattr(hyper, key, default)

    ts_init = float(_get("time_smoothness_weight", 0.0) or 0.0)
    l1_init = float(_get("l1_time_planes", 0.0) or 0.0)
    ptv_init = float(_get("plane_tv_weight", 0.0) or 0.0)

    anneal_iter = _get("regularizer_anneal_iter", 0)
    if anneal_iter is None:
        anneal_iter = 0
    anneal_iter = int(anneal_iter)

    if anneal_iter <= 0 or iteration < anneal_iter:
        return ts_init, l1_init, ptv_init

    ts_final_val = _get("time_smoothness_weight_final", None)
    l1_final_val = _get("l1_time_planes_final", None)
    ptv_final_val = _get("plane_tv_weight_final", None)

    ts_target = float(ts_final_val) if (ts_final_val is not None and float(ts_final_val) >= 0) else ts_init
    l1_target = float(l1_final_val) if (l1_final_val is not None and float(l1_final_val) >= 0) else l1_init
    ptv_target = float(ptv_final_val) if (ptv_final_val is not None and float(ptv_final_val) >= 0) else ptv_init

    anneal_type = str(_get("regularizer_anneal_type", "step") or "step").lower()
    end_iter = _get("regularizer_anneal_end_iter", 0)
    if end_iter is None:
        end_iter = 0
    end_iter = int(end_iter)

    if anneal_type == "linear" and end_iter > anneal_iter:
        if iteration >= end_iter:
            return ts_target, l1_target, ptv_target
        alpha = (iteration - anneal_iter) / float(end_iter - anneal_iter)
        ts_weight = (1.0 - alpha) * ts_init + alpha * ts_target
        l1_weight = (1.0 - alpha) * l1_init + alpha * l1_target
        ptv_weight = (1.0 - alpha) * ptv_init + alpha * ptv_target
        return ts_weight, l1_weight, ptv_weight

    # Default: "step" iteration-gated weight switch
    return ts_target, l1_target, ptv_target


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
    # ram_gb 16->0 (2026-08-06): the host-free-RAM gate is miscalibrated for Docker-Desktop
    # stages -- this stage's RAM lives inside the WSL2 VM, which is now hard-capped via
    # ~/.wslconfig (memory=20GB), so the cap, not the gate, protects the host. A nonzero
    # estimate just deadlocked batches: vmmemWSL legitimately holds its cap between runs,
    # leaving "host free" permanently below any estimate.
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=0.0)

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

        model_meta: dict[str, Any] = {
            "port": cfg.get("port", 6009),
            "expname": cfg.get("expname") or ctx.run_id,
        }
        offsets_json = Path(scene.path) / "offsets.json"
        if offsets_json.is_file():
            model_meta["offsets_file_sha256"] = hashlib.sha256(offsets_json.read_bytes()).hexdigest()

        learned_offsets_file = model_host / "learned_offsets.json"
        if learned_offsets_file.is_file():
            try:
                model_meta["learned_offsets"] = json.loads(
                    learned_offsets_file.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        artifacts = {
            "model": Artifact(
                name="model",
                kind="model",
                path=str(model_host),
                producing_stage=ctx.stage_name,
                metadata=model_meta,
            )
        }
        if learned_offsets_file.is_file():
            artifacts["learned_offsets"] = Artifact(
                name="learned_offsets",
                kind="json",
                path=str(learned_offsets_file),
                producing_stage=ctx.stage_name,
                metadata={"is_learned_offsets": True},
            )
        curves_file = model_host / "training_curves.json"
        if curves_file.is_file():
            artifacts["training_curves"] = Artifact(
                name="training_curves",
                kind="json",
                path=str(curves_file),
                producing_stage=ctx.stage_name,
                metadata={"is_training_curves": True},
            )
        return artifacts
