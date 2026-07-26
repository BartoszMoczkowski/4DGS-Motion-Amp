"""``segment.mbs`` — runs the vendored, ported copy of ``motion-seg/motion_seg/mbs_infer.py``'s CLI
(``pipeline/vendored/cuda/mbs_infer.py``) inside the ``cuda`` container (T08/T09's container-exec
model; T10 is the first task to put a *second* impl behind an existing role — see
``pipeline.stages.registry``'s module docstring for why that's exactly "add a new idea = register
an impl + a preset", no core edits).

Unlike ``segment.rigid`` (T07, ``host`` environment, pure numpy/scipy, no GPU), Option A needs
MotNet (``submodules/multibody-sync-4dgs``) on the GPU, so this stage follows the T09 ``cuda``-
stage shape instead: build a CLI invocation, exec it as a separate process inside the warm ``cuda``
container via ``pipeline.stages.cuda_common`` rather than importing anything from
``pipeline.vendored.cuda`` in-process (its real deps — ``torch`` and the MBS ``ext/`` CUDA ops —
only exist there). Unlike ``train``/``render``/``seg_extract``/``amp``, this script needs none of
the 4DGS ``ModelParams``/``PipelineParams``/``ModelHiddenParams``/``OptimizationParams`` groups, so
it never calls ``write_stage_bridge`` — ``pipeline.api._stage_config_for``'s ``"_bridge"`` merge
is scoped to exactly those four roles (T09) and deliberately excludes ``segment``.

Kept to the identical on-disk artifact shape ``segment.rigid`` writes (``points``/``labels`` npz,
see ``mbs_infer.py``'s own ``main()``) — that's the whole point of T10: ``seg_eval.default``
(T07) scores either backend's output without caring which one ran.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import flag, run_cuda_script
from .registry import register


def _resolve_checkpoint_host(ctx: StageContext, checkpoint: str) -> Path:
    """``SegmentMbsConfig.checkpoint`` has no sensible default (models.py) — it's expected to be
    either an absolute host path, or (the documented convention, see ``WINDOWS_SETUP.md``'s MBS
    setup step) a path relative to the repo root, e.g.
    ``"submodules/multibody-sync-4dgs/ckpt/<downloaded>.pth.tar"`` — the same ``ckpt/`` directory
    the reference script's own docstring points at. Relative-to-repo-root is resolved here (a
    stage-local convenience, like ``train.default``'s own ``expname`` fallback) rather than in
    ``pipeline.paths`` (T06) itself, since it's just filling in a missing base, not a host<->
    container space conversion — ``ctx.paths.to_container`` still does the real translation right
    after, and still raises if the resolved path isn't under a known root.
    """
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ctx.paths.get_roots().repo_root_host / path
    return path


@register("segment.mbs")
class SegmentMbsStage(Stage):
    """MultiBodySync MotNet inference (Option A) over a trajectories ``.npz`` — same inputs/
    outputs contract as ``segment.rigid`` (T07) so a preset's ``segment.impl: "rigid" -> "mbs"``
    is a pure config switch, nothing else about the DAG changes.
    """

    inputs = ("trajectories",)
    outputs = ("segmentation",)
    environment = "cuda"
    # MotNet itself is small (operates on an n_points~4000, n_sub=256 working set, not the full
    # Gaussian count) — far lighter than train/render/amp's full 4DGS forward/backward passes.
    # Rough estimate (T12's resource manager isn't built yet to measure real headroom), padded
    # the same way T09's stages were.
    resources = ResourceRequest(needs_gpu=True, vram_gb=4.0, ram_gb=4.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        traj = ctx.inputs["trajectories"]
        traj_container = ctx.paths.to_container(traj.path, env="cuda")

        out_host = ctx.run_dir / "segmentation.npz"
        out_container = ctx.paths.to_container(out_host, env="cuda")

        cfg = ctx.config  # SegmentMbsConfig's own fields (checkpoint/n_points/.../seed).
        checkpoint = cfg.get("checkpoint", "")
        if not checkpoint:
            # Belt-and-suspenders: SegmentConfig._check_impl_ready (T02) already rejects this at
            # config-validation time, well before a run gets anywhere near this stage. Failing
            # fast here too means a hand-built stage_configs dict (bypassing validate_config)
            # can't silently exec the container with an empty --checkpoint.
            raise ValueError(
                "segment.mbs requires a 'checkpoint' path (see SegmentConfig._check_impl_ready "
                "and WINDOWS_SETUP.md's MBS setup step)"
            )
        checkpoint_host = _resolve_checkpoint_host(ctx, checkpoint)
        checkpoint_container = ctx.paths.to_container(checkpoint_host, env="cuda")

        n_points = int(cfg.get("n_points", 4000))
        n_views = int(cfg.get("n_views", 4))

        args = [
            *flag("trajectories", str(traj_container)),
            *flag("out", str(out_container)),
            *flag("checkpoint", str(checkpoint_container)),
            *flag("n-points", n_points),
            *flag("n-views", n_views),
            *flag("n-sub", int(cfg.get("n_sub", 256))),
            *flag("opacity-thresh", cfg.get("opacity_thresh", 0.1)),
            *flag("alpha", cfg.get("alpha", 0.05)),
            *flag("seed", int(cfg.get("seed", 0))),
        ]

        run_cuda_script(ctx, "mbs_infer", args, log_name="segment_mbs")

        return {
            "segmentation": Artifact(
                name="segmentation",
                kind="npz",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={"n_points": n_points, "n_views": n_views},
            )
        }
