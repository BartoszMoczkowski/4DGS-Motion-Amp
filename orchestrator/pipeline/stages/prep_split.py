"""``prep_split.default`` — runs the vendored, ported copy of ``split_mesh.py``'s CLI
(``pipeline/vendored/isaac/split_mesh.py``) inside the ``isaac`` container via
``/isaac-sim/python.sh`` (T11, see ``pipeline.stages.isaac_common``'s module docstring).

First stage of the new prep/capture front end this task adds: splits a single fused CAD mesh
(``inputs["raw_mesh"]``, e.g. ``CONJUNTO_BOMBAS.usd`` — an external asset, no producer in this
repo, pre-seeded like ``capture``/``gt_segmentation`` were before this task, see
``pipeline.stages.convert``'s docstring for the established pattern) into per-part-labelled USD
prims by connected components. Feeds ``prep_motion.default``, which authors rigid per-part
animation onto this stage's output.

Named ``prep_split.default`` (role ``prep_split``, matching ``PipelineConfig.prep_split`` — T02's
own top-level config section name), not ``prep.split`` as ``planning/ARCHITECTURE.md``'s original
stage table sketched: the registry's ``role.impl`` split (``pipeline.stages.registry.register``)
takes everything before the *first* dot as the role, so ``prep.split``/``prep.motion`` would both
have collided into one ``"prep"`` role with two impls (``split``/``motion``) — not two different
alternative implementations of the same thing, which is what "multiple impls of one role" is
supposed to mean (like ``segment.rigid``/``segment.mbs``). ``pipeline.api._auto_stage_plan`` would
then treat ``"prep"`` as needing a ``resolved_config["prep"]["impl"]`` disambiguator that doesn't
exist, and ``_stage_config_for`` would fail to slice either stage's own section (there is no
top-level ``resolved_config["prep"]`` — only ``prep_split``/``prep_motion``). Caught while wiring
this stage into a full auto-planned run for the first time; fixed by naming each its own
single-impl role instead, matching its config section 1:1 — see ``planning/TASKS.md``'s T11 note.
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .isaac_common import bool_flag, flag, opt_flag, run_isaac_script
from .registry import register


@register("prep_split.default")
class PrepSplitStage(Stage):
    """Splits a fused CAD mesh into labelled per-part USD prims (``prep_motion.default``'s
    downstream input) — no GPU needed (plain USD/trimesh CPU work), but runs in the ``isaac``
    container anyway since that's the only image with ``pxr`` (usd-core) available today; see
    ``pipeline.vendored.isaac``'s package docstring for why no separate small-CPU image exists yet.
    """

    inputs = ("raw_mesh",)
    outputs = ("segmented_mesh",)
    environment = "isaac"
    resources = ResourceRequest(needs_gpu=False, ram_gb=2.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        raw_mesh = ctx.inputs["raw_mesh"]
        in_container = ctx.paths.to_container(raw_mesh.path, env="isaac")

        out_host = ctx.run_dir / "segmented_mesh.usd"
        out_container = ctx.paths.to_container(out_host, env="isaac")

        cfg = ctx.config  # PrepSplitConfig's own fields (group/min_faces/preview/no_usd/weld_tol).
        args = [
            *flag("in", str(in_container)),
            *flag("out", str(out_container)),
            *flag("group", cfg.get("group", "CONJUNTO_BOMBAS")),
            *flag("min-faces", int(cfg.get("min_faces", 1))),
            *opt_flag("preview", cfg.get("preview")),
            *bool_flag("no-usd", cfg.get("no_usd", False)),
        ]
        # `PrepSplitConfig.weld_tol` has no effect: split_mesh.py's split_components() takes a
        # weld_tol parameter, but its own main() never passes it through via any CLI flag — a
        # pre-existing gap in the reference script itself (see
        # pipeline/vendored/isaac/split_mesh.py's module docstring), not something this port
        # fixes ("copy the logic in, don't rewrite").

        run_isaac_script(ctx, "split_mesh", args, log_name="prep_split")

        return {
            "segmented_mesh": Artifact(
                name="segmented_mesh",
                kind="usd",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={"group": cfg.get("group", "CONJUNTO_BOMBAS")},
            )
        }
