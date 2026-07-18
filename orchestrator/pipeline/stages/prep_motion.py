"""``prep_motion.default`` — runs the vendored, ported copy of ``add_motion.py``'s CLI
(``pipeline/vendored/isaac/add_motion.py``) inside the ``isaac`` container via
``/isaac-sim/python.sh`` (T11, see ``pipeline.stages.isaac_common``'s module docstring).

Consumes ``prep_split.default``'s ``segmented_mesh`` output and authors subtle, periodic, per-part
rigid motion onto it, producing the animated USD ``capture.isaac`` actually opens in Isaac Sim,
plus a ``motion_groups.json`` ground-truth motion-segment mapping (returned as a second, separate
output — a real deliverable the reference script always writes, not something this stage invents).

Named ``prep_motion.default`` (role ``prep_motion``, matching ``PipelineConfig.prep_motion``), not
``prep.motion`` — see ``pipeline.stages.prep_split``'s module docstring for why (the registry's
``role.impl`` split would otherwise collide ``prep.split``/``prep.motion`` into one ambiguous
``"prep"`` role).
"""

from __future__ import annotations

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .isaac_common import flag, opt_flag, run_isaac_script, star_list_flag
from .registry import register


@register("prep_motion.default")
class PrepMotionStage(Stage):
    """Authors per-part rigid SE(3) sinusoidal motion onto a segmented USD — same "no GPU, but
    runs in the ``isaac`` container for its ``pxr``" situation as ``prep_split.default`` (see that
    stage's docstring).
    """

    inputs = ("segmented_mesh",)
    outputs = ("animated_mesh", "motion_groups")
    environment = "isaac"
    resources = ResourceRequest(needs_gpu=False, ram_gb=2.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        segmented = ctx.inputs["segmented_mesh"]
        in_container = ctx.paths.to_container(segmented.path, env="isaac")

        out_host = ctx.run_dir / "animated_mesh.usd"
        out_container = ctx.paths.to_container(out_host, env="isaac")
        # add_motion.py always writes "<out-without-ext>_motion_groups.json" alongside --out.
        motion_groups_host = out_host.with_name(out_host.stem + "_motion_groups.json")

        cfg = ctx.config  # PrepMotionConfig's own fields.
        trans_amp = cfg.get("trans_amp_mm", [1.0, 4.0])
        rot_surface = cfg.get("rot_surface_mm", [0.5, 3.0])
        freq = cfg.get("freq", [2, 5])

        args = [
            *flag("in", str(in_container)),
            *flag("out", str(out_container)),
            *flag("group", cfg.get("group", "CONJUNTO_BOMBAS")),
            *flag("num-frames", int(cfg.get("num_frames", 60))),
            *flag("fps", cfg.get("fps", 24.0)),
            "--trans-amp-mm", str(trans_amp[0]), str(trans_amp[1]),
            "--rot-surface-mm", str(rot_surface[0]), str(rot_surface[1]),
            *flag("rot-deg-max", cfg.get("rot_deg_max", 3.0)),
            "--freq", str(int(freq[0])), str(int(freq[1])),
            *flag("groups", int(cfg.get("groups", 0))),
            *star_list_flag("exclude", cfg.get("exclude")),
            *flag("seed", int(cfg.get("seed", 0))),
            *opt_flag("plot", cfg.get("plot")),
        ]

        run_isaac_script(ctx, "add_motion", args, log_name="prep_motion")

        return {
            "animated_mesh": Artifact(
                name="animated_mesh",
                kind="usd",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={"num_frames": cfg.get("num_frames", 60), "fps": cfg.get("fps", 24.0)},
            ),
            "motion_groups": Artifact(
                name="motion_groups",
                kind="json",
                path=str(motion_groups_host),
                producing_stage=ctx.stage_name,
            ),
        }
