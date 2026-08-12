"""``roi.motion_gate`` — T19's band-limited energy gate + k-NN dilation +
rigidity-lock readmission (proposal 01, ``docs/proposals/01-motion-gated-roi.md``).

Produces the ``roi_mask`` artifact (``roi_mask.npz``) consumed optionally by downstream
``segment.*`` stages.  Points outside the ROI receive label ``-2`` ("static") in the
segmentation, while opacity floaters stay ``-1``.
"""

from __future__ import annotations

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.motion_gate import motion_gate as _motion_gate
from .base import ResourceRequest, Stage, StageContext
from .registry import register

#: Config keys forwarded verbatim to :func:`_motion_gate`.
_KWARG_KEYS = (
    "drive_freq", "harmonics", "dilation_hops", "readmit_mult", "k",
)


@register("roi.motion_gate")
class RoiMotionGateStage(Stage):
    inputs = ("trajectories",)
    outputs = ("roi_mask",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=1.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        data = np.load(ctx.inputs["trajectories"].path)
        xyz, traj = data["canonical_xyz"], data["traj"]

        kwargs = {key: ctx.config[key] for key in _KWARG_KEYS if key in ctx.config}

        roi_mask, snr, info = _motion_gate(xyz, traj, **kwargs)

        ctx.logger.info(
            "roi.motion_gate: n=%d moving_init=%d dilated=%d readmitted=%d freq=%s sigma_d=%.6f",
            info["n_points"], info["n_moving_init"], info["n_dilated"],
            info["n_readmitted"], info["drive_freq_used"], info["sigma_d"],
        )

        out_path = ctx.run_dir / "roi_mask.npz"
        np.savez(out_path, roi_mask=roi_mask, snr=snr)

        return {
            "roi_mask": Artifact(
                name="roi_mask",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={k: v for k, v in info.items()
                          if isinstance(v, (int, float, str, bool))},
            )
        }
