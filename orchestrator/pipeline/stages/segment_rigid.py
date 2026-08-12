"""``segment.rigid`` — calls the vendored, ported copy of
``motion-seg/motion_seg/segment_rigid.py``'s ``segment_trajectories()``.

Per the "copy the logic in, don't call the original script" rule (``planning/INSTRUCTIONS.md``,
2026-07-14, superseding "wrap, don't rewrite"), this calls
:func:`pipeline.vendored.host.segment_rigid.segment_trajectories` — an in-project, verbatim port
— rather than reaching into ``motion_seg`` from outside the package. Impl name ``rigid`` matches
``SegmentConfig.impl`` (T02) so a preset's ``segment.impl: "rigid"`` resolves straight to this
class via the registry (T04). Its real upstream producer is ``seg_extract`` (GPU, T09, out of
scope here) which writes ``trajectories.npz`` — so, like ``convert.default``'s ``capture`` input,
``trajectories`` is always an external input for T07, pre-seeded into the run's manifest before
``run_dag`` runs.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.segment_rigid import segment_trajectories as _segment_trajectories
from .base import ResourceRequest, Stage, StageContext
from .registry import register


@register("segment.rigid")
class SegmentRigidStage(Stage):
    """Local-rigidity-graph motion segmentation (Option B) over a trajectories ``.npz``.

    ``inputs["trajectories"]`` must have ``canonical_xyz``/``traj`` arrays (and optionally
    ``opacity``) — the exact shape ``extract_trajectories.py`` writes and ``segment_rigid.py``'s
    own CLI reads via ``np.load(args.trajectories)``.
    """

    inputs = ("trajectories",)
    outputs = ("segmentation",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=1.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        data = np.load(ctx.inputs["trajectories"].path)
        xyz, traj = data["canonical_xyz"], data["traj"]
        opacity = data["opacity"] if "opacity" in data.files else None

        k = int(ctx.config.get("k", 12))
        min_size = int(ctx.config.get("min_size", 15))
        threshold_mult = float(ctx.config.get("threshold_mult", 1.0))
        opacity_thresh = float(ctx.config.get("opacity_thresh", 0.1))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            labels, info = _segment_trajectories(
                xyz,
                traj,
                opacity=opacity,
                opacity_thresh=opacity_thresh,
                k=k,
                threshold_mult=threshold_mult,
                min_size=min_size,
            )
        for line in buf.getvalue().splitlines():
            ctx.logger.info(line)
        ctx.logger.info("segment.rigid: %s", info)

        out_path = ctx.run_dir / "segmentation.npz"
        # Same on-disk shape segment_rigid.py's own CLI writes (points, labels) — evaluate_
        # segmentation.py's --pred / seg_eval.default both read exactly this.
        # Optional ROI gating: points outside ROI get label -2 (static), preserving -1 floaters.
        roi_artifact = ctx.inputs.get("roi_mask")
        if roi_artifact is not None:
            roi_data = np.load(roi_artifact.path)
            roi_mask = roi_data["roi_mask"]
            labels[(~roi_mask) & (labels != -1)] = -2
            ctx.logger.info("roi gated: %d inside, %d outside (-2)", int(roi_mask.sum()), int((~roi_mask).sum()))

        np.savez(out_path, points=xyz.astype(np.float32), labels=labels)

        return {
            "segmentation": Artifact(
                name="segmentation",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={k2: v2 for k2, v2 in info.items()},
            )
        }
