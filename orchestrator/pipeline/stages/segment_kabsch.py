"""``segment.kabsch`` — T20's Kabsch EM rigid-body clustering (proposal 05,
``docs/proposals/05-iterative-kabsch-em.md``), a fourth impl behind the ``segment`` role.

Follows the exact same shape as ``segment.rigid`` (T07) and ``segment.rigid2`` (T18):
- ``inputs = ("trajectories",)`` / ``outputs = ("segmentation",)``
- Reads ``trajectories.npz`` (``canonical_xyz``/``traj``/optional ``opacity``)
- Writes ``segmentation.npz`` ``{points, labels}`` with label ``-1`` for dropped floaters
- ``seg_eval.default`` is backend-agnostic

Calls :func:`pipeline.vendored.host.kabsch_em.segment_by_kabsch` in-process (host stage).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.kabsch_em import segment_by_kabsch as _segment_by_kabsch
from .base import ResourceRequest, Stage, StageContext
from .registry import register

#: Config keys forwarded verbatim to ``segment_by_kabsch``.
_KWARG_KEYS = (
    "n_clusters", "k_range", "init", "max_iter", "sigma",
    "spatial_prior", "greedy_split", "fps_subsample", "propagate_q",
    "drive_freq", "harmonics", "tolerance", "min_size",
)


@register("segment.kabsch")
class SegmentKabschStage(Stage):
    inputs = ("trajectories",)
    outputs = ("segmentation",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=4.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        data = np.load(ctx.inputs["trajectories"].path)
        xyz, traj = data["canonical_xyz"], data["traj"]
        opacity = data["opacity"] if "opacity" in data.files else None

        opacity_thresh = float(ctx.config.get("opacity_thresh", 0.1))
        kwargs = {key: ctx.config[key] for key in _KWARG_KEYS if key in ctx.config}

        # Opacity filter floaters (same convention as segment.rigid/segment.rigid2)
        n = len(xyz)
        if opacity is not None:
            keep = opacity > opacity_thresh
        else:
            keep = np.ones(n, dtype=bool)

        labels_full = np.full(n, -1, dtype=np.int64)
        sub_labels, info = _segment_by_kabsch(
            xyz[keep], traj[keep],
            rng_seed=int(ctx.config.get("rng_seed", 0)),
            **kwargs,
        )
        labels_full[keep] = sub_labels
        info["n_floaters_dropped"] = int((~keep).sum())

        ctx.logger.info("segment.kabsch: %s", {k: v for k, v in info.items()
                        if isinstance(v, (int, float, str, bool))})

        out_path = ctx.run_dir / "segmentation.npz"
        # Optional ROI gating: points outside ROI get label -2 (static), preserving -1 floaters.
        roi_artifact = ctx.inputs.get("roi_mask")
        if roi_artifact is not None:
            roi_data = np.load(roi_artifact.path)
            roi_mask = roi_data["roi_mask"]
            labels_full[(~roi_mask) & (labels_full != -1)] = -2
            ctx.logger.info("roi gated: %d inside, %d outside (-2)", int(roi_mask.sum()), int((~roi_mask).sum()))

        np.savez(out_path, points=xyz.astype(np.float32), labels=labels_full)

        return {
            "segmentation": Artifact(
                name="segmentation",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={k: v for k, v in info.items()
                          if isinstance(v, (int, float, str, bool))},
            )
        }
