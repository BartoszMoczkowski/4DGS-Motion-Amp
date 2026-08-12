"""``roi.mask_oracle`` — T22 validation ceiling: a perfect ROI mask derived from ground-truth
part labels (proposal 02, ``docs/proposals/02-multiview-mask-lifting.md``).

This is a **host**-environment stage (no GPU needed).  It reads ``gt_segmentation.npz``,
maps GT labels onto the trajectory point cloud via nearest-neighbour matching in canonical
space, and writes an ``roi_mask`` artifact where every labelled (non-background) point is
marked as inside the machine ROI.

Use this to measure the upper-bound ARI achievable by any segmentation method that is
restricted to the true machine region — independent of mask-lifting quality.
"""

from __future__ import annotations

import numpy as np

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .registry import register


@register("roi.mask_oracle")
class RoiMaskOracleStage(Stage):
    inputs = ("trajectories", "gt_segmentation")
    outputs = ("roi_mask",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=1.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        traj_data = np.load(ctx.inputs["trajectories"].path)
        xyz = traj_data["canonical_xyz"]

        gt_data = np.load(ctx.inputs["gt_segmentation"].path)
        gt_points = gt_data["points"]
        gt_labels = gt_data["labels"]

        # Nearest-neighbour match: GT init cloud -> trajectory points (same convention as
        # segment_rigid2.py and seg_eval.default).
        from scipy.spatial import cKDTree
        _, nn = cKDTree(gt_points).query(xyz, k=1)
        mapped_labels = gt_labels[nn]

        # Background is conventionally label 0 or -1 in Omniverse instance segmentation.
        # Any positive label is considered part of the machine.
        roi_mask = mapped_labels > 0
        # Also include label 0 if it represents a valid part (some datasets use 0 as first part).
        # Conservative: include everything that has a non-negative label.
        roi_mask = mapped_labels >= 0

        # snr is a dummy diagnostic for the oracle (perfect knowledge = 1.0 inside, 0.0 outside).
        snr = roi_mask.astype(np.float32)

        ctx.logger.info(
            "roi.mask_oracle: n=%d gt_instances=%d roi=%d (%.1f%%)",
            len(xyz), len(np.unique(gt_labels)), int(roi_mask.sum()),
            100.0 * roi_mask.sum() / len(xyz) if len(xyz) else 0.0,
        )

        out_path = ctx.run_dir / "roi_mask.npz"
        np.savez(out_path, roi_mask=roi_mask, snr=snr)

        return {
            "roi_mask": Artifact(
                name="roi_mask",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={
                    "n_points": len(xyz),
                    "n_roi": int(roi_mask.sum()),
                    "n_gt_instances": len(np.unique(gt_labels)),
                },
            )
        }
