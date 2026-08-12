"""``segment.rigid2`` — T18's denoising + calibrated-z + spectral-partition impl behind the
``segment`` role (proposal 06, ``docs/proposals/06-multiscale-snr-multiscale.md``), a third
alternative to ``segment.rigid`` (T07) / ``segment.mbs`` (T10), selected by
``segment.impl: "rigid2"`` in a preset — no core edits, exactly the plugin shape T10 proved.

Calls :func:`pipeline.vendored.host.segment_rigid2.segment_trajectories2` in-process (host
stage, same shape as ``segment.rigid``). Same I/O contract as the other impls — reads
``trajectories.npz`` (``canonical_xyz``/``traj``/optional ``opacity``), writes
``segmentation.npz`` ``{points, labels}`` — so ``seg_eval.default`` is backend-agnostic.

The separability diagnostic (proposal 06's go/no-go signal): when the impl config's
``gt_segmentation_path`` is set (per-run, like ``run_grid_seg.py`` sets ``recolored_ply``),
z-score AUROC vs GT is computed and written to ``separability.json`` as an extra artifact.
It is a config-provided path rather than a declared DAG input so that preset runs *without*
any GT (the normal inference case) still work — ``gt_segmentation`` stays a declared input
only of ``seg_eval.default``.
"""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.segment_rigid2 import segment_trajectories2 as _segment_trajectories2
from .base import ResourceRequest, Stage, StageContext
from .registry import register

#: Config keys forwarded verbatim to ``segment_trajectories2`` -> ``segment_by_rigidity2``
#: (everything else in the section is stage-level: opacity_thresh / gt path / preview).
_KWARG_KEYS = (
    "k", "min_size", "denoise", "drive_freq", "harmonics", "calibrate_sigma", "z_thresh",
    "threshold_mult", "partition", "min_clusters", "max_clusters", "n_clusters",
    "n_subsample", "propagate_q",
)


@register("segment.rigid2")
class SegmentRigid2Stage(Stage):
    inputs = ("trajectories",)
    outputs = ("segmentation",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=2.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        data = np.load(ctx.inputs["trajectories"].path)
        xyz, traj = data["canonical_xyz"], data["traj"]
        opacity = data["opacity"] if "opacity" in data.files else None

        opacity_thresh = float(ctx.config.get("opacity_thresh", 0.1))
        kwargs = {key: ctx.config[key] for key in _KWARG_KEYS if key in ctx.config}
        # pydantic dumps None for the unset float|None fields; the vendored function treats
        # None as "auto-detect", which is the intent — forward as-is.

        gt_labels = None
        gt_path = ctx.config.get("gt_segmentation_path") or ""
        if gt_path:
            gt = np.load(gt_path)
            # GT points are NN-aligned to the trajectory points (same convention as
            # seg_eval.default's GT propagation — nearest neighbour in canonical space).
            from scipy.spatial import cKDTree

            _, nn = cKDTree(gt["points"]).query(xyz, k=1)
            gt_labels = gt["labels"][nn]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            labels, info = _segment_trajectories2(
                xyz, traj, opacity=opacity, opacity_thresh=opacity_thresh,
                gt_labels_full=gt_labels, **kwargs,
            )
        for line in buf.getvalue().splitlines():
            ctx.logger.info(line)
        ctx.logger.info("segment.rigid2: %s", {k2: v2 for k2, v2 in info.items() if k2 != "eigenvalues"})

        out_path = ctx.run_dir / "segmentation.npz"
        # Optional ROI gating: points outside ROI get label -2 (static), preserving -1 floaters.
        roi_artifact = ctx.inputs.get("roi_mask")
        if roi_artifact is not None:
            roi_data = np.load(roi_artifact.path)
            roi_mask = roi_data["roi_mask"]
            labels[(~roi_mask) & (labels != -1)] = -2
            ctx.logger.info("roi gated: %d inside, %d outside (-2)", int(roi_mask.sum()), int((~roi_mask).sum()))

        np.savez(out_path, points=xyz.astype(np.float32), labels=labels)

        artifacts: dict[str, Artifact] = {
            "segmentation": Artifact(
                name="segmentation",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={k2: v2 for k2, v2 in info.items()
                          if isinstance(v2, (int, float, str, bool))},
            )
        }

        if "separability" in info:
            sep = {"denoised_z": info["separability"], "raw_score": info.get("separability_raw"),
                   "sigma_d": info.get("sigma_d"), "drive_freq_used": info.get("drive_freq_used")}
            sep_path = ctx.run_dir / "separability.json"
            sep_path.write_text(json.dumps(sep, indent=2), encoding="utf-8")
            ctx.logger.info("wrote %s (auroc=%s)", sep_path, sep["denoised_z"].get("auroc"))
            artifacts["separability"] = Artifact(
                name="separability",
                kind="json",
                path=str(sep_path),
                producing_stage=ctx.stage_name,
                metadata={"auroc": info["separability"].get("auroc")},
            )

        return artifacts
