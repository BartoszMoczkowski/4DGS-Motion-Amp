"""``seg_eval.default`` — calls the vendored, ported copy of
``motion-seg/motion_seg/evaluate_segmentation.py``'s ``evaluate()``/``_write_colored_ply()``.

Per the "copy the logic in, don't call the original script" rule (``planning/INSTRUCTIONS.md``,
2026-07-14, superseding "wrap, don't rewrite"), this calls
:mod:`pipeline.vendored.host.seg_eval` — an in-project, verbatim port of the already-verified
scoring logic (``evaluate()`` was graduated out of the reference script's ``main()`` into a
public function before being ported, per ``planning/INSTRUCTIONS.md``'s "refactor into
importable functions only when it clearly pays off"; the port itself is unchanged from that
graduated function) — rather than reaching into ``motion_seg`` from outside the package.

Consumes ``segmentation`` (produced by whichever ``segment.*`` impl a preset selects) and an
external ``gt_segmentation`` artifact.  T19 addition: optionally consumes ``roi_mask`` for
ARI-within-ROI scoring.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.seg_eval import _write_colored_ply, evaluate
from .base import ResourceRequest, Stage, StageContext
from .registry import register


@register("seg_eval.default")
class SegEvalStage(Stage):
    """Scores a predicted segmentation against GT (ARI + best-match IoU) and writes a JSON
    summary (``outputs["seg_eval_result"]``).

    ``recolored_ply`` (``SegEvalConfig.recolored_ply``) is optional, matching the CLI's own
    ``--recolored-ply`` optionality — when set, a colored-by-predicted-label PLY is written too
    and returned under the extra (undeclared-in-``outputs``) key ``"recolored_ply"``; nothing
    downstream in this DAG depends on it, so it isn't part of the hard ``outputs`` contract.
    ``comparison_png`` isn't wired up here: it needs ``motion_seg.visualize`` (matplotlib), an
    extra dependency this CPU vertical-slice test has no other reason to pull in — a real preset
    that wants it can still get it by running the original CLI directly.
    """

    inputs = ("segmentation", "gt_segmentation")
    outputs = ("seg_eval_result",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=0.5)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        pred = np.load(ctx.inputs["segmentation"].path)
        gt = np.load(ctx.inputs["gt_segmentation"].path)

        drop_floaters = bool(ctx.config.get("drop_floaters", False))
        top_n = int(ctx.config.get("top_n", 15))

        # Optional ROI mask for ARI-within-ROI scoring (T19)
        roi_mask = None
        roi_artifact = ctx.inputs.get("roi_mask")
        if roi_artifact is not None:
            roi_data = np.load(roi_artifact.path)
            roi_mask = roi_data["roi_mask"]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = evaluate(
                pred["points"], pred["labels"], gt["points"], gt["labels"],
                drop_floaters=drop_floaters,
                roi_mask=roi_mask,
            )
        for line in buf.getvalue().splitlines():
            ctx.logger.info(line)

        summary = {
            "ari": result["ari"],
            "mean_iou": result["mean_iou"],
            "n_gt": result["n_gt"],
            "n_pred": result["n_pred"],
            "n_pred_points": int(len(result["pred_labels"])),
            "top_matches": [
                {
                    "gt_label": int(gt_l),
                    "gt_size": int(gt_sz),
                    "pred_label": int(pred_l),
                    "pred_size": int(pred_sz),
                    "iou": float(iou),
                }
                for gt_l, pred_l, iou, gt_sz, pred_sz in result["matches"][:top_n]
            ],
        }
        if "ari_within_roi" in result:
            summary["ari_within_roi"] = result["ari_within_roi"]
            summary["n_roi_points"] = result["n_roi_points"]

        summary_path = ctx.run_dir / "seg_eval_result.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        ctx.logger.info("wrote %s (ari=%.4f, mean_iou=%.4f)", summary_path, summary["ari"], summary["mean_iou"])

        metadata = {"ari": summary["ari"], "mean_iou": summary["mean_iou"]}
        if "ari_within_roi" in summary:
            metadata["ari_within_roi"] = summary["ari_within_roi"]
            metadata["n_roi_points"] = summary["n_roi_points"]

        artifacts: dict[str, Artifact] = {
            "seg_eval_result": Artifact(
                name="seg_eval_result",
                kind="json",
                path=str(summary_path),
                producing_stage=ctx.stage_name,
                metadata=metadata,
            )
        }

        recolored_ply = ctx.config.get("recolored_ply")
        if recolored_ply:
            ply_path = Path(recolored_ply)
            if not ply_path.is_absolute():
                ply_path = ctx.run_dir / ply_path
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            _write_colored_ply(str(ply_path), result["pred_points"], result["pred_labels"])
            ctx.logger.info("wrote %s", ply_path)
            artifacts["recolored_ply"] = Artifact(
                name="recolored_ply",
                kind="ply",
                path=str(ply_path),
                producing_stage=ctx.stage_name,
            )

        return artifacts
