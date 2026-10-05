"""``segment.multicut`` — Multi-channel affinity graph fusion with lifted multicut solver.

Fuses spatial adjacency, appearance/identity similarity, trajectory coherence/phasors,
and optional 2D mask agreement into a single affinity graph, partitioned with
Correlation Clustering / Lifted Multicut (GAEC + KL / RAMA solver).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path

import numpy as np

from ..artifacts import Artifact
from ..vendored.host.multicut import (
    DEFAULT_LOGIT_WEIGHTS,
    build_multichannel_graph,
    compute_edge_logits,
    fit_edge_weights,
    postprocess_min_cluster_size,
    solve_multicut,
)
from .base import ResourceRequest, Stage, StageContext
from .registry import register

logger = logging.getLogger(__name__)


def _read_ply_rgb(ply_path: Path) -> np.ndarray | None:
    """Extract RGB colors from point_cloud.ply SH DC components."""
    if not ply_path.is_file():
        return None
    try:
        with open(ply_path, "rb") as f:
            header = b""
            while b"end_header\n" not in header:
                header += f.readline()
            header_len = f.tell()

        props = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ]
        for i in range(45):
            props.append((f"f_rest_{i}", "<f4"))
        props.extend([
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ])
        dt = np.dtype(props)
        with open(ply_path, "rb") as f:
            f.seek(header_len)
            data = np.fromfile(f, dtype=dt)

        f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        rgb = np.clip(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0)
        return rgb
    except Exception as exc:
        logger.warning("Failed to read PLY RGB from %s: %s", ply_path, exc)
        return None


@register("segment.multicut")
class SegmentMulticutStage(Stage):
    inputs = ("trajectories",)
    outputs = ("segmentation",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=2.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        data = np.load(ctx.inputs["trajectories"].path)
        xyz, traj = data["canonical_xyz"], data["traj"]
        opacity = data["opacity"] if "opacity" in data.files else None

        k = int(ctx.config.get("k", 12))
        radius_max = float(ctx.config.get("radius_max", 0.05))
        drive_freq = ctx.config.get("drive_freq")
        f0 = float(drive_freq) if drive_freq is not None else 10.0
        use_appearance = bool(ctx.config.get("use_appearance", True))
        use_motion = bool(ctx.config.get("use_motion", True))
        lifted_edges = bool(ctx.config.get("lifted_edges", False))
        lifted_k = int(ctx.config.get("lifted_k", 4))
        lifted_min_dist = float(ctx.config.get("lifted_min_dist", 0.05))
        min_size = int(ctx.config.get("min_size", 15))
        opacity_thresh = float(ctx.config.get("opacity_thresh", 0.1))
        refine_kl = bool(ctx.config.get("refine_kl", True))
        calibrate_on_gt = bool(ctx.config.get("calibrate_on_gt", False))

        # Find point_cloud.ply in train_out
        rgb = None
        if use_appearance:
            ply_files = sorted((ctx.run_dir / "train_out" / "point_cloud").glob("*/*.ply"))
            if ply_files:
                rgb = _read_ply_rgb(ply_files[-1])
                if rgb is not None and len(rgb) != len(xyz):
                    logger.warning("RGB length %d != xyz length %d, discarding", len(rgb), len(xyz))
                    rgb = None

        traj_in = traj if use_motion else None

        # Build multi-channel affinity graph
        edges, feats, graph_info = build_multichannel_graph(
            xyz,
            traj=traj_in,
            rgb=rgb,
            k=k,
            radius_max=radius_max,
            f0=f0,
            include_lifted=lifted_edges,
            lifted_k=lifted_k,
            lifted_min_dist=lifted_min_dist,
        )
        ctx.logger.info("Graph built: %s", graph_info)

        # Optional GT calibration / diagnostic
        weights_dict = None
        gt_path = ctx.config.get("gt_segmentation_path") or ""
        gt_labels = None
        if gt_path and Path(gt_path).is_file():
            gt = np.load(gt_path)
            from scipy.spatial import cKDTree

            _, nn = cKDTree(gt["points"]).query(xyz, k=1)
            gt_labels = gt["labels"][nn]
            if calibrate_on_gt:
                weights_dict = fit_edge_weights(feats, edges, gt_labels)

        # Compute logit edge affinities
        weights = compute_edge_logits(feats, weights=weights_dict)
        ctx.logger.info(
            "Edge logits: min=%.2f max=%.2f mean=%.2f pos_ratio=%.2f%%",
            float(weights.min()) if len(weights) else 0.0,
            float(weights.max()) if len(weights) else 0.0,
            float(weights.mean()) if len(weights) else 0.0,
            float((weights > 0).mean() * 100) if len(weights) else 0.0,
        )

        # Solve multicut / correlation clustering
        labels, solve_meta = solve_multicut(
            len(xyz), edges, weights, use_rama_if_available=True, refine_kl=refine_kl
        )
        ctx.logger.info("Multicut solver: %s", solve_meta)

        # Postprocess small clusters
        if min_size > 1:
            labels = postprocess_min_cluster_size(xyz, labels, min_size=min_size)

        # Filter low opacity as floater (-1)
        if opacity is not None and opacity_thresh > 0:
            low_op = opacity < opacity_thresh
            labels[low_op] = -1

        # Optional ROI gating: points outside ROI get label -2 (static)
        roi_artifact = ctx.inputs.get("roi_mask")
        if roi_artifact is not None:
            roi_data = np.load(roi_artifact.path)
            roi_mask = roi_data["roi_mask"]
            labels[(~roi_mask) & (labels != -1)] = -2
            ctx.logger.info("ROI gated: %d inside, %d outside (-2)", int(roi_mask.sum()), int((~roi_mask).sum()))

        out_path = ctx.run_dir / "segmentation.npz"
        np.savez(out_path, points=xyz.astype(np.float32), labels=labels)

        info_dict = {
            "k_pred": int(len(np.unique(labels[labels >= 0]))),
            "solver": solve_meta.get("solver"),
            "wall_time_s": solve_meta.get("wall_time_s"),
            "cut_energy": solve_meta.get("cut_energy"),
            "uncut_energy": solve_meta.get("uncut_energy"),
            "n_edges": graph_info.get("n_edges"),
        }

        artifacts: dict[str, Artifact] = {
            "segmentation": Artifact(
                name="segmentation",
                kind="npz",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata=info_dict,
            )
        }

        # Write separability / multicut report if GT was provided
        if gt_labels is not None and len(edges) > 0:
            u = edges[:, 0]
            v = edges[:, 1]
            same_mask = gt_labels[u] == gt_labels[v]
            if same_mask.any() and (~same_mask).any():
                same_w = weights[same_mask]
                diff_w = weights[~same_mask]
                # AUROC: probability same_w > diff_w
                n_eval = min(5000, len(same_w), len(diff_w))
                rng = np.random.default_rng(42)
                s_sub = rng.choice(same_w, size=n_eval, replace=False)
                d_sub = rng.choice(diff_w, size=n_eval, replace=False)
                auroc = float(np.mean(s_sub[:, None] > d_sub[None, :]) + 0.5 * np.mean(s_sub[:, None] == d_sub[None, :]))
            else:
                auroc = 0.5

            report = {
                "multicut": solve_meta,
                "graph": graph_info,
                "edge_weights": {k: float(v) for k, v in (weights_dict or DEFAULT_LOGIT_WEIGHTS).items()},
                "edge_auroc": auroc,
            }
            sep_path = ctx.run_dir / "separability.json"
            sep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            ctx.logger.info("Wrote %s (edge_auroc=%.4f)", sep_path, auroc)
            artifacts["separability"] = Artifact(
                name="separability",
                kind="json",
                path=str(sep_path),
                producing_stage=ctx.stage_name,
                metadata={"edge_auroc": auroc},
            )

        return artifacts
