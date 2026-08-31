#!/usr/bin/env python3
"""
build_gt_motion_classes.py — Build ground-truth motion-equivalence classes for pump01.

Clusters the 107 CAD parts of pump01 by trajectory similarity into motion classes:
- Class 0: Exact-zero-motion group (static, frame_base, label 0).
- Classes 1..K: Motion-equivalence classes of moving parts based on SE(3) trajectory curve
  similarity (direction, frequency, phase) independent of amplitude scaling.

Emits:
- data/multipleview/pump01/gt_motion_classes.npy (aligned with 100,000 GT points in gt_segmentation.npz)
- gt_motion_classes.npy (at root for easy reference)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from scipy.cluster.hierarchy import fcluster, linkage

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_motion_classes(usd_path: str, gt_seg_path: str, cos_thresh: float = 0.985):
    """
    Load USD trajectories and gt_segmentation.npz, cluster 107 parts by motion similarity,
    and return (gt_motion_point_labels, part_to_class_map, summary_dict).
    """
    print(f"[build_gt] Opening USD stage: {usd_path}")
    stage = Usd.Stage.Open(usd_path)

    # Collect part prims under CONJUNTO_BOMBAS
    parts = []
    for p in stage.Traverse():
        if p.GetTypeName() == "Xform" and p.GetParent().GetName() == "CONJUNTO_BOMBAS":
            parts.append(p)
    
    part_prims = {p.GetName(): p for p in parts}
    ordered_names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, 107)]
    
    num_eval_frames = 60
    time_codes = np.linspace(0, stage.GetEndTimeCode(), num_eval_frames)

    transforms = np.zeros((107, num_eval_frames, 4, 4))
    for idx, name in enumerate(ordered_names):
        prim = part_prims[name]
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        for t_i, tc in enumerate(time_codes):
            if not ops:
                M = np.eye(4)
            else:
                mat_gf = ops[0].Get(Usd.TimeCode(tc))
                M = np.eye(4) if mat_gf is None else np.array(mat_gf, dtype=np.float64).T
            transforms[idx, t_i] = M

    # Trajectories relative to t=0
    disp_t = transforms[:, :, :3, 3] - transforms[:, :1, :3, 3] # (107, 60, 3)
    traj_vecs = disp_t.reshape(107, -1)                         # (107, 180)
    norms = np.linalg.norm(traj_vecs, axis=1)

    is_static = norms < 1e-5

    # Normalized trajectories for moving parts
    moving_indices = np.where(~is_static)[0]
    norm_trajs = np.zeros_like(traj_vecs)
    for idx in moving_indices:
        norm_trajs[idx] = traj_vecs[idx] / norms[idx]

    # Pairwise cosine similarity matrix
    cos_sim = norm_trajs @ norm_trajs.T # (107, 107)

    # Hierarchical clustering on moving parts with distance 1 - |cos_sim|
    dist_condensed = []
    for i in range(len(moving_indices)):
        for j in range(i+1, len(moving_indices)):
            idx_i = moving_indices[i]
            idx_j = moving_indices[j]
            d_ij = max(0.0, 1.0 - abs(cos_sim[idx_i, idx_j]))
            dist_condensed.append(d_ij)

    Z = linkage(dist_condensed, method="complete")
    cutoff = 1.0 - cos_thresh
    clusters = fcluster(Z, t=cutoff, criterion="distance")

    # Map part index (0..106) -> motion class index (0..K-1)
    part_to_class = np.zeros(107, dtype=np.int32)
    part_to_class[0] = 0 # Class 0 = Static (frame_base)

    for idx_in_moving, cluster_id in enumerate(clusters):
        part_idx = moving_indices[idx_in_moving]
        part_to_class[part_idx] = cluster_id # 1..K_moving

    # Calculate statistics
    class_counts = np.bincount(part_to_class)
    n_static = class_counts[0]
    n_singletons = int((class_counts[1:] == 1).sum())
    n_coupled_groups = int((class_counts[1:] > 1).sum())
    n_coupled_parts = int(class_counts[1:][class_counts[1:] > 1].sum())
    n_total_classes = len(class_counts)

    print(f"\n[build_gt] Motion Equivalence Classes Summary (cos_thresh={cos_thresh}):")
    print(f"  - Total CAD Parts: 107")
    print(f"  - Total Motion Classes: {n_total_classes}")
    print(f"  - (a) Static parts: {n_static} ({ordered_names[0]})")
    print(f"  - (b) Independently moving (singletons): {n_singletons}")
    print(f"  - (c) Kinematically coupled duplicates: {n_coupled_parts} parts across {n_coupled_groups} groups")

    # Load point-level GT labels and map to motion classes
    print(f"\n[build_gt] Loading point-level GT: {gt_seg_path}")
    gt_seg = np.load(gt_seg_path)
    gt_points = gt_seg["points"]
    gt_cad_labels = gt_seg["labels"]

    gt_motion_point_labels = part_to_class[gt_cad_labels]

    summary = {
        "n_cad_parts": 107,
        "n_motion_classes": n_total_classes,
        "n_static": int(n_static),
        "n_independently_moving": int(n_singletons),
        "n_kinematically_coupled_parts": int(n_coupled_parts),
        "n_kinematically_coupled_groups": int(n_coupled_groups),
        "cos_thresh": float(cos_thresh),
    }

    return gt_motion_point_labels, part_to_class, summary


def main():
    usd_path = REPO_ROOT / "omniverse-pipeline" / "data" / "scenes" / "grid" / "pump_A20mm_M2_animated.usd"
    gt_seg_path = REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_segmentation.npz"
    out_npy_path = REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_motion_classes.npy"
    root_npy_path = REPO_ROOT / "gt_motion_classes.npy"

    gt_motion_point_labels, part_to_class, summary = build_motion_classes(
        str(usd_path), str(gt_seg_path), cos_thresh=0.985
    )

    np.save(str(out_npy_path), gt_motion_point_labels)
    np.save(str(root_npy_path), gt_motion_point_labels)
    print(f"[build_gt] Saved gt_motion_classes.npy to:\n  - {out_npy_path}\n  - {root_npy_path}")

    # Also save metadata json
    meta_path = REPO_ROOT / "data" / "multipleview" / "pump01" / "gt_motion_classes_meta.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[build_gt] Saved metadata to: {meta_path}")

if __name__ == "__main__":
    main()
