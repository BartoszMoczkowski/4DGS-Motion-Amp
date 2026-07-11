#!/usr/bin/env python3
"""Baseline rigid motion-segmentation for a trained 4DGS scene ("Option B" in
.claude_notes/NOTES_4dgs_motion_segmentation.md): local-rigidity graph + connected
components (see motion_seg/rigidity_graph.py for the algorithm and why it's a reasonable
fit for free-correspondence 4DGS trajectories).

Pure numpy/scipy — does NOT need a GPU or the 4DGS/torch stack. Run this after
`extract_trajectories.py` (which does need the trained model + GPU) has produced a
trajectories.npz.

Usage:
    python -m motion_seg.segment_rigid --trajectories output/multipleview/pump01/trajectories.npz \
        --out output/multipleview/pump01/segmentation.npz

Self-test (no data needed, verifies the algorithm on synthetic rigid bodies):
    python -m motion_seg.segment_rigid --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from motion_seg.rigidity_graph import segment_by_rigidity


def segment_trajectories(
    xyz: np.ndarray,
    traj: np.ndarray,
    opacity: np.ndarray | None = None,
    opacity_thresh: float = 0.1,
    k: int = 12,
    threshold_mult: float = 1.0,
    min_size: int = 15,
):
    """Opacity-filter floaters, run the rigidity-graph segmentation on the rest, then map
    labels back onto the full (unfiltered) point set. Floaters get label -1.

    Returns (labels (N,) int — -1 for dropped floaters, info dict).
    """
    n = len(xyz)
    if opacity is not None:
        keep = opacity > opacity_thresh
    else:
        keep = np.ones(n, dtype=bool)

    labels_full = np.full(n, -1, dtype=np.int64)
    sub_labels, info = segment_by_rigidity(
        xyz[keep], traj[keep], k=k, threshold_mult=threshold_mult, min_size=min_size
    )
    labels_full[keep] = sub_labels
    info["n_floaters_dropped"] = int((~keep).sum())
    return labels_full, info


def _selftest() -> int:
    """Synthetic scene: one big static base + several small rigid parts, each rotating
    about its own centroid at a distinct frequency/phase (mirrors the pump capture: a static
    frame + many independently-moving parts). Checks the recovered labels against the known
    ground truth via ARI, with no GPU / trained model required."""
    from motion_seg.metrics import adjusted_rand_index

    rng = np.random.RandomState(0)
    T = 60
    times = np.linspace(0.0, 1.0, T, endpoint=False)

    xyz_list, traj_list, gt_list = [], [], []
    lid = 0

    # Static base: a big, spatially spread-out blob that never moves.
    base_pts = rng.uniform(-1.0, 1.0, size=(2000, 3)) * np.array([3.0, 0.3, 3.0])
    xyz_list.append(base_pts)
    traj_list.append(np.repeat(base_pts[:, None, :], T, axis=1))
    gt_list.append(np.full(len(base_pts), lid))
    lid += 1

    # Several small rigid parts, spatially separated, each with its own rotation axis,
    # frequency (integer cycles, like the real pump's periodic motion) and small amplitude.
    n_parts = 6
    for p in range(n_parts):
        center = np.array([(p - n_parts / 2) * 1.2, 1.0, 0.0])
        pts = center + rng.uniform(-0.15, 0.15, size=(150, 3))
        centroid = pts.mean(axis=0)
        freq = 2 + p  # integer cycles over the clip, like add_motion.py
        amp_deg = 3.0 + p  # small rigid rotation amplitude
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        traj = np.empty((len(pts), T, 3))
        rel = pts - centroid
        for ti, t in enumerate(times):
            theta = np.deg2rad(amp_deg) * np.sin(2 * np.pi * freq * t)
            # Rodrigues' rotation formula about `axis` through `centroid`.
            cos, sin = np.cos(theta), np.sin(theta)
            rotated = (
                rel * cos
                + np.cross(axis, rel) * sin
                + axis * (rel @ axis)[:, None] * (1 - cos)
            )
            traj[:, ti, :] = centroid + rotated
        xyz_list.append(pts)
        traj_list.append(traj)
        gt_list.append(np.full(len(pts), lid))
        lid += 1

    xyz = np.concatenate(xyz_list).astype(np.float64)
    traj = np.concatenate(traj_list).astype(np.float64)
    gt = np.concatenate(gt_list)

    labels, info = segment_trajectories(xyz, traj, opacity=None, k=10, min_size=10)
    ari = adjusted_rand_index(gt, labels)
    print(f"[selftest] {info}")
    print(
        f"[selftest] recovered {info['n_components_final']} segments "
        f"(ground truth: {lid}); ARI={ari:.4f}"
    )
    ok = ari > 0.9
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--trajectories", help=".npz written by extract_trajectories.py")
    ap.add_argument("--out", help="output segmentation .npz path")
    ap.add_argument("--k", type=int, default=12, help="k-NN graph neighbors")
    ap.add_argument(
        "--min-size", type=int, default=15, help="merge components smaller than this"
    )
    ap.add_argument(
        "--threshold-mult",
        type=float,
        default=1.0,
        help="multiply the auto (Otsu) rigidity threshold by this; >1 = more "
        "permissive (fewer, bigger segments), <1 = stricter (more, smaller)",
    )
    ap.add_argument(
        "--opacity-thresh",
        type=float,
        default=0.1,
        help="drop Gaussians with opacity <= this before segmenting (floaters)",
    )
    ap.add_argument(
        "--preview-png",
        default=None,
        help="write a 3-view (top/front/side) colored scatter PNG here "
        "(default: <out>.png next to --out; pass '' to skip)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    if not args.trajectories or not args.out:
        ap.error("--trajectories and --out are required (or use --selftest)")

    data = np.load(args.trajectories)
    xyz, traj = data["canonical_xyz"], data["traj"]
    opacity = data["opacity"] if "opacity" in data else None

    labels, info = segment_trajectories(
        xyz,
        traj,
        opacity=opacity,
        opacity_thresh=args.opacity_thresh,
        k=args.k,
        threshold_mult=args.threshold_mult,
        min_size=args.min_size,
    )
    np.savez(args.out, points=xyz.astype(np.float32), labels=labels)
    print(f"[ok] {info}")
    print(
        f"[ok] {info['n_components_final']} segments, "
        f"{info['n_floaters_dropped']} floaters dropped -> {args.out}"
    )

    preview_png = args.preview_png
    if preview_png is None:
        base, _ext = os.path.splitext(args.out)
        preview_png = (base or args.out) + "_preview.png"
    if preview_png:
        from motion_seg.visualize import render_segmentation_png

        render_segmentation_png(xyz, labels, preview_png)
        print(f"[ok] wrote {preview_png}")


if __name__ == "__main__":
    main()
