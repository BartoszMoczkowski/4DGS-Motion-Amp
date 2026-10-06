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


def _build_rotating_part(center, n_pts, half, freq, amp_deg, phase, trans_amp,
                         trans_dir, axis, times, rng, noise_sigma=0.0):
    """One rigid part: `n_pts` points in a cube of half-size `half` around `center`,
    rotating rigidly about its own centroid at `freq` cycles over the clip with amplitude
    `amp_deg` degrees and phase `phase`, plus a sinusoidal translation of amplitude
    `trans_amp` along `trans_dir` at the same frequency/phase (sinusoidal periodic motion,
    like add_motion.py). The translation term matters: with rotation alone, boundary
    points of adjacent parts with similar rotation move nearly identically and their
    cross-part edges are NOT separable by rigidity score (verified 2026-10-05).
    `noise_sigma` adds iid Gaussian jitter per point per timestep — models 4DGS
    reconstruction jitter, which breaks exact rigidity and is the real-data regime the
    rigidity threshold must be robust to."""
    pts = center + rng.uniform(-half, half, size=(n_pts, 3))
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    traj = np.empty((len(pts), len(times), 3))
    for ti, t in enumerate(times):
        theta = np.deg2rad(amp_deg) * np.sin(2 * np.pi * freq * t + phase)
        # Rodrigues' rotation formula about `axis` through `centroid`.
        cos, sin = np.cos(theta), np.sin(theta)
        rotated = (
            rel * cos
            + np.cross(axis, rel) * sin
            + axis * (rel @ axis)[:, None] * (1 - cos)
        )
        shift = trans_amp * np.sin(2 * np.pi * freq * t + phase) * trans_dir
        traj[:, ti, :] = centroid + rotated + shift
    if noise_sigma > 0.0:
        traj = traj + rng.normal(scale=noise_sigma, size=traj.shape)
    return pts, traj


def _selftest(threshold_mult: float = 1.0) -> int:
    """Two synthetic fixtures mirroring the pump capture (static base + independently
    rotating rigid parts), checked against known ground truth via ARI — no GPU / trained
    model required.

    CASE A (sanity): the pre-2026-10-05 fixture — noiseless, spatially disjoint parts.
    The kNN graph is already disconnected, so this validates kNN construction, connected
    components and label mapping, but NOT edge cutting. Bar: ARI > 0.99.

    CASE B (primary, added 2026-10-05 for review item M1): parts are placed ADJACENT
    (cube faces touching) so the kNN graph is connected across part boundaries and
    cross-part edges MUST be cut by the rigidity score for the parts to separate; and
    every trajectory carries iid Gaussian jitter (sigma = 0.002, ~1.3% of the part
    half-size, comparable to observed 4DGS reconstruction jitter), so same-part edges
    are not exactly zero either. Parts move with distinct frequencies, phases, rotation
    axes and translation directions, so their relative motion is separable from the
    jitter floor. The test asserts (i) the full kNN graph bridges parts (otherwise the
    fixture is vacuous), (ii) edges were actually cut (n_kept_edges < n_edges), and
    (iii) ARI >= 0.99. Bar rationale: with these parameters the fixture scores ARI 1.0
    across seeds 0-4 (verified 2026-10-05); 0.99 leaves headroom for platform float
    jitter while remaining unreachable without correct edge cutting (cutting nothing
    yields 2 components / ARI ~0.84; cutting at a noise-split threshold yields ~0.94).
    """
    from motion_seg.metrics import adjusted_rand_index
    from motion_seg.rigidity_graph import build_knn_edges
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rng = np.random.RandomState(0)
    T = 60
    times = np.linspace(0.0, 1.0, T, endpoint=False)

    def build_scene(spacing, noise_sigma, n_parts=6, trans_amp=0.0):
        xyz_list, traj_list, gt_list = [], [], []
        lid = 0
        # Static base: a big, spatially spread-out slab that never moves.
        base_pts = rng.uniform(-1.0, 1.0, size=(2000, 3)) * np.array([3.0, 0.3, 3.0])
        xyz_list.append(base_pts)
        traj_list.append(np.repeat(base_pts[:, None, :], T, axis=1))
        if noise_sigma > 0.0:
            traj_list[-1] = traj_list[-1] + rng.normal(
                scale=noise_sigma, size=traj_list[-1].shape)
        gt_list.append(np.full(len(base_pts), lid))
        lid += 1
        half = 0.15
        for p in range(n_parts):
            center = np.array([(p - (n_parts - 1) / 2) * spacing, 1.0, 0.0])
            freq = 2 + p          # integer cycles over the clip, like add_motion.py
            amp_deg = 8.0 + 2 * p  # small rigid rotation amplitude
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            trans_dir = rng.normal(size=3)
            trans_dir /= np.linalg.norm(trans_dir)
            pts, traj = _build_rotating_part(
                center, 150, half, freq, amp_deg, phase=p * 2.1,
                trans_amp=trans_amp, trans_dir=trans_dir, axis=axis,
                times=times, rng=rng, noise_sigma=noise_sigma)
            xyz_list.append(pts)
            traj_list.append(traj)
            gt_list.append(np.full(len(pts), lid))
            lid += 1
        return (np.concatenate(xyz_list).astype(np.float64),
                np.concatenate(traj_list).astype(np.float64),
                np.concatenate(gt_list), lid)

    # --- Case A: noiseless, disjoint (legacy sanity check) -------------------
    xyz, traj, gt, n_gt = build_scene(spacing=1.2, noise_sigma=0.0)
    labels, info = segment_trajectories(xyz, traj, opacity=None, k=10, min_size=10,
                                        threshold_mult=threshold_mult)
    ari_a = adjusted_rand_index(gt, labels)
    print(f"[selftest][A: noiseless/disjoint] {info}")
    print(f"[selftest][A] recovered {info['n_components_final']} segments "
          f"(ground truth: {n_gt}); ARI={ari_a:.4f} (bar: > 0.99)")
    ok_a = ari_a > 0.99

    # --- Case B: noisy, adjacent — REQUIRES edge cutting ----------------------
    xyz, traj, gt, n_gt = build_scene(spacing=0.30, noise_sigma=0.002, trans_amp=0.08)
    # Fixture-vacuity check: the full kNN graph must bridge at least some part
    # boundaries, otherwise "segmentation" could succeed on connectivity alone.
    edges = build_knn_edges(xyz, k=10)
    n = len(xyz)
    full_graph = coo_matrix(
        (np.ones(2 * len(edges), dtype=np.int8),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n, n))
    n_cc_full, cc_full = connected_components(full_graph, directed=False)
    bridged = n_cc_full < n_gt
    labels, info = segment_trajectories(xyz, traj, opacity=None, k=10, min_size=10,
                                        threshold_mult=threshold_mult)
    ari_b = adjusted_rand_index(gt, labels)
    cut_happened = info["n_kept_edges"] < info["n_edges"]
    print(f"[selftest][B: noisy/adjacent] {info}")
    print(f"[selftest][B] full kNN graph components: {n_cc_full} (parts bridged: "
          f"{bridged}); edges cut: {info['n_edges'] - info['n_kept_edges']} of "
          f"{info['n_edges']}")
    print(f"[selftest][B] recovered {info['n_components_final']} segments "
          f"(ground truth: {n_gt}); ARI={ari_b:.4f} (bar: >= 0.99)")
    ok_b = bridged and cut_happened and ari_b >= 0.99
    if not bridged:
        print("[selftest][B] FIXTURE VACUOUS: kNN graph does not bridge parts — "
              "edge cutting is not being exercised!")
    if not cut_happened:
        print("[selftest][B] no edges were cut — thresholding is not being exercised!")

    ok = ok_a and ok_b
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} "
          f"(A={'PASS' if ok_a else 'FAIL'} ARI={ari_a:.4f}, "
          f"B={'PASS' if ok_b else 'FAIL'} ARI={ari_b:.4f})")
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
        # --threshold-mult also applies in selftest mode, so the edge-cutting requirement
        # can be deliberately broken (e.g. --selftest --threshold-mult 1e9) to prove the
        # test is able to FAIL.
        sys.exit(_selftest(threshold_mult=args.threshold_mult))

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
