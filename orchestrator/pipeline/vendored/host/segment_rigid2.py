"""Opacity-filter/floater wrapper around :func:`rigidity_graph2.segment_by_rigidity2`, mirroring
:mod:`pipeline.vendored.host.segment_rigid`'s structure (T18 — ``segment.rigid2``) so the two
impls are drop-in interchangeable behind the ``segment`` role: same inputs, same label
conventions (``-1`` = dropped floater), same ``segmentation.npz`` shape downstream.
"""

from __future__ import annotations

import numpy as np

from .rigidity_graph2 import segment_by_rigidity2


def segment_trajectories2(
    xyz: np.ndarray,
    traj: np.ndarray,
    opacity: np.ndarray | None = None,
    opacity_thresh: float = 0.1,
    gt_labels_full: np.ndarray | None = None,
    **kwargs,
):
    """Opacity-filter floaters, run the T18 segmentation on the rest, map labels back onto the
    full point set. ``gt_labels_full`` (aligned with the *unfiltered* xyz) is sliced alongside
    and only feeds the separability diagnostic. Returns (labels (N,), info dict)."""
    n = len(xyz)
    if opacity is not None:
        keep = opacity > opacity_thresh
    else:
        keep = np.ones(n, dtype=bool)

    labels_full = np.full(n, -1, dtype=np.int64)
    sub_labels, info = segment_by_rigidity2(
        xyz[keep],
        traj[keep],
        gt_labels=gt_labels_full[keep] if gt_labels_full is not None else None,
        **kwargs,
    )
    labels_full[keep] = sub_labels
    info["n_floaters_dropped"] = int((~keep).sum())
    return labels_full, info
