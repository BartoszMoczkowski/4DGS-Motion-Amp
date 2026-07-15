"""Vendored, verbatim copy of ``motion_seg/segment_rigid.py``'s ``segment_trajectories()``
(2026-07-14 copy-in rework of T07; see ``pipeline.vendored``'s module docstring). The function
body is byte-for-byte the reference script's; the only change is the import of
``segment_by_rigidity`` now pointing at the sibling vendored module
(:mod:`pipeline.vendored.host.rigidity_graph`) instead of ``motion_seg.rigidity_graph``. The
reference script's CLI/argparse/``_selftest`` are intentionally not ported — nothing in
``pipeline/stages/segment_rigid.py`` calls them; only ``segment_trajectories()`` is a
production dependency.

Original docstring:

    Baseline rigid motion-segmentation for a trained 4DGS scene ("Option B" in
    .claude_notes/NOTES_4dgs_motion_segmentation.md): local-rigidity graph + connected
    components (see rigidity_graph.py for the algorithm and why it's a reasonable fit for
    free-correspondence 4DGS trajectories). Pure numpy/scipy — does NOT need a GPU or the
    4DGS/torch stack.
"""

from __future__ import annotations

import numpy as np

from .rigidity_graph import segment_by_rigidity


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
