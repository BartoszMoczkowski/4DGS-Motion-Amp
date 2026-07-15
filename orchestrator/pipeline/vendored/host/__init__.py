"""Vendored CPU-only stage logic (runs in the orchestrator's own ``host`` venv).

Each module here is a verbatim port of an already-verified function from the reference scripts
under ``omniverse_pipeline/``/``motion_seg/`` (see ``pipeline.vendored``'s module docstring for
why this exists instead of importing/subprocessing those scripts directly):

- :mod:`pipeline.vendored.host.convert` — ported from ``omniverse_pipeline/omni_to_4dgs.py``'s
  ``convert()`` (+ the geometry/COLMAP-writer helpers it calls).
- :mod:`pipeline.vendored.host.rigidity_graph` / :mod:`pipeline.vendored.host.segment_rigid` —
  ported from ``motion_seg/rigidity_graph.py`` / ``motion_seg/segment_rigid.py``'s
  ``segment_trajectories()``.
- :mod:`pipeline.vendored.host.metrics` / :mod:`pipeline.vendored.host.seg_eval` — ported from
  ``motion_seg/metrics.py`` / ``motion_seg/evaluate_segmentation.py``'s ``evaluate()`` /
  ``_write_colored_ply()``.

Ported verbatim (copy, not reimplement/redesign) — see each module's docstring for the exact
provenance. ``pipeline/stages/{convert,segment_rigid,seg_eval}.py`` import from here.
"""

from __future__ import annotations
