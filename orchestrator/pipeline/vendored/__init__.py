"""Copied-in (vendored) stage logic, one subpackage per execution environment.

Per ``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule
(2026-07-14, superseding "wrap, don't rewrite"): ``omniverse_pipeline/``, ``motion_seg/``, and the
repo-root scripts (``train.py``, ``render.py``, ``render_amp.py``, ``mbs_infer.py``, ...) are
throwaway/testing scripts, useful as a *reference* for already-verified logic, never as a live
runtime dependency. A ``pipeline.stages`` module must never ``sys.path``-hack its way into
importing them and must never shell out to them as a subprocess. Instead the verified
function(s) are ported here, verbatim, and stages import from here via a normal in-project
import.

Subpackages (see ``planning/ARCHITECTURE.md``'s "Vendored stage logic"):

- :mod:`pipeline.vendored.host` — CPU-only logic that runs in the orchestrator's own ``host``
  venv (``convert``, ``segment.rigid``, ``seg_eval``).
- :mod:`pipeline.vendored.cuda` (T09) — ``train``/``render``/``seg_extract``/``amp`` logic that
  runs inside the ``cuda`` container as a separate process (``segment.mbs`` is T10); T08's repo
  bind-mount is what makes this directory available inside the running container.
- ``pipeline.vendored.isaac`` (T11) — logic that runs inside the ``isaac`` container
  (prep.split/prep.motion/capture.isaac); same rule.

The **only** thing this project depends on outside itself is the container *runtime* (the
``isaac``/``cuda`` images) — never the script files that happen to live outside ``orchestrator/``.
"""

from __future__ import annotations
