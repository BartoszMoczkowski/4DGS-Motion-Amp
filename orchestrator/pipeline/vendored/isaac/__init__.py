"""Vendored Isaac-container stage logic (runs inside the ``isaac`` Docker image, T11).

Per ``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule and
``pipeline.vendored``'s module docstring: each module here is a verbatim, byte-for-byte port of
the corresponding repo-root reference script under ``omniverse-pipeline/omniverse_pipeline/``, including its own
``argparse``-based ``if __name__ == "__main__":`` entry point (where it has one). Like
``pipeline.vendored.cuda`` (T09/T10) and unlike ``pipeline.vendored.host`` (T07), these modules
are never imported by the orchestrator's own host process — they only ever run as a **separate
process inside the ``isaac`` container**, via Isaac Sim's own bundled interpreter
(``/isaac-sim/python.sh``, not the container's plain ``python`` — see
``pipeline/stages/isaac_common.py``), because that's the only place ``pxr``/``omni.*``
(Isaac Sim's USD/Omniverse Kit bindings) are actually importable.

- :mod:`pipeline.vendored.isaac.rig` — verbatim port of ``omniverse-pipeline/omniverse_pipeline/rig.py`` (pure-math
  camera-rig generation, numpy-only — no Isaac Sim dependency itself, but vendored alongside
  ``omni_capture.py`` since that's the one same-directory import it makes).
- :mod:`pipeline.vendored.isaac.split_mesh` — verbatim port of
  ``omniverse-pipeline/omniverse_pipeline/split_mesh.py`` (stage ``prep_split.default``).
- :mod:`pipeline.vendored.isaac.add_motion` — verbatim port of
  ``omniverse-pipeline/omniverse_pipeline/add_motion.py`` (stage ``prep_motion.default``).
- :mod:`pipeline.vendored.isaac.omni_capture` — verbatim port of
  ``omniverse-pipeline/omniverse_pipeline/omni_capture.py`` (stage ``capture.isaac``).

Env decision (``planning/ARCHITECTURE.md``'s "isaac/host*" footnote, resolved by this task):
``split_mesh``/``add_motion`` are plain-Python USD/trimesh CPU work with no actual Isaac Sim
runtime dependency (no ``SimulationApp``, no GPU) — they could in principle run in a separate
small CPU image. This project has no such image yet, and adding one is out of this task's
"contained task" scope (a new ``Env`` literal value, a new Dockerfile, new container config) for
what would only save a bit of container-startup weight. They run in the existing ``isaac``
container instead (it already has ``pxr``/usd-core via ``/isaac-sim/python.sh``); ``trimesh``
needs a one-time manual ``pip install`` there (not preinstalled) — see
``planning/WINDOWS_SETUP.md``'s Isaac stages setup step. Revisit if/when a lighter CPU image
becomes worth building for its own sake.

None of these modules are imported by anything under ``orchestrator/`` at module scope (only
referenced as a container-side script path, as a plain string) — importing ``pipeline`` (or even
``pipeline.vendored``) must stay safe on a CPU-only host with no Isaac Sim/``pxr`` installed. This
package's own ``__init__.py`` intentionally does not import any of the three modules above, same
as ``pipeline.vendored.cuda``'s ``__init__.py``.
"""

from __future__ import annotations
