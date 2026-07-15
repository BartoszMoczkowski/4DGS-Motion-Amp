"""Vendored CUDA-container stage logic (runs inside the ``cuda`` Docker image, T08).

Per ``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule and
``pipeline.vendored``'s module docstring: each module here is a verbatim, byte-for-byte port of
the corresponding repo-root reference script, including its own ``argparse``-based
``if __name__ == "__main__":`` entry point. Unlike ``pipeline.vendored.host`` (T07) — whose
modules are graduated *functions* imported and called in-process by a ``host``-environment
stage — these modules are never imported by the orchestrator's own (native-Windows, no-torch)
process. They only ever run as a **separate process inside the ``cuda`` container**
(``python pipeline/vendored/cuda/<name>.py <args>``, via ``ctx.containers.exec_in_container``),
because that's the only place their real dependencies (``torch``, ``arguments``, ``scene``,
``gaussian_renderer``, ``diff_gaussian_rasterization``, ``motion_amp``, ...) exist. The stage
classes in ``pipeline/stages/{train,render,seg_extract,amp}.py`` build each script's CLI
invocation from the resolved ``PipelineConfig`` (T02) — see ``pipeline/stages/cuda_common.py``
for the shared argument-building/bridge-file/container-exec plumbing.

- :mod:`pipeline.vendored.cuda.train` — verbatim port of ``train.py``.
- :mod:`pipeline.vendored.cuda.render` — verbatim port of ``render.py``.
- :mod:`pipeline.vendored.cuda.seg_extract` — verbatim port of
  ``motion_seg/extract_trajectories.py``.
- :mod:`pipeline.vendored.cuda.amp` — verbatim port of ``render_amp.py``.
- :mod:`pipeline.vendored.cuda.mbs_infer` — verbatim port of ``motion_seg/mbs_infer.py`` (T10,
  segmentation "Option A": MultiBodySync MotNet inference — a second impl behind the same
  ``segment`` role ``segment.rigid`` (T07) already occupies, see
  ``pipeline/stages/segment_mbs.py``).

None of these modules are imported by anything under ``orchestrator/`` at module scope (only
referenced as a container-side script path, as a plain string) — importing ``pipeline`` (or even
``pipeline.vendored``) must stay safe on a CPU-only host with no ``torch`` installed. This
package's own ``__init__.py`` intentionally does not import any of the five modules above, unlike
``pipeline.vendored.host``'s ``__init__.py``.
"""

from __future__ import annotations
