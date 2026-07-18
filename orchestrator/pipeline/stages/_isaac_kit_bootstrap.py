"""Orchestrator-owned bootstrap, run standalone inside the ``isaac`` container ahead of a vendored
CPU-only USD script (``split_mesh.py``/``add_motion.py``) that needs ``pxr`` but never launches its
own Kit runtime.

**Not a ported/copy-in file** — ``pipeline/vendored/isaac/*.py`` is a byte-for-byte copy of the
reference scripts (per ``planning/INSTRUCTIONS.md``'s "copy the logic in, don't rewrite" rule) and
stays untouched; this is new orchestration glue, found necessary on T11's first real-hardware run
(2026-07-16, see ``.claude_notes/NOTES_pipeline_orchestration.md``).

Why this exists: ``split_mesh.py``/``add_motion.py`` were ported assuming ``pxr`` (USD Python
bindings) is importable straight from Isaac Sim's bundled interpreter (``/isaac-sim/python.sh``)
with no Kit bootstrap — true on some earlier Isaac Sim releases, false on the
``nvcr.io/nvidia/isaac-sim:6.0.1`` image this project actually pulls: real execution raised a plain
``ModuleNotFoundError: No module named 'pxr'`` from a bare ``from pxr import Usd, UsdGeom``, even
though ``pxr`` unquestionably works in the *same* container once ``omni_capture.py`` launches its
own ``isaacsim.SimulationApp`` first (that's Kit's extension loader populating ``sys.path`` at
runtime — not a static, pre-computed ``PYTHONPATH``). This script does exactly that minimal
bootstrap — nothing else — then hands off to the real target script's own ``main()`` unchanged.

Usage (see ``pipeline.stages.isaac_common.run_isaac_script``, which builds this automatically for
``split_mesh``/``add_motion`` only — never for ``omni_capture``, which already does its own
``SimulationApp`` launch and must not be double-wrapped):

    /isaac-sim/python.sh _isaac_kit_bootstrap.py <real_script.py> [args...]

The target script sees exactly the ``argv`` it would have seen if invoked directly (``argv[0]`` is
rewritten to its own path before ``runpy.run_path`` executes it as ``__main__``), so its own
``argparse`` setup needs no changes.
"""
from __future__ import annotations

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _isaac_kit_bootstrap.py <script.py> [args...]", file=sys.stderr)
        raise SystemExit(2)

    target = sys.argv[1]
    forwarded_argv = sys.argv[1:]  # target's own argv[0] is its own path, same as a direct call

    from isaacsim import SimulationApp  # only importable via /isaac-sim/python.sh

    app = SimulationApp(launch_config={"headless": True})
    try:
        sys.argv = forwarded_argv
        runpy.run_path(target, run_name="__main__")
    finally:
        app.close()


if __name__ == "__main__":
    main()
