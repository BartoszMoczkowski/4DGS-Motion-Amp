"""``convert.default`` — calls the vendored, ported copy of
``omniverse-pipeline/omniverse_pipeline/omni_to_4dgs.py``'s ``convert()``.

Per the "copy the logic in, don't call the original script" rule (``planning/INSTRUCTIONS.md``,
2026-07-14, superseding "wrap, don't rewrite"), this calls
:func:`pipeline.vendored.host.convert.convert` — an in-project, verbatim port of the
already-verified function — in-process, no subprocess, no CLI-arg building, and no import-path
reach outside ``orchestrator/``. ``convert()``'s real downstream consumer is ``train`` (GPU, T09,
out of scope here); this stage just needs to prove the port end-to-end on a synthetic capture
fixture (see ``tests/test_stages_cpu.py``).
"""

from __future__ import annotations

import contextlib
import io

from ..artifacts import Artifact
from ..vendored.host.convert import convert as _convert
from .base import ResourceRequest, Stage, StageContext
from .registry import register


@register("convert.default")
class ConvertStage(Stage):
    """Converts an Omniverse capture directory into a 4DGS ``multipleview`` scene.

    ``inputs["capture"]`` must be a ``dataset`` artifact pointing at a capture directory
    (``cameras_gt.json`` + ``camNN/rgb_*.png``, see ``omni_capture.py``'s output convention) —
    nothing in this repo's Phase-0 scope produces one yet (that's ``capture.isaac``, T11), so it's
    always an external input, pre-seeded into the run's manifest before ``run_dag`` runs.
    """

    inputs = ("capture",)
    outputs = ("scene",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=1.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        capture_dir = ctx.inputs["capture"].path
        name = ctx.config.get("name", "omni_scene")
        near = ctx.config.get("near", 0.1)
        far = ctx.config.get("far", 1000.0)
        target_radius = ctx.config.get("target_radius", 4.0)

        out_dir = ctx.run_dir / "convert_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            scene_dir = _convert(capture_dir, str(out_dir), name, near, far, target_radius)
        for line in buf.getvalue().splitlines():
            ctx.logger.info(line)

        return {
            "scene": Artifact(
                name="scene",
                kind="dataset",
                path=str(scene_dir),
                producing_stage=ctx.stage_name,
                metadata={"name": name, "near": near, "far": far, "target_radius": target_radius},
            )
        }
