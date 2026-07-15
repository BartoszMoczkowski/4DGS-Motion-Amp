"""``EchoStage`` — a dummy stage with no real work, used to exercise the registry + ``run(ctx)``
contract in tests (T04 acceptance: "a dummy stage runs through run(ctx) with a fake ctx and
produces a valid artifact").

Registered under role ``test``, impl ``echo`` — deliberately outside any real pipeline role
(``capture``/``convert``/``train``/.../``segment``/``amp``) so it can never collide with or be
mistaken for a real stage in a preset.
"""

from __future__ import annotations

import json

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .registry import register


@register("test.echo")
class EchoStage(Stage):
    """Writes its input ``message`` (default ``"hello"``) to ``echo.json`` in the run dir."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("echo",)
    environment = "host"
    resources = ResourceRequest(needs_gpu=False, ram_gb=0.1)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        message = ctx.config.get("message", "hello")
        out_path = ctx.run_dir / "echo.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"message": message}))
        ctx.logger.info("EchoStage wrote %s", out_path)

        return {
            "echo": Artifact(
                name="echo",
                kind="json",
                path=str(out_path),
                producing_stage=ctx.stage_name,
                metadata={"message": message},
            )
        }
