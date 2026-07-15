"""``amp.default`` — runs the vendored, ported copy of ``render_amp.py``'s CLI
(``pipeline/vendored/cuda/amp.py``) inside the ``cuda`` container (T08/T09).

See ``pipeline.stages.train``'s and ``pipeline.stages.cuda_common``'s module docstrings for the
general CLI-invocation-in-a-container design this follows.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import Artifact
from ..config.models import AMP_CHANNELS
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import bool_flag, flag, run_cuda_script, write_stage_bridge
from .registry import register


class AmpFactorNotIntegerError(ValueError):
    """A channel's ``factor`` isn't a whole number.

    ``pipeline/vendored/cuda/amp.py``'s ``--amp_factors`` is declared ``type=int`` (a verbatim,
    pre-existing quirk of ``render_amp.py`` — see that module's docstring); passing a non-integer
    string would fail argparse *inside* the container with a much less legible error. Caught here
    instead, before the exec call.
    """


def _int_factor(channel: str, value: float) -> int:
    if float(value).is_integer():
        return int(value)
    raise AmpFactorNotIntegerError(
        f"amp.channels[{channel!r}].factor = {value!r} is not a whole number, but "
        "pipeline/vendored/cuda/amp.py's --amp_factors is int-typed (a pre-existing "
        "render_amp.py quirk, kept as-is per the 'copy the logic in' rule) — use an integer "
        "factor (e.g. 2 or -1 to disable this channel)"
    )


@register("amp.default")
class AmpStage(Stage):
    """Per-channel motion amplification + video render (``render_amp.py``'s ``render_sets`` /
    ``render_set_amp``).

    ``inputs["model"]`` is ``train.default``'s trained model directory. Channel order is fixed
    (``pipeline.config.models.AMP_CHANNELS``) and matches ``render_amp.py``'s positional
    ``amp_factors``/``freq_cutoffs`` lists; a channel missing from ``AmpConfig.channels`` (it
    shouldn't be, since the pydantic default fills all eight) falls back to "don't amplify"
    (``factor=-1``). The compiled video is written by the reference script to a fixed, iteration-
    independent location — ``<model_path>/video/<video_path>`` — so (unlike ``render.default``)
    this stage can compute the exact output path itself rather than needing to introspect the
    container afterward.
    """

    inputs = ("model",)
    outputs = ("amp_video",)
    environment = "cuda"
    resources = ResourceRequest(needs_gpu=True, vram_gb=10.0, ram_gb=10.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_container = ctx.paths.to_container(model.path, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        cfg = ctx.config  # AmpConfig's own fields; `configs` ignored, same as the other T09
        # stages — this stage always generates its own bridge file from the resolved config.
        channels = cfg.get("channels", {})
        factors: list[int] = []
        freq_low: list[float] = []
        freq_high: list[float] = []
        for name in AMP_CHANNELS:
            channel = channels.get(name, {"factor": -1.0, "freq_low": 0.0, "freq_high": 1.0})
            factors.append(_int_factor(name, channel.get("factor", -1.0)))
            freq_low.append(float(channel.get("freq_low", 0.0)))
            freq_high.append(float(channel.get("freq_high", 1.0)))

        video_path = cfg.get("video_path", "render.mp4")

        args = [
            *flag("model_path", str(model_container)),
            *flag("iteration", cfg.get("iteration", -1)),
            *flag("configs", str(bridge_container)),
            *bool_flag("skip_train", cfg.get("skip_train", False)),
            *bool_flag("skip_test", cfg.get("skip_test", False)),
            *bool_flag("skip_video", cfg.get("skip_video", False)),
            *bool_flag("quiet", cfg.get("quiet", False)),
            "--amp_factors", *[str(f) for f in factors],
            "--freq_low", *[str(f) for f in freq_low],
            "--freq_high", *[str(f) for f in freq_high],
            *flag("video_path", video_path),
            *flag("video_fps", cfg.get("video_fps", 20)),
            *flag("method", cfg.get("method", "eulerian")),
            *bool_flag("low_vram", cfg.get("low_vram_mode", False)),
            *bool_flag("frozen_cam", cfg.get("frozen_cam", False)),
        ]

        run_cuda_script(ctx, "amp", args, log_name="amp")

        # render_set_amp writes the compiled video at <model_path>/video/<video_path> (the
        # hardcoded split name "video", not the "train"/"test"/"video" render.default deals with)
        # — deterministic, so no need to introspect the container's filesystem afterward.
        out_host = Path(model.path) / "video" / video_path

        return {
            "amp_video": Artifact(
                name="amp_video",
                kind="video",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={
                    "method": cfg.get("method", "eulerian"),
                    "factors": dict(zip(AMP_CHANNELS, factors)),
                },
            )
        }
