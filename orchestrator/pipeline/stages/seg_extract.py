"""``seg_extract.default`` — runs the vendored, ported copy of
``motion-seg/motion_seg/extract_trajectories.py``'s CLI (``pipeline/vendored/cuda/seg_extract.py``) inside
the ``cuda`` container (T08/T09).

See ``pipeline.stages.train``'s and ``pipeline.stages.cuda_common``'s module docstrings for the
general CLI-invocation-in-a-container design this follows. Needs the same GPU environment as
``train``/``render`` (loads the trained deformation network) — see the reference script's own
module docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .cuda_common import flag, run_cuda_script, write_stage_bridge
from .registry import register


def _capture_frame_count(ctx: StageContext) -> Optional[int]:
    """Best-effort frame count of the capture this run's model was trained on, from whatever
    upstream artifacts happen to be in the manifest (``ctx.inputs`` carries *all* run artifacts,
    not just this stage's declared inputs). ``None`` when nothing usable is present — the guard
    in ``run()`` then simply doesn't fire, same as before this check existed.

    Prefers the converted ``scene`` (``camNN/frame_*.jpg`` — exactly what training sampled its
    time base over); falls back to the raw ``capture`` (``camNN/rgb_*.png``, possibly under an
    ``rgb/`` subfolder — the same convention ``omni_to_4dgs.convert`` itself handles).
    """

    scene = ctx.inputs.get("scene")
    if scene is not None:
        cams = sorted(p for p in Path(scene.path).glob("cam*") if p.is_dir())
        if cams:
            return len(list(cams[0].glob("frame_*.jpg")))
    capture = ctx.inputs.get("capture")
    if capture is not None:
        cams = sorted(p for p in Path(capture.path).glob("cam*") if p.is_dir())
        if cams:
            cam = cams[0]
            rgb = cam / "rgb" if (cam / "rgb").is_dir() else cam
            return len(list(rgb.glob("rgb_*.png")))
    return None


@register("seg_extract.default")
class SegExtractStage(Stage):
    """Samples the trained deformation field at ``n_times`` evenly spaced timesteps into a dense
    per-Gaussian trajectory tensor (``trajectories.npz``) — the data adapter feeding
    ``segment.rigid``/``segment.mbs`` (T07/T10).

    ``inputs["model"]`` is ``train.default``'s trained model directory. The output path is always
    computed and passed explicitly via ``--out`` (rather than relying on the reference script's
    own ``<model_path>/trajectories.npz`` default, which it would compute from the *container*
    path) so this stage knows exactly where to find the result afterward without re-deriving the
    container's path-resolution logic on the host side.
    """

    inputs = ("model",)
    outputs = ("trajectories",)
    environment = "cuda"
    resources = ResourceRequest(needs_gpu=True, vram_gb=4.0, ram_gb=1.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        model = ctx.inputs["model"]
        model_container = ctx.paths.to_container(model.path, env="cuda")

        bridge_container = write_stage_bridge(ctx)

        out_host = ctx.run_dir / "trajectories.npz"
        out_container = ctx.paths.to_container(out_host, env="cuda")

        cfg = ctx.config  # SegExtractConfig's own fields (iteration/n_times); `configs` ignored,
        # same as `render.default` — this stage always generates its own bridge file.
        n_times = int(cfg.get("n_times", 60))

        # O2 (reviews/2026-10-05-omniverse-pipeline-review.md): extracting at fewer timesteps
        # than the capture has frames aliases fast periodic motion past Nyquist — the grid
        # scenes (240 frames, 40 motion cycles/clip) were extracted at the default 60, which
        # every frequency-calibrated downstream stage (rigid2 FFT denoising, kabsch FFT
        # fingerprint, trajectory_denoise) then operated on the wrong waveform for. Grid runs
        # now use the `grid_seg` preset (n_times: 240); this guard catches the same mistake for
        # any future capture/preset combination.
        frame_count = _capture_frame_count(ctx)
        if frame_count is not None and n_times < frame_count:
            ctx.logger.warning(
                "seg_extract n_times=%d is BELOW the capture's frame count (%d): periodic "
                "motion faster than ~%.3g cycles/clip will alias past Nyquist in the extracted "
                "trajectories. Set seg_extract.n_times >= the frame count (e.g. the grid_seg "
                "preset's 240 for grid captures).",
                n_times,
                frame_count,
                n_times / 2.0,
            )

        args = [
            *flag("model_path", str(model_container)),
            *flag("iteration", cfg.get("iteration", -1)),
            *flag("configs", str(bridge_container)),
            "--n-times", str(n_times),
            *flag("out", str(out_container)),
        ]

        run_cuda_script(ctx, "seg_extract", args, log_name="seg_extract")

        return {
            "trajectories": Artifact(
                name="trajectories",
                kind="npz",
                path=str(out_host),
                producing_stage=ctx.stage_name,
                metadata={"n_times": n_times},
            )
        }
