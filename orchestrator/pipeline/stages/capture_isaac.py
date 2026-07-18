"""``capture.isaac`` -- runs the vendored, ported copy of ``omni_capture.py``'s CLI
(``pipeline/vendored/isaac/omni_capture.py``) as a **native Windows subprocess** against Bartosz's
own real Isaac Sim install (T11, revised 2026-07-16 -- see ``pipeline.stages.isaac_common``'s module
docstring's "Revised again" note and ``.claude_notes/NOTES_pipeline_orchestration.md``'s "adjust
the project plan" entry). Unlike ``prep.split``/``prep.motion`` (still exec'd inside the ``isaac``
Docker container, CPU-only, unaffected), this one genuinely needs the GPU's RTX/Hydra renderer for
headless rendering -- which needs Vulkan, which NVIDIA has confirmed isn't supported under WSL2
(what backs Docker Desktop's Linux containers on Windows). Running natively sidesteps that gap
entirely; no other stage in this project needed to move.

Because this now runs natively, it does **not** go through ``ctx.paths.to_container(...)``/
``ctx.containers`` at all -- a native subprocess shares this orchestrator process's own filesystem,
so every path this stage builds (the capture dir, the ``--config`` YAML, the input USD) stays a
plain host path, passed straight through to :func:`pipeline.stages.isaac_common.
run_native_isaac_script`.

Consumes ``prep_motion.default``'s ``animated_mesh`` and produces ``capture`` -- the same artifact
name ``convert.default`` (T07) has always declared as its own external input (see that stage's
docstring: "nothing in this repo's Phase-0 scope produces one yet"). This is the stage that
finally does: with ``capture.isaac`` registered, a preset's auto-planned DAG chains
``prep_split.default -> prep_motion.default -> capture.isaac -> convert.default -> ...`` end to
end from a raw USD asset (Milestone M3), and ``convert.default`` needs no change at all.

``capture`` itself keeps role ``capture`` / impl ``isaac`` (not renamed, unlike ``prep_split``/
``prep_motion`` -- see ``pipeline.stages.prep_split``'s docstring for why those two needed
renaming): ``resolved_config["capture"]`` already exists as its own top-level section (T02), so
``role.impl`` = ``capture.isaac`` slices correctly with no collision -- there's no second
``capture.*`` impl to collide with.
"""

from __future__ import annotations

from typing import Any

import yaml

from ..artifacts import Artifact
from .base import ResourceRequest, Stage, StageContext
from .isaac_common import IsaacStageError, flag, run_native_isaac_script
from .registry import register

#: `omni_capture.py`'s `capture.{near,far}` use `cap_cfg.get(key, computed_default)` -- a key
#: present with an explicit `None` value would shadow that computed default (dict.get returns the
#: stored `None`, not the fallback), unlike every other Optional field this stage passes through
#: (which have no such fallback-shadowing risk -- see CaptureFrameConfig's docstring). Dropped here
#: rather than written as `null` so the reference script's own radius-based default kicks in.
_NONE_SHADOWS_DEFAULT = ("near", "far")


def _write_capture_config(ctx: StageContext) -> Any:
    """Write this stage-call's ``--config`` YAML under ``ctx.run_dir``, shaped exactly like
    ``omni_capture.py``'s ``_load_config`` expects (``app``/``scene``/``rig``/``capture``/
    ``output``/``lighting`` top-level keys -- ``CaptureConfig`` (T02) flattens ``app.headless`` to
    a top-level ``headless`` field, so it's re-nested here). ``scene.usd_path``/
    ``output.capture_dir`` are left as whatever the config says (mostly irrelevant -- this stage
    always overrides both via the script's own ``--usd``/``--out`` CLI flags with the DAG's real
    artifact paths instead, the same "derive from the DAG's own artifact wiring, not the static
    config value" pattern ``train.default`` uses for ``source_path``/``model_path``).
    """
    cfg = ctx.config
    capture_section = dict(cfg.get("capture", {}))
    for key in _NONE_SHADOWS_DEFAULT:
        if capture_section.get(key) is None:
            capture_section.pop(key, None)

    doc = {
        "app": {"headless": bool(cfg.get("headless", True))},
        "scene": dict(cfg.get("scene", {})),
        "rig": dict(cfg.get("rig", {})),
        "capture": capture_section,
        "output": dict(cfg.get("output", {})),
        "lighting": dict(cfg.get("lighting", {})),
    }
    yaml_host = ctx.run_dir / f"{ctx.stage_name.replace('.', '_')}_capture_config.yaml"
    # Explicit UTF-8 regardless of the writer's OS locale (cp1252 on Bartosz's native Windows) --
    # same bug class as `pipeline.config.bridge.write_bridge`'s 2026-07-18 fix, where an
    # unspecified encoding on write produced a file `omni_capture.py`'s own (UTF-8-assuming) YAML
    # loader couldn't read back if it ever contained a non-ASCII character.
    yaml_host.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return yaml_host


@register("capture.isaac")
class CaptureIsaacStage(Stage):
    """Headless multi-camera synthetic capture: opens the animated USD in Isaac Sim, rings it
    with a camera rig, and writes synchronized RGB + instance/semantic segmentation + GT camera
    poses -- exactly the directory shape ``convert.default`` already expects.
    """

    inputs = ("animated_mesh",)
    outputs = ("capture",)
    environment = "isaac"
    # Isaac Sim's headless rendering (Replicator RTX path-tracing) is comfortably more demanding
    # than the plain-Python prep stages, though far lighter than 4DGS training -- a rough estimate
    # (T12's resource manager isn't built yet to measure real headroom), padded the same way T09's
    # stages were.
    resources = ResourceRequest(needs_gpu=True, vram_gb=8.0, ram_gb=8.0)

    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        animated = ctx.inputs["animated_mesh"]

        capture_dir_host = ctx.run_dir / "capture"
        capture_dir_host.mkdir(parents=True, exist_ok=True)

        yaml_host = _write_capture_config(ctx)

        args = [
            *flag("config", str(yaml_host)),
            *flag("usd", str(animated.path)),
            *flag("out", str(capture_dir_host)),
        ]
        # --n-cameras/--headless/--frames overrides are left unset here: the yaml already carries
        # rig.n_cameras/headless/capture.num_frames from the resolved config (T02), and
        # omni_capture.py's own CLI overrides exist only to let a caller diverge from a YAML file
        # it doesn't otherwise want to touch -- not a distinction this stage needs, since it always
        # writes a fresh YAML from the resolved config on every call anyway.

        run_native_isaac_script(ctx, "omni_capture", args, log_name="capture")

        # `run_native_isaac_script` only checks the subprocess's exit code, and Isaac Sim's own
        # shutdown path has been observed (2026-07-16, first real-hardware run, see
        # .claude_notes/NOTES_pipeline_orchestration.md) to still exit 0 even when the capture
        # didn't really happen. Two distinct ways that's shown up so far, so two checks:
        #
        # 1. A fatal in-process exception right after Kit startup (e.g. a Replicator extension
        #    failing to start because of an unrelated cache-permission error, leaving `BasicWriter`
        #    unregistered) skips straight to `simulation_app` tearing down -- `cameras_gt.json`
        #    (written near the very end of `omni_capture.py`'s `main()`, right after the writer is
        #    attached) never gets written at all.
        # 2. Kit starts fine, `main()` runs to completion with no exception, `cameras_gt.json` +
        #    the point-cloud files *do* get written (they come from stage geometry, not from
        #    rendering) -- but the RTX render products themselves never produce frames (seen once
        #    as repeated "Timed out while waiting for pending Replicator writer schedules to
        #    drain" warnings, one per frame, alongside an "IHydraTexture ... no GPU foundation"
        #    error -- looks like a renderer/GPU-passthrough problem, not something this stage can
        #    fix). `BasicWriter` never wrote a single `camNN/` output directory, so `cameras_gt.json`
        #    existing alone isn't a strong enough signal -- checking the camera-directory *count*
        #    (the same thing this stage's own consumer, `convert.default`, needs and
        #    `test_stages_isaac_gpu.py` already asserts) catches this too.
        #
        # Either way: without this check the stage would report `"success"` -- and get permanently
        # cached as such via `pipeline.dag.cache.put_cached`, poisoning every future run with the
        # same config -- despite `convert.default` having nothing real to consume.
        gt_path = capture_dir_host / "cameras_gt.json"
        n_cameras = ctx.config.get("rig", {}).get("n_cameras")
        cam_dirs = [p for p in capture_dir_host.iterdir() if p.is_dir() and p.name.startswith("cam")]
        log_path = ctx.run_dir / "logs" / "capture.log"
        if not gt_path.is_file():
            raise IsaacStageError(
                f"omni_capture exited 0 but never wrote {gt_path} -- Isaac Sim likely hit a "
                f"fatal in-process error after Kit startup (it can exit 0 even after an "
                f"unhandled exception in main()); see log at {log_path}"
            )
        if n_cameras is not None and len(cam_dirs) != int(n_cameras):
            raise IsaacStageError(
                f"omni_capture exited 0 and wrote {gt_path}, but only {len(cam_dirs)}/{n_cameras} "
                f"'camNN' output directories exist under {capture_dir_host} -- the Replicator "
                f"writer likely never actually rendered any frames (e.g. a GPU-passthrough/"
                f"renderer problem, not just a startup crash); see log at {log_path}"
            )

        return {
            "capture": Artifact(
                name="capture",
                kind="dataset",
                path=str(capture_dir_host),
                producing_stage=ctx.stage_name,
                metadata={
                    "n_cameras": n_cameras,
                    "num_frames": ctx.config.get("capture", {}).get("num_frames"),
                },
            )
        }
