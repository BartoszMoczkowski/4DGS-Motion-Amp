"""OOM detection + single reduced-memory retry (``planning/tasks/T12-resource-manager.md``:
"OOM-retry: on CUDA OOM, retry the stage once with reduced-memory settings before failing; record
the fallback in the manifest.").

Every ``cuda``/``isaac``-environment stage (T09/T11) runs its real work as a separate process
(container exec or native subprocess) and reports failure by raising ``CudaStageError``/
``IsaacStageError`` with a ``log_path`` attribute pointing at that process's captured stdout/
stderr (see ``pipeline.stages.cuda_common``/``isaac_common``) — :func:`is_oom_error` reads that
log rather than the exception message alone, since the message is just
``"<script> exited with code <n>; see log at <path>"``, not the actual CUDA error text.

:func:`reduced_memory_config` is deliberately narrow: only stages with a *known-safe* config knob
that plausibly reduces peak memory get a fallback at all (``amp``'s ``low_vram_mode``,
``segment.mbs``'s working-set size, ``capture.isaac``'s ``rt_subframes``) — ``train``/``render``/
``seg_extract`` have no such knob exposed yet, so a real OOM there re-raises immediately rather
than silently guessing at an unproven mitigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..artifacts import Artifact
from ..stages.base import Stage, StageContext

#: substrings that show up in a vendored CUDA script's captured output on an actual GPU
#: out-of-memory condition — PyTorch's own OOM message, plus the raw CUDA runtime error text it
#: sometimes wraps instead.
OOM_MARKERS: tuple[str, ...] = (
    "CUDA out of memory",
    "cudaErrorMemoryAllocation",
    "OutOfMemoryError",
    "CUDA error: out of memory",
)


def _read_log_text(log_path: Optional[str]) -> str:
    if not log_path:
        return ""
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def is_oom_error(exc: Exception) -> bool:
    """``True`` if ``exc`` (a stage's ``run()`` failure) looks like a GPU out-of-memory condition,
    judged by scanning the log file its own ``log_path`` attribute points at (``CudaStageError``/
    ``IsaacStageError`` — see module docstring). ``False`` for any exception with no such
    attribute (a plain ``ValueError``/config error, or a toy test stage), or whose log has none of
    :data:`OOM_MARKERS`.
    """
    text = _read_log_text(getattr(exc, "log_path", None))
    if not text:
        return False
    return any(marker in text for marker in OOM_MARKERS)


def reduced_memory_config(stage_name: str, cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """One reduced-memory variant of ``cfg`` for a single retry, or ``None`` if ``stage_name`` has
    no known memory-reduction knob left to try (including "already tried it" — e.g. ``amp`` with
    ``low_vram_mode`` already ``True``, or a working-set size already at its floor) — in which case
    the original OOM should just propagate rather than retry with an unchanged config that would
    only OOM identically again.
    """
    role = stage_name.split(".", 1)[0]

    if role == "amp":
        if cfg.get("low_vram_mode"):
            return None
        return {**cfg, "low_vram_mode": True}

    if stage_name == "segment.mbs":
        n_points = int(cfg.get("n_points", 4000))
        n_sub = int(cfg.get("n_sub", 256))
        new_points = max(500, n_points // 2)
        new_sub = max(64, n_sub // 2)
        if new_points == n_points and new_sub == n_sub:
            return None
        return {**cfg, "n_points": new_points, "n_sub": new_sub}

    if stage_name == "capture.isaac":
        capture = dict(cfg.get("capture", {}))
        rt_subframes = int(capture.get("rt_subframes", 8))
        new_rt_subframes = max(2, rt_subframes // 2)
        if new_rt_subframes == rt_subframes:
            return None
        capture["rt_subframes"] = new_rt_subframes
        return {**cfg, "capture": capture}

    return None


def run_with_oom_retry(
    stage_cls: type[Stage],
    ctx: StageContext,
    stage_name: str,
) -> tuple[dict[str, Artifact], Optional[dict[str, Any]]]:
    """Run ``stage_cls().run(ctx)``; on a failure :func:`is_oom_error` recognizes, retry exactly
    once with :func:`reduced_memory_config`'s fallback, if one exists for ``stage_name``.

    Returns ``(result, fallback_info)`` — ``fallback_info`` is ``None`` on a first-try success,
    or a small JSON-able dict describing what changed (``pipeline.dag.scheduler`` records this
    straight into ``StageRecord.oom_fallback``, T12's own new manifest field) on a successful
    retry. Re-raises the *original* exception unchanged if no retry was attempted (not an OOM, or
    no fallback exists for this stage) or the retry itself also fails — a caller sees exactly the
    same exception shape as any other stage failure, never something masked or re-typed.
    """
    try:
        result = stage_cls().run(ctx)
        return result, None
    except Exception as exc:  # noqa: BLE001 - inspect, then either retry or re-raise as-is
        if not is_oom_error(exc):
            raise
        fallback_cfg = reduced_memory_config(stage_name, ctx.config)
        if fallback_cfg is None:
            raise
        original_cfg = ctx.config
        changed = {k: v for k, v in fallback_cfg.items() if original_cfg.get(k) != v}
        ctx.logger.warning(
            "stage %s hit an apparent CUDA OOM (see %s); retrying once with reduced-memory "
            "settings: %s",
            stage_name,
            getattr(exc, "log_path", "?"),
            changed,
        )
        ctx.config = fallback_cfg
        try:
            result = stage_cls().run(ctx)
        except Exception:
            ctx.config = original_cfg
            raise
        return result, {"reason": "cuda_oom", "changed": changed}
