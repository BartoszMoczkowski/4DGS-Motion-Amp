"""Adaptive knobs derived from measured headroom (``planning/tasks/T12-resource-manager.md``:
"Adaptive knobs from measured headroom: ``low_vram_mode`` (amp/render_amp), segmentation
working-set / subsample (``mbs_infer.py``), ``rt_subframes`` (capture), opacity thresholds.").

Each function here is a pure "given this much free memory, what value should this knob take"
calculation — no I/O, no config mutation. ``pipeline.dag.scheduler`` (or a future caller) queries
headroom once via :mod:`pipeline.resources.query`, then asks these functions what to override in
a stage's resolved config before running it. Every function degrades to the *unmodified default*
when headroom is unknown (``free_*_mb=None``) — "don't know" must never be treated as "assume the
worst and cripple every run," only real measured tightness should adjust anything.

Thresholds/scaling are deliberately simple linear ramps, not tuned curves — the task's own "Notes
/ gotchas" says estimates start rough and get refined from real observed peak-mem over real runs
(now that T12 fills that field in). Revisit these once there's real data from Bartosz's machine.
"""

from __future__ import annotations

from typing import Optional

#: How much headroom above a stage's own VRAM estimate counts as "comfortable" — below this
#: margin, adaptive knobs start tightening. 20% is a rough safety buffer, not a measured constant.
_COMFORTABLE_MARGIN = 1.2


def should_use_low_vram_mode(
    free_vram_mb: Optional[float],
    estimated_vram_gb: float,
    *,
    default: bool = False,
) -> bool:
    """``AmpConfig.low_vram_mode`` (``core/render_amp.py``/``amp-ui/amp_ui/ampUI.py``'s ``--low_vram`` /
    "Low VRAM mode" checkbox — moves intermediate tensors off-GPU aggressively between amplify
    steps, at a speed cost). Forced ``True`` when free VRAM is below the stage's own estimate times
    :data:`_COMFORTABLE_MARGIN`; otherwise left at whatever the config already said
    (``default`` — never forces it back to ``False``, since the user may have opted in deliberately
    even with plenty of headroom).
    """
    if free_vram_mb is None:
        return default
    if free_vram_mb < estimated_vram_gb * 1024 * _COMFORTABLE_MARGIN:
        return True
    return default


def scaled_working_set(
    free_vram_mb: Optional[float],
    estimated_vram_gb: float,
    default: int,
    *,
    floor: int,
) -> int:
    """Scale a GPU working-set size (``SegmentMbsConfig.n_points``/``n_sub`` — MotNet's
    subsampled point/FPS-subsample counts, ``motion-seg/motion_seg/mbs_infer.py``'s own default is 4000/256)
    down proportionally to how far free VRAM falls short of the stage's comfortable estimate, never
    below ``floor`` (below which the algorithm's own accuracy assumptions break down — MotNet needs
    a minimum working set to do anything meaningful) and never above ``default`` (plenty of
    headroom -> use the configured value as-is, this never scales *up* past what was asked for).
    """
    if free_vram_mb is None:
        return default
    comfortable_mb = estimated_vram_gb * 1024 * _COMFORTABLE_MARGIN
    if free_vram_mb >= comfortable_mb:
        return default
    # Linear ramp: at free_vram_mb == 0 -> floor; at free_vram_mb == comfortable_mb -> default.
    ratio = max(0.0, free_vram_mb / comfortable_mb)
    scaled = int(floor + (default - floor) * ratio)
    return max(floor, min(default, scaled))


def scaled_rt_subframes(
    free_vram_mb: Optional[float],
    estimated_vram_gb: float,
    default: int,
    *,
    floor: int = 2,
) -> int:
    """Scale ``CaptureFrameConfig.rt_subframes`` (Isaac Sim path-trace samples/frame —
    ``capture_config_pump.yaml``'s ``rt_subframes: 16``, ``omni_capture.py``'s
    ``rep.orchestrator.step(rt_subframes=...)``) down under tight VRAM — fewer samples/frame is
    noisier but cheaper, the direct rendering-time/VRAM knob for ``capture.isaac``. Same linear-ramp
    shape as :func:`scaled_working_set`, floor defaults to 2 (still produces a usable, if noisy,
    frame — 1 or 0 subframes isn't a meaningful path-trace sample count).
    """
    return scaled_working_set(free_vram_mb, estimated_vram_gb, default, floor=floor)


def scaled_opacity_thresh(
    free_vram_mb: Optional[float],
    estimated_vram_gb: float,
    default: float,
    *,
    ceiling: float = 0.5,
) -> float:
    """Scale ``SegmentRigidConfig``/``SegmentMbsConfig.opacity_thresh`` (both default ``0.1`` —
    ``motion-seg/motion_seg/segment_rigid.py``/``mbs_infer.py``'s pre-filter dropping near-transparent
    Gaussians before clustering/working-set selection) *up* under tight VRAM — the inverse
    direction from :func:`scaled_working_set`: a stricter opacity cutoff drops more points before
    any working-set subsample even runs, shrinking the set MotNet/the rigidity graph actually
    operates on for free. Same linear-ramp shape, capped at ``ceiling`` (past which the algorithm
    starts discarding real, opaque-enough geometry rather than noise) and never scaled *below*
    ``default`` (plenty of headroom -> leave the configured threshold alone).
    """
    if free_vram_mb is None:
        return default
    comfortable_mb = estimated_vram_gb * 1024 * _COMFORTABLE_MARGIN
    if free_vram_mb >= comfortable_mb:
        return default
    ratio = max(0.0, free_vram_mb / comfortable_mb)
    # Linear ramp: at free_vram_mb == 0 -> ceiling; at free_vram_mb == comfortable_mb -> default.
    scaled = ceiling - (ceiling - default) * ratio
    return max(default, min(ceiling, scaled))
