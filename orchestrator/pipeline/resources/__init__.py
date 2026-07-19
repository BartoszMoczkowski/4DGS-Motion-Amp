"""VRAM/RAM resource manager: pynvml/nvidia-smi/psutil query, serial gating, adaptive knobs,
OOM-retry, peak-mem monitoring (T12 — ``planning/tasks/T12-resource-manager.md``).

Four leaf-ish pieces, mirroring ``pipeline.containers``'s "pure data / one module that talks to
the real thing" split:

- :mod:`pipeline.resources.query` — read-only VRAM/RAM telemetry (``pynvml``/``nvidia-smi``/
  ``psutil``, all lazily imported — never at this package's module scope, so importing
  ``pipeline`` stays safe with no GPU/psutil installed, mirroring
  ``pipeline.containers.manager``'s "``docker`` only imported inside methods" convention;
  ``tests/test_import.py``'s ``test_no_heavy_imports_at_module_scope`` covers ``pynvml`` here the
  same way it already covers ``docker``/``torch``).
- :mod:`pipeline.resources.gating` — ``check_headroom``: the pre-flight check
  ``pipeline.dag.scheduler`` calls right before a stage runs, so a stage whose estimate exceeds
  currently-free VRAM/RAM fails cleanly instead of crashing mid-execution.
- :mod:`pipeline.resources.adaptive` — pure "given this much free memory, what should this knob
  be" calculations for ``low_vram_mode``/segmentation working-set/``rt_subframes``/
  ``opacity_thresh``.
- :mod:`pipeline.resources.monitor` — ``ResourceMonitor``: samples peak VRAM/RAM across a
  stage's execution, filling ``StageRecord.peak_vram_mb``/``peak_ram_mb`` (nullable placeholders
  T03 left for this task).
- :mod:`pipeline.resources.oom_retry` — ``run_with_oom_retry``: catches an apparent CUDA OOM,
  retries once with a stage-specific reduced-memory config if one exists, and reports what
  changed for the manifest's new ``StageRecord.oom_fallback`` field.
"""

from __future__ import annotations

from .adaptive import (
    scaled_opacity_thresh,
    scaled_rt_subframes,
    scaled_working_set,
    should_use_low_vram_mode,
)
from .gating import InsufficientResourcesError, check_headroom
from .monitor import ResourceMonitor
from .oom_retry import is_oom_error, reduced_memory_config, run_with_oom_retry
from .query import GpuMemoryInfo, RamInfo, query_gpu_memory, query_ram

__all__ = [
    # query
    "GpuMemoryInfo",
    "RamInfo",
    "query_gpu_memory",
    "query_ram",
    # gating
    "InsufficientResourcesError",
    "check_headroom",
    # adaptive
    "should_use_low_vram_mode",
    "scaled_working_set",
    "scaled_rt_subframes",
    "scaled_opacity_thresh",
    # monitor
    "ResourceMonitor",
    # oom_retry
    "is_oom_error",
    "reduced_memory_config",
    "run_with_oom_retry",
    # top-level convenience
    "gpu_status",
]


def gpu_status() -> dict:
    """Current VRAM/RAM snapshot as a plain dict — what ``pipeline.api.gpu_status`` exposes to
    Layers 2/3. ``None`` fields mean that dimension couldn't be measured on this machine (see
    :mod:`pipeline.resources.query`'s docstring), not an error.
    """
    gpu = query_gpu_memory()
    ram = query_ram()
    return {
        "gpu": None
        if gpu is None
        else {"total_mb": gpu.total_mb, "free_mb": gpu.free_mb, "used_mb": gpu.used_mb},
        "ram": None
        if ram is None
        else {"total_mb": ram.total_mb, "free_mb": ram.free_mb, "used_mb": ram.used_mb},
    }
