"""Pre-flight gating: never start a stage whose declared estimate exceeds *currently free*
VRAM/RAM — the one hook ``pipeline.dag.scheduler`` calls right before executing a stage
(``planning/tasks/T12-resource-manager.md``: "Gate the serial scheduler...").

Single-GPU serial scheduling (``ARCHITECTURE.md``) means this only ever needs to check "is there
enough headroom right now for the one stage about to start" — no reservation bookkeeping across
concurrently-running stages, because there never are any.

Fails **open**, not closed, whenever headroom can't be measured (:mod:`pipeline.resources.query`
returns ``None`` — no GPU, no working ``pynvml``/``nvidia-smi``/``psutil``): a CPU-only sandbox or
CI box must keep running the full test suite exactly as it did before T12 existed, and a stage
that doesn't declare any GPU/RAM need (``ResourceRequest()`` default) is never gated at all.

Imports :mod:`pipeline.resources.query` as a module (``_query``), not its individual functions —
mirrors ``pipeline.dag.scheduler``'s own ``from .. import containers as _containers`` convention —
so ``tests/conftest.py``'s autouse fixture (which forces "can't measure" for every test by
default, since this sandbox's own incidental RAM has nothing to do with what a stage would need on
a real target machine) can monkeypatch ``pipeline.resources.query``'s functions directly and have
every caller see it, rather than a stale name-imported reference.
"""

from __future__ import annotations

from typing import Optional

from ..stages.base import ResourceRequest
from . import query as _query
from .query import GpuMemoryInfo, RamInfo


class InsufficientResourcesError(RuntimeError):
    """A stage's ``ResourceRequest`` estimate exceeds currently free VRAM/RAM.

    Raised *before* the stage runs (not a real OOM) — caught by
    ``pipeline.dag.scheduler.run_dag``'s existing per-stage ``except Exception`` the same as any
    other stage failure, so it's recorded as a clean ``"failed"`` manifest entry with this
    message, never an opaque crash mid-execution (the acceptance criterion's "...or fails cleanly
    with a clear message").
    """


def check_headroom(
    resources: ResourceRequest,
    *,
    gpu: Optional[GpuMemoryInfo] = None,
    ram: Optional[RamInfo] = None,
) -> None:
    """Raise :class:`InsufficientResourcesError` if ``resources`` needs more VRAM/RAM than is
    currently free. A no-op (no query even performed) for a stage that declares no need
    (``needs_gpu=False`` and ``ram_gb <= 0``, the ``ResourceRequest`` default) — most CPU stages.

    ``gpu``/``ram`` are injectable (tests pass canned :class:`~pipeline.resources.query.GpuMemoryInfo`/
    :class:`~pipeline.resources.query.RamInfo` values); a real caller leaves them ``None`` and gets
    a fresh :func:`~pipeline.resources.query.query_gpu_memory`/:func:`~pipeline.resources.query.query_ram`
    call. Either query returning ``None`` (can't measure) means that dimension is never gated —
    see module docstring.
    """
    if resources.needs_gpu and resources.vram_gb > 0:
        gpu_info = gpu if gpu is not None else _query.query_gpu_memory()
        if gpu_info is not None:
            needed_mb = resources.vram_gb * 1024
            if needed_mb > gpu_info.free_mb:
                raise InsufficientResourcesError(
                    f"stage needs an estimated {resources.vram_gb:.1f} GB VRAM but only "
                    f"{gpu_info.free_mb / 1024:.1f} GB is currently free "
                    f"(total {gpu_info.total_mb / 1024:.1f} GB) — free up VRAM (close other "
                    f"GPU-using processes/containers) before retrying"
                )

    if resources.ram_gb > 0:
        ram_info = ram if ram is not None else _query.query_ram()
        if ram_info is not None:
            needed_mb = resources.ram_gb * 1024
            if needed_mb > ram_info.free_mb:
                raise InsufficientResourcesError(
                    f"stage needs an estimated {resources.ram_gb:.1f} GB RAM but only "
                    f"{ram_info.free_mb / 1024:.1f} GB is currently free "
                    f"(total {ram_info.total_mb / 1024:.1f} GB) — free up RAM before retrying"
                )
