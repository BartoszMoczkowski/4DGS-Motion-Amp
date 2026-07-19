"""Query current VRAM (GPU) and system RAM headroom.

Per the package docstring / ``planning/tasks/T12-resource-manager.md``: this is read-only
telemetry, never imported at ``pipeline.resources`` module scope with a heavy dependency —
``pynvml``/``psutil`` are only ever imported *inside* a function, mirroring
``pipeline.containers.manager``'s "``docker`` only imported inside methods" convention
(``tests/test_import.py``'s ``test_no_heavy_imports_at_module_scope`` covers ``pynvml`` the same
way it already covers ``docker``/``torch``).

Every query function degrades to ``None`` rather than raising when detection isn't possible (no
GPU, ``pynvml``/``nvidia-smi`` unavailable, ``psutil`` unavailable) — a CPU-only sandbox or a
machine with a GPU driver hiccup must still be able to import and run everything that isn't
actually gated on real headroom. Callers (``gating``/``adaptive``) treat ``None`` as "unknown,
don't block" — see their own docstrings.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuMemoryInfo:
    """One GPU's memory snapshot, in MB. ``used_mb`` is derived (``total_mb - free_mb``), not
    queried separately — matches what ``nvidia-smi``/``pynvml`` themselves report directly."""

    total_mb: float
    free_mb: float

    @property
    def used_mb(self) -> float:
        return self.total_mb - self.free_mb


@dataclass(frozen=True)
class RamInfo:
    """System RAM snapshot, in MB."""

    total_mb: float
    free_mb: float

    @property
    def used_mb(self) -> float:
        return self.total_mb - self.free_mb


def _query_gpu_memory_pynvml() -> Optional[GpuMemoryInfo]:
    try:
        import pynvml  # local import — see module docstring
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return GpuMemoryInfo(total_mb=info.total / 1e6, free_mb=info.free / 1e6)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - any pynvml/driver failure -> "can't measure", not a crash
        return None


def _query_gpu_memory_nvidia_smi() -> Optional[GpuMemoryInfo]:
    """Fallback for a machine with the ``nvidia-smi`` CLI on ``PATH`` but no working ``pynvml``
    binding (e.g. a driver/binding version mismatch) — same two numbers, parsed from
    ``--query-gpu=memory.total,memory.free``. Only the *first* GPU is read, matching
    :func:`_query_gpu_memory_pynvml`'s single-GPU assumption (``ARCHITECTURE.md``: "single-GPU
    serial scheduling is fine").
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first_line = out.stdout.strip().splitlines()[0]
        total_str, free_str = (p.strip() for p in first_line.split(","))
        return GpuMemoryInfo(total_mb=float(total_str), free_mb=float(free_str))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def query_gpu_memory() -> Optional[GpuMemoryInfo]:
    """Total + free VRAM for GPU 0, or ``None`` if it can't be measured (no GPU, no working
    ``pynvml``/``nvidia-smi``). Tries ``pynvml`` first (cheaper, no subprocess), falls back to
    ``nvidia-smi``.
    """
    info = _query_gpu_memory_pynvml()
    if info is not None:
        return info
    return _query_gpu_memory_nvidia_smi()


def query_ram() -> Optional[RamInfo]:
    """Total + free system RAM, or ``None`` if it can't be measured (``psutil`` unavailable)."""
    try:
        import psutil  # local import — see module docstring
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
        # `available` (not `free`) is psutil's own "actually usable without swapping" estimate —
        # accounts for reclaimable cache/buffers, matching what a new process could realistically
        # allocate, which is what a pre-flight gate/OOM-avoidance check cares about.
        return RamInfo(total_mb=vm.total / 1e6, free_mb=vm.available / 1e6)
    except Exception:  # noqa: BLE001 - psutil present but failing -> "can't measure"
        return None
