"""Peak VRAM/RAM sampling across a stage's execution — fills the ``StageRecord.peak_vram_mb``/
``peak_ram_mb`` fields ``pipeline.artifacts`` (T03) left nullable for exactly this task
(``planning/tasks/T12-resource-manager.md``: "Fill the peak VRAM/RAM fields left nullable in
T03.").

A stage's own body runs the actual GPU/CPU work out-of-process (inside a container, T08/T09, or
as a native subprocess, T11) — this orchestrator process never allocates that memory itself, so
the only way to see a stage's *peak* usage is to poll the host machine's overall VRAM/RAM
periodically while the stage is running and keep the maximum used-memory sample seen. This is a
coarse, whole-machine measurement (not per-process), consistent with the two GPU images never
running concurrently (``ARCHITECTURE.md``: single-GPU serial scheduling) — during any one stage's
execution, this machine's GPU usage *is* that stage's usage (modulo whatever baseline the OS/other
processes hold, which :meth:`ResourceMonitor.stop` subtracts out via the *pre*-start baseline).
"""

from __future__ import annotations

import threading
from typing import Optional

from . import query as _query

#: how often to poll while a stage is running. Cheap enough (one pynvml/psutil call) that a short
#: interval doesn't meaningfully perturb anything, but frequent enough to catch a short-lived spike
#: a slower poll would miss.
DEFAULT_POLL_INTERVAL_S = 0.5


class ResourceMonitor:
    """Samples GPU/RAM usage on a background thread between :meth:`start` and :meth:`stop`.

    Usage::

        mon = ResourceMonitor()
        mon.start()
        ... run the stage ...
        peak_vram_mb, peak_ram_mb = mon.stop()

    Either return value is ``None`` if that dimension was never measurable (no GPU / no working
    ``pynvml``+``nvidia-smi`` / no ``psutil`` — see :mod:`pipeline.resources.query`), never a
    crash — a stage's actual success/failure must never hinge on whether this machine happens to
    have working memory telemetry.

    Peak is reported as *usage above this monitor's own start-time baseline* (``used_mb`` at
    :meth:`start` time), not raw ``used_mb`` — a machine already running other GPU/RAM-hungry
    processes shouldn't have that baseline misattributed to the stage being measured. If a sample
    ever comes in *below* the baseline (memory freed by something else mid-run), the delta floors
    at 0 rather than going negative.
    """

    def __init__(self, *, poll_interval_s: float = DEFAULT_POLL_INTERVAL_S) -> None:
        self._poll_interval_s = poll_interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._peak_vram_mb: Optional[float] = None
        self._peak_ram_mb: Optional[float] = None
        self._baseline_vram_mb: Optional[float] = None
        self._baseline_ram_mb: Optional[float] = None

    def _sample(self) -> None:
        gpu = _query.query_gpu_memory()
        if gpu is not None:
            delta = max(0.0, gpu.used_mb - (self._baseline_vram_mb or 0.0))
            self._peak_vram_mb = delta if self._peak_vram_mb is None else max(self._peak_vram_mb, delta)
        ram = _query.query_ram()
        if ram is not None:
            delta = max(0.0, ram.used_mb - (self._baseline_ram_mb or 0.0))
            self._peak_ram_mb = delta if self._peak_ram_mb is None else max(self._peak_ram_mb, delta)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            self._sample()

    def start(self) -> None:
        gpu = _query.query_gpu_memory()
        self._baseline_vram_mb = gpu.used_mb if gpu is not None else None
        ram = _query.query_ram()
        self._baseline_ram_mb = ram.used_mb if ram is not None else None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[Optional[float], Optional[float]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s * 4)
            self._thread = None
        # One final sample so a stage shorter than one poll interval still gets a real reading
        # instead of `None` purely from bad timing.
        self._sample()
        return self._peak_vram_mb, self._peak_ram_mb
