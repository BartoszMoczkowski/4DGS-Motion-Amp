"""Tests for T12 (resource manager): VRAM/RAM query, gating, adaptive knobs, peak-mem
monitoring, and OOM-retry.

``tests/conftest.py``'s autouse fixture already forces ``pipeline.resources.query``'s two query
functions to return ``None`` for every test by default (this sandbox's own incidental RAM has no
bearing on what's under test) — every test here that wants real-looking values monkeypatches its
own canned return on top of that default, the same way any fixture default gets overridden.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from pipeline.artifacts import Artifact
from pipeline.resources.adaptive import (
    scaled_opacity_thresh,
    scaled_rt_subframes,
    scaled_working_set,
    should_use_low_vram_mode,
)
from pipeline.resources.gating import InsufficientResourcesError, check_headroom
from pipeline.resources.monitor import ResourceMonitor
from pipeline.resources.oom_retry import (
    is_oom_error,
    reduced_memory_config,
    run_with_oom_retry,
)
from pipeline.resources.query import GpuMemoryInfo, RamInfo, query_gpu_memory, query_ram
from pipeline.stages.base import ResourceRequest, Stage, StageContext


# --- query.py -------------------------------------------------------------------------------


def test_query_gpu_memory_returns_none_when_pynvml_and_nvidia_smi_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # `import pynvml` -> ImportError
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert query_gpu_memory() is None


def test_query_gpu_memory_uses_pynvml_when_available(monkeypatch):
    class _FakeInfo:
        total = 8_000_000_000
        free = 3_000_000_000

    class _FakePynvml:
        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return "handle0"

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return _FakeInfo()

        @staticmethod
        def nvmlShutdown():
            pass

    monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
    info = query_gpu_memory()
    assert info == GpuMemoryInfo(total_mb=8000.0, free_mb=3000.0)
    assert info.used_mb == pytest.approx(5000.0)


def test_query_gpu_memory_falls_back_to_nvidia_smi_when_pynvml_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")

    class _FakeCompleted:
        stdout = "8000, 3000\n"

    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeCompleted()
    )
    info = query_gpu_memory()
    assert info == GpuMemoryInfo(total_mb=8000.0, free_mb=3000.0)


def test_query_ram_returns_none_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert query_ram() is None


def test_query_ram_uses_psutil_when_available(monkeypatch):
    class _FakeVM:
        total = 16_000_000_000
        available = 6_000_000_000

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return _FakeVM()

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil())
    info = query_ram()
    assert info == RamInfo(total_mb=16000.0, free_mb=6000.0)
    assert info.used_mb == pytest.approx(10000.0)


# --- gating.py --------------------------------------------------------------------------------


def test_check_headroom_is_a_noop_for_default_resources():
    # ResourceRequest() default (needs_gpu=False, vram_gb=0, ram_gb=0) -> never even queries.
    check_headroom(ResourceRequest())


def test_check_headroom_raises_when_vram_insufficient():
    resources = ResourceRequest(needs_gpu=True, vram_gb=10.0)
    gpu = GpuMemoryInfo(total_mb=8000.0, free_mb=2000.0)
    with pytest.raises(InsufficientResourcesError, match="VRAM"):
        check_headroom(resources, gpu=gpu)


def test_check_headroom_passes_when_vram_sufficient():
    resources = ResourceRequest(needs_gpu=True, vram_gb=4.0)
    gpu = GpuMemoryInfo(total_mb=8000.0, free_mb=6000.0)
    check_headroom(resources, gpu=gpu)  # no raise


def test_check_headroom_raises_when_ram_insufficient():
    resources = ResourceRequest(ram_gb=16.0)
    ram = RamInfo(total_mb=8000.0, free_mb=3000.0)
    with pytest.raises(InsufficientResourcesError, match="RAM"):
        check_headroom(resources, ram=ram)


def test_check_headroom_unknown_headroom_is_a_noop():
    # gpu/ram explicitly None (unmeasurable) -> fails open even for a huge estimate.
    resources = ResourceRequest(needs_gpu=True, vram_gb=999.0, ram_gb=999.0)
    check_headroom(resources, gpu=None, ram=None)


# --- adaptive.py ------------------------------------------------------------------------------


def test_should_use_low_vram_mode_forces_true_under_tight_headroom():
    assert should_use_low_vram_mode(free_vram_mb=8000.0, estimated_vram_gb=10.0) is True


def test_should_use_low_vram_mode_leaves_default_when_comfortable():
    assert should_use_low_vram_mode(free_vram_mb=20000.0, estimated_vram_gb=10.0, default=False) is False


def test_should_use_low_vram_mode_returns_default_when_unknown():
    assert should_use_low_vram_mode(free_vram_mb=None, estimated_vram_gb=10.0, default=True) is True
    assert should_use_low_vram_mode(free_vram_mb=None, estimated_vram_gb=10.0, default=False) is False


def test_scaled_working_set_scales_down_under_tight_headroom():
    # comfortable_mb = 4.0 * 1024 * 1.2 = 4915.2; free at half of that -> roughly midway between
    # floor and default.
    value = scaled_working_set(free_vram_mb=2457.6, estimated_vram_gb=4.0, default=4000, floor=500)
    assert 500 < value < 4000


def test_scaled_working_set_never_exceeds_default_when_comfortable():
    assert scaled_working_set(free_vram_mb=100_000.0, estimated_vram_gb=4.0, default=4000, floor=500) == 4000


def test_scaled_working_set_floors_at_zero_free():
    assert scaled_working_set(free_vram_mb=0.0, estimated_vram_gb=4.0, default=4000, floor=500) == 500


def test_scaled_working_set_returns_default_when_unknown():
    assert scaled_working_set(free_vram_mb=None, estimated_vram_gb=4.0, default=4000, floor=500) == 4000


def test_scaled_rt_subframes_floors_at_two_by_default():
    assert scaled_rt_subframes(free_vram_mb=0.0, estimated_vram_gb=8.0, default=16) == 2


def test_scaled_opacity_thresh_rises_under_tight_headroom_and_caps_at_ceiling():
    assert scaled_opacity_thresh(free_vram_mb=0.0, estimated_vram_gb=4.0, default=0.1, ceiling=0.5) == pytest.approx(0.5)


def test_scaled_opacity_thresh_stays_at_default_when_comfortable():
    assert scaled_opacity_thresh(free_vram_mb=100_000.0, estimated_vram_gb=4.0, default=0.1) == 0.1


def test_scaled_opacity_thresh_returns_default_when_unknown():
    assert scaled_opacity_thresh(free_vram_mb=None, estimated_vram_gb=4.0, default=0.1) == 0.1


# --- oom_retry.py -----------------------------------------------------------------------------


class _FakeGpuError(RuntimeError):
    def __init__(self, message: str, *, log_path: str | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def test_is_oom_error_detects_a_marker_in_the_log(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("some output\nRuntimeError: CUDA out of memory. Tried to allocate...\n")
    assert is_oom_error(_FakeGpuError("exit 1", log_path=str(log_path))) is True


def test_is_oom_error_false_for_unrelated_failure(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("some output\nFileNotFoundError: no such file\n")
    assert is_oom_error(_FakeGpuError("exit 1", log_path=str(log_path))) is False


def test_is_oom_error_false_when_exception_has_no_log_path():
    assert is_oom_error(RuntimeError("boom")) is False


def test_is_oom_error_false_when_log_missing_on_disk(tmp_path: Path):
    assert is_oom_error(_FakeGpuError("exit 1", log_path=str(tmp_path / "does_not_exist.log"))) is False


def test_reduced_memory_config_amp_forces_low_vram_then_gives_up():
    cfg = {"low_vram_mode": False, "method": "eulerian"}
    fallback = reduced_memory_config("amp.default", cfg)
    assert fallback == {"low_vram_mode": True, "method": "eulerian"}
    # already tried -> nothing left to reduce.
    assert reduced_memory_config("amp.default", fallback) is None


def test_reduced_memory_config_segment_mbs_halves_working_set_then_floors():
    cfg = {"n_points": 4000, "n_sub": 256}
    fallback = reduced_memory_config("segment.mbs", cfg)
    assert fallback == {"n_points": 2000, "n_sub": 128}
    fallback2 = reduced_memory_config("segment.mbs", fallback)
    assert fallback2 == {"n_points": 1000, "n_sub": 64}
    # keep halving until both hit their floor, then no further fallback.
    floor_cfg = {"n_points": 500, "n_sub": 64}
    assert reduced_memory_config("segment.mbs", floor_cfg) is None


def test_reduced_memory_config_capture_isaac_halves_rt_subframes_then_floors():
    cfg = {"capture": {"rt_subframes": 16}}
    fallback = reduced_memory_config("capture.isaac", cfg)
    assert fallback == {"capture": {"rt_subframes": 8}}
    floor_cfg = {"capture": {"rt_subframes": 2}}
    assert reduced_memory_config("capture.isaac", floor_cfg) is None


def test_reduced_memory_config_unknown_stage_returns_none():
    assert reduced_memory_config("train.default", {"iterations": 30000}) is None
    assert reduced_memory_config("render.default", {}) is None


def _ctx(tmp_path: Path, *, config: dict, stage_name: str = "amp.toy") -> StageContext:
    return StageContext(
        run_id="r1",
        stage_name=stage_name,
        config=config,
        run_dir=tmp_path,
        logger=logging.getLogger(f"test.{stage_name}"),
        inputs={},
    )


def test_run_with_oom_retry_succeeds_on_first_try(tmp_path: Path):
    calls = []

    class _OkStage(Stage):
        def run(self, ctx: StageContext):
            calls.append(dict(ctx.config))
            return {}

    ctx = _ctx(tmp_path, config={"low_vram_mode": False})
    result, fallback = run_with_oom_retry(_OkStage, ctx, "amp.toy")
    assert result == {}
    assert fallback is None
    assert len(calls) == 1


def test_run_with_oom_retry_retries_once_after_oom_and_succeeds(tmp_path: Path):
    log_path = tmp_path / "amp.log"
    log_path.write_text("...\nCUDA out of memory\n")
    calls = []

    class _FlakyAmpStage(Stage):
        def run(self, ctx: StageContext):
            calls.append(dict(ctx.config))
            if len(calls) == 1:
                raise _FakeGpuError("exit 1", log_path=str(log_path))
            return {"amp_video": Artifact(name="amp_video", kind="video", path="x", producing_stage="amp.toy")}

    ctx = _ctx(tmp_path, config={"low_vram_mode": False, "method": "eulerian"})
    result, fallback = run_with_oom_retry(_FlakyAmpStage, ctx, "amp.toy")

    assert "amp_video" in result
    assert fallback == {"reason": "cuda_oom", "changed": {"low_vram_mode": True}}
    assert len(calls) == 2
    assert calls[0] == {"low_vram_mode": False, "method": "eulerian"}
    assert calls[1] == {"low_vram_mode": True, "method": "eulerian"}
    assert ctx.config == {"low_vram_mode": True, "method": "eulerian"}  # left applied on success


def test_run_with_oom_retry_reraises_when_no_fallback_exists(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("CUDA out of memory\n")

    class _AlwaysOomStage(Stage):
        def run(self, ctx: StageContext):
            raise _FakeGpuError("exit 1", log_path=str(log_path))

    ctx = _ctx(tmp_path, config={"iterations": 30000}, stage_name="train.default")
    with pytest.raises(_FakeGpuError):
        run_with_oom_retry(_AlwaysOomStage, ctx, "train.default")
    assert ctx.config == {"iterations": 30000}  # untouched -- no retry was attempted


def test_run_with_oom_retry_reraises_non_oom_errors_without_retrying(tmp_path: Path):
    calls = []

    class _PlainFailStage(Stage):
        def run(self, ctx: StageContext):
            calls.append(1)
            raise ValueError("not a memory problem")

    ctx = _ctx(tmp_path, config={"low_vram_mode": False}, stage_name="amp.toy")
    with pytest.raises(ValueError):
        run_with_oom_retry(_PlainFailStage, ctx, "amp.toy")
    assert len(calls) == 1  # never retried


def test_run_with_oom_retry_restores_config_when_retry_also_fails(tmp_path: Path):
    log_path = tmp_path / "amp.log"
    log_path.write_text("CUDA out of memory\n")
    calls = []

    class _DoubleFailStage(Stage):
        def run(self, ctx: StageContext):
            calls.append(dict(ctx.config))
            raise _FakeGpuError("exit 1", log_path=str(log_path))

    ctx = _ctx(tmp_path, config={"low_vram_mode": False}, stage_name="amp.toy")
    with pytest.raises(_FakeGpuError):
        run_with_oom_retry(_DoubleFailStage, ctx, "amp.toy")
    assert len(calls) == 2
    assert ctx.config == {"low_vram_mode": False}  # restored after the retry itself failed too


# --- monitor.py -------------------------------------------------------------------------------


def test_resource_monitor_reports_none_when_telemetry_unavailable():
    # conftest's autouse fixture already forces both query fns to None.
    mon = ResourceMonitor(poll_interval_s=100)
    mon.start()
    peak_vram_mb, peak_ram_mb = mon.stop()
    assert peak_vram_mb is None
    assert peak_ram_mb is None


def test_resource_monitor_tracks_peak_delta_above_baseline(monkeypatch):
    from pipeline.resources import monitor as monitor_mod

    used_values = iter([1000.0, 4000.0])  # first call (start's baseline), second (stop's sample)

    def fake_gpu():
        v = next(used_values, 4000.0)
        return GpuMemoryInfo(total_mb=10_000.0, free_mb=10_000.0 - v)

    monkeypatch.setattr(monitor_mod._query, "query_gpu_memory", fake_gpu)
    monkeypatch.setattr(monitor_mod._query, "query_ram", lambda: None)

    # A huge poll interval means the background thread never wakes up during this short test --
    # only start()'s baseline read and stop()'s final catch-up sample run, so exactly two
    # `fake_gpu` calls happen, deterministically.
    mon = ResourceMonitor(poll_interval_s=100)
    mon.start()
    peak_vram_mb, peak_ram_mb = mon.stop()
    assert peak_vram_mb == pytest.approx(3000.0)
    assert peak_ram_mb is None
