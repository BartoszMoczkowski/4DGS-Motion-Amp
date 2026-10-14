"""tests/test_capture_studio.py

Unit tests for DahuaController, CaptureManager, and FastAPI Web Studio endpoints.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rtsp_capture.dahua_api import DahuaController
from rtsp_capture.manager import CaptureManager
from rtsp_capture.web_app import (
    BatchConfigRequest,
    ExposureRequest,
    FlashRequest,
    WhiteBalanceRequest,
    create_app,
)


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text(
        """
credentials:
  user: admin
  password: "test_password"

cameras:
  cam01: {ip: 192.168.3.184, subtype: 0}
  cam02: {ip: 192.168.3.159, subtype: 0}
  cam03: {ip: 192.168.3.162, subtype: 0}
""",
        encoding="utf-8",
    )
    return cfg


def test_dahua_controller_mock():
    ctrl = DahuaController(mock_mode=True)
    ctrl.register_camera("cam01", "192.168.3.184")

    # Color & White Balance
    ok, _ = ctrl.set_white_balance("cam01", style="Manual", gain_red=62, gain_blue=41)
    assert ok
    col = ctrl.get_color_config("cam01")
    assert col.wb_style == "Manual"
    assert col.gain_red == 62
    assert col.gain_blue == 41

    # Exposure
    ok, _ = ctrl.set_exposure("cam01", mode="Manual", shutter="1/250", gain_max=70)
    assert ok
    exp = ctrl.get_exposure_config("cam01")
    assert exp.mode == "Manual"
    assert exp.shutter == "1/250"
    assert exp.gain_max == 70

    # Snapshot returns valid JPEG
    jpeg = ctrl.get_snapshot("cam01")
    assert len(jpeg) > 100
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker

    # Light Flash
    assert ctrl.flash_camera_light("cam01", duration_sec=0.2, pulses=2)


def test_capture_manager_mock(dummy_config: Path, tmp_path: Path):
    mgr = CaptureManager(dummy_config, mock_mode=True)
    assert len(mgr.cameras) == 3

    telem = mgr.get_telemetry()
    assert telem["state"] == "IDLE"
    assert telem["camera_count"] == 3
    assert telem["mock_mode"] is True

    # Probe in mock mode
    probe_res = mgr.probe_all(duration=1.0)
    assert len(probe_res) == 3
    for r in probe_res:
        assert r["ok"] is True
        assert r["srs"] >= 2

    # Start and stop recording
    rec_out = tmp_path / "test_recordings"
    mgr.start_recording(rec_out, session_name="test_run", duration_sec=1.0)
    time.sleep(0.3)
    t_active = mgr.get_telemetry()
    assert t_active["state"] == "RECORDING"

    mgr.stop_recording()
    t_idle = mgr.get_telemetry()
    assert t_idle["state"] == "IDLE"


def test_web_app_endpoint_logic(dummy_config: Path):
    mgr = CaptureManager(dummy_config, mock_mode=True)
    app = create_app(mgr)

    # Find registered endpoint functions
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}

    # 1. /api/status
    status_fn = routes["/api/status"]
    status_data = status_fn()
    assert status_data["camera_count"] == 3
    assert "cam01" in status_data["cameras"]

    # 2. /api/cameras/{camera_id}/snapshot
    snap_fn = routes["/api/cameras/{camera_id}/snapshot"]
    resp_snap = snap_fn("cam01")
    assert resp_snap.media_type == "image/jpeg"
    assert resp_snap.body[:2] == b"\xff\xd8"

    # 3. /api/cameras/{camera_id}/flash
    flash_fn = routes["/api/cameras/{camera_id}/flash"]
    flash_res = flash_fn("cam01", FlashRequest(duration_sec=0.2, pulses=1))
    assert flash_res["status"] == "flashing"

    # 4. /api/cameras/batch-config
    batch_fn = routes["/api/cameras/batch-config"]
    batch_res = batch_fn(
        BatchConfigRequest(
            white_balance=WhiteBalanceRequest(style="Manual", gain_red=55, gain_blue=48),
            exposure=ExposureRequest(mode="Manual", shutter="1/200"),
        )
    )
    assert len(batch_res["updated_cameras"]) == 3

    # Verify state was updated
    cam02_state = mgr.dahua.get_state("cam02")
    assert cam02_state.color.gain_red == 55
    assert cam02_state.exposure.shutter == "1/200"

    # 5. /api/probe
    probe_fn = routes["/api/probe"]
    probe_res = probe_fn(duration=1.0)
    assert len(probe_res["probe_results"]) == 3
