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
    PictureRequest,
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
            picture=PictureRequest(brightness=55, contrast=52),
        )
    )
    assert len(batch_res["updated_cameras"]) == 3
    assert batch_res["ok"] is True

    # Verify state was updated
    cam02_state = mgr.dahua.get_state("cam02")
    assert cam02_state.color.gain_red == 55
    assert cam02_state.color.brightness == 55
    assert cam02_state.color.contrast == 52
    assert cam02_state.exposure.shutter == "1/200"

    read_fn = next(
        route.endpoint
        for route in app.routes
        if route.path == "/api/cameras/{camera_id}/config" and "GET" in route.methods
    )
    readback = read_fn("cam02")
    assert readback["ok"] is True
    assert readback["color"]["brightness"] == 55
    assert readback["exposure"]["shutter"] == "1/200"

    # 5. /api/probe
    probe_fn = routes["/api/probe"]
    probe_res = probe_fn(duration=1.0)
    assert len(probe_res["probe_results"]) == 3


def test_dahua_controller_live_schema_and_readback(monkeypatch):
    ctrl = DahuaController()
    ctrl.register_camera("cam01", "192.168.3.183")
    values = {
        "VideoInWhiteBalance[0][0].Mode": "Auto",
        "VideoInWhiteBalance[0][0].GainRed": "75",
        "VideoInWhiteBalance[0][0].GainBlue": "30",
        "VideoInWhiteBalance[0][0].GainGreen": "50",
        "VideoInColor[0][0].Brightness": "55",
        "VideoInColor[0][0].Contrast": "52",
        "VideoInColor[0][0].Saturation": "50",
        "VideoInSharpness[0][0].Sharpness": "50",
        "VideoInExposure[0][0].Mode": "0",
        "VideoInExposure[0][0].Speed": "0",
        "VideoInExposure[0][0].GainMax": "50",
    }
    calls = []

    def fake_cgi(ip, script, params, method="GET"):
        if params["action"] == "getConfig":
            prefix = f"{params['name']}["
            text = "\r\n".join(
                f"table.{key}={value}"
                for key, value in values.items()
                if key.startswith(prefix)
            )
            return True, text + "\r\n"
        calls.append(params.copy())
        for key, value in params.items():
            if key != "action":
                values[key] = str(value)
        return True, "OK\r\n"

    monkeypatch.setattr(ctrl, "_cgi_request", fake_cgi)

    color = ctrl.get_color_config("cam01")
    exposure = ctrl.get_exposure_config("cam01")
    assert color.wb_style == "Auto"
    assert color.gain_red == 75
    assert color.gain_blue == 30
    assert color.brightness == 55
    assert color.contrast == 52
    assert exposure.mode == "Auto"

    ok, msg = ctrl.set_white_balance("cam01", style="Manual", gain_red=62, gain_blue=41)
    assert ok, msg
    assert calls[-1] == {
        "action": "setConfig",
        "VideoInWhiteBalance[0][0].Mode": "Manual",
        "VideoInWhiteBalance[0][0].GainRed": 62,
        "VideoInWhiteBalance[0][0].GainBlue": 41,
        "VideoInWhiteBalance[0][0].GainGreen": 50,
    }

    ok, msg = ctrl.set_picture("cam01", brightness=50, contrast=50, saturation=51, sharpness=60)
    assert ok, msg
    assert calls[-1] == {
        "action": "setConfig",
        "VideoInColor[0][0].Brightness": 50,
        "VideoInColor[0][0].Contrast": 50,
        "VideoInColor[0][0].Saturation": 51,
        "VideoInSharpness[0][0].Sharpness": 60,
    }

    ok, msg = ctrl.set_exposure("cam01", mode="Manual", shutter="1/200", gain_max=70)
    assert ok, msg
    assert calls[-1] == {
        "action": "setConfig",
        "VideoInExposure[0][0].Mode": 1,
        "VideoInExposure[0][0].Speed": 200,
        "VideoInExposure[0][0].GainMax": 70,
    }
    assert ctrl.get_state("cam01").exposure.shutter == "1/200"
    assert ctrl.get_state("cam01").exposure.gain_max == 70

    ok, msg = ctrl.set_exposure("cam01", mode="Auto")
    assert ok, msg
    assert calls[-1]["VideoInExposure[0][0].Mode"] == 0
    assert calls[-1]["VideoInExposure[0][0].Speed"] == 0
    assert ctrl.get_state("cam01").exposure.mode == "Auto"


def test_cgi_http_200_error_is_failure(monkeypatch):
    class ErrorResponse:
        status_code = 200
        text = "Error\r\nInvalid parameter"

    monkeypatch.setattr(
        "rtsp_capture.dahua_api.requests.get",
        lambda *args, **kwargs: ErrorResponse(),
    )
    ctrl = DahuaController()
    ok, msg = ctrl._cgi_request("192.168.3.183", "configManager.cgi", {"action": "setConfig"})
    assert not ok
    assert "Invalid parameter" in msg


def test_web_app_batch_reports_partial_failure(dummy_config: Path, monkeypatch):
    mgr = CaptureManager(dummy_config, mock_mode=True)
    app = create_app(mgr)
    batch_fn = next(
        route.endpoint
        for route in app.routes
        if route.path == "/api/cameras/batch-config" and "POST" in route.methods
    )

    def partial_failure(camera_ids, **kwargs):
        return {
            cid: (cid != "cam02", "OK" if cid != "cam02" else "readback mismatch")
            for cid in camera_ids
        }

    monkeypatch.setattr(mgr.dahua, "batch_set_white_balance", partial_failure)
    result = batch_fn(
        BatchConfigRequest(
            white_balance=WhiteBalanceRequest(style="Manual", gain_red=55, gain_blue=48),
        )
    )
    assert result["ok"] is False
    assert result["results"]["white_balance"]["cam02"] == {
        "ok": False,
        "message": "readback mismatch",
    }
