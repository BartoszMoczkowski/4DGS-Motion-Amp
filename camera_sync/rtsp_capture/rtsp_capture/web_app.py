"""rtsp_capture.web_app

FastAPI server providing REST endpoints, SSE telemetry, snapshot proxying,
and static frontend delivery for the Multi-Camera Capture Studio.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .manager import CaptureManager

STATIC_DIR = Path(__file__).parent / "static"


class WhiteBalanceRequest(BaseModel):
    style: str = "Manual"  # "Auto", "Manual", "Custom"
    gain_red: int = Field(default=50, ge=0, le=100)
    gain_blue: int = Field(default=50, ge=0, le=100)
    gain_green: int = Field(default=50, ge=0, le=100)


class ExposureRequest(BaseModel):
    mode: str = "Manual"  # "Auto", "Manual"
    shutter: str = "1/100"
    gain_max: int = Field(default=50, ge=0, le=100)


class PictureRequest(BaseModel):
    brightness: int | None = Field(default=None, ge=0, le=100)
    contrast: int | None = Field(default=None, ge=0, le=100)
    saturation: int | None = Field(default=None, ge=0, le=100)
    sharpness: int | None = Field(default=None, ge=0, le=100)


class BatchConfigRequest(BaseModel):
    cameras: list[str] | None = None  # None = apply to all
    white_balance: WhiteBalanceRequest | None = None
    exposure: ExposureRequest | None = None
    picture: PictureRequest | None = None


class RecordStartRequest(BaseModel):
    session_name: str | None = None
    output_dir: str = "recordings"
    duration_sec: float | None = None
    auto_extract: bool = False


class FlashRequest(BaseModel):
    duration_sec: float = 3.0
    pulses: int = 4


def create_app(manager: CaptureManager) -> FastAPI:
    app = FastAPI(title="Multi-Camera Capture Studio", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Telemetry & Status
    # ------------------------------------------------------------------

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        return manager.get_telemetry()

    @app.get("/api/events")
    async def sse_events(request: Request):
        """Server-Sent Events for real-time telemetry updates (2 Hz)."""
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break
                telem = manager.get_telemetry()
                yield f"data: {json.dumps(telem)}\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Live Preview Snapshots
    # ------------------------------------------------------------------

    @app.get("/api/cameras/{camera_id}/snapshot")
    def get_snapshot(camera_id: str):
        if not manager.dahua:
            raise HTTPException(status_code=500, detail="Controller not initialized")
        jpeg = manager.dahua.get_snapshot(camera_id, subtype=1)
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # ------------------------------------------------------------------
    # Camera Tuning & Identification
    # ------------------------------------------------------------------

    @app.post("/api/cameras/{camera_id}/config")
    def update_camera_config(
        camera_id: str,
        wb: WhiteBalanceRequest | None = None,
        exp: ExposureRequest | None = None,
        pic: PictureRequest | None = None,
    ):
        if not manager.dahua:
            raise HTTPException(status_code=500, detail="Controller not initialized")
        results = {}
        if wb:
            ok, msg = manager.dahua.set_white_balance(
                camera_id,
                style=wb.style,
                gain_red=wb.gain_red,
                gain_blue=wb.gain_blue,
                gain_green=wb.gain_green,
            )
            results["white_balance"] = {"ok": ok, "message": msg}
        if exp:
            ok, msg = manager.dahua.set_exposure(
                camera_id,
                mode=exp.mode,
                shutter=exp.shutter,
                gain_max=exp.gain_max,
            )
            results["exposure"] = {"ok": ok, "message": msg}
        if pic:
            ok, msg = manager.dahua.set_picture(
                camera_id,
                brightness=pic.brightness,
                contrast=pic.contrast,
                saturation=pic.saturation,
                sharpness=pic.sharpness,
            )
            results["picture"] = {"ok": ok, "message": msg}
        return {"camera_id": camera_id, "results": results}

    @app.post("/api/cameras/batch-config")
    def batch_update_config(req: BatchConfigRequest):
        if not manager.dahua:
            raise HTTPException(status_code=500, detail="Controller not initialized")
        cids = req.cameras or list(manager.cameras.keys())
        results = {}
        if req.white_balance:
            res = manager.dahua.batch_set_white_balance(
                cids,
                style=req.white_balance.style,
                gain_red=req.white_balance.gain_red,
                gain_blue=req.white_balance.gain_blue,
                gain_green=req.white_balance.gain_green,
            )
            results["white_balance"] = res
        if req.exposure:
            res = manager.dahua.batch_set_exposure(
                cids,
                mode=req.exposure.mode,
                shutter=req.exposure.shutter,
                gain_max=req.exposure.gain_max,
            )
            results["exposure"] = res
        return {"updated_cameras": cids, "results": results}

    @app.post("/api/cameras/{camera_id}/flash")
    def flash_camera(camera_id: str, req: FlashRequest = Body(default=FlashRequest())):
        if not manager.dahua:
            raise HTTPException(status_code=500, detail="Controller not initialized")
        ok = manager.dahua.flash_camera_light(
            camera_id, duration_sec=req.duration_sec, pulses=req.pulses
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        return {"camera_id": camera_id, "status": "flashing", "duration": req.duration_sec}

    @app.post("/api/cameras/{camera_id}/light")
    def toggle_light(camera_id: str, mode: str = "Manual", brightness: int = 100):
        if not manager.dahua:
            raise HTTPException(status_code=500, detail="Controller not initialized")
        ok, msg = manager.dahua.set_lighting(camera_id, mode=mode, brightness=brightness)
        return {"camera_id": camera_id, "ok": ok, "mode": mode, "message": msg}

    # ------------------------------------------------------------------
    # Probe & Diagnostics
    # ------------------------------------------------------------------

    @app.post("/api/probe")
    def run_probe(duration: float = 4.0):
        results = manager.probe_all(duration=duration)
        return {"probe_results": results}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @app.post("/api/record/start")
    def start_recording(req: RecordStartRequest):
        try:
            out_p = Path(req.output_dir)
            if not out_p.is_absolute():
                out_p = manager.config_path.parent / out_p
            res = manager.start_recording(
                output_dir=out_p,
                session_name=req.session_name,
                duration_sec=req.duration_sec,
                auto_extract=req.auto_extract,
            )
            return {"status": "started", "session": res}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/record/stop")
    def stop_recording():
        res = manager.stop_recording()
        return res

    @app.post("/api/record/extract")
    def extract_frames(recording_dir: str, out_dir: str, fps: float | None = None, tolerance: float | None = None):
        try:
            rec_p = Path(recording_dir)
            out_p = Path(out_dir)
            res = manager.run_extract(rec_p, out_p, fps=fps, tolerance=tolerance)
            return {"status": "extracted", "manifest": res}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index():
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Multi-Camera Capture Studio</h1><p>Frontend loading...</p>")

    return app
