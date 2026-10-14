"""rtsp_capture.manager

Central session manager for multi-camera capture, RTCP synchronization,
probe diagnostics, and Dahua camera hardware controls.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dahua_api import DahuaController
from .probe import probe_camera
from .recorder import (
    CameraConfig,
    CameraRecorder,
    RecordingSession,
    _write_manifest,
    load_camera_config,
)
from .sync_extract import extract_synced_frames


class CaptureManager:
    """Coordinates camera recording, diagnostics, extraction, and hardware tuning."""

    def __init__(
        self,
        config_path: Path,
        mock_mode: bool = False,
    ):
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.cameras: dict[str, CameraConfig] = {}
        self.dahua: DahuaController | None = None
        self._lock = threading.Lock()

        # State machine: "IDLE", "RECORDING", "PROBING", "EXTRACTING"
        self.state = "IDLE"
        self.current_session: RecordingSession | None = None
        self.record_event = threading.Event()
        self.stop_event = threading.Event()
        self.recording_thread: threading.Thread | None = None
        self.last_probe_results: list[dict] = []
        self.last_extract_result: dict | None = None
        self.active_error: str | None = None
        self.auto_extract = False
        self.session_duration: float | None = None

        self.reload_config()

    def reload_config(self) -> None:
        """Reload camera definitions and initialize Dahua controller."""
        with self._lock:
            self.cameras = load_camera_config(self.config_path)

            # Read credentials for Dahua controller
            import yaml
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            creds = data.get("credentials") or {}
            user = os.environ.get("RTSP_CAM_USER") or creds.get("user", "admin")
            password = os.environ.get("RTSP_CAM_PASSWORD") or creds.get("password", "")

            self.dahua = DahuaController(
                user=user,
                password=password,
                mock_mode=self.mock_mode,
            )

            # Register camera IPs
            raw_cams = data.get("cameras", {})
            for cid, info in raw_cams.items():
                if isinstance(info, dict):
                    ip = info.get("ip")
                    if not ip and info.get("url"):
                        # Extract host from url if needed
                        import urllib.parse
                        parsed = urllib.parse.urlparse(info["url"])
                        ip = parsed.hostname or "127.0.0.1"
                    if ip:
                        self.dahua.register_camera(cid, ip)

    # ------------------------------------------------------------------
    # Telemetry & Status
    # ------------------------------------------------------------------

    def get_telemetry(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed = 0.0
            if self.current_session and self.current_session.wall_start > 0:
                end = self.current_session.wall_stop if self.current_session.wall_stop > 0 else now
                elapsed = max(0.0, end - self.current_session.wall_start)

            cam_stats = {}
            total_frames = 0
            total_srs = 0
            total_gaps = 0

            # Pull stats from recording threads if recording
            if self.current_session and self.current_session.threads:
                for th in self.current_session.threads:
                    cid = th.cfg.camera_id
                    st = th.stats
                    total_frames += st.get("frames", 0)
                    total_srs += st.get("srs", 0)
                    total_gaps += st.get("gaps", 0)
                    cam_stats[cid] = {
                        "state": th.state,
                        "frames": st.get("frames", 0),
                        "srs": st.get("srs", 0),
                        "gaps": st.get("gaps", 0),
                        "reconnects": st.get("reconnects", 0),
                        "error": th.error,
                    }

            # Merge with dahua states
            dahua_states = {}
            if self.dahua:
                for cid, c_state in self.dahua.get_all_states().items():
                    dahua_states[cid] = {
                        "camera_id": cid,
                        "ip": c_state.ip,
                        "online": c_state.online or self.mock_mode,
                        "color": asdict(c_state.color),
                        "exposure": asdict(c_state.exposure),
                        "light": asdict(c_state.light),
                        "error": c_state.error,
                        "stream": cam_stats.get(cid, {"state": "IDLE"}),
                    }

            return {
                "state": self.state,
                "mock_mode": self.mock_mode,
                "camera_count": len(self.cameras),
                "elapsed_sec": round(elapsed, 1),
                "total_frames": total_frames,
                "total_srs": total_srs,
                "total_gaps": total_gaps,
                "session": {
                    "run_id": self.current_session.run_id if self.current_session else None,
                    "output_dir": str(self.current_session.output_dir) if self.current_session else None,
                } if self.current_session else None,
                "cameras": dahua_states,
                "last_extract": self.last_extract_result,
                "error": self.active_error,
            }

    # ------------------------------------------------------------------
    # Recording Lifecycle
    # ------------------------------------------------------------------

    def start_recording(
        self,
        output_dir: Path,
        session_name: str | None = None,
        duration_sec: float | None = None,
        auto_extract: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.state == "RECORDING":
                raise RuntimeError("Recording session is already running.")
            if not self.cameras:
                raise ValueError("No cameras configured.")

            run_id = session_name or time.strftime("run_%Y%m%d_%H%M%S")
            target_dir = output_dir / run_id
            target_dir.mkdir(parents=True, exist_ok=True)

            self.session_duration = duration_sec
            self.auto_extract = auto_extract
            self.current_session = RecordingSession(
                run_id=run_id,
                output_dir=target_dir,
                cameras=self.cameras,
            )
            self.record_event.clear()
            self.stop_event.clear()
            self.state = "RECORDING"
            self.active_error = None

        def _recording_worker() -> None:
            try:
                # In mock mode, emulate frame generation instead of real RTSP sockets
                if self.mock_mode:
                    self._run_mock_recording()
                    return

                for _, cfg in self.cameras.items():
                    thread = CameraRecorder(cfg, target_dir, self.record_event, self.stop_event)
                    self.current_session.threads.append(thread)
                    thread.start()

                # Start writing bitstreams
                time.sleep(0.5)
                self.record_event.set()
                self.current_session.wall_start = time.time()

                # Wait for duration or stop event
                while not self.stop_event.is_set():
                    if (
                        self.session_duration is not None
                        and (time.time() - self.current_session.wall_start) >= self.session_duration
                    ):
                        break
                    self.stop_event.wait(0.2)

            except Exception as exc:
                self.active_error = str(exc)
            finally:
                self.stop_recording()

        self.recording_thread = threading.Thread(target=_recording_worker, daemon=True)
        self.recording_thread.start()
        return {"run_id": run_id, "output_dir": str(target_dir)}

    def _run_mock_recording(self) -> None:
        """Simulate recording thread updates when cameras are offline."""
        self.current_session.wall_start = time.time()
        self.record_event.set()
        start = time.time()
        fps = 25.0
        frame_interval = 1.0 / fps

        # Synthetic recorder stubs
        class MockThread:
            def __init__(self, cid: str):
                self.cfg = CameraConfig(cid, f"mock://{cid}")
                self.state = "RECORDING"
                self.codec = "H264"
                self.clock_rate = 90000
                self.stats = {"frames": 0, "srs": 0, "gaps": 0, "reconnects": 0}
                self.bitstream_name = f"{cid}.h264"
                self.error = None

            def join(self, timeout=None):
                pass

        self.current_session.threads = [MockThread(cid) for cid in self.cameras]

        next_sr = start + 1.0
        while not self.stop_event.is_set():
            t = time.time()
            if self.session_duration and (t - start) >= self.session_duration:
                break
            for th in self.current_session.threads:
                th.stats["frames"] += 1
                if t >= next_sr:
                    th.stats["srs"] += 1
            if t >= next_sr:
                next_sr = t + 2.0  # SR every 2 sec
            time.sleep(frame_interval)

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            if self.state != "RECORDING" and not self.current_session:
                return {"status": "not_recording"}
            self.stop_event.set()

        session = self.current_session
        if session:
            for thread in session.threads:
                thread.join(timeout=5.0)

            if session.wall_stop == 0.0:
                session.wall_stop = time.time()
            manifest_path = _write_manifest(session)

            out_dir = session.output_dir
            auto_ex = self.auto_extract
        else:
            manifest_path = None
            out_dir = None
            auto_ex = False

        with self._lock:
            self.state = "IDLE"

        # Auto-extract if requested and not in mock mode
        if auto_ex and out_dir and not self.mock_mode:
            try:
                self.run_extract(out_dir, out_dir / "synced_output")
            except Exception as exc:
                self.active_error = f"Auto-extract failed: {exc}"

        return {
            "status": "stopped",
            "run_id": session.run_id if session else None,
            "manifest": str(manifest_path) if manifest_path else None,
        }

    # ------------------------------------------------------------------
    # Diagnostics & Probing
    # ------------------------------------------------------------------

    def probe_all(self, duration: float = 4.0) -> list[dict]:
        with self._lock:
            self.state = "PROBING"
        try:
            if self.mock_mode:
                results = []
                for cid, cfg in self.cameras.items():
                    results.append({
                        "camera_id": cid,
                        "url": cfg.url,
                        "ok": True,
                        "error": None,
                        "codec": "H264",
                        "clock_rate": 90000,
                        "resolution": "2688x1520",
                        "frames": int(duration * 25),
                        "srs": max(2, int(duration // 2)),
                    })
                self.last_probe_results = results
                return results

            results: list[dict] = []
            threads = []
            for cid, cfg in self.cameras.items():
                t = threading.Thread(
                    target=lambda c=cid, u=cfg.url: results.append(probe_camera(c, u, duration)),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=duration + 5.0)
            self.last_probe_results = sorted(results, key=lambda x: x["camera_id"])
            return self.last_probe_results
        finally:
            with self._lock:
                self.state = "IDLE"

    # ------------------------------------------------------------------
    # Frame Extraction
    # ------------------------------------------------------------------

    def run_extract(
        self,
        recording_dir: Path,
        out_dir: Path,
        fps: float | None = None,
        tolerance: float | None = None,
        quality: int = 2,
    ) -> dict[str, Any]:
        with self._lock:
            self.state = "EXTRACTING"
        try:
            res = extract_synced_frames(
                recording_dir=recording_dir,
                out_dir=out_dir,
                fps=fps,
                tolerance=tolerance,
                quality=quality,
            )
            self.last_extract_result = res
            return res
        finally:
            with self._lock:
                self.state = "IDLE"
