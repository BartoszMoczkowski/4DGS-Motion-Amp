"""rtsp_capture.recorder

Thread-per-camera RTSP recorder writing raw Annex-B bitstreams plus
per-frame and per-Sender-Report JSONL sidecars.

Threads connect and PLAY immediately (so RTCP SRs accumulate), but only
write to disk while the shared record event is set:

    Enter   -- start writing (shared event flipped in the main thread)
    Ctrl+C  -- stop writing, TEARDOWN, close files, write manifest

Camera YAML schema
------------------
credentials:
  user: admin
  password: "secret"          # env RTSP_CAM_USER / RTSP_CAM_PASSWORD override
cameras:
  cam01:
    ip: 192.168.3.183
    subtype: 0                # 0 = main stream, 1 = sub stream
  cam02:
    url: rtsp://user:pass@192.168.3.159:554/cam/realmonitor?channel=1&subtype=0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rtsp_client import RtspClient, redact_url

URL_TEMPLATE = "rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype={subtype}"

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0


def _import_yaml() -> Any:
    try:
        import yaml

        return yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to load camera configs.  "
            "Install the package with:  uv sync --package camera-sync-rtcp"
        ) from exc


@dataclass
class CameraConfig:
    camera_id: str
    url: str


def load_camera_config(path: Path) -> dict[str, CameraConfig]:
    """Load a YAML camera definition file.

    Expected schema::

        credentials: {user: ..., password: ...}
        cameras:
          cam01: {ip: 192.168.3.183, subtype: 0}
          cam02: {url: rtsp://user:pass@host:554/custom/path}

    Entries with ``ip`` get the Dahua ``realmonitor`` URL built from the
    ``credentials`` section; entries may instead give a full ``url``.
    ``RTSP_CAM_USER`` / ``RTSP_CAM_PASSWORD`` environment variables
    override the YAML credentials.
    """
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    creds = data.get("credentials") or {}
    user = os.environ.get("RTSP_CAM_USER") or creds.get("user")
    password = os.environ.get("RTSP_CAM_PASSWORD") or creds.get("password")

    raw = data.get("cameras", {})
    if not raw:
        raise ValueError(f"No 'cameras' section found in config file: {path}")

    result: dict[str, CameraConfig] = {}
    for camera_id, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(
                f"Camera '{camera_id}' must be a mapping with 'ip' or 'url'."
            )
        url = info.get("url")
        if not url:
            ip = info.get("ip")
            if not ip:
                raise ValueError(f"Camera '{camera_id}' needs an 'ip' or a 'url'.")
            if not user or not password:
                raise ValueError(
                    f"Camera '{camera_id}' uses the URL template but no "
                    "credentials are configured (YAML 'credentials' section or "
                    "RTSP_CAM_USER / RTSP_CAM_PASSWORD env vars)."
                )
            subtype = info.get("subtype", 0)
            url = URL_TEMPLATE.format(user=user, password=password, ip=ip, subtype=subtype)
        result[camera_id] = CameraConfig(camera_id=camera_id, url=url)
    return result


class CameraRecorder(threading.Thread):
    """One camera: connect, PLAY, and write while the record event is set.

    States: CONNECTING -> ARMED -> RECORDING, with RECONNECTING on loss
    (exponential backoff up to 30 s) and STOPPED at the end.
    """

    def __init__(
        self,
        cfg: CameraConfig,
        output_dir: Path,
        record_event: threading.Event,
        stop_event: threading.Event,
    ):
        super().__init__(name=f"rec-{cfg.camera_id}", daemon=True)
        self.cfg = cfg
        self.output_dir = output_dir
        self.record_event = record_event
        self.stop_event = stop_event
        self.state = "CONNECTING"
        self.codec: str | None = None
        self.clock_rate: int = 90000
        self.stats: dict[str, int] = {
            "frames": 0,
            "srs": 0,
            "packets": 0,
            "gaps": 0,
            "reconnects": 0,
        }
        self.error: str | None = None
        self._frame_idx = 0
        self._bitstream = None
        self._frames_log = None
        self._rtcp_log = None
        self.bitstream_name: str | None = None

    # ------------------------------------------------------------------
    # thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        backoff = BACKOFF_INITIAL
        while not self.stop_event.is_set():
            client = RtspClient(self.cfg.url)
            conn_stats: dict[str, int] = {}
            try:
                self.state = "CONNECTING"
                client.connect()
                client.describe()
                client.setup()
                client.play()
                self.codec = client.codec
                self.clock_rate = client.clock_rate
                backoff = BACKOFF_INITIAL
                print(f"[{self.cfg.camera_id}] connected "
                      f"({self.codec} @ {self.clock_rate} Hz)", flush=True)
                client.receive_loop(
                    self._on_frame, self._on_sr, self.stop_event, stats=conn_stats
                )
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self.error = str(exc)
                self.state = "RECONNECTING"
                self.stats["reconnects"] += 1
                print(
                    f"[{self.cfg.camera_id}] connection lost ({exc}); "
                    f"retrying in {backoff:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            finally:
                self.stats["packets"] += conn_stats.get("packets", 0)
                self.stats["gaps"] += conn_stats.get("gaps", 0)
                client.teardown()
                client.close()
        self.state = "STOPPED"
        self._close_files()

    # ------------------------------------------------------------------
    # callbacks (called from this thread's receive loop)
    # ------------------------------------------------------------------

    def _on_frame(self, data: bytes, rtp_ts: int, seq_start: int, arrival: float) -> None:
        if not self.record_event.is_set():
            return
        self._ensure_files()
        self._frame_idx += 1
        self._bitstream.write(data)
        self._frames_log.write(
            json.dumps(
                {
                    "frame_idx": self._frame_idx,
                    "rtp_ts": rtp_ts,
                    "seq_start": seq_start,
                    "n_bytes": len(data),
                    "arrival_mono": arrival,
                }
            )
            + "\n"
        )
        self.stats["frames"] += 1

    def _on_sr(self, ntp_unix: float, rtp_ts: int, arrival: float) -> None:
        if not self.record_event.is_set():
            return
        self._ensure_files()
        self._rtcp_log.write(
            json.dumps(
                {"ntp_unix": ntp_unix, "rtp_ts": rtp_ts, "arrival_mono": arrival}
            )
            + "\n"
        )
        self.stats["srs"] += 1

    # ------------------------------------------------------------------
    # file handling
    # ------------------------------------------------------------------

    def _ensure_files(self) -> None:
        if self._bitstream is not None:
            return
        ext = ".h265" if (self.codec or "").upper() in ("H265", "HEVC") else ".h264"
        self.bitstream_name = f"{self.cfg.camera_id}{ext}"
        self._bitstream = open(
            self.output_dir / self.bitstream_name, "wb", buffering=1 << 20
        )
        self._frames_log = open(
            self.output_dir / f"{self.cfg.camera_id}_frames.jsonl",
            "w",
            encoding="utf-8",
            buffering=1 << 16,
        )
        self._rtcp_log = open(
            self.output_dir / f"{self.cfg.camera_id}_rtcp.jsonl",
            "w",
            encoding="utf-8",
            buffering=1 << 16,
        )

    def _close_files(self) -> None:
        for fh in (self._bitstream, self._frames_log, self._rtcp_log):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        self._bitstream = self._frames_log = self._rtcp_log = None


@dataclass
class RecordingSession:
    run_id: str
    output_dir: Path
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    threads: list[CameraRecorder] = field(default_factory=list)
    wall_start: float = 0.0
    wall_stop: float = 0.0
    exit_code: int = 0


def _write_manifest(session: RecordingSession) -> Path:
    cameras: dict[str, Any] = {}
    for thread in session.threads:
        entry: dict[str, Any] = {
            "url": redact_url(thread.cfg.url),
            "codec": thread.codec,
            "clock_rate": thread.clock_rate,
            "stats": thread.stats,
        }
        if thread.bitstream_name:
            entry["bitstream"] = thread.bitstream_name
            entry["frames_log"] = f"{thread.cfg.camera_id}_frames.jsonl"
            entry["rtcp_log"] = f"{thread.cfg.camera_id}_rtcp.jsonl"
        if thread.error:
            entry["last_error"] = thread.error
        cameras[thread.cfg.camera_id] = entry

    manifest = {
        "run_id": session.run_id,
        "wall_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(session.wall_start)),
        "wall_stop": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(session.wall_stop)),
        "record_start_unix": session.wall_start,
        "record_stop_unix": session.wall_stop,
        "camera_count": len(session.cameras),
        "cameras": cameras,
    }
    manifest_path = session.output_dir / "recording_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def run_recording(
    config_path: Path,
    output_dir: Path,
    max_duration_sec: float | None = None,
) -> RecordingSession:
    """Connect all cameras, wait for Enter to start writing, stop on Ctrl+C."""
    cameras = load_camera_config(config_path)
    if not cameras:
        raise ValueError("No cameras defined in config file.")
    if len(cameras) > 20:
        print(f"[WARN] Config defines {len(cameras)} cameras; "
              f"20 is the recommended maximum.", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    session = RecordingSession(
        run_id=time.strftime("%Y%m%d_%H%M%S"),
        output_dir=output_dir,
        cameras=cameras,
    )

    record_event = threading.Event()
    stop_event = threading.Event()
    try:
        print(f"[RECORDER] Connecting {len(cameras)} camera(s)…")
        for camera_id, cfg in cameras.items():
            thread = CameraRecorder(cfg, output_dir, record_event, stop_event)
            session.threads.append(thread)
            thread.start()

        print("[RECORDER] Press Enter to start recording (Ctrl+C to abort).")
        try:
            input()
            record_event.set()
        except (EOFError, KeyboardInterrupt):
            stop_event.set()

        if stop_event.is_set():
            print("[RECORDER] Aborted before recording started.", file=sys.stderr)
            session.exit_code = 1
        else:
            session.wall_start = time.time()
            print("[RECORDER] Recording.  Press Ctrl+C to stop.")
            try:
                while not stop_event.is_set():
                    if max_duration_sec is not None and (
                        time.time() - session.wall_start >= max_duration_sec
                    ):
                        break
                    stop_event.wait(0.5)
            except KeyboardInterrupt:
                pass
            stop_event.set()
            session.wall_stop = time.time()
    finally:
        stop_event.set()

    print("\n[RECORDER] Stopping cameras…")
    for thread in session.threads:
        thread.join(timeout=15.0)

    if session.wall_stop == 0.0:
        session.wall_stop = time.time()
    manifest_path = _write_manifest(session)
    print(f"[RECORDER] Manifest written: {manifest_path}")
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtsp-record",
        description="RTCP-aware multi-camera RTSP recorder (raw Annex-B + JSONL sidecars)",
    )
    parser.add_argument(
        "--config", "-c", type=Path, required=True,
        help="YAML file with camera definitions (see cameras.yaml.example)",
    )
    parser.add_argument(
        "--out", "-o", type=Path, required=True,
        help="Output directory for bitstreams, JSONL logs and the manifest",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=None, metavar="SEC",
        help="Maximum recording duration in seconds (default: until Ctrl+C)",
    )
    args = parser.parse_args(argv)

    try:
        session = run_recording(
            config_path=args.config,
            output_dir=args.out,
            max_duration_sec=args.duration,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for thread in session.threads:
        s = thread.stats
        print(
            f"  [{thread.cfg.camera_id}] frames={s['frames']} srs={s['srs']} "
            f"gaps={s['gaps']} reconnects={s['reconnects']}"
        )
    return session.exit_code


if __name__ == "__main__":
    sys.exit(main())
