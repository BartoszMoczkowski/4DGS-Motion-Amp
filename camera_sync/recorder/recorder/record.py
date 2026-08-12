"""
recorder.record

FFmpeg-based multi-camera recorder.  Key design points:

  - Uses stream-copy (`-c copy`) so CPU load stays minimal even with
    16 simultaneous RTSP feeds.
  - Preserves original RTP timestamps via `-copyts` and writes them
    into the MKV container.
  - Produces a sidecar `recording_manifest.json` that ties each file
    back to its camera ID and RTSP URL.

Typical usage:

    recorder --config cameras.yaml --out ./recordings/run_001

CLI arguments
-------------
--config, -c    YAML file listing camera IDs and RTSP URLs
--out, -o       Output directory (created if missing)
--duration, -d  Max recording duration in seconds (optional)
--transport     RTSP transport: tcp (default) or udp
--timeout       Per-camera startup timeout in seconds (default: 15)

Camera YAML schema
------------------
cameras:
  cam01:
    url: rtsp://192.168.1.101:554/stream
  cam02:
    url: rtsp://192.168.1.102:554/stream
  # ... up to 16 entries
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# YAML is imported lazily so the module stays importable even when pyyaml
# is absent (e.g. in sandbox test environments).
# ---------------------------------------------------------------------------

def _import_yaml() -> Any:
    try:
        import yaml
        return yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to load camera configs.  "
            "Install the package with:  uv sync --package camera-sync-recorder"
        ) from exc


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    camera_id: str
    url: str


@dataclass
class RecordingSession:
    run_id: str
    start_wall_time: str
    output_dir: Path
    transport: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)
    exit_code: int = 0


# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------

def build_ffmpeg_command(
    rtsp_url: str,
    output_path: Path,
    camera_id: str,
    transport: str = "tcp",
) -> list[str]:
    """Return the FFmpeg argument list for a single camera."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-rtsp_transport", transport,
        "-fflags", "+genpts",
        "-i", rtsp_url,
        "-c", "copy",
        "-f", "matroska",
        "-copyts",
        "-start_at_zero",
        "-metadata", f"camera_id={camera_id}",
        "-metadata", f"rtsp_url={rtsp_url}",
        str(output_path),
    ]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_camera_config(path: Path) -> dict[str, CameraConfig]:
    """Load a YAML camera definition file.

    Expected schema::

        cameras:
          cam01:
            url: rtsp://192.168.1.101:554/stream
          cam02:
            url: rtsp://192.168.1.102:554/stream

    Returns a mapping ``camera_id -> CameraConfig``.
    """
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    raw = data.get("cameras", {})
    if not raw:
        raise ValueError(f"No 'cameras' section found in config file: {path}")

    result: dict[str, CameraConfig] = {}
    for camera_id, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(
                f"Camera '{camera_id}' must be a mapping with at least a 'url' key."
            )
        url = info.get("url")
        if not url:
            raise ValueError(f"Camera '{camera_id}' is missing the 'url' field.")
        result[camera_id] = CameraConfig(camera_id=camera_id, url=url)

    return result


# ---------------------------------------------------------------------------
# Recording orchestration
# ---------------------------------------------------------------------------

def _kill_all(session: RecordingSession, sig: int = signal.SIGTERM) -> None:
    """Forward *sig* to every running FFmpeg process."""
    for camera_id, proc in list(session.processes.items()):
        if proc.poll() is None:
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass


def _wait_all(session: RecordingSession, timeout_sec: float = 30.0) -> None:
    """Block until every FFmpeg process has exited or *timeout_sec* elapses."""
    deadline = time.monotonic() + timeout_sec
    for camera_id, proc in list(session.processes.items()):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[WARN] Camera '{camera_id}' did not exit in time; forcing kill.",
                  file=sys.stderr)
            proc.kill()
            proc.wait()


def _build_manifest(session: RecordingSession) -> dict:
    """Assemble the recording_manifest.json payload."""
    camera_entries: dict[str, dict] = {}
    for camera_id, cfg in session.cameras.items():
        proc = session.processes.get(camera_id)
        entry: dict[str, Any] = {
            "rtsp_url": cfg.url,
            "file": f"{camera_id}.mkv",
        }
        if proc is not None:
            entry["pid"] = proc.pid
            entry["returncode"] = proc.returncode
        camera_entries[camera_id] = entry

    return {
        "run_id": session.run_id,
        "start_wall_time": session.start_wall_time,
        "transport": session.transport,
        "camera_count": len(session.cameras),
        "cameras": camera_entries,
    }


def _write_manifest(session: RecordingSession) -> Path:
    """Write the sidecar manifest and return its path."""
    manifest = _build_manifest(session)
    manifest_path = session.output_dir / "recording_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def run_recording(
    config_path: Path,
    output_dir: Path,
    transport: str = "tcp",
    max_duration_sec: float | None = None,
    startup_timeout_sec: float = 15.0,
) -> RecordingSession:
    """Launch one FFmpeg process per camera and wait for completion.

    Parameters
    ----------
    config_path
        YAML file with camera definitions.
    output_dir
        Directory that will receive MKV files and the manifest.
    transport
        RTSP transport protocol (``tcp`` or ``udp``).
    max_duration_sec
        If given, FFmpeg will stop after this many seconds.  If ``None`` the
        recording runs until the user interrupts it.
    startup_timeout_sec
        Maximum time to wait for FFmpeg to open each RTSP stream before
        considering it a failure.

    Returns
    -------
    RecordingSession
        Populated session object; inspect ``.exit_code`` and ``.processes``
        to determine success.
    """
    # ---- load config -------------------------------------------------------
    cameras = load_camera_config(config_path)
    if not cameras:
        raise ValueError("No cameras defined in config file.")
    if len(cameras) > 16:
        print(f"[WARN] Config defines {len(cameras)} cameras; "
              f"16 is the recommended maximum.", file=sys.stderr)

    # ---- prepare output directory ------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    start_wall_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    session = RecordingSession(
        run_id=run_id,
        start_wall_time=start_wall_time,
        output_dir=output_dir,
        transport=transport,
        cameras=cameras,
    )

    # ---- per-camera log files ----------------------------------------------
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # ---- signal / interrupt handling ---------------------------------------
    shutdown_event = threading.Event()
    original_sigint = signal.signal(signal.SIGINT, lambda _s, _f: shutdown_event.set())
    try:
        # ---- spawn FFmpeg processes ----------------------------------------
        print(f"[RECORDER] Starting {len(cameras)} camera(s)…")
        for camera_id, cfg in cameras.items():
            out_file = output_dir / f"{camera_id}.mkv"
            cmd = build_ffmpeg_command(
                rtsp_url=cfg.url,
                output_path=out_file,
                camera_id=camera_id,
                transport=transport,
            )
            # Append -t <duration> right before the output file if requested.
            if max_duration_sec is not None:
                cmd = cmd[:-1] + ["-t", str(max_duration_sec)] + cmd[-1:]

            log_path = log_dir / f"{camera_id}.log"
            log_file = log_path.open("w", encoding="utf-8")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
            )
            session.processes[camera_id] = proc
            print(f"  [{camera_id}] pid={proc.pid}  -> {out_file}")

        # ---- wait for startup (basic health check) -------------------------
        print(f"[RECORDER] Waiting up to {startup_timeout_sec}s for streams to open…")
        time.sleep(min(2.0, startup_timeout_sec))   # give FFmpeg time to connect
        for camera_id, proc in list(session.processes.items()):
            if proc.poll() is not None:
                print(f"[ERROR] Camera '{camera_id}' exited early "
                      f"(code={proc.returncode}).  Check logs/{camera_id}.log",
                      file=sys.stderr)
                session.exit_code = 1

        if session.exit_code != 0:
            print("[RECORDER] One or more cameras failed to start.  Aborting.",
                  file=sys.stderr)
            _kill_all(session)
            _wait_all(session)
            return session

        # ---- main wait loop ------------------------------------------------
        print("[RECORDER] Recording.  Press Ctrl+C to stop.")
        while not shutdown_event.is_set():
            # Check whether any camera has finished (e.g. due to -t timeout).
            all_done = all(
                proc.poll() is not None for proc in session.processes.values()
            )
            if all_done:
                print("[RECORDER] All cameras finished.")
                break
            time.sleep(0.5)

        # ---- graceful shutdown ---------------------------------------------
        if shutdown_event.is_set():
            print("\n[RECORDER] Shutdown signal received.  Stopping cameras…")
        _kill_all(session, signal.SIGTERM)
        _wait_all(session)

    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # ---- write manifest ----------------------------------------------------
    manifest_path = _write_manifest(session)
    print(f"[RECORDER] Manifest written: {manifest_path}")
    return session


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recorder",
        description="Multi-camera RTSP recorder via FFmpeg",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML file with camera definitions",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        required=True,
        help="Output directory for MKV files and manifest",
    )
    parser.add_argument(
        "--transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="RTSP transport protocol (default: tcp)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        metavar="SEC",
        help="Maximum recording duration in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        metavar="SEC",
        help="Per-camera startup timeout (default: 15)",
    )
    args = parser.parse_args(argv)

    # Verify FFmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] FFmpeg not found in PATH.  Please install FFmpeg.",
              file=sys.stderr)
        return 2

    try:
        session = run_recording(
            config_path=args.config,
            output_dir=args.out,
            transport=args.transport,
            max_duration_sec=args.duration,
            startup_timeout_sec=args.timeout,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # Summarise
    ok = sum(
        1 for p in session.processes.values() if p.returncode == 0
    )
    total = len(session.processes)
    print(f"[RECORDER] Done.  {ok}/{total} camera(s) exited cleanly.")
    return session.exit_code


if __name__ == "__main__":
    sys.exit(main())
