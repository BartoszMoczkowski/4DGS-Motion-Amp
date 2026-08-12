"""
camera-sync-recorder

Lightweight FFmpeg wrapper for recording multiple RTSP streams in parallel
to timestamped MKV files.  Designed for up to 16 IP cameras that do *not*
emit RTCP Sender Reports.

Main entry point: ``recorder.record:main`` (``recorder`` CLI).
"""

__version__ = "0.1.0"

from recorder.record import (
    CameraConfig,
    RecordingSession,
    build_ffmpeg_command,
    load_camera_config,
    run_recording,
)

__all__ = [
    "CameraConfig",
    "RecordingSession",
    "build_ffmpeg_command",
    "load_camera_config",
    "run_recording",
]
