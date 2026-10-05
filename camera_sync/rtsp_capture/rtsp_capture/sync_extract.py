"""rtsp_capture.sync_extract

Post-hoc synced frame extraction from an ``rtsp-record`` output directory.

Each frame's RTP timestamp is mapped to wall-clock NTP time by
piecewise-linear interpolation between the camera's RTCP Sender Reports
(extrapolating from the nearest SR at the edges using the SDP clock
rate).  Cameras are aligned on a reference timeline; for every instant
the nearest frame of each camera is taken, and instants where any
camera's nearest frame is further than ``--tolerance`` away are dropped.

Frames are decoded to JPEG with the system ``ffmpeg`` binary (not managed
by uv).  Output::

    <out>/synced/frame_000001/cam01.jpg
    <out>/synced/frame_000001/cam02.jpg
    ...
    <out>/sync_manifest.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CLOCK_RATE = 90000
DEFAULT_FPS_FALLBACK = 25.0


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rtp_to_ntp(sr_points: list[tuple[int, float]], clock_rate: float = DEFAULT_CLOCK_RATE):
    """Build an RTP-timestamp -> NTP-unix-seconds mapping from Sender Reports.

    *sr_points* is a list of ``(rtp_ts, ntp_unix)`` pairs (64-bit extended
    RTP timestamps).  Between bracketing SRs the mapping is linear; at the
    edges it is extrapolated from the nearest SR using *clock_rate*.
    """
    points = sorted(sr_points)
    if not points:
        raise ValueError("no sender reports available for RTP->NTP mapping")
    ts_list = [p[0] for p in points]

    def map_ts(ts: int) -> float:
        if len(points) == 1:
            t0, n0 = points[0]
            return n0 + (ts - t0) / clock_rate
        idx = bisect.bisect_left(ts_list, ts)
        if idx == 0:
            t0, n0 = points[0]
            return n0 + (ts - t0) / clock_rate
        if idx >= len(points):
            t1, n1 = points[-1]
            return n1 + (ts - t1) / clock_rate
        t0, n0 = points[idx - 1]
        t1, n1 = points[idx]
        if t1 == t0:
            return n0
        return n0 + (n1 - n0) * (ts - t0) / (t1 - t0)

    return map_ts


def _load_cameras(recording_dir: Path) -> dict[str, dict]:
    with (recording_dir / "recording_manifest.json").open("r", encoding="utf-8") as fh:
        rec_manifest = json.load(fh)
    cameras: dict[str, dict] = {}
    for camera_id, info in rec_manifest["cameras"].items():
        if "frames_log" not in info:
            print(f"[EXTRACT] {camera_id}: no recorded data, skipping.", file=sys.stderr)
            continue
        frames = load_jsonl(recording_dir / info["frames_log"])
        srs = load_jsonl(recording_dir / info["rtcp_log"])
        clock_rate = info.get("clock_rate") or DEFAULT_CLOCK_RATE
        if len(srs) < 2:
            print(f"[WARN] {camera_id}: only {len(srs)} Sender Report(s); "
                  f"sync confidence is low.", file=sys.stderr)
        if not frames or not srs:
            print(f"[WARN] {camera_id}: empty frames/SR log, skipping.", file=sys.stderr)
            continue
        mapper = build_rtp_to_ntp(
            [(s["rtp_ts"], s["ntp_unix"]) for s in srs], clock_rate
        )
        for frame in frames:
            frame["ntp_unix"] = mapper(frame["rtp_ts"])
        cameras[camera_id] = {
            "frames": frames,
            "bitstream": recording_dir / info["bitstream"],
            "srs": len(srs),
        }
    return cameras


def _estimate_fps(frames: list[dict]) -> float:
    times = [f["ntp_unix"] for f in frames]
    if len(times) < 2:
        return DEFAULT_FPS_FALLBACK
    diffs = sorted(b - a for a, b in zip(times, times[1:]))
    median = diffs[len(diffs) // 2]
    return 1.0 / median if median > 0 else DEFAULT_FPS_FALLBACK


def _nearest_frame(frames: list[dict], times: list[float], t: float) -> tuple[dict, float]:
    idx = bisect.bisect_left(times, t)
    best = None
    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(frames):
            residual = frames[candidate]["ntp_unix"] - t
            if best is None or abs(residual) < abs(best[1]):
                best = (frames[candidate], residual)
    return best


def _decode_camera(
    ffmpeg: str, bitstream: Path, tmp_dir: Path, camera_id: str, quality: int
) -> Path:
    out_dir = tmp_dir / camera_id
    out_dir.mkdir(parents=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(bitstream),
        "-q:v", str(quality),
        "-start_number", "1",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return out_dir


def extract(
    recording_dir: Path,
    out_dir: Path,
    fps: float | None = None,
    tolerance: float | None = None,
    ref_camera: str | None = None,
    quality: int = 2,
    ffmpeg: str = "ffmpeg",
) -> dict:
    """Build a synced frame series from an rtsp-record output directory."""
    if shutil.which(ffmpeg) is None:
        raise RuntimeError(f"ffmpeg binary not found: {ffmpeg!r}")

    cameras = _load_cameras(recording_dir)
    if not cameras:
        raise RuntimeError("no usable camera data in recording directory")
    ref = ref_camera or next(iter(cameras))
    if ref not in cameras:
        raise ValueError(f"reference camera {ref!r} not in recording")

    uniform_grid = fps is not None
    if fps is None:
        fps = _estimate_fps(cameras[ref]["frames"])
    if tolerance is None:
        tolerance = 0.5 / fps

    times = {cid: [f["ntp_unix"] for f in c["frames"]] for cid, c in cameras.items()}
    if uniform_grid:
        t = max(v[0] for v in times.values())
        end = min(v[-1] for v in times.values())
        timeline = []
        while t <= end:
            timeline.append(t)
            t += 1.0 / fps
    else:
        timeline = times[ref]

    instants = []
    for t in timeline:
        picks = {}
        for camera_id, cam in cameras.items():
            frame, residual = _nearest_frame(cam["frames"], times[camera_id], t)
            if abs(residual) > tolerance:
                picks = None
                break
            picks[camera_id] = {
                "frame_idx": frame["frame_idx"],
                "residual_ms": residual * 1000.0,
            }
        if picks is not None:
            instants.append({"ntp_unix": t, "frames": picks})

    if not instants:
        raise RuntimeError(
            "no synced instants survived the tolerance filter "
            f"(tolerance={tolerance * 1000:.1f} ms)"
        )

    synced_dir = out_dir / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rtsp_extract_") as tmp:
        tmp_dir = Path(tmp)
        decoded: dict[str, Path] = {}
        for camera_id, cam in cameras.items():
            print(f"[EXTRACT] decoding {camera_id} ({cam['bitstream'].name})…")
            decoded[camera_id] = _decode_camera(
                ffmpeg, cam["bitstream"], tmp_dir, camera_id, quality
            )
        for index, instant in enumerate(instants, start=1):
            instant["index"] = index
            frame_dir = synced_dir / f"frame_{index:06d}"
            frame_dir.mkdir()
            for camera_id, pick in instant["frames"].items():
                src = decoded[camera_id] / f"{pick['frame_idx']:06d}.jpg"
                shutil.copyfile(src, frame_dir / f"{camera_id}.jpg")

    residuals: dict[str, dict] = {}
    for camera_id in cameras:
        values = [i["frames"][camera_id]["residual_ms"] for i in instants]
        residuals[camera_id] = {
            "mean_ms": statistics.fmean(values),
            "max_ms": max(abs(v) for v in values),
        }

    sync_manifest = {
        "recording_dir": str(recording_dir),
        "reference_camera": ref,
        "fps": fps,
        "tolerance_ms": tolerance * 1000.0,
        "instants_total": len(timeline),
        "instants_kept": len(instants),
        "sr_counts": {cid: c["srs"] for cid, c in cameras.items()},
        "per_camera_residuals": residuals,
        "instants": instants,
    }
    manifest_path = out_dir / "sync_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(sync_manifest, fh, indent=2)
    print(f"[EXTRACT] {len(instants)}/{len(timeline)} instants kept "
          f"(tolerance {tolerance * 1000:.1f} ms).  Manifest: {manifest_path}")
    return sync_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtsp-extract",
        description="Post-hoc synced frame extraction via RTCP Sender Reports",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Recording directory produced by rtsp-record",
    )
    parser.add_argument(
        "--out", "-o", type=Path, required=True,
        help="Output directory for synced/ and sync_manifest.json",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Uniform grid rate in Hz (default: reference camera's frames)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=None, metavar="SEC",
        help="Max per-camera residual in seconds (default: half frame interval)",
    )
    parser.add_argument(
        "--ref-camera", default=None,
        help="Camera ID providing the reference timeline (default: first)",
    )
    parser.add_argument(
        "--quality", "-q", type=int, default=2,
        help="ffmpeg JPEG quality -q:v (default: 2)",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg",
        help="Path to the ffmpeg binary (default: from PATH)",
    )
    args = parser.parse_args(argv)

    try:
        sync_manifest = extract(
            recording_dir=args.input,
            out_dir=args.out,
            fps=args.fps,
            tolerance=args.tolerance,
            ref_camera=args.ref_camera,
            quality=args.quality,
            ffmpeg=args.ffmpeg,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for camera_id, stats in sync_manifest["per_camera_residuals"].items():
        print(f"  [{camera_id}] residual mean={stats['mean_ms']:+.2f} ms "
              f"max={stats['max_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
