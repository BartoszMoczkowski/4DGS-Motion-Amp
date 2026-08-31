"""
sync_analyzer.analyze

Post-processing pipeline for extracting per-camera time offsets from
recorded MKV files that contain frames of the Sync Display.

Pipeline Architecture:
----------------------
1. Frame Sampling:
   Read sparse frames (e.g. 5-10 fps) from each camera's MKV recording.
2. Visual Detection:
   Locate ArUco markers (IDs 0 and 1) to define the screen ROI and orientation.
   Decode QR code within the ROI (with full-frame fallback) using multi-pass
   pyzbar / OpenCV QRCodeDetector to retrieve JSON payload {"u": <unix_ms>, "s": <seq>}.
3. Outlier Filtering & Time Mapping:
   Validate sequence monotonicity and time continuity. For each camera, collect
   (pts_sec, qr_unix_ms) pairs.
4. Offset & Drift Fitting:
   Fit a robust linear model per camera:
       pts = alpha + beta * (qr_unix_sec - T_ref_epoch)
   where alpha is the camera's PTS at the session start epoch (seconds) and
   beta is clock drift (~1.0).
5. Alignment & Reporting:
   Pick a reference camera, compute relative offsets (alpha_i - alpha_ref),
   write sync_report.json / sync_report.md, and optionally remux aligned clips
   via FFmpeg -itsoffset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional / lazy imports
# ---------------------------------------------------------------------------

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except (ImportError, FileNotFoundError):
    HAS_PYZBAR = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1)  # 0: top-left, 1: bottom-right
DEFAULT_SAMPLE_FPS = 15.0


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class QRObservation:
    """A single decoded timestamp observation from a video frame."""
    pts_sec: float
    unix_ms: int
    seq: int
    frame_idx: int
    aruco_found: bool = False
    aruco_ids: list[int] = field(default_factory=list)
    bbox: list[int] | None = None  # [x, y, w, h]


@dataclass
class CameraFit:
    """Fitted offset and drift parameters for one camera."""
    camera_id: str
    file_name: str
    offset_sec: float = 0.0          # Relative offset vs reference camera (seconds)
    absolute_offset_sec: float = 0.0 # Fitted intercept alpha at T_ref_epoch (seconds)
    drift: float = 1.0               # Slope beta (clock drift ratio)
    samples: int = 0
    rmse_ms: float = 0.0             # Root mean square error of fit in ms
    max_error_ms: float = 0.0        # Max residual error in ms
    confidence: str = "failed"       # "high" | "medium" | "low" | "failed"
    status: str = "ok"               # "ok" | "no_detections" | "fit_failed" | "insufficient_data"
    fps: float = 0.0
    duration_sec: float = 0.0
    first_qr_pts: float | None = None
    last_qr_pts: float | None = None


@dataclass
class SyncReport:
    """Complete synchronization report across all cameras."""
    run_id: str
    created_at: str
    reference_camera: str
    sample_fps: float
    base_epoch_unix_ms: int = 0
    cameras: dict[str, CameraFit] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "reference_camera": self.reference_camera,
            "sample_fps": self.sample_fps,
            "base_epoch_unix_ms": self.base_epoch_unix_ms,
            "cameras": {
                cid: asdict(fit) for cid, fit in self.cameras.items()
            },
        }


# ---------------------------------------------------------------------------
# ArUco Marker Detection & ROI Extraction
# ---------------------------------------------------------------------------

def _detect_aruco_markers(gray_img: np.ndarray) -> tuple[dict[int, np.ndarray], list[int]]:
    """Detect ArUco markers in a grayscale image.

    Returns:
        marker_centers: mapping of marker_id -> center coordinate (x, y)
        detected_ids: list of all detected marker IDs
    """
    marker_centers: dict[int, np.ndarray] = {}
    detected_ids: list[int] = []

    try:
        # Modern OpenCV 4.7+
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray_img)
    except AttributeError:
        try:
            # Legacy OpenCV 4.x
            dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray_img, dictionary, parameters=parameters)
        except Exception:
            return marker_centers, detected_ids

    if ids is not None and len(ids) > 0:
        for i, marker_id_arr in enumerate(ids):
            m_id = int(np.ravel(marker_id_arr)[0])
            detected_ids.append(m_id)
            c = np.reshape(corners[i], (-1, 2))
            center = np.mean(c, axis=0)
            marker_centers[m_id] = center

    return marker_centers, detected_ids


def _extract_qr_roi(img: np.ndarray, marker_centers: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int] | None]:
    """Compute region of interest for QR code from detected ArUco markers 0 and 1.

    Returns:
        roi_img: cropped image (or full image if markers not found)
        bbox: [x, y, w, h] of ROI in original image coordinates
    """
    h, w = img.shape[:2]

    # If both markers 0 (top-left) and 1 (bottom-right) are found
    if 0 in marker_centers and 1 in marker_centers:
        c0 = marker_centers[0]
        c1 = marker_centers[1]

        # Estimated display center
        center_x = (c0[0] + c1[0]) / 2.0
        center_y = (c0[1] + c1[1]) / 2.0

        # Estimated bounding box span
        span_x = abs(c1[0] - c0[0])
        span_y = abs(c1[1] - c0[1])

        # QR code is centered and occupies roughly 40-70% of the display width/height
        half_w = max(50, int(span_x * 0.45))
        half_h = max(50, int(span_y * 0.45))

        x0 = max(0, int(center_x - half_w))
        y0 = max(0, int(center_y - half_h))
        x1 = min(w, int(center_x + half_w))
        y1 = min(h, int(center_y + half_h))

        if x1 > x0 + 20 and y1 > y0 + 20:
            return img[y0:y1, x0:x1], [x0, y0, x1 - x0, y1 - y0]

    # Fallback to full image
    return img, None


# ---------------------------------------------------------------------------
# Multi-Pass QR Code Decoding
# ---------------------------------------------------------------------------

def _parse_qr_payload(data_str: str) -> tuple[int, int] | None:
    """Parse JSON payload {"u": <unix_ms>, "s": <seq>}.

    Returns:
        (unix_ms, seq) if valid, else None.
    """
    try:
        data = json.loads(data_str)
        if isinstance(data, dict) and "u" in data and "s" in data:
            unix_ms = int(data["u"])
            seq = int(data["s"])
            # Sanity check: valid unix timestamp (> 1e12 ms) and non-negative sequence
            if unix_ms > 1_000_000_000_000 and seq >= 0:
                return unix_ms, seq
    except Exception:
        pass
    return None


def _decode_qr_image(img: np.ndarray) -> tuple[int, int, list[int] | None] | None:
    """Attempt multi-pass QR decode on a BGR or Grayscale image.

    Returns:
        (unix_ms, seq, bbox) on success, or None.
    """
    if img is None or img.size == 0:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Multi-pass candidate images for QR detection
    candidates = [
        ("raw_gray", gray),
    ]

    # Pass 2: Otsu binary thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    candidates.append(("otsu", otsu))

    # Pass 3: Inverted Otsu
    candidates.append(("otsu_inv", 255 - otsu))

    # Pass 4: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )
    candidates.append(("adaptive", adaptive))

    # Pass 5: 2x Upscaled if image is small
    if gray.shape[0] < 300 or gray.shape[1] < 300:
        scaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        candidates.append(("scaled", scaled))

    # 1. Try pyzbar first if available
    if HAS_PYZBAR:
        for _, cand in candidates:
            try:
                decoded = pyzbar.decode(cand)
                for item in decoded:
                    text = item.data.decode("utf-8", errors="ignore")
                    parsed = _parse_qr_payload(text)
                    if parsed is not None:
                        unix_ms, seq = parsed
                        rect = item.rect
                        bbox = [int(rect.left), int(rect.top), int(rect.width), int(rect.height)]
                        return unix_ms, seq, bbox
            except Exception:
                continue

    # 2. Fallback to OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        for _, cand in candidates:
            text, points, _ = detector.detectAndDecode(cand)
            if text:
                parsed = _parse_qr_payload(text)
                if parsed is not None:
                    unix_ms, seq = parsed
                    bbox = None
                    if points is not None and len(points) > 0:
                        pts = points[0]
                        min_x, min_y = np.min(pts, axis=0)
                        max_x, max_y = np.max(pts, axis=0)
                        bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
                    return unix_ms, seq, bbox
    except Exception:
        pass

    return None


def detect_qr_in_frame(frame: np.ndarray) -> dict[str, Any] | None:
    """Detect ArUco markers and decode QR payload in a single video frame.

    Returns:
        dict with keys {"unix_ms", "seq", "aruco_found", "aruco_ids", "bbox"}
        or None if no QR was found.
    """
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    marker_centers, detected_ids = _detect_aruco_markers(gray)
    aruco_found = (0 in marker_centers or 1 in marker_centers)

    # 1. Try ROI based on ArUco first
    roi_img, roi_bbox = _extract_qr_roi(frame, marker_centers)
    res = _decode_qr_image(roi_img)

    if res is not None:
        unix_ms, seq, local_bbox = res
        global_bbox = roi_bbox
        if local_bbox is not None and roi_bbox is not None:
            global_bbox = [
                roi_bbox[0] + local_bbox[0],
                roi_bbox[1] + local_bbox[1],
                local_bbox[2],
                local_bbox[3],
            ]
        return {
            "unix_ms": unix_ms,
            "seq": seq,
            "aruco_found": aruco_found,
            "aruco_ids": detected_ids,
            "bbox": global_bbox,
        }

    # 2. If ROI decode failed and ROI was cropped, try full frame
    if roi_bbox is not None:
        res_full = _decode_qr_image(frame)
        if res_full is not None:
            unix_ms, seq, bbox = res_full
            return {
                "unix_ms": unix_ms,
                "seq": seq,
                "aruco_found": aruco_found,
                "aruco_ids": detected_ids,
                "bbox": bbox,
            }

    return None


# ---------------------------------------------------------------------------
# Video Sampling & Observation Collection
# ---------------------------------------------------------------------------

def extract_frame_observations(
    video_path: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    save_overlays_dir: Path | None = None,
    max_frames: int | None = None,
) -> tuple[list[QRObservation], float, float]:
    """Extract sparse frames from video and collect timestamp observations.

    Parameters:
        video_path: Path to the MKV/MP4 video file.
        sample_fps: Desired sampling rate in Hz.
        save_overlays_dir: Optional directory to save annotated debug frames.
        max_frames: Optional upper limit on processed frames.

    Returns:
        observations: list of valid QRObservation objects.
        fps: nominal video framerate.
        duration_sec: estimated total video duration in seconds.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0  # Fallback default if container doesn't report FPS

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1

    if save_overlays_dir is not None:
        save_overlays_dir.mkdir(parents=True, exist_ok=True)

    observations: list[QRObservation] = []
    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Compute PTS in seconds
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_sec = (pos_msec / 1000.0) if pos_msec > 0 else (frame_idx / fps)

            detection = detect_qr_in_frame(frame)
            if detection is not None:
                obs = QRObservation(
                    pts_sec=pts_sec,
                    unix_ms=detection["unix_ms"],
                    seq=detection["seq"],
                    frame_idx=frame_idx,
                    aruco_found=detection["aruco_found"],
                    aruco_ids=detection["aruco_ids"],
                    bbox=detection["bbox"],
                )
                observations.append(obs)

                # Save annotated debug overlay if requested
                if save_overlays_dir is not None:
                    _save_annotated_frame(
                        frame=frame,
                        obs=obs,
                        out_path=save_overlays_dir / f"frame_{frame_idx:06d}_pts_{pts_sec:.3f}s.jpg",
                    )

            sampled_count += 1
            if max_frames is not None and sampled_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return observations, fps, duration_sec


def _save_annotated_frame(frame: np.ndarray, obs: QRObservation, out_path: Path) -> None:
    """Draw diagnostic detection overlays on a frame and write to disk."""
    vis = frame.copy()
    # Draw QR bounding box
    if obs.bbox is not None:
        x, y, w, h = obs.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Info overlay text
    clock_str = time.strftime("%H:%M:%S", time.gmtime(obs.unix_ms / 1000.0))
    ms_rem = obs.unix_ms % 1000
    info_text = f"PTS: {obs.pts_sec:.3f}s | SEQ: {obs.seq:04d} | TIME: {clock_str}.{ms_rem:03d}"
    cv2.putText(
        vis,
        info_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), vis)


# ---------------------------------------------------------------------------
# Outlier Filtering & Monotonicity
# ---------------------------------------------------------------------------

def filter_observations(observations: list[QRObservation]) -> list[QRObservation]:
    """Filter out spurious readings, backward sequence jumps, and time outliers."""
    if not observations:
        return []

    # Sort by video PTS
    sorted_obs = sorted(observations, key=lambda o: o.pts_sec)

    cleaned: list[QRObservation] = []
    max_seq_seen = -1

    for obs in sorted_obs:
        # Sequence monotonicity: reject severe backward jumps (e.g. > 2 sequence steps back)
        if max_seq_seen >= 0 and obs.seq < max_seq_seen - 2:
            continue

        # If previous observation exists, check timestamp consistency
        if cleaned:
            prev = cleaned[-1]
            dt_video = obs.pts_sec - prev.pts_sec
            dt_qr = (obs.unix_ms - prev.unix_ms) / 1000.0

            # If video advanced but QR jumped backwards by > 1.0s, ignore
            if dt_video >= 0 and dt_qr < -1.0:
                continue

        cleaned.append(obs)
        if obs.seq > max_seq_seen:
            max_seq_seen = obs.seq

    return cleaned


# ---------------------------------------------------------------------------
# Offset & Drift Fitting
# ---------------------------------------------------------------------------

def fit_camera_offset(
    observations: list[QRObservation],
    camera_id: str = "camera",
    file_name: str = "",
    fps: float = 30.0,
    duration_sec: float = 0.0,
    base_epoch_unix_sec: float | None = None,
) -> CameraFit:
    """Fit offset + drift model: pts = alpha + beta * (unix_sec - base_epoch_unix_sec).

    Uses sequence transition boundary detection when available (for sub-frame accuracy),
    with fallback to iterative sigma-clipped linear regression.

    Parameters:
        observations: list of timestamp observations.
        camera_id: ID of the camera.
        file_name: Video file name.
        fps: Nominal video FPS.
        duration_sec: Video duration.
        base_epoch_unix_sec: Common reference unix epoch (seconds). If None,
            uses min(unix_sec) of this camera.

    Returns:
        CameraFit object with fitted alpha, beta, RMSE, and confidence.
    """
    fit = CameraFit(
        camera_id=camera_id,
        file_name=file_name,
        fps=fps,
        duration_sec=duration_sec,
    )

    cleaned = filter_observations(observations)
    if not cleaned:
        fit.status = "no_detections"
        fit.confidence = "failed"
        return fit

    fit.samples = len(cleaned)
    fit.first_qr_pts = cleaned[0].pts_sec
    fit.last_qr_pts = cleaned[-1].pts_sec

    # Extract raw arrays
    pts_arr = np.array([o.pts_sec for o in cleaned], dtype=np.float64)
    unix_sec_arr = np.array([o.unix_ms / 1000.0 for o in cleaned], dtype=np.float64)

    # Reference origin t0 (seconds) to keep condition number well-behaved
    t0 = base_epoch_unix_sec if base_epoch_unix_sec is not None else np.min(unix_sec_arr)

    # 1. Look for sequence transition boundaries (first frame of a new QR payload)
    trans_x: list[float] = []
    trans_y: list[float] = []

    for i in range(1, len(cleaned)):
        prev_obs = cleaned[i - 1]
        curr_obs = cleaned[i]
        if curr_obs.seq > prev_obs.seq:
            # Transition occurs between prev and curr frame
            mid_pts = (prev_obs.pts_sec + curr_obs.pts_sec) / 2.0
            t_unix_sec = curr_obs.unix_ms / 1000.0
            trans_x.append(t_unix_sec - t0)
            trans_y.append(mid_pts)

    if len(trans_x) >= 3:
        x = np.array(trans_x, dtype=np.float64)
        y = np.array(trans_y, dtype=np.float64)
    else:
        # Fallback: use all frame observations
        x = unix_sec_arr - t0
        y = pts_arr

    if len(x) < 2:
        # Single observation: assume drift beta = 1.0
        alpha = float(y[0] - x[0])
        fit.absolute_offset_sec = alpha
        fit.offset_sec = alpha
        fit.drift = 1.0
        fit.rmse_ms = 0.0
        fit.max_error_ms = 0.0
        fit.confidence = "low"
        fit.status = "insufficient_data"
        return fit

    # Theil-Sen robust slope estimator (immune to leverage outliers)
    slopes: list[float] = []
    n_pts = len(x)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = x[j] - x[i]
            if abs(dx) > 1e-4:
                slopes.append((y[j] - y[i]) / dx)

    if slopes:
        beta_init = float(np.median(slopes))
        alpha_init = float(np.median(y - beta_init * x))
    else:
        beta_init, alpha_init = 1.0, float(y[0] - x[0])

    # Compute residuals against Theil-Sen baseline
    residuals = np.abs(y - (alpha_init + beta_init * x))
    med_res = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med_res)))
    sigma_mad = 1.4826 * mad
    thresh = max(0.035, 3.0 * sigma_mad)

    mask = residuals <= thresh

    # Final fit on inliers
    if np.sum(mask) >= 2:
        beta, alpha = np.polyfit(x[mask], y[mask], deg=1)
        inlier_residuals = (y[mask] - (alpha + beta * x[mask])) * 1000.0  # in ms
        rmse_ms = float(np.sqrt(np.mean(inlier_residuals ** 2)))
        max_error_ms = float(np.max(np.abs(inlier_residuals)))
    else:
        beta, alpha = 1.0, float(y[0] - x[0])
        rmse_ms = 0.0
        max_error_ms = 0.0

    fit.absolute_offset_sec = float(alpha)
    fit.offset_sec = float(alpha)
    fit.drift = float(beta)
    fit.rmse_ms = rmse_ms
    fit.max_error_ms = max_error_ms
    fit.samples = int(np.sum(mask))

    # Determine confidence
    if fit.samples >= 6 and rmse_ms < 25.0 and abs(beta - 1.0) < 0.005:
        fit.confidence = "high"
    elif fit.samples >= 3 and rmse_ms < 50.0 and abs(beta - 1.0) < 0.02:
        fit.confidence = "medium"
    elif fit.samples >= 2:
        fit.confidence = "low"
    else:
        fit.confidence = "failed"
        fit.status = "fit_failed"

    return fit


# ---------------------------------------------------------------------------
# Multi-Camera Relative Alignment
# ---------------------------------------------------------------------------

def compute_relative_offsets(
    camera_fits: dict[str, CameraFit],
    ref_camera_id: str | None = None,
) -> tuple[str, dict[str, CameraFit]]:
    """Express all camera offsets relative to a chosen reference camera.

    Returns:
        (reference_camera_id, aligned_camera_fits)
    """
    valid_cams = [
        cid for cid, fit in camera_fits.items()
        if fit.status == "ok" and fit.confidence in ("high", "medium", "low")
    ]

    if not valid_cams:
        # No successfully fitted cameras; return as-is
        ref = ref_camera_id or (next(iter(camera_fits.keys())) if camera_fits else "none")
        return ref, camera_fits

    # Select reference camera
    if ref_camera_id and ref_camera_id in camera_fits and camera_fits[ref_camera_id].status == "ok":
        ref = ref_camera_id
    else:
        # Auto-select: prioritize "high" confidence, most samples, lowest RMSE
        def score_cam(cid: str) -> tuple[int, int, float]:
            f = camera_fits[cid]
            conf_score = 3 if f.confidence == "high" else (2 if f.confidence == "medium" else 1)
            return (conf_score, f.samples, -f.rmse_ms)

        ref = max(valid_cams, key=score_cam)

    ref_abs_offset = camera_fits[ref].absolute_offset_sec

    # Update relative offsets: offset_{i -> ref} = alpha_i - alpha_ref
    aligned = dict(camera_fits)
    for cid, fit in aligned.items():
        if fit.status == "ok":
            fit.offset_sec = float(fit.absolute_offset_sec - ref_abs_offset)
        else:
            fit.offset_sec = 0.0

    return ref, aligned


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_sync_report(report: SyncReport, out_dir: Path) -> Path:
    """Serialize the sync report to JSON and Markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "sync_report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    md_path = out_dir / "sync_report.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(generate_markdown_summary(report))

    return json_path


def generate_markdown_summary(report: SyncReport) -> str:
    """Generate a clean Markdown summary table from a SyncReport."""
    lines = [
        f"# Multi-Camera Synchronization Report",
        f"",
        f"- **Run ID**: `{report.run_id}`",
        f"- **Generated At**: {report.created_at}",
        f"- **Reference Camera**: `{report.reference_camera}`",
        f"- **Sample Rate**: {report.sample_fps:.1f} Hz",
        f"",
        f"| Camera | File | Rel Offset (s) | Frame Offset | Drift | Samples | RMSE (ms) | Confidence | Status |",
        f"|--------|------|----------------|--------------|-------|---------|-----------|------------|--------|",
    ]

    for cid, fit in sorted(report.cameras.items()):
        fps = fit.fps if fit.fps > 0 else 30.0
        frame_offset = int(round(fit.offset_sec * fps))
        drift_str = f"{fit.drift:.6f}" if fit.status == "ok" else "N/A"
        rmse_str = f"{fit.rmse_ms:.2f}" if fit.status == "ok" else "N/A"
        rel_sec_str = f"{fit.offset_sec:+.4f}" if fit.status == "ok" else "N/A"

        lines.append(
            f"| `{cid}` | `{fit.file_name}` | {rel_sec_str} | {frame_offset:+d} frames | "
            f"{drift_str} | {fit.samples} | {rmse_str} | **{fit.confidence}** | {fit.status} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FFmpeg Remuxing
# ---------------------------------------------------------------------------

def remux_synced_videos(
    input_dir: Path,
    report: SyncReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Apply FFmpeg -itsoffset to create time-aligned video clips.

    Returns mapping of camera_id -> synced_mkv_path.
    """
    synced_dir = output_dir / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for cid, fit in report.cameras.items():
        if fit.status != "ok":
            print(f"[WARN] Skipping remux for '{cid}' (status={fit.status})", file=sys.stderr)
            continue

        in_file = input_dir / fit.file_name
        if not in_file.exists():
            print(f"[WARN] Video file not found for remux: {in_file}", file=sys.stderr)
            continue

        out_file = synced_dir / f"{cid}_synced.mkv"
        offset = fit.offset_sec

        # FFmpeg command using stream-copy and -itsoffset
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-itsoffset", f"{offset:.6f}",
            "-i", str(in_file),
            "-c", "copy",
            str(out_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results[cid] = out_file
            print(f"  [{cid}] Remuxed -> {out_file.name} (offset: {offset:+.4f}s)")
        except Exception as exc:
            print(f"[ERROR] Failed to remux {cid}: {exc}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Full Pipeline Orchestration
# ---------------------------------------------------------------------------

def analyze_recording(
    input_path: Path,
    output_dir: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    ref_camera: str | None = None,
    save_overlays: bool = False,
    remux: bool = False,
) -> SyncReport:
    """Run full synchronization analysis on a recording directory or video file.

    Parameters:
        input_path: Directory with MKVs + recording_manifest.json, or single video.
        output_dir: Directory to store sync_report.json, summary, and synced clips.
        sample_fps: Frame extraction sample rate in Hz.
        ref_camera: Optional explicit reference camera ID.
        save_overlays: Whether to save annotated debug frames.
        remux: Whether to generate aligned video files with FFmpeg.

    Returns:
        SyncReport object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover videos to analyze
    camera_files: dict[str, Path] = {}
    run_id = time.strftime("%Y%m%d_%H%M%S")

    if input_path.is_file():
        # Single video mode
        cid = input_path.stem
        camera_files[cid] = input_path
    elif input_path.is_dir():
        # Directory mode: check for recording_manifest.json first
        manifest_path = input_path / "recording_manifest.json"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                run_id = manifest.get("run_id", run_id)
                for cid, cinfo in manifest.get("cameras", {}).items():
                    fname = cinfo.get("file", f"{cid}.mkv")
                    fpath = input_path / fname
                    if fpath.exists():
                        camera_files[cid] = fpath
            except Exception as e:
                print(f"[WARN] Error reading manifest: {e}; falling back to glob.", file=sys.stderr)

        # If manifest was absent or didn't find files, glob for video files
        if not camera_files:
            for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov"):
                for vf in input_path.glob(ext):
                    if not vf.stem.endswith("_synced"):
                        camera_files[vf.stem] = vf
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not camera_files:
        raise ValueError(f"No video files found in {input_path}")

    print(f"[ANALYZER] Processing {len(camera_files)} camera(s) at {sample_fps} fps…")

    # Step 1: Extract observations for all cameras
    all_observations: dict[str, list[QRObservation]] = {}
    video_metas: dict[str, tuple[float, float]] = {}

    for cid, fpath in sorted(camera_files.items()):
        print(f"  [{cid}] Extracting frames from {fpath.name}…")
        overlay_dir = (output_dir / "detection_log" / cid) if save_overlays else None

        try:
            obs, fps, duration = extract_frame_observations(
                video_path=fpath,
                sample_fps=sample_fps,
                save_overlays_dir=overlay_dir,
            )
            all_observations[cid] = obs
            video_metas[cid] = (fps, duration)
            print(f"    -> {len(obs)} QRs detected")
        except Exception as exc:
            print(f"    -> [ERROR] Failed extracting {cid}: {exc}", file=sys.stderr)
            all_observations[cid] = []
            video_metas[cid] = (30.0, 0.0)

    # Step 2: Determine global session base epoch
    all_unix_ms = [
        o.unix_ms
        for obs_list in all_observations.values()
        for o in obs_list
    ]

    base_epoch_unix_ms = int(np.min(all_unix_ms)) if all_unix_ms else 0
    base_epoch_sec = base_epoch_unix_ms / 1000.0

    # Step 3: Fit models using shared base epoch
    raw_fits: dict[str, CameraFit] = {}
    for cid, fpath in sorted(camera_files.items()):
        obs = all_observations.get(cid, [])
        fps, duration = video_metas.get(cid, (30.0, 0.0))

        fit = fit_camera_offset(
            observations=obs,
            camera_id=cid,
            file_name=fpath.name,
            fps=fps,
            duration_sec=duration,
            base_epoch_unix_sec=base_epoch_sec,
        )
        raw_fits[cid] = fit

    # Step 4: Relative offset alignment
    selected_ref, aligned_fits = compute_relative_offsets(raw_fits, ref_camera_id=ref_camera)

    report = SyncReport(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reference_camera=selected_ref,
        sample_fps=sample_fps,
        base_epoch_unix_ms=base_epoch_unix_ms,
        cameras=aligned_fits,
    )

    # Step 5: Write reports
    report_json_path = write_sync_report(report, output_dir)
    print(f"\n[ANALYZER] Sync report written to: {report_json_path}")
    print("\n" + generate_markdown_summary(report))

    # Step 6: Optional remux
    if remux:
        print("\n[ANALYZER] Remuxing synchronized video clips…")
        remux_synced_videos(input_dir=input_path, report=report, output_dir=output_dir)

    return report


# ---------------------------------------------------------------------------
# Self-Test Mode
# ---------------------------------------------------------------------------

def run_selftest() -> int:
    """Generate synthetic multi-camera test videos and assert sub-frame precision."""
    print("[SELFTEST] Running camera-sync-analyzer synthetic self-test…")

    try:
        import qrcode
        from PIL import Image
    except ImportError:
        print("[SELFTEST] ERROR: qrcode and pillow required for self-test.", file=sys.stderr)
        return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="sync_selftest_"))
    try:
        width, height = 640, 480
        fps = 30.0
        duration_sec = 2.5
        total_frames = int(duration_sec * fps)

        # Ground truth offsets for 3 cameras (in seconds)
        gt_offsets = {
            "cam01": 0.000,
            "cam02": +0.120,   # 120 ms delay (~3.6 frames)
            "cam03": -0.080,   # -80 ms lead (~2.4 frames)
        }

        # ArUco markers 0 and 1
        marker_size = 80
        margin = 20
        dict_aruco = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        try:
            m0_img = cv2.aruco.generateImageMarker(dict_aruco, 0, marker_size)
            m1_img = cv2.aruco.generateImageMarker(dict_aruco, 1, marker_size)
        except AttributeError:
            m0_img = cv2.aruco.drawMarker(dict_aruco, 0, marker_size)
            m1_img = cv2.aruco.drawMarker(dict_aruco, 1, marker_size)

        m0_bgr = cv2.cvtColor(m0_img, cv2.COLOR_GRAY2BGR)
        m1_bgr = cv2.cvtColor(m1_img, cv2.COLOR_GRAY2BGR)

        # Base wall-clock epoch
        t_base_ms = 1_700_000_000_000

        print(f"  [1/3] Generating synthetic videos in {temp_dir}…")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for cid, offset in gt_offsets.items():
            video_path = temp_dir / f"{cid}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for f_idx in range(total_frames):
                frame_pts = f_idx / fps
                # Real wall-clock moment when this frame was captured:
                # pts = offset + (t_real - t0) => t_real = (pts - offset) + t0
                t_real_sec = frame_pts - offset
                # QR flips at 5 Hz (every 200 ms)
                seq = max(0, int(t_real_sec * 5.0))
                qr_unix_ms = int(t_base_ms + (seq * 200))

                # Synthesize screen frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw ArUco markers
                frame[margin:margin+marker_size, margin:margin+marker_size] = m0_bgr
                frame[
                    height-marker_size-margin:height-margin,
                    width-marker_size-margin:width-margin
                ] = m1_bgr

                # Generate QR code
                payload = json.dumps({"u": qr_unix_ms, "s": seq}, separators=(",", ":"))
                qr = qrcode.QRCode(box_size=4, border=2)
                qr.add_data(payload)
                qr.make(fit=True)
                qr_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
                qr_np = np.array(qr_pil)
                qr_bgr = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)

                qw, qh = qr_bgr.shape[1], qr_bgr.shape[0]
                qx = (width - qw) // 2
                qy = (height - qh) // 2
                frame[qy:qy+qh, qx:qx+qw] = qr_bgr

                writer.write(frame)

            writer.release()

        # Run analysis
        print("  [2/3] Executing analyze_recording…")
        report_dir = temp_dir / "report"
        report = analyze_recording(
            input_path=temp_dir,
            output_dir=report_dir,
            sample_fps=DEFAULT_SAMPLE_FPS,
            ref_camera="cam01",
        )

        # Assertions
        print("  [3/3] Validating accuracy against ground truth…")
        max_allowed_error_sec = 0.020  # 20 ms tolerance (sub-frame at 30 fps)

        for cid, gt_off in gt_offsets.items():
            expected_rel = gt_off - gt_offsets["cam01"]
            fit = report.cameras.get(cid)
            if fit is None or fit.status != "ok":
                raise AssertionError(f"Camera {cid} failed to fit properly: {fit}")

            error_sec = abs(fit.offset_sec - expected_rel)
            print(f"    [{cid}] Ground Truth: {expected_rel:+.4f}s | Fitted: {fit.offset_sec:+.4f}s | Error: {error_sec*1000.0:.2f}ms")

            if error_sec > max_allowed_error_sec:
                raise AssertionError(
                    f"Offset error for {cid} ({error_sec*1000:.1f}ms) exceeds tolerance ({max_allowed_error_sec*1000:.1f}ms)"
                )

        print("[SELFTEST] PASS: All camera offsets verified within sub-frame tolerance (<= 20ms).")
        return 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync-analyzer",
        description="Extract per-camera time offsets from recorded sync footage",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Directory containing MKVs and recording_manifest.json, or single video file",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output directory for sync report and remuxed clips",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        metavar="HZ",
        help=f"Frame sampling rate for QR detection (default: {DEFAULT_SAMPLE_FPS})",
    )
    parser.add_argument(
        "--ref-camera",
        type=str,
        default=None,
        metavar="ID",
        help="Camera ID to use as time reference (default: auto-select)",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save annotated detection frames to output/detection_log/",
    )
    parser.add_argument(
        "--remux",
        action="store_true",
        help="Remux synchronized video clips using FFmpeg -itsoffset",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run self-test with synthetic multi-camera video streams and exit",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    if args.input is None or args.out is None:
        parser.error("--input and --out are required unless running with --selftest.")

    try:
        analyze_recording(
            input_path=args.input,
            output_dir=args.out,
            sample_fps=args.sample_fps,
            ref_camera=args.ref_camera,
            save_overlays=args.save_overlays,
            remux=args.remux,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
