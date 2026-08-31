"""
Unit and integration tests for camera-sync-analyzer.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode

from sync_analyzer.analyze import (
    ARUCO_DICT,
    CameraFit,
    QRObservation,
    SyncReport,
    _decode_qr_image,
    _detect_aruco_markers,
    _extract_qr_roi,
    _parse_qr_payload,
    analyze_recording,
    compute_relative_offsets,
    detect_qr_in_frame,
    filter_observations,
    fit_camera_offset,
    generate_markdown_summary,
    write_sync_report,
)


def _generate_synthetic_test_frame(
    unix_ms: int = 1700000000000,
    seq: int = 1,
    width: int = 640,
    height: int = 480,
    marker_size: int = 80,
    margin: int = 20,
) -> np.ndarray:
    """Helper to synthesize a test frame with ArUco 0, 1 and centered QR."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    dict_aruco = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        m0 = cv2.aruco.generateImageMarker(dict_aruco, 0, marker_size)
        m1 = cv2.aruco.generateImageMarker(dict_aruco, 1, marker_size)
    except AttributeError:
        m0 = cv2.aruco.drawMarker(dict_aruco, 0, marker_size)
        m1 = cv2.aruco.drawMarker(dict_aruco, 1, marker_size)

    # ArUco markers need a white border / quiet zone on black background
    m0_b = cv2.copyMakeBorder(m0, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    m1_b = cv2.copyMakeBorder(m1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    m0_bgr = cv2.cvtColor(m0_b, cv2.COLOR_GRAY2BGR)
    m1_bgr = cv2.cvtColor(m1_b, cv2.COLOR_GRAY2BGR)

    bw, bh = m0_bgr.shape[1], m0_bgr.shape[0]
    frame[margin:margin+bh, margin:margin+bw] = m0_bgr
    frame[
        height-bh-margin:height-margin,
        width-bw-margin:width-margin
    ] = m1_bgr

    # QR
    payload = json.dumps({"u": unix_ms, "s": seq}, separators=(",", ":"))
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_bgr = cv2.cvtColor(np.array(qr_pil), cv2.COLOR_RGB2BGR)

    qw, qh = qr_bgr.shape[1], qr_bgr.shape[0]
    qx = (width - qw) // 2
    qy = (height - qh) // 2
    frame[qy:qy+qh, qx:qx+qw] = qr_bgr

    return frame


def test_parse_qr_payload():
    valid = '{"u":1722948723000,"s":42}'
    assert _parse_qr_payload(valid) == (1722948723000, 42)

    invalid_json = '{"u": 123'
    assert _parse_qr_payload(invalid_json) is None

    missing_keys = '{"time": 1722948723000}'
    assert _parse_qr_payload(missing_keys) is None

    invalid_epoch = '{"u": 500,"s": 1}'
    assert _parse_qr_payload(invalid_epoch) is None


def test_aruco_and_qr_detection():
    unix_ms = 1700000050000
    seq = 15
    frame = _generate_synthetic_test_frame(unix_ms=unix_ms, seq=seq)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_centers, detected_ids = _detect_aruco_markers(gray)
    assert 0 in detected_ids
    assert 1 in detected_ids

    detection = detect_qr_in_frame(frame)
    assert detection is not None
    assert detection["unix_ms"] == unix_ms
    assert detection["seq"] == seq
    assert detection["aruco_found"] is True


def test_filter_observations():
    obs = [
        QRObservation(pts_sec=0.0, unix_ms=1000, seq=1, frame_idx=0),
        QRObservation(pts_sec=0.2, unix_ms=1200, seq=2, frame_idx=6),
        QRObservation(pts_sec=0.4, unix_ms=1400, seq=3, frame_idx=12),
        QRObservation(pts_sec=0.6, unix_ms=500, seq=0, frame_idx=18), # Outlier (backward jump)
        QRObservation(pts_sec=0.8, unix_ms=1800, seq=5, frame_idx=24),
    ]

    filtered = filter_observations(obs)
    assert len(filtered) == 4
    assert [o.seq for o in filtered] == [1, 2, 3, 5]


def test_fit_camera_offset():
    # Simulate camera with offset +0.100s at 30 fps (6 frames per 200 ms QR)
    t0_ms = 1700000000000
    true_offset = 0.100
    fps = 30.0

    observations = []
    for f in range(60):
        pts_sec = f / fps
        dt = pts_sec - true_offset
        seq = max(0, int(dt * 5.0))
        unix_ms = int(t0_ms + (seq * 200))
        observations.append(
            QRObservation(pts_sec=pts_sec, unix_ms=unix_ms, seq=seq, frame_idx=f)
        )

    # Add an outlier
    observations.append(
        QRObservation(pts_sec=3.5, unix_ms=int(t0_ms + 9000.0), seq=99, frame_idx=100)
    )

    fit = fit_camera_offset(
        observations,
        camera_id="cam01",
        file_name="cam01.mkv",
        base_epoch_unix_sec=t0_ms / 1000.0,
    )
    assert fit.status == "ok"
    assert fit.confidence == "high"
    assert abs(fit.drift - 1.0) < 0.01
    assert fit.rmse_ms < 20.0


def test_compute_relative_offsets():
    fits = {
        "cam01": CameraFit(camera_id="cam01", file_name="cam01.mkv", absolute_offset_sec=10.0, confidence="high", status="ok"),
        "cam02": CameraFit(camera_id="cam02", file_name="cam02.mkv", absolute_offset_sec=10.150, confidence="high", status="ok"),
        "cam03": CameraFit(camera_id="cam03", file_name="cam03.mkv", absolute_offset_sec=9.920, confidence="high", status="ok"),
    }

    ref_cam, aligned = compute_relative_offsets(fits, ref_camera_id="cam01")
    assert ref_cam == "cam01"
    assert aligned["cam01"].offset_sec == pytest.approx(0.000, abs=1e-4)
    assert aligned["cam02"].offset_sec == pytest.approx(0.150, abs=1e-4)
    assert aligned["cam03"].offset_sec == pytest.approx(-0.080, abs=1e-4)


def test_sync_report_io_and_summary():
    report = SyncReport(
        run_id="test_run",
        created_at="2026-08-12T19:00:00Z",
        reference_camera="cam01",
        sample_fps=15.0,
        cameras={
            "cam01": CameraFit(
                camera_id="cam01",
                file_name="cam01.mkv",
                offset_sec=0.0,
                absolute_offset_sec=1.234,
                drift=1.0,
                samples=20,
                rmse_ms=2.5,
                confidence="high",
                status="ok",
                fps=30.0,
            )
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_path = write_sync_report(report, tmp_path)
        assert json_path.exists()

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["reference_camera"] == "cam01"
        assert "cam01" in data["cameras"]

        summary = generate_markdown_summary(report)
        assert "test_run" in summary
        assert "cam01" in summary


def test_end_to_end_analyze_recording():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        width, height, fps = 640, 480, 30.0
        duration_sec = 2.0
        total_frames = int(duration_sec * fps)
        t_base_ms = 1700000000000

        gt_offsets = {"c1": 0.000, "c2": 0.100}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for cid, offset in gt_offsets.items():
            video_path = dir_path / f"{cid}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            for f in range(total_frames):
                pts = f / fps
                t_real = pts - offset
                seq = max(0, int(t_real * 5.0))
                qr_unix_ms = int(t_base_ms + (seq * 200))
                frame = _generate_synthetic_test_frame(unix_ms=qr_unix_ms, seq=seq, width=width, height=height)
                writer.write(frame)
            writer.release()

        report_dir = dir_path / "out_report"
        report = analyze_recording(
            input_path=dir_path,
            output_dir=report_dir,
            sample_fps=15.0,
            ref_camera="c1",
        )

        assert "c1" in report.cameras
        assert "c2" in report.cameras
        assert report.cameras["c1"].status == "ok"
        assert report.cameras["c2"].status == "ok"

        # Check relative offset between c2 and c1
        rel_c2 = report.cameras["c2"].offset_sec
        assert abs(rel_c2 - 0.100) <= 0.035  # Frame-level accuracy (<= 1 frame / 33.3 ms at 30 fps)
