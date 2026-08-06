# Manual Camera Sync via Encoded Screen Display — Software Plan

> **Status**: Planning document — no code yet.  
> **Problem**: IP cameras stream RTSP/RTP but do **not** emit RTCP Sender Reports (SR), making standard RTP timestamp synchronization impossible.  
> **Solution**: Record all cameras while they briefly film a display showing a time-encoded visual pattern. Post-process to extract per-camera time offsets.

---

## 1. Goals & Non-Goals

**Goals**
- Synchronize up to 16 IP camera recordings to a common timeline after capture.
- Achieve frame-level accuracy (≈ ±1 frame / ±33 ms at 30 fps) without hardware genlock.
- Be operable by one person with a laptop and the existing camera network.

**Non-Goals**
- Real-time live sync during recording (this is a post-hoc method).
- Sub-frame / genlock-grade accuracy (that requires hardware).
- Replacing the cameras or modifying their firmware.

---

## 2. Why Drop Viseron for This Workflow

Viseron is a full VMS (Video Management System) built around motion detection, object detection, and NVR storage. For our sync use-case it adds unnecessary complexity:

| Concern | Viseron | FFmpeg + Custom Scripts |
|---------|---------|------------------------|
| RTSP without RTCP | Handles it, but abstracts timestamps away | Native, transparent, preserves RTP timestamps |
| 16 parallel streams | Heavy container orchestration | 16 independent OS processes |
| Container / GPU need | Requires Docker, often a GPU for AI features | No container, no GPU needed for recording |
| Timestamp access | Hard to get raw PTS values | `-copyts` gives exact PTS in MKV |
| Post-processing | Not designed for frame-level offset math | We build exactly what we need |

**Decision**: Use **FFmpeg** for recording and a small Python wrapper for orchestration. Viseron can remain for live viewing if desired, but it is not part of the sync pipeline.

---

## 3. High-Level Architecture

Three independent software components:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. SYNC DISPLAY │     │  2. RECORDER     │     │  3. SYNC ANALYZER│
│  (One laptop)    │     │  (Workstation)   │     │  (Post-process)  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
   Shows encoded time      Records 16 RTSP         Decodes markers,
   pattern on screen       streams to MKV          computes offsets,
   ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲         + sidecar JSON          generates report
   ││││││││││││││││
   └┴┴┴┴┴┴┴┴┴┴┴┴┴┘
   All cameras film
   the display
```

---

## 4. Component 1 — Sync Display

A full-screen application running on a laptop or external monitor/TV. All cameras point at this screen for a short calibration window (≈ 5–10 seconds per camera group).

### 4.1 Visual Layout

```
┌─────────────────────────────────────────┐
│  ┌─────┐                                │
│  │ArUco│   HH:MM:SS.mmm                 │
│  │  0  │   14:52:03.427                 │
│  └─────┘                                │
│                                         │
│         ┌─────────────────────┐         │
│         │                     │         │
│         │    QR CODE          │         │
│         │    (updates 5×/s)   │         │
│         │                     │         │
│         └─────────────────────┘         │
│                                         │
│                                ┌─────┐  │
│                                │ArUco│  │
│                                │  1  │  │
│                                └─────┘  │
└─────────────────────────────────────────┘
```

**Elements**

| Element | Purpose | Spec |
|---------|---------|------|
| **Digital Clock** | Human-readable verification | Monospace, white on black, `HH:MM:SS.mmm` |
| **QR Code** | Machine-readable timestamp | Updates at **5 Hz** (every 200 ms). Encodes JSON: `{"u":<unix_ms>,"s":<seq>}`. High ECC (30%). Minimum 400×400 px on screen. |
| **ArUco Markers** | Auto-detection & ROI localization | Two static markers in opposite corners. Used to quickly locate the QR region and correct perspective if the camera is angled. |
| **Background** | Maximize contrast | Pure black (`#000000`) to reduce LCD blooming and moiré. |

### 4.2 Update Strategy

- **Text clock**: Updates every display refresh cycle (e.g., 60 Hz) so it is always readable.
- **QR code**: Updates at **5 Hz** only. This is the critical design decision:
  - At 30 fps camera capture, a 200 ms dwell time gives **6 consecutive frames** showing the identical QR code.
  - This avoids motion blur caused by a code that changes every frame.
  - 5 Hz is frequent enough to collect many data points in a 5-second window (≈ 25 samples per camera).
- **Sequence number** (`s`): Monotonically increases with every QR update. Detecting a gap in `s` flags possible dropped frames or mis-reads.

### 4.3 Display Technology Notes

- **Filming a screen with digital cameras** introduces risks:
  - *Moiré patterns*: Mitigate by using a large QR (occupies ≥ 5% of the camera frame height) and high QR error correction.
  - *Rolling shutter partial capture*: The 200 ms dwell time is much longer than typical rolling-shutter readout (≈ 20–50 ms), so the QR is fully captured in most frames.
  - *LCD flicker / beat frequency*: Use maximum screen brightness. A 60 Hz LCD filmed at 30 fps is usually stable.
- **Logistics for 16 cameras**:
  - *Option A — Single large TV*: If all cameras are in one room, a 55"+ TV may be visible to many at once.
  - *Option B — Mobile laptop walk*: Carry the laptop to each camera cluster. The recording is continuous, so each camera only needs 5 seconds of the display. A 90-second walk-around is sufficient.

---

## 5. Component 2 — Recorder

A lightweight Python launcher that orchestrates FFmpeg processes. No heavy VMS required.

### 5.1 Why FFmpeg

- Native RTSP/RTP support without requiring RTCP SR.
- `-copyts` preserves the original RTP timestamps into the container.
- MKV (Matroska) is the recommended container: robust, supports arbitrary codecs, and stores timestamps with high precision.
- Battle-tested for running dozens of simultaneous network streams.

### 5.2 Per-Camera FFmpeg Command

```bash
ffmpeg \
  -hide_banner -loglevel error \
  -rtsp_transport tcp \
  -fflags +genpts \
  -i rtsp://CAMERA_IP:554/stream \
  -c copy \
  -f matroska \
  -copyts \
  -start_at_zero \
  -metadata camera_id="cam01" \
  -metadata rtsp_url="rtsp://CAMERA_IP:554/stream" \
  recordings/cam01_<start_unixtime>.mkv
```

**Key flags explained**

| Flag | Purpose |
|------|---------|
| `-rtsp_transport tcp` | UDP multicast can drop packets; TCP is more reliable for sync. |
| `-fflags +genpts` | Generates presentation timestamps if the source lacks them. |
| `-c copy` | No re-encoding; minimal CPU load, preserves original quality. |
| `-copyts` | Copies source timestamps (RTP-derived) into the output without shifting them to wall-clock. |
| `-start_at_zero` | Ensures the first timestamp starts near 0 for easier math later. |
| `-metadata` | Embeds camera ID directly in the file header. |

### 5.3 Multi-Camera Wrapper

A Python script (`record_cameras.py`) that:
1. Reads a YAML/JSON config listing up to 16 RTSP URLs and camera IDs.
2. Creates an output directory with a timestamped run name.
3. Launches one FFmpeg subprocess per camera.
4. Captures stdout/stderr to per-camera log files.
5. On `Ctrl+C` (SIGINT), forwards the signal to all children and waits for graceful finalization.
6. Writes a sidecar JSON file `recording_manifest.json`:
   ```json
   {
     "run_id": "2026-08-06_145203",
     "start_wall_time": "2026-08-06T14:52:03.427Z",
     "cameras": {
       "cam01": {
         "file": "cam01_1722948723.mkv",
         "rtsp_url": "rtsp://...",
         "pid": 12345
       }
     }
   }
   ```

**Hardware considerations for 16 streams**
- Network: Ensure the switch and NIC can handle 16 × stream bitrate. Gigabit Ethernet is usually sufficient for 16 × 4 Mbps H.264 streams.
- Disk: Writing 16 streams to one HDD may saturate IOPS. Use an SSD or RAID, or distribute across multiple disks.
- CPU: Because we use `-c copy`, CPU usage is negligible (mostly I/O bound).

---

## 6. Component 3 — Sync Analyzer

The post-processing tool that turns the recorded MKV files into a sync report.

### 6.1 Processing Pipeline

```
Input MKV files
      │
      ▼
┌─────────────────────────────────────┐
│  A. Frame Sampling                  │
│  Extract 5 fps JPEGs (or raw) using │
│  FFmpeg seek-to-keyframes. We do    │
│  NOT need every frame.              │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  B. Detection                       │
│  1. Locate ArUco markers → define   │
│     region of interest (ROI).       │
│  2. Perspective-correct ROI if      │
│     camera is angled.               │
│  3. Run QR decoder (pyzbar /        │
│     cv2.QRCodeDetector).            │
│  4. Validate JSON schema & sequence │
│     monotonicity.                   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  C. Time Mapping                    │
│  For each valid detection:          │
│    (video_pts_seconds, qr_unix_ms)  │
│                                     │
│  Fit linear model per camera:       │
│    pts = offset + drift × qr_time   │
│                                     │
│  (drift is usually ≈ 1.0; solved    │
│  via RANSAC or least-squares)       │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  D. Offset Computation              │
│  Choose a reference camera (e.g.    │
│  the one with smallest latency, or  │
│  arbitrary). Express all offsets    │
│  relative to it.                    │
└─────────────────────────────────────┘
      │
      ▼
   Reports & Outputs
```

### 6.2 Detection Details

- **ArUco first**: The static ArUco markers are extremely robust and fast to detect. They tell us exactly where the QR code is, even if the camera is at an angle or the QR is small in the frame.
- **QR decode attempts**: Only run on the ROI defined by ArUco. This speeds up processing dramatically.
- **Validation rules**:
  - JSON must contain numeric `u` (unix ms) and `s` (sequence).
  - Sequence `s` must increase monotonically (allowing gaps, but never decreases).
  - `u` must increase by ≈ 200 ms per step.
- **Outlier rejection**: If a QR decode yields a timestamp inconsistent with neighboring frames by > 500 ms, discard it (likely a misread).

### 6.3 Offset Math

For each camera *i*, we have observations `(pts_ij, t_j)` where `t_j` is the QR timestamp and `pts_ij` is the video presentation timestamp for that frame.

Because `t_j` is the **same real moment** across all cameras, we fit:

```
pts_i = α_i + β_i × t
```

- `α_i` — offset (seconds). The camera with the smallest `α` is the "fastest" (least network/capture delay).
- `β_i` — drift (unitless, ≈ 1.0). Accounts for the camera's clock running slightly fast or slow relative to the display's clock.

For recordings under 1 hour, `β_i` is usually within ±0.001 of 1.0, but we still estimate it for robustness.

**Relative offsets** (what we actually use):
```
offset_{i→ref} = α_i - α_ref
```

### 6.4 Expected Accuracy

| Scenario | Accuracy |
|----------|----------|
| QR stable period (200 ms) | We know the real time to within ±100 ms just from which QR is visible. |
| Frame-level alignment (30 fps) | ±1 frame = ±33 ms by matching frames showing the same QR. |
| With sub-frame interpolation at QR transition boundaries | ±10–20 ms (estimating the exact moment the QR flips based on motion blur analysis). |

For almost all multi-camera editing purposes, **±1 frame is sufficient**.

---

## 7. Output Artifacts

The analyzer produces:

1. **`sync_report.json`** — The primary deliverable:
   ```json
   {
     "reference_camera": "cam03",
     "cameras": {
       "cam01": { "offset_sec": 0.127, "drift": 1.00001, "samples": 24, "confidence": "high" },
       "cam02": { "offset_sec": -0.043, "drift": 0.99998, "samples": 25, "confidence": "high" },
       "cam03": { "offset_sec": 0.000, "drift": 1.00000, "samples": 25, "confidence": "high" }
     }
   }
   ```

2. **`synced_preview/`** (optional) — FFmpeg-remuxed clips trimmed to a common overlapping window:
   ```bash
   ffmpeg -itsoffset 0.127 -i cam01.mkv -c copy synced/cam01_synced.mkv
   ffmpeg -itsoffset -0.043 -i cam02.mkv -c copy synced/cam02_synced.mkv
   ```

3. **`timeline.fcpxml` / `timeline.xml`** (optional) — Importable timeline for DaVinci Resolve, Premiere Pro, or Final Cut Pro with clips already placed on synchronized tracks.

4. **`detection_log/`** — Diagnostic images showing detected QR frames with overlays, useful for verifying that the algorithm found the correct sync points.

---

## 8. Operational Workflow (Step-by-Step)

1. **Start Recorder**
   ```bash
   python recorder.py --config cameras.yaml --out ./recordings/run_001
   ```
   All 16 cameras begin recording to MKV files.

2. **Wait for stabilization** (5–10 seconds) so all streams are fully buffered and writing.

3. **Start Sync Display** on a laptop.
   ```bash
   python sync_display.py --fullscreen
   ```

4. **Show the display to cameras**
   - If cameras are clustered: Hold the laptop up or place it near the groups for ~5 seconds each.
   - If a large TV is available and visible: Leave it in view for 10 seconds.
   - The recording is continuous — moving between cameras is fine.

5. **Stop Sync Display** (`Esc` or `Ctrl+C`). The actual event recording continues.

6. **Stop Recorder** when the event is over (`Ctrl+C`).

7. **Run Sync Analyzer**
   ```bash
   python sync_analyzer.py --input ./recordings/run_001 --out ./synced/run_001
   ```

8. **Review `sync_report.json`** and optionally generate synced clips.

---

## 9. Technology Stack Summary

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Display** | Python + Pygame or PyQt6 | Simple, cross-platform, stable frame timing. |
| **Recording** | FFmpeg 6.0+ | Industry standard, handles RTSP natively, minimal overhead. |
| **Launcher** | Python 3.12 + `subprocess` | Light orchestration, YAML config parsing. |
| **Frame Extraction** | FFmpeg or OpenCV | Fast seeking to keyframes; no full decode needed for detection. |
| **ArUco Detection** | OpenCV `aruco` module | Mature, fast, subpixel accuracy. |
| **QR Decoding** | `pyzbar` or `cv2.QRCodeDetector` | `pyzbar` (ZBar) is generally more robust than OpenCV's native decoder. |
| **Math / Fitting** | NumPy + SciPy | Linear regression, RANSAC outlier rejection. |
| **Optional UI** | PyQt6 | For a visual tool showing detection overlays and allowing manual offset adjustment. |

---

## 10. Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **QR too small in camera frame** | Medium | High | Display QR at least 1/4 of screen height. Instruct operator to hold laptop close or zoom camera. |
| **Moiré / screen filming artifacts prevent QR decode** | Medium | High | Use high QR error correction (30%). Invert colors (white QR on black). Slight camera angle helps. ArUco markers provide backup detectability. |
| **Camera has severe rolling shutter skew** | Low | Medium | 200 ms QR dwell is much longer than rolling-shutter readout; unlikely to split a QR across frames. |
| **Network drop causes gaps in recording** | Medium | Medium | FFmpeg `-rtsp_transport tcp` and MKV container are resilient. Gaps are visible as timeline discontinuities; sync still works as long as some QR frames are present. |
| **Camera clock drift over long recording** | Low | Low | Fit drift coefficient `β_i` during analysis. Correctable unless drift is extreme. |
| **Cannot position display so all cameras see it** | High (for 16 cams) | Low | The mobile walk method works fine. Each camera needs only 5 seconds of visibility. |

---

## 11. Future Enhancements (Post-MVP)

- **Audio chirp**: The display laptop plays a short audible chirp at the start of the QR sequence. This provides a secondary sync channel for cameras that also record audio.
- **Real-time validator**: A companion script that decodes QRs live from one camera feed to verify that the display is actually readable before the operator walks away.
- **Rolling-shutter skew estimation**: Use a vertical line sweep on the display to measure per-camera rolling-shutter readout time.
- **ONVIF integration**: Auto-discover camera RTSP URLs instead of manual YAML configuration.

---

## 12. Success Criteria

This plan is considered successful when:
1. A single operator can sync 16 cameras with no hardware modifications.
2. The sync accuracy is ≤ ±1 video frame (≤ ±33 ms for 30 fps).
3. The entire post-processing step for 16 cameras finishes in under 10 minutes on a standard workstation.
4. No dependency on RTCP SR, NTP, or camera-side timestamping.
