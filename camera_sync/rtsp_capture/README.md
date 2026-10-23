# camera-sync-rtcp

RTSP multi-camera capture with **RTCP Sender Report** cross-camera
synchronization, written against the four Dahua cameras on the lab LAN
(`192.168.3.183/.159/.184/.162`).

Unlike the FFmpeg-based `camera-sync-recorder`, this package implements a
minimal stdlib-only RTSP client over TCP interleaved transport, so it sees
the **RTCP Sender Reports** that Dahua cameras emit (FFmpeg never exposes
them).  Each SR maps one RTP timestamp to one NTP wall-clock timestamp per
camera; piecewise-linear interpolation between SRs timestamps every frame,
giving frame series synced across cameras *without* filming a QR display.

Recording writes the **raw H.264/H.265 Annex-B bitstream** (no decode), so
CPU per camera is negligible.

## Quick Start

### 1. Web Studio (GUI for visual setup, previews, calibration & recording)

```bash
uv run rtsp-gui
# Or with mock mode if cameras are currently offline:
uv run rtsp-gui --mock
```
Opens `http://127.0.0.1:8080`:
- **Multi-Camera Grid:** Low-latency 2 FPS live previews for all cameras.
- **Hardware Identification:** Click **"⚡ Identify"** on any camera card to flash its physical white light illuminator in the rig.
- **Universal Synchronization:** One-click "Apply to All Cameras" locks White Balance gains and Exposure/Shutter speed across all cameras to eliminate color grading discrepancies.
- **Probing & Recording:** One-click pre-flight check and live recording deck with timer and auto-extraction.

### 2. Unified CLI (`rtsp-capture`)

For headless operations and fast automated reruns:

```bash
# Launch Web GUI
uv run rtsp-capture gui [--port 8080] [--mock]

# Diagnostic probe
uv run rtsp-capture probe -c cameras.yaml [-d 5]

# Hardware identify (flash physical camera light)
uv run rtsp-capture flash -c cameras.yaml --camera cam02 [--duration 3]

# Lock white balance and exposure across all cameras
uv run rtsp-capture config -c cameras.yaml --all --wb-red 10 --wb-blue 50 --shutter 1/100

# Headless record
uv run rtsp-capture record -c cameras.yaml -o ./recordings/run_001 [-d 10] [--extract]

# Post-hoc frame synchronization extraction
uv run rtsp-capture extract -i ./recordings/run_001 -o ./synced/run_001
```

## Legacy Commands (Preserved)

```bash
uv run rtsp-probe -c cameras.yaml
uv run rtsp-record -c cameras.yaml -o ./recordings/run_001
uv run rtsp-extract -i ./recordings/run_001 -o ./synced/run_001
```

## Configuration

Copy `cameras.yaml.example` to `cameras.yaml` (gitignored — never commit
real credentials) and fill in the password:

```yaml
credentials:
  user: admin
  password: "CHANGEME"      # or set RTSP_CAM_USER / RTSP_CAM_PASSWORD env vars

cameras:
  cam01: {ip: 192.168.3.183, subtype: 0}   # subtype 0 = main stream
  cam02: {ip: 192.168.3.159, subtype: 0}
  cam03: {ip: 192.168.3.184, subtype: 0}
  cam04: {ip: 192.168.3.162, subtype: 0}
```

Entries with `ip` get the Dahua URL
`rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype={subtype}`;
any entry may instead give a full `url:`.

## Workflow

### 1. Probe (do this first)

```bash
uv run --package camera-sync-rtcp rtsp-probe -c cameras.yaml
```

Connects to every camera for ~10 s (in parallel) and reports auth, codec,
resolution, clock rate, frames received, and **RTCP SRs received (need
≥ 2)**.  Exit code is non-zero if any camera fails.

### 2. One-time camera NTP setup

Sync accuracy across cameras is bounded by how well the **camera clocks**
agree — the RTCP SR timestamps come from each camera's own clock.  Point
all cameras at the **same NTP source** once:

- **Web UI:** `http://<camera-ip>` → Setup → System → General → Date & Time
  → enable **NTP**, set the server address (e.g. a LAN NTP server or
  `pool.ntp.org`), same on every camera.
- **CGI (scriptable):**

  ```
  curl "http://<user>:<pass>@<ip>/cgi-bin/configManager.cgi?action=setConfig&NTP.Enable=true&NTP.Address=<server>&NTP.Port=123&NTP.UpdatePeriod=30"
  ```

Residual per-camera offsets are reported in `sync_manifest.json`
(`per_camera_residuals`), so clock misalignment is visible, not silent.

> **Dahua timezone gotcha (observed on the lab cameras):** the SR NTP
> timestamp is derived from the camera's *local* time using the configured
> timezone — and daylight-saving time is **not** applied.  A camera set to a
> DST zone (e.g. Sarajevo, UTC+1/+2) reports SRs exactly 1 h off during
> summer.  Set **all cameras to the same no-DST timezone** (the lab cameras
> use Beijing, UTC+8: `NTP.TimeZone=13`) in addition to enabling NTP.
> Symptoms are obvious in `sync_manifest.json`: residuals of hours, or a
> year-2000 clock for a camera that was never synced.

### 3. Record

```bash
uv run --package camera-sync-rtcp rtsp-record -c cameras.yaml -o ./recordings/run_001
```

All camera threads connect and PLAY immediately (so SRs accumulate), then:

- **Enter** — start writing bitstreams to disk
- **Ctrl+C** — stop writing, TEARDOWN, close files, write manifest

Dropped connections reconnect automatically with exponential backoff
(1 s → 30 s max).  Output per camera:

```
recordings/run_001/
├── recording_manifest.json    # run id, wall times, per-camera stats
├── cam01.h264                 # raw Annex-B bitstream (or .h265)
├── cam01_frames.jsonl         # {frame_idx, rtp_ts, seq_start, n_bytes, arrival_mono}
├── cam01_rtcp.jsonl           # {ntp_unix, rtp_ts, arrival_mono} per Sender Report
└── cam02.*                    # … per camera
```

### 4. Extract synced frames

```bash
uv run --package camera-sync-rtcp rtsp-extract -i ./recordings/run_001 -o ./synced/run_001
```

Maps each frame's RTP timestamp to NTP wall time via the camera's SRs
(piecewise-linear; extrapolation at the edges uses the SDP clock rate,
90000 Hz by default), aligns all cameras on the reference camera's
timeline (`--ref-camera`, or a uniform `--fps` grid), keeps only instants
where every camera has a frame within `--tolerance` (default: half a frame
interval), and decodes those frames to JPEG with `ffmpeg`:

```
synced/run_001/
├── sync_manifest.json         # per-instant NTP times, per-camera frame idx
│                              # + residual ms, per-camera mean/max residuals
└── synced/
    ├── frame_000001/{cam01.jpg, cam02.jpg, cam03.jpg, cam04.jpg}
    └── …
```

The per-camera residuals in `sync_manifest.json` double as an NTP
misalignment report: if one camera's mean residual sits near the tolerance
bound while others are near zero, that camera's clock is off.

## Scaling notes

- One thread per camera; all work is network/disk I/O (no decode during
  recording).  Bitstream writes go through a 1 MiB `BufferedWriter`.
- Main streams run ~4–8 Mbps per camera → ~160 Mbps total at 20 cameras,
  fine on gigabit Ethernet + SSD.
- Dropped packets are counted per camera from RTP sequence gaps (see
  `gaps` in the recording manifest); reconnects are counted separately.
- Tested target: 4 cameras; architecture is designed for 10–20.
