# camera_sync — Manual Camera Synchronization Toolkit

This folder contains three independently installable Python packages for
synchronizing up to 16 IP cameras that stream RTSP/RTP **without** RTCP
Sender Reports.  The workflow is entirely post-hoc: you record all cameras
while they briefly film a laptop screen displaying an encoded time pattern,
then analyze the recordings to extract per-camera offsets.

## Packages

| Package | Path | Purpose | Status |
|---------|------|---------|--------|
| `camera-sync-display` | `sync_display/` | Full-screen QR + clock + ArUco display | **Implemented** |
| `camera-sync-recorder` | `recorder/` | FFmpeg multi-camera recording wrapper | **Implemented** |
| `camera-sync-analyzer` | `sync_analyzer/` | QR detection, offset fitting, reporting | Stub / planned |

## Quick Start

### 1. Sync Display

```bash
uv sync --package camera-sync-display
uv run --package camera-sync-display sync-display --windowed
```

Controls: **ESC** or **Q** — quit; **F** — toggle fullscreen.

The display shows a digital clock, a QR code that refreshes at 5 Hz
(encoding `{"u": <unix_ms>, "s": <seq>}`), and two ArUco markers
for automatic region detection later.

### 2. Recorder

Create a `cameras.yaml` (see `recorder/cameras.yaml.example`):

```yaml
cameras:
  cam01:
    url: rtsp://192.168.1.101:554/stream
  cam02:
    url: rtsp://192.168.1.102:554/stream
```

Run the recorder:

```bash
uv sync --package camera-sync-recorder
uv run --package camera-sync-recorder recorder -c cameras.yaml -o ./recordings/run_001
```

Press **Ctrl+C** to stop.  The output directory will contain:

```
recordings/run_001/
├── recording_manifest.json      # sidecar metadata
├── cam01.mkv
├── cam02.mkv
└── logs/
    ├── cam01.log                # FFmpeg stderr
    └── cam02.log
```

CLI options:
```bash
recorder --help
# --config, -c     YAML file with camera definitions (required)
# --out, -o        Output directory (required)
# --transport      RTSP transport: tcp (default) or udp
# --duration, -d   Maximum recording duration in seconds
# --timeout        Per-camera startup timeout (default: 15s)
```

### 3. Analyzer (planned)

```bash
# Not yet implemented
uv run --package camera-sync-analyzer sync-analyzer -i ./recordings/run_001 -o ./synced/run_001
```

## Full Workflow

```bash
# 1. Start recording all cameras
uv run --package camera-sync-recorder recorder -c cameras.yaml -o ./recordings/run_001

# 2. Show the sync display to every camera for ~5 seconds each
uv run --package camera-sync-display sync-display

# 3. Stop recording (Ctrl+C)

# 4. Analyse and compute offsets (future)
# uv run --package camera-sync-analyzer sync-analyzer -i ./recordings/run_001 -o ./synced/run_001
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  sync-display   │     │    recorder     │     │ sync-analyzer   │
│   (Pygame)      │     │   (FFmpeg)      │     │  (OpenCV+pyzbar)│
│                 │     │                 │     │                 │
│  · Clock        │     │  · 16× RTSP     │     │  · ArUco detect │
│  · QR @ 5 Hz    │     │  · -copyts      │     │  · QR decode    │
│  · ArUco corners│     │  · MKV + JSON   │     │  · Offset fit   │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │
         └──────────┬────────────┘
                    │
            All cameras film
            the display briefly
```

## Design Notes

- Each package has its own `pyproject.toml` and can be installed in
  isolation.  They share no runtime code.
- The **QR payload schema** (`{"u": <int>, "s": <int>}`) is the only
  contract between the display and the analyser.
- FFmpeg is a runtime dependency for `recorder` and `sync-analyzer`
  (frame extraction) but is **not** managed by uv — it must be installed
  on the host system.
- The recorder uses **stream-copy** (`-c copy`) so CPU load is minimal
  even with 16 parallel feeds.  All heavy lifting is I/O bound.
