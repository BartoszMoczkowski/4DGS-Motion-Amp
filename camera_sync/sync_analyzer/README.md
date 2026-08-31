# camera-sync-analyzer

Post-processing tool for multi-camera video synchronization using ArUco and QR timestamp detection.

## Overview

`camera-sync-analyzer` extracts sparse frames from multi-camera recordings (MKV/MP4), locates ArUco markers 0 and 1 to define the display ROI, decodes the 5 Hz QR timestamp stream, fits a robust linear model (offset + drift) per camera, computes relative timeline offsets against a reference camera, and optionally outputs remuxed synchronized videos using FFmpeg `-itsoffset`.

## Installation

```bash
uv sync --package camera-sync-analyzer
```

## Usage

### Analyze a multi-camera recording run

```bash
uv run --package camera-sync-analyzer sync-analyzer -i ./recordings/run_001 -o ./synced/run_001 --remux
```

### Self-Test

Run the built-in synthetic self-test to verify algorithm accuracy:

```bash
uv run --package camera-sync-analyzer sync-analyzer --selftest
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Path to recording directory containing MKVs + `recording_manifest.json`, or single video | Required |
| `--out`, `-o` | Output directory for `sync_report.json`, summary, and remuxed clips | Required |
| `--sample-fps` | Frame extraction sampling rate in Hz | `5.0` |
| `--ref-camera` | Camera ID to use as time reference | Auto-select |
| `--save-overlays` | Save annotated detection frames to `output/detection_log/` | `False` |
| `--remux` | Remux synchronized video clips via FFmpeg `-itsoffset` | `False` |
| `--selftest` | Run built-in synthetic test with simulated multi-camera delays | `False` |

## Outputs

- `sync_report.json`: Machine-readable per-camera offsets, drift, RMSE, sample counts, and confidence ratings.
- `sync_report.md`: Markdown summary table.
- `synced/`: Time-aligned video files (when `--remux` is specified).
- `detection_log/`: Diagnostic images showing detected bounding boxes and timestamps (when `--save-overlays` is specified).
