# Known issue: instance/semantic segmentation PNGs look pure black

**Status:** diagnosed 2026-08-05 — **not a bug in the data; cosmetic display issue.**

## Symptom

`instance_segmentation_XXXX.png` and `semantic_segmentation_XXXX.png` in capture
output dirs (e.g. `Q:/Omniverse/renders/capture_pump_warehouse/cam01/`) open as pure
black in image viewers.

## Root cause

`omni_capture.py` initializes Replicator's `BasicWriter` with
`colorize_instance_segmentation=False` and `colorize_semantic_segmentation=False`
(`omniverse-pipeline/omniverse_pipeline/omni_capture.py:240-241`). The writer therefore
saves **raw label IDs as 16-bit grayscale PNGs** (`I;16`). Part IDs are small integers
(0–99 measured in the warehouse capture); in a 16-bit image (max 65535) those values
are ≤0.16% of full scale, so every standard viewer renders them as black.

## Evidence

Measured on `capture_pump_warehouse/cam01/*_segmentation_0000.png`:

- mode `I;16`, shape 900×1600, values 0–99, **49 unique labels** — the segmentation
  content is fully present.
- The accompanying `instance_segmentation_mapping_XXXX.json` maps IDs to prim paths
  (`0 -> BACKGROUND`, `1 -> UNLABELLED`, `2 -> /World/pump/frame_base`,
  `3 -> /World/pump/part_001`, ...), so semantics are reaching Replicator correctly
  through the referenced `/World/pump` subject.

Instance and semantic masks are identical here by design: each part gets exactly one
`class` label equal to its name, so instance == semantic.

## How to read the masks

```python
import numpy as np
from PIL import Image
ids = np.array(Image.open("instance_segmentation_0000.png"))  # uint16 label IDs
```

## Optional fixes (not applied)

- Set `colorize_instance_segmentation=True` / `colorize_semantic_segmentation=True` in
  `omni_capture.py` for human-inspectable color PNGs (the raw-ID output should be kept
  for GT use — colorized PNGs lose exact IDs).
- Or normalize on inspection: `Image.fromarray((ids * (65535 // ids.max())).astype(np.uint16))`.
- Or convert to an 8-bit palette image in `omni_to_4dgs.py` for QA previews.
