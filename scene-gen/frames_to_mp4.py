#!/usr/bin/env python3
"""
frames_to_mp4.py — compose omni_capture RGB frames into mp4 videos.

Writes one mp4 per camera (<capture_dir>/<cam>/rgb -> <out_dir>/<cam>.mp4) plus a
tiled overview mp4 with all cameras in a grid (2 rows x 5 cols for 10 cams), each
tile labeled with its camera name. Uses imageio-ffmpeg (bundled ffmpeg binary),
so no system ffmpeg is needed.

Usage:
    python scene-gen/frames_to_mp4.py --capture Q:/Omniverse/renders/capture_pump_A16mm_M8 \
        --out-dir Q:/Omniverse/renders/capture_pump_A16mm_M8/videos [--fps 24]
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np


def _collect(capture_dir: str):
    cams = {}
    for entry in sorted(os.listdir(capture_dir)):
        rgb_dir = os.path.join(capture_dir, entry, "rgb")
        if os.path.isdir(rgb_dir):
            frames = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg")))
            if frames:
                cams[entry] = [os.path.join(rgb_dir, f) for f in frames]
    if not cams:
        raise SystemExit(f"[mp4] no cam*/rgb frames found under {capture_dir}")
    return cams


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a small camera label in the top-left corner (Pillow default font)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, 4 + 9 * len(text) + 8, 24], fill=(0, 0, 0))
    d.text((10, 8), text, fill=(255, 255, 255))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="omni_capture output dir")
    ap.add_argument("--out-dir", default=None, help="default: <capture>/videos")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--tile-cols", type=int, default=5)
    ap.add_argument("--tile-width", type=int, default=480,
                    help="per-tile width in the overview video")
    ap.add_argument("--no-per-cam", action="store_true", help="skip per-camera mp4s")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import Image

    out_dir = args.out_dir or os.path.join(args.capture, "videos")
    os.makedirs(out_dir, exist_ok=True)
    cams = _collect(args.capture)
    names = sorted(cams)
    n_frames = min(len(v) for v in cams.values())
    print(f"[mp4] {len(names)} cameras, {n_frames} frames -> {out_dir}")

    if not args.no_per_cam:
        for name in names:
            out = os.path.join(out_dir, f"{name}.mp4")
            with imageio.get_writer(out, fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=2) as w:
                for f in cams[name][:n_frames]:
                    w.append_data(np.array(Image.open(f).convert("RGB")))
            print(f"[mp4] {out}")

    # tiled overview
    cols = min(args.tile_cols, len(names))
    rows = int(np.ceil(len(names) / cols))
    sample = Image.open(cams[names[0]][0])
    aspect = sample.height / sample.width
    tw, th = args.tile_width, int(args.tile_width * aspect)
    th += th % 2  # h264 wants even dims
    out = os.path.join(out_dir, "overview.mp4")
    with imageio.get_writer(out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=2) as w:
        for fi in range(n_frames):
            tiles = []
            for name in names:
                fr = Image.open(cams[name][fi]).convert("RGB").resize((tw, th))
                tiles.append(_label(np.array(fr), name))
            grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
            for i, t in enumerate(tiles):
                r, c = divmod(i, cols)
                grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
            w.append_data(grid)
    print(f"[mp4] {out}")


if __name__ == "__main__":
    main()
