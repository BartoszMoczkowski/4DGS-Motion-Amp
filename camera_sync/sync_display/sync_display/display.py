"""
sync_display.display

Full-screen display showing encoded time pattern for manual camera
synchronization.  Displays:

  - A wall-clock in HH:MM:SS.mmm (updates every frame)
  - A QR code that flips at a fixed cadence (default 5 Hz).  Each QR
    encodes a JSON payload {"u": <unix_ms>, "s": <seq>}.
  - Two ArUco markers (ids 0 and 1) in opposite corners so the
    post-processing analyser can locate the QR region automatically
    even when the camera is angled.

The background is pure black (#000000) for maximum contrast when the
screen is filmed by IP cameras.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame
import qrcode
from PIL import Image
from qrcode.constants import ERROR_CORRECT_H

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QR_SIZE = 400          # pixels
DEFAULT_MARKER_SIZE = 150      # pixels
DEFAULT_UPDATE_HZ = 5.0        # QR refresh rate
DEFAULT_WINDOWED = False

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1)

QR_ECC = ERROR_CORRECT_H       # 30 % recovery

# Background / foreground colours (RGB)
BG_COLOUR = (0, 0, 0)
FG_COLOUR = (255, 255, 255)
DIM_COLOUR = (128, 128, 128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_aruco(marker_id: int, size: int) -> pygame.Surface:
    """Return a pygame Surface containing an ArUco marker.

    Tries the modern OpenCV 4.7+ API first, then falls back to the
    legacy 4.x API, and finally to a plain checkerboard so the app
    still starts even if OpenCV is missing the aruco module.
    """
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    except AttributeError:
        try:
            dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
            marker_img = cv2.aruco.drawMarker(dictionary, marker_id, size)
        except Exception:
            # Last-resort fallback – not detectable as ArUco, but visually
            # distinct enough for manual ROI selection.
            marker_img = np.zeros((size, size), dtype=np.uint8)
            cell = size // 4
            for r in range(4):
                for c in range(4):
                    if (r + c) % 2 == 0:
                        y0, y1 = r * cell, (r + 1) * cell
                        x0, x1 = c * cell, (c + 1) * cell
                        marker_img[y0:y1, x0:x1] = 255

    # OpenCV images are BGR by default, but aruco generators return
    # single-channel.  Convert to RGB for pygame.
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
    return pygame.image.frombuffer(marker_rgb.tobytes(), (size, size), "RGB")


def _generate_qr(data: str, pixel_size: int) -> pygame.Surface:
    """Return a pygame Surface containing a QR code encoding *data*."""
    qr = qrcode.QRCode(
        version=None,                 # auto-fit
        error_correction=QR_ECC,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Render to PIL Image, force RGB, resize to target pixel size.
    pil_img = qr.make_image(fill_color="black", back_color="white")
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((pixel_size, pixel_size), Image.NEAREST)

    return pygame.image.frombuffer(pil_img.tobytes(), (pixel_size, pixel_size), "RGB")


def _load_monospace_font(size: int) -> pygame.font.Font:
    """Try a list of known monospace system fonts, fall back to default."""
    candidates = [
        "consolas",
        "dejavusansmono",
        "liberationmono",
        "couriernew",
        "monospace",
    ]
    for name in candidates:
        try:
            font = pygame.font.SysFont(name, size, bold=True)
            # SysFont returns a Font object even if the name does not exist;
            # it silently falls back to the default font.  We can check whether
            # the name was actually resolved by comparing the font path/name,
            # but that is platform-specific.  Instead we simply accept it –
            # the default pygame font is acceptable for debugging.
            return font
        except Exception:
            continue
    return pygame.font.Font(None, size)


# ---------------------------------------------------------------------------
# Main display class
# ---------------------------------------------------------------------------

class SyncDisplay:
    def __init__(
        self,
        *,
        windowed: bool = DEFAULT_WINDOWED,
        qr_size: int = DEFAULT_QR_SIZE,
        marker_size: int = DEFAULT_MARKER_SIZE,
        update_hz: float = DEFAULT_UPDATE_HZ,
        font_size: int | None = None,
    ) -> None:
        pygame.init()

        self.windowed = windowed
        self.qr_size = qr_size
        self.marker_size = marker_size
        self.update_interval_ms = int(1000.0 / update_hz)

        # ---- display surface ------------------------------------------------
        flags = 0 if windowed else pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((0, 0), flags)
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Camera Sync Display")

        # ---- fonts ----------------------------------------------------------
        if font_size is None:
            font_size = int(self.height * 0.08)
        small_font_size = max(12, int(font_size * 0.22))

        self.font = _load_monospace_font(font_size)
        self.small_font = _load_monospace_font(small_font_size)

        # ---- ArUco markers --------------------------------------------------
        self.marker_surfaces = [
            _generate_aruco(MARKER_IDS[0], marker_size),
            _generate_aruco(MARKER_IDS[1], marker_size),
        ]

        # ---- QR state -------------------------------------------------------
        self.current_qr_surface: pygame.Surface | None = None
        self.last_qr_update_ms = 0
        self.sequence = 0
        self.current_payload = ""
        self._update_qr(0)            # generate first QR immediately

        # ---- timing ---------------------------------------------------------
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.start_time = time.time()

    # ------------------------------------------------------------------

    def _update_qr(self, now_ms: int) -> None:
        """Generate a fresh QR code with the current timestamp."""
        self.sequence += 1
        unix_ms = int(time.time() * 1000)
        payload = {"u": unix_ms, "s": self.sequence}
        self.current_payload = json.dumps(payload, separators=(",", ":"))
        self.current_qr_surface = _generate_qr(self.current_payload, self.qr_size)
        self.last_qr_update_ms = now_ms

    def _toggle_fullscreen(self) -> None:
        """Switch between windowed and fullscreen."""
        try:
            pygame.display.toggle_fullscreen()
        except Exception:
            # Some backends / platforms do not support runtime toggling.
            pass
        self.width, self.height = self.screen.get_size()

    def _draw(self) -> None:
        """Blit one complete frame."""
        self.screen.fill(BG_COLOUR)

        # ---- ArUco markers --------------------------------------------------
        margin = int(self.marker_size * 0.25)
        self.screen.blit(self.marker_surfaces[0], (margin, margin))
        self.screen.blit(
            self.marker_surfaces[1],
            (self.width - self.marker_size - margin,
             self.height - self.marker_size - margin),
        )

        # ---- Clock ----------------------------------------------------------
        now = time.time()
        ms_part = int((now % 1) * 1000)
        clock_text = time.strftime("%H:%M:%S") + f".{ms_part:03d}"
        clock_surf = self.font.render(clock_text, True, FG_COLOUR)
        clock_rect = clock_surf.get_rect(
            center=(self.width // 2, int(self.height * 0.12))
        )
        self.screen.blit(clock_surf, clock_rect)

        # ---- QR code --------------------------------------------------------
        if self.current_qr_surface is not None:
            qr_rect = self.current_qr_surface.get_rect(
                center=(self.width // 2, self.height // 2)
            )
            self.screen.blit(self.current_qr_surface, qr_rect)

        # ---- Footer info ----------------------------------------------------
        elapsed = now - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        footer_lines = [
            f"SEQ {self.sequence:04d}  |  PAYLOAD  {self.current_payload}",
            f"ESC/Q quit  ·  F fullscreen  ·  {fps:.1f} FPS",
        ]
        for i, line in enumerate(footer_lines):
            surf = self.small_font.render(line, True, DIM_COLOUR)
            rect = surf.get_rect(
                center=(
                    self.width // 2,
                    self.height - 30 - i * (self.small_font.get_height() + 4),
                )
            )
            self.screen.blit(surf, rect)

        pygame.display.flip()

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop.  Runs until the user quits."""
        running = True
        while running:
            # ---- events -----------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_f:
                        self._toggle_fullscreen()

            # ---- update QR if interval elapsed ------------------------------
            now_ms = pygame.time.get_ticks()
            if now_ms - self.last_qr_update_ms >= self.update_interval_ms:
                self._update_qr(now_ms)

            # ---- render -----------------------------------------------------
            self._draw()
            self.frame_count += 1
            self.clock.tick(60)          # cap at ~60 fps; VSync where available

        pygame.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync-display",
        description="Full-screen sync display with encoded QR timestamps and ArUco markers.",
    )
    parser.add_argument(
        "--windowed", "-w",
        action="store_true",
        help="Run in a resizable window instead of fullscreen",
    )
    parser.add_argument(
        "--qr-size",
        type=int,
        default=DEFAULT_QR_SIZE,
        metavar="PX",
        help=f"QR code width/height in pixels (default: {DEFAULT_QR_SIZE})",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=DEFAULT_MARKER_SIZE,
        metavar="PX",
        help=f"ArUco marker width/height in pixels (default: {DEFAULT_MARKER_SIZE})",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=DEFAULT_UPDATE_HZ,
        metavar="HZ",
        help=f"How often the QR code refreshes (default: {DEFAULT_UPDATE_HZ})",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        metavar="PX",
        help="Clock font size in pixels (default: auto from screen height)",
    )
    args = parser.parse_args(argv)

    display = SyncDisplay(
        windowed=args.windowed,
        qr_size=args.qr_size,
        marker_size=args.marker_size,
        update_hz=args.update_hz,
        font_size=args.font_size,
    )
    display.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
