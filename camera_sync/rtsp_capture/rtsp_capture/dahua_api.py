"""rtsp_capture.dahua_api

Dahua IP Camera HTTP CGI interface and offline simulation engine.

Handles:
- White balance (Auto / Manual / Gains)
- Exposure & Shutter controls
- Picture settings (Brightness, Contrast, Saturation, Sharpness)
- Illuminator / White light controls (On, Off, Auto, Brightness)
- Hardware camera identification (light flashing / pulsing)
- Low-resolution snapshot retrieval (/cgi-bin/snapshot.cgi)
- Complete offline mock simulator for development without live cameras
"""

from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
    from requests.auth import HTTPDigestAuth
except ImportError:  # pragma: no cover
    requests = None
    HTTPDigestAuth = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass
class CameraColorConfig:
    wb_style: str = "Manual"  # "Auto", "Manual", "Custom", "Sunny", "Incandescent"
    gain_red: int = 50        # 0 - 100
    gain_blue: int = 50       # 0 - 100
    gain_green: int = 50      # 0 - 100
    brightness: int = 50      # 0 - 100
    contrast: int = 50        # 0 - 100
    saturation: int = 50      # 0 - 100
    sharpness: int = 50       # 0 - 100


@dataclass
class CameraExposureConfig:
    mode: str = "Manual"     # "Auto", "Manual"
    shutter: str = "1/100"   # "1/25", "1/50", "1/100", "1/250", "1/500", "1/1000"
    gain_max: int = 50       # 0 - 100
    anti_flicker: str = "50Hz"  # "50Hz", "60Hz", "Outdoor"


@dataclass
class CameraLightConfig:
    mode: str = "Off"        # "Manual", "Off", "Auto"
    brightness: int = 100    # 0 - 100
    is_flashing: bool = False


@dataclass
class CameraState:
    camera_id: str
    ip: str
    online: bool = False
    color: CameraColorConfig = field(default_factory=CameraColorConfig)
    exposure: CameraExposureConfig = field(default_factory=CameraExposureConfig)
    light: CameraLightConfig = field(default_factory=CameraLightConfig)
    last_seen: float = 0.0
    error: str | None = None


def _parse_dahua_response(text: str) -> dict[str, str]:
    """Parse key=value pairs from Dahua CGI responses."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


class DahuaController:
    """Controls Dahua IP cameras via HTTP Digest CGI endpoints."""

    def __init__(
        self,
        user: str = "admin",
        password: str = "",
        timeout: float = 2.0,
        mock_mode: bool = False,
    ):
        self.user = user
        self.password = password
        self.timeout = timeout
        self.mock_mode = mock_mode
        self._states: dict[str, CameraState] = {}
        self._lock = threading.Lock()
        self._flash_threads: dict[str, threading.Thread] = {}

    def register_camera(self, camera_id: str, ip: str) -> CameraState:
        with self._lock:
            if camera_id not in self._states:
                self._states[camera_id] = CameraState(camera_id=camera_id, ip=ip)
            return self._states[camera_id]

    def get_state(self, camera_id: str) -> CameraState | None:
        with self._lock:
            return self._states.get(camera_id)

    def get_all_states(self) -> dict[str, CameraState]:
        with self._lock:
            return dict(self._states)

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _cgi_request(
        self,
        ip: str,
        script: str,
        params: dict[str, Any],
        method: str = "GET",
    ) -> tuple[bool, str]:
        if self.mock_mode:
            return True, "OK"
        if not requests:
            return False, "requests library not installed"

        url = f"http://{ip}/cgi-bin/{script}"
        auth = HTTPDigestAuth(self.user, self.password)
        try:
            if method.upper() == "GET":
                resp = requests.get(url, params=params, auth=auth, timeout=self.timeout)
            else:
                resp = requests.post(url, data=params, auth=auth, timeout=self.timeout)
            if resp.status_code == 200:
                if resp.text.strip().lower().startswith("error"):
                    return False, resp.text
                return True, resp.text
            return False, f"HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Color & White Balance
    # ------------------------------------------------------------------

    def get_color_config(self, camera_id: str) -> CameraColorConfig:
        state = self.get_state(camera_id)
        if not state:
            return CameraColorConfig()
        if self.mock_mode:
            return state.color

        parsed_configs = {}
        errors = []
        for name in ("VideoInColor", "VideoInWhiteBalance", "VideoInSharpness"):
            ok, text = self._cgi_request(state.ip, "configManager.cgi", {
                "action": "getConfig",
                "name": name,
            })
            if ok:
                parsed_configs[name] = _parse_dahua_response(text)
            else:
                errors.append(f"{name}: {text.strip()}")
        if errors:
            state.online = False
            state.error = "; ".join(errors)
            return state.color

        color = parsed_configs["VideoInColor"]
        wb = parsed_configs["VideoInWhiteBalance"]
        sharpness = parsed_configs["VideoInSharpness"]
        try:
            state.color = CameraColorConfig(
                wb_style=wb.get("table.VideoInWhiteBalance[0][0].Mode", state.color.wb_style),
                gain_red=int(wb.get("table.VideoInWhiteBalance[0][0].GainRed", state.color.gain_red)),
                gain_blue=int(wb.get("table.VideoInWhiteBalance[0][0].GainBlue", state.color.gain_blue)),
                gain_green=int(wb.get("table.VideoInWhiteBalance[0][0].GainGreen", state.color.gain_green)),
                brightness=int(color.get("table.VideoInColor[0][0].Brightness", state.color.brightness)),
                contrast=int(color.get("table.VideoInColor[0][0].Contrast", state.color.contrast)),
                saturation=int(color.get("table.VideoInColor[0][0].Saturation", state.color.saturation)),
                sharpness=int(sharpness.get("table.VideoInSharpness[0][0].Sharpness", state.color.sharpness)),
            )
        except ValueError as exc:
            state.error = f"Invalid colour config response: {exc}"
        else:
            state.online = True
            state.last_seen = time.time()
            state.error = None
        return state.color

    def set_white_balance(
        self,
        camera_id: str,
        style: str = "Manual",
        gain_red: int | None = None,
        gain_blue: int | None = None,
        gain_green: int | None = None,
    ) -> tuple[bool, str]:
        state = self.get_state(camera_id)
        if not state:
            return False, f"Unknown camera {camera_id}"

        red = max(0, min(100, gain_red if gain_red is not None else state.color.gain_red))
        blue = max(0, min(100, gain_blue if gain_blue is not None else state.color.gain_blue))
        green = max(0, min(100, gain_green if gain_green is not None else state.color.gain_green))
        if self.mock_mode:
            state.color.wb_style = style
            state.color.gain_red = red
            state.color.gain_blue = blue
            state.color.gain_green = green
            return True, "OK"

        params = {
            "action": "setConfig",
            "VideoInWhiteBalance[0][0].Mode": style,
        }
        if style in ("Manual", "Custom"):
            params["VideoInWhiteBalance[0][0].GainRed"] = red
            params["VideoInWhiteBalance[0][0].GainBlue"] = blue
            params["VideoInWhiteBalance[0][0].GainGreen"] = green

        ok, text = self._cgi_request(state.ip, "configManager.cgi", params)
        if not ok:
            state.error = text
            return False, text

        actual = self.get_color_config(camera_id)
        if state.error:
            return False, state.error
        mismatches = []
        if actual.wb_style != style:
            mismatches.append(f"mode={actual.wb_style!r}")
        if style in ("Manual", "Custom"):
            for name, expected in (("gain_red", red), ("gain_blue", blue), ("gain_green", green)):
                value = getattr(actual, name)
                if value != expected:
                    mismatches.append(f"{name}={value}")
        if mismatches:
            message = f"White balance readback mismatch: {', '.join(mismatches)}"
            state.error = message
            return False, message
        return True, "OK"

    def set_picture(
        self,
        camera_id: str,
        brightness: int | None = None,
        contrast: int | None = None,
        saturation: int | None = None,
        sharpness: int | None = None,
    ) -> tuple[bool, str]:
        state = self.get_state(camera_id)
        if not state:
            return False, f"Unknown camera {camera_id}"

        requested = {
            "brightness": max(0, min(100, brightness)) if brightness is not None else None,
            "contrast": max(0, min(100, contrast)) if contrast is not None else None,
            "saturation": max(0, min(100, saturation)) if saturation is not None else None,
            "sharpness": max(0, min(100, sharpness)) if sharpness is not None else None,
        }
        if self.mock_mode:
            for name, value in requested.items():
                if value is not None:
                    setattr(state.color, name, value)
            return True, "OK"

        params = {"action": "setConfig"}
        for name in ("brightness", "contrast", "saturation"):
            value = requested[name]
            if value is not None:
                params[f"VideoInColor[0][0].{name.capitalize()}"] = value
        if requested["sharpness"] is not None:
            params["VideoInSharpness[0][0].Sharpness"] = requested["sharpness"]

        ok, text = self._cgi_request(state.ip, "configManager.cgi", params)
        if not ok:
            state.error = text
            return False, text

        actual = self.get_color_config(camera_id)
        if state.error:
            return False, state.error
        mismatches = [
            f"{name}={getattr(actual, name)}"
            for name, expected in requested.items()
            if expected is not None and getattr(actual, name) != expected
        ]
        if mismatches:
            message = f"Picture readback mismatch: {', '.join(mismatches)}"
            state.error = message
            return False, message
        return True, "OK"

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------

    def get_exposure_config(self, camera_id: str) -> CameraExposureConfig:
        state = self.get_state(camera_id)
        if not state:
            return CameraExposureConfig()
        if self.mock_mode:
            return state.exposure

        ok, text = self._cgi_request(state.ip, "configManager.cgi", {
            "action": "getConfig",
            "name": "VideoInExposure",
        })
        if not ok:
            state.online = False
            state.error = text
            return state.exposure

        parsed = _parse_dahua_response(text)
        raw_mode = parsed.get("table.VideoInExposure[0][0].Mode", state.exposure.mode)
        try:
            speed = int(parsed.get("table.VideoInExposure[0][0].Speed", "0"))
            gain_max = int(parsed.get("table.VideoInExposure[0][0].GainMax", state.exposure.gain_max))
            state.exposure = CameraExposureConfig(
                mode={"0": "Auto", "1": "Manual"}.get(raw_mode, raw_mode),
                shutter=f"1/{speed}" if speed > 0 else state.exposure.shutter,
                gain_max=gain_max,
                anti_flicker=state.exposure.anti_flicker,
            )
        except ValueError as exc:
            state.error = f"Invalid exposure config response: {exc}"
        else:
            state.online = True
            state.last_seen = time.time()
            state.error = None
        return state.exposure

    def set_exposure(
        self,
        camera_id: str,
        mode: str = "Manual",
        shutter: str | None = None,
        gain_max: int | None = None,
    ) -> tuple[bool, str]:
        state = self.get_state(camera_id)
        if not state:
            return False, f"Unknown camera {camera_id}"
        if mode not in ("Auto", "Manual"):
            return False, f"Unsupported exposure mode {mode!r}"

        requested_shutter = shutter or state.exposure.shutter
        speed = 0
        if mode == "Manual":
            try:
                numerator, denominator = requested_shutter.split("/", 1)
                if int(numerator) != 1:
                    raise ValueError
                speed = int(denominator)
                if speed <= 0:
                    raise ValueError
            except ValueError:
                message = f"Invalid shutter {requested_shutter!r}; expected 1/N"
                state.error = message
                return False, message
            requested_shutter = f"1/{speed}"
        gain = max(0, min(100, gain_max if gain_max is not None else state.exposure.gain_max))

        if self.mock_mode:
            state.exposure.mode = mode
            state.exposure.shutter = requested_shutter
            state.exposure.gain_max = gain
            return True, "OK"

        params = {
            "action": "setConfig",
            "VideoInExposure[0][0].Mode": 0 if mode == "Auto" else 1,
            "VideoInExposure[0][0].Speed": speed,
        }
        if gain_max is not None:
            params["VideoInExposure[0][0].GainMax"] = gain

        ok, text = self._cgi_request(state.ip, "configManager.cgi", params)
        if not ok:
            state.error = text
            return False, text

        actual = self.get_exposure_config(camera_id)
        if state.error:
            return False, state.error
        mismatches = []
        if actual.mode != mode:
            mismatches.append(f"mode={actual.mode!r}")
        if mode == "Manual" and actual.shutter != requested_shutter:
            mismatches.append(f"shutter={actual.shutter!r}")
        if gain_max is not None and actual.gain_max != gain:
            mismatches.append(f"gain_max={actual.gain_max}")
        if mismatches:
            message = f"Exposure readback mismatch: {', '.join(mismatches)}"
            state.error = message
            return False, message
        return True, "OK"

    # ------------------------------------------------------------------
    # Lighting & Hardware Identification Flash
    # ------------------------------------------------------------------

    def set_lighting(
        self,
        camera_id: str,
        mode: str = "Manual",
        brightness: int = 100,
    ) -> tuple[bool, str]:
        state = self.get_state(camera_id)
        if not state:
            return False, f"Unknown camera {camera_id}"

        state.light.mode = mode
        state.light.brightness = max(0, min(100, brightness))

        if self.mock_mode:
            return True, "OK"

        params = {
            "action": "setConfig",
            "Lighting[0][0].Mode": mode,
            "Lighting[0][0].Correction": state.light.brightness,
        }
        ok, text = self._cgi_request(state.ip, "configManager.cgi", params)
        return ok, text

    def flash_camera_light(
        self,
        camera_id: str,
        duration_sec: float = 3.0,
        pulses: int = 4,
    ) -> bool:
        """Flash or pulse the camera's light to identify its physical unit."""
        state = self.get_state(camera_id)
        if not state:
            return False

        def _worker() -> None:
            state.light.is_flashing = True
            try:
                dt = duration_sec / (pulses * 2)
                for _ in range(pulses):
                    self.set_lighting(camera_id, mode="Manual", brightness=100)
                    time.sleep(dt)
                    self.set_lighting(camera_id, mode="Off", brightness=0)
                    time.sleep(dt)
            finally:
                self.set_lighting(camera_id, mode="Off", brightness=0)
                state.light.is_flashing = False

        thread = threading.Thread(target=_worker, name=f"flash-{camera_id}", daemon=True)
        thread.start()
        self._flash_threads[camera_id] = thread
        return True

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def batch_set_white_balance(
        self,
        camera_ids: list[str],
        style: str = "Manual",
        gain_red: int = 50,
        gain_blue: int = 50,
        gain_green: int = 50,
    ) -> dict[str, tuple[bool, str]]:
        results = {}
        threads = []
        for cid in camera_ids:
            def _set(c=cid):
                results[c] = self.set_white_balance(c, style, gain_red, gain_blue, gain_green)
            t = threading.Thread(target=_set, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.timeout * 2)
        return results

    def batch_set_picture(
        self,
        camera_ids: list[str],
        brightness: int | None = None,
        contrast: int | None = None,
        saturation: int | None = None,
        sharpness: int | None = None,
    ) -> dict[str, tuple[bool, str]]:
        results = {}
        threads = []
        for cid in camera_ids:
            def _set(c=cid):
                results[c] = self.set_picture(c, brightness, contrast, saturation, sharpness)
            t = threading.Thread(target=_set, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.timeout * 2)
        return results

    def batch_set_exposure(
        self,
        camera_ids: list[str],
        mode: str = "Manual",
        shutter: str = "1/100",
        gain_max: int = 50,
    ) -> dict[str, tuple[bool, str]]:
        results = {}
        threads = []
        for cid in camera_ids:
            def _set(c=cid):
                results[c] = self.set_exposure(c, mode, shutter, gain_max)
            t = threading.Thread(target=_set, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.timeout * 2)
        return results

    # ------------------------------------------------------------------
    # Snapshot / Live Preview
    # ------------------------------------------------------------------

    def get_snapshot(self, camera_id: str, subtype: int = 1) -> bytes:
        """Fetch low-res JPEG snapshot from camera or generate mock test card."""
        state = self.get_state(camera_id)
        if not state:
            return self._generate_mock_frame("unknown", "0.0.0.0")

        if not self.mock_mode and requests:
            url = f"http://{state.ip}/cgi-bin/snapshot.cgi"
            auth = HTTPDigestAuth(self.user, self.password)
            try:
                resp = requests.get(
                    url,
                    params={"channel": 1, "subtype": subtype},
                    auth=auth,
                    timeout=self.timeout,
                )
                if resp.status_code == 200 and resp.content.startswith(b"\xff\xd8"):
                    state.online = True
                    state.last_seen = time.time()
                    return resp.content
            except Exception as exc:
                state.error = str(exc)
                state.online = False

        # Fallback to mock frame
        return self._generate_mock_frame(
            camera_id,
            state.ip,
            color=state.color,
            exposure=state.exposure,
            is_flashing=state.light.is_flashing,
        )

    def _generate_mock_frame(
        self,
        camera_id: str,
        ip: str,
        color: CameraColorConfig | None = None,
        exposure: CameraExposureConfig | None = None,
        is_flashing: bool = False,
        width: int = 480,
        height: int = 270,
    ) -> bytes:
        """Render a synthetic camera preview frame reflecting current settings."""
        color = color or CameraColorConfig()
        exposure = exposure or CameraExposureConfig()

        if Image is None:
            # Fallback 1x1 minimal JPEG
            return (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
                b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
                b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
                b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
                b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
            )

        # Base background color influenced by white balance tint & exposure
        exp_factor = 0.5 + (color.brightness / 100.0) * 0.8
        red_tint = int(30 * exp_factor * (color.gain_red / 50.0))
        blue_tint = int(35 * exp_factor * (color.gain_blue / 50.0))
        green_tint = int(32 * exp_factor * (color.gain_green / 50.0))

        if is_flashing:
            bg_color = (240, 240, 255)
        else:
            bg_color = (
                max(10, min(250, red_tint)),
                max(10, min(250, green_tint)),
                max(15, min(250, blue_tint)),
            )

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        grid_col = (60, 70, 85) if not is_flashing else (180, 180, 200)
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=grid_col, width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=grid_col, width=1)

        bar_height = 24
        bar_colors = [
            (220, 220, 220), (220, 220, 30), (30, 220, 220), (30, 220, 30),
            (220, 30, 220), (220, 30, 30), (30, 30, 220), (20, 20, 20)
        ]
        bw = width // len(bar_colors)
        for i, c in enumerate(bar_colors):
            r = max(0, min(255, int(c[0] * (color.gain_red / 50.0) * exp_factor)))
            g = max(0, min(255, int(c[1] * (color.gain_green / 50.0) * exp_factor)))
            b = max(0, min(255, int(c[2] * (color.gain_blue / 50.0) * exp_factor)))
            draw.rectangle(
                [(i * bw, height - bar_height), ((i + 1) * bw, height)],
                fill=(r, g, b),
            )

        cx, cy = width // 2, height // 2 - 12
        r = 30
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 230, 180), width=2)
        draw.line([(cx - r - 10, cy), (cx + r + 10, cy)], fill=(0, 230, 180), width=1)
        draw.line([(cx, cy - r - 10), (cx, cy + r + 10)], fill=(0, 230, 180), width=1)

        sec = time.time() % 60
        angle = (sec / 60.0) * 2 * math.pi - math.pi / 2
        tx = cx + int(math.cos(angle) * (r - 4))
        ty = cy + int(math.sin(angle) * (r - 4))
        draw.line([(cx, cy), (tx, ty)], fill=(255, 80, 80), width=2)

        header_text = f"[{camera_id.upper()}]  {ip}"
        draw.rectangle([(8, 8), (240, 32)], fill=(0, 0, 0))
        draw.text((14, 12), header_text, fill=(255, 255, 255))

        info_text = (
            f"WB: {color.wb_style} (R:{color.gain_red} B:{color.gain_blue}) | "
            f"EXP: {exposure.shutter} | {time.strftime('%H:%M:%S')}"
        )
        draw.rectangle([(8, height - bar_height - 24), (width - 8, height - bar_height - 4)], fill=(0, 0, 0))
        draw.text((14, height - bar_height - 20), info_text, fill=(0, 230, 180))

        if is_flashing:
            draw.rectangle([(width - 120, 8), (width - 8, 36)], fill=(255, 50, 50))
            draw.text((width - 110, 14), "⚡ FLASH ON", fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
