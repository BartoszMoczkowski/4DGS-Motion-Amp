"""Minimal RTSP client over TCP interleaved transport (stdlib only).

Supports Basic and Digest authentication, SDP parsing for the video track
(codec / clock rate / control URL), and a receive loop that dispatches
RTP (channel 0) through a depacketizer and RTCP Sender Reports (channel 1)
to caller callbacks.

Written for Dahua ``/cam/realmonitor`` streams but uses only standard
RTSP 1.0 requests.
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
import socket
import struct
import time
from typing import Callable
from urllib.parse import urlsplit

from .rtp import RtpTsExtender, depay_for_encoding, parse_rtcp_sr, parse_rtp

USER_AGENT = "camera-sync-rtcp/0.1"


class RtspError(Exception):
    pass


class AuthError(RtspError):
    pass


def redact_url(url: str) -> str:
    """Strip credentials from an RTSP URL for logs and manifests."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    out = f"{parts.scheme}://{host}{parts.path}"
    if parts.query:
        out += f"?{parts.query}"
    return out


def parse_sdp(sdp: str) -> dict:
    """Extract video-track info from an SDP description.

    Returns a dict with keys ``codec``, ``clock_rate``, ``payload_type``,
    ``control`` and optionally ``width`` / ``height``.
    """
    info: dict = {
        "codec": None,
        "clock_rate": 90000,
        "payload_type": None,
        "control": None,
    }
    in_video = False
    for line in sdp.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line.startswith("m="):
            in_video = line.startswith("m=video")
            if in_video and info["payload_type"] is None:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        info["payload_type"] = int(parts[3])
                    except ValueError:
                        pass
        elif not in_video:
            continue
        elif line.startswith("a=rtpmap:"):
            rest = line[len("a=rtpmap:") :]
            _, _, encoding = rest.partition(" ")
            enc_parts = encoding.split("/")
            info["codec"] = enc_parts[0].upper()
            if len(enc_parts) > 1:
                try:
                    info["clock_rate"] = int(enc_parts[1])
                except ValueError:
                    pass
        elif line.startswith("a=control:"):
            info["control"] = line[len("a=control:") :]
        elif line.startswith("a=framesize:"):
            m = re.search(r"(\d+)-(\d+)", line)
            if m:
                info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    return info


class RtspClient:
    """Single-camera RTSP session over TCP interleaved transport."""

    def __init__(self, url: str, timeout: float = 10.0):
        parts = urlsplit(url)
        if parts.scheme != "rtsp":
            raise ValueError(f"not an rtsp:// URL: {url}")
        if not parts.hostname:
            raise ValueError(f"missing host in URL: {url}")
        self.host = parts.hostname
        self.port = parts.port or 554
        self.user = parts.username or ""
        self.password = parts.password or ""
        self.base_url = f"rtsp://{self.host}:{self.port}{parts.path}"
        if parts.query:
            self.base_url += f"?{parts.query}"
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.session_id: str | None = None
        self.codec: str | None = None
        self.clock_rate: int = 90000
        self.sdp_info: dict = {}
        self.track_url: str | None = None
        self._buf = b""
        self._cseq = 0
        self._auth_scheme: str | None = None  # None / "basic" / "digest"
        self._digest_params: dict[str, str] = {}

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._buf = b""

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def describe(self) -> dict:
        status, _, body = self._request(
            "DESCRIBE", self.base_url, {"Accept": "application/sdp"}
        )
        if status != 200:
            raise RtspError(f"DESCRIBE failed: status {status}")
        self.sdp_info = parse_sdp(body.decode("utf-8", errors="replace"))
        if not self.sdp_info.get("codec"):
            raise RtspError("no video track found in SDP")
        self.codec = self.sdp_info["codec"]
        self.clock_rate = self.sdp_info["clock_rate"]
        control = self.sdp_info.get("control")
        if not control or control == "*":
            self.track_url = self.base_url
        elif control.startswith("rtsp://"):
            self.track_url = control
        else:
            self.track_url = self.base_url.rstrip("/") + "/" + control
        return self.sdp_info

    def setup(self) -> None:
        status, headers, _ = self._request(
            "SETUP",
            self.track_url or self.base_url,
            {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"},
        )
        if status != 200:
            raise RtspError(f"SETUP failed: status {status}")
        session = headers.get("session")
        if not session:
            raise RtspError("SETUP response missing Session header")
        self.session_id = session.split(";")[0].strip()

    def play(self) -> None:
        status, _, _ = self._request("PLAY", self.base_url, {"Range": "npt=0.000-"})
        if status != 200:
            raise RtspError(f"PLAY failed: status {status}")

    def teardown(self) -> None:
        try:
            self._request("TEARDOWN", self.base_url)
        except (RtspError, OSError, socket.timeout):
            pass

    # ------------------------------------------------------------------
    # receive loop
    # ------------------------------------------------------------------

    def receive_loop(
        self,
        on_frame: Callable[[bytes, int, int, float], None],
        on_sender_report: Callable[[float, int, float], None],
        stop_event=None,
        keepalive_interval: float = 30.0,
        stats: dict | None = None,
    ) -> None:
        """Dispatch interleaved RTP/RTCP until *stop_event* is set or the
        connection drops (raises).

        ``on_frame(frame_bytes, rtp_ts64, seq_start, arrival_mono)`` fires
        once per completed frame; ``on_sender_report(ntp_unix, rtp_ts64,
        arrival_mono)`` once per RTCP Sender Report.  RTP timestamps are
        already extended past the 2**32 wraparound.  ``stats``, if given,
        accumulates ``packets`` and ``gaps`` (RTP sequence gaps).
        """
        if self.sock is None:
            raise RtspError("not connected")
        depay = depay_for_encoding(self.codec or "")
        extender = RtpTsExtender()
        last_keepalive = time.monotonic()
        frame_start_seq: int | None = None
        expected_seq: int | None = None

        def handle_packet(channel: int, payload: bytes) -> None:
            nonlocal frame_start_seq, expected_seq
            arrival = time.monotonic()
            if channel == 0:
                try:
                    pkt = parse_rtp(payload)
                except ValueError:
                    return
                if stats is not None:
                    stats["packets"] = stats.get("packets", 0) + 1
                    if expected_seq is not None:
                        gap = (pkt.seq - expected_seq) & 0xFFFF
                        if 0 < gap < 0x8000:
                            stats["gaps"] = stats.get("gaps", 0) + gap
                    expected_seq = (pkt.seq + 1) & 0xFFFF
                if frame_start_seq is None:
                    frame_start_seq = pkt.seq
                for frame in depay.feed(pkt):
                    on_frame(frame, extender.extend(pkt.rtp_ts), frame_start_seq, arrival)
                    frame_start_seq = None
            elif channel == 1:
                sr = parse_rtcp_sr(payload)
                if sr is not None:
                    _, ntp_unix, rtp_ts = sr
                    on_sender_report(ntp_unix, extender.extend(rtp_ts), arrival)

        self.sock.settimeout(1.0)
        while stop_event is None or not stop_event.is_set():
            try:
                self._fill(1)
            except socket.timeout:
                if time.monotonic() - last_keepalive >= keepalive_interval:
                    self._request("OPTIONS", self.base_url, on_packet=handle_packet)
                    last_keepalive = time.monotonic()
                continue
            if self._buf[:1] == b"$":
                header = self._read_exact(4)
                payload = self._read_exact(struct.unpack("!H", header[2:4])[0])
                handle_packet(header[1], payload)
            else:
                self._read_response(on_packet=handle_packet)
            if time.monotonic() - last_keepalive >= keepalive_interval:
                self._request("OPTIONS", self.base_url, on_packet=handle_packet)
                last_keepalive = time.monotonic()

    # ------------------------------------------------------------------
    # request/response machinery
    # ------------------------------------------------------------------

    def _fill(self, n: int = 1) -> None:
        assert self.sock is not None
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RtspError("connection closed by server")
            self._buf += chunk

    def _read_exact(self, n: int) -> bytes:
        self._fill(n)
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _read_line(self, on_packet=None) -> str:
        """Read one CRLF-terminated line, dispatching any interleaved
        packets encountered first (only possible while streaming)."""
        while True:
            self._fill(1)
            if self._buf[:1] == b"$" and on_packet is not None:
                header = self._read_exact(4)
                payload = self._read_exact(struct.unpack("!H", header[2:4])[0])
                on_packet(header[1], payload)
                continue
            idx = self._buf.find(b"\r\n")
            if idx >= 0:
                line, self._buf = self._buf[:idx], self._buf[idx + 2 :]
                return line.decode("latin-1")
            self._fill(len(self._buf) + 1)

    def _read_response(self, on_packet=None) -> tuple[int, dict, bytes]:
        status_line = self._read_line(on_packet)
        m = re.match(r"RTSP/\d\.\d\s+(\d+)\s*(.*)", status_line)
        if not m:
            raise RtspError(f"bad RTSP status line: {status_line!r}")
        status = int(m.group(1))
        headers: dict[str, str] = {}
        while True:
            line = self._read_line(on_packet)
            if not line:
                break
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        body = b""
        if "content-length" in headers:
            body = self._read_exact(int(headers["content-length"]))
        return status, headers, body

    def _request(
        self,
        method: str,
        url: str,
        extra_headers: dict | None = None,
        on_packet=None,
    ) -> tuple[int, dict, bytes]:
        if self.sock is None:
            raise RtspError("not connected")
        for attempt in (0, 1):
            self._cseq += 1
            headers = {"CSeq": str(self._cseq), "User-Agent": USER_AGENT}
            if self.session_id:
                headers["Session"] = self.session_id
            if self._auth_scheme:
                headers["Authorization"] = self._build_auth(method, url)
            if extra_headers:
                headers.update(extra_headers)
            lines = [f"{method} {url} RTSP/1.0"]
            lines += [f"{k}: {v}" for k, v in headers.items()]
            request = "\r\n".join(lines) + "\r\n\r\n"
            self.sock.sendall(request.encode("latin-1"))
            status, resp_headers, body = self._read_response(on_packet)
            if status != 401 or attempt == 1:
                if status == 401:
                    raise AuthError("authentication failed (check credentials)")
                return status, resp_headers, body
            challenge = resp_headers.get("www-authenticate", "")
            if not self.user:
                raise AuthError("server requires authentication but no credentials given")
            if challenge.lower().startswith("digest"):
                self._auth_scheme = "digest"
                self._digest_params = self._parse_auth_params(challenge)
            elif challenge.lower().startswith("basic"):
                self._auth_scheme = "basic"
            else:
                raise AuthError(f"unsupported auth challenge: {challenge!r}")
        raise AuthError("authentication failed")  # unreachable

    @staticmethod
    def _parse_auth_params(challenge: str) -> dict[str, str]:
        return {
            m.group(1).lower(): m.group(2) if m.group(2) is not None else m.group(3)
            for m in re.finditer(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', challenge)
        }

    def _build_auth(self, method: str, uri: str) -> str:
        if self._auth_scheme == "basic":
            token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            return f"Basic {token}"
        realm = self._digest_params.get("realm", "")
        nonce = self._digest_params.get("nonce", "")
        ha1 = hashlib.md5(f"{self.user}:{realm}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        qop = self._digest_params.get("qop", "").split(",")[0].strip()
        if qop:
            nc = "00000001"
            cnonce = secrets.token_hex(8)
            response = hashlib.md5(
                f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()
            ).hexdigest()
            return (
                f'Digest username="{self.user}", realm="{realm}", nonce="{nonce}", '
                f'uri="{uri}", response="{response}", qop={qop}, nc={nc}, '
                f'cnonce="{cnonce}"'
            )
        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
        return (
            f'Digest username="{self.user}", realm="{realm}", nonce="{nonce}", '
            f'uri="{uri}", response="{response}"'
        )
