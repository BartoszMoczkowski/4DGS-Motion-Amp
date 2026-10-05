"""RTP packet parsing, RTCP Sender Report parsing, and H.264/H.265
depacketization to Annex-B byte streams.

All timestamps returned here are raw 32-bit RTP timestamps; use
:class:`RtpTsExtender` to extend them to monotonic 64-bit values across
the 2**32 wraparound boundary.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

NTP_UNIX_EPOCH_OFFSET = 2208988800  # seconds between 1900-01-01 and 1970-01-01

ANNEX_B_PREFIX = b"\x00\x00\x00\x01"

RTCP_PT_SR = 200


class RtpPacket(NamedTuple):
    seq: int
    rtp_ts: int
    ssrc: int
    marker: bool
    payload: bytes
    payload_type: int


def parse_rtp(data: bytes) -> RtpPacket:
    """Parse one RTP packet.  Raises ValueError on malformed input."""
    if len(data) < 12:
        raise ValueError("RTP packet too short")
    b0, b1, seq, rtp_ts, ssrc = struct.unpack("!BBHII", data[:12])
    if b0 >> 6 != 2:
        raise ValueError(f"not an RTP packet (version {b0 >> 6})")
    csrc_count = b0 & 0x0F
    has_extension = (b0 >> 4) & 1
    offset = 12 + 4 * csrc_count
    if len(data) < offset:
        raise ValueError("RTP packet truncated (CSRC list)")
    if has_extension:
        if len(data) < offset + 4:
            raise ValueError("RTP packet truncated (extension header)")
        ext_words = struct.unpack("!H", data[offset + 2 : offset + 4])[0]
        offset += 4 + 4 * ext_words
        if len(data) < offset:
            raise ValueError("RTP packet truncated (extension body)")
    return RtpPacket(
        seq=seq,
        rtp_ts=rtp_ts,
        ssrc=ssrc,
        marker=bool(b1 & 0x80),
        payload=data[offset:],
        payload_type=b1 & 0x7F,
    )


def parse_rtcp_sr(data: bytes) -> tuple[int, float, int] | None:
    """Parse an RTCP Sender Report (PT=200), possibly inside a compound packet.

    Returns ``(ssrc, ntp_unix_seconds, rtp_ts)`` or ``None`` when the packet
    contains no Sender Report block.
    """
    offset = 0
    while offset + 4 <= len(data):
        b0, pt, length_words = struct.unpack("!BBH", data[offset : offset + 4])
        if b0 >> 6 != 2:
            return None
        block_len = (length_words + 1) * 4
        if pt == RTCP_PT_SR:
            if offset + 28 > len(data):
                return None
            ssrc, ntp_sec, ntp_frac, rtp_ts = struct.unpack(
                "!IIII", data[offset + 4 : offset + 20]
            )
            ntp_unix = (ntp_sec - NTP_UNIX_EPOCH_OFFSET) + ntp_frac / 2**32
            return ssrc, ntp_unix, rtp_ts
        offset += block_len
    return None


class RtpTsExtender:
    """Extend 32-bit RTP timestamps to a monotonic 64-bit range.

    Feed timestamps in arrival order; each forward wrap past 2**32 adds
    another 2**32 to the returned value.
    """

    def __init__(self) -> None:
        self._wraps = 0
        self._last: int | None = None

    def extend(self, ts: int) -> int:
        if self._last is None:
            self._last = ts
            return ts
        if ts < self._last and (self._last - ts) > 0x80000000:
            self._wraps += 1
        self._last = ts
        return ts + (self._wraps << 32)


class H264Depay:
    """H.264 RTP payload (RFC 6184) -> Annex-B frames.

    Handles single NAL units, STAP-A aggregation and FU-A fragmentation.
    Feed packets in arrival order; each call returns the list of completed
    frames (empty unless the packet carried the marker bit).
    """

    codec = "h264"
    extension = ".h264"

    def __init__(self) -> None:
        self._nals: list[bytes] = []
        self._fu_buf = bytearray()

    def feed(self, packet: RtpPacket) -> list[bytes]:
        payload = packet.payload
        if payload:
            nal_type = payload[0] & 0x1F
            if 1 <= nal_type <= 23:
                self._nals.append(payload)
            elif nal_type == 24:
                self._feed_stap_a(payload[1:])
            elif nal_type == 28:
                self._feed_fu_a(payload)
        if packet.marker:
            return self._flush_frame()
        return []

    def _flush_frame(self) -> list[bytes]:
        self._fu_buf.clear()  # drop any incomplete FU-A
        if not self._nals:
            return []
        frame = b"".join(ANNEX_B_PREFIX + nal for nal in self._nals)
        self._nals.clear()
        return [frame]

    def _feed_stap_a(self, data: bytes) -> None:
        while len(data) >= 2:
            size = struct.unpack("!H", data[:2])[0]
            data = data[2:]
            if len(data) < size:
                break
            self._nals.append(data[:size])
            data = data[size:]

    def _feed_fu_a(self, payload: bytes) -> None:
        if len(payload) < 2:
            return
        indicator, header = payload[0], payload[1]
        start = bool(header & 0x80)
        end = bool(header & 0x40)
        nal_type = header & 0x1F
        if start:
            self._fu_buf = bytearray([(indicator & 0xE0) | nal_type])
            self._fu_buf += payload[2:]
        elif self._fu_buf:
            self._fu_buf += payload[2:]
        if end and self._fu_buf:
            self._nals.append(bytes(self._fu_buf))
            self._fu_buf = bytearray()


class H265Depay:
    """H.265 RTP payload (RFC 7798) -> Annex-B frames.

    Handles single NAL units, aggregation packets (type 48) and
    fragmentation units (type 49).  DONL/DOND fields are not supported
    (Dahua cameras do not emit them).
    """

    codec = "h265"
    extension = ".h265"

    def __init__(self) -> None:
        self._nals: list[bytes] = []
        self._fu_buf = bytearray()

    def feed(self, packet: RtpPacket) -> list[bytes]:
        payload = packet.payload
        if len(payload) >= 2:
            nal_type = (payload[0] >> 1) & 0x3F
            if nal_type <= 47:
                self._nals.append(payload)
            elif nal_type == 48:
                self._feed_ap(payload[2:])
            elif nal_type == 49:
                self._feed_fu(payload)
        if packet.marker:
            return self._flush_frame()
        return []

    def _flush_frame(self) -> list[bytes]:
        self._fu_buf.clear()
        if not self._nals:
            return []
        frame = b"".join(ANNEX_B_PREFIX + nal for nal in self._nals)
        self._nals.clear()
        return [frame]

    def _feed_ap(self, data: bytes) -> None:
        while len(data) >= 2:
            size = struct.unpack("!H", data[:2])[0]
            data = data[2:]
            if len(data) < size:
                break
            self._nals.append(data[:size])
            data = data[size:]

    def _feed_fu(self, payload: bytes) -> None:
        if len(payload) < 3:
            return
        fu_header = payload[2]
        start = bool(fu_header & 0x80)
        end = bool(fu_header & 0x40)
        fu_type = fu_header & 0x3F
        if start:
            byte0 = (payload[0] & 0x81) | (fu_type << 1)
            self._fu_buf = bytearray([byte0, payload[1]])
            self._fu_buf += payload[3:]
        elif self._fu_buf:
            self._fu_buf += payload[3:]
        if end and self._fu_buf:
            self._nals.append(bytes(self._fu_buf))
            self._fu_buf = bytearray()


def depay_for_encoding(encoding_name: str) -> H264Depay | H265Depay:
    """Instantiate the depacketizer matching an SDP rtpmap encoding name."""
    name = encoding_name.upper()
    if name in ("H264", "AVC"):
        return H264Depay()
    if name in ("H265", "HEVC"):
        return H265Depay()
    raise ValueError(f"unsupported RTP encoding: {encoding_name}")
