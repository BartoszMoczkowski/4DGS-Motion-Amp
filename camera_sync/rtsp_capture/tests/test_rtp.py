"""Offline unit tests for rtsp_capture.rtp and the SR interpolation mapping.

All packets are hand-built synthetic byte strings; no network access.
"""

from __future__ import annotations

import struct

import pytest

from rtsp_capture.rtp import (
    ANNEX_B_PREFIX,
    NTP_UNIX_EPOCH_OFFSET,
    H264Depay,
    H265Depay,
    RtpTsExtender,
    depay_for_encoding,
    parse_rtcp_sr,
    parse_rtp,
)
from rtsp_capture.rtsp_client import parse_sdp
from rtsp_capture.sync_extract import build_rtp_to_ntp

SSRC = 0x12345678


def make_rtp(seq: int, ts: int, payload: bytes, marker: int = 0, pt: int = 96) -> bytes:
    return struct.pack("!BBHII", 0x80, (marker << 7) | pt, seq, ts, SSRC) + payload


def make_rtcp_sr(rtp_ts: int, unix_seconds: float, ssrc: int = SSRC) -> bytes:
    sec = int(unix_seconds) + NTP_UNIX_EPOCH_OFFSET
    frac = int((unix_seconds - int(unix_seconds)) * 2**32)
    body = struct.pack("!IIIIII", ssrc, sec, frac, rtp_ts, 0, 0)
    return struct.pack("!BBH", 0x80, 200, (len(body) // 4)) + body


# ---------------------------------------------------------------------------
# RTP parsing
# ---------------------------------------------------------------------------

class TestParseRtp:
    def test_basic_fields(self):
        payload = b"\x65\x88\x84"
        pkt = parse_rtp(make_rtp(seq=42, ts=90000, payload=payload, marker=1, pt=96))
        assert pkt.seq == 42
        assert pkt.rtp_ts == 90000
        assert pkt.ssrc == SSRC
        assert pkt.marker is True
        assert pkt.payload_type == 96
        assert pkt.payload == payload

    def test_marker_zero(self):
        pkt = parse_rtp(make_rtp(seq=1, ts=0, payload=b"\x41", marker=0))
        assert pkt.marker is False

    def test_too_short(self):
        with pytest.raises(ValueError):
            parse_rtp(b"\x80\x60\x00")

    def test_bad_version(self):
        bad = bytes([0x00, 0x60]) + b"\x00" * 10
        with pytest.raises(ValueError):
            parse_rtp(bad)


# ---------------------------------------------------------------------------
# RTCP Sender Report parsing
# ---------------------------------------------------------------------------

class TestParseRtcpSr:
    def test_ntp_to_unix(self):
        data = make_rtcp_sr(rtp_ts=123456, unix_seconds=1700000000.0)
        ssrc, ntp_unix, rtp_ts = parse_rtcp_sr(data)
        assert ssrc == SSRC
        assert rtp_ts == 123456
        assert ntp_unix == pytest.approx(1700000000.0, abs=1e-6)

    def test_fractional_seconds(self):
        data = make_rtcp_sr(rtp_ts=0, unix_seconds=1700000000.5)
        _, ntp_unix, _ = parse_rtcp_sr(data)
        assert ntp_unix == pytest.approx(1700000000.5, abs=1e-6)

    def test_non_sr_returns_none(self):
        # Receiver Report (PT=201), empty
        rr = struct.pack("!BBH", 0x80, 201, 1) + b"\x00" * 4
        assert parse_rtcp_sr(rr) is None

    def test_compound_packet_sr_second(self):
        rr = struct.pack("!BBH", 0x80, 201, 1) + b"\x00" * 4
        sr = make_rtcp_sr(rtp_ts=777, unix_seconds=1700000001.0)
        result = parse_rtcp_sr(rr + sr)
        assert result is not None
        assert result[2] == 777

    def test_too_short(self):
        assert parse_rtcp_sr(b"\x80") is None


# ---------------------------------------------------------------------------
# RTP timestamp wraparound
# ---------------------------------------------------------------------------

class TestRtpTsExtender:
    def test_no_wrap(self):
        ext = RtpTsExtender()
        assert ext.extend(1000) == 1000
        assert ext.extend(2000) == 2000

    def test_wraparound(self):
        ext = RtpTsExtender()
        assert ext.extend(0xFFFFFFF0) == 0xFFFFFFF0
        assert ext.extend(0x00000010) == 0x100000010
        assert ext.extend(0x00000020) == 0x100000020

    def test_double_wraparound(self):
        ext = RtpTsExtender()
        ext.extend(0xFFFFFFF0)
        ext.extend(0x10)
        ext.extend(0xFFFFFFF0)
        assert ext.extend(0x20) == 0x200000020


# ---------------------------------------------------------------------------
# H.264 depacketizer
# ---------------------------------------------------------------------------

class TestH264Depay:
    def test_single_nal(self):
        depay = H264Depay()
        nal = b"\x65\x88\x84\x00"  # IDR slice
        frames = depay.feed(parse_rtp(make_rtp(1, 100, nal, marker=1)))
        assert frames == [ANNEX_B_PREFIX + nal]

    def test_single_nal_no_marker_means_no_frame(self):
        depay = H264Depay()
        nal = b"\x41\x9a\x00"
        assert depay.feed(parse_rtp(make_rtp(1, 100, nal, marker=0))) == []

    def test_two_nals_one_frame(self):
        depay = H264Depay()
        sps = b"\x67\x42\x00\x1f"
        pps = b"\x68\xce\x06\xe2"
        assert depay.feed(parse_rtp(make_rtp(1, 100, sps, marker=0))) == []
        frames = depay.feed(parse_rtp(make_rtp(2, 100, pps, marker=1)))
        assert frames == [ANNEX_B_PREFIX + sps + ANNEX_B_PREFIX + pps]

    def test_stap_a(self):
        depay = H264Depay()
        sps = b"\x67\x42\x00\x1f"
        pps = b"\x68\xce\x06\xe2"
        stap = b"\x78" + struct.pack("!H", len(sps)) + sps + struct.pack("!H", len(pps)) + pps
        frames = depay.feed(parse_rtp(make_rtp(1, 100, stap, marker=1)))
        assert frames == [ANNEX_B_PREFIX + sps + ANNEX_B_PREFIX + pps]

    def test_fu_a_across_packets(self):
        depay = H264Depay()
        original_nal = b"\x65" + bytes(range(256)) * 4  # type 5 IDR, 1025 bytes
        indicator = 0x7C  # F=0, NRI=3 (0x60), type 28
        part1 = original_nal[1:400]
        part2 = original_nal[400:700]
        part3 = original_nal[700:]
        fu1 = bytes([indicator, 0x80 | 5]) + part1          # start
        fu2 = bytes([indicator, 0x00 | 5]) + part2          # middle
        fu3 = bytes([indicator, 0x40 | 5]) + part3          # end
        assert depay.feed(parse_rtp(make_rtp(1, 100, fu1, marker=0))) == []
        assert depay.feed(parse_rtp(make_rtp(2, 100, fu2, marker=0))) == []
        frames = depay.feed(parse_rtp(make_rtp(3, 100, fu3, marker=1)))
        assert frames == [ANNEX_B_PREFIX + original_nal]

    def test_fu_a_incomplete_dropped(self):
        depay = H264Depay()
        fu1 = bytes([0x7C, 0x85]) + b"\xaa" * 100  # start, never ended
        depay.feed(parse_rtp(make_rtp(1, 100, fu1, marker=0)))
        nal = b"\x41\x9a"
        frames = depay.feed(parse_rtp(make_rtp(2, 200, nal, marker=1)))
        assert frames == [ANNEX_B_PREFIX + nal]


# ---------------------------------------------------------------------------
# H.265 depacketizer
# ---------------------------------------------------------------------------

def h265_header(nal_type: int) -> bytes:
    # F=0, Type (6 bits), LayerId=0, TID=1
    return bytes([(nal_type << 1) & 0x7E, 0x01])


class TestH265Depay:
    def test_single_nal(self):
        depay = H265Depay()
        nal = h265_header(19) + b"\xde\xad\xbe\xef"  # IDR_W_RADL
        frames = depay.feed(parse_rtp(make_rtp(1, 100, nal, marker=1)))
        assert frames == [ANNEX_B_PREFIX + nal]

    def test_aggregation_packet(self):
        depay = H265Depay()
        vps = h265_header(32) + b"\x01\x02"
        sps = h265_header(33) + b"\x03\x04"
        ap = h265_header(48) + struct.pack("!H", len(vps)) + vps + struct.pack("!H", len(sps)) + sps
        frames = depay.feed(parse_rtp(make_rtp(1, 100, ap, marker=1)))
        assert frames == [ANNEX_B_PREFIX + vps + ANNEX_B_PREFIX + sps]

    def test_fu_across_packets(self):
        depay = H265Depay()
        original_nal = h265_header(19) + bytes(range(256)) * 4
        part1 = original_nal[2:400]
        part2 = original_nal[400:700]
        part3 = original_nal[700:]
        fu_hdr = h265_header(49)
        fu1 = fu_hdr + bytes([0x80 | 19]) + part1
        fu2 = fu_hdr + bytes([19]) + part2
        fu3 = fu_hdr + bytes([0x40 | 19]) + part3
        assert depay.feed(parse_rtp(make_rtp(1, 100, fu1, marker=0))) == []
        assert depay.feed(parse_rtp(make_rtp(2, 100, fu2, marker=0))) == []
        frames = depay.feed(parse_rtp(make_rtp(3, 100, fu3, marker=1)))
        assert frames == [ANNEX_B_PREFIX + original_nal]


class TestDepayFactory:
    def test_names(self):
        assert isinstance(depay_for_encoding("H264"), H264Depay)
        assert isinstance(depay_for_encoding("h264"), H264Depay)
        assert isinstance(depay_for_encoding("H265"), H265Depay)
        assert isinstance(depay_for_encoding("HEVC"), H265Depay)

    def test_unsupported(self):
        with pytest.raises(ValueError):
            depay_for_encoding("VP8")


# ---------------------------------------------------------------------------
# SDP parsing
# ---------------------------------------------------------------------------

class TestParseSdp:
    SDP = (
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.3.183\r\n"
        "s=Media Presentation\r\n"
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=framesize:96 1920-1080\r\n"
        "a=control:trackID=0\r\n"
    )

    def test_video_track(self):
        info = parse_sdp(self.SDP)
        assert info["codec"] == "H264"
        assert info["clock_rate"] == 90000
        assert info["payload_type"] == 96
        assert info["control"] == "trackID=0"
        assert info["width"] == 1920
        assert info["height"] == 1080

    def test_h265(self):
        info = parse_sdp("m=video 0 RTP/AVP 96\r\na=rtpmap:96 H265/90000\r\n")
        assert info["codec"] == "H265"

    def test_ignores_audio(self):
        sdp = (
            "m=audio 0 RTP/AVP 8\r\n"
            "a=rtpmap:8 PCMA/8000\r\n"
            + self.SDP
        )
        info = parse_sdp(sdp)
        assert info["codec"] == "H264"


# ---------------------------------------------------------------------------
# SR interpolation mapping
# ---------------------------------------------------------------------------

class TestRtpToNtp:
    SR_POINTS = [(90000 * 10, 1000.0), (90000 * 12, 1002.0)]  # slope exactly 1

    def test_interpolation(self):
        f = build_rtp_to_ntp(self.SR_POINTS, clock_rate=90000)
        assert f(90000 * 11) == pytest.approx(1001.0)
        assert f(90000 * 10) == pytest.approx(1000.0)
        assert f(90000 * 12) == pytest.approx(1002.0)

    def test_extrapolation_edges(self):
        f = build_rtp_to_ntp(self.SR_POINTS, clock_rate=90000)
        assert f(90000 * 9) == pytest.approx(999.0)
        assert f(90000 * 13) == pytest.approx(1003.0)

    def test_single_sr_extrapolates(self):
        f = build_rtp_to_ntp([(90000 * 10, 1000.0)], clock_rate=90000)
        assert f(90000 * 11) == pytest.approx(1001.0)
        assert f(90000 * 9) == pytest.approx(999.0)

    def test_no_srs_raises(self):
        with pytest.raises(ValueError):
            build_rtp_to_ntp([], clock_rate=90000)

    def test_extended_timestamps_past_wrap(self):
        # SRs on either side of the 2**32 boundary (already 64-bit extended)
        points = [(2**32 - 45000, 1000.0), (2**32 + 45000, 1001.0)]
        f = build_rtp_to_ntp(points, clock_rate=90000)
        assert f(2**32) == pytest.approx(1000.5)
