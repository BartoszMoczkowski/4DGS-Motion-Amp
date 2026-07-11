# Viseron NVR setup notes

Date: 2026-07-05
Location: `viseron/` (docker-compose.yml + config/config.yaml)

## What this is

Viseron (self-hosted NVR / CV) was set up via `docker compose` per the official
install docs (https://viseron.netlify.app/docs/documentation/installation),
recording a single RTSP camera (`camera_1`, host 192.168.3.177).

## Issues hit and fixes

1. **"NVR is idle" blocking manual recording.**
   Root cause: `nvr.operation_state` only leaves `IDLE` when the camera is
   actively recording or a motion/object detector is actively scanning
   (`viseron/components/nvr/nvr.py::update_operation_state`). With only
   `ffmpeg.camera` + `nvr:` configured (no detector), it's always idle, and
   the manual-recording API (`components/webserver/api/v1/camera.py`)
   explicitly refuses to start while idle.
   Fix: added a `background_subtractor` motion detector for `camera_1`.
   Its default `trigger_event_recording: false` means it does NOT start
   automatic recordings — it only keeps the NVR out of the idle state so the
   manual record button works.

2. **Recording works, but playback 404s ("Recording with id X not found" /
   "Could not find init.mp4 file for camera camera_1").**
   Root cause: Docker Desktop on Windows bind-mounts (`./segments:/segments`
   etc., backing a `C:\Users\...` folder) do not reliably deliver inotify
   file-created events into the container. Viseron's storage component
   (`components/storage/tier_handler.py`) uses native inotify
   (`watchdog.observers.Observer`) by default to notice new recording
   segments and register them in its Postgres `Files` table. Without that,
   segments are written to disk correctly (confirmed on disk) but never
   indexed in the DB, so the HLS playlist endpoint can't find them — even
   though the files genuinely exist.
   Fix: added a `storage` block setting `poll: true` on both the `recorder`
   and `snapshots` tiers (path `/`), which switches to `PollingObserverVFS`
   instead of inotify. This is the standard workaround for Docker
   bind-mount/network-filesystem setups.

   Side note: an intermediate detour switched the image to
   `roflcoopter/amd64-cuda-viseron:latest` without wiring up
   `runtime: nvidia`, which is unstable without GPU passthrough and likely
   caused container restarts (possibly explaining an earlier orphaned
   "recording id 1" disappearing). Reverted to the plain image; if GPU accel
   is wanted later, `runtime: nvidia` line is already present (commented
   guidance) in `docker-compose.yml` — needs NVIDIA Container Toolkit on
   the host.

## Current config summary (`config/config.yaml`)

- `ffmpeg.camera.camera_1`: RTSP camera at 192.168.3.177:554/stream1
- `background_subtractor.motion_detector.cameras.camera_1`: motion scanning
  on, not set to auto-trigger recording
- `storage.recorder.tiers` / `storage.snapshots.tiers`: `path: /`, `poll: true`
  (Windows bind-mount workaround, see above)
- `nvr.camera_1`: ties it together
- `go2rtc.streams.camera_1`: RTSP restream for live view

## Multi-camera time sync (NTP) — researched 2026-07-05

Question: does Viseron have built-in NTP sync options for cameras?

**No.** Viseron has no per-camera NTP configuration surface (unlike Hikvision/
Dahua NVRs with a "Sync Time with NVR" toggle per channel). It ingests each
camera's RTSP/H.264 stream via FFmpeg and timestamps on arrival at the host —
it doesn't manage or query camera clocks at all.

Evidence: GitHub Q&A #1030 (roflcoopter/viseron) — user saw cameras drift
30-60s out of sync during playback. Root cause turned out to be GOP/B-frame
encoder settings (Axis Zipstream Storage Profile) at low framerate, not clock
drift. The maintainer's own troubleshooting step for suspected clock issues
was checking the *host's* wall clock (`date`), confirming Viseron relies on
host system time, not camera-side NTP.

**What to actually do for multi-camera sync:**
1. Configure NTP on each camera individually via its own admin web UI
   (Date & Time > NTP), pointed at one common NTP server (e.g. local chrony).
2. NTP-sync the Viseron host itself (chrony/systemd-timesyncd) — this is the
   clock Viseron timestamps against.
3. Caveat: LAN NTP typically only gets ~single-digit-to-tens-of-ms accuracy,
   and RTSP adds its own variable network/buffering latency on top that
   Viseron doesn't compensate for. May not be tight enough for frame-accurate
   multi-view capture feeding 4DGS reconstruction of subtle periodic motion.
4. If frame-accurate sync becomes necessary: check whether cameras support
   PTP/IEEE 1588 (sub-microsecond, some higher-end Axis models do); hardware
   genlock/trigger input (rare on consumer IP cams); or fall back to a
   physical fiducial (LED flash/clap visible to all cameras) to calibrate
   per-recording offsets post-hoc.

## LAN NTP sync plan (decided 2026-07-06)

Precision requirement relaxed: cameras max out at 60fps (16.7ms/frame), and LAN
NTP accuracy (~single-digit-to-tens of ms) is acceptable — no need for
PTP/genlock/hardware trigger. Decided to keep Viseron rather than switch NVRs.

Adjustments needed to get synced recordings out of Viseron:

1. **Host clock** (this is what actually drives Viseron's recording
   timestamps, since Viseron stamps on FFmpeg frame arrival, not camera
   clock): install/configure chrony on the Docker host against the LAN NTP
   source. Verify `chronyc tracking`, and confirm container time matches
   host (`docker exec <container> date`).
2. **Camera clocks** (only matters for camera-side OSD/embedded metadata,
   not for Viseron's own recording timestamps): set NTP server on each
   camera's own admin UI, same source as host. Must verify it actually took
   (cheap cameras often ignore NTP config) via OSD or ONVIF
   `GetSystemDateAndTime` — don't trust the settings screen. Shorten poll
   interval if adjustable (5-10 min vs daily).
3. **Network path consistency matters more than clocks for cross-camera
   alignment in the actual recordings**: wired Ethernet only, same
   switch/LAN segment for all cameras, consistent `-rtsp_transport tcp` and
   buffering flags across camera configs in Viseron. This is because
   Viseron's segment timestamps come from host arrival time, so skew is
   dominated by per-camera network/decode latency variance, not clock
   offset.
4. **Avoid adaptive/dynamic-GOP encoder features** (e.g. Axis Zipstream
   Storage Profile) — confirmed root cause of a real Viseron cross-camera
   desync case (GitHub discussion #1030) unrelated to clocks. Use fixed
   frame rate encoding.
5. **Verify empirically, not just by config**: physical sync test (LED
   flash/clap visible to all cameras) then check actual offset in recorded
   segments/frames across cameras to confirm within the ~16.7ms budget.

## Camera-clock (RTCP-based) timestamps instead of host-arrival — researched 2026-07-06

Investigated whether Viseron can use each camera's own clock (via RTCP Sender
Report NTP↔RTP mapping) for recording timestamps instead of host arrival
time, since that would be strictly better than LAN-NTP-synced host stamping.

**Confirmed by reading Viseron source directly** (not just docs):
- `viseron/components/ffmpeg/const.py::CAMERA_INPUT_ARGS` hardcodes
  `-use_wallclock_as_timestamps 1` in the default input args, along with
  `-avoid_negative_ts make_zero -fflags +nobuffer+genpts+discardcorrupt
  -flags low_delay -strict experimental -err_detect ignore_err -vsync 0`.
  This flag explicitly discards RTP/RTCP-derived capture-time and stamps
  frames by host arrival — confirmed this is what both the live pipe *and*
  the recorder/segment ffmpeg command use (same `stream_command()` builds
  both, per `stream.py`).
- **Override path**: setting the documented `input_args` config key on a
  camera *fully replaces* `CAMERA_INPUT_ARGS` (not appended) — see
  `stream.py::stream_command()`. So dropping `-use_wallclock_as_timestamps 1`
  from a custom `input_args` list is a supported (if advanced/untested-by-
  Viseron-itself) way to make ffmpeg fall back to RTCP SR-anchored PTS —
  i.e. camera-clock-based timestamps — without a full `raw_command` rewrite.
- Risk: a Viseron GitHub discussion (#1027) noted this flag is a "saver" —
  some cameras send broken/inconsistent RTP timestamps, and removing the
  wallclock fallback can break recording entirely on those cameras. Needs
  per-camera testing, not a blind global change.

**How to verify a given camera actually sends usable RTCP Sender Reports**
(prerequisite for camera-clock timestamps to work at all):
1. Wireshark/tcpdump the RTSP session, filter `rtcp`, confirm Sender Report
   (type 200) packets with a sensibly-advancing NTP timestamp field.
2. `GST_DEBUG=rtspsrc:5 gst-launch-1.0 rtspsrc location=rtsp://user:pass@<cam-ip>/<path> ! fakesink`
   — dumps the RTCP SR NTP↔RTP mapping to the debug log, no packet capture
   tool needed.
3. `ffmpeg -loglevel debug -i rtsp://user:pass@<cam-ip>/<path> -t 10 -f null - 2>&1 | grep -i rtcp`
   — check ffmpeg is actually processing sender-report NTP fields.
4. Empirical: apply the `input_args` override to one test camera, record,
   check for clean/even frame spacing (good) vs jitter or `frame_timeout`
   restarts (camera's RTP timestamps aren't trustworthy).
5. Confirm the camera's own clock is well-disciplined separately (ONVIF
   `GetSystemDateAndTime` vs host NTP time) — RTCP SR accuracy is only as
   good as the clock behind it. See LAN NTP sync plan above.

### Result for camera_1 (192.168.3.177, /stream1, "HTMS"-branded stream) — tested 2026-07-06

Ran `ffmpeg -loglevel debug -rtsp_transport tcp -i rtsp://...@192.168.3.177:554/stream1
-t 60 -c copy -f null -`, captured full 60s / 3600 frames to a log file, converted
from PowerShell's UTF-16 redirect encoding to UTF-8 for searching (plain grep against
the raw UTF-16 bytes silently found nothing — false negative, not evidence of anything).

**Confirmed: zero RTCP/Sender Report/NTP mentions across the entire session** (SDP
negotiation through 3600 demuxed frames). This is conclusive, not inconclusive —
ffmpeg's RTSP demuxer logs SR processing at debug level, and none appeared in a full
minute. **This camera does not send RTCP Sender Reports**, at least not on this
stream/transport.

Consequence: the earlier "clean recording" result from dropping
`-use_wallclock_as_timestamps` is **not evidence of camera-clock sync** — it's likely
worse for absolute timing than the default. Without RTCP SR, ffmpeg only anchors the
first frame to a reference point once at stream start, then counts up via RTP
timestamp deltas (nominally 90kHz per `rtpmap:96 H264/90000`) with no periodic
re-anchoring to true time — any clock-rate error accumulates uncorrected over the
whole recording. The default `-use_wallclock_as_timestamps 1` samples host wall-clock
per frame instead — jittery frame-to-frame, but no long-run drift.

**Decision: reverted the `input_args` override on camera_1, back to Viseron's
default (wallclock-arrival timestamps) + host/camera NTP sync as the sync strategy.**
Camera-clock/RTCP-based timestamping is not available for this camera. Still need to
run the same 60s RTCP check on any other cameras before assuming they behave the same
— cheap RTSP stacks vary even within the same vendor.

### Follow-up: ONVIF Media service may expose a different, RTCP-compliant stream URI

Camera is Anjvision-branded but the SDP `tool:HTMS` tag suggests an OEM firmware
stack underneath, and the tested path `/stream1` doesn't match the crowd-sourced
Anjvision default (`/h264_stream` per iSpyConnect) — likely the vendor's proprietary
RTSP endpoint, not necessarily the same code path as its ONVIF media profile.

Read the ONVIF Streaming Spec (v17.06) directly: Section 5.1.2.2.1 states "A device
**shall** support RTCP Sender Report for media synchronization" for live streaming —
this is a mandatory conformance requirement, but it applies to the stream URI the
camera's **ONVIF Media service** advertises via `GetStreamUri`, not necessarily
whatever proprietary path we tested.

**Next step (not yet done):** use ONVIF Device Manager (free Windows GUI tool) against
192.168.3.177 to get the actual ONVIF-advertised stream URI from `GetStreamUri`, then
rerun the same `ffmpeg -loglevel debug ... -t 60 -c copy -f null -` RTCP check against
*that* URI instead of `/stream1`. If the ONVIF-profile stream is served by different,
more spec-compliant code than the proprietary path, it may actually emit RTCP SR where
`/stream1` doesn't.

Camera also has a manufacturer resource portal at online.anjvision.com — turned out
to be a plain Apache/nginx directory index (not JS-rendered as first assumed), with
folders `client/ debug/ doc/ firmware/ instructions/ pdf/ sdk/`. Browsed it directly
with the browser tool once connected.

### Found: camera's proprietary HTTP API includes a real NTP config endpoint

`doc/API/Anjvision HTTP API说明书 1.5.pdf` (107 pages, bilingual CN/EN, "HTTP API
Specifications" v1.5) documents the vendor's CGI API — separate from ONVIF entirely.
Searched full extracted text for "rtcp"/"onvif": **zero matches anywhere in the
manual** — confirms this API has nothing to do with ONVIF streaming compliance, so
the RTCP-via-ONVIF-URI question (below) is still open and independent of this.

But it does document real NTP configuration, which ODM's UI doesn't surface (this is
the vendor's own CGI API, not ONVIF, so ONVIF tools wouldn't show it):

- **`/HAPI/V1.0/systime/setntp`** (GET or PUT) — params: `serverIP`, `serverPort`
  (optional, default 123), `refreshInterval` (optional, default 60s), plus
  `username`/`password` (or `uid`) for auth.
  Example: `http://192.168.1.202/HAPI/V1.0/systime/setntp?serverIP=ipvs.icamra.com&username=admin&password=e10adc3949ba59abbe56e057f20f883e`
  **Note:** the password param is MD5-hashed, not plaintext (that example hash is
  MD5("123456"), a well-known default-password hash) — must hash before sending.
- **`/HAPI/V1.0/systime/gettime`** (2.7.1) — returns current `timeMode`
  (`MANUAL`/`P2P`/`NTP`) and `nowtime`. Useful for confirming NTP mode actually took
  and for spot-checking drift against host time without going through ONVIF.
- **`/HAPI/V1.0/systime/settime`** (2.7.2) — manual time/timezone/mode set, in case
  `setntp` doesn't flip `timeMode` to `NTP` by itself.

**Next step:** use `setntp` to point camera_1 at the LAN NTP server from the sync
plan above, then poll `gettime` to confirm `timeMode: NTP` and check drift.

### RESOLVED: ONVIF/RTCP question closed, real NTP config found instead

ODM's Live Video tab reports the ONVIF media profile URI as the **same** `/stream1`
already tested — no separate ONVIF-compliant stream exists on this camera, so
there's nothing further to test. The `doc/API` HTTP API manual turned out to be for
an unrelated Anjvision firmware/product line — confirmed empirically, a test call to
`/HAPI/V1.0/uid/getuid` against camera_1 returned 404 (API not implemented on this
device). Also found a real customer review on the AliExpress listing for this exact
camera (WGWK-branded OEM board, resold under multiple names incl. Anjvision):
*"Camera with onvif enabled not identifiable by odm. rtsp links are not accessible by
vlc."* — independent confirmation this hardware's ONVIF/RTSP stack is known-flaky,
not a config mistake on our end.

**Conclusion: camera-clock/RTCP-based timestamping is not available on this camera,
full stop.** No further investigation needed here. Sticking with Viseron default
(wallclock-arrival stamping) + NTP as the sync strategy, per the plan above.

Side note: camera's own web UI (Platform Manager → GB28181) shows this firmware is
primarily built around the Chinese GB28181 video-interconnection standard (SIP-based
registration), with ONVIF/RTSP as a secondary compatibility layer — plausible
explanation for why the RTSP/ONVIF side is under-baked.

### Found: actual working NTP config, in the camera's own web UI (not ODM, not the HAPI doc)

`System Manager → Time Setting` in the camera's native web UI
(`http://192.168.3.177/view/time_setting.html`, rendered in an iframe — not visible
via simple page-text extraction, had to inspect the iframe's DOM directly). This is
NOT exposed through ODM/ONVIF, which is why ODM's UI didn't show it.

Current state found on camera_1: `Update Mode: NTP Server` (already enabled),
`NTP Server: time.windows.com`, `NTP Port: 123`, `Refresh Time: 12 Hours`.

**Changes made/needed:**
1. Change `NTP Server` from `time.windows.com` to the local LAN NTP source's IP.
2. Change `Refresh Time` unit — dropdown has both `Hours` and `Minutes` (confirmed via
   DOM inspection: `interval_unit` select options are `["Hours","Minutes"]`) — switch
   to Minutes, set ~5-10, matching the earlier "shorten poll interval" guidance.
3. Save.

This fully replaces the need for ONVIF `SetNTP` push or the HAPI `setntp` endpoint —
the camera already has a working native NTP client, just pointed at the wrong server
with too long a refresh interval.

## Open items / things to watch

- Haven't set explicit retention rules (`max_age`/`max_size`) — defaults to
  keep continuous recordings indefinitely (disk usage will grow).
- `create_event_clip` not enabled — no `.mp4` files in `/event_clips`, only
  raw `.m4s` segments in `/segments` (fine for in-browser HLS playback, but
  matters if he wants downloadable clips outside the web UI beyond the
  built-in download feature).
