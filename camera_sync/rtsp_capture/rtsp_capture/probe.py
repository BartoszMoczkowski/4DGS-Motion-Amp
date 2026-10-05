"""rtsp_capture.probe

Pre-flight diagnostic: connect to every camera in the config for a few
seconds and verify authentication, codec, clock rate, frame flow and --
critically -- that RTCP Sender Reports actually arrive (need >= 2 for
cross-camera sync to work).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from .recorder import load_camera_config
from .rtsp_client import RtspClient, redact_url

MIN_SRS = 2


def probe_camera(camera_id: str, url: str, duration: float = 10.0) -> dict:
    result = {
        "camera_id": camera_id,
        "url": redact_url(url),
        "ok": False,
        "error": None,
        "codec": None,
        "clock_rate": None,
        "resolution": None,
        "frames": 0,
        "srs": 0,
    }
    client = RtspClient(url)
    try:
        client.connect()
        info = client.describe()
        client.setup()
        client.play()
    except Exception as exc:
        result["error"] = str(exc)
        client.close()
        return result

    result["codec"] = client.codec
    result["clock_rate"] = client.clock_rate
    if "width" in info:
        result["resolution"] = f"{info['width']}x{info['height']}"

    stop = threading.Event()
    loop_error: list[BaseException] = []

    def _loop() -> None:
        try:
            client.receive_loop(
                lambda *_: result.__setitem__("frames", result["frames"] + 1),
                lambda *_: result.__setitem__("srs", result["srs"] + 1),
                stop,
            )
        except BaseException as exc:  # noqa: BLE001 - report, don't crash probe
            loop_error.append(exc)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    time.sleep(duration)
    stop.set()
    thread.join(timeout=5.0)
    client.teardown()
    client.close()

    if loop_error and result["frames"] == 0:
        result["error"] = str(loop_error[0])
    result["ok"] = result["error"] is None and result["srs"] >= MIN_SRS
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtsp-probe",
        description="Diagnostic RTSP connect: auth, codec, clock rate, RTCP SR check",
    )
    parser.add_argument(
        "--config", "-c", type=Path, required=True,
        help="YAML file with camera definitions (see cameras.yaml.example)",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=10.0, metavar="SEC",
        help="How long to listen per camera (default: 10)",
    )
    args = parser.parse_args(argv)

    try:
        cameras = load_camera_config(args.config)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[PROBE] Probing {len(cameras)} camera(s) for {args.duration:.0f}s…")
    results: list[dict] = []
    threads = []
    for camera_id, cfg in cameras.items():
        thread = threading.Thread(
            target=lambda cid=camera_id, u=cfg.url: results.append(
                probe_camera(cid, u, args.duration)
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    failures = 0
    for result in sorted(results, key=lambda r: r["camera_id"]):
        cid = result["camera_id"]
        if result["error"]:
            failures += 1
            print(f"  [{cid}] FAIL  {result['url']}  error: {result['error']}")
            continue
        status = "OK  " if result["ok"] else "FAIL"
        if not result["ok"]:
            failures += 1
        res = result["resolution"] or "?"
        print(
            f"  [{cid}] {status}  {result['url']}  codec={result['codec']} "
            f"{res}  clock={result['clock_rate']} Hz  "
            f"frames={result['frames']}  SRs={result['srs']}"
            + ("" if result["ok"] else f"  (need >= {MIN_SRS} SRs)")
        )

    total = len(results)
    if failures:
        print(f"[PROBE] {total - failures}/{total} camera(s) OK.")
        return 1
    print(f"[PROBE] All {total} camera(s) OK: auth, stream and >= {MIN_SRS} RTCP SRs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
