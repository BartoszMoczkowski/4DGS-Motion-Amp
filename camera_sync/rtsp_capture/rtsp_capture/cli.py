"""rtsp_capture.cli

Unified CLI suite for multi-camera capture, RTCP synchronization,
probe diagnostics, camera tuning, and Web Studio GUI launcher.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .manager import CaptureManager


def _find_default_config() -> Path:
    candidates = [
        Path("cameras.yaml"),
        Path(__file__).parent.parent / "cameras.yaml",
        Path("camera_sync/rtsp_capture/cameras.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


# ----------------------------------------------------------------------
# Subcommand: gui
# ----------------------------------------------------------------------

def cmd_gui(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is required for the Web GUI. Install via 'uv sync'.", file=sys.stderr)
        return 1

    from .web_app import create_app

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    manager = CaptureManager(config_path, mock_mode=args.mock)
    app = create_app(manager)

    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"  4DGS Multi-Camera Capture Studio")
    print(f"  Web GUI:   {url}")
    print(f"  Config:    {config_path}")
    print(f"  Cameras:   {len(manager.cameras)} configured")
    print(f"  Mode:      {'MOCK SIMULATION (offline)' if args.mock else 'LIVE HARDWARE'}")
    print(f"=======================================================\n")

    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# ----------------------------------------------------------------------
# Subcommand: record
# ----------------------------------------------------------------------

def cmd_record(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    manager = CaptureManager(config_path, mock_mode=args.mock)
    out_dir = args.out.resolve()

    print(f"[RECORDER] Target: {out_dir} | Duration: {args.duration or 'until Enter/Ctrl+C'}")
    try:
        res = manager.start_recording(
            output_dir=out_dir.parent,
            session_name=out_dir.name,
            duration_sec=args.duration,
            auto_extract=args.extract,
        )
        print(f"[RECORDER] Session active: {res['run_id']}")
        if manager.recording_thread:
            manager.recording_thread.join()
        print(f"[RECORDER] Completed.")
        return 0
    except KeyboardInterrupt:
        print("\n[RECORDER] Stopping on user request...")
        manager.stop_recording()
        return 0


# ----------------------------------------------------------------------
# Subcommand: probe
# ----------------------------------------------------------------------

def cmd_probe(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    manager = CaptureManager(config_path, mock_mode=args.mock)
    print(f"[PROBE] Probing {len(manager.cameras)} camera(s) for {args.duration:.0f}s...")
    results = manager.probe_all(duration=args.duration)

    print("\n" + "=" * 78)
    print(f"{'Camera':<10} {'Status':<10} {'Codec':<8} {'Resolution':<12} {'Frames':<8} {'SRs':<6} {'Confidence'}")
    print("-" * 78)
    for r in results:
        status = "ONLINE" if r["ok"] else "FAIL"
        srs = r.get("srs", 0)
        conf = "SYNC READY" if srs >= 2 else "LOW"
        print(
            f"{r['camera_id']:<10} {status:<10} {r.get('codec') or '-':<8} "
            f"{r.get('resolution') or '-':<12} {r.get('frames', 0):<8} {srs:<6} {conf}"
        )
    print("=" * 78 + "\n")
    return 0


# ----------------------------------------------------------------------
# Subcommand: config
# ----------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    manager = CaptureManager(config_path, mock_mode=args.mock)
    if not manager.dahua:
        print("[ERROR] Dahua controller not initialized", file=sys.stderr)
        return 1

    targets = list(manager.cameras.keys()) if args.all else ([args.camera] if args.camera else [])
    if not targets:
        print("[ERROR] Specify --camera <id> or --all", file=sys.stderr)
        return 1

    print(f"[CONFIG] Applying settings to: {', '.join(targets)}")
    if args.wb_style or args.wb_red is not None or args.wb_blue is not None:
        style = args.wb_style or "Manual"
        red = args.wb_red if args.wb_red is not None else 50
        blue = args.wb_blue if args.wb_blue is not None else 50
        res = manager.dahua.batch_set_white_balance(targets, style=style, gain_red=red, gain_blue=blue)
        for cid, (ok, msg) in res.items():
            print(f"  [{cid}] White Balance: {'OK' if ok else 'FAIL'} ({msg.strip()})")

    if args.shutter or args.exp_mode:
        shutter = args.shutter or "1/100"
        mode = args.exp_mode or "Manual"
        res = manager.dahua.batch_set_exposure(targets, mode=mode, shutter=shutter)
        for cid, (ok, msg) in res.items():
            print(f"  [{cid}] Exposure: {'OK' if ok else 'FAIL'} ({msg.strip()})")

    return 0


# ----------------------------------------------------------------------
# Subcommand: flash
# ----------------------------------------------------------------------

def cmd_flash(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    manager = CaptureManager(config_path, mock_mode=args.mock)
    if not manager.dahua:
        print("[ERROR] Dahua controller not initialized", file=sys.stderr)
        return 1

    print(f"[FLASH] Flashing light on camera '{args.camera}' for {args.duration:.1f}s ({args.pulses} pulses)...")
    ok = manager.dahua.flash_camera_light(args.camera, duration_sec=args.duration, pulses=args.pulses)
    if ok:
        print(f"[FLASH] Light pulsed on {args.camera}.")
        return 0
    print(f"[ERROR] Failed to flash {args.camera}.", file=sys.stderr)
    return 1


# ----------------------------------------------------------------------
# Subcommand: extract
# ----------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> int:
    from .sync_extract import extract_synced_frames

    print(f"[EXTRACT] Extracting synced frames from: {args.input}")
    try:
        manifest = extract_synced_frames(
            recording_dir=args.input.resolve(),
            out_dir=args.out.resolve(),
            fps=args.fps,
            tolerance=args.tolerance,
        )
        print(f"[EXTRACT] Complete. Kept {manifest['instants_kept']}/{manifest['instants_total']} instants.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


# ----------------------------------------------------------------------
# Main Argument Parser
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    default_cfg = _find_default_config()

    parser = argparse.ArgumentParser(
        prog="rtsp-capture",
        description="Multi-Camera Capture Studio with RTCP cross-camera synchronization and Dahua hardware controls",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # gui
    p_gui = subparsers.add_parser("gui", help="Launch the Web Studio GUI")
    p_gui.add_argument("--config", "-c", type=Path, default=default_cfg, help="Path to cameras.yaml")
    p_gui.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    p_gui.add_argument("--port", "-p", type=int, default=8080, help="Web server port (default: 8080)")
    p_gui.add_argument("--mock", action="store_true", help="Run with simulated mock cameras")
    p_gui.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    p_gui.set_defaults(func=cmd_gui)

    # record
    p_rec = subparsers.add_parser("record", help="Record synchronized bitstreams and RTCP logs")
    p_rec.add_argument("--config", "-c", type=Path, default=default_cfg, help="Path to cameras.yaml")
    p_rec.add_argument("--out", "-o", type=Path, required=True, help="Output directory")
    p_rec.add_argument("--duration", "-d", type=float, default=None, help="Duration in seconds")
    p_rec.add_argument("--extract", action="store_true", help="Auto-extract synced frames after recording")
    p_rec.add_argument("--mock", action="store_true", help="Run in mock mode")
    p_rec.set_defaults(func=cmd_record)

    # probe
    p_probe = subparsers.add_parser("probe", help="Pre-flight diagnostic for RTSP & RTCP Sender Reports")
    p_probe.add_argument("--config", "-c", type=Path, default=default_cfg, help="Path to cameras.yaml")
    p_probe.add_argument("--duration", "-d", type=float, default=5.0, help="Probe listen duration (default: 5s)")
    p_probe.add_argument("--mock", action="store_true", help="Run in mock mode")
    p_probe.set_defaults(func=cmd_probe)

    # config
    p_cfg = subparsers.add_parser("config", help="View or adjust camera parameters (WB, Exposure)")
    p_cfg.add_argument("--config", "-c", type=Path, default=default_cfg, help="Path to cameras.yaml")
    p_cfg.add_argument("--camera", help="Target camera ID")
    p_cfg.add_argument("--all", action="store_true", help="Apply to all cameras")
    p_cfg.add_argument("--wb-style", choices=["Manual", "Auto", "Custom"])
    p_cfg.add_argument("--wb-red", type=int, help="White balance Red gain (0-100)")
    p_cfg.add_argument("--wb-blue", type=int, help="White balance Blue gain (0-100)")
    p_cfg.add_argument("--exp-mode", choices=["Manual", "Auto"])
    p_cfg.add_argument("--shutter", help="Shutter speed, e.g. 1/50, 1/100, 1/250")
    p_cfg.add_argument("--mock", action="store_true", help="Run in mock mode")
    p_cfg.set_defaults(func=cmd_config)

    # flash
    p_flash = subparsers.add_parser("flash", help="Flash camera light to identify physical unit in rig")
    p_flash.add_argument("--config", "-c", type=Path, default=default_cfg, help="Path to cameras.yaml")
    p_flash.add_argument("--camera", required=True, help="Target camera ID, e.g. cam02")
    p_flash.add_argument("--duration", type=float, default=3.0, help="Flash duration in seconds")
    p_flash.add_argument("--pulses", type=int, default=4, help="Number of light pulses")
    p_flash.add_argument("--mock", action="store_true", help="Run in mock mode")
    p_flash.set_defaults(func=cmd_flash)

    # extract
    p_ext = subparsers.add_parser("extract", help="Extract synced JPEG frames from recording")
    p_ext.add_argument("--input", "-i", type=Path, required=True, help="Recording folder")
    p_ext.add_argument("--out", "-o", type=Path, required=True, help="Output folder for synced frames")
    p_ext.add_argument("--fps", type=float, default=None, help="Target uniform FPS")
    p_ext.add_argument("--tolerance", type=float, default=None, help="Residual tolerance in seconds")
    p_ext.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def gui_main() -> int:
    parser = build_parser()
    args = parser.parse_args(["gui"] + sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
