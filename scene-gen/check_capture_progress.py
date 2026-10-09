#!/usr/bin/env python3
"""
check_capture_progress.py — CLI monitoring tool for Isaac Sim multi-view synthetic captures.

Back-compat wrapper (T26): the inspection logic moved to
``omniverse_pipeline.capture_check`` (capture QA belongs to the capture pipeline); this
script keeps its original CLI working by importing from there. For single-capture
pass/fail QA use ``amp capture verify <dir>`` instead.

Usage:
    python scene-gen/check_capture_progress.py
    python scene-gen/check_capture_progress.py --watch 2
    python scene-gen/check_capture_progress.py --capture-dir Q:/Omniverse/renders/capture_cubes_k2_both
    python scene-gen/check_capture_progress.py --renders-dir Q:/Omniverse/renders
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from omniverse_pipeline.capture_check import (
    inspect_capture_dir,
    print_status_table,
)

DEFAULT_RENDERS_DIR = Path(r"Q:\Omniverse\renders")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renders-dir",
        type=str,
        default=str(DEFAULT_RENDERS_DIR),
        help=f"Directory containing capture folders (default: {DEFAULT_RENDERS_DIR})",
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        default=None,
        help="Path to a single capture directory to monitor",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="Refresh interval in seconds (e.g. --watch 2)",
    )
    args = parser.parse_args()

    try:
        while True:
            captures = []
            if args.capture_dir:
                cap_path = Path(args.capture_dir)
                if cap_path.is_dir():
                    captures.append(inspect_capture_dir(cap_path))
                else:
                    print(f"[error] Specified directory does not exist: {cap_path}")
                    sys.exit(1)
            else:
                renders_dir = Path(args.renders_dir)
                if renders_dir.is_dir():
                    dirs = sorted([d for d in renders_dir.iterdir() if d.is_dir() and d.name.startswith("capture_")])
                    for d in dirs:
                        captures.append(inspect_capture_dir(d))
                else:
                    print(f"[warning] Renders directory not found: {renders_dir}")

            if args.watch:
                # Clear terminal screen
                os.system("cls" if os.name == "nt" else "clear")
                print(f"[check_capture_progress] Watching captures in {args.capture_dir or args.renders_dir} (interval: {args.watch}s) — Press Ctrl+C to exit")

            print_status_table(captures)

            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
