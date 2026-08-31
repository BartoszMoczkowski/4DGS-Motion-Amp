#!/usr/bin/env python3
"""gen_cubes_dataset_all.py — master batch script to generate the synthetic benchmark dataset.

Specs:
  - Cubes N in {2, 4, 8}
  - Motion modes: rotation (pure spin), translation (floor linear oscillation), both (spin + translation)
  - Camera Rig: 16 cameras arranged in two stacked 80-degree elevation arcs (0.5m vertical separation: h_lower=0.8m, h_upper=1.3m)
  - Checkerboard: 3x larger tiles (tile_size = 1.5m)
  - Framerate & Duration: 60 FPS, 120 frames (2.0s duration)

Usage:
  python scene-gen/gen_cubes_dataset_all.py                  # generate all 9 scene variants
  python scene-gen/gen_cubes_dataset_all.py --test            # generate single test scene (k=2, mode=both)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO_ROOT / "omniverse-pipeline" / "data" / "scenes" / "cubes_dataset_60fps"
sys.path.insert(0, str(REPO_ROOT / "scene-gen"))

from gen_spinning_cubes import generate_spinning_cubes_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Cube counts N (default: 2 4 8)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["rotation", "translation", "both"],
        default=["rotation", "translation", "both"],
        help="Motion modes to generate (default: rotation translation both)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only generate single test scene (N=2, mode=both)",
    )
    parser.add_argument(
        "--camera-layout",
        type=str,
        choices=["stacked_ring", "stacked_arcs", "circle", "ring"],
        default="stacked_ring",
        help="Camera rig layout: stacked_ring (default) or stacked_arcs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SCENES_DIR),
        help=f"Output directory for scenes (default: {SCENES_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        n_list = [2]
        modes_list = ["both"]
        print("=== Generating Single Test Scene (N=2, motion_mode=both, 16 cams, 360 deg stacked ring, 0.5m sep, 60fps, 120 frames) ===")
    else:
        n_list = args.n
        modes_list = args.modes
        print(f"=== Generating Dataset Variants: N={n_list}, modes={modes_list}, layout={args.camera_layout} ===")

    manifests = []
    for k_val in n_list:
        for mode in modes_list:
            scene_name = f"cubes_k{k_val}_{mode}"
            print(f"\n[dataset] Generating USD scene: {scene_name} (N={k_val}, mode={mode}, layout={args.camera_layout})...")
            m = generate_spinning_cubes_scene(
                k=k_val,
                circle_radius=1.5,
                cube_size=0.35,
                num_cameras=16,
                camera_radius=3.0,
                camera_height=1.2,
                camera_layout=args.camera_layout,
                arc_deg=80.0,
                center_deg=0.0,
                height_lower=0.8,
                height_upper=1.3,  # 0.5m vertical separation
                motion_mode=mode,
                wall_margin=2.5,
                room_height=4.5,
                tile_size=1.5,      # 3x larger checkerboard tiles
                num_frames=120,    # 120 frames
                fps=60.0,          # 60 FPS
                seed=42 + k_val * 10,
                out_dir=out_dir,
                name=scene_name,
                capture_root="Q:/Omniverse/renders",
            )
            manifests.append(m)

    batch_manifest_path = out_dir / "dataset_manifest.json"
    with open(batch_manifest_path, "w") as f:
        json.dump({"scenes": manifests, "total_variants": len(manifests)}, f, indent=2)

    print(f"\n================================================================================")
    print(f"Successfully generated {len(manifests)} scene variants in {out_dir}")
    print(f"Manifest written -> {batch_manifest_path}")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
