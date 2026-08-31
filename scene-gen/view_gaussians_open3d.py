"""Launch an interactive Open3D window to view 4DGS point clouds or colored segmentations.

Usage:
    uv run python view_gaussians_open3d.py --ply runs/grid-A20mm_M2/segmentation_colored_kabsch.ply
    uv run python view_gaussians_open3d.py --ply runs/grid-A20mm_M2/segmentation_colored.ply
    uv run python view_gaussians_open3d.py --raw runs/grid-A20mm_M2/train_out/point_cloud/iteration_15000/point_cloud.ply
"""

import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Gaussian / Segmentation Point Cloud Viewer")
    parser.add_argument("--ply", type=str, default="runs/grid-A20mm_M2/segmentation_colored_kabsch.ply", help="Path to colored PLY file")
    parser.add_argument("--raw", type=str, default=None, help="Path to raw 3DGS point_cloud.ply")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed in the current environment. Run: uv pip install open3d")
        sys.exit(1)

    if args.raw and os.path.exists(args.raw):
        print(f"Loading raw Gaussian point cloud from: {args.raw}")
        pcd = o3d.io.read_point_cloud(args.raw)
    elif os.path.exists(args.ply):
        print(f"Loading segmented/colored point cloud from: {args.ply}")
        pcd = o3d.io.read_point_cloud(args.ply)
    else:
        print(f"File not found: {args.ply or args.raw}")
        sys.exit(1)

    print(f"Loaded point cloud with {len(pcd.points)} points.")
    print("Controls:")
    print(" - Left Click + Drag: Orbit / Rotate")
    print(" - Right Click / Shift + Left Click: Pan")
    print(" - Scroll Wheel: Zoom")
    print(" - +/- keys: Increase/Decrease point size")
    print(" - N key: Toggle normals")
    print(" - Q / ESC: Exit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="4DGS Point Cloud Viewer", width=1280, height=800)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.asarray([0.08, 0.08, 0.10])
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
