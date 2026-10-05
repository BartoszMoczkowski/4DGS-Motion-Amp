#!/usr/bin/env python3
"""export_cube_gt_pointcloud.py — standalone GT point cloud generator for USD cube scenes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.float32)
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float red\nproperty float green\nproperty float blue\n"
        "end_header\n"
    )
    data = np.concatenate([xyz, np.zeros_like(xyz), rgb], axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def sample_mesh_quad_surface(corners: np.ndarray, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points uniformly on a 3D planar quadrilateral given 4 corners."""
    # corners: (4, 3)
    u = rng.uniform(0, 1, size=(n_samples, 1))
    v = rng.uniform(0, 1, size=(n_samples, 1))
    # Bilinear interpolation
    p00, p10, p11, p01 = corners[0], corners[1], corners[2], corners[3]
    pts = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01
    return pts


def export_gt_pointcloud(usd_path: str | Path, capture_dir: str | Path, n_target: int = 50000, scale: float = 1.0):
    usd_path = Path(usd_path).resolve()
    capture_dir = Path(capture_dir).resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    xf_cache = UsdGeom.XformCache(Usd.TimeCode(0))
    rng = np.random.RandomState(42)

    xyz_all, lab_all = [], []
    label_names = {"background": 0}

    # 1. Sample Background from /World/Room
    room_root = stage.GetPrimAtPath("/World/Room")
    if room_root.IsValid():
        room_pts = []
        for prim in Usd.PrimRange(room_root):
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                pts_attr = mesh.GetPointsAttr().Get()
                if pts_attr and len(pts_attr) == 4:
                    M = xf_cache.GetLocalToWorldTransform(prim)
                    corners = np.array([M.Transform(Gf.Vec3d(*p)) for p in pts_attr], dtype=np.float64)
                    # Sample 5,000 points per quad (floor, ceiling, 4 walls = 30k pts)
                    sampled = sample_mesh_quad_surface(corners, 5000, rng)
                    room_pts.append(sampled)
        if room_pts:
            bg_xyz = np.concatenate(room_pts, axis=0)
            xyz_all.append(bg_xyz)
            lab_all.append(np.zeros(len(bg_xyz), dtype=np.int32))
            print(f"[gt_export] Sampled {len(bg_xyz)} background room points (label 0)")

    # 2. Sample Moving Cubes from /World/Cubes
    cubes_root = stage.GetPrimAtPath("/World/Cubes")
    if cubes_root.IsValid():
        cube_id = 1
        for prim in Usd.PrimRange(cubes_root):
            if prim.GetTypeName() in ("Cube", "Mesh"):
                parent = prim.GetParent()
                name = parent.GetName() if (parent and parent.IsValid() and parent.GetName() != "Cubes") else prim.GetName()
                if name not in label_names:
                    label_names[name] = cube_id
                    cube_id += 1

                cid = label_names[name]
                if prim.GetTypeName() == "Cube":
                    cube = UsdGeom.Cube(prim)
                    size = float(cube.GetSizeAttr().Get() or 0.35)
                    hs = size / 2.0
                    grid = np.linspace(-hs, hs, 25)
                    gx, gy, gz = np.meshgrid(grid, grid, grid)
                    mask = (np.abs(gx) == hs) | (np.abs(gy) == hs) | (np.abs(gz) == hs)
                    P = np.column_stack([gx[mask], gy[mask], gz[mask]]).astype(np.float64)
                else:
                    mesh = UsdGeom.Mesh(prim)
                    pts = mesh.GetPointsAttr().Get()
                    P = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)

                M = xf_cache.GetLocalToWorldTransform(prim)
                Pw = np.array([M.Transform(Gf.Vec3d(*p)) for p in P], dtype=np.float64)

                xyz_all.append(Pw)
                lab_all.append(np.full(len(Pw), cid, dtype=np.int32))
                print(f"[gt_export] Extracted {name} (label {cid}): {len(Pw)} points at {Pw.mean(axis=0)}")

    if not xyz_all:
        print(f"[gt_export] No geometry found under {usd_path}")
        return

    xyz = np.concatenate(xyz_all) * scale
    lab = np.concatenate(lab_all)

    if len(xyz) > n_target:
        # Keep all cube points, downsample background if needed
        cube_mask = (lab > 0)
        n_cube = cube_mask.sum()
        n_bg_avail = n_target - n_cube
        if n_bg_avail > 0 and (~cube_mask).sum() > n_bg_avail:
            bg_indices = np.where(~cube_mask)[0]
            chosen_bg = rng.choice(bg_indices, n_bg_avail, replace=False)
            cube_indices = np.where(cube_mask)[0]
            keep_idx = np.sort(np.concatenate([cube_indices, chosen_bg]))
            xyz, lab = xyz[keep_idx], lab[keep_idx]

    rs = np.random.RandomState(42)
    palette = rs.randint(40, 230, size=(lab.max() + 1, 3), dtype=np.uint8)
    palette[0] = np.array([80, 80, 80], dtype=np.uint8)  # grey for background
    rgb = palette[lab]

    ply_path = capture_dir / "points3D_gt.ply"
    npy_path = capture_dir / "points3D_labels.npy"
    lbl_path = capture_dir / "label_names.json"
    npz_path = capture_dir / "gt_segmentation.npz"

    write_ply(ply_path, xyz, rgb)
    np.save(npy_path, lab)
    np.savez(npz_path, points=xyz.astype(np.float32), labels=lab.astype(np.int32))
    with open(lbl_path, "w") as f:
        json.dump({v: k for k, v in label_names.items()}, f, indent=2)

    print(f"[gt_export] Successfully exported GT point cloud ({len(xyz)} points, {len(label_names)} instances: {list(label_names.keys())})")
    print(f" -> PLY: {ply_path}")
    print(f" -> NPY: {npy_path}")
    print(f" -> NPZ: {npz_path}")
    print(f" -> Labels: {lbl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    export_gt_pointcloud(args.usd, args.out, n_target=args.points, scale=args.scale)


if __name__ == "__main__":
    main()
