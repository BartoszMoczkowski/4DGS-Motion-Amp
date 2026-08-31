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


def export_gt_pointcloud(usd_path: str | Path, capture_dir: str | Path, n_target: int = 50000):
    usd_path = Path(usd_path).resolve()
    capture_dir = Path(capture_dir).resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    cubes_root = stage.GetPrimAtPath("/World/Cubes")
    it = Usd.PrimRange(cubes_root) if cubes_root.IsValid() else stage.Traverse()

    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    xyz_all, lab_all = [], []
    label_names = {}
    lid = 0

    for prim in it:
        type_name = prim.GetTypeName()
        if type_name not in ("Cube", "Mesh"):
            continue

        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            P = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
        else: # Cube
            cube = UsdGeom.Cube(prim)
            size = float(cube.GetSizeAttr().Get() or 1.0)
            hs = size / 2.0
            grid = np.linspace(-hs, hs, 25)
            gx, gy, gz = np.meshgrid(grid, grid, grid)
            mask = (np.abs(gx) == hs) | (np.abs(gy) == hs) | (np.abs(gz) == hs)
            P = np.column_stack([gx[mask], gy[mask], gz[mask]]).astype(np.float64)

        M = xf_cache.GetLocalToWorldTransform(prim)
        Pw = np.array([M.Transform((x, y, z)) for x, y, z in P])

        parent = prim.GetParent()
        name = parent.GetName() if (parent and parent.IsValid() and parent.GetName()) else prim.GetName()
        if name not in label_names:
            label_names[name] = lid
            lid += 1

        xyz_all.append(Pw)
        lab_all.append(np.full(len(Pw), label_names[name], np.int32))

    if not xyz_all:
        print(f"[gt_export] No cubes/meshes found under {usd_path}")
        return

    xyz = np.concatenate(xyz_all)
    lab = np.concatenate(lab_all)

    if len(xyz) > n_target:
        idx = np.random.RandomState(0).choice(len(xyz), n_target, replace=False)
        xyz, lab = xyz[idx], lab[idx]

    rs = np.random.RandomState(42)
    palette = rs.randint(40, 230, size=(lab.max() + 1, 3), dtype=np.uint8)
    rgb = palette[lab]

    ply_path = capture_dir / "points3D_gt.ply"
    npy_path = capture_dir / "points3D_labels.npy"
    lbl_path = capture_dir / "label_names.json"

    write_ply(ply_path, xyz, rgb)
    np.save(npy_path, lab)
    with open(lbl_path, "w") as f:
        json.dump({v: k for k, v in label_names.items()}, f, indent=2)

    print(f"[gt_export] Successfully exported GT point cloud ({len(xyz)} points, {len(label_names)} instances)")
    print(f" -> PLY: {ply_path}")
    print(f" -> NPY: {npy_path}")
    print(f" -> Labels: {lbl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--points", type=int, default=50000)
    args = parser.parse_args()

    export_gt_pointcloud(args.usd, args.out, n_target=args.points)


if __name__ == "__main__":
    main()
