"""Sweep-variant materialization (T27): derived capture dirs with subsampled init clouds.

Moved from the deleted ``scene-gen/run_grid_4dgs.py`` (Phase C of T27). A
sweep variant ``<capture_dir>_g<N>`` holds a subsampled ``points3D_gt.ply`` /
``points3D_labels.npy`` (seeded RNG), a copied ``cameras_gt.json``, and **Windows directory
junctions** (``mklink /J``) to the original per-camera ``camNN`` dirs — no frame copying.
Junctions require the captures to be on the same volume and are Windows-only by design
(inherited constraint from the grid pipeline; the function raises on other platforms).

Also home to ``count_gaussians`` (numeric-iteration PLY header parse), used by the results
store.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..vendored.host.convert import _read_ply_xyz_rgb, write_ply


def count_ply_vertices(ply_path: Path) -> int | None:
    """``element vertex N`` from a PLY header, without reading the payload."""
    with open(ply_path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return None


def count_gaussians(run_dir: Path) -> int | None:
    """Final trained Gaussian count of a run, from its last point_cloud.ply header.

    O6 (reviews/2026-10-05-omniverse-pipeline-review.md): sort by the NUMERIC iteration
    suffix — a plain lexicographic sorted() ranks "iteration_7000" after "iteration_30000".
    """

    def _iteration(p: Path) -> int:
        return int(p.parent.name.rsplit("_", 1)[1])

    plys = sorted(
        run_dir.glob("train_out/point_cloud/iteration_*/point_cloud.ply"), key=_iteration
    )
    if not plys:
        return None
    return count_ply_vertices(plys[-1])


def make_capture_variant(src: Path, dst: Path, n_points: int, seed: int = 0) -> Path:
    """Derived capture dir with a subsampled init cloud; camNN dirs are junctions.

    Idempotent: an existing ``dst`` whose ``points3D_gt.ply`` already has exactly ``n_points``
    vertices is reused as-is; an existing ``dst`` with a different count is an error (never
    silently clobbered). Returns ``src`` unchanged when ``n_points`` covers the whole source
    cloud (the 100%-target case).
    """
    src_ply = src / "points3D_gt.ply"
    available = count_ply_vertices(src_ply)
    if available is not None and n_points >= available:
        return src

    if dst.exists():
        existing = count_ply_vertices(dst / "points3D_gt.ply") if (dst / "points3D_gt.ply").is_file() else None
        if existing == n_points:
            return dst
        raise RuntimeError(
            f"capture variant {dst} already exists with {existing} points, expected "
            f"{n_points} — delete it explicitly to rebuild"
        )
    dst.mkdir(parents=True)
    xyz, rgb = _read_ply_xyz_rgb(str(src_ply))
    labels = np.load(src / "points3D_labels.npy")
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(xyz), size=n_points, replace=False))
    write_ply(str(dst / "points3D_gt.ply"), xyz[idx], rgb[idx].astype(np.float32) / 255.0)
    np.save(dst / "points3D_labels.npy", labels[idx])
    shutil.copy2(src / "cameras_gt.json", dst / "cameras_gt.json")
    if sys.platform != "win32":
        raise NotImplementedError(
            "capture-variant junctions are Windows-only (mklink /J) — see module docstring"
        )
    for cam_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("cam")):
        link = dst / cam_dir.name
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link).replace("/", "\\"), str(cam_dir).replace("/", "\\")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"mklink {link} -> {cam_dir} failed: {r.stdout}{r.stderr}")
    return dst
