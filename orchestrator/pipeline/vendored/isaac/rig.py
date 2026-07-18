"""Vendored, verbatim copy of ``omniverse_pipeline/rig.py`` (T11 copy-in, per
``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule; see
``pipeline.vendored.isaac``'s package docstring). Body is byte-for-byte the reference script's.

Vendored alongside ``omni_capture.py`` (same directory) because that's the one import it makes
(``import rig as rigmod`` after ``sys.path.insert(0, dirname(__file__))``) — a same-directory
import Python already resolves via the script's own directory on ``sys.path[0]``, so nothing
about that import needed to change when this file moved from ``omniverse_pipeline/`` here.

Pure numpy, no Isaac Sim/USD dependency — unlike its sibling modules in this package, it's safe to
import directly in the orchestrator's own host process too (``tests/test_stages_isaac.py`` does,
for a cheap sandbox-verifiable sanity check of the camera-rig math itself, even though the stage
that actually drives it, ``capture.isaac``, only ever execs it inside the container like every
other ``isaac``-environment stage).

Original docstring:

    rig.py — pure-math camera-rig generation for the Omniverse capture.

    No Isaac Sim / USD dependency (numpy only) so it can be unit-tested standalone.
    Produces OpenCV camera-to-world matrices (axes: +X right, +Y down, +Z look/forward)
    and intrinsics, which is exactly what omni_to_4dgs.py consumes.

    Run `python rig.py --selftest` for a quick sanity check.
"""
from __future__ import annotations
import argparse, numpy as np


def look_at_opencv(eye, target, world_up=(0, 0, 1)) -> np.ndarray:
    """Return an OpenCV camera-to-world 4x4 for a camera at `eye` looking at `target`.

    OpenCV camera frame: +X right, +Y down, +Z forward (toward target).
    """
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    world_up = np.asarray(world_up, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-12)
    # Degenerate guard: if fwd ~ world_up, pick a different up.
    if abs(np.dot(fwd, world_up / np.linalg.norm(world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up); right /= (np.linalg.norm(right) + 1e-12)
    down = np.cross(fwd, right)                     # +Y down in OpenCV
    R = np.stack([right, down, fwd], axis=1)        # columns = camera axes in world
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = eye
    return c2w


def ring(center, radius, n, height, world_up=(0, 0, 1), start_deg=0.0):
    """n cameras evenly spaced on a horizontal ring at `height` above center, looking in."""
    center = np.asarray(center, float)
    up = np.asarray(world_up, float)
    # Build an in-plane basis orthogonal to world_up.
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2)
    poses = []
    for i in range(n):
        ang = np.deg2rad(start_deg) + 2 * np.pi * i / n
        eye = center + radius * (np.cos(ang) * e1 + np.sin(ang) * e2) + height * up
        poses.append(look_at_opencv(eye, center, up))
    return poses


def dome(center, radius, n, world_up=(0, 0, 1), n_rings=3, min_elev_deg=10, max_elev_deg=70):
    """n cameras spread over a dome (upper hemisphere) looking at center."""
    center = np.asarray(center, float)
    up = np.asarray(world_up, float)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2)
    elevs = np.linspace(np.deg2rad(min_elev_deg), np.deg2rad(max_elev_deg), n_rings)
    per = [n // n_rings + (1 if r < n % n_rings else 0) for r in range(n_rings)]
    poses = []
    for r, (el, k) in enumerate(zip(elevs, per)):
        for i in range(k):
            az = 2 * np.pi * i / max(k, 1) + (r * np.pi / n_rings)
            dir_h = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2)
            eye = center + radius * (dir_h + np.sin(el) * up)
            poses.append(look_at_opencv(eye, center, up))
    return poses


def build_rig(cfg: dict, bbox_center, bbox_radius):
    """cfg keys: layout(ring|dome), n_cameras, radius_scale, height_scale, world_up,
    (ring: start_deg) (dome: n_rings, min_elev_deg, max_elev_deg).
    Returns list of OpenCV c2w (4x4 np arrays)."""
    center = np.asarray(bbox_center, float)
    up = np.asarray(cfg.get("world_up", [0, 0, 1]), float)
    radius = bbox_radius * float(cfg.get("radius_scale", 2.5))
    n = int(cfg.get("n_cameras", 8))
    layout = cfg.get("layout", "ring")
    if layout == "dome":
        return dome(center, radius, n, up, int(cfg.get("n_rings", 3)),
                    cfg.get("min_elev_deg", 10), cfg.get("max_elev_deg", 70))
    height = bbox_radius * float(cfg.get("height_scale", 0.3))
    return ring(center, radius, n, height, up, float(cfg.get("start_deg", 0.0)))


def intrinsics_from_fov(width, height, vfov_deg):
    """Pinhole intrinsics from a vertical field of view."""
    f = (height / 2) / np.tan(np.deg2rad(vfov_deg) / 2)
    return {"width": int(width), "height": int(height),
            "fx": float(f), "fy": float(f), "cx": width / 2, "cy": height / 2}


def _selftest():
    ok = True
    poses = ring(center=[0, 0, 0], radius=3, n=8, height=1)
    for c2w in poses:
        R = c2w[:3, :3]
        # orthonormal, right-handed
        ok &= np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        ok &= abs(np.linalg.det(R) - 1) < 1e-6
        # +Z (forward) should point toward origin from the camera center
        fwd = R[:, 2]; to_center = -c2w[:3, 3] / np.linalg.norm(c2w[:3, 3])
        ok &= np.dot(fwd, to_center) > 0.9
    d = dome(center=[0, 0, 0], radius=5, n=10)
    ok &= len(d) == 10
    intr = intrinsics_from_fov(1600, 900, 50)
    ok &= intr["fx"] > 0
    print("rig selftest:", "PASS" if ok else "FAIL",
          f"(ring={len(poses)}, dome={len(d)}, fx={intr['fx']:.1f})")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import sys
    sys.exit(_selftest() if a.selftest else 0)
