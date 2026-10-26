#!/usr/bin/env python3
"""
scripts/gen_synthetic_wobble_scene.py

Task T38: Synthetic sync pre-flight scene generator.
Constructs a deterministic, multi-view 4DGS dataset with:
1. 10 cameras (matching the 2-tier Dahua capture rig geometry: cam01..cam10).
2. A known rigid 7.11 Hz linear oscillation of a central test object (rotor assembly)
   with calibrated ~2.8 px projected image amplitude.
3. Injected 15–25 ms per-camera offsets (reproducing the measured RTSP capture offsets).
4. Full 4DGS multipleview dataset layout:
   - cam01/ .. cam10/ (frame_00001.jpg .. frame_00060.jpg)
   - sparse_/ (cameras.bin, images.bin, points3D.bin)
   - points3D_multipleview.ply
   - poses_bounds_multipleview.npy
   - offsets.json (authoritative schema matching T32)
   - optical_offsets.json (matching T33 format)
   - gt_motion_info.json (complete ground-truth parameters)
   - motion_masks/ (precomputed masks matching T36)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "multipleview" / "synthetic_turbine_preflight"

# Nominal shaft speed & measured timing offsets from T32 archaeology
NOMINAL_WOBBLE_HZ = 7.112
DEFAULT_FPS = 20.0
MEASURED_OFFSETS_S = {
    "cam01": 0.0,
    "cam02": 0.0180,
    "cam03": -0.0086,
    "cam04": 0.0230,
    "cam05": -0.0226,
    "cam06": 0.0200,
    "cam07": -0.0035,
    "cam08": -0.0220,
    "cam09": -0.0160,
    "cam10": 0.0120,
}


# -----------------------------------------------------------------------------
# Geometry and COLMAP Binary Serializers
# -----------------------------------------------------------------------------

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0:
        q = -q
    return q


def opencv_c2w_to_colmap_qt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera-to-world (4x4) -> COLMAP (qvec, tvec) world-to-camera."""
    c2w = np.asarray(c2w, dtype=np.float64)
    R_c2w = c2w[:3, :3]
    C = c2w[:3, 3]
    R_wc = R_c2w.T
    t_wc = -R_wc @ C
    return rotmat_to_qvec(R_wc), t_wc


def opencv_c2w_to_llff_row(c2w: np.ndarray, H: float, W: float, focal: float, near: float = 0.5, far: float = 5.0) -> np.ndarray:
    """Build one LLFF poses_bounds row (17,) from OpenCV c2w."""
    c2w = np.asarray(c2w, dtype=np.float64)
    R_c2w = c2w[:3, :3]
    C = c2w[:3, 3]
    right = R_c2w[:, 0]
    down = R_c2w[:, 1]
    fwd = R_c2w[:, 2]
    up = -down
    back = -fwd
    m = np.stack([-up, right, back, C, np.array([H, W, focal])], axis=1)
    return np.concatenate([m.reshape(-1), [near, far]]).astype(np.float64)


def write_cameras_bin(path: Path | str, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Shared PINHOLE camera (id=1, model=1: fx, fy, cx, cy)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, int(width), int(height)))
        f.write(struct.pack("<dddd", float(fx), float(fy), float(cx), float(cy)))


def write_images_bin(path: Path | str, cams: list[dict[str, Any]]):
    """Write images.bin with image1.jpg .. imageN.jpg matching 4DGS conventions."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for i, c in enumerate(cams, start=1):
            q = c["qvec"]
            t = c["tvec"]
            f.write(struct.pack("<i", i))
            f.write(struct.pack("<dddd", *[float(v) for v in q]))
            f.write(struct.pack("<ddd", *[float(v) for v in t]))
            f.write(struct.pack("<i", 1))
            f.write(c["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points3D_bin(path: Path | str, xyz: np.ndarray, rgb: np.ndarray):
    """Minimal points3D.bin."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *xyz[i]))
            f.write(struct.pack("<BBB", *rgb[i]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))


def write_ply(path: Path | str, xyz: np.ndarray, rgb: np.ndarray):
    """Write points3D_multipleview.ply with format matching scene/dataset_readers.py."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float red\nproperty float green\nproperty float blue\n"
        "end_header\n"
    )
    normals = np.zeros_like(xyz)
    data = np.concatenate([xyz, normals, rgb], axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# -----------------------------------------------------------------------------
# 10-Camera Rig Geometry
# -----------------------------------------------------------------------------

def build_10cam_rig(
    radius: float = 1.8,
    target_center: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Builds a 10-camera 2-tier rig looking at target_center:
    - Tier 1 (cam01..cam05): lower ring, height z = 0.35m, azimuths -50° to +50°
    - Tier 2 (cam06..cam10): upper ring, height z = 0.65m, azimuths -45° to +45°
    """
    if target_center is None:
        target_center = np.array([0.0, 0.0, 0.45], dtype=np.float64)

    # Azimuth angles in degrees (view from front looking toward +Y)
    lower_azimuths = [-48.0, -24.0, 0.0, 24.0, 48.0]
    upper_azimuths = [-42.0, -20.0, 4.0, 22.0, 44.0]

    cameras = []
    cam_id = 1

    for az in lower_azimuths:
        rad = math.radians(az)
        pos = np.array([radius * math.sin(rad), -radius * math.cos(rad), 0.35], dtype=np.float64)
        c2w = look_at(pos, target_center)
        qvec, tvec = opencv_c2w_to_colmap_qt(c2w)
        cameras.append({
            "cam_id": f"cam{cam_id:02d}",
            "name": f"image{cam_id}.jpg",
            "c2w": c2w,
            "qvec": qvec,
            "tvec": tvec,
            "tier": "lower",
            "azimuth_deg": az,
            "pos": pos,
        })
        cam_id += 1

    for az in upper_azimuths:
        rad = math.radians(az)
        pos = np.array([radius * math.sin(rad), -radius * math.cos(rad), 0.65], dtype=np.float64)
        c2w = look_at(pos, target_center)
        qvec, tvec = opencv_c2w_to_colmap_qt(c2w)
        cameras.append({
            "cam_id": f"cam{cam_id:02d}",
            "name": f"image{cam_id}.jpg",
            "c2w": c2w,
            "qvec": qvec,
            "tvec": tvec,
            "tier": "upper",
            "azimuth_deg": az,
            "pos": pos,
        })
        cam_id += 1

    return cameras


def look_at(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """OpenCV camera-to-world 4x4 matrix for eye looking at target."""
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)

    right = np.cross(fwd, world_up)
    right = right / (np.linalg.norm(right) + 1e-12)

    down = np.cross(fwd, right)

    R = np.stack([right, down, fwd], axis=1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


# -----------------------------------------------------------------------------
# 3D Point Cloud and Visual Model
# -----------------------------------------------------------------------------

def generate_3d_point_cloud(
    target_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic 3D points (xyz, rgb, is_dynamic_mask):
    - Static baseplate & frame (~600 points)
    - Static motor casing (~300 points)
    - Static backdrop (~300 points)
    - Dynamic oscillating rotor hub & blades (~600 points)
    Returns: (xyz, rgb_255, is_dynamic)
    """
    rng = np.random.default_rng(42)
    pts_list = []
    rgb_list = []
    dyn_list = []

    # 1. Static baseplate: grid from X=[-0.7, 0.7], Y=[-0.7, 0.7], Z=0.0
    gx, gy = np.meshgrid(np.linspace(-0.65, 0.65, 25), np.linspace(-0.65, 0.65, 25))
    base_xyz = np.stack([gx.flatten(), gy.flatten(), np.zeros_like(gx).flatten()], axis=1)
    base_rgb = np.zeros_like(base_xyz)
    # Checker color pattern
    checker = ((np.floor((base_xyz[:, 0] + 0.65) / 0.13) + np.floor((base_xyz[:, 1] + 0.65) / 0.13)) % 2 == 0)
    base_rgb[checker] = [210, 210, 215]
    base_rgb[~checker] = [45, 45, 50]
    pts_list.append(base_xyz)
    rgb_list.append(base_rgb)
    dyn_list.append(np.zeros(len(base_xyz), dtype=bool))

    # 2. Static vertical frame pillars
    pillar_x = [-0.45, 0.45, -0.45, 0.45]
    pillar_y = [-0.45, -0.45, 0.45, 0.45]
    pillar_pts = []
    pillar_rgbs = []
    for px, py in zip(pillar_x, pillar_y):
        pz = np.linspace(0.0, 0.85, 30)
        p = np.stack([np.full_like(pz, px), np.full_like(pz, py), pz], axis=1)
        pillar_pts.append(p)
        pillar_rgbs.append(np.full_like(p, [120, 125, 130]))
    pillar_pts = np.concatenate(pillar_pts, axis=0)
    pillar_rgbs = np.concatenate(pillar_rgbs, axis=0)
    pts_list.append(pillar_pts)
    rgb_list.append(pillar_rgbs)
    dyn_list.append(np.zeros(len(pillar_pts), dtype=bool))

    # 3. Static motor casing: box behind rotor, Y in [0.08, 0.35], X in [-0.15, 0.15], Z in [0.35, 0.55]
    mc_x, mc_z = np.meshgrid(np.linspace(-0.14, 0.14, 15), np.linspace(0.36, 0.54, 15))
    motor_front = np.stack([mc_x.flatten(), np.full_like(mc_x.flatten(), 0.08), mc_z.flatten()], axis=1)
    motor_rgb = np.full_like(motor_front, [75, 80, 85])
    pts_list.append(motor_front)
    rgb_list.append(motor_rgb)
    dyn_list.append(np.zeros(len(motor_front), dtype=bool))

    # 4. Static backdrop at Y=0.75, X in [-0.7, 0.7], Z in [0.1, 0.8]
    bg_x, bg_z = np.meshgrid(np.linspace(-0.6, 0.6, 20), np.linspace(0.15, 0.8, 15))
    bg_xyz = np.stack([bg_x.flatten(), np.full_like(bg_x.flatten(), 0.75), bg_z.flatten()], axis=1)
    bg_rgb = np.full_like(bg_xyz, [180, 185, 190])
    pts_list.append(bg_xyz)
    rgb_list.append(bg_rgb)
    dyn_list.append(np.zeros(len(bg_xyz), dtype=bool))

    # 5. Dynamic oscillating rotor hub & blades at target_center (c_x=0, c_y=0, c_z=0.45)
    cx, cy, cz = target_center
    hub_pts = []
    hub_rgbs = []

    # Concentric rings on rotor face (Y = -0.02)
    for r in [0.02, 0.05, 0.08, 0.11, 0.14]:
        n_ring = int(round(2 * math.pi * r * 200))
        angles = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
        px = cx + r * np.cos(angles)
        pz = cz + r * np.sin(angles)
        py = np.full_like(px, cy - 0.02)
        p = np.stack([px, py, pz], axis=1)
        # Alternate high-contrast marker colors
        col = np.zeros_like(p)
        col[::2] = [230, 230, 30]   # Bright yellow
        col[1::2] = [20, 20, 25]    # Deep black
        hub_pts.append(p)
        hub_rgbs.append(col)

    # 4 Rotor blades (horizontal & vertical arms with high-contrast anchor cross markers)
    for angle in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        arm_r = np.linspace(0.12, 0.28, 20)
        px = cx + arm_r * math.cos(angle)
        pz = cz + arm_r * math.sin(angle)
        py = np.full_like(px, cy - 0.025)
        p = np.stack([px, py, pz], axis=1)
        col = np.full_like(p, [220, 50, 40])  # Vibrant red blades
        hub_pts.append(p)
        hub_rgbs.append(col)

    hub_pts = np.concatenate(hub_pts, axis=0)
    hub_rgbs = np.concatenate(hub_rgbs, axis=0)
    pts_list.append(hub_pts)
    rgb_list.append(hub_rgbs)
    dyn_list.append(np.ones(len(hub_pts), dtype=bool))

    xyz = np.concatenate(pts_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0)
    is_dynamic = np.concatenate(dyn_list, axis=0)
    return xyz, rgb, is_dynamic


# -----------------------------------------------------------------------------
# Anti-Aliased Subpixel Frame Renderer
# -----------------------------------------------------------------------------

def project_point(
    p_world: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Projects a 3D world point into pixel (u, v) and camera depth z_c."""
    R_c2w = c2w[:3, :3]
    C = c2w[:3, 3]
    R_wc = R_c2w.T
    p_c = R_wc @ (p_world - C)
    zc = p_c[2]
    if zc <= 1e-3:
        return -9999.0, -9999.0, zc
    u = fx * (p_c[0] / zc) + cx
    v = fy * (p_c[1] / zc) + cy
    return float(u), float(v), float(zc)


def render_synthetic_frame(
    cam: dict[str, Any],
    time_s: float,
    target_center: np.ndarray,
    oscillation_axis: np.ndarray,
    amplitude_m: float,
    freq_hz: float,
    width: int = 1600,
    height: int = 900,
    focal: float = 1400.0,
    cx: float = 800.0,
    cy: float = 450.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders one frame at exact time_s with anti-aliased subpixel drawing.
    Returns: (rgb_img [H, W, 3] uint8, gt_motion_mask [H, W] uint8 in {0, 255})
    """
    shift = 4
    scale = 1 << shift
    c2w = cam["c2w"]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Background gradient (calm studio lighting)
    y_coords = np.linspace(35, 75, height, dtype=np.uint8)[:, None, None]
    img[:, :] = np.tile(y_coords, (1, width, 3))

    # 1. Render static backdrop grid (Y = 0.75)
    # 1. Render static backdrop panel (soft gradient without sharp corner grid)
    p_bg0 = np.array([-0.7, 0.75, 0.1])
    p_bg1 = np.array([0.7, 0.75, 0.85])
    u0, v0, z0 = project_point(p_bg0, c2w, focal, focal, cx, cy)
    u1, v1, z1 = project_point(p_bg1, c2w, focal, focal, cx, cy)
    if z0 > 0 and z1 > 0:
        cv2.rectangle(img, (int(round(u0)), int(round(v1))), (int(round(u1)), int(round(v0))), (65, 70, 78), -1)

    # 2. Render static checkerboard floor (Z = 0.0)
    tile_size = 0.15
    for xi in range(-4, 5):
        for yi in range(-4, 5):
            x0, x1 = xi * tile_size, (xi + 1) * tile_size
            y0, y1 = yi * tile_size, (yi + 1) * tile_size
            quad_3d = [
                np.array([x0, y0, 0.0]),
                np.array([x1, y0, 0.0]),
                np.array([x1, y1, 0.0]),
                np.array([x0, y1, 0.0]),
            ]
            quad_2d = []
            valid = True
            for qp in quad_3d:
                u, v, z = project_point(qp, c2w, focal, focal, cx, cy)
                if z <= 0.1:
                    valid = False
                    break
                quad_2d.append([int(round(u * scale)), int(round(v * scale))])
            if valid:
                pts = np.array(quad_2d, dtype=np.int32).reshape((-1, 1, 2))
                col = (180, 185, 190) if (xi + yi) % 2 == 0 else (45, 48, 52)
                cv2.fillPoly(img, [pts], col, cv2.LINE_AA, shift=shift)

    # 3. Render static motor casing behind rotor (Y in [0.08, 0.35], X in [-0.15, 0.15], Z in [0.32, 0.58])
    box_corners = [
        np.array([-0.14, 0.08, 0.32]), np.array([0.14, 0.08, 0.32]),
        np.array([0.14, 0.08, 0.58]), np.array([-0.14, 0.08, 0.58]),
    ]
    box_2d = []
    box_valid = True
    for bp in box_corners:
        u, v, z = project_point(bp, c2w, focal, focal, cx, cy)
        if z <= 0.1:
            box_valid = False
            break
        box_2d.append([int(round(u * scale)), int(round(v * scale))])
    if box_valid:
        pts = np.array(box_2d, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (80, 85, 95), cv2.LINE_AA, shift=shift)
        cv2.polylines(img, [pts], True, (30, 32, 36), 2, cv2.LINE_AA, shift=shift)

    # 4. Render DYNAMIC oscillating rotor assembly
    # Physical sinusoidal translation: d(t) = A * sin(2*pi*f*t) * u
    d_trans = (amplitude_m * math.sin(2.0 * math.pi * freq_hz * time_s)) * oscillation_axis
    dyn_center = target_center + d_trans

    # Project rotor center & radius
    cu, cv, cz = project_point(dyn_center, c2w, focal, focal, cx, cy)
    if cz <= 0.1:
        return img, mask

    c_sub = (int(round(cu * scale)), int(round(cv * scale)))
    r_hub_m = 0.13
    r_hub_px = (focal / cz) * r_hub_m
    r_sub = int(round(r_hub_px * scale))

    # Mask moving region
    cv2.circle(mask, (int(round(cu)), int(round(cv))), int(math.ceil(r_hub_px * 2.3)), 255, -1)

    # Draw 4 Rotor Blades
    blade_length_m = 0.28
    blade_width_m = 0.045
    blade_angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    for b_ang in blade_angles:
        # Radial vector in (X, Z) plane
        rad_vec = np.array([math.cos(b_ang), 0.0, math.sin(b_ang)])
        tan_vec = np.array([-math.sin(b_ang), 0.0, math.cos(b_ang)])

        b_pts_3d = [
            dyn_center + 0.06 * rad_vec - (blade_width_m / 2) * tan_vec,
            dyn_center + blade_length_m * rad_vec - (blade_width_m / 3) * tan_vec,
            dyn_center + blade_length_m * rad_vec + (blade_width_m / 3) * tan_vec,
            dyn_center + 0.06 * rad_vec + (blade_width_m / 2) * tan_vec,
        ]
        b_pts_2d = []
        b_valid = True
        for bp in b_pts_3d:
            u, v, z = project_point(bp, c2w, focal, focal, cx, cy)
            if z <= 0.1:
                b_valid = False
                break
            b_pts_2d.append([int(round(u * scale)), int(round(v * scale))])
        if b_valid:
            pts = np.array(b_pts_2d, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (210, 55, 45), cv2.LINE_AA, shift=shift)
            cv2.polylines(img, [pts], True, (40, 20, 20), 2, cv2.LINE_AA, shift=shift)

            # High-contrast optical tracking anchor markers on each blade tip
            tip_3d = dyn_center + (blade_length_m - 0.035) * rad_vec
            tu, tv, tz = project_point(tip_3d, c2w, focal, focal, cx, cy)
            if tz > 0.1:
                t_sub = (int(round(tu * scale)), int(round(tv * scale)))
                m_rad_px = (focal / tz) * 0.018
                cv2.circle(img, t_sub, int(round(m_rad_px * scale)), (245, 245, 250), -1, cv2.LINE_AA, shift=shift)
                cv2.circle(img, t_sub, int(round((m_rad_px * 0.5) * scale)), (15, 15, 20), -1, cv2.LINE_AA, shift=shift)

    # Draw Rotor Hub Disc
    cv2.circle(img, c_sub, r_sub, (230, 230, 235), -1, cv2.LINE_AA, shift=shift)
    cv2.circle(img, c_sub, r_sub, (30, 30, 35), 2, cv2.LINE_AA, shift=shift)

    # High-Contrast Quadrant Pattern on Rotor Hub Face
    r_quad_px = r_hub_px * 0.85
    r_quad_sub = int(round(r_quad_px * scale))
    cv2.ellipse(img, c_sub, (r_quad_sub, r_quad_sub), 0, 0, 90, (25, 25, 30), -1, cv2.LINE_AA, shift=shift)
    cv2.ellipse(img, c_sub, (r_quad_sub, r_quad_sub), 0, 180, 270, (25, 25, 30), -1, cv2.LINE_AA, shift=shift)
    cv2.ellipse(img, c_sub, (r_quad_sub, r_quad_sub), 0, 90, 180, (240, 240, 30), -1, cv2.LINE_AA, shift=shift)
    cv2.ellipse(img, c_sub, (r_quad_sub, r_quad_sub), 0, 270, 360, (240, 240, 30), -1, cv2.LINE_AA, shift=shift)

    # Central Anchor Bullseye & Crosshair
    r_center_px = r_hub_px * 0.35
    r_center_sub = int(round(r_center_px * scale))
    cv2.circle(img, c_sub, r_center_sub, (245, 245, 250), -1, cv2.LINE_AA, shift=shift)
    cv2.circle(img, c_sub, int(round(r_center_sub * 0.45)), (20, 20, 25), -1, cv2.LINE_AA, shift=shift)

    ch_len = int(round(r_hub_px * 1.15 * scale))
    cv2.line(img, (c_sub[0] - ch_len, c_sub[1]), (c_sub[0] + ch_len, c_sub[1]), (245, 245, 250), 2, cv2.LINE_AA, shift=shift)
    cv2.line(img, (c_sub[0], c_sub[1] - ch_len), (c_sub[0], c_sub[1] + ch_len), (245, 245, 250), 2, cv2.LINE_AA, shift=shift)

    # 4 High-contrast checkerboard corner targets on rotor face (Shi-Tomasi corner anchors)
    for off_x, off_z in [(-0.065, -0.065), (0.065, -0.065), (-0.065, 0.065), (0.065, 0.065)]:
        anchor_3d = dyn_center + np.array([off_x, -0.022, off_z])
        au, av, az = project_point(anchor_3d, c2w, focal, focal, cx, cy)
        if az > 0.1:
            tile_px = max(4, int(round((focal / az) * 0.012)))
            au_i, av_i = int(round(au)), int(round(av))
            for row in range(-2, 2):
                for col in range(-2, 2):
                    c_col = (250, 250, 250) if (row + col) % 2 == 0 else (10, 10, 15)
                    cv2.rectangle(img, (au_i + col * tile_px, av_i + row * tile_px),
                                  (au_i + (col + 1) * tile_px, av_i + (row + 1) * tile_px), c_col, -1)

    return img, mask


# -----------------------------------------------------------------------------
# Main Dataset Generation Orchestrator
# -----------------------------------------------------------------------------

def generate_synthetic_scene(
    out_dir: Path | str = DEFAULT_OUT_DIR,
    n_frames: int = 60,
    fps: float = DEFAULT_FPS,
    freq_hz: float = NOMINAL_WOBBLE_HZ,
    target_amplitude_px: float = 2.8,
    width: int = 1600,
    height: int = 900,
    focal: float = 1400.0,
    offsets_table: dict[str, float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Constructs the synthetic pre-flight scene in out_dir.
    Returns the metadata summary dictionary.
    """
    out_path = Path(out_dir).resolve()
    if offsets_table is None:
        offsets_table = dict(MEASURED_OFFSETS_S)

    print(f"[T38] Generating synthetic pre-flight scene at: {out_path}")
    print(f"      Resolution: {width}x{height}, FPS: {fps}, Frames: {n_frames}, Frequency: {freq_hz:.3f} Hz")

    out_path.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_path / "sparse_"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_path / "motion_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    target_center = np.array([0.0, 0.0, 0.45], dtype=np.float64)
    oscillation_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Lateral oscillation (X-axis)

    # Calculate physical 3D amplitude in meters to produce target_amplitude_px in frontal views
    # dx_px = (focal / distance) * A_m  ==>  A_m = dx_px * distance / focal
    rig_radius = 1.8
    amplitude_m = target_amplitude_px * rig_radius / focal
    print(f"      Target amplitude: {target_amplitude_px:.2f} px  ->  Physical 3D amplitude: {amplitude_m * 1000.0:.3f} mm")

    # 1. Build camera rig
    cameras = build_10cam_rig(radius=rig_radius, target_center=target_center)
    cx = width / 2.0
    cy = height / 2.0

    # 2. Write COLMAP binary structures
    write_cameras_bin(sparse_dir / "cameras.bin", width, height, focal, focal, cx, cy)
    write_images_bin(sparse_dir / "images.bin", cameras)

    # 3. Generate and write 3D point cloud
    xyz, rgb, is_dynamic = generate_3d_point_cloud(target_center)
    write_points3D_bin(sparse_dir / "points3D.bin", xyz, rgb)
    write_ply(out_path / "points3D_multipleview.ply", xyz, rgb)

    # 4. Write poses_bounds_multipleview.npy (LLFF format, shape [10, 17])
    llff_rows = [opencv_c2w_to_llff_row(c["c2w"], height, width, focal, near=0.5, far=5.0) for c in cameras]
    poses_bounds = np.stack(llff_rows, axis=0)
    np.save(out_path / "poses_bounds_multipleview.npy", poses_bounds)

    # 5. Render frames with injected offsets
    cam_motion_masks = {}
    per_cam_projected_amplitudes = {}

    for cam in cameras:
        cid = cam["cam_id"]
        cam_dir = out_path / cid
        cam_dir.mkdir(parents=True, exist_ok=True)
        offset_s = offsets_table.get(cid, 0.0)

        # Projected 2D amplitude along principal motion direction for this camera view
        # R_c2w column 0 is camera right vector
        right_vec = cam["c2w"][:3, 0]
        proj_factor = abs(float(np.dot(oscillation_axis, right_vec)))
        per_cam_projected_amplitudes[cid] = float(target_amplitude_px * proj_factor)

        last_mask = None
        for i in range(n_frames):
            # Injected continuous sampling instant: t_k(i) = i / fps + delta_k
            t_k = (i / fps) + offset_s
            frame_img, mask_img = render_synthetic_frame(
                cam=cam,
                time_s=t_k,
                target_center=target_center,
                oscillation_axis=oscillation_axis,
                amplitude_m=amplitude_m,
                freq_hz=freq_hz,
                width=width,
                height=height,
                focal=focal,
                cx=cx,
                cy=cy,
            )
            frame_path = cam_dir / f"frame_{i + 1:05d}.jpg"
            cv2.imwrite(str(frame_path), frame_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            last_mask = mask_img

        cam_motion_masks[cid] = last_mask

    # 6. Save precomputed motion masks (T36 format)
    mask_manifest = {
        "scene": out_path.name,
        "n_cameras": len(cameras),
        "bandpass_hz": [6.0, 9.0],
        "mask_max_weight": 50.0,
        "cameras": {},
    }
    for cid, m in cam_motion_masks.items():
        norm_mask = (m.astype(np.float32) / 255.0)
        np.save(masks_dir / f"{cid}.npy", norm_mask)
        cv2.imwrite(str(masks_dir / f"{cid}_vis.png"), m)

        # Generate color heatmap
        heatmap = cv2.applyColorMap((norm_mask * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(masks_dir / f"{cid}_heatmap.png"), heatmap)

        mask_manifest["cameras"][cid] = {
            "active_pixel_count": int(np.sum(m > 0)),
            "active_fraction": float(np.mean(m > 0)),
        }

    with open(masks_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(mask_manifest, f, indent=2)

    # 7. Write authoritative offsets.json (matching T32 schema)
    offsets_data = {
        "scene": out_path.name,
        "fps": float(fps),
        "frame_count": int(n_frames),
        "duration_s": float(n_frames / fps),
        "reference_camera": "cam01",
        "exposure_s": 0.010,
        "mid_exposure_corrected": True,
        "offsets_s": {c["cam_id"]: float(offsets_table[c["cam_id"]]) for c in cameras},
        "mean_offsets_s": {c["cam_id"]: float(offsets_table[c["cam_id"]]) for c in cameras},
        "drift_s_per_s": {c["cam_id"]: 0.0 for c in cameras},
        "cameras": {},
        "provenance": {
            "method": "synthetic_ground_truth_injection",
            "source": "scripts/gen_synthetic_wobble_scene.py",
            "nominal_frequency_hz": freq_hz,
            "target_3d_amplitude_mm": float(amplitude_m * 1000.0),
        },
    }
    for c in cameras:
        cid = c["cam_id"]
        off_s = offsets_table[cid]
        offsets_data["cameras"][cid] = {
            "offset_s": float(off_s),
            "offset_ms": float(off_s * 1000.0),
            "mean_offset_s": float(off_s),
            "mean_offset_ms": float(off_s * 1000.0),
            "drift_s_per_s": 0.0,
            "drift_ms_per_s": 0.0,
            "requires_affine": False,
            "exposure_s": 0.010,
            "residual_stats": {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            },
            "split_reproducibility": {
                "first_half_offset_s": float(off_s),
                "second_half_offset_s": float(off_s),
                "drift_consistency_s": 0.0,
            },
        }

    with open(out_path / "offsets.json", "w", encoding="utf-8") as f:
        json.dump(offsets_data, f, indent=2)

    # 8. Write gt_motion_info.json
    gt_info = {
        "scene": out_path.name,
        "generator": "scripts/gen_synthetic_wobble_scene.py",
        "wobble_frequency_hz": float(freq_hz),
        "wobble_period_s": float(1.0 / freq_hz),
        "fps": float(fps),
        "n_frames": int(n_frames),
        "oscillation_axis": [float(v) for v in oscillation_axis],
        "target_center": [float(v) for v in target_center],
        "amplitude_3d_m": float(amplitude_m),
        "amplitude_3d_mm": float(amplitude_m * 1000.0),
        "target_projected_amplitude_px": float(target_amplitude_px),
        "per_camera_projected_amplitudes_px": per_cam_projected_amplitudes,
        "injected_offsets_s": offsets_table,
        "n_total_points3d": int(len(xyz)),
        "n_dynamic_points3d": int(np.sum(is_dynamic)),
        "n_static_points3d": int(np.sum(~is_dynamic)),
    }
    with open(out_path / "gt_motion_info.json", "w", encoding="utf-8") as f:
        json.dump(gt_info, f, indent=2)

    print(f"[T38] Successfully generated synthetic scene in {out_path}")
    print(f"      Wrote 10 camera folders ({n_frames} frames each), sparse COLMAP model,")
    print(f"      points3D_multipleview.ply ({len(xyz)} vertices), offsets.json, and motion_masks/.")
    return gt_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output scene directory")
    parser.add_argument("--n-frames", type=int, default=60, help="Number of frames per camera (default: 60)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame rate (default: 20.0)")
    parser.add_argument("--freq-hz", type=float, default=NOMINAL_WOBBLE_HZ, help="Wobble frequency in Hz (default: 7.112)")
    parser.add_argument("--amp-px", type=float, default=2.8, help="Target 2D projected pixel amplitude (default: 2.8)")
    parser.add_argument("--width", type=int, default=1600, help="Image width (default: 1600)")
    parser.add_argument("--height", type=int, default=900, help="Image height (default: 900)")
    parser.add_argument("--focal", type=float, default=1400.0, help="Camera focal length (default: 1400.0)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")

    args = parser.parse_args()
    generate_synthetic_scene(
        out_dir=args.out_dir,
        n_frames=args.n_frames,
        fps=args.fps,
        freq_hz=args.freq_hz,
        target_amplitude_px=args.amp_px,
        width=args.width,
        height=args.height,
        focal=args.focal,
        force=args.force,
    )


if __name__ == "__main__":
    main()
