#!/usr/bin/env python3
"""
gen_spinning_cubes.py — generate synthetic benchmark scenes with k independently spinning cubes.

Scene layout:
* k plain cubes spinning independently around their vertical axes, non-overlapping,
  placed at random within an invisible 3 m circle (diameter 3.0 m, radius 1.5 m) on the floor.
* Encased in a simple room with checkerboard patterns on the floor and walls.
* Room walls placed 2–3 m behind the cameras (e.g. 2.5 m margin).
* 10 cameras arranged in a circle looking at the center of the scene.
* A single light source located near one of the top corners of the room.
* Stage uses Z-up coordinates and 1.0 m units (metersPerUnit = 1.0).
* Compatible with OpenUSD / Isaac Sim capture (omni_capture.py) and 4DGS pipeline (omni_to_4dgs.py).

Usage:
    python scene-gen/gen_spinning_cubes.py --k 1 2 3 4 5 6 7 8 --out-dir omniverse-pipeline/data/scenes/cubes
    python scene-gen/gen_spinning_cubes.py --selftest
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Procedural Checkerboard Texture Generator (Stdlib + zlib, zero heavy deps)
# -----------------------------------------------------------------------------

def write_png_rgb(path: str | Path, img: np.ndarray) -> None:
    """Write an RGB uint8 HxWx3 array as PNG using Python stdlib (no Pillow required)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def generate_checkerboard_texture(
    out_path: str | Path,
    img_size: int = 1024,
    tiles_per_axis: int = 8,
    color_a: Tuple[int, int, int] = (240, 240, 240),
    color_b: Tuple[int, int, int] = (30, 30, 35),
    line_color: Tuple[int, int, int] = (15, 15, 18),
    border_width: int = 2,
) -> str:
    """Generate a high-resolution crisp checkerboard texture with subtle border lines."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    tile_px = img_size // tiles_per_axis

    for row in range(tiles_per_axis):
        for col in range(tiles_per_axis):
            c = color_a if (row + col) % 2 == 0 else color_b
            y0, y1 = row * tile_px, (row + 1) * tile_px
            x0, x1 = col * tile_px, (col + 1) * tile_px
            img[y0:y1, x0:x1] = c
            if border_width > 0:
                img[y0 : y0 + border_width, x0:x1] = line_color
                img[y1 - border_width : y1, x0:x1] = line_color
                img[y0:y1, x0 : x0 + border_width] = line_color
                img[y0:y1, x1 - border_width : x1] = line_color

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_png_rgb(out_path, img)
    return str(out_path)


# -----------------------------------------------------------------------------
# Camera Rig Mathematics
# -----------------------------------------------------------------------------

def look_at_opencv(
    eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> np.ndarray:
    """Return an OpenCV camera-to-world 4x4 matrix for camera at eye looking at target."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    fwd = target - eye
    norm_fwd = np.linalg.norm(fwd)
    if norm_fwd < 1e-9:
        fwd = np.array([0.0, 1.0, 0.0])
    else:
        fwd /= norm_fwd

    if abs(np.dot(fwd, world_up / np.linalg.norm(world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)  # +Y down in OpenCV

    R = np.stack([right, down, fwd], axis=1)  # columns = camera axes in world
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


def build_camera_circle(
    n_cameras: int = 10,
    radius: float = 3.0,
    height: float = 1.2,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.2),
    start_deg: float = 0.0,
) -> List[np.ndarray]:
    """Generate n_cameras in a circle around the origin looking toward target."""
    poses = []
    target_arr = np.asarray(target, dtype=float)
    for i in range(n_cameras):
        angle = math.radians(start_deg) + 2.0 * math.pi * i / n_cameras
        eye = np.array([radius * math.cos(angle), radius * math.sin(angle), height])
        poses.append(look_at_opencv(eye, target_arr, world_up=np.array([0.0, 0.0, 1.0])))
    return poses


def build_camera_stacked_arcs(
    n_cameras: int = 10,
    radius: float = 3.0,
    height_lower: float = 0.8,
    height_upper: float = 1.6,
    arc_deg: float = 60.0,
    center_deg: float = 0.0,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.2),
) -> List[np.ndarray]:
    """Generate n_cameras in two stacked elevation arcs spanning arc_deg on one side of center."""
    n_lower = n_cameras // 2
    n_upper = n_cameras - n_lower
    poses = []
    target_arr = np.asarray(target, dtype=float)

    # Lower arc
    angles_lower = np.linspace(center_deg - arc_deg / 2.0, center_deg + arc_deg / 2.0, n_lower)
    for deg in angles_lower:
        ang = math.radians(deg)
        eye = np.array([radius * math.cos(ang), radius * math.sin(ang), height_lower])
        poses.append(look_at_opencv(eye, target_arr, world_up=np.array([0.0, 0.0, 1.0])))

    # Upper arc
    angles_upper = np.linspace(center_deg - arc_deg / 2.0, center_deg + arc_deg / 2.0, n_upper)
    for deg in angles_upper:
        ang = math.radians(deg)
        eye = np.array([radius * math.cos(ang), radius * math.sin(ang), height_upper])
        poses.append(look_at_opencv(eye, target_arr, world_up=np.array([0.0, 0.0, 1.0])))

    return poses


def build_camera_stacked_ring(
    n_cameras: int = 16,
    radius: float = 3.0,
    height_lower: float = 0.8,
    height_upper: float = 1.3,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.2),
) -> List[np.ndarray]:
    """Generate n_cameras in two stacked 360-degree rings (half on lower tier, half on upper tier)."""
    n_lower = n_cameras // 2
    n_upper = n_cameras - n_lower
    poses = []
    target_arr = np.asarray(target, dtype=float)

    # Lower tier
    for i in range(n_lower):
        ang = 2.0 * math.pi * i / n_lower
        eye = np.array([radius * math.cos(ang), radius * math.sin(ang), height_lower])
        poses.append(look_at_opencv(eye, target_arr, world_up=np.array([0.0, 0.0, 1.0])))

    # Upper tier (offset by half step for uniform coverage)
    offset = math.pi / n_upper
    for i in range(n_upper):
        ang = 2.0 * math.pi * i / n_upper + offset
        eye = np.array([radius * math.cos(ang), radius * math.sin(ang), height_upper])
        poses.append(look_at_opencv(eye, target_arr, world_up=np.array([0.0, 0.0, 1.0])))

    return poses


# -----------------------------------------------------------------------------
# Cube Placement & Non-Overlapping Rejection Sampling
# -----------------------------------------------------------------------------

def sample_cube_placements(
    k: int,
    circle_radius: float = 1.5,
    cube_size: float = 0.35,
    margin: float = 0.08,
    motion_mode: str = "both",
    seed: int = 42,
    max_tries: int = 20000,
) -> List[Dict[str, Any]]:
    """Sample k non-overlapping cube positions placed randomly on the floor within circle_radius."""
    rng = np.random.default_rng(seed)
    placed: List[Dict[str, Any]] = []

    # Maximum bounding radius of a cube in 2D (xy plane diagonal radius)
    diag_radius = (math.sqrt(2.0) * cube_size) / 2.0
    valid_placement_radius = circle_radius - diag_radius

    if valid_placement_radius <= 0:
        raise ValueError(
            f"Cube size {cube_size}m is too large for circle radius {circle_radius}m!"
        )

    for i in range(k):
        success = False
        for _ in range(max_tries):
            # Uniform random sampling in circle
            r = valid_placement_radius * math.sqrt(rng.uniform(0.0, 1.0))
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x = float(r * math.cos(theta))
            y = float(r * math.sin(theta))

            # Check overlap with all previously placed cubes
            overlap = False
            for p in placed:
                dist = math.hypot(x - p["x"], y - p["y"])
                min_dist = 2.0 * diag_radius + margin
                if dist < min_dist:
                    overlap = True
                    break

            if not overlap:
                # Random integer spin cycles over the clip (e.g. 1 to 5 cycles, clockwise or counter-clockwise)
                cycles = int(rng.integers(1, 6)) * int(rng.choice([-1, 1]))
                init_rot_deg = float(rng.uniform(0.0, 360.0))
                trans_angle_deg = float(rng.uniform(0.0, 360.0))
                trans_amp_m = float(rng.uniform(0.20, 0.40))
                trans_cycles = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
                # Distinct saturated HSV color (golden ratio spacing)
                r_col, g_col, b_col = colorsys.hsv_to_rgb(
                    (0.15 + i * 0.61803398875) % 1.0, 0.85, 0.85
                )

                placed.append(
                    {
                        "index": i,
                        "name": f"cube_{i+1:02d}",
                        "x": x,
                        "y": y,
                        "z": cube_size / 2.0,  # Resting on the floor at z=0
                        "size": cube_size,
                        "cycles": cycles,
                        "init_rot_deg": init_rot_deg,
                        "motion_mode": motion_mode,
                        "trans_angle_deg": trans_angle_deg,
                        "trans_amp_m": trans_amp_m,
                        "trans_cycles": trans_cycles,
                        "color": [float(r_col), float(g_col), float(b_col)],
                    }
                )
                success = True
                break

        if not success:
            raise RuntimeError(
                f"Could not place {k} non-overlapping cubes within {circle_radius*2:.1f}m circle "
                f"after {max_tries} iterations. Try reducing k, cube_size, or margin."
            )

    return placed


# -----------------------------------------------------------------------------
# USD Stage Authoring (Pure OpenUSD / pxr)
# -----------------------------------------------------------------------------

def build_usd_scene(
    usd_out_path: str | Path,
    cubes_info: List[Dict[str, Any]],
    room_width: float,
    room_depth: float,
    room_height: float,
    checker_tex_rel_path: str,
    camera_poses: List[np.ndarray],
    light_pos: Tuple[float, float, float],
    num_frames: int = 60,
    fps: float = 24.0,
    vfov_deg: float = 45.0,
    width_px: int = 1600,
    height_px: int = 900,
    tile_size: float = 1.5,
) -> None:
    """Build the complete OpenUSD scene with room, textures, light, cameras, and animated cubes."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    if os.path.exists(usd_out_path):
        os.remove(usd_out_path)

    stage = Usd.Stage.CreateNew(str(usd_out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)

    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    # -------------------------------------------------------------------------
    # 1. Materials: Checkerboard Material
    # -------------------------------------------------------------------------
    mats_scope = stage.DefinePrim("/World/Materials", "Scope")
    checker_mat_path = "/World/Materials/CheckerMaterial"
    checker_mat = UsdShade.Material.Define(stage, checker_mat_path)

    pbr_shader = UsdShade.Shader.Define(stage, f"{checker_mat_path}/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    tex_shader = UsdShade.Shader.Define(stage, f"{checker_mat_path}/CheckerTexture")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(checker_tex_rel_path))
    tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    st_reader = UsdShade.Shader.Define(stage, f"{checker_mat_path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_shader.ConnectableAPI(), "rgb"
    )
    checker_mat.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")

    # -------------------------------------------------------------------------
    # 2. Room Geometry: Floor, 4 Walls, and Ceiling
    # -------------------------------------------------------------------------
    room_scope = stage.DefinePrim("/World/Room", "Scope")
    hw = room_width / 2.0
    hd = room_depth / 2.0

    def create_quad_mesh(
        name: str,
        corners: List[Tuple[float, float, float]],
        uvs: List[Tuple[float, float]],
    ) -> UsdGeom.Mesh:
        mesh_path = f"/World/Room/{name}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*c) for c in corners]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))

        # UV mapping
        st_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        st_primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))

        # Bind checkerboard material
        UsdShade.MaterialBindingAPI(mesh).Bind(checker_mat)
        return mesh

    # Floor quad (z = 0)
    uv_w = room_width / tile_size
    uv_d = room_depth / tile_size
    uv_h = room_height / tile_size

    create_quad_mesh(
        "Floor",
        [(-hw, -hd, 0.0), (hw, -hd, 0.0), (hw, hd, 0.0), (-hw, hd, 0.0)],
        [(0.0, 0.0), (uv_w, 0.0), (uv_w, uv_d), (0.0, uv_d)],
    )

    # Ceiling quad (z = room_height)
    create_quad_mesh(
        "Ceiling",
        [(-hw, -hd, room_height), (-hw, hd, room_height), (hw, hd, room_height), (hw, -hd, room_height)],
        [(0.0, 0.0), (0.0, uv_d), (uv_w, uv_d), (uv_w, 0.0)],
    )

    # Wall East (+X)
    create_quad_mesh(
        "Wall_East",
        [(hw, -hd, 0.0), (hw, hd, 0.0), (hw, hd, room_height), (hw, -hd, room_height)],
        [(0.0, 0.0), (uv_d, 0.0), (uv_d, uv_h), (0.0, uv_h)],
    )

    # Wall West (-X)
    create_quad_mesh(
        "Wall_West",
        [(-hw, hd, 0.0), (-hw, -hd, 0.0), (-hw, -hd, room_height), (-hw, hd, room_height)],
        [(0.0, 0.0), (uv_d, 0.0), (uv_d, uv_h), (0.0, uv_h)],
    )

    # Wall North (+Y)
    create_quad_mesh(
        "Wall_North",
        [(hw, hd, 0.0), (-hw, hd, 0.0), (-hw, hd, room_height), (hw, hd, room_height)],
        [(0.0, 0.0), (uv_w, 0.0), (uv_w, uv_h), (0.0, uv_h)],
    )

    # Wall South (-Y)
    create_quad_mesh(
        "Wall_South",
        [(-hw, -hd, 0.0), (hw, -hd, 0.0), (hw, -hd, room_height), (-hw, -hd, room_height)],
        [(0.0, 0.0), (uv_w, 0.0), (uv_w, uv_h), (0.0, uv_h)],
    )

    # -------------------------------------------------------------------------
    # 3. Single Light Source near Top Corner
    # -------------------------------------------------------------------------
    lights_scope = stage.DefinePrim("/World/Lights", "Scope")
    light_prim = UsdLux.SphereLight.Define(stage, "/World/Lights/CornerLight")
    light_prim.CreateIntensityAttr(250000.0)
    light_prim.CreateRadiusAttr(0.25)
    light_prim.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    xf_light = UsdGeom.Xformable(light_prim.GetPrim())
    xf_light.AddTranslateOp().Set(Gf.Vec3f(*light_pos))

    # -------------------------------------------------------------------------
    # 4. Animated Cubes
    # -------------------------------------------------------------------------
    cubes_root = stage.DefinePrim("/World/Cubes", "Xform")

    for cube_info in cubes_info:
        name = cube_info["name"]
        cube_xform = UsdGeom.Xform.Define(stage, f"/World/Cubes/{name}")
        cube_mesh = UsdGeom.Cube.Define(stage, f"/World/Cubes/{name}/mesh")
        cube_mesh.GetSizeAttr().Set(cube_info["size"])

        # Cube Material (Solid distinct color)
        cube_mat_path = f"/World/Materials/{name}_Mat"
        cube_mat = UsdShade.Material.Define(stage, cube_mat_path)
        cube_pbr = UsdShade.Shader.Define(stage, f"{cube_mat_path}/PBRShader")
        cube_pbr.CreateIdAttr("UsdPreviewSurface")
        col = cube_info["color"]
        cube_pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(col[0], col[1], col[2]))
        cube_pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        cube_pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        cube_mat.CreateSurfaceOutput().ConnectToSource(cube_pbr.ConnectableAPI(), "surface")

        UsdShade.MaterialBindingAPI(cube_mesh).Bind(cube_mat)
        cube_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(col[0], col[1], col[2])]))

        # Animation (Rotation, Translation, or Both)
        xf = UsdGeom.Xformable(cube_xform.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        r_op = xf.AddRotateZOp()

        cycles = cube_info["cycles"]
        init_rot = cube_info["init_rot_deg"]
        x0, y0, z0 = cube_info["x"], cube_info["y"], cube_info["z"]

        motion_mode = cube_info.get("motion_mode", "both")
        trans_angle = cube_info.get("trans_angle_deg", 45.0)
        trans_amp = cube_info.get("trans_amp_m", 0.3)
        trans_cycles = cube_info.get("trans_cycles", 1.5)

        dx = trans_amp * math.cos(math.radians(trans_angle))
        dy = trans_amp * math.sin(math.radians(trans_angle))

        for fi in range(num_frames):
            u = fi / max(num_frames - 1, 1)

            if motion_mode in ("rotation", "both", "rotation+translation"):
                angle = (init_rot + 360.0 * cycles * u) % 360.0
            else:
                angle = init_rot

            if motion_mode in ("translation", "both", "rotation+translation"):
                osc = math.sin(2.0 * math.pi * trans_cycles * u)
                x_t = x0 + dx * osc
                y_t = y0 + dy * osc
            else:
                x_t, y_t = x0, y0

            t_op.Set(Gf.Vec3f(float(x_t), float(y_t), float(z0)), Usd.TimeCode(fi))
            r_op.Set(float(angle), Usd.TimeCode(fi))

    # -------------------------------------------------------------------------
    # 5. Authored 10-Camera Rig in USD
    # -------------------------------------------------------------------------
    cam_scope = stage.DefinePrim("/World/CaptureCameras", "Scope")
    v_ap = 20.955
    focal_mm = (v_ap / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    h_ap = v_ap * (width_px / height_px)

    for i, c2w in enumerate(camera_poses, start=1):
        cam_path = f"/World/CaptureCameras/cam{i:02d}"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(float(focal_mm))
        cam.CreateVerticalApertureAttr(float(v_ap))
        cam.CreateHorizontalApertureAttr(float(h_ap))

        # OpenCV c2w -> USD camera basis
        R = c2w[:3, :3]
        right, up, back = R[:, 0], -R[:, 1], -R[:, 2]
        M = Gf.Matrix4d(
            float(right[0]), float(right[1]), float(right[2]), 0.0,
            float(up[0]), float(up[1]), float(up[2]), 0.0,
            float(back[0]), float(back[1]), float(back[2]), 0.0,
            float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3]), 1.0,
        )
        xf_cam = UsdGeom.Xformable(cam.GetPrim())
        xf_cam.ClearXformOpOrder()
        xf_cam.AddTransformOp().Set(M)

    stage.GetRootLayer().Save()


# -----------------------------------------------------------------------------
# Main Scene Generator & Capture Config Writer
# -----------------------------------------------------------------------------

def generate_spinning_cubes_scene(
    k: int = 2,
    circle_radius: float = 1.5,
    cube_size: float = 0.35,
    num_cameras: int = 16,
    camera_radius: float = 3.0,
    camera_height: float = 1.2,
    camera_layout: str = "stacked_arcs",
    arc_deg: float = 80.0,
    center_deg: float = 0.0,
    height_lower: float = 0.8,
    height_upper: float = 1.3,
    motion_mode: str = "both",
    wall_margin: float = 2.5,
    room_height: float = 4.5,
    tile_size: float = 1.5,
    num_frames: int = 120,
    fps: float = 60.0,
    vfov_deg: float = 45.0,
    width_px: int = 1280,
    height_px: int = 720,
    seed: int = 42,
    out_dir: str | Path = "omniverse-pipeline/data/scenes/cubes",
    name: str = "spinning_cubes",
    capture_root: str = "Q:/Omniverse/renders",
) -> Dict[str, Any]:
    """Generate the complete spinning cubes benchmark scene files."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Texture Generation (fewer tiles_per_axis for larger checker size)
    tex_filename = f"{name}_checker.png"
    tex_path = out_dir / tex_filename
    generate_checkerboard_texture(tex_path, img_size=1024, tiles_per_axis=4)

    # 2. Placement and Geometry calculations
    cubes_info = sample_cube_placements(
        k=k, circle_radius=circle_radius, cube_size=cube_size, motion_mode=motion_mode, seed=seed
    )

    room_width = 2.0 * (camera_radius + wall_margin)
    room_depth = 2.0 * (camera_radius + wall_margin)

    # Corner light placed near top corner (+X, +Y, +Z)
    light_pos = (
        room_width / 2.0 - 0.6,
        room_depth / 2.0 - 0.6,
        room_height - 0.5,
    )

    # 3. Camera Poses (OpenCV convention)
    if camera_layout in ("stacked_ring", "ring_stacked"):
        camera_poses = build_camera_stacked_ring(
            n_cameras=num_cameras,
            radius=camera_radius,
            height_lower=height_lower,
            height_upper=height_upper,
            target=(0.0, 0.0, cube_size / 2.0),
        )
        rig_yaml = f"""rig:
  layout: stacked_ring
  n_cameras: {num_cameras}
  radius: {camera_radius}
  height_lower: {height_lower}
  height_upper: {height_upper}
  center: [0.0, 0.0, 0.2]
  world_up: [0, 0, 1]"""
    elif camera_layout in ("stacked_arcs", "arc_stacked"):
        camera_poses = build_camera_stacked_arcs(
            n_cameras=num_cameras,
            radius=camera_radius,
            height_lower=height_lower,
            height_upper=height_upper,
            arc_deg=arc_deg,
            center_deg=center_deg,
            target=(0.0, 0.0, cube_size / 2.0),
        )
        rig_yaml = f"""rig:
  layout: stacked_arcs
  n_cameras: {num_cameras}
  radius: {camera_radius}
  height_lower: {height_lower}
  height_upper: {height_upper}
  arc_deg: {arc_deg}
  center_deg: {center_deg}
  center: [0.0, 0.0, 0.2]
  world_up: [0, 0, 1]"""
    else:
        camera_poses = build_camera_circle(
            n_cameras=num_cameras,
            radius=camera_radius,
            height=camera_height,
            target=(0.0, 0.0, cube_size / 2.0),
        )
        rig_yaml = f"""rig:
  layout: ring
  n_cameras: {num_cameras}
  radius: {camera_radius}
  height: {camera_height}
  center: [0.0, 0.0, 0.2]
  world_up: [0, 0, 1]"""

    # 4. USD Generation
    usd_path = out_dir / f"{name}.usd"
    build_usd_scene(
        usd_out_path=usd_path,
        cubes_info=cubes_info,
        room_width=room_width,
        room_depth=room_depth,
        room_height=room_height,
        checker_tex_rel_path=tex_filename,
        camera_poses=camera_poses,
        light_pos=light_pos,
        num_frames=num_frames,
        fps=fps,
        vfov_deg=vfov_deg,
        width_px=width_px,
        height_px=height_px,
        tile_size=tile_size,
    )

    # 5. Capture YAML Config (for Isaac Sim omni_capture.py)
    capture_dir = f"{capture_root}/capture_{name}"
    yaml_content = f"""# Capture config for spinning cubes test scene
app:
  headless: true

scene:
  usd_path: "{str(usd_path).replace(os.sep, '/')}"
  subject_prim: "/World/Cubes"
  semantic_roots: ["/World/Cubes"]

{rig_yaml}

capture:
  width: {width_px}
  height: {height_px}
  vfov_deg: {vfov_deg}
  num_frames: {num_frames}
  camera_chunk_size: 4
  rt_subframes: 4
  near: 0.1
  far: 20.0

output:
  capture_dir: "{capture_dir}"
  instance_segmentation: true
  semantic_segmentation: false
  colorize_instance_segmentation: false
  colorize_semantic_segmentation: false
  num_init_points: 50000

lighting:
  enabled: true
  force: false
  add_dome: false
  add_distant: false
"""
    yaml_path = out_dir / f"capture_config_{name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # 6. Manifest JSON
    manifest = {
        "scene_name": name,
        "k_cubes": k,
        "circle_radius_m": circle_radius,
        "cube_size_m": cube_size,
        "room_dimensions_m": [room_width, room_depth, room_height],
        "wall_margin_behind_cameras_m": wall_margin,
        "camera_radius_m": camera_radius,
        "num_cameras": num_cameras,
        "num_frames": num_frames,
        "fps": fps,
        "light_position_m": list(light_pos),
        "cubes": cubes_info,
        "usd_path": str(usd_path),
        "yaml_path": str(yaml_path),
        "capture_dir": capture_dir,
    }
    manifest_path = out_dir / f"{name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[gen_spinning_cubes] Generated '{name}': k={k} cubes, "
        f"room={room_width:.1f}x{room_depth:.1f}m -> {usd_path}"
    )

    return manifest


# -----------------------------------------------------------------------------
# Automated Verification & Selftest
# -----------------------------------------------------------------------------

def run_selftest() -> int:
    """Run full automated verification of the spinning cubes scene generator."""
    import tempfile
    from pxr import Usd, UsdGeom

    print("=== Running gen_spinning_cubes selftest ===")
    tmp_dir = tempfile.mkdtemp(prefix="spinning_cubes_selftest_")

    k = 5
    circle_radius = 1.5
    cube_size = 0.35
    num_cameras = 10
    camera_radius = 3.0
    wall_margin = 2.5
    num_frames = 60

    manifest = generate_spinning_cubes_scene(
        k=k,
        circle_radius=circle_radius,
        cube_size=cube_size,
        num_cameras=num_cameras,
        camera_radius=camera_radius,
        wall_margin=wall_margin,
        num_frames=num_frames,
        out_dir=tmp_dir,
        name="test_cubes",
    )

    checks = {}

    usd_path = manifest["usd_path"]
    yaml_path = manifest["yaml_path"]
    checks["USD file exists"] = os.path.isfile(usd_path)
    checks["YAML config exists"] = os.path.isfile(yaml_path)

    stage = Usd.Stage.Open(usd_path)
    checks["Stage valid"] = stage is not None
    checks["Stage Z-up"] = UsdGeom.GetStageUpAxis(stage) == "Z"
    checks["Stage metersPerUnit=1.0"] = abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-6
    checks["TimeCode 0..59"] = (
        stage.GetStartTimeCode() == 0 and stage.GetEndTimeCode() == num_frames - 1
    )

    cubes = manifest["cubes"]
    checks["Placed exactly k cubes"] = len(cubes) == k

    in_circle = True
    diag_radius = (math.sqrt(2.0) * cube_size) / 2.0
    for c in cubes:
        dist_center = math.hypot(c["x"], c["y"])
        if dist_center + diag_radius > circle_radius + 1e-5:
            in_circle = False
            break
    checks["All cubes strictly within 3m circle"] = in_circle

    no_overlap = True
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            d = math.hypot(cubes[i]["x"] - cubes[j]["x"], cubes[i]["y"] - cubes[j]["y"])
            if d < 2.0 * diag_radius:
                no_overlap = False
                break
    checks["No cubes overlap"] = no_overlap

    room_prim = stage.GetPrimAtPath("/World/Room")
    checks["Room prim exists"] = room_prim and room_prim.IsValid()
    checks["Floor mesh exists"] = stage.GetPrimAtPath("/World/Room/Floor").IsValid()
    checks["4 Walls exist"] = all(
        stage.GetPrimAtPath(f"/World/Room/Wall_{w}").IsValid()
        for w in ["East", "West", "North", "South"]
    )
    expected_room_w = 2.0 * (camera_radius + wall_margin)
    checks["Room width == 11m (2.5m behind cameras)"] = (
        abs(manifest["room_dimensions_m"][0] - expected_room_w) < 1e-6
    )

    checker_mat = stage.GetPrimAtPath("/World/Materials/CheckerMaterial")
    checks["Checker material exists"] = checker_mat.IsValid()

    light = stage.GetPrimAtPath("/World/Lights/CornerLight")
    checks["Corner light exists"] = light.IsValid() and light.GetTypeName() == "SphereLight"
    light_pos = manifest["light_position_m"]
    checks["Light in top corner (z > 3.5m)"] = light_pos[2] >= 3.5 and light_pos[0] > 4.0

    cams = [
        stage.GetPrimAtPath(f"/World/CaptureCameras/cam{i:02d}")
        for i in range(1, num_cameras + 1)
    ]
    checks["10 Cameras authored in USD"] = len(cams) == 10 and all(
        c.IsValid() and c.GetTypeName() == "Camera" for c in cams
    )

    all_passed = True
    for desc, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        if not passed:
            all_passed = False

    print(f"=== Selftest Result: {'ALL PASS' if all_passed else 'SOME CHECKS FAILED'} ===")
    return 0 if all_passed else 1


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5],
        help="Number of spinning cubes (e.g. --k 5 or --k 1 2 3 4 5 6 7 8)",
    )
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=1.5,
        help="Radius of placement circle in meters (default: 1.5, i.e. 3.0 m circle diameter)",
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=0.35,
        help="Cube side length in meters (default: 0.35)",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=16,
        help="Number of cameras in the rig (default: 16)",
    )
    parser.add_argument(
        "--camera-radius",
        type=float,
        default=3.0,
        help="Radius of camera circle in meters (default: 3.0)",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.2,
        help="Height of camera circle in meters (default: 1.2)",
    )
    parser.add_argument(
        "--camera-layout",
        type=str,
        choices=["stacked_arcs", "stacked_ring", "circle", "ring"],
        default="stacked_arcs",
        help="Camera rig layout: stacked_arcs (two stacked arcs on one side) or ring/circle (default: stacked_arcs)",
    )
    parser.add_argument(
        "--arc-deg",
        type=float,
        default=80.0,
        help="Horizontal angle span in degrees for stacked_arcs layout (default: 80.0)",
    )
    parser.add_argument(
        "--center-deg",
        type=float,
        default=0.0,
        help="Center azimuth angle in degrees for stacked_arcs layout (default: 0.0)",
    )
    parser.add_argument(
        "--height-lower",
        type=float,
        default=0.8,
        help="Height of lower camera arc in meters (default: 0.8)",
    )
    parser.add_argument(
        "--height-upper",
        type=float,
        default=1.3,
        help="Height of upper camera arc in meters (default: 1.3, 0.5m vertical separation)",
    )
    parser.add_argument(
        "--motion-mode",
        type=str,
        choices=["rotation", "translation", "both"],
        default="both",
        help="Cube motion mode: rotation (pure spin), translation (linear oscillation), or both (default: both)",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=1.5,
        help="Checkerboard tile square size in meters (default: 1.5)",
    )
    parser.add_argument(
        "--wall-margin",
        type=float,
        default=2.5,
        help="Distance from camera ring to room walls in meters (default: 2.5, i.e. 2.5m behind cameras)",
    )
    parser.add_argument(
        "--room-height",
        type=float,
        default=4.5,
        help="Height of the room in meters (default: 4.5)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=120,
        help="Number of animation frames/timecodes (default: 120)",
    )
    parser.add_argument(
        "--fps", type=float, default=60.0, help="Frames per second (default: 60.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for placement and rotation (default: 42)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "omniverse-pipeline",
            "data",
            "scenes",
            "cubes",
        ),
        help="Output directory for USD, textures, and configs",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="spinning_cubes",
        help="Base prefix for scene names (default: spinning_cubes)",
    )
    parser.add_argument(
        "--capture-root",
        type=str,
        default="Q:/Omniverse/renders",
        help="Root path for capture output in the YAML config",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run self-test validation in a temporary directory",
    )

    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for k_val in args.k:
        scene_name = f"{args.name_prefix}_k{k_val}"
        m = generate_spinning_cubes_scene(
            k=k_val,
            circle_radius=args.circle_radius,
            cube_size=args.cube_size,
            num_cameras=args.num_cameras,
            camera_radius=args.camera_radius,
            camera_height=args.camera_height,
            camera_layout=args.camera_layout,
            arc_deg=args.arc_deg,
            center_deg=args.center_deg,
            height_lower=args.height_lower,
            height_upper=args.height_upper,
            motion_mode=args.motion_mode,
            wall_margin=args.wall_margin,
            room_height=args.room_height,
            tile_size=args.tile_size,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed + k_val * 100,
            out_dir=out_dir,
            name=scene_name,
            capture_root=args.capture_root,
        )
        manifests.append(m)

    if len(manifests) > 1:
        batch_manifest_path = out_dir / "cubes_grid_manifest.json"
        with open(batch_manifest_path, "w") as f:
            json.dump({"scenes": manifests}, f, indent=2)
        print(f"[gen_spinning_cubes] Wrote batch manifest with {len(manifests)} scenes -> {batch_manifest_path}")


if __name__ == "__main__":
    main()
