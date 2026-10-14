#!/usr/bin/env python3
"""
gen_scenes.py — generate parametric test scenes for 4DGS motion segmentation and amplification.

Supports:
1. Legacy amplitude grid: (base amplitude A [mm] × multiplier M) with independent random
   integer-cycle sinusoidal SE(3) motion per part.
2. Capture-limitation series (Series A / R / T / F):
   - Series A: common-mode in-sync vibration of non-target parts at amplitude A,
     with deviant target part at B (A0: B=0; A1..A4: B=2A/5A/10A/20A; A5: B=A, non-target 0).
     Relative interface motion at deviant part is |B - A|; common-mode elsewhere is 0.
   - Series R: capture ladder (resolution 1600x900 to 3000x2000, distance radius_scale 2.5 to 1.25,
     frames 60 to 120).
   - Series T: material ablation (T0: pastel displayColors; T1: realistic uniform brushed steel PBR;
     T2: T1 + procedural surface texture: lathe machining marks, rust, labels, sub-pixel texture).
   - Series F: honest combined cells (F1: A3 motion × T1 metallic × R4 capture; F2: with T2 texture).

Runs with plain python + usd-core + numpy + PIL (pxr imported lazily inside functions).

Usage:
    # Generate from YAML preset:
    python scene-gen/gen_scenes.py --preset scene-gen/presets/pump01_sA0_common_target_static.yaml

    # Legacy grid generation:
    python scene-gen/gen_scenes.py \
        --segmented-usd Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd \
        --env Q:/Omniverse/assets/physAI_start/Factory/Factory.usd \
        --out-dir omniverse-pipeline/data/scenes/grid \
        --base-amp-mm 8 20 40 --multipliers 2 4 8 --seed 0

    python scene-gen/gen_scenes.py --selftest
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys
from typing import Any

import numpy as np

# Reference scripts live in the editable workspace package omniverse_pipeline.
from omniverse_pipeline import add_motion, compose_scene

GROUP = "CONJUNTO_BOMBAS"          # part group name inside the segmented/animated USD
SUBJECT_NAME = "pump"              # renamed subject root in the composed scene
# Artist-intended pump placement in the physAI_start factory (SceneAssembly.usd),
# see .claude_notes/NOTES_warehouse_scene.md.
TRANSLATE = (-130.29, -437.17, 0.35)
SCALE = 0.2
# Dome top ring at the template's 75° puts cameras at z≈10.4 m, inside the factory
# roof structure (frames come out pure gray). 50° keeps the top ring at z≈8.7 m,
# just above the verified-good middle ring (45°, z≈8.2 m).
MAX_ELEV_DEG = 50.0
TEMPLATE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "omniverse-pipeline", "omniverse_pipeline",
    "capture_config_pump_warehouse.yaml")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _fmt(v: float) -> str:
    """Compact number for filenames: 1, 4, 16, 0.5 ..."""
    return ("%g" % v)


# ---------------------------------------------------------------- motion ----

def author_motion(
    inp: str,
    out: str,
    base_amp_mm: float,
    multiplier: float,
    amplified_name: str,
    num_frames: int,
    fps: float,
    seed: int,
    freq_hz: float = 10.0,
    motion_mode: str = "random",
    common_amp_mm: float | None = None,
    target_amp_mm: float | None = None,
    direction: tuple[float, float, float] | None = None,
    phase_rad: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Write <out> with per-part periodic motion; `amplified_name` scaled by `multiplier`
    or explicit target_amp_mm.

    Supports:
    - motion_mode='random': legacy random independent draws per part.
    - motion_mode='common_mode': non-target parts translate in sync at common_amp_mm (amplitude A),
      while `amplified_name` translates with target_amp_mm (amplitude B) along the identical
      frequency, phase, and direction. Relative interface motion at the target part is |B - A|,
      and identically 0 elsewhere.

    Returns (movable_names, peak_surface_mm dict) in SUBJECT-internal millimetres."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(inp)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    units_per_mm = 1e-3 / mpu
    stage.GetRootLayer().Export(out)
    stage = Usd.Stage.Open(out)

    parts = add_motion.collect_parts(stage, GROUP, UsdGeom, Usd)
    movable = [(p, c, r) for (p, c, r) in parts if p.GetName() != "frame_base"]
    movable_names = [p.GetName() for p, _, _ in movable]
    if amplified_name not in movable_names:
        raise SystemExit(f"[gen] target part {amplified_name} not among movable parts: {movable_names[:5]}...")

    # Integer cycles over the clip so the motion loops seamlessly.
    cycles = int(round(freq_hz * (num_frames - 1) / fps))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    peak_mm: dict[str, float] = {}

    if motion_mode == "common_mode":
        # All non-target parts translate in unison (A); target part translates with B.
        A_mm = base_amp_mm if common_amp_mm is None else float(common_amp_mm)
        B_mm = (base_amp_mm * multiplier) if target_amp_mm is None else float(target_amp_mm)
        dir_vec = _unit(np.asarray(direction if direction is not None else [0.0, 0.0, 1.0], dtype=float))
        phase = float(phase_rad if phase_rad is not None else 0.0)

        for prim, _, _ in movable:
            name = prim.GetName()
            amp_val_mm = B_mm if name == amplified_name else A_mm
            amp_stage = amp_val_mm * units_per_mm

            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            for fi in range(num_frames):
                u = fi / max(num_frames - 1, 1)
                disp_stage = amp_stage * np.sin(2.0 * np.pi * cycles * u + phase) * dir_vec
                M = np.eye(4)
                M[:3, 3] = disp_stage
                op.Set(Gf.Matrix4d(*M.T.flatten()), Usd.TimeCode(fi))
            peak_mm[name] = float(amp_val_mm)
    else:
        # Legacy random per-part motion
        cfg = dict(
            trans_amp=[0.5 * base_amp_mm * units_per_mm, 1.5 * base_amp_mm * units_per_mm],
            rot_surface=[0.25 * base_amp_mm * units_per_mm, 0.75 * base_amp_mm * units_per_mm],
            rot_deg_max=3.0,
            freq=[cycles, cycles],
        )
        rng = np.random.default_rng(seed)
        for prim, c, r in movable:
            name = prim.GetName()
            mp = add_motion._rand_motion(cfg, rng, r)
            if name == amplified_name:
                mult_to_use = (target_amp_mm / base_amp_mm) if target_amp_mm is not None else multiplier
                mp["trans_amp"] *= mult_to_use
                mp["rot_amp"] *= mult_to_use
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            peak = 0.0
            for fi in range(num_frames):
                u = fi / max(num_frames - 1, 1)
                M = add_motion.part_matrix(np.asarray(c), mp, u)
                op.Set(Gf.Matrix4d(*M.T.flatten()), Usd.TimeCode(fi))
                peak = max(peak, add_motion._surface_disp(mp, r, u))
            peak_mm[name] = peak / units_per_mm

    stage.GetRootLayer().Save()
    return movable_names, peak_mm


# ------------------------------------------------------------- materials ----

def create_procedural_texture(out_path: str, width: int = 1024, height: int = 1024) -> str:
    """Generate high-frequency brushed alloy + lathe machining marks + rust/decals PNG
    for sub-pixel texture evaluation (Diamond et al. 2017)."""
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rng = np.random.default_rng(42)

    # Base neutral alloy
    base = np.full((height, width, 3), 190, dtype=np.uint8)
    # Horizontal brushed striations
    striations = rng.normal(0, 10, size=(height, 1, 1)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + striations, 0, 255).astype(np.uint8)
    # High frequency surface grain
    grain = rng.normal(0, 5, size=(height, width, 3)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Machining concentric lathe arcs
    for r in range(40, width // 2, 35):
        draw.arc([width // 2 - r, height // 2 - r, width // 2 + r, height // 2 + r],
                 start=0, end=360, fill=(160, 162, 168), width=2)

    # Rust & patina spots
    for _ in range(30):
        rx = int(rng.integers(50, width - 50))
        ry = int(rng.integers(50, height - 50))
        rad = int(rng.integers(8, 26))
        draw.ellipse([rx - rad, ry - rad, rx + rad, ry + rad], fill=(138, 92, 52))

    # Machine label badges & serial markings
    draw.rectangle([80, 80, 420, 210], fill=(35, 40, 45), outline=(220, 180, 30), width=4)
    draw.text((100, 95), "HIGH PRESSURE INDUSTRIAL PUMP", fill=(255, 255, 255))
    draw.text((100, 125), "SN: 4DGS-PUMP-2026-T2", fill=(210, 215, 220))
    draw.text((100, 155), "RATED: 250 BAR / 1750 RPM", fill=(230, 190, 40))
    draw.text((100, 180), "SUB-PIXEL TEXTURE FIDELITY TEST", fill=(160, 165, 170))

    img.save(out_path)
    return out_path


def apply_materials(
    usd_path: str,
    preset: str = "saturated_metallic",
    out_dir: str | None = None,
) -> int:
    """Apply material preset to every part under /World/<group>:
    - 'pastel': retains original pastel displayColor, removes metallic UsdShade materials
    - 'metallic': uniform brushed steel PBR (metallic=1.0, roughness=0.32), uniform displayColor
    - 'metallic_texture': metallic PBR with procedural texture connected + cylindrical UV primvars:st
    - 'saturated_metallic': legacy mode (golden-ratio HSV palette per part)
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.Open(usd_path)
    grp = stage.GetPrimAtPath(f"/World/{GROUP}")
    if not grp:
        return 0
    parts = [p for p in grp.GetChildren() if p.GetTypeName() == "Xform"]
    n = len(parts)

    if preset == "pastel":
        # Keep original displayColor, remove UsdShade material bindings
        for part in parts:
            for child in part.GetChildren():
                if child.GetTypeName() == "Mesh":
                    UsdShade.MaterialBindingAPI(child).UnbindAllBindings()
        mats_prim = stage.GetPrimAtPath(f"/World/{GROUP}/Materials")
        if mats_prim:
            stage.RemovePrim(mats_prim.GetPath())
        stage.GetRootLayer().Save()
        return n

    mats_scope = stage.DefinePrim(f"/World/{GROUP}/Materials", "Scope")

    if preset == "saturated_metallic":
        # Legacy behavior: saturated golden-ratio HSV palette
        for i, part in enumerate(sorted(parts, key=lambda p: p.GetName())):
            r_, g_, b_ = colorsys.hsv_to_rgb((0.62 + i * 0.61803) % 1.0, 0.85, 0.8)
            mat_path = f"{mats_scope.GetPath()}/{part.GetName()}_mat"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r_, g_, b_))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for child in part.GetChildren():
                if child.GetTypeName() == "Mesh":
                    UsdShade.MaterialBindingAPI(child).Bind(mat)
                    UsdGeom.Mesh(child).CreateDisplayColorAttr(
                        Vt.Vec3fArray([Gf.Vec3f(r_, g_, b_)]))
        stage.GetRootLayer().Save()
        return n

    if preset == "metallic":
        # Uniform brushed steel PBR across all parts (no per-part color coding)
        mat_path = f"{mats_scope.GetPath()}/BrushedSteel_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.73, 0.75))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.32)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        base_mat_path = f"{mats_scope.GetPath()}/CastIronBase_mat"
        base_mat = UsdShade.Material.Define(stage, base_mat_path)
        base_shader = UsdShade.Shader.Define(stage, f"{base_mat_path}/PBRShader")
        base_shader.CreateIdAttr("UsdPreviewSurface")
        base_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.35, 0.38))
        base_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
        base_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        base_mat.CreateSurfaceOutput().ConnectToSource(base_shader.ConnectableAPI(), "surface")

        for part in parts:
            is_base = (part.GetName() == "frame_base")
            target_mat = base_mat if is_base else mat
            color_vec = Gf.Vec3f(0.35, 0.35, 0.38) if is_base else Gf.Vec3f(0.72, 0.73, 0.75)
            for child in part.GetChildren():
                if child.GetTypeName() == "Mesh":
                    UsdShade.MaterialBindingAPI(child).Bind(target_mat)
                    UsdGeom.Mesh(child).CreateDisplayColorAttr(Vt.Vec3fArray([color_vec]))
        stage.GetRootLayer().Save()
        return n

    if preset == "metallic_texture":
        # T2: Uniform metallic + procedural texture + cylindrical UV primvars:st
        tex_dir = out_dir or os.path.dirname(usd_path)
        tex_path = os.path.join(tex_dir, "machined_metal_texture.png")
        if not os.path.isfile(tex_path):
            create_procedural_texture(tex_path)

        mat_path = f"{mats_scope.GetPath()}/TexturedSteel_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.32)

        uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex_shader = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTexture")
        tex_shader.CreateIdAttr("UsdUVTexture")
        rel_tex = os.path.relpath(tex_path, os.path.dirname(usd_path)).replace("\\", "/")
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel_tex))
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        for part in parts:
            for child in part.GetChildren():
                if child.GetTypeName() == "Mesh":
                    m = UsdGeom.Mesh(child)
                    pts = np.array(m.GetPointsAttr().Get() or [], dtype=np.float32)
                    if len(pts) > 0:
                        cen = pts.mean(axis=0)
                        rel = pts - cen
                        u = (np.arctan2(rel[:, 1], rel[:, 0]) / (2.0 * np.pi) + 0.5) % 1.0
                        z_span = max(rel[:, 2].max() - rel[:, 2].min(), 1e-4)
                        v = ((rel[:, 2] - rel[:, 2].min()) / z_span) % 1.0
                        uvs = [Gf.Vec2f(float(u[k]), float(v[k])) for k in range(len(pts))]
                        st_pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
                            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                        st_pv.Set(Vt.Vec2fArray(uvs))
                        # Subtle vertex noise on displayColor fallback
                        noise = np.sin(pts[:, 0] * 0.05) * np.cos(pts[:, 1] * 0.05) * 0.03
                        cols = [Gf.Vec3f(float(np.clip(0.72 + n_val, 0, 1)),
                                         float(np.clip(0.73 + n_val, 0, 1)),
                                         float(np.clip(0.75 + n_val, 0, 1))) for n_val in noise]
                        m.CreateDisplayColorAttr(Vt.Vec3fArray(cols))
                    UsdShade.MaterialBindingAPI(child).Bind(mat)
        stage.GetRootLayer().Save()
        return n

    raise ValueError(f"Unknown material preset: {preset}")


def apply_metallic_materials(usd_path: str) -> int:
    """Backward-compatible wrapper for apply_materials(usd_path, preset='saturated_metallic')."""
    return apply_materials(usd_path, preset="saturated_metallic")


# ------------------------------------------------------------ capture yaml ----

def write_capture_config(
    template_path: str,
    out_path: str,
    scene_usd: str,
    capture_dir: str,
    num_frames: int | None = None,
    max_elev_deg: float | None = None,
    width: int | None = None,
    height: int | None = None,
    radius_scale: float | None = None,
    rt_subframes: int | None = None,
) -> None:
    """Clone the warehouse capture config, swapping usd_path / capture_dir and any
    rig/capture overrides via text substitution so all other settings survive verbatim."""
    with open(template_path) as f:
        text = f.read()
    text = re.sub(r'usd_path: ".*?"', f'usd_path: "{scene_usd}"', text, count=1)
    text = re.sub(r'capture_dir: ".*?"', f'capture_dir: "{capture_dir}"', text, count=1)
    if num_frames is not None:
        text = re.sub(r'num_frames: \d+', f'num_frames: {num_frames}', text, count=1)
    if max_elev_deg is not None:
        text = re.sub(r'max_elev_deg: [\d.]+', f'max_elev_deg: {max_elev_deg:g}', text, count=1)
    if width is not None:
        text = re.sub(r'width: \d+', f'width: {width}', text, count=1)
    if height is not None:
        text = re.sub(r'height: \d+', f'height: {height}', text, count=1)
    if radius_scale is not None:
        text = re.sub(r'radius_scale: [\d.]+', f'radius_scale: {radius_scale:g}', text, count=1)
    if rt_subframes is not None:
        text = re.sub(r'rt_subframes: \d+', f'rt_subframes: {rt_subframes}', text, count=1)
    with open(out_path, "w") as f:
        f.write(text)


# ----------------------------------------------------------------- driver ----

def cell_name(amp: float, mult: float) -> str:
    return f"pump_A{_fmt(amp)}mm_M{_fmt(mult)}"


def pick_amplified_part(segmented_usd: str, seed: int) -> str:
    """Seeded pick of one movable part; identical for every cell."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(segmented_usd)
    parts = add_motion.collect_parts(stage, GROUP, UsdGeom, Usd)
    names = sorted(p.GetName() for p, _, _ in parts if p.GetName() != "frame_base")
    rng = np.random.default_rng(seed + 99991)
    return names[int(rng.integers(0, len(names)))]


def generate_cell(
    segmented_usd: str,
    env_usd: str,
    out_dir: str,
    amp: float,
    mult: float,
    amplified: str,
    num_frames: int,
    fps: float,
    seed: int,
    capture_root: str,
    freq_hz: float,
    custom_name: str | None = None,
    motion_mode: str = "random",
    common_amp_mm: float | None = None,
    target_amp_mm: float | None = None,
    direction: tuple[float, float, float] | None = None,
    phase_rad: float | None = None,
    material_preset: str = "saturated_metallic",
    width: int | None = None,
    height: int | None = None,
    radius_scale: float | None = None,
    rt_subframes: int | None = None,
    max_elev_deg: float = MAX_ELEV_DEG,
) -> dict[str, Any]:
    """Generate a single scene variant, authoring motion, materials, composition,
    capture config, and motion.json."""
    name = custom_name or cell_name(amp, mult)
    animated = os.path.join(out_dir, f"{name}_animated.usd")
    scene = os.path.join(out_dir, f"{name}_scene.usd")
    yaml_path = os.path.join(out_dir, f"{name}.yaml")
    capture_dir = f"{capture_root}/capture_{name}"

    movable, peak_mm = author_motion(
        inp=segmented_usd,
        out=animated,
        base_amp_mm=amp,
        multiplier=mult,
        amplified_name=amplified,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        freq_hz=freq_hz,
        motion_mode=motion_mode,
        common_amp_mm=common_amp_mm,
        target_amp_mm=target_amp_mm,
        direction=direction,
        phase_rad=phase_rad,
    )
    n_mats = apply_materials(animated, preset=material_preset, out_dir=out_dir)
    compose_scene.compose(
        in_usd=animated, out_usd=scene, group=SUBJECT_NAME,
        src_prim=f"/World/{GROUP}", env_usd=env_usd, env_root="/World/Factory",
        translate=TRANSLATE, scale=SCALE)
    write_capture_config(
        TEMPLATE_CONFIG, yaml_path,
        os.path.abspath(scene).replace(os.sep, "/"), capture_dir,
        num_frames=num_frames, max_elev_deg=max_elev_deg,
        width=width, height=height, radius_scale=radius_scale, rt_subframes=rt_subframes)

    peak_world_mm = {k: v * SCALE for k, v in peak_mm.items()}
    c_amp_world = (amp if common_amp_mm is None else common_amp_mm) * SCALE
    t_amp_world = (target_amp_mm if target_amp_mm is not None else (amp * mult)) * SCALE
    rel_amp_world = abs(t_amp_world - c_amp_world)

    groups = {
        "num_frames": num_frames,
        "fps": fps,
        "freq_hz": freq_hz,
        "base_amp_mm": amp,
        "multiplier": mult,
        "amplified_part": amplified,
        "target_part": amplified,
        "motion_mode": motion_mode,
        "common_mode_amp_world_mm": c_amp_world,
        "target_amp_world_mm": t_amp_world,
        "relative_amp_world_mm": rel_amp_world,
        "material_preset": material_preset,
        "seed": seed,
        "compose_scale": SCALE,
        "peak_surface_mm": peak_world_mm,
    }
    with open(os.path.join(out_dir, f"{name}_motion.json"), "w") as f:
        json.dump(groups, f, indent=2)

    amplified_peak_world = peak_world_mm[amplified]
    info = {
        "cell": name,
        "base_amp_mm": amp,
        "multiplier": mult,
        "target_part": amplified,
        "motion_mode": motion_mode,
        "material_preset": material_preset,
        "animated": animated,
        "scene": scene,
        "config": yaml_path,
        "capture_dir": capture_dir,
        "materials": n_mats,
        "amplified_peak_mm": amplified_peak_world,
        "relative_peak_mm": rel_amp_world,
    }
    print(f"[cell {name}] parts={len(movable)} mats={n_mats} "
          f"target={amplified} peak={amplified_peak_world:.2f}mm rel={rel_amp_world:.2f}mm (scale={SCALE})")
    return info


def generate_from_preset(
    preset_path: str,
    out_dir: str | None = None,
    capture_root: str = "Q:/Omniverse/renders",
) -> dict[str, Any]:
    """Load a YAML preset and generate the corresponding scene variant."""
    import yaml
    with open(preset_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    name = spec.get("name", os.path.splitext(os.path.basename(preset_path))[0])
    target_out_dir = os.path.abspath(out_dir or spec.get("out_dir") or os.path.join(
        os.path.dirname(__file__), "..", "omniverse-pipeline", "data", "scenes", "grid"))
    os.makedirs(target_out_dir, exist_ok=True)

    segmented_usd = spec.get("segmented_usd", "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd")
    env_usd = spec.get("env", "Q:/Omniverse/assets/physAI_start/Factory/Factory.usd")

    motion_spec = spec.get("motion", {})
    motion_mode = motion_spec.get("mode", "random")
    base_amp = float(motion_spec.get("base_amp_mm", 5.0))
    mult = float(motion_spec.get("multiplier", 1.0))
    common_amp = motion_spec.get("common_amp_mm")
    target_amp = motion_spec.get("target_amp_mm")
    target_part = motion_spec.get("target_part", "part_019")
    freq_hz = float(motion_spec.get("freq_hz", 10.0))
    seed = int(motion_spec.get("seed", 0))

    mat_spec = spec.get("material", {})
    mat_preset = mat_spec.get("preset", "pastel")

    cap_spec = spec.get("capture", {})
    width = cap_spec.get("width", 1600)
    height = cap_spec.get("height", 900)
    radius_scale = cap_spec.get("radius_scale", 2.5)
    num_frames = cap_spec.get("num_frames", 60)
    fps = cap_spec.get("fps", 60.0)
    rt_subframes = cap_spec.get("rt_subframes", 16)
    max_elev_deg = cap_spec.get("max_elev_deg", MAX_ELEV_DEG)

    return generate_cell(
        segmented_usd=segmented_usd,
        env_usd=env_usd,
        out_dir=target_out_dir,
        amp=base_amp,
        mult=mult,
        amplified=target_part,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        capture_root=capture_root,
        freq_hz=freq_hz,
        custom_name=name,
        motion_mode=motion_mode,
        common_amp_mm=common_amp,
        target_amp_mm=target_amp,
        material_preset=mat_preset,
        width=width,
        height=height,
        radius_scale=radius_scale,
        rt_subframes=rt_subframes,
        max_elev_deg=max_elev_deg,
    )


# ----------------------------------------------------------------- CLI ----

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None,
                    help="path to a scene preset YAML file (e.g. scene-gen/presets/pump01_sA0_common_target_static.yaml)")
    ap.add_argument("--segmented-usd",
                    default="Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd")
    ap.add_argument("--env", default="Q:/Omniverse/assets/physAI_start/Factory/Factory.usd")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "omniverse-pipeline", "data", "scenes", "grid"))
    ap.add_argument("--base-amp-mm", type=float, nargs="+", default=[8, 20, 40])
    ap.add_argument("--multipliers", type=float, nargs="+", default=[2, 4, 8])
    ap.add_argument("--only-cell", default=None,
                    help="e.g. 16x8 — generate just this (amp x multiplier) cell")
    ap.add_argument("--num-frames", type=int, default=240)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--freq-hz", type=float, default=10.0,
                    help="target motion frequency in Hz; rounded to integer cycles "
                         "over the clip so the motion loops seamlessly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture-root", default="Q:/Omniverse/renders")
    ap.add_argument("--target-part", default="part_019",
                    help="named target/deviant part (default: part_019, Cat A neighbor)")
    ap.add_argument("--motion-mode", choices=["random", "common_mode"], default="random",
                    help="random (legacy grid) or common_mode (Series A in-sync motion)")
    ap.add_argument("--common-amp-mm", type=float, default=None,
                    help="amplitude A for in-sync parts in common_mode")
    ap.add_argument("--target-amp-mm", type=float, default=None,
                    help="amplitude B for target part in common_mode")
    ap.add_argument("--material-preset", choices=["pastel", "metallic", "metallic_texture", "saturated_metallic"],
                    default="saturated_metallic", help="material look for meshes")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--radius-scale", type=float, default=None)
    ap.add_argument("--rt-subframes", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    if args.preset:
        info = generate_from_preset(args.preset, out_dir=args.out_dir, capture_root=args.capture_root)
        print(f"[gen] generated preset variant: {info['cell']}")
        return

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cells = [(a, m) for a in args.base_amp_mm for m in args.multipliers]
    if args.only_cell:
        a_s, m_s = re.split(r"[x,]", args.only_cell)
        cells = [(float(a_s), float(m_s))]

    amplified = args.target_part or pick_amplified_part(args.segmented_usd, args.seed)
    cycles = int(round(args.freq_hz * (args.num_frames - 1) / args.fps))
    print(f"[gen] target part (all cells): {amplified}; "
          f"freq {args.freq_hz} Hz -> {cycles} cycles/clip "
          f"({cycles * args.fps / (args.num_frames - 1):.2f} Hz effective)")

    manifest = {"seed": args.seed, "amplified_part": amplified,
                "freq_hz": args.freq_hz, "cycles": cycles, "cells": []}
    for amp, mult in cells:
        manifest["cells"].append(generate_cell(
            segmented_usd=args.segmented_usd,
            env_usd=args.env,
            out_dir=out_dir,
            amp=amp,
            mult=mult,
            amplified=amplified,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
            capture_root=args.capture_root,
            freq_hz=args.freq_hz,
            motion_mode=args.motion_mode,
            common_amp_mm=args.common_amp_mm,
            target_amp_mm=args.target_amp_mm,
            material_preset=args.material_preset,
            width=args.width,
            height=args.height,
            radius_scale=args.radius_scale,
            rt_subframes=args.rt_subframes,
        ))
    with open(os.path.join(out_dir, "grid_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gen] {len(manifest['cells'])} cell(s) -> {out_dir} (+ grid_manifest.json)")


# --------------------------------------------------------------- selftest ----

def _selftest() -> int:
    """Check (1) legacy amplified peak ~4x and material binding, (2) common-mode in-sync motion
    (zero relative motion between non-target parts, |B-A| relative motion at deviant part),
    (3) material presets (pastel, metallic, metallic_texture with UV primvars), (4) capture overrides."""
    import tempfile
    from pxr import Gf, Usd, UsdGeom, UsdShade, Vt

    tmp = tempfile.mkdtemp(prefix="gen_scenes_selftest_")
    src = os.path.join(tmp, "seg.usd")
    s = Usd.Stage.CreateNew(src)
    UsdGeom.SetStageUpAxis(s, "Z")
    UsdGeom.SetStageMetersPerUnit(s, 0.01)
    s.DefinePrim("/World", "Xform")
    s.DefinePrim(f"/World/{GROUP}", "Xform")
    for i, name in enumerate(["frame_base", "part_001", "part_002", "part_003"]):
        body = UsdGeom.Xform.Define(s, f"/World/{GROUP}/{name}")
        mesh = UsdGeom.Mesh.Define(s, f"/World/{GROUP}/{name}/mesh")
        o = 50.0 * i
        mesh.CreatePointsAttr(Vt.Vec3fArray(
            [Gf.Vec3f(o, 0, 0), Gf.Vec3f(o + 10, 0, 0), Gf.Vec3f(o, 10, 0)]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    s.GetRootLayer().Save()

    checks: dict[str, bool] = {}

    # 1. Legacy random checks
    amp, seed = 4.0, 0
    out1 = os.path.join(tmp, "m1.usd")
    out4 = os.path.join(tmp, "m4.usd")
    _, peak1 = author_motion(src, out1, amp, 1.0, "part_002", 60, 24.0, seed)
    _, peak4 = author_motion(src, out4, amp, 4.0, "part_002", 60, 24.0, seed)
    n_mat = apply_metallic_materials(out4)

    checks["amplified peak ~4x"] = abs(peak4["part_002"] / peak1["part_002"] - 4.0) < 0.05
    checks["other parts unchanged"] = all(abs(peak1[n] - peak4[n]) < 1e-9 for n in ("part_001", "part_003"))
    checks["motion is mm-scale"] = 1.0 < peak1["part_001"] < 20.0
    checks["4 materials authored"] = (n_mat == 4)

    st = Usd.Stage.Open(out4)
    mesh = st.GetPrimAtPath(f"/World/{GROUP}/part_002/mesh")
    bound = UsdShade.MaterialBindingAPI(mesh).GetDirectBinding().GetMaterial()
    checks["material bound to mesh"] = bool(bound) and bound.GetPrim().GetName() == "part_002_mat"
    surf = UsdShade.Shader(st.GetPrimAtPath(f"/World/{GROUP}/Materials/part_002_mat/PBRShader"))
    checks["metallic=1"] = abs(surf.GetInput("metallic").Get() - 1.0) < 1e-6

    # 2. Common-mode motion checks (Series A)
    # A0: A=5mm, target static (B=0)
    out_a0 = os.path.join(tmp, "a0.usd")
    author_motion(src, out_a0, base_amp_mm=5.0, multiplier=0.0, amplified_name="part_002",
                  num_frames=60, fps=60.0, seed=0, motion_mode="common_mode",
                  common_amp_mm=5.0, target_amp_mm=0.0, direction=(0.0, 0.0, 1.0))
    st_a0 = Usd.Stage.Open(out_a0)
    op1_a0 = UsdGeom.Xformable(st_a0.GetPrimAtPath(f"/World/{GROUP}/part_001")).GetOrderedXformOps()[0]
    op3_a0 = UsdGeom.Xformable(st_a0.GetPrimAtPath(f"/World/{GROUP}/part_003")).GetOrderedXformOps()[0]
    op2_a0 = UsdGeom.Xformable(st_a0.GetPrimAtPath(f"/World/{GROUP}/part_002")).GetOrderedXformOps()[0]

    # Non-target parts must have identical displacement at every frame (zero relative motion)
    zero_rel_non_target = True
    target_is_static = True
    rel_motion_is_A = True
    for fi in range(60):
        m1 = np.array(op1_a0.Get(fi))
        m3 = np.array(op3_a0.Get(fi))
        m2 = np.array(op2_a0.Get(fi))
        if not np.allclose(m1, m3, atol=1e-6):
            zero_rel_non_target = False
        if not np.allclose(m2, np.eye(4), atol=1e-6):
            target_is_static = False
    checks["common_mode: non-target zero relative motion"] = zero_rel_non_target
    checks["common_mode A0: target is static"] = target_is_static

    # A3: A=5mm, B=50mm (10x). Peak relative motion must be |B - A| = 45mm
    out_a3 = os.path.join(tmp, "a3.usd")
    _, peak_a3 = author_motion(src, out_a3, base_amp_mm=5.0, multiplier=10.0, amplified_name="part_002",
                               num_frames=60, fps=60.0, seed=0, motion_mode="common_mode",
                               common_amp_mm=5.0, target_amp_mm=50.0, direction=(0.0, 0.0, 1.0))
    st_a3 = Usd.Stage.Open(out_a3)
    op1_a3 = UsdGeom.Xformable(st_a3.GetPrimAtPath(f"/World/{GROUP}/part_001")).GetOrderedXformOps()[0]
    op2_a3 = UsdGeom.Xformable(st_a3.GetPrimAtPath(f"/World/{GROUP}/part_002")).GetOrderedXformOps()[0]
    max_rel_a3 = 0.0
    mpu = UsdGeom.GetStageMetersPerUnit(st_a3) or 1.0
    units_per_mm = 1e-3 / mpu
    for fi in range(60):
        t1 = np.array(op1_a3.Get(fi))[3, :3]  # row vector USD convention
        t2 = np.array(op2_a3.Get(fi))[3, :3]
        max_rel_a3 = max(max_rel_a3, float(np.linalg.norm(t2 - t1)))
    checks["common_mode A3: relative motion is |B-A|"] = abs(max_rel_a3 / units_per_mm - 45.0) < 0.05

    # 3. Material Presets (pastel, metallic, metallic_texture)
    out_pastel = os.path.join(tmp, "mat_pastel.usd")
    s_p = Usd.Stage.Open(out4)
    s_p.GetRootLayer().Export(out_pastel)
    apply_materials(out_pastel, preset="pastel")
    st_pastel = Usd.Stage.Open(out_pastel)
    m_p = st_pastel.GetPrimAtPath(f"/World/{GROUP}/part_002/mesh")
    checks["material preset pastel: no UsdShade material bound"] = not bool(
        UsdShade.MaterialBindingAPI(m_p).GetDirectBinding().GetMaterial())

    out_met = os.path.join(tmp, "mat_met.usd")
    s_p.GetRootLayer().Export(out_met)
    apply_materials(out_met, preset="metallic")
    st_met = Usd.Stage.Open(out_met)
    m_met = st_met.GetPrimAtPath(f"/World/{GROUP}/part_002/mesh")
    bound_met = UsdShade.MaterialBindingAPI(m_met).GetDirectBinding().GetMaterial()
    checks["material preset metallic: bound"] = bool(bound_met) and bound_met.GetPrim().GetName() == "BrushedSteel_mat"

    out_tex = os.path.join(tmp, "mat_tex.usd")
    s_p.GetRootLayer().Export(out_tex)
    apply_materials(out_tex, preset="metallic_texture", out_dir=tmp)
    st_tex = Usd.Stage.Open(out_tex)
    m_tex = st_tex.GetPrimAtPath(f"/World/{GROUP}/part_002/mesh")
    st_pv = UsdGeom.PrimvarsAPI(m_tex).GetPrimvar("st")
    checks["material preset metallic_texture: primvars:st exists"] = bool(st_pv) and st_pv.Get() is not None

    for name_chk, ok in checks.items():
        print(f"[selftest] {name_chk}: {'ok' if ok else 'FAIL'}")
    all_ok = all(checks.values())
    print("SELFTEST:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    main()
