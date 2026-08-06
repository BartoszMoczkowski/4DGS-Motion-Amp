#!/usr/bin/env python3
"""
gen_scenes.py — generate a parameter grid of pump-in-factory test scenes.

Each grid cell = (base motion amplitude A [mm], amplification multiplier M):
* Every movable pump part gets its own random integer-cycle sinusoidal SE(3) motion
  (same scheme as omniverse_pipeline/add_motion.py), with translation amplitudes drawn
  from [0.5A, 1.5A] mm and rotational surface displacements from [0.25A, 0.75A] mm.
* ONE part (same seeded pick for every cell) has its amplitudes multiplied by M —
  the "amplified" part the downstream motion-amplification methods should recover.
* Every part gets a metallic UsdPreviewSurface material (metallic=1, roughness 0.35)
  with a distinct saturated color (HSV-spaced palette), replacing the pastel
  displayColor look. displayColor is also updated as a fallback.
* The animated USD is then composed into the factory hall via
  omniverse_pipeline/compose_scene.py (subject renamed to /World/pump), and a capture
  YAML is emitted by cloning capture_config_pump_warehouse.yaml with only
  scene.usd_path / output.capture_dir changed (10 cams, 60 frames etc. untouched).

All cells use the SAME rng seed, so per-part motion draws (directions, frequencies,
phases, normalized amplitude positions) are identical across the grid — cells differ
only in amplitude scale and the multiplier. This keeps the grid comparable.

Runs with plain python + usd-core + numpy (pxr imported lazily inside functions).

Usage:
    python scene-gen/gen_scenes.py \
        --segmented-usd Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd \
        --env Q:/Omniverse/assets/physAI_start/Factory/Factory.usd \
        --out-dir omniverse-pipeline/data/scenes/grid \
        --base-amp-mm 1 4 16 --multipliers 2 4 8 --seed 0

    python scene-gen/gen_scenes.py ... --only-cell 16x8   # just the largest cell
    python scene-gen/gen_scenes.py --selftest
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys

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


def _fmt(v: float) -> str:
    """Compact number for filenames: 1, 4, 16, 0.5 ..."""
    return ("%g" % v)


# ---------------------------------------------------------------- motion ----

def author_motion(inp: str, out: str, base_amp_mm: float, multiplier: float,
                  amplified_name: str, num_frames: int, fps: float, seed: int,
                  freq_hz: float = 10.0):
    """Write <out> with per-part periodic motion; `amplified_name` scaled by `multiplier`.

    Returns (movable_names, peak_surface_mm dict)."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(inp)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    units_per_mm = 1e-3 / mpu
    stage.GetRootLayer().Export(out)
    stage = Usd.Stage.Open(out)

    parts = add_motion.collect_parts(stage, GROUP, UsdGeom, Usd)
    movable = [(p, c, r) for (p, c, r) in parts if p.GetName() != "frame_base"]
    if amplified_name not in [p.GetName() for p, _, _ in movable]:
        raise SystemExit(f"[gen] amplified part {amplified_name} not among movable parts")

    # 10 Hz target: integer cycles over the clip so the motion loops seamlessly.
    # 240 frames @ 60 fps -> 239/60 s -> 40 cycles = 10.04 Hz (closest integer to 10 Hz).
    cycles = int(round(freq_hz * (num_frames - 1) / fps))
    cfg = dict(
        trans_amp=[0.5 * base_amp_mm * units_per_mm, 1.5 * base_amp_mm * units_per_mm],
        rot_surface=[0.25 * base_amp_mm * units_per_mm, 0.75 * base_amp_mm * units_per_mm],
        rot_deg_max=3.0,
        freq=[cycles, cycles],
    )
    rng = np.random.default_rng(seed)   # same seed every cell -> comparable grids

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    peak_mm = {}
    for prim, c, r in movable:
        name = prim.GetName()
        mp = add_motion._rand_motion(cfg, rng, r)
        if name == amplified_name:
            mp["trans_amp"] *= multiplier
            mp["rot_amp"] *= multiplier   # amplified part may exceed the 3 deg cap: intended
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
    return [p.GetName() for p, _, _ in movable], peak_mm


# ------------------------------------------------------------- materials ----

def apply_metallic_materials(usd_path: str):
    """Bind a metallic UsdPreviewSurface with a distinct saturated color to every part
    mesh under /World/<group>. Also refreshes displayColor as a fallback."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.Open(usd_path)
    grp = stage.GetPrimAtPath(f"/World/{GROUP}")
    parts = [p for p in grp.GetChildren() if p.GetTypeName() == "Xform"]
    n = len(parts)
    mats_scope = stage.DefinePrim(f"/World/{GROUP}/Materials", "Scope")

    for i, part in enumerate(sorted(parts, key=lambda p: p.GetName())):
        # saturated hues, golden-ratio spaced so adjacent parts never share a hue family;
        # offset 0.62 starts in blue (avoids the red-heavy look of hue = i/N).
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


# ------------------------------------------------------------ capture yaml ----

def write_capture_config(template_path: str, out_path: str, scene_usd: str,
                         capture_dir: str, num_frames: int | None = None,
                         max_elev_deg: float | None = None):
    """Clone the warehouse capture config, swapping usd_path / capture_dir (and
    optionally num_frames / max_elev_deg) via text substitution so all other
    settings/comments survive verbatim."""
    with open(template_path) as f:
        text = f.read()
    text = re.sub(r'usd_path: ".*?"', f'usd_path: "{scene_usd}"', text, count=1)
    text = re.sub(r'capture_dir: ".*?"', f'capture_dir: "{capture_dir}"', text, count=1)
    if num_frames is not None:
        text = re.sub(r'num_frames: \d+', f'num_frames: {num_frames}', text, count=1)
    if max_elev_deg is not None:
        text = re.sub(r'max_elev_deg: [\d.]+', f'max_elev_deg: {max_elev_deg:g}',
                      text, count=1)
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
    rng = np.random.default_rng(seed + 99991)  # separate stream from motion draws
    return names[int(rng.integers(0, len(names)))]


def generate_cell(segmented_usd: str, env_usd: str, out_dir: str, amp: float,
                  mult: float, amplified: str, num_frames: int, fps: float,
                  seed: int, capture_root: str, freq_hz: float) -> dict:
    name = cell_name(amp, mult)
    animated = os.path.join(out_dir, f"{name}_animated.usd")
    scene = os.path.join(out_dir, f"{name}_scene.usd")
    yaml_path = os.path.join(out_dir, f"{name}.yaml")
    capture_dir = f"{capture_root}/capture_{name}"

    movable, peak_mm = author_motion(segmented_usd, animated, amp, mult, amplified,
                                     num_frames, fps, seed, freq_hz)
    n_mats = apply_metallic_materials(animated)
    compose_scene.compose(
        in_usd=animated, out_usd=scene, group=SUBJECT_NAME,
        src_prim=f"/World/{GROUP}", env_usd=env_usd, env_root="/World/Factory",
        translate=TRANSLATE, scale=SCALE)
    write_capture_config(TEMPLATE_CONFIG, yaml_path,
                         os.path.abspath(scene).replace(os.sep, "/"), capture_dir,
                         num_frames=num_frames, max_elev_deg=MAX_ELEV_DEG)

    groups = {
        "num_frames": num_frames, "fps": fps, "freq_hz": freq_hz,
        "base_amp_mm": amp, "multiplier": mult, "amplified_part": amplified,
        "seed": seed, "peak_surface_mm": peak_mm,
    }
    with open(os.path.join(out_dir, f"{name}_motion.json"), "w") as f:
        json.dump(groups, f, indent=2)

    info = {"cell": name, "base_amp_mm": amp, "multiplier": mult,
            "animated": animated, "scene": scene, "config": yaml_path,
            "capture_dir": capture_dir, "materials": n_mats,
            "amplified_peak_mm": peak_mm[amplified]}
    print(f"[cell {name}] parts={len(movable)} mats={n_mats} "
          f"amplified={amplified} peak={peak_mm[amplified]:.1f} mm")
    return info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cells = [(a, m) for a in args.base_amp_mm for m in args.multipliers]
    if args.only_cell:
        a_s, m_s = re.split(r"[x,]", args.only_cell)
        cells = [(float(a_s), float(m_s))]

    amplified = pick_amplified_part(args.segmented_usd, args.seed)
    cycles = int(round(args.freq_hz * (args.num_frames - 1) / args.fps))
    print(f"[gen] amplified part (all cells): {amplified}; "
          f"freq {args.freq_hz} Hz -> {cycles} cycles/clip "
          f"({cycles * args.fps / (args.num_frames - 1):.2f} Hz effective)")

    manifest = {"seed": args.seed, "amplified_part": amplified,
                "freq_hz": args.freq_hz, "cycles": cycles, "cells": []}
    for amp, mult in cells:
        manifest["cells"].append(generate_cell(
            args.segmented_usd, args.env, out_dir, amp, mult, amplified,
            args.num_frames, args.fps, args.seed, args.capture_root, args.freq_hz))
    with open(os.path.join(out_dir, "grid_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gen] {len(manifest['cells'])} cell(s) -> {out_dir} (+ grid_manifest.json)")


# --------------------------------------------------------------- selftest ----

def _selftest() -> int:
    """Tiny 3-part subject: check (1) amplified part peak scales ~M×, (2) non-amplified
    parts are identical between M=1 and M=4 cells (same seed), (3) materials bind."""
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

    amp, seed = 4.0, 0
    out1 = os.path.join(tmp, "m1.usd")
    out4 = os.path.join(tmp, "m4.usd")
    _, peak1 = author_motion(src, out1, amp, 1.0, "part_002", 60, 24.0, seed)
    _, peak4 = author_motion(src, out4, amp, 4.0, "part_002", 60, 24.0, seed)
    n_mat = apply_metallic_materials(out4)

    checks = {
        "amplified peak ~4x": abs(peak4["part_002"] / peak1["part_002"] - 4.0) < 0.05,
        "other parts unchanged": all(abs(peak1[n] - peak4[n]) < 1e-9
                                     for n in ("part_001", "part_003")),
        "motion is mm-scale": 1.0 < peak1["part_001"] < 20.0,
        "4 materials authored": n_mat == 4,
    }
    st = Usd.Stage.Open(out4)
    mesh = st.GetPrimAtPath(f"/World/{GROUP}/part_002/mesh")
    bound = UsdShade.MaterialBindingAPI(mesh).GetDirectBinding().GetMaterial()
    checks["material bound to mesh"] = bool(bound) and \
        bound.GetPrim().GetName() == "part_002_mat"
    surf = UsdShade.Shader(st.GetPrimAtPath(
        f"/World/{GROUP}/Materials/part_002_mat/PBRShader"))
    checks["metallic=1"] = abs(surf.GetInput("metallic").Get() - 1.0) < 1e-6

    for name, ok in checks.items():
        print(f"[selftest] {name}: {'ok' if ok else 'FAIL'}")
    ok = all(checks.values())
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
