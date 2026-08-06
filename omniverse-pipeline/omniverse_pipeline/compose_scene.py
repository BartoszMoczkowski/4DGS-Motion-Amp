#!/usr/bin/env python3
"""
compose_scene.py — composite an animated subject USD (e.g. CONJUNTO_BOMBAS_animated.usd)
into a static environment USD (e.g. the physAI_start Factory hall), producing one stage
that omni_capture.py can open: industrial background geometry + the animated subject in
one timeline.

Design:
* Pure composition — no flattening. The output stage holds two references:
  the environment under `--env-root` (default /World/Environment, static background) and
  the subject at /World/<group> (same path the rest of the pipeline already expects:
  capture's `scene.semantic_roots`, subject bbox, and init-point sampling all key off it).
  The subject's per-part animation rides along through the reference untouched.
* Placement: the subject reference gets `translate` / `rotateX` / `scale` xform ops
  (applied in that order: scale, then rotate, then translate). Defaults are identity —
  pass explicit values to seat the subject in the environment. For the pump01 scene the
  values mirror physAI_start/SceneAssembly.usd, which places this exact pump in this
  exact factory at translate=(-130.29, -437.17, 0.35), scale=0.2.
* Stage metadata (metersPerUnit, framesPerSecond, start/end timeCode) is copied from the
  subject input so the capture timeline still lines up with the authored keyframes;
  upAxis comes from the environment when one is given, else from the subject.
* Reference paths are written verbatim: `--ref-in`/`--ref-env` default to `--in`/`--env`,
  but callers that compose on one filesystem view and render on another (the orchestrator
  writes this file inside the `isaac` container, then capture.isaac opens it natively on
  the Windows host) pass the *host* paths via `--ref-*` while `--in`/`--env` stay
  container-local (they are only opened for metadata, never resolved through the output).
* No environment (`--env` omitted) is a valid pass-through: the output just re-references
  the subject at the same path. Keeps downstream wiring (capture opens the composed
  scene) uniform whether or not a preset wants a background.

Runs with plain python (usd-core only) — same situation as split_mesh.py / add_motion.py.

Usage:
    python compose_scene.py --in CONJUNTO_BOMBAS_animated.usd \
        --out CONJUNTO_BOMBAS_warehouse.usd --group CONJUNTO_BOMBAS \
        --env Q:/Omniverse/assets/physAI_start/Factory/Factory.usd \
        --env-root /World/Factory \
        --translate -130.29 -437.17 0.35 --scale 0.2

    python compose_scene.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys


def compose(
    in_usd: str,
    out_usd: str,
    group: str,
    env_usd: str | None = None,
    env_root: str = "/World/Environment",
    src_prim: str | None = None,
    ref_in: str | None = None,
    ref_env: str | None = None,
    translate=(0.0, 0.0, 0.0),
    rotate_x_deg: float = 0.0,
    scale: float = 1.0,
) -> dict:
    from pxr import Gf, Usd, UsdGeom

    ref_in = ref_in or in_usd
    ref_env = ref_env or env_usd

    src = Usd.Stage.Open(in_usd)
    subject_path = f"/World/{group}"
    src_path = src_prim or subject_path
    if not src.GetPrimAtPath(src_path):
        raise SystemExit(f"[compose] subject prim {src_path} not found in {in_usd}")

    env = Usd.Stage.Open(env_usd) if env_usd else None

    if os.path.exists(out_usd):
        os.remove(out_usd)
    stage = Usd.Stage.CreateNew(out_usd)

    up_axis = UsdGeom.GetStageUpAxis(env) if env else UsdGeom.GetStageUpAxis(src)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    stage.SetMetadata("metersPerUnit", src.GetMetadata("metersPerUnit"))
    stage.SetFramesPerSecond(src.GetFramesPerSecond())
    stage.SetStartTimeCode(src.GetStartTimeCode())
    stage.SetEndTimeCode(src.GetEndTimeCode())
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    if env:
        env_prim = stage.DefinePrim(env_root, "Xform")
        # Reference the env's defaultPrim explicitly (falling back to /World): a bare
        # file reference only resolves if the env layer declares a defaultPrim, and
        # hand-built envs often don't.
        target = env.GetDefaultPrim()
        if not target or not target.IsValid():
            target = env.GetPrimAtPath("/World")
        if not target:
            raise SystemExit(f"[compose] no defaultPrim or /World in env {env_usd}")
        env_prim.GetReferences().AddReference(ref_env, target.GetPath())

    subj = stage.DefinePrim(subject_path, "Xform")
    subj.GetReferences().AddReference(ref_in, src_path)
    xf = UsdGeom.Xformable(subj)
    ops = []
    if any(abs(v) > 1e-12 for v in translate):
        ops.append(xf.AddTranslateOp())
        ops[-1].Set(tuple(float(v) for v in translate))
    if abs(rotate_x_deg) > 1e-12:
        ops.append(xf.AddRotateXOp())
        ops[-1].Set(float(rotate_x_deg))
    if abs(scale - 1.0) > 1e-12:
        ops.append(xf.AddScaleOp())
        ops[-1].Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    if ops:
        xf.SetXformOpOrder(ops)

    stage.GetRootLayer().Save()
    info = {
        "out": out_usd,
        "subject": subject_path,
        "env_root": env_root if env else None,
        "up_axis": str(up_axis),
        "frames": [src.GetStartTimeCode(), src.GetEndTimeCode()],
        "fps": src.GetFramesPerSecond(),
    }
    print(f"[compose] {info}")
    return info


def _selftest() -> int:
    """Build a tiny animated subject + a tiny env in a temp dir, compose, and verify the
    output stage: both references resolve, the subject lands at /World/<group> with the
    requested transform, and the animation survives composition."""
    import tempfile

    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    tmp = tempfile.mkdtemp(prefix="compose_selftest_")

    # Subject: /World/BODY/part_001 — a cube mesh translated over 60 frames.
    subj_path = os.path.join(tmp, "subject.usd")
    s = Usd.Stage.CreateNew(subj_path)
    UsdGeom.SetStageUpAxis(s, "Y")
    s.SetMetadata("metersPerUnit", 0.01)
    s.SetFramesPerSecond(24)
    s.SetStartTimeCode(0)
    s.SetEndTimeCode(59)
    s.DefinePrim("/World", "Xform")
    body = UsdGeom.Xform.Define(s, "/World/BODY/part_001")
    tattr = body.AddTranslateOp()
    for f in range(60):
        tattr.Set(Gf.Vec3d(float(f), 0.0, 0.0), Usd.TimeCode(f))
    mesh = UsdGeom.Mesh.Define(s, "/World/BODY/part_001/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    s.GetRootLayer().Save()

    # Env: a static floor plane under /World/FloorGeo.
    env_path = os.path.join(tmp, "env.usd")
    e = Usd.Stage.CreateNew(env_path)
    UsdGeom.SetStageUpAxis(e, "Z")
    e.SetMetadata("metersPerUnit", 0.01)
    e.DefinePrim("/World", "Xform")
    floor = UsdGeom.Mesh.Define(e, "/World/FloorGeo/floor")
    floor.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    e.GetRootLayer().Save()

    out_path = os.path.join(tmp, "composed.usd")
    compose(
        in_usd=subj_path,
        out_usd=out_path,
        group="BODY",
        env_usd=env_path,
        env_root="/World/Factory",
        translate=(10.0, 20.0, 30.0),
        scale=0.5,
    )

    c = Usd.Stage.Open(out_path)
    checks = {
        "subject exists": bool(c.GetPrimAtPath("/World/BODY")),
        "subject part exists": bool(c.GetPrimAtPath("/World/BODY/part_001")),
        "env exists": bool(c.GetPrimAtPath("/World/Factory")),
        "env floor resolves": bool(c.GetPrimAtPath("/World/Factory/FloorGeo/floor")),
        "upAxis from env (Z)": UsdGeom.GetStageUpAxis(c) == "Z",
        "timeline kept": c.GetStartTimeCode() == 0 and c.GetEndTimeCode() == 59,
    }
    # Animation rides through the reference: world position of the cube at frame 7 must
    # reflect part anim (x=7), subject scale (0.5) and translate (+10 in x).
    cache = UsdGeom.XformCache(Usd.TimeCode(7))
    pts = np.array(
        UsdGeom.Mesh(c.GetPrimAtPath("/World/BODY/part_001/mesh")).GetPointsAttr().Get(),
        dtype=float,
    )
    M = np.array(cache.GetLocalToWorldTransform(c.GetPrimAtPath("/World/BODY/part_001/mesh"))).T
    world = (M[:3, :3] @ pts.T).T + M[:3, 3]
    expect = np.array([(10.0 + 7 * 0.5) + pts[:, 0] * 0.5, 20.0 + pts[:, 1] * 0.5, 30.0 + pts[:, 2] * 0.5]).T
    checks["anim+transform compose"] = bool(np.allclose(world, expect, atol=1e-6))

    for name, ok in checks.items():
        print(f"[selftest] {name}: {'ok' if ok else 'FAIL'}")
    ok = all(checks.values())
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="in_usd", help="animated subject USD (opened for metadata)")
    ap.add_argument("--out", dest="out_usd", help="output composed USD")
    ap.add_argument("--group", default="CONJUNTO_BOMBAS",
                    help="subject root prim name in the OUTPUT; composed as /World/<group>")
    ap.add_argument("--src-prim", default=None,
                    help="subject prim path in the INPUT (default: /World/<group>). Set this "
                    "to rename the subject in the composed scene, e.g. --group pump "
                    "--src-prim /World/CONJUNTO_BOMBAS.")
    ap.add_argument("--env", default=None, help="environment USD (opened for metadata); omit for pass-through")
    ap.add_argument("--env-root", default="/World/Environment",
                    help="prim path the environment is referenced under")
    ap.add_argument("--ref-in", default=None,
                    help="path written into the subject reference (default: --in)")
    ap.add_argument("--ref-env", default=None,
                    help="path written into the environment reference (default: --env)")
    ap.add_argument("--translate", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"), help="subject placement in stage units")
    ap.add_argument("--rotate-x", type=float, default=0.0,
                    help="subject rotation about X in degrees (e.g. -90 for Y-up asset in Z-up world)")
    ap.add_argument("--scale", type=float, default=1.0, help="subject uniform scale")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.in_usd or not args.out_usd:
        ap.error("--in and --out are required (or use --selftest)")

    compose(
        in_usd=args.in_usd,
        out_usd=args.out_usd,
        group=args.group,
        env_usd=args.env,
        env_root=args.env_root,
        src_prim=args.src_prim,
        ref_in=args.ref_in,
        ref_env=args.ref_env,
        translate=tuple(args.translate),
        rotate_x_deg=args.rotate_x,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
