#!/usr/bin/env python3
"""
add_motion.py — author subtle, periodic, per-part rigid motion onto a segmented USD
(e.g. CONJUNTO_BOMBAS_segmented.usd from split_mesh.py), turning a static parts assembly
into a controlled motion-segmentation / motion-amplification test case.

Design (matches project goals: rigid machines, whole-clip periodic, small amplitude):
* Each movable part gets ONE rigid SE(3) sinusoidal motion:
    M(t) = Trans(c + A_t·sin(2π f_t·u + φ_t)·d_t) · Rot(A_r·sin(2π f_r·u + φ_r), d_r) · Trans(-c)
  where c = part centroid (so rotation pivots about the part, not the world origin),
  u = t / (num_frames-1) ∈ [0,1], and frequencies f are INTEGER cycles over the clip so the
  motion is exactly periodic and loops seamlessly.
* `frame_base` (and any --exclude) stays static.
* Motion is authored as a time-sampled `xformOp:transform` (Matrix4d) per part.
* Amplitudes are small by default (few mm translation, ~1° rotation) — the regime motion
  amplification is meant to reveal. Units follow the stage (pump: metersPerUnit=0.01 → cm,
  so --trans-amp-mm is converted to stage units automatically).

Each part therefore has its own distinct rigid motion, so the part's instance label doubles
as its ground-truth motion-segment label. Use --groups K to instead cluster parts into K
shared-motion groups (K+1 segments incl. static frame); a `motion_groups.json` is written.

Runs with plain python (usd-core + numpy). Verify with --report / --plot.

Usage:
    python add_motion.py --in CONJUNTO_BOMBAS_segmented.usd --out CONJUNTO_BOMBAS_animated.usd \
        --group CONJUNTO_BOMBAS --num-frames 60 --fps 24 \
        --trans-amp-mm 1 4 --rot-amp-deg 0.5 2 --freq 2 5 --plot motion_qa.png
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0, 0])


def axis_angle_matrix(axis, angle_rad):
    """3x3 rotation via Rodrigues."""
    a = _unit(np.asarray(axis, float)); th = float(angle_rad)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def collect_parts(stage, group, UsdGeom, Usd):
    """Return list of (prim, centroid, radius) for each part Xform under /World/<group>.
    radius = max distance of any mesh point from the centroid (used to bound rotational
    surface motion so it stays subtle regardless of part size)."""
    parts = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for p in stage.Traverse():
        if p.GetTypeName() == "Xform" and p.GetParent().GetName() == group:
            pts = None
            for c in p.GetChildren():
                if c.GetTypeName() == "Mesh":
                    pts = np.array(UsdGeom.Mesh(c).GetPointsAttr().Get(), dtype=np.float64)
                    M = np.array(cache.GetLocalToWorldTransform(c)).T
                    pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
                    break
            if pts is None or len(pts) == 0:
                continue
            cen = pts.mean(axis=0)
            rad = float(np.linalg.norm(pts - cen, axis=1).max())
            parts.append((p, cen, rad))
    return parts


def assign_motion_params(part_names, centroids, radii, cfg, rng):
    """Per-part motion parameters. If cfg['groups']>0, parts sharing a group share params.
    Rotation amplitude is derived from a target SURFACE displacement (mm-scale) divided by the
    representative part radius, so rotational motion stays subtle for large and small parts."""
    n = len(part_names)
    groups = cfg.get("groups", 0)
    if groups and groups > 0:
        C = np.asarray(centroids)
        idx = rng.choice(n, min(groups, n), replace=False)
        cent = C[idx].copy()
        for _ in range(25):
            d = np.linalg.norm(C[:, None] - cent[None], axis=2)
            lab = d.argmin(1)
            for k in range(len(cent)):
                if (lab == k).any():
                    cent[k] = C[lab == k].mean(0)
        group_of = lab
    else:
        group_of = np.arange(n)                     # each part its own group

    radii = np.asarray(radii)
    params = {}
    for g in np.unique(group_of):
        rep_radius = float(radii[group_of == g].max())     # bound worst-case surface motion
        params[int(g)] = _rand_motion(cfg, rng, rep_radius)
    return group_of, params


def _rand_motion(cfg, rng, radius):
    ta = rng.uniform(*cfg["trans_amp"])             # stage units (from mm)
    # rotation: pick a target surface displacement (stage units) -> angle = disp / radius,
    # capped at cfg['rot_deg_max'] so tiny parts don't spin.
    surf = rng.uniform(*cfg["rot_surface"])         # stage units
    ang = min(surf / max(radius, 1e-6), np.deg2rad(cfg["rot_deg_max"]))
    ft = int(rng.integers(cfg["freq"][0], cfg["freq"][1] + 1))
    fr = int(rng.integers(cfg["freq"][0], cfg["freq"][1] + 1))
    return dict(
        trans_amp=ta, trans_dir=_unit(rng.normal(size=3)), trans_freq=ft,
        trans_phase=rng.uniform(0, 2*np.pi),
        rot_amp=ang, rot_dir=_unit(rng.normal(size=3)), rot_freq=fr,
        rot_phase=rng.uniform(0, 2*np.pi),
    )


def part_matrix(c, mp, u):
    """4x4 (numpy, row-vector/USD convention on return) at normalized time u∈[0,1]."""
    off = mp["trans_amp"] * np.sin(2*np.pi*mp["trans_freq"]*u + mp["trans_phase"]) * mp["trans_dir"]
    ang = mp["rot_amp"] * np.sin(2*np.pi*mp["rot_freq"]*u + mp["rot_phase"])
    R = axis_angle_matrix(mp["rot_dir"], ang)
    # world transform: translate(-c), rotate, translate(c+off)  (column-vector math)
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = (c + off) - R @ c
    return T


def author(inp, out, group, cfg, rng):
    from pxr import Usd, UsdGeom, Gf, Sdf
    stage = Usd.Stage.Open(inp)
    # export to a fresh layer so we don't mutate the segmented source
    stage.GetRootLayer().Export(out)
    stage = Usd.Stage.Open(out)

    parts = collect_parts(stage, group, UsdGeom, Usd)
    exclude = set(cfg.get("exclude", ["frame_base"]))
    movable = [(p, c, r) for (p, c, r) in parts if p.GetName() not in exclude]
    names = [p.GetName() for p, _, _ in movable]
    centroids = [c for _, c, _ in movable]
    radii = [r for _, _, r in movable]
    group_of, params = assign_motion_params(names, centroids, radii, cfg, rng)

    n_frames = int(cfg["num_frames"]); fps = float(cfg["fps"])
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(n_frames - 1)
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)

    peak = {}
    for (prim, c, r), name, g in zip(movable, names, group_of):
        mp = params[int(g)]
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        surf = []                                    # peak SURFACE displacement (trans + rot·radius)
        for fi in range(n_frames):
            u = fi / max(n_frames - 1, 1)
            M = part_matrix(np.asarray(c), mp, u)
            gm = Gf.Matrix4d(*M.T.flatten())        # USD is row-vector -> transpose
            op.Set(gm, Usd.TimeCode(fi))
            off = mp["trans_amp"] * np.sin(2*np.pi*mp["trans_freq"]*u + mp["trans_phase"])
            ang = mp["rot_amp"] * np.sin(2*np.pi*mp["rot_freq"]*u + mp["rot_phase"])
            surf.append(abs(off) + abs(ang) * r)     # worst-case surface point displacement
        peak[name] = float(max(surf))

    stage.GetRootLayer().Save()
    # write group / label mapping (GT motion segments)
    seg = {name: int(g) for name, g in zip(names, group_of)}
    seg["frame_base"] = -1
    with open(os.path.splitext(out)[0] + "_motion_groups.json", "w") as f:
        json.dump({"num_groups": int(len(params)), "segment_of_part": seg,
                   "num_frames": n_frames, "fps": fps}, f, indent=2)
    return stage, movable, params, group_of, peak, n_frames


def make_config(a):
    return dict(num_frames=a.num_frames, fps=a.fps,
                trans_amp=[a.trans_amp_units[0], a.trans_amp_units[1]],
                rot_surface=[a.rot_surface_units[0], a.rot_surface_units[1]],
                rot_deg_max=a.rot_deg_max, freq=a.freq, groups=a.groups,
                exclude=a.exclude)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="CONJUNTO_BOMBAS")
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--trans-amp-mm", type=float, nargs=2, default=[1.0, 4.0],
                    help="min max translation amplitude in MILLIMETRES")
    ap.add_argument("--rot-surface-mm", type=float, nargs=2, default=[0.5, 3.0],
                    help="min max ROTATIONAL surface displacement in MILLIMETRES "
                         "(angle derived per part from its radius, so rotation stays subtle)")
    ap.add_argument("--rot-deg-max", type=float, default=3.0,
                    help="hard cap on rotation angle (deg) so tiny parts don't over-spin")
    ap.add_argument("--freq", type=int, nargs=2, default=[2, 5],
                    help="min max integer cycles over the clip (keeps motion periodic)")
    ap.add_argument("--groups", type=int, default=0,
                    help="0 = each part independent; K = cluster parts into K shared-motion groups")
    ap.add_argument("--exclude", nargs="*", default=["frame_base"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", help="write a motion-QA image (displacement curves + peak map)")
    args = ap.parse_args()

    # convert mm -> stage units using the source stage's metersPerUnit
    from pxr import Usd, UsdGeom
    mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(args.inp)) or 1.0
    units_per_mm = 1e-3 / mpu                       # mm -> stage units
    args.trans_amp_units = [args.trans_amp_mm[0] * units_per_mm,
                            args.trans_amp_mm[1] * units_per_mm]
    args.rot_surface_units = [args.rot_surface_mm[0] * units_per_mm,
                              args.rot_surface_mm[1] * units_per_mm]
    print(f"[units] metersPerUnit={mpu} -> {units_per_mm:.3g} stage-units/mm; "
          f"trans {args.trans_amp_mm} mm, rot-surface {args.rot_surface_mm} mm")

    rng = np.random.default_rng(args.seed)
    stage, movable, params, group_of, peak, nf = author(
        args.inp, args.out, args.group, make_config(args), rng)
    n_seg = len(set(int(g) for g in group_of))
    pk = np.array(list(peak.values()))
    print(f"[motion] {len(movable)} movable parts, {n_seg} motion segment(s), {nf} frames")
    print(f"[motion] peak displacement (units): min={pk.min():.3f} max={pk.max():.3f} "
          f"mean={pk.mean():.3f}  (= {pk.mean()/units_per_mm:.2f} mm mean)")
    print(f"[out] {args.out}  (+ {os.path.splitext(args.out)[0]}_motion_groups.json)")

    if args.plot:
        _plot(movable, params, group_of, nf, args.plot, units_per_mm)
        print(f"[plot] {args.plot}")


def _surface_disp(mp, r, u):
    """Worst-case surface-point displacement (stage units): translation + rotation·radius."""
    off = abs(mp["trans_amp"] * np.sin(2*np.pi*mp["trans_freq"]*u + mp["trans_phase"]))
    ang = abs(mp["rot_amp"] * np.sin(2*np.pi*mp["rot_freq"]*u + mp["rot_phase"]))
    return off + ang * r


def _plot(movable, params, group_of, nf, out_png, units_per_mm):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    us = np.linspace(0, 1, nf)
    fig = plt.figure(figsize=(15, 5))
    # (1) surface displacement-vs-time for up to 8 parts -> periodicity & looping
    ax1 = fig.add_subplot(1, 3, 1)
    for i in range(min(8, len(movable))):
        prim, c, r = movable[i]; mp = params[int(group_of[i])]
        d = [_surface_disp(mp, r, u) / units_per_mm for u in us]
        ax1.plot(us, d, lw=1, label=prim.GetName())
    ax1.set_title("surface displacement (mm) vs normalized time")
    ax1.set_xlabel("u = t/(N-1)"); ax1.set_ylabel("mm"); ax1.legend(fontsize=6, ncol=2)
    # (2) start-vs-end transform residual (should be ~0 => motion loops)
    ax2 = fig.add_subplot(1, 3, 2)
    res = [np.abs(part_matrix(np.asarray(c), params[int(group_of[i])], 0.0)
                  - part_matrix(np.asarray(c), params[int(group_of[i])], 1.0)).max()
           for i, (_, c, _) in enumerate(movable)]
    ax2.hist(res, bins=20); ax2.set_title("start-vs-end residual\n(~0 => seamless loop)")
    ax2.set_xlabel("max |M(0)-M(1)|")
    # (3) 3D scatter of part centroids colored by peak surface displacement
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    C = np.array([c for _, c, _ in movable])
    pk = np.array([max(_surface_disp(params[int(group_of[i])], r, u) / units_per_mm for u in us)
                   for i, (_, _, r) in enumerate(movable)])
    s = ax3.scatter(C[:, 0], C[:, 1], C[:, 2], c=pk, cmap="viridis", s=12)
    fig.colorbar(s, ax=ax3, shrink=0.6, label="peak surface disp (mm)")
    ax3.set_title("per-part peak motion"); ax3.set_axis_off()
    plt.tight_layout(); plt.savefig(out_png, dpi=95, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
