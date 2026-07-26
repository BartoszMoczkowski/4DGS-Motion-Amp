#!/usr/bin/env python3
"""
split_mesh.py — split a single fused mesh (e.g. an STL-imported CAD assembly) into its
separate rigid parts by connected components, and write a structured, per-part-labelled USD.

Motivation: `CONJUNTO BOMBAS.usd` is one welded Mesh with no sub-parts, so it can't be
motion-segmented or per-part-animated. CAD-exported STLs almost always keep each physical
body as a topologically separate shell, so welding coincident vertices and splitting by
connected components recovers the original parts.

Output USD layout (each part is an Xform+Mesh so it can be moved/animated & labelled):
    /World  (Xform, up-axis + metersPerUnit preserved)
      /World/<group>            (Xform)
        /World/<group>/frame_base   (Xform, class="frame_base")  <- largest component
          .../mesh                  (Mesh, per-part displayColor)
        /World/<group>/part_001     (Xform, class="part_001")
          .../mesh
        ...

Runs with plain python (trimesh + usd-core). Verify with `--preview out.png`.

Usage:
    python split_mesh.py --in "CONJUNTO BOMBAS.usd" --out CONJUNTO_BOMBAS_segmented.usd \
        --group CONJUNTO_BOMBAS --min-faces 12 --preview seg_preview.png
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def load_geometry(path):
    """Return (vertices Nx3 float, faces Mx3 int, up_axis, meters_per_unit).
    Reads a .usd/.usda/.usdc mesh (first Mesh prim, baked to world) or a mesh trimesh can load."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".usd", ".usda", ".usdc", ".usdz"):
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(path)
        up = UsdGeom.GetStageUpAxis(stage)
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        verts_all, faces_all, offset = [], [], 0
        for p in stage.Traverse():
            if p.GetTypeName() != "Mesh":
                continue
            m = UsdGeom.Mesh(p)
            pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.array(m.GetFaceVertexCountsAttr().Get())
            idx = np.array(m.GetFaceVertexIndicesAttr().Get())
            if not np.all(counts == 3):                     # triangulate fans if needed
                tri, i = [], 0
                for c in counts:
                    for k in range(1, c - 1):
                        tri.append([idx[i], idx[i + k], idx[i + k + 1]])
                    i += c
                faces = np.array(tri)
            else:
                faces = idx.reshape(-1, 3)
            M = np.array(cache.GetLocalToWorldTransform(p)).T   # USD is row-vector; transpose
            ptw = (M[:3, :3] @ pts.T).T + M[:3, 3]
            verts_all.append(ptw); faces_all.append(faces + offset); offset += len(ptw)
        verts = np.concatenate(verts_all); faces = np.concatenate(faces_all)
        return verts, faces, up, mpu
    else:
        import trimesh
        m = trimesh.load(path, process=False)
        return np.asarray(m.vertices), np.asarray(m.faces), "Y", 1.0


def split_components(verts, faces, weld_tol=1e-5):
    """Weld coincident vertices, split into connected components.
    Returns list of (verts, faces) sorted by face count descending."""
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)   # process welds verts
    comps = m.split(only_watertight=False)
    comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
    return [(np.asarray(c.vertices), np.asarray(c.faces)) for c in comps]


def build_usd(parts, out_path, group, up_axis="Y", mpu=0.01,
              names=None, seed=42):
    from pxr import Usd, UsdGeom, Sdf, Gf, Vt

    def bake_semantics(prim, label):
        """Best-effort: write a 'class' label with whichever USD semantics schema exists.
        (Runtime Replicator semantics are also applied by omni_capture.py via semantic_roots.)"""
        done = False
        try:                                    # modern (USD >= 24.11 / Isaac 5.x)
            from pxr import UsdSemantics
            api = UsdSemantics.LabelsAPI.Apply(prim, "class")
            api.CreateLabelsAttr(Vt.TokenArray([label])); done = True
        except Exception:
            pass
        try:                                    # legacy (older Kit / Replicator)
            from pxr import Semantics
            api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            api.CreateSemanticTypeAttr().Set("class")
            api.CreateSemanticDataAttr().Set(label); done = True
        except Exception:
            pass
        return done

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    grp = UsdGeom.Xform.Define(stage, f"/World/{group}")

    rs = np.random.RandomState(seed)
    palette = rs.randint(40, 235, size=(len(parts), 3)) / 255.0
    if names is None:
        names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, len(parts))]

    for i, ((v, f), name) in enumerate(zip(parts, names)):
        part_xf = UsdGeom.Xform.Define(stage, f"/World/{group}/{name}")
        sem_ok = bake_semantics(part_xf.GetPrim(), name)   # label read by Replicator seg
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{group}/{name}/mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
            np.full(len(f), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
            f.reshape(-1).astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        c = palette[i]
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))]))
        # extent (bbox) — good practice for USD meshes
        mn, mx = v.min(0), v.max(0)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*mn.astype(float)),
                                             Gf.Vec3f(*mx.astype(float))]))
    stage.GetRootLayer().Save()
    return stage


def preview(parts, out_png, max_pts=40000, seed=42):
    """Colored 3-view scatter of per-part sampled vertices — quick visual QA (no GPU)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = np.random.RandomState(seed)
    palette = rs.randint(40, 235, size=(len(parts), 3)) / 255.0
    # parts[0] is the largest (frame_base): draw it light grey & sparse so movers stand out.
    per = [min(1500, len(v)) for v, _ in parts]
    per[0] = min(6000, len(parts[0][0]))
    fig = plt.figure(figsize=(16, 6))
    views = [(20, -60, "persp"), (0, 0, "front (XZ)"), (90, -90, "top (XY)")]
    for a, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, a + 1, projection="3d")
        for i, (v, _) in enumerate(parts):
            k = min(per[i], len(v)); idx = rs.choice(len(v), k, replace=False)
            s = v[idx]
            col, sz, al = (([[0.8, 0.8, 0.8]], 1, 0.25) if i == 0
                           else ([palette[i]], 4, 0.9))
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=sz, c=col, alpha=al, depthshade=False)
        ax.set_title(f"{title} — {len(parts)} parts (frame grey)"); ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        try: ax.set_box_aspect(np.ptp(np.concatenate([p[0] for p in parts]), axis=0))
        except Exception: pass
    plt.tight_layout(); plt.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input fused mesh (.usd/.stl/...)")
    ap.add_argument("--out", required=True, help="output segmented .usd")
    ap.add_argument("--group", default="CONJUNTO_BOMBAS", help="parent Xform name")
    ap.add_argument("--min-faces", type=int, default=1,
                    help="components with fewer faces are merged into the nearest larger part")
    ap.add_argument("--preview", help="also write a colored preview PNG")
    ap.add_argument("--no-usd", action="store_true", help="analysis/preview only, skip USD write")
    args = ap.parse_args()

    verts, faces, up, mpu = load_geometry(args.inp)
    print(f"[load] {len(verts)} verts, {len(faces)} faces, up={up}, mpu={mpu}")
    parts = split_components(verts, faces)
    print(f"[split] {len(parts)} connected components")

    if args.min_faces > 1:
        big = [p for p in parts if len(p[1]) >= args.min_faces]
        small = [p for p in parts if len(p[1]) < args.min_faces]
        if small:
            centers = [v.mean(0) for v, _ in big]
            for v, f in small:
                c = v.mean(0)
                j = int(np.argmin([np.linalg.norm(c - bc) for bc in centers]))
                bv, bf = big[j]
                big[j] = (np.concatenate([bv, v]), np.concatenate([bf, f + len(bv)]))
            print(f"[merge] folded {len(small)} sub-{args.min_faces}-face fragments into nearest parts")
        parts = sorted(big, key=lambda c: len(c[1]), reverse=True)

    print(f"[parts] final {len(parts)}; largest={len(parts[0][1])} faces -> 'frame_base'")
    if args.preview:
        preview(parts, args.preview); print(f"[preview] {args.preview}")
    if not args.no_usd:
        build_usd(parts, args.out, args.group, up, mpu)
        tot = sum(len(f) for _, f in parts)
        print(f"[usd] {args.out}  ({len(parts)} parts, {tot} faces total)")


if __name__ == "__main__":
    main()
