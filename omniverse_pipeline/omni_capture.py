#!/usr/bin/env python3
"""
omni_capture.py — headless multi-camera synthetic capture for the 4DGS testing pipeline.

Opens a USD stage in Isaac Sim 5.1, rings it with a configurable static camera rig,
and captures synchronized RGB + instance/semantic segmentation + ground-truth camera
poses over the scene's animation timeline.  Output is consumed by `omni_to_4dgs.py`.

RUN IT WITH ISAAC SIM'S PYTHON (not plain python):
    Windows:  Q:\\Omniverse\\ISAAC_SIM\\IsaacSim\\tools\\packman\\python.bat ^
                  omni_capture.py --config capture_config.yaml
    (or the KAT python at Physical-AI-Learning-KAT\\tools\\packman\\python.bat)

This file CANNOT run in the assistant sandbox (needs a GPU + the Isaac Sim runtime).
The Isaac/Replicator API calls follow the Isaac Sim 5.1 docs; treat the first run as a
smoke test.  The camera-rig math (rig.py) and the downstream converter (omni_to_4dgs.py)
ARE unit-tested independently.

Config: see capture_config.yaml.  CLI flags override config values.
"""
from __future__ import annotations
import argparse, json, os, sys

# --- 1. Launch the app FIRST (must precede any omni.* / pxr import that needs the runtime) ---
def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="capture_config.yaml")
    ap.add_argument("--usd", help="override scene.usd_path")
    ap.add_argument("--out", help="override output.capture_dir")
    ap.add_argument("--n-cameras", type=int, help="override rig.n_cameras")
    ap.add_argument("--headless", type=int, choices=[0, 1], help="override app.headless")
    ap.add_argument("--frames", type=int, help="override capture.num_frames")
    return ap.parse_args()


def _load_config(path):
    """Load the YAML config. Uses PyYAML if available, else a small built-in parser that
    handles the subset these config files use (nested maps, scalars, inline [lists],
    comments) so it works on Isaac Sim's bundled python without extra installs."""
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text):
    """Minimal YAML parser for the pipeline configs: 2-space-indented nested maps, scalar
    values (str/int/float/bool/null), inline lists like [a, b], and # comments. No block
    sequences or multi-line strings (not used by our configs)."""
    def strip_comment(s):
        out, q = [], None
        for ch in s:
            if q:
                out.append(ch)
                if ch == q:
                    q = None
            elif ch in "\"'":
                q = ch; out.append(ch)
            elif ch == "#":
                break
            else:
                out.append(ch)
        return "".join(out)

    def scalar(v):
        v = v.strip()
        if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
            return v[1:-1]
        low = v.lower()
        if low in ("null", "~", "none", ""):
            return None
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def value(v):
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            return [scalar(x) for x in _split_top(inner)] if inner else []
        return scalar(v)

    root = {}
    # stack of (indent, container_dict)
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        rest = rest.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = value(rest)
    return root


def _split_top(s):
    """Split a comma list, respecting quotes (no nested brackets in our configs)."""
    parts, cur, q = [], [], None
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch; cur.append(ch)
        elif ch == ",":
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    if args.usd: cfg["scene"]["usd_path"] = args.usd
    if args.out: cfg["output"]["capture_dir"] = args.out
    if args.n_cameras: cfg["rig"]["n_cameras"] = args.n_cameras
    if args.frames is not None: cfg["capture"]["num_frames"] = args.frames
    headless = bool(cfg["app"].get("headless", True)) if args.headless is None else bool(args.headless)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp(launch_config={"headless": headless})

    # Imports that require the running app:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, Gf, UsdLux
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig as rigmod

    scene_cfg, rig_cfg = cfg["scene"], cfg["rig"]
    cap_cfg, out_cfg = cfg["capture"], cfg["output"]
    capture_dir = out_cfg["capture_dir"]
    os.makedirs(capture_dir, exist_ok=True)

    # --- 2. Open the stage ---
    usd_path = scene_cfg["usd_path"]
    print(f"[capture] opening {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    for _ in range(60):                     # let payloads/references resolve
        simulation_app.update()

    up_token = UsdGeom.GetStageUpAxis(stage)
    world_up = {"Y": [0, 1, 0], "Z": [0, 0, 1]}.get(up_token, [0, 0, 1])
    rig_cfg.setdefault("world_up", world_up)

    # --- 3. Scene bbox -> rig center/size ---
    subject = scene_cfg.get("subject_prim")   # optional: focus on one prim
    root = stage.GetPrimAtPath(subject) if subject else stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
    bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
    if not np.all(np.isfinite(bmin)):
        bmin, bmax = np.array([-100.]*3), np.array([100.]*3)
    center = (bmin + bmax) / 2
    radius = float(np.linalg.norm(bmax - bmin) / 2) or 100.0
    print(f"[capture] up={up_token} bbox center={center} radius={radius:.1f}")

    # --- 4. Build camera rig (OpenCV c2w) via the tested rig module ---
    poses = rigmod.build_rig(rig_cfg, center, radius)
    n_cams = len(poses)
    W, H = int(cap_cfg["width"]), int(cap_cfg["height"])
    intr = rigmod.intrinsics_from_fov(W, H, float(cap_cfg.get("vfov_deg", 45)))

    # --- 5. Create USD cameras + render products ---
    cam_scope = "/World/CaptureCameras"
    UsdGeom.Scope.Define(stage, cam_scope)
    render_products, cam_records = [], []
    # focalLength/aperture so the rendered vertical FoV matches `intr`
    v_ap = 20.955
    focal_mm = (v_ap / 2) / np.tan(np.deg2rad(float(cap_cfg.get("vfov_deg", 45))) / 2)
    h_ap = v_ap * (W / H)
    for i, c2w in enumerate(poses, start=1):
        path = f"{cam_scope}/cam{i:02d}"
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(float(focal_mm))
        cam.CreateVerticalApertureAttr(float(v_ap))
        cam.CreateHorizontalApertureAttr(float(h_ap))
        # OpenCV c2w (cols right,down,fwd) -> USD camera basis (right, up=-down, back=-fwd)
        R = c2w[:3, :3]
        right, up, back = R[:, 0], -R[:, 1], -R[:, 2]
        M = Gf.Matrix4d(
            float(right[0]), float(right[1]), float(right[2]), 0.0,
            float(up[0]),    float(up[1]),    float(up[2]),    0.0,
            float(back[0]),  float(back[1]),  float(back[2]),  0.0,
            float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3]), 1.0,
        )
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(M)
        rp = rep.create.render_product(path, (W, H), name=f"cam{i:02d}")
        render_products.append(rp)
        cam_records.append({"folder": f"cam{i:02d}",
                            "c2w": [float(v) for v in c2w.flatten()],
                            "near": float(cap_cfg.get("near", radius * 0.05)),
                            "far": float(cap_cfg.get("far", radius * 4.0))})

    # --- 6. Semantics (needed for instance/semantic segmentation GT) ---
    _apply_semantics(stage, scene_cfg, UsdGeom)

    # --- 6b. Lighting (a headless render is black without a light) ---
    _setup_lighting(stage, cfg.get("lighting", {}), center, radius, up_token, capture_dir, UsdLux, UsdGeom, Gf, np)

    # --- 7. Writer ---
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        output_dir=capture_dir,
        rgb=True,
        instance_segmentation=bool(out_cfg.get("instance_segmentation", True)),
        semantic_segmentation=bool(out_cfg.get("semantic_segmentation", True)),
        colorize_instance_segmentation=False,
        colorize_semantic_segmentation=False,
        camera_params=True,
    )
    writer.attach(render_products)
    rep.orchestrator.set_capture_on_play(False)

    # --- 8. Author authoritative GT (poses + intrinsics) ---
    with open(os.path.join(capture_dir, "cameras_gt.json"), "w") as f:
        json.dump({"intrinsics": intr, "up_axis": up_token,
                   "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                   "cameras": cam_records}, f, indent=2)

    # --- 9. Sample the meshes for the init point cloud (+ GT labels) ---
    _sample_pointcloud(stage, root, capture_dir, int(out_cfg.get("num_init_points", 100000)),
                       UsdGeom, Usd, np)

    # --- 10. Step over the animation timeline ---
    tl_start = stage.GetStartTimeCode()
    tl_end = stage.GetEndTimeCode()
    n_frames = int(cap_cfg["num_frames"])
    fps = stage.GetFramesPerSecond() or 24.0
    if tl_end > tl_start:
        times = np.linspace(tl_start, tl_end, n_frames)
    else:
        times = np.arange(n_frames)
    print(f"[capture] {n_cams} cams x {n_frames} frames, timecode {tl_start}->{tl_end}")
    for k, tc in enumerate(times):
        # advance the timeline to this timecode, then capture one frame from all products
        delta = (times[1] - times[0]) / fps if len(times) > 1 else 1.0 / fps
        omni.timeline.get_timeline_interface().set_current_time(float(tc) / fps)
        rep.orchestrator.step(rt_subframes=int(cap_cfg.get("rt_subframes", 8)),
                              delta_time=0.0, pause_timeline=True)
    rep.orchestrator.wait_until_complete()
    print(f"[capture] done -> {capture_dir}")
    simulation_app.close()


def _write_png(path, img, np):
    """Write an RGB uint8 HxWx3 array as PNG using only stdlib (no Pillow)."""
    import zlib, struct
    H, W, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(H))
    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def _make_bg_texture(path, np, H=1024, W=2048, base=(6, 7, 11), line=(28, 32, 44),
                     dot=(48, 54, 70), step=64, line_w=2):
    """Generate a dark equirectangular grid backdrop (near-black base, faint grid lines,
    brighter dots at intersections) — dark for contrast, patterned to give the background
    stable features for 4DGS."""
    img = np.zeros((H, W, 3), np.uint8); img[:] = base
    for i in range(line_w):
        img[i::step, :] = line; img[:, i::step] = line
    ys = np.arange(0, H, step); xs = np.arange(0, W, step)
    for y in ys:
        for x in xs:
            img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = dot
    _write_png(path, img, np)


def _setup_lighting(stage, lcfg, center, radius, up_token, capture_dir, UsdLux, UsdGeom, Gf, np):
    """Add lights so the headless capture isn't black. Config keys (all optional):
    add_dome(bool), dome_intensity, add_distant(bool), distant_intensity,
    distant_elev_deg, distant_azim_deg. Skips if the scene already has lights and
    lcfg.get('force') is not True.

    Background: the DomeLight's environment IS the visible background. Set
    dome_texture: 'generate' to render a dark grid backdrop (good contrast + features for
    4DGS) generated to <capture_dir>/bg_dome.png; or give a path to your own EXR/PNG."""
    if lcfg.get("enabled", True) is False:
        return
    existing = [p for p in stage.Traverse() if p.GetTypeName() in
                ("DomeLight", "DistantLight", "SphereLight", "RectLight", "DiskLight")]
    if existing and not lcfg.get("force", False):
        print(f"[lighting] scene already has {len(existing)} light(s); skipping (set lighting.force to override)")
        return
    scope = "/World/CaptureLights"
    UsdGeom.Scope.Define(stage, scope)
    if lcfg.get("add_dome", True):
        dome = UsdLux.DomeLight.Define(stage, f"{scope}/dome")
        dome.CreateIntensityAttr(float(lcfg.get("dome_intensity", 1000.0)))
        tex = lcfg.get("dome_texture")            # None | "generate" | path
        if tex == "generate":
            tex = os.path.join(capture_dir, "bg_dome.png")
            _make_bg_texture(tex, np,
                             base=lcfg.get("bg_base", [6, 7, 11]),
                             line=lcfg.get("bg_line", [28, 32, 44]),
                             dot=lcfg.get("bg_dot", [48, 54, 70]),
                             step=int(lcfg.get("bg_step", 64)))
        if tex:
            dome.CreateTextureFileAttr(str(tex))
            dome.CreateTextureFormatAttr("latlong")
            print(f"[lighting] dome background texture: {tex}")
        print(f"[lighting] dome light @ {lcfg.get('dome_intensity', 1000.0)}")
    if lcfg.get("add_distant", True):
        dist = UsdLux.DistantLight.Define(stage, f"{scope}/key")
        dist.CreateIntensityAttr(float(lcfg.get("distant_intensity", 3000.0)))
        dist.CreateAngleAttr(0.53)
        # aim it via elevation/azimuth about the scene up-axis
        elev = np.deg2rad(lcfg.get("distant_elev_deg", 45.0))
        azim = np.deg2rad(lcfg.get("distant_azim_deg", -45.0))
        xf = UsdGeom.Xformable(dist.GetPrim())
        xf.ClearXformOpOrder()
        # DistantLight emits along its local -Z; rotate so -Z points down-ish from elev/azim
        rx = UsdGeom.XformCommonAPI(dist.GetPrim())
        rx.SetRotate(Gf.Vec3f(float(-np.rad2deg(elev) - 90 if up_token == "Y" else -np.rad2deg(elev)),
                              float(np.rad2deg(azim)), 0.0))
        print(f"[lighting] distant key light @ {lcfg.get('distant_intensity', 3000.0)}")


def _apply_semantics(stage, scene_cfg, UsdGeom):
    """Tag prims with a 'class' semantic label so instance/semantic seg has content.
    Strategy: label each *direct child* of each configured `semantic_roots` prim with
    that child's name (one instance per moving part).  Falls back to all meshes."""
    import importlib

    def tag(prim, label):
        """Apply a ('class', label) semantic that Replicator segmentation reads. Tries the
        stable Isaac Sim util first, then raw USD schemas. Each attempt is guarded so a wrong
        signature/module falls through instead of aborting the run."""
        # 1) Isaac Sim core util (5.x path, then legacy path). Signature is stable:
        #    add_update_semantics(prim, semantic_label, type_label="class")
        for modname in ("isaacsim.core.utils.semantics", "omni.isaac.core.utils.semantics"):
            try:
                m = importlib.import_module(modname)
                m.add_update_semantics(prim, semantic_label=label, type_label="class")
                return True
            except Exception:
                pass
        # 2) Raw USD schemas — apply whichever import works (harmless to apply both).
        ok = False
        try:
            from pxr import UsdSemantics, Vt
            UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr(Vt.TokenArray([label]))
            ok = True
        except Exception:
            pass
        try:
            from pxr import Semantics
            s = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            s.CreateSemanticTypeAttr().Set("class")
            s.CreateSemanticDataAttr().Set(label)
            ok = True
        except Exception:
            pass
        return ok

    roots = scene_cfg.get("semantic_roots")
    tagged = 0
    if roots:
        for rp in roots:
            root = stage.GetPrimAtPath(rp)
            if not root or not root.IsValid():
                print(f"[semantics] root not found: {rp}"); continue
            for child in root.GetChildren():
                tag(child, child.GetName()); tagged += 1
    else:  # fallback: every mesh gets its own name
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                tag(prim, prim.GetName()); tagged += 1
    print(f"[semantics] tagged {tagged} prims")


def _sample_pointcloud(stage, root, out_dir, n_target, UsdGeom, Usd, np):
    """Collect mesh vertices (world space) with a per-mesh integer label; subsample to
    n_target; write points3D_gt.ply (+ points3D_labels.npy).  Colors are per-label
    pseudo-colors (real appearance is learned by 4DGS; init color is not critical)."""
    xyz_all, lab_all = [], []
    label_names = {}
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = [root] if root != stage.GetPseudoRoot() else stage.Traverse()
    it = stage.Traverse() if root == stage.GetPseudoRoot() else Usd.PrimRange(root)
    lid = 0
    for prim in it:
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        M = xf_cache.GetLocalToWorldTransform(prim)
        P = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
        Pw = np.array([M.Transform((x, y, z)) for x, y, z in P])
        # Use the *parent* Xform's name, not the mesh prim's own name: split_mesh.py builds
        # every part as .../<part_name>/mesh, so prim.GetName() is literally "mesh" for every
        # part and every point collapsed onto a single label. The parent Xform name is the
        # actual per-part identifier (matches _apply_semantics, which tags root.GetChildren()
        # the same way -> keeps point-cloud labels consistent with the rendered instance-seg
        # masks). Falls back to the mesh's own name if it has no useful parent.
        parent = prim.GetParent()
        name = parent.GetName() if (parent and parent.IsValid() and parent.GetName()) else prim.GetName()
        if name not in label_names:
            label_names[name] = lid; lid += 1
        xyz_all.append(Pw)
        lab_all.append(np.full(len(Pw), label_names[name], np.int32))
    if not xyz_all:
        print("[pointcloud] no meshes found; skipping"); return
    xyz = np.concatenate(xyz_all); lab = np.concatenate(lab_all)
    if len(xyz) > n_target:
        idx = np.random.RandomState(0).choice(len(xyz), n_target, replace=False)
        xyz, lab = xyz[idx], lab[idx]
    # pseudo-colors per label
    rs = np.random.RandomState(42)
    palette = rs.randint(40, 230, size=(lab.max() + 1, 3), dtype=np.uint8)
    rgb = palette[lab]
    # write PLY (same format omni_to_4dgs._read_ply_xyz_rgb expects)
    _write_ply(os.path.join(out_dir, "points3D_gt.ply"), xyz, rgb, np)
    np.save(os.path.join(out_dir, "points3D_labels.npy"), lab)
    with open(os.path.join(out_dir, "label_names.json"), "w") as f:
        json.dump({v: k for k, v in label_names.items()}, f, indent=2)
    print(f"[pointcloud] {len(xyz)} points, {len(label_names)} instances -> points3D_gt.ply")


def _write_ply(path, xyz, rgb, np):
    xyz = np.asarray(xyz, np.float32); rgb = np.asarray(rgb, np.float32)
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property float red\nproperty float green\nproperty float blue\n"
              "end_header\n")
    data = np.concatenate([xyz, np.zeros_like(xyz), rgb], axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); f.write(data.tobytes())


if __name__ == "__main__":
    main()
