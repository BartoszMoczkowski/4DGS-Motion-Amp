"""Vendored, verbatim copy of ``omniverse_pipeline/omni_to_4dgs.py``'s ``convert()`` and the
geometry/COLMAP-writer helpers it calls (2026-07-14 copy-in rework of T07; see
``pipeline.vendored``'s module docstring). Function bodies are byte-for-byte the reference
script's — no import changes needed here (the reference script's helpers have no cross-repo
imports of their own). The reference script's CLI/argparse/``_selftest``/``qvec_to_rotmat``
(only used by its own selftest, never by ``convert()``) are intentionally not ported — only what
``convert()`` actually calls is a production dependency of ``pipeline/stages/convert.py``.

Original docstring:

    omni_to_4dgs.py — convert an Omniverse/Isaac-Sim capture (produced by omni_capture.py)
    into the exact on-disk layout that 4DGS's `multipleview` loader expects.

    This module is PURE PYTHON (numpy + Pillow; plyfile optional) and has NO Isaac Sim
    dependency, so it runs anywhere and is unit-tested (see `--selftest`).

    --------------------------------------------------------------------------------
    Input  (a capture directory, see omni_capture.py):
        <capture>/
            cameras_gt.json                 # intrinsics + per-camera OpenCV cam->world 4x4
            camNN/rgb_XXXX.png              # synchronized RGB, one folder per camera
            points3D_gt.ply                 # (optional) mesh-sampled init cloud w/ colors+labels
            camNN/instance_segmentation_XXXX.png (+ *_mapping.json)   # (optional) GT masks

    Output (a 4DGS multipleview scene):
        <out>/data/multipleview/<name>/
            cam01/frame_00001.jpg ...       # renamed/re-encoded frames (1-indexed, jpg)
            sparse_/cameras.bin             # 1 shared PINHOLE camera (id=1)
            sparse_/images.bin              # image1.jpg .. imageN.jpg  (loader maps -> camNN)
            sparse_/points3D.bin
            points3D_multipleview.ply
            poses_bounds_multipleview.npy   # LLFF (N,17) — used only for the spiral test path
            gt_segmentation.npz             # (optional) per-init-point instance labels for MBS eval

    --------------------------------------------------------------------------------
    Conventions (verified against scene/colmap_loader.py + scene/multipleview_dataset.py):

    * cameras_gt.json stores each camera as an OpenCV camera-to-world matrix `c2w`
      (camera axes: +X right, +Y down, +Z forward/look).  Camera CENTER = c2w[:3,3].
    * COLMAP stores world-to-camera: qvec (w,x,y,z) + tvec, with X_cam = R_wc X_world + t.
      The 4DGS multipleview loader then does R = qvec2rotmat(qvec).T ; T = tvec.
    * Image names MUST be `image{N}.jpg` (N = 1-based camera index): the loader derives the
      camera folder as `cam` + name[5:-4].zfill(2).  So image1.jpg -> cam01, image12.jpg -> cam12.
    * All cameras are assumed to share intrinsics (loader reads cam_intrinsics[1] globally).
    * LLFF poses_bounds row (17) = 3x5 [ -up | right | back | center | (H,W,focal) ] flattened
      + [near, far].  (The loader remaps [c1,-c0,c2,c3] -> [right,up,back,center].)
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

# ----------------------------------------------------------------------------- #
#  Geometry helpers
# ----------------------------------------------------------------------------- #


def opencv_c2w_to_colmap_qt(c2w: np.ndarray):
    """OpenCV camera-to-world (4x4) -> COLMAP (qvec[w,x,y,z], tvec) world-to-camera."""
    c2w = np.asarray(c2w, dtype=np.float64)
    R_c2w = c2w[:3, :3]
    C = c2w[:3, 3]                      # camera center in world
    R_wc = R_c2w.T                      # world->camera rotation
    t = -R_wc @ C                       # world->camera translation
    return rotmat_to_qvec(R_wc), t


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion (w,x,y,z).  Matches colmap_loader.qvec2rotmat inverse."""
    R = np.asarray(R, dtype=np.float64)
    # Standard, numerically-stable conversion.
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]     # (w,x,y,z)
    if q[0] < 0:
        q = -q
    return q


def opencv_c2w_to_llff_row(c2w, H, W, focal, near, far) -> np.ndarray:
    """Build one LLFF poses_bounds row (17,) from an OpenCV c2w."""
    R_c2w = np.asarray(c2w, dtype=np.float64)[:3, :3]
    C = np.asarray(c2w, dtype=np.float64)[:3, 3]
    # OpenCV camera axes (world coords):
    right = R_c2w[:, 0]
    down = R_c2w[:, 1]
    fwd = R_c2w[:, 2]
    up = -down
    back = -fwd
    # LLFF stored layout: columns [ -up(=down), right, back, center ] + hwf column.
    m = np.stack([-up, right, back, C, np.array([H, W, focal])], axis=1)  # 3x5
    return np.concatenate([m.reshape(-1), [near, far]]).astype(np.float64)


# ----------------------------------------------------------------------------- #
#  COLMAP binary writers  (byte-for-byte compatible with scene/colmap_loader.py)
# ----------------------------------------------------------------------------- #


def write_cameras_bin(path, width, height, fx, fy, cx, cy):
    """Single shared PINHOLE camera, id=1 (loader indexes cam_intrinsics[1])."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))                       # num_cameras
        f.write(struct.pack("<iiQQ", 1, 1, int(width), int(height)))  # id, model_id=1(PINHOLE), w, h
        f.write(struct.pack("<dddd", float(fx), float(fy), float(cx), float(cy)))


def write_images_bin(path, cams):
    """cams: list of dicts with keys qvec(w,x,y,z), tvec(3,), name. camera_id fixed to 1."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))               # num_reg_images
        for i, c in enumerate(cams, start=1):
            q = c["qvec"]; t = c["tvec"]
            f.write(struct.pack("<i", i))                   # image_id
            f.write(struct.pack("<dddd", *[float(v) for v in q]))
            f.write(struct.pack("<ddd", *[float(v) for v in t]))
            f.write(struct.pack("<i", 1))                   # camera_id (shared)
            f.write(c["name"].encode("utf-8") + b"\x00")    # null-terminated name
            f.write(struct.pack("<Q", 0))                   # num_points2D = 0


def write_points3D_bin(path, xyz, rgb):
    """Minimal points3D.bin (rgb uint8, empty tracks)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q", i + 1))               # point3D_id
            f.write(struct.pack("<ddd", *xyz[i]))
            f.write(struct.pack("<BBB", *rgb[i]))
            f.write(struct.pack("<d", 0.0))                 # reprojection error
            f.write(struct.pack("<Q", 0))                   # track length = 0


# ----------------------------------------------------------------------------- #
#  PLY (matches scene/dataset_readers.storePly / fetchPly: x y z nx ny nz r g b, all f4)
# ----------------------------------------------------------------------------- #


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)               # fetchPly divides by 255
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


# ----------------------------------------------------------------------------- #
#  Main conversion
# ----------------------------------------------------------------------------- #


def _load_frames_list(cam_dir):
    imgs = sorted(fn for fn in os.listdir(cam_dir)
                  if fn.lower().endswith((".png", ".jpg", ".jpeg"))
                  and ("rgb" in fn.lower() or "frame" in fn.lower()))
    return imgs


def _nerfpp_radius(centers):
    """Same radius 4DGS/3DGS derive from cameras (scene/dataset_readers.getNerfppNorm):
    max distance from the mean camera center, *1.1. This directly sets `spatial_lr_scale`,
    which every position/deformation/grid learning rate is multiplied by."""
    centers = np.asarray(centers, dtype=np.float64)
    center = centers.mean(axis=0)
    dist = np.linalg.norm(centers - center, axis=1)
    return float(dist.max() * 1.1)


def convert(capture_dir, out_dir, name, default_near=0.1, default_far=1000.0, target_radius=4.0):
    from PIL import Image
    with open(os.path.join(capture_dir, "cameras_gt.json")) as f:
        gt = json.load(f)

    intr = gt["intrinsics"]
    W, H = int(intr["width"]), int(intr["height"])
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx = float(intr.get("cx", W / 2)); cy = float(intr.get("cy", H / 2))
    cameras = gt["cameras"]                     # list, one per camera, ordered
    n_cams = len(cameras)

    scene_dir = os.path.join(out_dir, "data", "multipleview", name)
    sparse_dir = os.path.join(scene_dir, "sparse_")
    os.makedirs(sparse_dir, exist_ok=True)

    # --- world-scale normalization ---
    # Omniverse stages carry their own unit ("meters_per_unit"); this converter used to write
    # raw stage-unit coordinates straight through. For a physically large scene (e.g. a
    # multi-meter machine captured in centimeters) that leaves camera centers thousands of
    # units from the scene center -> Scene.cameras_extent (`spatial_lr_scale`, see
    # scene/__init__.py + scene/dataset_readers.getNerfppNorm) becomes huge, and EVERY
    # position/deformation/grid learning rate in arguments/__init__.py is multiplied by it
    # (gaussian_model.training_setup). Those LR constants were tuned for COLMAP-scale scenes
    # (cameras_extent ~ O(1-10)); a huge spatial_lr_scale makes the very first optimizer step
    # on the (until-then-untouched) deformation/grid params explode -> nan loss right when the
    # fine stage engages them for the first time. Fix: convert to meters via `meters_per_unit`,
    # then uniformly rescale (translations, near/far, point cloud) so the scene's nerf++-style
    # camera radius lands at `target_radius` (pass --target-radius 0 to only apply the
    # meters_per_unit conversion, no extra normalization).
    meters_per_unit = float(gt.get("meters_per_unit", 1.0))
    raw_centers = np.array(
        [np.array(c["c2w"], dtype=np.float64).reshape(4, 4)[:3, 3] for c in cameras]
    ) * meters_per_unit
    radius_after_mpu = _nerfpp_radius(raw_centers)
    norm_factor = (target_radius / radius_after_mpu) if (target_radius and radius_after_mpu > 1e-9) else 1.0
    scale = meters_per_unit * norm_factor
    print(f"[scale] meters_per_unit={meters_per_unit} camera radius after unit-conversion="
          f"{radius_after_mpu:.4g} -> target_radius={target_radius} "
          f"(normalization x{norm_factor:.4g}, total scale x{scale:.6g})")
    with open(os.path.join(scene_dir, "scene_scale.json"), "w") as f:
        json.dump({
            "meters_per_unit": meters_per_unit,
            "target_radius": target_radius,
            "normalization_factor": norm_factor,
            "total_scale": scale,
            "note": "internal_units = physical_meters * total_scale; divide measured "
                    "internal-unit displacements/positions by total_scale to recover meters.",
        }, f, indent=2)

    # --- frames: camNN/rgb_XXXX.* -> camNN/frame_XXXXX.jpg (1-indexed) ---
    n_frames_per_cam = []
    for ci, cam in enumerate(cameras, start=1):
        cam_dir = os.path.join(capture_dir, cam.get("folder", f"cam{ci:02d}"))
        # omni_capture.py (Replicator BasicWriter) nests RGB frames under an "rgb"
        # subfolder alongside camera_params/instance_segmentation/semantic_segmentation.
        src_folder = os.path.join(cam_dir, "rgb") if os.path.isdir(os.path.join(cam_dir, "rgb")) else cam_dir
        dst_folder = os.path.join(scene_dir, f"cam{ci:02d}")
        os.makedirs(dst_folder, exist_ok=True)
        frames = _load_frames_list(src_folder)
        for fi, fn in enumerate(frames, start=1):
            im = Image.open(os.path.join(src_folder, fn)).convert("RGB")
            im.save(os.path.join(dst_folder, f"frame_{fi:05d}.jpg"), quality=95)
        n_frames_per_cam.append(len(frames))
    if len(set(n_frames_per_cam)) > 1:
        print(f"[WARN] cameras have differing frame counts {n_frames_per_cam}; "
              "4DGS multipleview assumes equal counts.", file=sys.stderr)

    # --- COLMAP sparse_ from GT poses ---
    colmap_cams = []
    llff_rows = []
    for ci, cam in enumerate(cameras, start=1):
        c2w = np.array(cam["c2w"], dtype=np.float64).reshape(4, 4)
        c2w[:3, 3] *= scale                     # world-scale normalization (see above)
        q, t = opencv_c2w_to_colmap_qt(c2w)
        colmap_cams.append({"qvec": q, "tvec": t, "name": f"image{ci}.jpg"})
        # near/far come from the capture in the same raw stage units as c2w -> scale them too;
        # the --near/--far CLI fallbacks are already meant for the converter's output space.
        near = float(cam["near"]) * scale if "near" in cam else default_near
        far = float(cam["far"]) * scale if "far" in cam else default_far
        llff_rows.append(opencv_c2w_to_llff_row(c2w, H, W, fx, near, far))

    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"), W, H, fx, fy, cx, cy)
    write_images_bin(os.path.join(sparse_dir, "images.bin"), colmap_cams)
    np.save(os.path.join(scene_dir, "poses_bounds_multipleview.npy"),
            np.stack(llff_rows).astype(np.float64))

    # --- init point cloud (+ optional GT labels) ---
    pts_src = os.path.join(capture_dir, "points3D_gt.ply")
    labels_src = os.path.join(capture_dir, "points3D_labels.npy")
    if os.path.exists(pts_src):
        xyz, rgb = _read_ply_xyz_rgb(pts_src)
        xyz = xyz * scale                       # keep the point cloud consistent with cameras
    else:
        print("[WARN] no points3D_gt.ply in capture; writing a tiny placeholder cloud. "
              "Provide a mesh-sampled cloud for real training.", file=sys.stderr)
        xyz = np.random.RandomState(0).uniform(-1, 1, size=(2000, 3))
        rgb = np.full((2000, 3), 128, np.uint8)
    write_ply(os.path.join(scene_dir, "points3D_multipleview.ply"), xyz, rgb / 255.0)
    write_points3D_bin(os.path.join(sparse_dir, "points3D.bin"), xyz, rgb)

    if os.path.exists(labels_src):
        labels = np.load(labels_src)
        np.savez(os.path.join(scene_dir, "gt_segmentation.npz"),
                 points=xyz.astype(np.float32), labels=labels)
        print(f"[ok] wrote gt_segmentation.npz ({len(np.unique(labels))} instances)")

    print(f"[done] {n_cams} cameras, {n_frames_per_cam} frames -> {scene_dir}")
    return scene_dir


def _read_ply_xyz_rgb(path):
    """Minimal binary/ascii PLY reader for x,y,z,(nx,ny,nz),red,green,blue."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip().split()[1]
        n = 0; props = []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        if fmt == b"binary_little_endian":
            arr = np.frombuffer(f.read(n * 4 * len(props)), "<f4").reshape(n, len(props))
        else:
            arr = np.array([f.readline().split() for _ in range(n)], dtype=np.float64)
    idx = {p: i for i, p in enumerate(props)}
    xyz = arr[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    if "red" in idx:
        rgb = arr[:, [idx["red"], idx["green"], idx["blue"]]]
        rgb = (rgb * 255 if rgb.max() <= 1.0 else rgb).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 128, np.uint8)
    return xyz, rgb
