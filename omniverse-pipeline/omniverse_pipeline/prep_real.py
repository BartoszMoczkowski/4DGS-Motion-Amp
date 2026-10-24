"""Prepares RTSP multi-camera captures into standard 4DGS multipleview dataset format (T32).

1. Restructures synchronized multi-cam frames into ``cam{01..10}/frame_{00001..N}.jpg``
2. Runs or imports COLMAP calibration on static reference frame across all cameras
3. Idempotently writes ``sparse_/`` (cameras.bin, images.bin, points3D.bin)
4. Exports ``points3D_multipleview.ply`` and ``poses_bounds_multipleview.npy``
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path: str | Path) -> dict[int, dict]:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            prop = read_next_bytes(fid, 24, "iiQQ")
            cam_id, model_id, width, height = prop[0], prop[1], prop[2], prop[3]
            num_params_map = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
            num_params = num_params_map.get(model_id, 4)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {
                "id": cam_id,
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path: str | Path) -> dict[int, dict]:
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            prop = read_next_bytes(fid, 64, "idddddddi")
            image_id = prop[0]
            qvec = np.array(prop[1:5])
            tvec = np.array(prop[5:8])
            camera_id = prop[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            _ = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
            }
    return images


def read_points3D_binary(path: str | Path) -> dict[int, dict]:
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            prop = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = prop[0]
            xyz = np.array(prop[1:4])
            rgb = np.array(prop[4:7])
            error = prop[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            points3D[point3D_id] = {
                "id": point3D_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": image_ids,
            }
    return points3D


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
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


def compute_poses_bounds(sparse_dir: str | Path) -> np.ndarray:
    """Computes LLFF poses_bounds_multipleview.npy from COLMAP sparse reconstruction."""
    sparse_path = Path(sparse_dir)
    cams = read_cameras_binary(sparse_path / "cameras.bin")
    imgs = read_images_binary(sparse_path / "images.bin")
    pts3d = read_points3D_binary(sparse_path / "points3D.bin")

    if not cams:
        raise ValueError(f"No cameras in {sparse_path / 'cameras.bin'}")
    if not imgs:
        raise ValueError(f"No images in {sparse_path / 'images.bin'}")

    cam = cams[list(cams.keys())[0]]
    h, w, f = cam["height"], cam["width"], cam["params"][0]
    hwf = np.array([h, w, f]).reshape([3, 1])

    # Sort images by number extracted from name: e.g. image1.jpg .. image10.jpg
    sorted_img_items = sorted(
        imgs.items(),
        key=lambda kv: int("".join(filter(str.isdigit, kv[1]["name"])) or "0"),
    )

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    for _, im in sorted_img_items:
        R = qvec2rotmat(im["qvec"])
        t = im["tvec"].reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    # Convert coordinates to standard LLFF: [-u, r, -t]
    poses = np.concatenate(
        [poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1
    )

    pts_arr = np.array([p["xyz"] for p in pts3d.values()]) if pts3d else np.zeros((0, 3))
    vis_arr = []
    id_to_idx = {im_id: i for i, (im_id, _) in enumerate(sorted_img_items)}
    for p in pts3d.values():
        cams_vis = [0] * len(sorted_img_items)
        for im_id in p["image_ids"]:
            if im_id in id_to_idx:
                cams_vis[id_to_idx[im_id]] = 1
        vis_arr.append(cams_vis)
    vis_arr = np.array(vis_arr) if vis_arr else np.zeros((0, len(sorted_img_items)))

    if len(pts_arr) > 0 and len(vis_arr) > 0:
        zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    else:
        zvals = np.zeros((0, len(sorted_img_items)))

    save_arr = []
    for i in range(len(sorted_img_items)):
        if len(zvals) > 0:
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis == 1]
            if len(zs) > 0:
                close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)
            else:
                close_depth, inf_depth = 0.1, 1000.0
        else:
            close_depth, inf_depth = 0.1, 1000.0
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))

    return np.array(save_arr, dtype=np.float64)


def restructure_frames(src_synced_dir: Path, dst_scene_dir: Path) -> int:
    """Restructures synced/frame_XXXXXX/camYY.jpg into dst_scene_dir/camYY/frame_XXXXX.jpg."""
    frame_dirs = sorted([
        d for d in src_synced_dir.iterdir()
        if d.is_dir() and (d.name.startswith("frame_") or d.name.isdigit())
    ])
    if not frame_dirs:
        # Check if already restructured
        existing_cams = [d for d in dst_scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam")]
        if existing_cams:
            return len(list(existing_cams[0].glob("frame_*.jpg")))
        raise FileNotFoundError(f"No frame folders found under {src_synced_dir}")

    total_frames = len(frame_dirs)
    for new_idx, fdir in enumerate(frame_dirs, start=1):
        cam_files = [f for f in fdir.iterdir() if f.is_file() and f.name.startswith("cam") and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for cfile in cam_files:
            cam_name = cfile.stem.split("_")[0] if "_" in cfile.stem else cfile.stem
            # Normalize to cam01..cam10
            digits = "".join(filter(str.isdigit, cam_name))
            if digits:
                cam_name = f"cam{int(digits):02d}"

            dst_cam_dir = dst_scene_dir / cam_name
            dst_cam_dir.mkdir(parents=True, exist_ok=True)

            dst_frame_file = dst_cam_dir / f"frame_{new_idx:05d}.jpg"
            if not dst_frame_file.exists() or dst_frame_file.stat().st_size == 0:
                try:
                    os.link(cfile, dst_frame_file)
                except OSError:
                    shutil.copyfile(cfile, dst_frame_file)

    return total_frames


def find_colmap_binary(explicit_bin: Optional[str | Path] = None) -> str:
    """Finds COLMAP executable on PATH or in repo directory."""
    if explicit_bin:
        p = Path(explicit_bin)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Specified COLMAP binary not found: {explicit_bin}")

    # Check repo-root / colmap / COLMAP.bat
    repo_root = Path(__file__).resolve().parents[2]
    repo_bat = repo_root / "colmap" / "COLMAP.bat"
    if repo_bat.is_file():
        return str(repo_bat.resolve())

    # Check PATH
    for name in ("colmap.bat", "COLMAP.bat", "colmap.exe", "colmap"):
        found = shutil.which(name)
        if found:
            return str(found)

    raise FileNotFoundError(
        "COLMAP executable not found. Please ensure COLMAP is on PATH, in colmap/COLMAP.bat, "
        "or pass --colmap-bin <path>."
    )


def run_colmap_calibration(
    ref_frame_dir: Path,
    dst_scene_dir: Path,
    colmap_bin: str,
    camera_model: str = "SIMPLE_RADIAL",
) -> Path:
    """Runs COLMAP feature extraction, matching, and mapping on 10 camera views."""
    colmap_tmp = dst_scene_dir / "colmap_tmp"
    shutil.rmtree(colmap_tmp, ignore_errors=True)
    images_dir = colmap_tmp / "images"
    sparse_dir = colmap_tmp / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    db_path = colmap_tmp / "database.db"

    # Copy / name images image1.jpg .. imageN.jpg matching 4DGS conventions
    cam_files = sorted([f for f in ref_frame_dir.iterdir() if f.is_file() and f.name.startswith("cam")])
    for i, src_img in enumerate(cam_files, start=1):
        shutil.copyfile(src_img, images_dir / f"image{i}.jpg")

    # 1. Feature extractor
    subprocess.run([
        colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
    ], check=True)

    # 2. Matcher
    subprocess.run([
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(db_path),
    ], check=True)

    # 3. Mapper
    subprocess.run([
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], check=True)

    model_dir = sparse_dir / "0"
    if not model_dir.is_dir():
        raise RuntimeError(f"COLMAP mapper did not produce sparse/0 in {sparse_dir}")

    sparse_target = dst_scene_dir / "sparse_"
    sparse_target.mkdir(parents=True, exist_ok=True)
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        src_f = model_dir / fname
        if src_f.exists():
            shutil.copyfile(src_f, sparse_target / fname)

    shutil.rmtree(colmap_tmp, ignore_errors=True)
    return sparse_target


def prep_real_dataset(
    capture_dir: str | Path,
    out_name_or_dir: str | Path,
    colmap_source: Optional[str | Path] = None,
    camera_model: str = "SIMPLE_RADIAL",
    colmap_bin: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Prepares synced capture directory into 4DGS multipleview layout idempotently."""
    cap_path = Path(capture_dir).resolve()
    if not cap_path.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {cap_path}")

    # Determine src_synced_dir
    if (cap_path / "synced").is_dir():
        src_synced_dir = cap_path / "synced"
    else:
        src_synced_dir = cap_path

    # Determine dst_scene_dir
    repo_root = Path(__file__).resolve().parents[2]
    out_str = str(out_name_or_dir)
    if os.path.sep in out_str or "/" in out_str:
        dst_scene_dir = Path(out_str).resolve()
    else:
        dst_scene_dir = repo_root / "data" / "multipleview" / out_str

    dst_scene_dir.mkdir(parents=True, exist_ok=True)
    sparse_target = dst_scene_dir / "sparse_"
    sparse_target.mkdir(parents=True, exist_ok=True)

    # 1. Restructure frames
    n_frames = restructure_frames(src_synced_dir, dst_scene_dir)

    ply_path = dst_scene_dir / "points3D_multipleview.ply"
    npy_path = dst_scene_dir / "poses_bounds_multipleview.npy"

    # Check idempotency: if already calibrated and not force, skip COLMAP
    colmap_done = (
        not force
        and (sparse_target / "cameras.bin").exists()
        and (sparse_target / "images.bin").exists()
        and (sparse_target / "points3D.bin").exists()
        and ply_path.exists()
        and npy_path.exists()
    )

    if not colmap_done:
        bin_path = find_colmap_binary(colmap_bin)

        if colmap_source:
            source_sparse = None
            src_p = Path(colmap_source)
            if (src_p / "sparse_").is_dir():
                source_sparse = src_p / "sparse_"
            elif (src_p / "cameras.bin").exists():
                source_sparse = src_p
            elif (repo_root / "data" / "multipleview" / colmap_source / "sparse_").is_dir():
                source_sparse = repo_root / "data" / "multipleview" / colmap_source / "sparse_"

            if source_sparse is None or not (source_sparse / "cameras.bin").exists():
                raise FileNotFoundError(f"Could not locate COLMAP calibration in colmap-source: {colmap_source}")

            for f in ("cameras.bin", "images.bin", "points3D.bin"):
                shutil.copyfile(source_sparse / f, sparse_target / f)

            # Optional triangulation for reference frame
            ref_dirs = sorted([d for d in src_synced_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])
            if ref_dirs:
                ref_frame = ref_dirs[0]
                colmap_tmp = dst_scene_dir / "colmap_tmp"
                shutil.rmtree(colmap_tmp, ignore_errors=True)
                images_dir = colmap_tmp / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                cam_files = sorted([f for f in ref_frame.iterdir() if f.is_file() and f.name.startswith("cam")])
                for i, c_f in enumerate(cam_files, start=1):
                    shutil.copyfile(c_f, images_dir / f"image{i}.jpg")

                db_path = colmap_tmp / "database.db"
                try:
                    subprocess.run([
                        bin_path, "feature_extractor",
                        "--database_path", str(db_path),
                        "--image_path", str(images_dir),
                        "--ImageReader.single_camera", "1",
                        "--ImageReader.camera_model", camera_model,
                    ], check=True)
                    subprocess.run([bin_path, "exhaustive_matcher", "--database_path", str(db_path)], check=True)
                    tri_out = colmap_tmp / "triangulated"
                    tri_out.mkdir(parents=True, exist_ok=True)
                    subprocess.run([
                        bin_path, "point_triangulator",
                        "--database_path", str(db_path),
                        "--image_path", str(images_dir),
                        "--input_path", str(sparse_target),
                        "--output_path", str(tri_out),
                        "--clear_points", "1",
                    ], check=True)

                    if (tri_out / "points3D.bin").exists():
                        pts_tri = read_points3D_binary(tri_out / "points3D.bin")
                        if len(pts_tri) >= 200:
                            shutil.copyfile(tri_out / "points3D.bin", sparse_target / "points3D.bin")
                except Exception:
                    pass  # Keep baseline points3D.bin
                finally:
                    shutil.rmtree(colmap_tmp, ignore_errors=True)
        else:
            ref_dirs = sorted([d for d in src_synced_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])
            if not ref_dirs:
                raise FileNotFoundError(f"No frame_XXXXXX directories found in {src_synced_dir}")
            run_colmap_calibration(ref_dirs[0], dst_scene_dir, bin_path, camera_model=camera_model)

        # 3. Export points3D_multipleview.ply
        pts3d = read_points3D_binary(sparse_target / "points3D.bin")
        if pts3d:
            xyz = np.array([p["xyz"] for p in pts3d.values()])
            rgb = np.array([p["rgb"] for p in pts3d.values()])
        else:
            xyz = np.zeros((0, 3), dtype=np.float32)
            rgb = np.zeros((0, 3), dtype=np.float32)
        write_ply(ply_path, xyz, rgb)

        # 4. Export poses_bounds_multipleview.npy
        poses_bounds = compute_poses_bounds(sparse_target)
        np.save(npy_path, poses_bounds)

    return dst_scene_dir
