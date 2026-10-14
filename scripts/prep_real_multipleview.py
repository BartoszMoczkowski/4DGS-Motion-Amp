"""
Prepares RTSP multi-camera captures (e.g. controler_test_01, hand_test_002)
into the standard 4DGS `multipleview` dataset layout.

Steps:
1. Link/copy frames into data/multipleview/<scene>/cam{01..10}/frame_{00001..N}.jpg
2. Run COLMAP calibration on static reference frame across all 10 cameras
3. Save sparse_/ (cameras.bin, images.bin, points3D.bin)
4. Generate points3D_multipleview.ply and poses_bounds_multipleview.npy
"""

import os
import sys
import shutil
import struct
import subprocess
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# COLMAP geometry helpers
def qvec2rotmat(qvec):
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

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            prop = read_next_bytes(fid, 24, "iiQQ")
            cam_id, model_id, width, height = prop[0], prop[1], prop[2], prop[3]
            # Simple radial has 4 params: f, cx, cy, k
            # PINHOLE has 4 params: fx, fy, cx, cy
            # We determine num_params based on model_id
            num_params_map = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
            num_params = num_params_map.get(model_id, 4)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {
                "id": cam_id, "model_id": model_id,
                "width": width, "height": height,
                "params": np.array(params)
            }
    return cameras

def read_images_binary(path):
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
                "id": image_id, "qvec": qvec, "tvec": tvec,
                "camera_id": camera_id, "name": image_name
            }
    return images

def read_points3D_binary(path):
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
                "id": point3D_id, "xyz": xyz, "rgb": rgb,
                "error": error, "image_ids": image_ids
            }
    return points3D

def write_ply(path, xyz, rgb):
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

def compute_poses_bounds(sparse_dir):
    """Computes LLFF poses_bounds_multipleview.npy from COLMAP sparse reconstruction."""
    cams = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    imgs = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    pts3d = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))

    # Shared camera
    cam = cams[list(cams.keys())[0]]
    h, w, f = cam["height"], cam["width"], cam["params"][0]
    hwf = np.array([h, w, f]).reshape([3, 1])

    # Sort images by name so perm is [image1.jpg, image2.jpg, ... image10.jpg]
    sorted_img_items = sorted(imgs.items(), key=lambda kv: int(''.join(filter(str.isdigit, kv[1]["name"]))))
    
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

    # Must switch to [-u, r, -t] from [r, -u, t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    pts_arr = np.array([p["xyz"] for p in pts3d.values()])
    vis_arr = []
    id_to_idx = {im_id: i for i, (im_id, _) in enumerate(sorted_img_items)}
    for p in pts3d.values():
        cams_vis = [0] * len(sorted_img_items)
        for im_id in p["image_ids"]:
            if im_id in id_to_idx:
                cams_vis[id_to_idx[im_id]] = 1
        vis_arr.append(cams_vis)
    vis_arr = np.array(vis_arr)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)

    save_arr = []
    for i in range(len(sorted_img_items)):
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        if len(zs) > 0:
            close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)
        else:
            close_depth, inf_depth = 0.1, 1000.0
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))

    return np.array(save_arr, dtype=np.float64)


def restructure_frames(src_synced_dir, dst_scene_dir):
    """
    Restructures synced/frame_XXXXXX/camYY.jpg into:
    dst_scene_dir/camYY/frame_XXXXX.jpg (1-indexed, 5 digits)
    """
    frame_dirs = sorted([d for d in os.listdir(src_synced_dir) if d.startswith("frame_") and os.path.isdir(os.path.join(src_synced_dir, d))])
    print(f"Found {len(frame_dirs)} synced frame folders.")

    for new_idx, fdir in enumerate(frame_dirs, start=1):
        src_frame_path = os.path.join(src_synced_dir, fdir)
        cam_files = [f for f in os.listdir(src_frame_path) if f.startswith("cam") and f.endswith(".jpg")]
        
        for cfile in cam_files:
            cam_name = os.path.splitext(cfile)[0] # e.g. cam01
            dst_cam_dir = os.path.join(dst_scene_dir, cam_name)
            os.makedirs(dst_cam_dir, exist_ok=True)
            
            dst_frame_file = os.path.join(dst_cam_dir, f"frame_{new_idx:05d}.jpg")
            src_file = os.path.join(src_frame_path, cfile)
            
            if not os.path.exists(dst_frame_file):
                try:
                    os.link(src_file, dst_frame_file)
                except OSError:
                    shutil.copyfile(src_file, dst_frame_file)

    print(f"Successfully structured frames in {dst_scene_dir}")


def run_colmap_calibration(ref_frame_dir, colmap_dir, colmap_bat):
    """
    Runs COLMAP calibration on 10 camera images from reference frame.
    """
    colmap_tmp = os.path.join(colmap_dir, "colmap_tmp")
    shutil.rmtree(colmap_tmp, ignore_errors=True)
    images_dir = os.path.join(colmap_tmp, "images")
    sparse_dir = os.path.join(colmap_tmp, "sparse")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    db_path = os.path.join(colmap_tmp, "database.db")

    # Name images as image1.jpg .. image10.jpg matching 4DGS convention
    for i in range(1, 11):
        src_img = os.path.join(ref_frame_dir, f"cam{i:02d}.jpg")
        dst_img = os.path.join(images_dir, f"image{i}.jpg")
        shutil.copyfile(src_img, dst_img)

    # 1. Feature extractor
    print("Running COLMAP feature extractor...")
    cmd1 = [
        colmap_bat, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL"
    ]
    subprocess.run(cmd1, check=True)

    # 2. Exhaustive matcher
    print("Running COLMAP exhaustive matcher...")
    cmd2 = [
        colmap_bat, "exhaustive_matcher",
        "--database_path", db_path
    ]
    subprocess.run(cmd2, check=True)

    # 3. Mapper
    print("Running COLMAP mapper...")
    cmd3 = [
        colmap_bat, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir
    ]
    subprocess.run(cmd3, check=True)

    model_dir = os.path.join(sparse_dir, "0")
    if not os.path.exists(model_dir):
        raise RuntimeError("COLMAP did not produce model sparse/0")

    return colmap_tmp, model_dir


def prepare_dataset(capture_name, colmap_source_capture=None):
    """
    Prepares a capture into data/multipleview/<capture_name>.
    If colmap_source_capture is provided, calibration is borrowed/triangulated from it.
    """
    capture_root = REPO_ROOT / "camera_sync" / "rtsp_capture" / "synced" / capture_name
    src_synced_dir = capture_root / "synced"
    dst_scene_dir = REPO_ROOT / "data" / "multipleview" / capture_name
    os.makedirs(dst_scene_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"Preparing Dataset: {capture_name}")
    print(f"==========================================")

    # 1. Frames restructuring
    restructure_frames(str(src_synced_dir), str(dst_scene_dir))

    sparse_target = dst_scene_dir / "sparse_"
    os.makedirs(sparse_target, exist_ok=True)
    colmap_bat = str(REPO_ROOT / "colmap" / "COLMAP.bat")

    # 2. COLMAP Calibration
    if colmap_source_capture:
        print(f"Using calibrated camera rig from {colmap_source_capture}...")
        source_sparse = REPO_ROOT / "data" / "multipleview" / colmap_source_capture / "sparse_"
        for f in ["cameras.bin", "images.bin", "points3D.bin"]:
            shutil.copyfile(source_sparse / f, sparse_target / f)
        
        # Triangulate points for this specific scene's frame 1
        ref_frame_dir = str(src_synced_dir / "frame_000001")
        colmap_tmp = dst_scene_dir / "colmap_tmp"
        shutil.rmtree(colmap_tmp, ignore_errors=True)
        images_dir = colmap_tmp / "images"
        os.makedirs(images_dir, exist_ok=True)
        for i in range(1, 11):
            shutil.copyfile(os.path.join(ref_frame_dir, f"cam{i:02d}.jpg"), images_dir / f"image{i}.jpg")

        db_path = colmap_tmp / "database.db"
        subprocess.run([colmap_bat, "feature_extractor", "--database_path", str(db_path), "--image_path", str(images_dir), "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL"], check=True)
        subprocess.run([colmap_bat, "exhaustive_matcher", "--database_path", str(db_path)], check=True)
        
        # Point triangulator with existing cameras
        tri_out = colmap_tmp / "triangulated"
        os.makedirs(tri_out, exist_ok=True)
        subprocess.run([colmap_bat, "point_triangulator", "--database_path", str(db_path), "--image_path", str(images_dir), "--input_path", str(sparse_target), "--output_path", str(tri_out), "--clear_points", "1"], check=True)

        if (tri_out / "points3D.bin").exists() and os.path.getsize(tri_out / "points3D.bin") > 100:
            shutil.copyfile(tri_out / "points3D.bin", sparse_target / "points3D.bin")
            print("Successfully triangulated new points3D.bin for this scene!")

    else:
        ref_frame_dir = str(src_synced_dir / "frame_000001")
        colmap_tmp, model_dir = run_colmap_calibration(ref_frame_dir, str(dst_scene_dir), colmap_bat)

        # Copy model 0 files to sparse_
        for f in ["cameras.bin", "images.bin", "points3D.bin"]:
            shutil.copyfile(os.path.join(model_dir, f), sparse_target / f)

    # 3. Export points3D_multipleview.ply
    pts3d = read_points3D_binary(sparse_target / "points3D.bin")
    xyz = np.array([p["xyz"] for p in pts3d.values()])
    rgb = np.array([p["rgb"] for p in pts3d.values()])
    ply_path = dst_scene_dir / "points3D_multipleview.ply"
    write_ply(str(ply_path), xyz, rgb)
    print(f"Wrote {len(xyz)} points to {ply_path}")

    # 4. Export poses_bounds_multipleview.npy
    poses_bounds = compute_poses_bounds(str(sparse_target))
    np_path = dst_scene_dir / "poses_bounds_multipleview.npy"
    np.save(str(np_path), poses_bounds)
    print(f"Wrote poses_bounds (shape {poses_bounds.shape}) to {np_path}")

    # Clean up temp colmap folder
    tmp_path = dst_scene_dir / "colmap_tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)

    print(f"[Done] Dataset {capture_name} is fully prepared for 4DGS training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=str, default="all", help="Dataset to prepare: hand_test_002, controler_test_01, or all")
    args = parser.parse_args()

    if args.capture in ["hand_test_002", "all"]:
        prepare_dataset("hand_test_002")

    if args.capture in ["controler_test_01", "all"]:
        # Use calibrated camera rig from hand_test_002 as both share the same physical rig setup
        prepare_dataset("controler_test_01", colmap_source_capture="hand_test_002")


if __name__ == "__main__":
    main()
