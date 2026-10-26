import json
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_orbit
import torch
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        use_camera_offsets=True,
        offsets_file=None
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.use_camera_offsets = use_camera_offsets
        self.offsets_file = offsets_file
        self._load_timing_and_offsets(cam_folder)
        self.image_paths, self.image_poses, self.image_times, self.image_cam_names = self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        self.camera_masks = self._load_motion_masks(cam_folder, cam_extrinsics)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)

    def _load_timing_and_offsets(self, cam_folder):
        """Load per-camera exposure offsets and drift from offsets.json (or optical_offsets.json).
        
        Anchors offsets to reference_camera (default cam01, delta=0.0).
        Evaluates physical time t_phys = (1 + drift) * (i / fps) + delta_k
        and normalized time t_norm = ((1 + drift) * i + delta_k * fps) / image_length.
        """
        self.fps = 20.0
        self.offsets_s = {}
        self.drift_s_per_s = {}
        self.has_offsets = False
        self.offsets_file_loaded = None
        self.reference_camera = "cam01"
        self.exposure_s = None

        env_disable = os.environ.get("PIPELINE_USE_CAMERA_OFFSETS", "").lower() in ("0", "false", "no")
        if not self.use_camera_offsets or env_disable:
            self.offsets_enabled = False
            return

        self.offsets_enabled = True
        candidates = []
        if self.offsets_file:
            candidates.append(self.offsets_file)
        else:
            candidates.append(os.path.join(cam_folder, "offsets.json"))
            candidates.append(os.path.join(cam_folder, "optical_offsets.json"))

        target_file = None
        for path in candidates:
            if path and os.path.exists(path):
                target_file = path
                break

        if not target_file:
            return

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[multipleview_dataset] Warning: failed to load offsets file {target_file}: {e}")
            return

        self.offsets_file_loaded = target_file
        self.fps = float(data.get("fps", 20.0))
        self.reference_camera = data.get("reference_camera", data.get("ref_camera", "cam01"))
        self.exposure_s = data.get("exposure_s", None)

        raw_offsets = dict(data.get("offsets_s", data.get("optical_offsets_s", {})))
        raw_drift = dict(data.get("drift_s_per_s", {}))

        if not raw_offsets and "cameras" in data and isinstance(data["cameras"], dict):
            for cname, cinfo in data["cameras"].items():
                if isinstance(cinfo, dict):
                    if "offset_s" in cinfo:
                        raw_offsets[cname] = cinfo["offset_s"]
                    elif "mean_offset_s" in cinfo:
                        raw_offsets[cname] = cinfo["mean_offset_s"]
                    if "drift_s_per_s" in cinfo:
                        raw_drift[cname] = cinfo["drift_s_per_s"]

        if not raw_offsets:
            return

        ref_offset = float(raw_offsets.get(self.reference_camera, 0.0))
        for cam, val in raw_offsets.items():
            self.offsets_s[cam] = float(val) - ref_offset
        for cam, val in raw_drift.items():
            self.drift_s_per_s[cam] = float(val)

        self.has_offsets = True

        
    def _load_motion_masks(self, cam_folder, cam_extrinsics):
        masks = {}
        mask_dirs = [
            os.path.join(cam_folder, "motion_masks"),
            os.path.join(cam_folder, "masks"),
        ]
        target_dir = None
        for md in mask_dirs:
            if os.path.isdir(md):
                target_dir = md
                break
        if target_dir is None:
            return masks

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            number = os.path.basename(extr.name)[5:-4]
            cam_name = "cam" + number.zfill(2)
            npy_path = os.path.join(target_dir, f"{cam_name}.npy")
            png_path = os.path.join(target_dir, f"{cam_name}.png")
            alt_png = os.path.join(target_dir, f"{cam_name}_mask.png")
            if os.path.exists(npy_path):
                arr = np.load(npy_path).astype(np.float32)
                masks[cam_name] = torch.from_numpy(arr)
            elif os.path.exists(png_path):
                img = Image.open(png_path).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                masks[cam_name] = torch.from_numpy(arr)
            elif os.path.exists(alt_png):
                img = Image.open(alt_png).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                masks[cam_name] = torch.from_numpy(arr)
        return masks

    def get_mask(self, index):
        if hasattr(self, "image_cam_names") and index < len(self.image_cam_names):
            cam_name = self.image_cam_names[index]
            return self.camera_masks.get(cam_name, None)
        return None

    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        cam01_path = os.path.join(cam_folder, "cam01")
        image_length = len(os.listdir(cam01_path)) if os.path.isdir(cam01_path) else 0
        self.image_length = image_length
        denom = float(image_length) if image_length > 0 else 1.0

        image_paths = []
        image_poses = []
        image_times = []
        image_cam_names = []
        image_physical_times = []

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = os.path.basename(extr.name)[5:-4]
            cam_name = "cam" + number.zfill(2)
            images_folder = os.path.join(cam_folder, cam_name)

            delta_k = self.offsets_s.get(cam_name, 0.0) if self.has_offsets else 0.0
            drift_k = self.drift_s_per_s.get(cam_name, 0.0) if self.has_offsets else 0.0

            image_range = range(image_length)
            if split == "test":
                image_range = [image_range[0], image_range[int(image_length / 3)], image_range[int(image_length * 2 / 3)]]

            for i in image_range:
                num = i + 1
                image_path = os.path.join(images_folder, "frame_" + str(num).zfill(5) + ".jpg")
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_cam_names.append(cam_name)

                # Physical seconds: (1 + drift) * (i / fps) + delta_k
                t_phys = (1.0 + drift_k) * (float(i) / self.fps) + delta_k
                # Normalized time: ((1 + drift) * i + delta_k * fps) / image_length
                t_norm = ((1.0 + drift_k) * float(i) + delta_k * self.fps) / denom

                image_physical_times.append(float(t_phys))
                image_times.append(float(t_norm))

        self.image_physical_times = image_physical_times
        return image_paths, image_poses, image_times, image_cam_names

    def get_physical_time(self, index):
        if hasattr(self, "image_physical_times") and index < len(self.image_physical_times):
            return self.image_physical_times[index]
        return None

    def get_camera_name(self, index):
        if hasattr(self, "image_cam_names") and index < len(self.image_cam_names):
            return self.image_cam_names[index]
        return None

    def get_camera_offset(self, cam_name):
        return self.offsets_s.get(cam_name, 0.0)

    def get_camera_drift(self, cam_name):
        return self.drift_s_per_s.get(cam_name, 0.0)

    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        # omniverse_pipeline captures are a static 360-degree ring/dome of cameras, not a
        # forward-facing array -> use a calm circular orbit for the test video instead of
        # get_spiral's LLFF-style heuristic (which degenerates into an erratic, fast-looking
        # path for this camera layout; see scene/neural_3D_dataset_NDC.get_orbit).
        val_poses = get_orbit(poses, N_views=N_views, n_rots=1)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]