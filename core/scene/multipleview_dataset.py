import hashlib
import json
import math
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_orbit
from scene.camera_offsets import compute_normalized_time, compute_physical_time
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
        offsets_file=None,
        strict_camera_offsets=False,
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height = cam_intrinsics[1].height
        width = cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.use_camera_offsets = use_camera_offsets
        self.offsets_file = offsets_file
        self.strict_camera_offsets = bool(strict_camera_offsets)
        self._load_timing_and_offsets(cam_folder, cam_extrinsics=cam_extrinsics)
        self.image_paths, self.image_poses, self.image_times, self.image_cam_names = (
            self.load_images_path(cam_folder, cam_extrinsics, cam_intrinsics, split)
        )
        self.camera_masks = self._load_motion_masks(cam_folder, cam_extrinsics)
        if split == "test":
            self.video_cam_infos = self.get_video_cam_infos(cam_folder)

    @staticmethod
    def _extract_cam_name(extr) -> str:
        number = os.path.basename(extr.name)[5:-4]
        return "cam" + number.zfill(2)

    def _load_timing_and_offsets(self, cam_folder, cam_extrinsics=None):
        """Load and validate per-camera exposure offsets and drift from offsets.json.

        Anchors offsets to reference_camera (default cam01, delta=0.0).
        Evaluates physical time t_phys = (1 + drift) * (i / fps) + delta_k
        and normalized time t_norm = ((1 + drift) * i + delta_k * fps) / image_length
        via the central helpers in `scene.camera_offsets`.

        Validation rules (W1 / T49):
        - Any present timing JSON must be valid JSON with finite `fps > 0`, finite
          `exposure_s > 0` (when present), non-empty `offsets_s`, finite offset/drift
          values, and full camera coverage for `cam_extrinsics`. Malformed files raise
          `ValueError` immediately (never ignored silently).
        - When `strict_camera_offsets=True` or `offsets_file` is explicitly provided,
          missing timing files raise `FileNotFoundError` and `optical_offsets.json`
          is never used as a fallback.
        """
        self.fps = 20.0
        self.offsets_s = {}
        self.drift_s_per_s = {}
        self.has_offsets = False
        self.offsets_file_loaded = None
        self.offsets_file_sha256 = None
        self.reference_camera = "cam01"
        self.exposure_s = None
        self.mid_exposure_corrected = None

        env_disable = os.environ.get("PIPELINE_USE_CAMERA_OFFSETS", "").lower() in (
            "0",
            "false",
            "no",
        )
        if not self.use_camera_offsets or env_disable:
            self.offsets_enabled = False
            return

        self.offsets_enabled = True
        target_file = None

        if self.offsets_file:
            candidate = (
                self.offsets_file
                if os.path.isabs(self.offsets_file)
                else os.path.join(cam_folder, self.offsets_file)
            )
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"[multipleview_dataset] Configured offsets_file does not exist: {candidate}"
                )
            target_file = candidate
        elif self.strict_camera_offsets:
            candidate = os.path.join(cam_folder, "offsets.json")
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"[multipleview_dataset] strict_camera_offsets=True requires {candidate} (no fallback allowed)"
                )
            target_file = candidate
        else:
            for path in (
                os.path.join(cam_folder, "offsets.json"),
                os.path.join(cam_folder, "optical_offsets.json"),
            ):
                if os.path.exists(path):
                    target_file = path
                    break

        if not target_file:
            return

        try:
            with open(target_file, "rb") as bf:
                raw_bytes = bf.read()
            self.offsets_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(
                f"[multipleview_dataset] Malformed JSON in timing file {target_file}: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"[multipleview_dataset] Timing file {target_file} must contain a JSON object"
            )

        raw_fps = data.get("fps", 20.0)
        try:
            fps_val = float(raw_fps)
        except (TypeError, ValueError) as e:
            raise ValueError(f"[multipleview_dataset] Invalid fps {raw_fps!r} in {target_file}") from e
        if not math.isfinite(fps_val) or fps_val <= 0.0:
            raise ValueError(
                f"[multipleview_dataset] fps must be finite and > 0 in {target_file}, got {fps_val}"
            )
        self.fps = fps_val

        self.reference_camera = str(
            data.get("reference_camera", data.get("ref_camera", "cam01"))
        )
        self.mid_exposure_corrected = data.get("mid_exposure_corrected", None)

        raw_exposure = data.get("exposure_s", None)
        if raw_exposure is not None:
            try:
                exp_val = float(raw_exposure)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"[multipleview_dataset] Invalid exposure_s {raw_exposure!r} in {target_file}"
                ) from e
            if not math.isfinite(exp_val) or exp_val <= 0.0:
                raise ValueError(
                    f"[multipleview_dataset] exposure_s must be finite and > 0 in {target_file}, got {exp_val}"
                )
            self.exposure_s = exp_val

        raw_offsets_obj = data.get("offsets_s", data.get("optical_offsets_s", {}))
        if not isinstance(raw_offsets_obj, dict):
            raise ValueError(
                f"[multipleview_dataset] offsets_s must be a dict in {target_file}"
            )
        raw_offsets = dict(raw_offsets_obj)

        raw_drift_obj = data.get("drift_s_per_s", {})
        if not isinstance(raw_drift_obj, dict):
            raise ValueError(
                f"[multipleview_dataset] drift_s_per_s must be a dict in {target_file}"
            )
        raw_drift = dict(raw_drift_obj)

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
            raise ValueError(
                f"[multipleview_dataset] Timing file {target_file} contains no camera offsets"
            )

        if self.reference_camera not in raw_offsets:
            raise ValueError(
                f"[multipleview_dataset] Reference camera {self.reference_camera!r} missing from offsets in {target_file}"
            )

        parsed_offsets = {}
        for cam, val in raw_offsets.items():
            try:
                fval = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"[multipleview_dataset] Non-numeric offset for {cam!r} in {target_file}: {val!r}"
                ) from e
            if not math.isfinite(fval):
                raise ValueError(
                    f"[multipleview_dataset] Non-finite offset for {cam!r} in {target_file}: {fval}"
                )
            parsed_offsets[cam] = fval

        parsed_drift = {}
        for cam, val in raw_drift.items():
            try:
                dval = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"[multipleview_dataset] Non-numeric drift for {cam!r} in {target_file}: {val!r}"
                ) from e
            if not math.isfinite(dval):
                raise ValueError(
                    f"[multipleview_dataset] Non-finite drift for {cam!r} in {target_file}: {dval}"
                )
            parsed_drift[cam] = dval

        if cam_extrinsics:
            expected_cams = [
                self._extract_cam_name(cam_extrinsics[k]) for k in cam_extrinsics
            ]
            missing_cams = [c for c in expected_cams if c not in parsed_offsets]
            if missing_cams:
                raise ValueError(
                    f"[multipleview_dataset] Timing file {target_file} is missing offsets for cameras: {missing_cams}"
                )

        ref_offset = parsed_offsets[self.reference_camera]
        for cam, fval in parsed_offsets.items():
            self.offsets_s[cam] = fval - ref_offset
        self.offsets_s[self.reference_camera] = 0.0
        for cam, dval in parsed_drift.items():
            self.drift_s_per_s[cam] = dval

        self.offsets_file_loaded = target_file
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
            cam_name = self._extract_cam_name(extr)
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

        image_paths = []
        image_poses = []
        image_times = []
        image_cam_names = []
        image_physical_times = []
        image_frame_indices = []

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            cam_name = self._extract_cam_name(extr)
            images_folder = os.path.join(cam_folder, cam_name)

            delta_k = self.offsets_s.get(cam_name, 0.0) if self.has_offsets else 0.0
            drift_k = self.drift_s_per_s.get(cam_name, 0.0) if self.has_offsets else 0.0

            image_range = range(image_length)
            if split == "test":
                image_range = [
                    image_range[0],
                    image_range[int(image_length / 3)],
                    image_range[int(image_length * 2 / 3)],
                ]

            for i in image_range:
                num = i + 1
                image_path = os.path.join(
                    images_folder, "frame_" + str(num).zfill(5) + ".jpg"
                )
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_cam_names.append(cam_name)

                t_phys = compute_physical_time(
                    i, delta_k, self.fps, drift_s_per_s=drift_k
                )
                t_norm = compute_normalized_time(
                    i, delta_k, self.fps, image_length, drift_s_per_s=drift_k
                )

                image_physical_times.append(float(t_phys))
                image_times.append(float(t_norm))
                image_frame_indices.append(int(i))

        self.image_physical_times = image_physical_times
        self.image_frame_indices = image_frame_indices
        self.cam_names = list(dict.fromkeys(image_cam_names))
        return image_paths, image_poses, image_times, image_cam_names

    def apply_offsets_dict(self, new_offsets_s: dict):
        """Update per-camera offsets and recompute all `image_times` and `image_physical_times`
        using the centralized time conversion helpers.
        """
        if not isinstance(new_offsets_s, dict):
            return
        ref_val = float(new_offsets_s.get(self.reference_camera, 0.0))
        for cam in self.cam_names:
            if cam in new_offsets_s and isinstance(new_offsets_s[cam], (int, float)):
                self.offsets_s[cam] = float(new_offsets_s[cam]) - ref_val
        self.offsets_s[self.reference_camera] = 0.0
        self.has_offsets = True

        for idx in range(len(self.image_times)):
            cam_name = self.image_cam_names[idx]
            frame_i = self.image_frame_indices[idx]
            delta_k = self.offsets_s.get(cam_name, 0.0)
            drift_k = self.drift_s_per_s.get(cam_name, 0.0)
            self.image_physical_times[idx] = float(
                compute_physical_time(frame_i, delta_k, self.fps, drift_s_per_s=drift_k)
            )
            self.image_times[idx] = float(
                compute_normalized_time(
                    frame_i, delta_k, self.fps, self.image_length, drift_s_per_s=drift_k
                )
            )

    def get_physical_time(self, index):
        if hasattr(self, "image_physical_times") and index < len(
            self.image_physical_times
        ):
            return self.image_physical_times[index]
        return None

    def get_camera_name(self, index):
        if hasattr(self, "image_cam_names") and index < len(self.image_cam_names):
            return self.image_cam_names[index]
        return None

    def get_frame_idx(self, index):
        if hasattr(self, "image_frame_indices") and index < len(
            self.image_frame_indices
        ):
            return self.image_frame_indices[index]
        return 0

    def get_camera_names(self):
        if hasattr(self, "cam_names"):
            return list(self.cam_names)
        return []

    def get_camera_offset(self, cam_name):
        return self.offsets_s.get(cam_name, 0.0)

    def get_camera_drift(self, cam_name):
        return self.drift_s_per_s.get(cam_name, 0.0)

    def get_video_cam_infos(self, datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_orbit(poses, N_views=N_views, n_rots=1)

        cameras = []
        len_poses = len(val_poses)
        times = [i / len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3, :] = p[:3, :]
            R = pose[:3, :3]
            R = -R
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.shape[2],
                    height=image.shape[1],
                    time=time,
                    mask=None,
                )
            )
        return cameras

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]

    def load_pose(self, index):
        return self.image_poses[index]
