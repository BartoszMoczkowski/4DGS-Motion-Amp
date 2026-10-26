from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                res = self.dataset[index]
                if isinstance(res, tuple) and len(res) == 4:
                    image, w2c, time, mask = res
                else:
                    image, w2c, time = res
                    mask = getattr(self.dataset, "get_mask", lambda idx: None)(index)
                R, T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            physical_time = getattr(self.dataset, "get_physical_time", lambda idx: None)(index)
            cam_name = getattr(self.dataset, "get_camera_name", lambda idx: None)(index)
            frame_idx = getattr(self.dataset, "get_frame_idx", lambda idx: 0)(index)
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, physical_time=physical_time,
                              cam_name=cam_name, frame_idx=frame_idx)
        else:
            return self.dataset[index]

    @property
    def fps(self):
        return getattr(self.dataset, "fps", 20.0)

    @property
    def offsets_s(self):
        return getattr(self.dataset, "offsets_s", {})

    @property
    def has_offsets(self):
        return getattr(self.dataset, "has_offsets", False)

    @property
    def drift_s_per_s(self):
        return getattr(self.dataset, "drift_s_per_s", {})

    def get_physical_time(self, index):
        getter = getattr(self.dataset, "get_physical_time", None)
        if getter is not None:
            return getter(index)
        return None

    def get_camera_name(self, index):
        getter = getattr(self.dataset, "get_camera_name", None)
        if getter is not None:
            return getter(index)
        return None

    def get_frame_idx(self, index):
        getter = getattr(self.dataset, "get_frame_idx", None)
        if getter is not None:
            return getter(index)
        return 0

    def get_camera_names(self):
        getter = getattr(self.dataset, "get_camera_names", None)
        if getter is not None:
            return getter()
        return []

    def __len__(self):
        return len(self.dataset)


