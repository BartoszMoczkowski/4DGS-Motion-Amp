"""core/scene/camera_offsets.py

Learnable per-camera continuous timing offsets module (R7).
Refines per-camera exposure offsets delta_k jointly with the 4DGS model.

Enforces mandatory guardrails:
1. cam01 anchor: reference camera offset is strictly fixed at 0.0.
2. Boundary clamp: total offset delta_k is strictly clamped to [-max_clamp_s, +max_clamp_s]
   (default +/-40 ms).
3. Dedicated optimizer group in core/train.py.
4. Continuous differentiable normalized timestamps:
   t_norm = ((1 + drift) * frame_idx + delta_k * fps) / image_length.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class CameraOffsets(nn.Module):
    def __init__(
        self,
        cam_names: List[str],
        init_offsets_s: Optional[Dict[str, float]] = None,
        reference_camera: str = "cam01",
        max_clamp_s: float = 0.040,
        fps: float = 20.0,
        image_length: int = 60,
        drift_s_per_s: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.cam_names = list(cam_names)
        self.reference_camera = reference_camera
        self.max_clamp_s = float(max_clamp_s)
        self.fps = float(fps)
        self.image_length = int(image_length) if image_length > 0 else 1
        self.drift_dict = dict(drift_s_per_s) if drift_s_per_s else {}

        # Name <-> Index mappings
        self.cam2idx = {name: idx for idx, name in enumerate(self.cam_names)}
        self.anchor_idx = self.cam2idx.get(reference_camera, 0)

        # Base offsets from offsets.json (fixed baseline)
        base = torch.zeros(len(self.cam_names), dtype=torch.float32)
        if init_offsets_s:
            for name, val in init_offsets_s.items():
                if name in self.cam2idx:
                    base[self.cam2idx[name]] = float(val)
        base[self.anchor_idx] = 0.0
        self.register_buffer("base_offsets", base)

        # Trainable residual delta offsets
        self.delta_offsets = nn.Parameter(torch.zeros(len(self.cam_names), dtype=torch.float32))

    def get_offset(self, cam_name_or_idx: Union[str, int]) -> torch.Tensor:
        """Returns the differentiable effective offset in seconds for a camera."""
        if isinstance(cam_name_or_idx, str):
            idx = self.cam2idx.get(cam_name_or_idx, 0)
        else:
            idx = int(cam_name_or_idx)

        if idx == self.anchor_idx:
            return torch.zeros((), device=self.delta_offsets.device, dtype=self.delta_offsets.dtype)

        raw = self.base_offsets[idx] + self.delta_offsets[idx]
        return torch.clamp(raw, -self.max_clamp_s, self.max_clamp_s)

    def get_normalized_time(self, cam_name_or_idx: Union[str, int], frame_idx: int) -> torch.Tensor:
        """Computes continuous differentiable normalized timestamp t_norm."""
        offset = self.get_offset(cam_name_or_idx)
        if isinstance(cam_name_or_idx, str):
            cam_name = cam_name_or_idx
        else:
            cam_name = self.cam_names[cam_name_or_idx] if cam_name_or_idx < len(self.cam_names) else "cam01"

        drift = self.drift_dict.get(cam_name, 0.0)
        denom = float(self.image_length) if self.image_length > 0 else 1.0
        t_norm = ((1.0 + drift) * float(frame_idx) + offset * self.fps) / denom
        return t_norm

    @torch.no_grad()
    def clamp_and_anchor(self):
        """Guardrail enforcement: strictly anchors cam01 at 0 and clamps total offset to +/-max_clamp_s."""
        self.delta_offsets[self.anchor_idx] = 0.0
        total = self.base_offsets + self.delta_offsets
        clamped_total = torch.clamp(total, -self.max_clamp_s, self.max_clamp_s)
        self.delta_offsets.copy_(clamped_total - self.base_offsets)
        self.delta_offsets[self.anchor_idx] = 0.0

    def get_effective_offsets(self) -> Dict[str, float]:
        """Returns dictionary of effective learned offsets in seconds."""
        out = {}
        with torch.no_grad():
            for name, idx in self.cam2idx.items():
                if idx == self.anchor_idx:
                    out[name] = 0.0
                else:
                    total = self.base_offsets[idx] + self.delta_offsets[idx]
                    clamped = torch.clamp(total, -self.max_clamp_s, self.max_clamp_s)
                    out[name] = float(clamped.item())
        return out
