"""core/scene/fourier_rigid.py

Fourier-Rigid Deformation Head (R9).
Models periodic physical turbine wobble as a rigid body transform (R(t), T(t))
shared across all Gaussians within a defined Region of Interest (ROI).

Key Characteristics:
1. Low-dimensional physics parameterization: ~20 parameters instead of ~10^6 HexPlane decoders.
2. Translation T(t): multi-harmonic Fourier series along a learnable dominant axis (and secondary
   transverse axis), capturing fundamental wobble at ~7.11 Hz and optional harmonics (n=1..3).
3. Rotation R(t): small-angle rotation around a learnable pivot p in SO(3).
4. Fully auditable: exposes readable frequency, amplitude (mm), phase (deg), and dominant 3D axis.
5. ROI Gated & Additive: Gaussians outside the turbine ROI follow the standard HexPlane path;
   Gaussians inside the ROI follow the structured Fourier-rigid motion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graphics_utils import batch_quaternion_multiply


def rodrigues_rotation(points: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate 3D points about a unit axis by angle using Rodrigues formula.
    
    Args:
        points: [N, 3] points relative to pivot.
        axis: [1, 3] or [N, 3] unit rotation axis.
        angle: [1, 1] or [N, 1] rotation angle in radians.
        
    Returns:
        [N, 3] rotated points relative to pivot.
    """
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    # axis x points
    cross_prod = torch.cross(axis.expand_as(points), points, dim=-1)
    # axis . points
    dot_prod = (axis.expand_as(points) * points).sum(dim=-1, keepdim=True)
    
    # R * y = y * cos(theta) + (axis x y) * sin(theta) + axis * (axis . y) * (1 - cos(theta))
    rotated = (
        points * cos_theta
        + cross_prod * sin_theta
        + axis.expand_as(points) * dot_prod * (1.0 - cos_theta)
    )
    return rotated


class FourierRigidHead(nn.Module):
    def __init__(
        self,
        base_frequency_hz: float = 7.112,
        learn_frequency: bool = False,
        num_harmonics: int = 3,
        fps: float = 20.0,
        image_length: int = 60,
        roi_bbox: Optional[List[float]] = None,
        pure_rigid: bool = False,
        static_background: bool = False,
        init_dominant_axis: Optional[List[float]] = None,
        init_amplitude_m: float = 0.001,
        init_pivot: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_frequency_hz = float(base_frequency_hz)
        self.learn_frequency = bool(learn_frequency)
        self.num_harmonics = int(num_harmonics)
        self.fps = float(fps)
        self.image_length = int(image_length) if image_length > 0 else 1
        self.roi_bbox = list(roi_bbox) if roi_bbox is not None else None
        self.pure_rigid = bool(pure_rigid)
        # r14: when True, points OUTSIDE the ROI render at their canonical pose
        # (HexPlane output discarded), so the background is static by construction.
        # The HexPlane then receives no image gradient anywhere (inside ROI it is
        # replaced by the head, outside it is bypassed) and cannot bleed the ROI
        # oscillation scene-wide. Requires pure_rigid=True to make sense.
        self.static_background = bool(static_background)

        # 1. Frequency Parameter (fixed or learnable with strict clamping)
        if self.learn_frequency:
            self.freq_offset = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer("freq_offset", torch.zeros(1, dtype=torch.float32))

        # 2. Translation Parameters
        # Dominant axis u (unnormalized vector parameter)
        if init_dominant_axis is not None and len(init_dominant_axis) == 3:
            axis_init = torch.tensor(init_dominant_axis, dtype=torch.float32)
        else:
            axis_init = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        self.axis_param = nn.Parameter(axis_init)

        # Secondary transverse axis v (orthogonal to u)
        self.transverse_axis_param = nn.Parameter(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))

        # Harmonic amplitudes and phases along dominant axis
        amps = torch.zeros(self.num_harmonics, dtype=torch.float32)
        if self.num_harmonics > 0:
            amps[0] = float(init_amplitude_m)
        self.amplitudes = nn.Parameter(amps)
        self.phases = nn.Parameter(torch.zeros(self.num_harmonics, dtype=torch.float32))

        # Secondary (transverse) harmonic amplitudes and phases
        self.transverse_amplitudes = nn.Parameter(torch.zeros(self.num_harmonics, dtype=torch.float32))
        self.transverse_phases = nn.Parameter(torch.zeros(self.num_harmonics, dtype=torch.float32))

        # 3. Rotation Parameters
        # 3D Pivot point p
        if init_pivot is not None and len(init_pivot) == 3:
            pivot_init = torch.tensor(init_pivot, dtype=torch.float32)
        elif self.roi_bbox is not None and len(self.roi_bbox) == 6:
            b = self.roi_bbox
            pivot_init = torch.tensor([(b[0] + b[3]) * 0.5, (b[1] + b[4]) * 0.5, (b[2] + b[5]) * 0.5], dtype=torch.float32)
        else:
            pivot_init = torch.zeros(3, dtype=torch.float32)
        self.pivot = nn.Parameter(pivot_init)

        # Rotation axis w
        self.rot_axis_param = nn.Parameter(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        # Rotation angular harmonic amplitudes (in radians) and phases
        self.rot_amplitudes = nn.Parameter(torch.zeros(self.num_harmonics, dtype=torch.float32))
        self.rot_phases = nn.Parameter(torch.zeros(self.num_harmonics, dtype=torch.float32))

    @property
    def effective_frequency_hz(self) -> torch.Tensor:
        """Returns the effective frequency in Hz."""
        if self.learn_frequency:
            clamped_offset = torch.clamp(self.freq_offset, -0.5, 0.5)
            return self.base_frequency_hz + clamped_offset
        return torch.tensor(self.base_frequency_hz, device=self.freq_offset.device)

    def get_unit_axes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute orthonormal dominant and transverse translation axes, and rotation axis."""
        # 1. Dominant axis u
        u_norm = torch.norm(self.axis_param)
        u = self.axis_param / (u_norm + 1e-7)

        # 2. Transverse axis v orthogonalized to u
        v_proj = self.transverse_axis_param - (self.transverse_axis_param * u).sum() * u
        v_norm = torch.norm(v_proj)
        v = v_proj / (v_norm + 1e-7)

        # 3. Rotation axis w
        w_norm = torch.norm(self.rot_axis_param)
        w = self.rot_axis_param / (w_norm + 1e-7)

        return u, v, w

    def compute_translation_and_rotation(
        self,
        time_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes translation vector T(t), rotation axis w, and rotation angle theta(t).
        
        Args:
            time_input: Normalized time tensor [..., 1] in [0, 1] or physical seconds.
            
        Returns:
            translation: [M, 3] translation vectors in world units.
            rot_axis: [1, 3] unit rotation axis.
            rot_angle: [M, 1] rotation angle in radians.
        """
        # Convert normalized time to physical seconds if within [0, 1]
        t = time_input
        if t.max() <= 1.05 and self.fps > 0:
            duration_s = float(self.image_length) / self.fps
            t_sec = t * duration_s
        else:
            t_sec = t

        f = self.effective_frequency_hz
        u, v, w = self.get_unit_axes()

        # Reshape t_sec to [M, 1]
        if t_sec.dim() == 0:
            t_sec = t_sec.view(1, 1)
        elif t_sec.dim() == 1:
            t_sec = t_sec.unsqueeze(1)
        elif t_sec.dim() > 2:
            t_sec = t_sec.view(-1, 1)

        # Translation along dominant axis u and transverse axis v
        trans_u = torch.zeros_like(t_sec)
        trans_v = torch.zeros_like(t_sec)
        rot_angle = torch.zeros_like(t_sec)

        two_pi = 2.0 * math.pi
        for n in range(1, self.num_harmonics + 1):
            idx = n - 1
            omega_n = two_pi * float(n) * f
            angle_base = omega_n * t_sec

            # Translation components
            trans_u = trans_u + self.amplitudes[idx] * torch.sin(angle_base + self.phases[idx])
            trans_v = trans_v + self.transverse_amplitudes[idx] * torch.sin(angle_base + self.transverse_phases[idx])

            # Rotation components
            rot_angle = rot_angle + self.rot_amplitudes[idx] * torch.sin(angle_base + self.rot_phases[idx])

        # T = trans_u * u + trans_v * v (shape [M, 3])
        translation = trans_u * u.unsqueeze(0) + trans_v * v.unsqueeze(0)
        return translation, w.unsqueeze(0), rot_angle

    def compute_quaternion(self, rot_axis: torch.Tensor, rot_angle: torch.Tensor) -> torch.Tensor:
        """Construct rotation quaternion [w, x, y, z] from unit axis and angle."""
        half_theta = rot_angle * 0.5  # [M, 1]
        cos_half = torch.cos(half_theta)  # [M, 1]
        sin_half = torch.sin(half_theta)  # [M, 1]
        
        # [M, 3] = [M, 1] * [1, 3]
        xyz = sin_half * rot_axis
        q = torch.cat([cos_half, xyz], dim=-1)  # [M, 4]
        return F.normalize(q, dim=-1)

    def forward(
        self,
        means3D_canon: torch.Tensor,
        means3D_deformed: torch.Tensor,
        rotations_canon: torch.Tensor,
        rotations_deformed: torch.Tensor,
        time: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Fourier-rigid motion to Gaussians within the ROI.
        
        Args:
            means3D_canon: [N, 3] canonical coordinates.
            means3D_deformed: [N, 3] HexPlane-deformed coordinates.
            rotations_canon: [N, 4] canonical rotation quaternions (w, x, y, z).
            rotations_deformed: [N, 4] HexPlane-deformed rotation quaternions.
            time: [N, 1] or [1, 1] timestamp tensor.
            roi_mask: Optional [N] boolean mask selecting Gaussians in the turbine ROI.
            
        Returns:
            means3D_out: [N, 3] transformed Gaussian coordinates.
            rotations_out: [N, 4] transformed Gaussian quaternions.
        """
        # Determine ROI mask
        if roi_mask is None and self.roi_bbox is not None and len(self.roi_bbox) == 6:
            b = self.roi_bbox
            roi_mask = (
                (means3D_canon[:, 0] >= b[0])
                & (means3D_canon[:, 0] <= b[3])
                & (means3D_canon[:, 1] >= b[1])
                & (means3D_canon[:, 1] <= b[4])
                & (means3D_canon[:, 2] >= b[2])
                & (means3D_canon[:, 2] <= b[5])
            )

        # If no ROI points, return deformed inputs untouched
        if roi_mask is None or not roi_mask.any():
            return means3D_deformed, rotations_deformed

        # Compute rigid motion parameters at the given time
        # Extract scalar or per-frame time
        t_sample = time[roi_mask][0:1] if time.shape[0] == means3D_canon.shape[0] else time[0:1]
        translation, rot_axis, rot_angle = self.compute_translation_and_rotation(t_sample)

        # Extract ROI points
        pts_roi = means3D_canon[roi_mask]  # [M, 3]
        pivot = self.pivot.unsqueeze(0).to(pts_roi.device)

        # Relative to pivot
        pts_rel = pts_roi - pivot
        pts_rot = rodrigues_rotation(pts_rel, rot_axis.to(pts_roi.device), rot_angle.to(pts_roi.device))
        pts_rigid = pivot + pts_rot + translation.to(pts_roi.device)

        # Displacement from canonical
        disp_rigid = pts_rigid - pts_roi

        # Quaternion rotation
        q_rot = self.compute_quaternion(rot_axis.to(pts_roi.device), rot_angle.to(pts_roi.device))
        q_rot_expanded = q_rot.expand(pts_roi.shape[0], -1)

        # Clone tensors to prevent in-place gradient corruption
        means3D_out = means3D_deformed.clone()
        rotations_out = rotations_deformed.clone()

        if self.pure_rigid:
            # Strictly replace deformation inside ROI with Fourier-rigid motion
            means3D_out[roi_mask] = pts_rigid
            rotations_out[roi_mask] = batch_quaternion_multiply(q_rot_expanded, rotations_canon[roi_mask])
            if self.static_background:
                # r14: freeze everything outside the ROI at the canonical pose
                means3D_out[~roi_mask] = means3D_canon[~roi_mask]
                rotations_out[~roi_mask] = rotations_canon[~roi_mask]
        else:
            # Additive hybrid: HexPlane base deformation + Fourier-rigid motion
            means3D_out[roi_mask] = means3D_deformed[roi_mask] + disp_rigid
            rotations_out[roi_mask] = batch_quaternion_multiply(q_rot_expanded, rotations_deformed[roi_mask])

        return means3D_out, rotations_out

    # -----------------------------------------------------------------------
    # Auditing, Inspection, and Serialization
    # -----------------------------------------------------------------------

    def get_dominant_axis(self) -> List[float]:
        """Returns the normalized dominant translation axis [ux, uy, uz]."""
        with torch.no_grad():
            u, _, _ = self.get_unit_axes()
            return [float(x) for x in u.cpu().numpy()]

    def get_transverse_axis(self) -> List[float]:
        """Returns the normalized transverse translation axis [vx, vy, vz]."""
        with torch.no_grad():
            _, v, _ = self.get_unit_axes()
            return [float(x) for x in v.cpu().numpy()]

    def get_amplitudes_m(self) -> List[float]:
        """Returns harmonic amplitudes along the dominant axis in meters/world units."""
        with torch.no_grad():
            return [float(x) for x in self.amplitudes.cpu().numpy()]

    def get_phases_rad(self) -> List[float]:
        """Returns harmonic phases in radians."""
        with torch.no_grad():
            return [float(x) for x in self.phases.cpu().numpy()]

    def get_phases_deg(self) -> List[float]:
        """Returns harmonic phases in degrees."""
        with torch.no_grad():
            return [float(math.degrees(x)) for x in self.phases.cpu().numpy()]

    def get_transverse_amplitudes_m(self) -> List[float]:
        """Returns harmonic amplitudes along the transverse axis."""
        with torch.no_grad():
            return [float(x) for x in self.transverse_amplitudes.cpu().numpy()]

    def get_rotation_amplitudes_deg(self) -> List[float]:
        """Returns harmonic angular rotation amplitudes in degrees."""
        with torch.no_grad():
            return [float(math.degrees(x)) for x in self.rot_amplitudes.cpu().numpy()]

    def get_summary(self) -> Dict[str, Any]:
        """Returns an auditable summary dictionary of all learned motion physics parameters."""
        with torch.no_grad():
            freq = float(self.effective_frequency_hz.item())
            u, v, w = self.get_unit_axes()
            return {
                "frequency_hz": freq,
                "num_harmonics": self.num_harmonics,
                "dominant_axis": [float(x) for x in u.cpu().numpy()],
                "transverse_axis": [float(x) for x in v.cpu().numpy()],
                "rotation_axis": [float(x) for x in w.cpu().numpy()],
                "pivot": [float(x) for x in self.pivot.cpu().numpy()],
                "amplitudes_m": [float(x) for x in self.amplitudes.cpu().numpy()],
                "phases_deg": [float(math.degrees(x)) for x in self.phases.cpu().numpy()],
                "transverse_amplitudes_m": [float(x) for x in self.transverse_amplitudes.cpu().numpy()],
                "transverse_phases_deg": [float(math.degrees(x)) for x in self.transverse_phases.cpu().numpy()],
                "rotation_amplitudes_deg": [float(math.degrees(x)) for x in self.rot_amplitudes.cpu().numpy()],
                "pure_rigid": self.pure_rigid,
                "static_background": self.static_background,
            }

    def init_from_optical_tracks(self, track_file_or_data: Union[str, Path, Dict[str, Any]]):
        """Initializes frequency, dominant axis, and amplitude from optical tracks or optical_offsets.json."""
        if isinstance(track_file_or_data, (str, Path)):
            p = Path(track_file_or_data)
            if not p.exists():
                return
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return
        else:
            data = dict(track_file_or_data)

        # Base frequency
        if "wobble_frequency_hz" in data:
            self.base_frequency_hz = float(data["wobble_frequency_hz"])

        # Principal axis angle
        angle_deg = None
        if "principal_axis_angle_deg" in data:
            angle_deg = float(data["principal_axis_angle_deg"])
        elif "cameras" in data and isinstance(data["cameras"], dict):
            # Take average principal axis angle across cameras
            angles = [
                float(c["principal_axis_angle_deg"])
                for c in data["cameras"].values()
                if "principal_axis_angle_deg" in c
            ]
            if angles:
                angle_deg = float(np.mean(angles))

        if angle_deg is not None:
            rad = math.radians(angle_deg)
            # Default dominant axis in XY plane
            with torch.no_grad():
                self.axis_param.copy_(torch.tensor([math.cos(rad), math.sin(rad), 0.0], dtype=torch.float32))
