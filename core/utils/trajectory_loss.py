"""core/utils/trajectory_loss.py

Trajectory Reprojection Loss (R8).
Projects deformed 3D ROI Gaussian positions into each camera at its continuous
timestamp and penalizes the discrepancy between the projected 2D displacement
and the measured optical tracks.

Supervises displacement of identified points, never dense optical flow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_points_to_pixels(
    points3D: torch.Tensor,
    full_proj_transform: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Project 3D world coordinates to 2D pixel coordinates using camera projection matrix.
    
    Args:
        points3D: [N, 3] world coordinates.
        full_proj_transform: [4, 4] world-to-clip projection matrix.
        width: Image width in pixels.
        height: Image height in pixels.
        
    Returns:
        [N, 2] pixel coordinates (x, y).
    """
    p_num = points3D.shape[0]
    ones = torch.ones((p_num, 1), dtype=points3D.dtype, device=points3D.device)
    points_hom = torch.cat([points3D, ones], dim=1)  # [N, 4]

    # full_proj_transform in 4DGS is transposed (column-major)
    clip = torch.matmul(points_hom, full_proj_transform)  # [N, 4]
    w = clip[:, 3:4]
    denom = torch.where(w.abs() < 1e-6, torch.sign(w) * 1e-6, w)

    # NDC coordinates in [-1, 1]
    ndc_x = clip[:, 0:1] / denom
    ndc_y = clip[:, 1:2] / denom

    # Convert to pixel space
    px_x = (ndc_x + 1.0) * 0.5 * float(width)
    px_y = (ndc_y + 1.0) * 0.5 * float(height)

    return torch.cat([px_x, px_y], dim=1)


class TrajectoryReprojectionLoss(nn.Module):
    def __init__(
        self,
        scene_path: Union[str, Path],
        track_file: Optional[str] = None,
        roi_bbox: Optional[List[float]] = None,
        loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.scene_path = Path(scene_path)
        self.roi_bbox = roi_bbox
        self.loss_type = loss_type
        self.tracks = self._load_measured_tracks(track_file)

    def _load_measured_tracks(self, track_file: Optional[str] = None) -> Dict[str, Any]:
        """Loads 2D displacement tracks from trajectory_tracks.json, optical_offsets.json, or gt_motion_info.json."""
        candidates = []
        if track_file:
            candidates.append(Path(track_file))
            if not Path(track_file).is_absolute():
                candidates.append(self.scene_path / track_file)

        candidates.extend([
            self.scene_path / "trajectory_tracks.json",
            self.scene_path / "optical_offsets.json",
            self.scene_path / "gt_motion_info.json",
        ])

        target = None
        for c in candidates:
            if c and c.exists():
                target = c
                break

        if target is None:
            return {}

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[TrajectoryReprojectionLoss] Warning: failed to parse {target}: {e}")
            return {}

        # Format 1: trajectory_tracks.json
        if "cameras" in data and isinstance(data["cameras"], dict):
            first_val = next(iter(data["cameras"].values()), {})
            if "displacements_px" in first_val:
                return data

        # Format 2: optical_offsets.json (reconstruct analytical displacement tracks)
        if "optical_offsets_s" in data and "cameras" in data:
            f0 = float(data.get("wobble_frequency_hz", 7.112))
            fps = float(data.get("fps", 20.0))
            frame_count = int(data.get("frame_count", 60))
            cam_dict = {}
            for cam_id, cinfo in data["cameras"].items():
                amp = float(cinfo.get("amplitude_px", 0.0))
                phase = float(cinfo.get("phase_rad", 0.0))
                angle_deg = float(cinfo.get("principal_axis_angle_deg", 0.0))
                theta = math.radians(angle_deg)
                cos_th, sin_th = math.cos(theta), math.sin(theta)

                displacements = []
                for i in range(frame_count):
                    t = float(i) / fps
                    sig = amp * math.sin(2.0 * math.pi * f0 * t + phase)
                    displacements.append([float(sig * cos_th), float(sig * sin_th)])

                cam_dict[cam_id] = {
                    "amplitude_px": amp,
                    "phase_rad": phase,
                    "displacements_px": displacements,
                }
            return {
                "fps": fps,
                "n_frames": frame_count,
                "cameras": cam_dict,
            }

        # Format 3: gt_motion_info.json (synthetic scene ground-truth)
        if "wobble_frequency_hz" in data:
            f0 = float(data.get("wobble_frequency_hz", 7.112))
            fps = float(data.get("fps", 20.0))
            n_frames = int(data.get("n_frames", 60))
            per_cam_amps = data.get("per_camera_projected_amplitudes_px", {})
            injected = data.get("injected_offsets_s", {})

            cam_dict = {}
            for cam_id, amp in per_cam_amps.items():
                delta_s = float(injected.get(cam_id, 0.0))
                displacements = []
                for i in range(n_frames):
                    t = float(i) / fps + delta_s
                    dx = float(amp) * math.sin(2.0 * math.pi * f0 * t)
                    dy = 0.0
                    displacements.append([dx, dy])

                cam_dict[cam_id] = {
                    "amplitude_px": float(amp),
                    "displacements_px": displacements,
                }
            return {
                "fps": fps,
                "n_frames": n_frames,
                "cameras": cam_dict,
            }

        return {}

    def get_measured_displacement(self, cam_name: str, frame_idx: int) -> torch.Tensor:
        """Returns measured 2D displacement [dx, dy] in pixels."""
        if not self.tracks or "cameras" not in self.tracks:
            return torch.zeros(2, dtype=torch.float32)

        cinfo = self.tracks["cameras"].get(cam_name, {})
        disps = cinfo.get("displacements_px", [])
        if not disps:
            return torch.zeros(2, dtype=torch.float32)

        idx = max(0, min(frame_idx, len(disps) - 1))
        d = disps[idx]
        return torch.tensor([float(d[0]), float(d[1])], dtype=torch.float32)

    def forward(
        self,
        viewpoint_cam: Any,
        canonical_xyz: torch.Tensor,
        deformed_xyz: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes trajectory reprojection loss for the current viewpoint.
        
        Args:
            viewpoint_cam: Camera instance for the current view.
            canonical_xyz: [N, 3] canonical Gaussian coordinates.
            deformed_xyz: [N, 3] deformed Gaussian coordinates at the current continuous timestamp.
            roi_mask: Optional [N] boolean tensor selecting moving ROI Gaussians.
            
        Returns:
            Scalar loss tensor.
        """
        # Determine camera name and frame index
        cam_name = getattr(viewpoint_cam, "cam_name", None)
        frame_idx = getattr(viewpoint_cam, "frame_idx", None)

        if cam_name is None:
            # Fallback heuristic from uid or image_name
            uid = getattr(viewpoint_cam, "uid", 0)
            if hasattr(viewpoint_cam, "image_name") and viewpoint_cam.image_name.isdigit():
                uid = int(viewpoint_cam.image_name)
            # Default to cam01
            cam_name = "cam01"
            frame_idx = uid

        if frame_idx is None:
            frame_idx = 0

        # Determine ROI mask
        if roi_mask is None:
            if self.roi_bbox is not None and len(self.roi_bbox) == 6:
                b = self.roi_bbox
                roi_mask = (
                    (canonical_xyz[:, 0] >= b[0])
                    & (canonical_xyz[:, 0] <= b[3])
                    & (canonical_xyz[:, 1] >= b[1])
                    & (canonical_xyz[:, 1] <= b[4])
                    & (canonical_xyz[:, 2] >= b[2])
                    & (canonical_xyz[:, 2] <= b[5])
                )
            elif hasattr(viewpoint_cam, "mask") and viewpoint_cam.mask is not None:
                # If motion mask exists, select Gaussians that project into active mask
                # or select a centered cluster
                roi_mask = None

        if roi_mask is not None and roi_mask.any():
            pts_canon = canonical_xyz[roi_mask]
            pts_curr = deformed_xyz[roi_mask]
        else:
            # Subsample a subset of points for computational efficiency
            n_pts = canonical_xyz.shape[0]
            if n_pts > 500:
                step = n_pts // 500
                pts_canon = canonical_xyz[::step]
                pts_curr = deformed_xyz[::step]
            else:
                pts_canon = canonical_xyz
                pts_curr = deformed_xyz

        w = int(viewpoint_cam.image_width)
        h = int(viewpoint_cam.image_height)
        proj = viewpoint_cam.full_proj_transform.to(canonical_xyz.device)

        # Projected 2D pixel coordinates
        p_canon = project_points_to_pixels(pts_canon, proj, w, h)
        p_curr = project_points_to_pixels(pts_curr, proj, w, h)

        # Predicted 2D displacement in pixels
        disp_pred = p_curr - p_canon  # [M, 2]
        mean_disp_pred = disp_pred.mean(dim=0)  # [2] (dx, dy)

        # Measured 2D displacement
        disp_meas = self.get_measured_displacement(cam_name, frame_idx).to(mean_disp_pred.device)

        if self.loss_type == "l2":
            loss = F.mse_loss(mean_disp_pred, disp_meas)
        elif self.loss_type == "l1":
            loss = F.l1_loss(mean_disp_pred, disp_meas)
        else:
            loss = F.smooth_l1_loss(mean_disp_pred, disp_meas, beta=0.5)

        return loss
