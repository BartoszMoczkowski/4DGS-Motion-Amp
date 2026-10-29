"""core/scene/camera_offsets.py

Centralized per-camera physical/normalized timestamp conversion and guarded
learnable timing offsets module (R6/R7, hardened in W1 / T49).

Time Convention (W1 Single Time Convention)
-------------------------------------------
1. Physical Reference Instant (`t_phys`, seconds):
   Each frame `(k, i)` has a physical exposure-reference instant `t_phys(k, i)` measured
   in seconds relative to the reference camera (`cam01`, frame `0` at `t_phys = 0.0 s`):
       t_phys(k, i) = (1 + drift_k) * (i / fps) + offset_k
   In the real RTSP/RTCP sidecar tables (`offsets.json`), all cameras share an identical
   1/100 s (10 ms) shutter exposure (`exposure_s = 0.01`, `mid_exposure_corrected = True`).
   Because the +5 ms mid-exposure shift (`exposure_s / 2`) is identical across all cameras,
   pairwise relative offsets `offset_k` are identical between sensor frame-start and
   mid-exposure conventions, and `exposure_s / 2` is never double-applied.
2. Normalized Model Time (`t_norm`, dimensionless in `[0, 1)` for canonical range):
   Conversion from `t_phys` to normalized 4DGS time happens through a single formula:
       t_norm(k, i) = (t_phys(k, i) * fps) / image_length
                    = ((1 + drift_k) * i + offset_k * fps) / image_length
   Training, held-out evaluation (`training_report`), offline rendering, and trajectory
   extraction all use this exact conversion.

Learnable Guardrails (W1 / E1b)
-------------------------------
1. Structural Anchor (`cam01`):
   `delta_offsets` is multiplied in the forward graph by a fixed binary buffer
   `anchor_mask` (with `0.0` at `anchor_idx` and `1.0` elsewhere) and guarded by a
   backward hook on `delta_offsets`. Thus `cam01` has identically zero gradient and
   remains at `0.0` by construction before, during, and after any optimizer step.
2. Quadratic Sidecar Prior:
       L_offset_prior = lambda_offset_prior * sum_{k != anchor} (delta_k / sigma_offset)^2
   penalizes deviations `delta_k` from the sidecar initialization `base_offsets[k]`.
3. Secondary Safety Clamp:
   `max_clamp_s` bounds total offset `base_offsets[k] + delta_k` to `[-max_clamp_s, +max_clamp_s]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


def compute_physical_time(
    frame_idx: Union[int, float, torch.Tensor],
    offset_s: Union[float, torch.Tensor],
    fps: float,
    drift_s_per_s: float = 0.0,
) -> Union[float, torch.Tensor]:
    """Compute physical timestamp in seconds relative to reference camera frame 0:
    t_phys = (1 + drift_s_per_s) * (frame_idx / fps) + offset_s.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    base_t = (1.0 + float(drift_s_per_s)) * (float(frame_idx) / float(fps))
    return base_t + offset_s


def physical_to_normalized_time(
    t_phys: Union[float, torch.Tensor],
    fps: float,
    image_length: int,
) -> Union[float, torch.Tensor]:
    """Convert physical time (seconds) to normalized 4DGS model time:
    t_norm = (t_phys * fps) / image_length.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    denom = float(image_length) if int(image_length) > 0 else 1.0
    return (t_phys * float(fps)) / denom


def compute_normalized_time(
    frame_idx: Union[int, float, torch.Tensor],
    offset_s: Union[float, torch.Tensor],
    fps: float,
    image_length: int,
    drift_s_per_s: float = 0.0,
) -> Union[float, torch.Tensor]:
    """Centralized conversion from `(frame_idx, offset_s, fps, image_length, drift_s_per_s)`
    to normalized model time `t_norm = ((1 + drift) * frame_idx + offset_s * fps) / image_length`.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    denom = float(image_length) if int(image_length) > 0 else 1.0
    return ((1.0 + float(drift_s_per_s)) * float(frame_idx) + offset_s * float(fps)) / denom


def resolve_camera_time(
    viewpoint_cam: Any,
    camera_offsets_module: Optional["CameraOffsets"] = None,
) -> Optional[torch.Tensor]:
    """Return differentiable/updated normalized time for `viewpoint_cam` when
    `camera_offsets_module` is active, or `None` to use `viewpoint_cam.time`.
    Used identically by training steps and `training_report` evaluation.
    """
    if camera_offsets_module is None:
        return None
    cam_name = getattr(viewpoint_cam, "cam_name", None)
    frame_idx = getattr(viewpoint_cam, "frame_idx", None)
    if cam_name is not None and frame_idx is not None:
        return camera_offsets_module.get_normalized_time(cam_name, int(frame_idx))
    return None


class CameraOffsets(nn.Module):
    def __init__(
        self,
        cam_names: List[str],
        init_offsets_s: Optional[Dict[str, float]] = None,
        reference_camera: str = "cam01",
        max_clamp_s: Optional[float] = 0.040,
        fps: float = 20.0,
        image_length: int = 60,
        drift_s_per_s: Optional[Dict[str, float]] = None,
        offset_prior_sigma_s: float = 0.005,
        lambda_offset_prior: float = 0.01,
        offsets_file_sha256: Optional[str] = None,
    ):
        super().__init__()
        if not cam_names:
            raise ValueError("cam_names must be a non-empty list of camera identifiers")
        if float(fps) <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        if float(offset_prior_sigma_s) <= 0.0:
            raise ValueError(f"offset_prior_sigma_s must be positive, got {offset_prior_sigma_s}")

        self.cam_names = list(cam_names)
        self.reference_camera = reference_camera
        self.max_clamp_s = float(max_clamp_s) if max_clamp_s is not None else None
        self.fps = float(fps)
        self.image_length = int(image_length) if image_length > 0 else 1
        self.drift_dict = dict(drift_s_per_s) if drift_s_per_s else {}
        self.offset_prior_sigma_s = float(offset_prior_sigma_s)
        self.lambda_offset_prior = float(lambda_offset_prior)
        self.offsets_file_sha256 = offsets_file_sha256

        # Name <-> Index mappings
        self.cam2idx = {name: idx for idx, name in enumerate(self.cam_names)}
        self.anchor_idx = self.cam2idx.get(reference_camera, 0)

        # Base offsets from offsets.json (fixed sidecar baseline, anchored at reference_camera)
        base = torch.zeros(len(self.cam_names), dtype=torch.float32)
        if init_offsets_s:
            ref_val = float(init_offsets_s.get(reference_camera, 0.0))
            for name, val in init_offsets_s.items():
                if name in self.cam2idx:
                    base[self.cam2idx[name]] = float(val) - ref_val
        base[self.anchor_idx] = 0.0
        self.register_buffer("base_offsets", base)

        # Structural anchor mask buffer: 0.0 at anchor_idx, 1.0 elsewhere
        anchor_mask = torch.ones(len(self.cam_names), dtype=torch.float32)
        anchor_mask[self.anchor_idx] = 0.0
        self.register_buffer("anchor_mask", anchor_mask)

        # Trainable residual delta offsets (initialized to 0.0)
        self.delta_offsets = nn.Parameter(torch.zeros(len(self.cam_names), dtype=torch.float32))
        self.delta_offsets.register_hook(self._zero_anchor_grad)

    def _zero_anchor_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook ensuring the reference camera gradient is identically zero."""
        return grad * self.anchor_mask

    def _resolve_idx_and_name(self, cam_name_or_idx: Union[str, int]) -> tuple[int, str]:
        if isinstance(cam_name_or_idx, str):
            idx = self.cam2idx.get(cam_name_or_idx, self.anchor_idx)
            name = cam_name_or_idx
        else:
            idx = int(cam_name_or_idx)
            name = self.cam_names[idx] if 0 <= idx < len(self.cam_names) else self.reference_camera
        return idx, name

    def get_masked_delta(self) -> torch.Tensor:
        """Return structurally anchored delta_offsets vector (`0.0` at `anchor_idx`)."""
        return self.delta_offsets * self.anchor_mask

    def get_offset(self, cam_name_or_idx: Union[str, int]) -> torch.Tensor:
        """Return differentiable effective offset in seconds for a camera.
        Anchor camera (`cam01`) is identically 0.0 by construction in the forward graph.
        """
        idx, _ = self._resolve_idx_and_name(cam_name_or_idx)
        masked_delta = self.get_masked_delta()
        raw = self.base_offsets[idx] + masked_delta[idx]
        if idx == self.anchor_idx:
            # Multiplying by anchor_mask[anchor_idx] == 0.0 keeps it in the graph with 0 gradient
            return raw * self.anchor_mask[idx]
        if self.max_clamp_s is not None and self.max_clamp_s > 0.0:
            return torch.clamp(raw, -self.max_clamp_s, self.max_clamp_s)
        return raw

    def get_physical_time(self, cam_name_or_idx: Union[str, int], frame_idx: int) -> torch.Tensor:
        """Compute continuous differentiable physical timestamp `t_phys` in seconds."""
        _, cam_name = self._resolve_idx_and_name(cam_name_or_idx)
        offset = self.get_offset(cam_name_or_idx)
        drift = self.drift_dict.get(cam_name, 0.0)
        return compute_physical_time(frame_idx, offset, self.fps, drift_s_per_s=drift)

    def get_normalized_time(self, cam_name_or_idx: Union[str, int], frame_idx: int) -> torch.Tensor:
        """Compute continuous differentiable normalized timestamp `t_norm`."""
        _, cam_name = self._resolve_idx_and_name(cam_name_or_idx)
        offset = self.get_offset(cam_name_or_idx)
        drift = self.drift_dict.get(cam_name, 0.0)
        return compute_normalized_time(
            frame_idx,
            offset,
            self.fps,
            self.image_length,
            drift_s_per_s=drift,
        )

    def compute_prior_loss(
        self,
        sigma_s: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Quadratic prior toward the sidecar initialization (`base_offsets`):
        L_offset_prior = weight * sum_{k != anchor} (delta_k / sigma_s)^2.
        """
        sig = float(sigma_s) if sigma_s is not None else self.offset_prior_sigma_s
        if sig <= 0.0:
            raise ValueError(f"sigma_s must be positive, got {sig}")
        w = float(weight) if weight is not None else self.lambda_offset_prior
        masked_delta = self.get_masked_delta()
        return w * torch.sum((masked_delta / sig) ** 2)

    @torch.no_grad()
    def clamp_and_anchor(self):
        """Secondary safety projection: zeroes anchor residual and clamps total offset to +/-max_clamp_s."""
        self.delta_offsets.mul_(self.anchor_mask)
        if self.max_clamp_s is not None and self.max_clamp_s > 0.0:
            total = self.base_offsets + self.delta_offsets
            clamped_total = torch.clamp(total, -self.max_clamp_s, self.max_clamp_s)
            self.delta_offsets.copy_(clamped_total - self.base_offsets)
            self.delta_offsets.mul_(self.anchor_mask)

    def get_init_offsets(self) -> Dict[str, float]:
        """Return dictionary of initial sidecar offsets in seconds."""
        with torch.no_grad():
            return {
                name: (0.0 if idx == self.anchor_idx else float(self.base_offsets[idx].item()))
                for name, idx in self.cam2idx.items()
            }

    def get_effective_offsets(self) -> Dict[str, float]:
        """Return dictionary of effective learned offsets in seconds."""
        out = {}
        with torch.no_grad():
            for name, idx in self.cam2idx.items():
                if idx == self.anchor_idx:
                    out[name] = 0.0
                else:
                    out[name] = float(self.get_offset(idx).item())
        return out

    def get_delta_offsets(self) -> Dict[str, float]:
        """Return dictionary of learned residual deltas (`effective - init`) in seconds."""
        init_map = self.get_init_offsets()
        eff_map = self.get_effective_offsets()
        return {
            name: (0.0 if idx == self.anchor_idx else float(eff_map[name] - init_map[name]))
            for name, idx in self.cam2idx.items()
        }

    def get_telemetry_dict(
        self,
        sigma_s: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Export full auditable telemetry dictionary for `learned_offsets.json` and run manifests."""
        with torch.no_grad():
            prior_val = float(self.compute_prior_loss(sigma_s=sigma_s, weight=weight).item())
        eff = self.get_effective_offsets()
        return {
            "reference_camera": self.reference_camera,
            "fps": self.fps,
            "image_length": self.image_length,
            "offsets_file_sha256": self.offsets_file_sha256,
            "offset_prior_sigma_s": float(sigma_s) if sigma_s is not None else self.offset_prior_sigma_s,
            "lambda_offset_prior": float(weight) if weight is not None else self.lambda_offset_prior,
            "max_offset_clamp_s": self.max_clamp_s,
            "prior_loss": prior_val,
            "init_offsets_s": self.get_init_offsets(),
            "current_offsets_s": eff,
            "delta_offsets_s": self.get_delta_offsets(),
            "drift_s_per_s": dict(self.drift_dict),
            # Top-level camera map for backward-compatible readers of learned_offsets.json
            **eff,
        }
