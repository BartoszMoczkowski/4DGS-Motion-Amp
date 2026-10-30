"""core/scene/deformation_dispatch.py

Unified deformation dispatch across all 4 call sites:
1. core/gaussian_renderer/__init__.py: render()
2. core/utils/render_utils.py: get_state_at_time()
3. core/motion_amp/renderer.py: render_mod()
4. core/train.py and orchestrator/pipeline/vendored/cuda/train.py

Guarantees exact parity between training-time deformation and offline
rendering, trajectory export, and motion amplification.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import torch


def dispatch_deformation(
    pc: Any,
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    time: torch.Tensor,
    stage: str = "fine",
    motion_components: Optional[Any] = None,
    fourier_head: Optional[Any] = None,
    pipe: Optional[Any] = None,
    iteration: Optional[int] = None,
    return_residual: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Dispatch Gaussian deformation through HexPlane and learned motion components.

    Args:
        pc: GaussianModel instance.
        means3D: Canonical Gaussian centers [N, 3].
        scales: Canonical or pre-activation scales [N, 3].
        rotations: Canonical or pre-activation rotations [N, 4].
        opacity: Canonical or pre-activation opacities [N, 1].
        shs: Spherical harmonics feature tensor.
        time: Timestamp tensor [N, 1] (or [1, 1]).
        stage: "coarse" or "fine".
        motion_components: Optional MotionComponents module (or retrieved from pc.motion_components).
        fourier_head: Optional legacy FourierRigidHead (fallback if motion_components is None).
        pipe: Pipeline parameters.
        iteration: Current training iteration (used for alpha ramp). None for inference (alpha=1).
        return_residual: Whether to also return the HexPlane residual h_i(t) = means3D_hex - means3D.

    Returns:
        (means3D_final, scales_final, rotations_final, opacity_final, shs_final)
        or with h_residual if return_residual=True.
    """
    if time.dim() == 2 and time.shape[0] != means3D.shape[0]:
        time = time.expand(means3D.shape[0], time.shape[1])

    h_residual = torch.zeros_like(means3D)

    if "coarse" in stage:
        if getattr(pipe, "gated_coarse", False):
            # Option B (R10): Gated coarse stage (legacy diagnostic mode)
            try:
                from gaussian_renderer import _resolve_roi_mask
                roi_mask = _resolve_roi_mask(pc, pipe, means3D)
            except Exception:
                roi_mask = getattr(pc, "_roi_mask", None)

            if roi_mask is not None and roi_mask.any():
                means3D_final = means3D.clone()
                scales_final = scales.clone()
                rotations_final = rotations.clone()
                opacity_final = opacity.clone()
                shs_final = shs.clone()
                t_in = time[roi_mask] if time.shape[0] == means3D.shape[0] else time
                m_def, s_def, r_def, o_def, sh_def = pc._deformation(
                    means3D[roi_mask],
                    scales[roi_mask],
                    rotations[roi_mask],
                    opacity[roi_mask],
                    shs[roi_mask],
                    t_in,
                )
                means3D_final[roi_mask] = m_def
                scales_final[roi_mask] = s_def
                rotations_final[roi_mask] = r_def
                opacity_final[roi_mask] = o_def
                shs_final[roi_mask] = sh_def
                h_residual[roi_mask] = m_def - means3D[roi_mask]
            else:
                means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
                    pc._deformation(means3D, scales, rotations, opacity, shs, time)
                )
                h_residual = means3D_final - means3D
        else:
            means3D_final = means3D
            scales_final = scales
            rotations_final = rotations
            opacity_final = opacity
            shs_final = shs
    elif "fine" in stage:
        means3D_hex, scales_hex, rotations_hex, opacity_hex, shs_hex = pc._deformation(
            means3D, scales, rotations, opacity, shs, time
        )
        h_residual = means3D_hex - means3D
        means3D_final = means3D_hex
        scales_final = scales_hex
        rotations_final = rotations_hex
        opacity_final = opacity_hex
        shs_final = shs_hex

        # 1. Active Motion Components (W4 / T52)
        active_mc = motion_components if motion_components is not None else getattr(pc, "motion_components", None)
        if active_mc is not None and getattr(active_mc, "is_active", True):
            means3D_final, rotations_final = active_mc.apply_deformation(
                pc=pc,
                means3D_canon=means3D,
                means3D_hex=means3D_hex,
                rotations_hex=rotations_hex,
                time=time,
                iteration=iteration,
            )
        # 2. Legacy Fourier Rigid Head fallback (Phase 12-13 R9)
        elif fourier_head is not None or getattr(pc, "fourier_head", None) is not None:
            active_fh = fourier_head if fourier_head is not None else getattr(pc, "fourier_head", None)
            try:
                from gaussian_renderer import _resolve_roi_mask
                roi_mask = _resolve_roi_mask(pc, pipe, means3D)
            except Exception:
                roi_mask = getattr(pc, "_roi_mask", None)
            means3D_final, rotations_final = active_fh(
                means3D_canon=means3D,
                means3D_deformed=means3D_final,
                rotations_canon=rotations,
                rotations_deformed=rotations_final,
                time=time,
                roi_mask=roi_mask,
            )
    else:
        raise NotImplementedError(f"Unknown stage: {stage}")

    if return_residual:
        return means3D_final, scales_final, rotations_final, opacity_final, shs_final, h_residual
    return means3D_final, scales_final, rotations_final, opacity_final, shs_final

