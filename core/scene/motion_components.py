"""core/scene/motion_components.py

Fixed-K Soft Learned Motion Components (W4 / T52).
Discovers shared low-dimensional rigid/periodic motion components during training
without binary ROI masks, without pure_rigid, and without static_background.
Every Gaussian retains its native HexPlane free-form residual pathway:

    p_final(t) = p_canon + h(t) + alpha(t) * sum_{k=1}^K q_k * g_k(t)
    r_final(t) = Delta q_comp(t) (x) r_hex(t)

where:
    h(t) = p_hex(t) - p_canon is the persistent native HexPlane residual,
    g_k(t) is candidate rigid displacement for component k,
    q_k are continuous soft responsibilities with q_0 = 1 - sum(q_k) >= 0,
    Delta q_comp(t) is composed in tangent (axis-angle) space before quaternion conversion,
    alpha(t) in [0, 1] is the smooth ramp-in schedule.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply batches of quaternions (w, x, y, z)."""
    w = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    x = q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2]
    y = q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1]
    z = q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]

    q3 = torch.stack((w, x, y, z), dim=-1)
    norm = torch.norm(q3, dim=-1, keepdim=True).clamp(min=1e-8)
    return q3 / norm


def axis_angle_to_quaternion(omega: torch.Tensor) -> torch.Tensor:
    """Convert tangent / axis-angle vector omega [..., 3] to unit quaternion (w, x, y, z).
    
    Uses Taylor expansions around phi = 0 for numerical stability and exact gradients:
        cos(phi/2) ~ 1 - phi^2/8 + phi^4/384
        sin(phi/2)/phi ~ 1/2 - phi^2/48 + phi^4/3840
    """
    phi_sq = (omega * omega).sum(dim=-1, keepdim=True)
    phi = torch.sqrt(phi_sq.clamp(min=1e-12))
    small_mask = phi < 1e-4

    # Taylor series for small angles
    w_small = 1.0 - phi_sq / 8.0 + (phi_sq * phi_sq) / 384.0
    scale_small = 0.5 - phi_sq / 48.0 + (phi_sq * phi_sq) / 3840.0

    # Normal computation
    safe_phi = torch.where(small_mask, torch.ones_like(phi), phi)
    w_normal = torch.cos(0.5 * safe_phi)
    scale_normal = torch.sin(0.5 * safe_phi) / safe_phi

    w = torch.where(small_mask, w_small, w_normal)
    scale = torch.where(small_mask, scale_small, scale_normal)
    xyz = scale * omega

    q = torch.cat([w, xyz], dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)


def rodrigues_rotation(points: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Rotate points [..., 3] about origin by axis-angle vector omega [..., 3].
    
    R * p = p * cos(phi) + (u x p) * sin(phi) + u * (u . p) * (1 - cos(phi))
    For small phi, uses second-order Taylor expansion:
    R * p ~ p + omega x p + 0.5 * omega x (omega x p)
    """
    phi_sq = (omega * omega).sum(dim=-1, keepdim=True)
    phi = torch.sqrt(phi_sq.clamp(min=1e-12))
    small_mask = phi < 1e-4

    # Small angle expansion
    cross_1 = torch.cross(omega, points, dim=-1)
    rot_small = points + cross_1 + 0.5 * torch.cross(omega, cross_1, dim=-1)

    # Standard Rodrigues
    safe_phi = torch.where(small_mask, torch.ones_like(phi), phi)
    axis = omega / safe_phi
    cos_theta = torch.cos(safe_phi)
    sin_theta = torch.sin(safe_phi)

    cross_axis = torch.cross(axis, points, dim=-1)
    dot_axis = (axis * points).sum(dim=-1, keepdim=True)
    rot_normal = (
        points * cos_theta
        + cross_axis * sin_theta
        + axis * dot_axis * (1.0 - cos_theta)
    )

    return torch.where(small_mask, rot_small, rot_normal)


class BaseComponentTrajectory(nn.Module):
    """Base interface for rigid component motion trajectory."""

    def __init__(self, init_pivot: Optional[Union[List[float], torch.Tensor]] = None):
        super().__init__()
        if init_pivot is not None:
            if isinstance(init_pivot, torch.Tensor):
                pivot_t = init_pivot.detach().clone().view(1, 3).to(dtype=torch.float32)
            else:
                pivot_t = torch.tensor(init_pivot, dtype=torch.float32).view(1, 3)
            self.register_buffer("pivot", pivot_t)
        else:
            self.register_buffer("pivot", torch.zeros((1, 3), dtype=torch.float32))

    def get_trajectory(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute translation T(t) [..., 3] and axis-angle rotation omega(t) [..., 3]."""
        raise NotImplementedError

    def compute_candidate_displacement(
        self, canon_xyz: torch.Tensor, time: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute candidate 3D displacement g_ik(t) and axis-angle rotation omega_k(t).
        
        g_ik(t) = (R_k(t) - I)(canon_xyz - c_k) + T_k(t)
        Returns shape [N, 3] if time is single instant/per-point [N, 1],
        or shape [N, T, 3] if time has T distinct sample timestamps.
        """
        trans, omega = self.get_trajectory(time)
        N = canon_xyz.shape[0]
        if trans.dim() == 2 and trans.shape[0] == 1 and N > 1:
            trans = trans.expand(N, 3)
            omega = omega.expand(N, 3)

        if trans.dim() == 2 and trans.shape[0] != N:
            # Trajectory across T timesteps for N points -> [N, T, 3]
            T = trans.shape[0]
            centered = (canon_xyz - self.pivot).unsqueeze(0)  # [1, N, 3]
            omega_exp = omega.unsqueeze(1)                     # [T, 1, 3]
            trans_exp = trans.unsqueeze(1)                     # [T, 1, 3]
            rotated = rodrigues_rotation(centered, omega_exp)  # [T, N, 3]
            g_ik = ((rotated - centered) + trans_exp).permute(1, 0, 2)  # [N, T, 3]
            omega_out = omega_exp.expand(-1, N, -1).permute(1, 0, 2)     # [N, T, 3]
            return g_ik, omega_out

        centered = canon_xyz - self.pivot
        rotated = rodrigues_rotation(centered, omega)
        g_ik = (rotated - centered) + trans
        return g_ik, omega


class FourierComponentTrajectory(BaseComponentTrajectory):
    """Periodic multi-harmonic Fourier rigid trajectory with strict Nyquist guardrail."""

    def __init__(
        self,
        base_frequency_hz: float = 5.691,
        fps: float = 20.004,
        num_harmonics: int = 1,
        learn_frequency: bool = False,
        init_amplitude_m: float = 0.001,
        init_dominant_axis: Optional[List[float]] = None,
        init_rotation_axis: Optional[List[float]] = None,
        init_pivot: Optional[Union[List[float], torch.Tensor]] = None,
    ):
        super().__init__(init_pivot=init_pivot)
        self.base_frequency_hz = float(base_frequency_hz)
        self.fps = float(fps)
        self.num_harmonics = int(num_harmonics)
        self.learn_frequency = bool(learn_frequency)

        # Nyquist guardrail check
        nyquist_limit = 0.5 * self.fps
        for h in range(1, self.num_harmonics + 1):
            freq_h = h * self.base_frequency_hz
            if freq_h >= nyquist_limit:
                raise ValueError(
                    f"Harmonic {h} at {freq_h:.3f} Hz meets or exceeds Nyquist limit {nyquist_limit:.3f} Hz at {self.fps:.3f} fps."
                )

        if self.learn_frequency:
            self.freq_offset = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer("freq_offset", torch.zeros(1, dtype=torch.float32))

        # Translation parameters
        if init_dominant_axis is not None and len(init_dominant_axis) == 3:
            u = torch.tensor(init_dominant_axis, dtype=torch.float32)
        else:
            u = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.u_axis = nn.Parameter(u / torch.norm(u).clamp(min=1e-8))

        # Harmonic Fourier coefficients [num_harmonics]
        self.a_trans = nn.Parameter(torch.full((self.num_harmonics,), init_amplitude_m, dtype=torch.float32))
        self.b_trans = nn.Parameter(torch.zeros((self.num_harmonics,), dtype=torch.float32))

        # Rotation parameters (axis and small angle harmonics)
        if init_rotation_axis is not None and len(init_rotation_axis) == 3:
            v = torch.tensor(init_rotation_axis, dtype=torch.float32)
        else:
            v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        self.v_axis = nn.Parameter(v / torch.norm(v).clamp(min=1e-8))

        self.a_rot = nn.Parameter(torch.zeros((self.num_harmonics,), dtype=torch.float32))
        self.b_rot = nn.Parameter(torch.zeros((self.num_harmonics,), dtype=torch.float32))

    @property
    def effective_frequency_hz(self) -> torch.Tensor:
        if self.learn_frequency:
            # Bound frequency within +/- 15% of base
            bound = 0.15 * self.base_frequency_hz
            return self.base_frequency_hz + torch.tanh(self.freq_offset) * bound
        return torch.tensor(self.base_frequency_hz, device=self.u_axis.device)

    def get_trajectory(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        freq = self.effective_frequency_hz

        u_norm = self.u_axis / torch.norm(self.u_axis).clamp(min=1e-8)
        v_norm = self.v_axis / torch.norm(self.v_axis).clamp(min=1e-8)

        total_trans = torch.zeros((*time.shape[:-1], 3), device=time.device, dtype=time.dtype)
        total_rot = torch.zeros((*time.shape[:-1], 3), device=time.device, dtype=time.dtype)

        for h in range(1, self.num_harmonics + 1):
            omega_h = 2.0 * math.pi * float(h) * freq * time
            cos_h = torch.cos(omega_h)
            sin_h = torch.sin(omega_h)

            t_mag = self.a_trans[h - 1] * cos_h + self.b_trans[h - 1] * sin_h
            r_mag = self.a_rot[h - 1] * cos_h + self.b_rot[h - 1] * sin_h

            total_trans = total_trans + t_mag * u_norm
            total_rot = total_rot + r_mag * v_norm

        return total_trans, total_rot


class ChebyshevComponentTrajectory(BaseComponentTrajectory):
    """Smooth low-dimensional Chebyshev polynomial trajectory for non-periodic rigid motion."""

    def __init__(
        self,
        degree: int = 4,
        t_min: float = 0.0,
        t_max: float = 1.0,
        init_pivot: Optional[Union[List[float], torch.Tensor]] = None,
    ):
        super().__init__(init_pivot=init_pivot)
        self.degree = int(degree)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        # Coefficients [3, degree + 1] initialized to 0
        self.trans_coeffs = nn.Parameter(torch.zeros((3, self.degree + 1), dtype=torch.float32))
        self.rot_coeffs = nn.Parameter(torch.zeros((3, self.degree + 1), dtype=torch.float32))

    def _eval_chebyshev_basis(self, tau: torch.Tensor) -> torch.Tensor:
        """Evaluate Chebyshev polynomials T_0(tau) .. T_D(tau) of shape [..., D+1]."""
        T = [torch.ones_like(tau)]
        if self.degree >= 1:
            T.append(tau)
        for m in range(2, self.degree + 1):
            T.append(2.0 * tau * T[-1] - T[-2])
        return torch.stack(T, dim=-1)

    def get_trajectory(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        # Map time to [-1, 1]
        denom = max(self.t_max - self.t_min, 1e-6)
        tau = (2.0 * (time - self.t_min) / denom - 1.0).clamp(-1.0, 1.0)

        # Basis [..., 1, D+1]
        basis = self._eval_chebyshev_basis(tau)  # [..., 1, D+1]
        if basis.shape[-2] == 1:
            basis = basis.squeeze(-2)  # [..., D+1]

        # T(t) = basis @ trans_coeffs.T  -> [..., 3]
        total_trans = torch.matmul(basis, self.trans_coeffs.T)
        total_rot = torch.matmul(basis, self.rot_coeffs.T)
        return total_trans, total_rot


class MotionComponents(nn.Module):
    """Container for K soft learned rigid motion components."""

    def __init__(
        self,
        num_components: int = 1,
        components: Optional[List[BaseComponentTrajectory]] = None,
        ramp_start_iter: int = 1000,
        ramp_end_iter: int = 3000,
        lambda_res_max: float = 1.0,
        is_active: bool = True,
    ):
        super().__init__()
        self.num_components = int(num_components)
        self.ramp_start_iter = int(ramp_start_iter)
        self.ramp_end_iter = int(ramp_end_iter)
        self.lambda_res_max = float(lambda_res_max)
        self.is_active = bool(is_active)

        if components is not None:
            assert len(components) == self.num_components, f"Expected {self.num_components} components, got {len(components)}"
            self.components = nn.ModuleList(components)
        else:
            self.components = nn.ModuleList([
                FourierComponentTrajectory() for _ in range(self.num_components)
            ])

    def get_alpha(self, iteration: Optional[int]) -> float:
        """Compute ramp-in factor alpha in [0, 1]. Offline inference (iteration=None) uses alpha=1.0."""
        if iteration is None:
            return 1.0
        if iteration < self.ramp_start_iter:
            return 0.0
        if iteration >= self.ramp_end_iter:
            return 1.0
        return float((iteration - self.ramp_start_iter) / max(self.ramp_end_iter - self.ramp_start_iter, 1))

    def get_responsibilities(self, pc: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft responsibilities q [N, K] and unassigned / null responsibility q0 [N, 1] on simplex.
        
        Note: q0 represents 'unassigned responsibility' (i.e. not explained by any shared rigid component).
        A Gaussian with q0 = 1.0 is NOT static: it continues to move freely through the persistent
        native HexPlane residual field h_i(t).
        """
        logits = getattr(pc, "_motion_component_logits", None)
        num_pts = pc.get_xyz.shape[0]
        device = pc.get_xyz.device

        if logits is None or logits.shape[-1] != self.num_components:
            # Default to unassigned: q0 = 1, q_k = 0
            q0 = torch.ones((num_pts, 1), device=device, dtype=torch.float32)
            q = torch.zeros((num_pts, self.num_components), device=device, dtype=torch.float32)
            return q, q0

        # Unconstrained logits z with fixed null / unassigned logit z0 = 0
        z0 = torch.zeros((num_pts, 1), device=logits.device, dtype=logits.dtype)
        z_full = torch.cat([z0, logits], dim=-1)
        q_full = F.softmax(z_full, dim=-1)

        q0 = q_full[:, :1]  # unassigned / null component responsibility
        q = q_full[:, 1:]   # shared component responsibilities
        return q, q0

    def apply_deformation(
        self,
        pc: Any,
        means3D_canon: torch.Tensor,
        means3D_hex: torch.Tensor,
        rotations_hex: torch.Tensor,
        time: torch.Tensor,
        iteration: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate candidate displacements per component and blend into final pose.
        
        p_final = p_hex + alpha * sum_{k=1}^K q_k * g_k(t)
        r_final = Delta q_comp(t) (x) r_hex
        """
        if not self.is_active or self.num_components == 0:
            return means3D_hex, rotations_hex

        alpha = self.get_alpha(iteration)
        if alpha <= 0.0:
            return means3D_hex, rotations_hex

        q, q0 = self.get_responsibilities(pc)  # [N, K], [N, 1]

        # Check if all responsibilities are effectively null
        if (q < 1e-7).all():
            return means3D_hex, rotations_hex

        num_pts = means3D_canon.shape[0]
        cand_disps = []
        cand_omegas = []

        for k in range(self.num_components):
            g_k, omega_k = self.components[k].compute_candidate_displacement(means3D_canon, time)
            cand_disps.append(g_k)      # [N, 3]
            cand_omegas.append(omega_k)  # [N, 3]

        # Stack into [N, K, 3]
        G = torch.stack(cand_disps, dim=1)
        Omega = torch.stack(cand_omegas, dim=1)

        # Blend: sum_{k=1}^K q_ik * g_ik(t) and sum_{k=1}^K q_ik * omega_k(t)
        q_weight = q.unsqueeze(-1)  # [N, K, 1]
        blended_disp = (q_weight * G).sum(dim=1)      # [N, 3]
        blended_omega = (q_weight * Omega).sum(dim=1)  # [N, 3]

        # Position composition
        means3D_final = means3D_hex + alpha * blended_disp

        # Tangent rotation composition
        omega_final = alpha * blended_omega
        delta_q = axis_angle_to_quaternion(omega_final)  # [N, 4]
        rotations_final = batch_quaternion_multiply(delta_q, rotations_hex)

        return means3D_final, rotations_final

    def compute_residual_loss(
        self,
        hex_residual: torch.Tensor,
        pc: Any,
        iteration: int,
    ) -> torch.Tensor:
        """Detached responsibility-weighted HexPlane residual penalty.
        
        Normalizes by sum(1 - stopgrad(q_i0)) so the penalty remains invariant to point count:
        L_res = lambda_res(t) * [ sum_{i=1}^N (1 - stopgrad(q_i0)) ||h_i(t)||_2^2 ] / [ sum_{i=1}^N (1 - stopgrad(q_i0)) + eps ]
        """
        if not self.is_active or self.num_components == 0:
            return torch.tensor(0.0, device=hex_residual.device)

        alpha = self.get_alpha(iteration)
        if alpha <= 0.0 or self.lambda_res_max <= 0.0:
            return torch.tensor(0.0, device=hex_residual.device)

        _, q0 = self.get_responsibilities(pc)
        # Detach q0 to prevent the optimizer from driving q0 -> 1 to cheat the penalty
        weight = (1.0 - q0.detach()).clamp(min=0.0)  # [N, 1]

        res_norm_sq = (hex_residual * hex_residual).sum(dim=-1, keepdim=True)  # [N, 1]
        weighted_res = weight * res_norm_sq

        lambda_cur = self.lambda_res_max * alpha
        weight_mass = weight.sum().clamp(min=1e-6)
        return lambda_cur * (weighted_res.sum() / weight_mass)

    def update_responsibilities_ema(
        self,
        pc: Any,
        hex_residuals: torch.Tensor,
        candidate_displacements: torch.Tensor,
        beta: float = 0.1,
        sigma: Optional[float] = None,
    ):
        """EMA responsibility update comparing residual errors against candidate models.
        
        Args:
            pc: GaussianModel.
            hex_residuals: [N, T, 3] or [N, 3] observed HexPlane displacements.
            candidate_displacements: [N, K, T, 3] or [N, K, 3] candidate displacements.
            beta: EMA blending factor in (0, 1].
            sigma: Softmax temperature scale in meters. If None, auto-scaled from residuals.
        """
        if hex_residuals.dim() == 2:
            hex_residuals = hex_residuals.unsqueeze(1)
        if candidate_displacements.dim() == 3:
            candidate_displacements = candidate_displacements.unsqueeze(2)

        # Residual under null component (k=0): E_i0 = mean_t ||h_i(t)||^2
        e0 = (hex_residuals * hex_residuals).sum(dim=-1).mean(dim=-1, keepdim=True)  # [N, 1]

        # Residual under component k: E_ik = mean_t ||h_i(t) - g_ik(t)||^2
        diff = hex_residuals.unsqueeze(1) - candidate_displacements  # [N, K, T, 3]
        ek = (diff * diff).sum(dim=-1).mean(dim=-1)  # [N, K]

        all_errors = torch.cat([e0, ek], dim=-1)  # [N, K+1]

        if sigma is None:
            # Auto-temperature scale based on moving points magnitude
            moving_mags = torch.sqrt(e0[e0 > 1e-8])
            if moving_mags.numel() > 0:
                sig_val = float(torch.median(moving_mags).item() * 0.3)
                sigma_sq = max(sig_val * sig_val, 1e-8)
            else:
                sigma_sq = 1e-4
        else:
            sigma_sq = max(sigma * sigma, 1e-8)

        target_q_full = F.softmax(-all_errors / (2.0 * sigma_sq), dim=-1)
        target_qk = target_q_full[:, 1:]  # [N, K]

        cur_q, _ = self.get_responsibilities(pc)
        new_q = (1.0 - beta) * cur_q + beta * target_qk

        # Map back to logits z_ik = log(q_ik / q_i0)
        new_q0 = (1.0 - new_q.sum(dim=-1, keepdim=True)).clamp(min=1e-6)
        new_logits = torch.log(new_q.clamp(min=1e-6) / new_q0)

        if hasattr(pc, "_motion_component_logits") and pc._motion_component_logits is not None:
            pc._motion_component_logits.data.copy_(new_logits)
        else:
            pc._motion_component_logits = nn.Parameter(new_logits.to(pc.get_xyz.device))

    def get_summary(self) -> Dict[str, Any]:
        """Telemetry dict for run manifest and logging."""
        summary = {
            "num_components": self.num_components,
            "ramp_start_iter": self.ramp_start_iter,
            "ramp_end_iter": self.ramp_end_iter,
            "lambda_res_max": self.lambda_res_max,
            "is_active": self.is_active,
            "components": [],
        }
        for k, comp in enumerate(self.components):
            comp_info: Dict[str, Any] = {
                "index": k,
                "type": comp.__class__.__name__,
                "pivot": comp.pivot.squeeze().tolist(),
            }
            if isinstance(comp, FourierComponentTrajectory):
                comp_info.update({
                    "base_frequency_hz": comp.base_frequency_hz,
                    "effective_frequency_hz": float(comp.effective_frequency_hz.item()),
                    "num_harmonics": comp.num_harmonics,
                    "dominant_axis": comp.u_axis.squeeze().tolist(),
                    "rotation_axis": comp.v_axis.squeeze().tolist(),
                    "a_trans": comp.a_trans.detach().cpu().tolist(),
                    "b_trans": comp.b_trans.detach().cpu().tolist(),
                })
            elif isinstance(comp, ChebyshevComponentTrajectory):
                comp_info.update({
                    "degree": comp.degree,
                    "trans_coeffs_norm": float(torch.norm(comp.trans_coeffs).item()),
                    "rot_coeffs_norm": float(torch.norm(comp.rot_coeffs).item()),
                })
            summary["components"].append(comp_info)
        return summary

