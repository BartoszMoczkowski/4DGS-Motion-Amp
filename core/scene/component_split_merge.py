"""core/scene/component_split_merge.py

Reversible Outer-Loop Component Split and Merge Controller (W5 / T53).
Treats structural changes (K -> K+1 or K -> K-1) as testable hypotheses validated
on held-out evidence, with exact bit-for-bit rollback if held-out reconstruction
or motion coherence fails to improve.

Key Properties:
1. Outer-loop controller running at configurable intervals after component warm-up.
2. Soft candidate initialization: child responsibilities softly partition parent responsibility
   (q_{i, ka} + q_{i, kb} = q_{ik}) preserving the simplex constraint and HexPlane residual.
3. In-memory / disk structural snapshot with exact bit-for-bit rollback of all parameters,
   responsibilities, optimizer moments, and RNG state upon rejected hypothesis.
4. Cooldown ledger preventing re-testing previously rejected structural configurations.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.motion_components import (
    MotionComponents,
    FourierComponentTrajectory,
    ChebyshevComponentTrajectory,
    BaseComponentTrajectory,
)


@dataclass
class StructuralSnapshot:
    """Exact snapshot of model, component, and optimizer state prior to a hypothesis."""

    iteration: int
    num_components: int
    gaussian_params: Dict[str, torch.Tensor]
    mc_state_dict: Dict[str, Any]
    component_configs: List[Dict[str, Any]]
    opt_state_dict: Optional[Dict[str, Any]]
    mc_opt_state_dict: Optional[Dict[str, Any]]
    logits: Optional[torch.Tensor]
    rng_cpu: torch.Tensor
    rng_cuda: Optional[List[torch.Tensor]] = None

    @classmethod
    def capture(
        cls,
        pc: Any,
        mc: MotionComponents,
        iteration: int,
        opt: Optional[torch.optim.Optimizer] = None,
        mc_opt: Optional[torch.optim.Optimizer] = None,
    ) -> StructuralSnapshot:
        """Capture deep-cloned state of Gaussians, motion components, and optimizers."""
        g_params: Dict[str, torch.Tensor] = {}
        for attr in ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]:
            val = getattr(pc, attr, None)
            if val is not None and isinstance(val, torch.Tensor):
                g_params[attr] = val.detach().clone()

        if hasattr(pc, "_deformation"):
            g_params["_deformation"] = copy.deepcopy(pc._deformation.state_dict())

        logits = None
        if getattr(pc, "_motion_component_logits", None) is not None:
            logits = pc._motion_component_logits.detach().clone()

        opt_sd = copy.deepcopy(opt.state_dict()) if opt is not None else None
        mc_opt_sd = copy.deepcopy(mc_opt.state_dict()) if mc_opt is not None else None

        # Capture component configurations
        configs = []
        for c in mc.components:
            cfg = {"type": c.__class__.__name__, "pivot": c.pivot.detach().clone()}
            if isinstance(c, FourierComponentTrajectory):
                cfg.update({
                    "base_frequency_hz": c.base_frequency_hz,
                    "fps": c.fps,
                    "num_harmonics": c.num_harmonics,
                    "learn_frequency": c.learn_frequency,
                })
            elif isinstance(c, ChebyshevComponentTrajectory):
                cfg.update({
                    "degree": c.degree,
                    "t_min": c.t_min,
                    "t_max": c.t_max,
                })
            configs.append(cfg)

        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        return cls(
            iteration=iteration,
            num_components=mc.num_components,
            gaussian_params=g_params,
            mc_state_dict=copy.deepcopy(mc.state_dict()),
            component_configs=configs,
            opt_state_dict=opt_sd,
            mc_opt_state_dict=mc_opt_sd,
            logits=logits,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=rng_cuda,
        )

    def restore(
        self,
        pc: Any,
        mc: MotionComponents,
        opt: Optional[torch.optim.Optimizer] = None,
        mc_opt: Optional[torch.optim.Optimizer] = None,
    ):
        """Restore exact bit-for-bit state onto model, components, and optimizers."""
        # 1. Restore Gaussian tensors
        for attr, tensor in self.gaussian_params.items():
            if attr == "_deformation":
                pc._deformation.load_state_dict(tensor)
            else:
                curr = getattr(pc, attr, None)
                if curr is not None:
                    curr.data.copy_(tensor)

        # 2. Restore motion component logits
        if self.logits is not None:
            if hasattr(pc, "_motion_component_logits") and pc._motion_component_logits is not None:
                if pc._motion_component_logits.shape != self.logits.shape:
                    pc._motion_component_logits = nn.Parameter(self.logits.clone().to(pc._xyz.device))
                else:
                    pc._motion_component_logits.data.copy_(self.logits)
            else:
                pc._motion_component_logits = nn.Parameter(self.logits.clone().to(pc._xyz.device))

        # 3. Rebuild component list matching snapshot
        rebuilt_components: List[BaseComponentTrajectory] = []
        for cfg in self.component_configs:
            c_type = cfg["type"]
            pivot = cfg["pivot"]
            if c_type == "FourierComponentTrajectory":
                comp = FourierComponentTrajectory(
                    base_frequency_hz=cfg["base_frequency_hz"],
                    fps=cfg["fps"],
                    num_harmonics=cfg["num_harmonics"],
                    learn_frequency=cfg["learn_frequency"],
                    init_pivot=pivot,
                )
            elif c_type == "ChebyshevComponentTrajectory":
                comp = ChebyshevComponentTrajectory(
                    degree=cfg["degree"],
                    t_min=cfg["t_min"],
                    t_max=cfg["t_max"],
                    init_pivot=pivot,
                )
            else:
                raise ValueError(f"Unknown component type: {c_type}")
            rebuilt_components.append(comp.to(pc._xyz.device))

        mc.num_components = self.num_components
        mc.components = nn.ModuleList(rebuilt_components)
        mc.load_state_dict(self.mc_state_dict)

        # 4. Restore optimizer states
        if opt is not None and self.opt_state_dict is not None:
            opt.load_state_dict(self.opt_state_dict)
        if mc_opt is not None and self.mc_opt_state_dict is not None:
            mc_opt.param_groups.clear()
            mc_opt.state.clear()
            for saved_g in self.mc_opt_state_dict["param_groups"]:
                g_name = saved_g.get("name", "")
                lr = saved_g.get("lr", 0.001)
                if g_name == "mc_trajectories":
                    mc_opt.add_param_group({"params": list(mc.parameters()), "lr": lr, "name": g_name})
                elif g_name == "mc_logits":
                    mc_opt.add_param_group({"params": [pc._motion_component_logits], "lr": lr, "name": g_name})
                elif len(saved_g["params"]) == len(list(mc.parameters())):
                    mc_opt.add_param_group({"params": list(mc.parameters()), "lr": lr})
                elif len(saved_g["params"]) == 1 and pc._motion_component_logits is not None:
                    mc_opt.add_param_group({"params": [pc._motion_component_logits], "lr": lr})
            mc_opt.load_state_dict(self.mc_opt_state_dict)

        # 5. Restore RNG states
        torch.set_rng_state(self.rng_cpu)
        if self.rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.rng_cuda)


@dataclass
class SplitMergeHypothesis:
    """Active testable structural hypothesis."""

    action: str  # "split" or "merge"
    target_idx: Union[int, Tuple[int, int]]
    proposal_iter: int
    eval_window: int
    baseline_metric: float  # Held-out metric (e.g. PSNR or negative loss) before proposal


class ComponentSplitMergeController:
    """Outer-loop reversible controller for motion component split and merge."""

    def __init__(
        self,
        enabled: bool = False,
        start_iter: int = 1500,
        interval: int = 1000,
        max_components: int = 4,
        min_component_support: float = 10.0,
        eval_window: int = 300,
        min_metric_improvement: float = 0.05,
        cooldown_iters: int = 2000,
    ):
        self.enabled = bool(enabled)
        self.start_iter = int(start_iter)
        self.interval = int(interval)
        self.max_components = int(max_components)
        self.min_component_support = float(min_component_support)
        self.eval_window = int(eval_window)
        self.min_metric_improvement = float(min_metric_improvement)
        self.cooldown_iters = int(cooldown_iters)

        self.active_hypothesis: Optional[SplitMergeHypothesis] = None
        self.snapshot: Optional[StructuralSnapshot] = None
        self.ledger: List[Dict[str, Any]] = []
        self.rejection_cooldowns: Dict[str, int] = {}  # signature -> cooldown_until_iter

    def is_proposal_due(self, iteration: int) -> bool:
        """Check if an outer-loop proposal interval has arrived."""
        if not self.enabled or self.active_hypothesis is not None:
            return False
        if iteration < self.start_iter:
            return False
        return (iteration - self.start_iter) % self.interval == 0

    def propose_split(
        self,
        pc: Any,
        mc: MotionComponents,
        target_k: int,
        iteration: int,
        baseline_metric: float,
        hex_residuals: torch.Tensor,
        opt: Optional[torch.optim.Optimizer] = None,
        mc_opt: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        resp_lr: float = 0.01,
    ) -> bool:
        """Hypothesize splitting component target_k into two children (k_a, k_b).
        
        Softly divides responsibility: q_{i, ka} + q_{i, kb} = q_{ik}.
        Takes exact snapshot before modification.
        """
        if mc.num_components >= self.max_components:
            return False

        sig = f"split_{target_k}"
        if iteration < self.rejection_cooldowns.get(sig, 0):
            return False

        # Verify support mass
        q, q0 = mc.get_responsibilities(pc)
        support = q[:, target_k].sum().item()
        if support < self.min_component_support:
            return False

        # 1. Capture snapshot before applying structural change
        self.snapshot = StructuralSnapshot.capture(pc, mc, iteration, opt, mc_opt)

        # 2. Compute soft partition factor s_i in (0, 1) using residual direction
        parent_comp = mc.components[target_k]
        q_target = q[:, target_k]  # [N]
        mask_support = q_target > 0.05
        if not mask_support.any():
            self.snapshot = None
            return False

        # Find dominant directional variation of residual on supported Gaussians
        if hex_residuals.dim() == 3:
            res_pts = hex_residuals.mean(dim=1)  # [N, 3]
        else:
            res_pts = hex_residuals

        res_sup = res_pts[mask_support]  # [M, 3]
        centered_res = res_sup - res_sup.mean(dim=0, keepdim=True)
        # SVD / PCA on residual variation
        try:
            _, _, V = torch.pca_lowrank(centered_res, q=1)
            dominant_dir = V[:, 0]  # [3]
        except Exception:
            dominant_dir = torch.tensor([1.0, 0.0, 0.0], device=pc.get_xyz.device)

        proj = (res_pts * dominant_dir).sum(dim=-1)  # [N]
        proj_mean = proj[mask_support].mean()
        proj_std = proj[mask_support].std().clamp(min=1e-4)
        # Normalized soft partition s_i
        s = torch.sigmoid((proj - proj_mean) / proj_std)  # [N]

        # 3. Softly partition responsibilities: q_ka = s * q_k, q_kb = (1 - s) * q_k
        q_ka = (s * q_target).clamp(min=1e-7)
        q_kb = ((1.0 - s) * q_target).clamp(min=1e-7)

        # Replace component k with k_a, append k_b at end
        new_q_list = []
        for j in range(mc.num_components):
            if j == target_k:
                new_q_list.append(q_ka)
            else:
                new_q_list.append(q[:, j])
        new_q_list.append(q_kb)

        new_q = torch.stack(new_q_list, dim=-1)  # [N, K+1]
        # Recompute logits: z_ij = log(q_ij / q_i0)
        new_logits = torch.log(new_q / q0.clamp(min=1e-6))
        pc._motion_component_logits = nn.Parameter(new_logits.to(pc.get_xyz.device))

        # 4. Clone and parameterize child components
        child_a = copy.deepcopy(parent_comp)
        child_b = copy.deepcopy(parent_comp)

        # Shift child pivots / dominant axes slightly along partition
        with torch.no_grad():
            if isinstance(child_a, FourierComponentTrajectory):
                child_a.u_axis.add_(0.1 * dominant_dir)
                child_b.u_axis.sub_(0.1 * dominant_dir)
            elif isinstance(child_b, ChebyshevComponentTrajectory):
                child_a.trans_coeffs.add_(0.001 * dominant_dir.unsqueeze(-1))
                child_b.trans_coeffs.sub_(0.001 * dominant_dir.unsqueeze(-1))

        new_components = []
        for j in range(mc.num_components):
            if j == target_k:
                new_components.append(child_a)
            else:
                new_components.append(mc.components[j])
        new_components.append(child_b)

        mc.num_components = len(new_components)
        mc.components = nn.ModuleList(new_components)

        # 5. Rebuild optimizer param groups for K+1
        if mc_opt is not None:
            mc_opt.param_groups.clear()
            mc_opt.state.clear()
            mc_opt.add_param_group({"params": list(mc.parameters()), "lr": lr, "name": "mc_trajectories"})
            mc_opt.add_param_group({"params": [pc._motion_component_logits], "lr": resp_lr, "name": "mc_logits"})

        # Record active hypothesis
        self.active_hypothesis = SplitMergeHypothesis(
            action="split",
            target_idx=target_k,
            proposal_iter=iteration,
            eval_window=self.eval_window,
            baseline_metric=baseline_metric,
        )
        return True

    def propose_merge(
        self,
        pc: Any,
        mc: MotionComponents,
        k1: int,
        k2: int,
        iteration: int,
        baseline_metric: float,
        opt: Optional[torch.optim.Optimizer] = None,
        mc_opt: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        resp_lr: float = 0.01,
    ) -> bool:
        """Hypothesize merging components (k1, k2) into a single merged component.
        
        Sums responsibilities: q_{merged} = q_k1 + q_k2.
        Takes exact snapshot before modification.
        """
        if mc.num_components <= 1:
            return False

        k_low, k_high = min(k1, k2), max(k1, k2)
        sig = f"merge_{k_low}_{k_high}"
        if iteration < self.rejection_cooldowns.get(sig, 0):
            return False

        # 1. Capture snapshot before applying structural change
        self.snapshot = StructuralSnapshot.capture(pc, mc, iteration, opt, mc_opt)

        q, q0 = mc.get_responsibilities(pc)
        # Merged responsibility
        q_merged = (q[:, k_low] + q[:, k_high]).clamp(min=1e-7)

        new_q_list = []
        new_components = []
        for j in range(mc.num_components):
            if j == k_low:
                new_q_list.append(q_merged)
                # Merged component inherits k_low
                new_components.append(mc.components[k_low])
            elif j == k_high:
                continue  # Removed
            else:
                new_q_list.append(q[:, j])
                new_components.append(mc.components[j])

        new_q = torch.stack(new_q_list, dim=-1)
        new_logits = torch.log(new_q / q0.clamp(min=1e-6))
        pc._motion_component_logits = nn.Parameter(new_logits.to(pc.get_xyz.device))

        mc.num_components = len(new_components)
        mc.components = nn.ModuleList(new_components)

        # Rebuild optimizer
        if mc_opt is not None:
            mc_opt.param_groups.clear()
            mc_opt.state.clear()
            mc_opt.add_param_group({"params": list(mc.parameters()), "lr": lr, "name": "mc_trajectories"})
            mc_opt.add_param_group({"params": [pc._motion_component_logits], "lr": resp_lr, "name": "mc_logits"})

        self.active_hypothesis = SplitMergeHypothesis(
            action="merge",
            target_idx=(k_low, k_high),
            proposal_iter=iteration,
            eval_window=self.eval_window,
            baseline_metric=baseline_metric,
        )
        return True

    def evaluate_and_decide(
        self,
        current_iter: int,
        current_metric: float,
        pc: Any,
        mc: MotionComponents,
        opt: Optional[torch.optim.Optimizer] = None,
        mc_opt: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[bool, str]:
        """Evaluate active hypothesis against held-out evidence.
        
        Returns (is_decided, decision_string).
        """
        if self.active_hypothesis is None or self.snapshot is None:
            return False, "no_active_hypothesis"

        if current_iter < self.active_hypothesis.proposal_iter + self.active_hypothesis.eval_window:
            return False, "evaluating"

        # Evaluation window has elapsed: compare held-out evidence
        improvement = current_metric - self.active_hypothesis.baseline_metric
        hyp = self.active_hypothesis
        sig = f"{hyp.action}_{hyp.target_idx}"

        if improvement >= self.min_metric_improvement:
            # ACCEPT: commit structural change
            self.ledger.append({
                "action": hyp.action,
                "target": hyp.target_idx,
                "proposal_iter": hyp.proposal_iter,
                "decision_iter": current_iter,
                "status": "accepted",
                "metric_improvement": float(improvement),
                "resulting_components": mc.num_components,
            })
            self.snapshot = None
            self.active_hypothesis = None
            return True, "accepted"
        else:
            # REJECT: bit-for-bit exact rollback
            self.snapshot.restore(pc, mc, opt, mc_opt)
            self.rejection_cooldowns[sig] = current_iter + self.cooldown_iters
            self.ledger.append({
                "action": hyp.action,
                "target": hyp.target_idx,
                "proposal_iter": hyp.proposal_iter,
                "decision_iter": current_iter,
                "status": "rejected",
                "metric_improvement": float(improvement),
                "cooldown_until": current_iter + self.cooldown_iters,
                "resulting_components": mc.num_components,
            })
            self.snapshot = None
            self.active_hypothesis = None
            return True, "rejected"

    def get_summary(self) -> Dict[str, Any]:
        """Summary telemetry dict for manifest and logs."""
        return {
            "enabled": self.enabled,
            "start_iter": self.start_iter,
            "interval": self.interval,
            "max_components": self.max_components,
            "min_component_support": self.min_component_support,
            "eval_window": self.eval_window,
            "min_metric_improvement": self.min_metric_improvement,
            "rejections": len([e for e in self.ledger if e["status"] == "rejected"]),
            "acceptances": len([e for e in self.ledger if e["status"] == "accepted"]),
            "ledger": self.ledger,
        }
