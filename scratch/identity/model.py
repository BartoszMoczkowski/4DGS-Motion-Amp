"""scene-gen/identity/model.py — Identity Encodings module for Gaussian Grouping.

Implements:
1. 16-dimensional per-Gaussian learnable identity feature vector.
2. 1x1 conv linear classifier mapping 16D rendered feature maps to 2D instance class logits.
3. Chunked rasterization of 16D features using the existing 3-channel CUDA rasterizer.
4. 2D Cross-Entropy classification loss + 3D k-NN spatial consistency regularization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scipy.spatial import cKDTree


def build_knn_edges(xyz: np.ndarray, k: int = 10, max_edges: int = 500000) -> torch.Tensor:
    """Build k-NN edge index tensor (2, E) on canonical 3D coordinates."""
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    src = np.repeat(np.arange(len(xyz)), k)
    dst = idx[:, 1:].reshape(-1)

    edges = np.stack([src, dst], axis=0)  # (2, E)
    # Remove self-loops
    valid = edges[0] != edges[1]
    edges = edges[:, valid]

    if edges.shape[1] > max_edges:
        rng = np.random.default_rng(42)
        sel = rng.choice(edges.shape[1], size=max_edges, replace=False)
        edges = edges[:, sel]

    return torch.tensor(edges, dtype=torch.int64, device="cuda")


class GaussianGroupingModule(nn.Module):
    """Augments a trained 4DGS model with 16D identity encodings."""

    def __init__(
        self,
        gaussians: GaussianModel,
        num_classes: int,
        feature_dim: int = 16,
        lambda_3d: float = 0.1,
    ):
        super().__init__()
        self.gaussians = gaussians
        self.num_classes = max(num_classes, 2)
        self.feature_dim = feature_dim
        self.lambda_3d = lambda_3d

        n_pts = gaussians.get_xyz.shape[0]

        # Freeze all 4DGS geometry and deformation parameters
        gaussians._xyz.requires_grad = False
        gaussians._scaling.requires_grad = False
        gaussians._rotation.requires_grad = False
        gaussians._opacity.requires_grad = False
        gaussians._features_dc.requires_grad = False
        gaussians._features_rest.requires_grad = False
        gaussians._deformation.eval()
        for p in gaussians._deformation.parameters():
            p.requires_grad = False

        # Learnable per-Gaussian 16D identity embedding
        # Initialized with small random normal
        self.identity_params = nn.Parameter(
            0.01 * torch.randn(n_pts, feature_dim, device="cuda")
        )

        # Linear classifier: 1x1 conv mapping 16D features -> num_classes logits
        self.classifier = nn.Conv2d(feature_dim, self.num_classes, kernel_size=1).cuda()

        # Dummy pipe settings
        class Pipe:
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False

        self.pipe = Pipe()
        self.bg_zero = torch.zeros(3, device="cuda")

    def render_features(self, cam: Camera) -> torch.Tensor:
        """Render 16D feature map using chunked rasterization passes."""
        chunks = []
        n_pts = self.identity_params.shape[0]

        # Pass in 3-channel chunks: 16 = 3 + 3 + 3 + 3 + 3 + 1 (padded)
        for c in range(0, self.feature_dim, 3):
            c_end = min(c + 3, self.feature_dim)
            chunk_feat = self.identity_params[:, c:c_end]
            dim_chunk = chunk_feat.shape[1]

            if dim_chunk < 3:
                pad = torch.zeros(n_pts, 3 - dim_chunk, device="cuda")
                chunk_feat = torch.cat([chunk_feat, pad], dim=1)

            res = render(cam, self.gaussians, self.pipe, self.bg_zero, override_color=chunk_feat, stage="fine")
            chunks.append(res["render"][:dim_chunk])

        rendered = torch.cat(chunks, dim=0)  # (16, H, W)
        return rendered

    def forward_loss(
        self,
        cam: Camera,
        gt_mask: torch.Tensor,
        xyz: torch.Tensor,
        k: int = 5,
        sample_size: int = 800,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute 2D CrossEntropy loss + 3D spatial consistency loss.

        Args:
            cam: Camera viewpoint object.
            gt_mask: (H, W) int64 tensor of class labels (0..num_classes-1).
            xyz: (N, 3) float32 tensor of canonical coordinates.

        Returns:
            total_loss: scalar loss tensor with grad.
            loss_2d_val: float value of 2D CE loss.
            loss_3d_val: float value of 3D spatial loss.
        """
        rendered_feat = self.render_features(cam)  # (16, H, W)

        # 2D classification loss
        logits_2d = self.classifier(rendered_feat.unsqueeze(0))  # (1, num_classes, H, W)
        loss_2d = F.cross_entropy(logits_2d, gt_mask.unsqueeze(0).long())
        loss_2d_norm = loss_2d / math.log(max(self.num_classes, 2))

        # 3D spatial consistency loss (Gaussian Grouping ECCV 2024 KL divergence)
        loss_3d_val = 0.0
        if self.lambda_3d > 0.0:
            n_pts = self.identity_params.shape[0]
            if n_pts > sample_size:
                sample_idx = torch.randperm(n_pts, device="cuda")[:sample_size]
                cand_idx = torch.randperm(n_pts, device="cuda")[:min(n_pts, 50000)]
            else:
                sample_idx = torch.arange(n_pts, device="cuda")
                cand_idx = sample_idx

            w = self.classifier.weight.squeeze(-1).squeeze(-1)  # (num_classes, 16)
            b = self.classifier.bias  # (num_classes,)

            sample_feat = self.identity_params[sample_idx]
            cand_feat = self.identity_params[cand_idx]
            sample_preds = F.softmax(sample_feat @ w.T + b, dim=-1)
            cand_preds = F.softmax(cand_feat @ w.T + b, dim=-1)

            sample_xyz = xyz[sample_idx]
            cand_xyz = xyz[cand_idx]

            dists = torch.cdist(sample_xyz, cand_xyz)
            _, topk_idx = dists.topk(k, largest=False)
            neighbor_preds = cand_preds[topk_idx]  # (sample_size, k, num_classes)

            kl = sample_preds.unsqueeze(1) * (
                torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10)
            )
            loss_3d = kl.sum(dim=-1).mean() / self.num_classes
            loss_3d_val = float(loss_3d.item())
            total_loss = loss_2d_norm + self.lambda_3d * loss_3d
        else:
            total_loss = loss_2d_norm

        return total_loss, float(loss_2d.item()), loss_3d_val

    @torch.no_grad()
    def get_gaussian_labels(self) -> np.ndarray:
        """Assign discrete class label to each 3D Gaussian via classifier weights."""
        # Classifier weight: (num_classes, 16, 1, 1), bias: (num_classes,)
        w = self.classifier.weight.squeeze(-1).squeeze(-1)  # (num_classes, 16)
        b = self.classifier.bias  # (num_classes,)

        logits = self.identity_params @ w.T + b  # (N, num_classes)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        return pred

    @torch.no_grad()
    def get_normalized_features(self) -> np.ndarray:
        """Return (N, 16) L2-normalized identity features."""
        norm_feat = F.normalize(self.identity_params, p=2, dim=-1)
        return norm_feat.cpu().numpy()
