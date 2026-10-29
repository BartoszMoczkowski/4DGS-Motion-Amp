"""Voxel-grid ROI — authoritative turbine ROI definition (T41, r11 corrective step).

The bbox form of the turbine ROI is non-selective (covers 68–92% of Gaussians
because the table sits inside the box), so the ROI is stored as a dilated voxel
occupancy grid produced by scratch/derive_turbine_roi.py from multi-view motion
masks. The grid is positional (indexed by Gaussian means3D), which makes it
densification-safe: unlike a per-point boolean mask it stays valid as Gaussians
are cloned/split/pruned during training.

Artifact: <scene>/roi_voxels.npz with arrays
  grid       bool [nx, ny, nz]  occupied ROI voxels (dilated)
  origin     float64 [3]        world coordinates of grid voxel (0,0,0)'s index origin
  voxel_size float64            world units per voxel
"""
import numpy as np
import torch


def load_roi_grid(npz_path, device="cuda"):
    """Load roi_voxels.npz into torch tensors. Returns a dict with:
    grid:       bool tensor [nx, ny, nz] on `device`
    origin_vox: int64 tensor [3] — voxel index of world origin offset
    voxel_size: float
    """
    d = np.load(npz_path)
    voxel_size = float(d["voxel_size"])
    grid = torch.from_numpy(np.ascontiguousarray(d["grid"])).to(device=device)
    grid = grid.bool()
    origin_vox = torch.from_numpy(
        np.floor(np.asarray(d["origin"], dtype=np.float64) / voxel_size).astype(np.int64)
    ).to(device=device)
    return {"grid": grid, "origin_vox": origin_vox, "voxel_size": voxel_size}


def voxel_roi_mask(means3D, roi_grid):
    """Boolean [N] mask: True where the Gaussian's position falls in an ROI voxel."""
    idx = torch.floor(means3D / roi_grid["voxel_size"]).long() - roi_grid["origin_vox"]
    g = roi_grid["grid"]
    shape = torch.tensor(g.shape, device=idx.device, dtype=idx.dtype)
    inb = ((idx >= 0) & (idx < shape)).all(dim=1)
    mask = torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    if inb.any():
        ii = idx[inb]
        mask[inb] = g[ii[:, 0], ii[:, 1], ii[:, 2]]
    return mask
