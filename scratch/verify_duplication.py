import numpy as np
import torch
from pathlib import Path

# Load E3 trajectory file
npz_path = Path("runs/turbine_r1_missing_6_w4_e3-10e6ad8b/trajectories.npz")
data = np.load(npz_path)
traj_combined = data["traj"] # [N, T, 3]
canonical_xyz = data["canonical_xyz"] # [N, 3]

# Load motion components
st_mc = torch.load('runs/turbine_r1_missing_6_w4_e3-10e6ad8b/train_out/motion_components.pth', map_location='cpu')
st_logits = torch.load('runs/turbine_r1_missing_6_w4_e3-10e6ad8b/train_out/motion_component_logits.pth', map_location='cpu')

# Compute responsibilities
logits_with_null = torch.cat([torch.zeros_like(st_logits), st_logits], dim=-1)
probs = torch.softmax(logits_with_null, dim=-1)
q1 = probs[:, 1:2].numpy() # [N, 1]

# In FourierComponentTrajectory:
# a_trans = 0.001, u = [0, 0, 1]
# g_1(t) = a_trans * cos(2 pi * freq * t) * u
# time in [0, 1) over 261 steps
N, T, _ = traj_combined.shape
times = np.linspace(0.0, 1.0, T, endpoint=False) # [T]
freq = 5.690745

omega = 2.0 * np.pi * freq * times # [T]
cos_t = np.cos(omega) # [T]
u = np.array([0.0, 0.0, 1.0])
# g_1: [T, 3] = cos_t[:, None] * 0.001 * u
g1_t = cos_t[:, None] * 0.001 * u # [T, 3]

# Component displacement for each Gaussian: q1 * g1(t) -> [N, T, 3]
comp_disp = q1[:, :, None] * g1_t[None, :, :] # [N, 1, 3] * [1, T, 3] -> [N, T, 3]

# Combined displacement = traj_combined - canonical_xyz[:, None, :]
total_disp = traj_combined - canonical_xyz[:, None, :]

# HexPlane displacement = total_disp - comp_disp
hex_disp = total_disp - comp_disp

print(f"Total displacement: min={np.min(total_disp):.6f}, max={np.max(total_disp):.6f}, mean_abs={np.mean(np.abs(total_disp)):.6f}")
print(f"Component displacement (q1 * g1): min={np.min(comp_disp):.8f}, max={np.max(comp_disp):.8f}, mean_abs={np.mean(np.abs(comp_disp)):.8f}")
print(f"HexPlane displacement: min={np.min(hex_disp):.6f}, max={np.max(hex_disp):.6f}, mean_abs={np.mean(np.abs(hex_disp)):.6f}")

diff = np.abs(total_disp - hex_disp)
print(f"Max absolute difference between Combined and HexPlane-only: {np.max(diff):.8f} meters ({np.max(diff)*1000:.5f} mm)")
print(f"Max component contribution: {np.max(np.abs(comp_disp))*1000:.5f} mm")
