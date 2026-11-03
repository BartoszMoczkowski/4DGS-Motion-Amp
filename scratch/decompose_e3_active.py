import json
import math
from pathlib import Path
import numpy as np
import torch

run_id = "turbine_r1_missing_6_w4_e3-625cdb58"
run_dir = Path(f"runs/{run_id}")

# 1. Load trajectories
traj_data = np.load(run_dir / "trajectories.npz")
traj = traj_data["traj"]  # [N, T, 3]
canon_xyz = traj_data["canonical_xyz"]  # [N, 3]
N, T, _ = traj.shape
fps = 20.004025
duration_s = T / fps
t_sec = np.linspace(0.0, duration_s, T, endpoint=False)

# 2. Load MotionComponents state and logits
with open(run_dir / "train_out/motion_components_state.json", "r", encoding="utf-8") as f:
    mc_state = json.load(f)

c0_info = mc_state["components"][0]
base_freq = float(c0_info["base_frequency_hz"])
a_trans = np.array(c0_info["a_trans"])  # [1]
b_trans = np.array(c0_info["b_trans"])  # [1]
dom_axis = np.array(c0_info["dominant_axis"])  # [3]
dom_axis = dom_axis / np.linalg.norm(dom_axis)

logits = torch.load(run_dir / "train_out/motion_component_logits.pth", map_location="cpu").detach()
# logits shape [N, 1]
logits_null = torch.cat([torch.zeros_like(logits), logits], dim=-1)
probs = torch.softmax(logits_null, dim=-1).numpy()
q1 = probs[:, 1:2]  # [N, 1]

# 3. Compute Component Trajectory g1(t)
omega = 2.0 * math.pi * base_freq * t_sec  # [T]
cos_t = np.cos(omega)
sin_t = np.sin(omega)
# g1(t) = (a * cos(w t) + b * sin(w t)) * dom_axis
scalar_disp = (a_trans[0] * cos_t + b_trans[0] * sin_t)  # [T] in meters
g1_t = scalar_disp[:, None] * dom_axis[None, :]  # [T, 3]

# Component displacement for each point: [N, T, 3]
comp_disp = q1[:, :, None] * g1_t[None, :, :]  # [N, T, 3]

# Total displacement: [N, T, 3]
total_disp = traj - canon_xyz[:, None, :]

# HexPlane displacement: [N, T, 3]
hex_disp = total_disp - comp_disp

# 4. Amplitudes and Statistics
amp_total = np.max(np.linalg.norm(total_disp, axis=-1), axis=-1)  # [N]
amp_comp = np.max(np.linalg.norm(comp_disp, axis=-1), axis=-1)    # [N]
amp_hex = np.max(np.linalg.norm(hex_disp, axis=-1), axis=-1)      # [N]

print("=== 3-WAY DISPLACEMENT DECOMPOSITION (meters / mm) ===")
print(f"Total Displ:   p50={np.percentile(amp_total, 50)*1000:.3f} mm, p90={np.percentile(amp_total, 90)*1000:.3f} mm, p99={np.percentile(amp_total, 99)*1000:.3f} mm, max={np.max(amp_total)*1000:.3f} mm")
print(f"Component:     p50={np.percentile(amp_comp, 50)*1000:.3f} mm, p90={np.percentile(amp_comp, 90)*1000:.3f} mm, p99={np.percentile(amp_comp, 99)*1000:.3f} mm, max={np.max(amp_comp)*1000:.3f} mm")
print(f"HexPlane-only: p50={np.percentile(amp_hex, 50)*1000:.3f} mm, p90={np.percentile(amp_hex, 90)*1000:.3f} mm, p99={np.percentile(amp_hex, 99)*1000:.3f} mm, max={np.max(amp_hex)*1000:.3f} mm")

# 5. Harmonic Power at 5.69 Hz via FFT
def compute_harmonic_power(disp_array, f0, fps):
    # disp_array: [N, T, 3]
    # FFT along T axis
    fft_vals = np.fft.rfft(disp_array, axis=1)  # [N, F, 3]
    freqs = np.fft.rfftfreq(T, d=1.0/fps)
    # find target bin
    bin_idx = np.argmin(np.abs(freqs - f0))
    power_f0 = np.sum(np.abs(fft_vals[:, bin_idx, :])**2, axis=-1)  # [N]
    total_power = np.sum(np.abs(fft_vals)**2, axis=(1, 2))  # [N]
    return power_f0, total_power, freqs[bin_idx]

p_f0_tot, p_tot_tot, bin_freq = compute_harmonic_power(total_disp, base_freq, fps)
p_f0_comp, p_tot_comp, _ = compute_harmonic_power(comp_disp, base_freq, fps)
p_f0_hex, p_tot_hex, _ = compute_harmonic_power(hex_disp, base_freq, fps)

print(f"\nTarget Frequency Bin: {bin_freq:.4f} Hz (nominal {base_freq:.4f} Hz)")
print(f"Mean harmonic power at f0: Total={np.mean(p_f0_tot):.6e}, Component={np.mean(p_f0_comp):.6e}, HexPlane={np.mean(p_f0_hex):.6e}")
print(f"Component share of f0 harmonic power on mover points:")
for q_thresh in [0.01, 0.05, 0.10, 0.20]:
    mask = (q1[:, 0] > q_thresh)
    n_pts = np.sum(mask)
    if n_pts > 0:
        p_tot_mean = np.mean(p_f0_tot[mask])
        p_comp_mean = np.mean(p_f0_comp[mask])
        p_hex_mean = np.mean(p_f0_hex[mask])
        share = p_comp_mean / (p_tot_mean + 1e-12)
        print(f"  q1 > {q_thresh:.2f}: {n_pts} pts ({n_pts/N*100:.2f}%) | f0 Power: Tot={p_tot_mean:.4e}, Comp={p_comp_mean:.4e}, Hex={p_hex_mean:.4e} | Comp Share={share*100:.2f}%")

# Also top 1% and top 5% by q1
for pct in [99, 95, 90]:
    val = np.percentile(q1[:, 0], pct)
    mask = (q1[:, 0] >= val)
    n_pts = np.sum(mask)
    p_tot_mean = np.mean(p_f0_tot[mask])
    p_comp_mean = np.mean(p_f0_comp[mask])
    p_hex_mean = np.mean(p_f0_hex[mask])
    share = p_comp_mean / (p_tot_mean + 1e-12)
    print(f"  Top {100-pct}% (q1 >= {val:.4f}, {n_pts} pts): Tot={p_tot_mean:.4e}, Comp={p_comp_mean:.4e}, Hex={p_hex_mean:.4e} | Comp Share={share*100:.2f}%")

# Save summary
out_data = {
    "run_id": run_id,
    "N": int(N),
    "mover_count_q05": int(np.sum(q1[:, 0] > 0.05)),
    "mover_count_q01": int(np.sum(q1[:, 0] > 0.01)),
    "component_amp_mm": float(math.sqrt(a_trans[0]**2 + b_trans[0]**2) * 1000.0),
    "total_p99_mm": float(np.percentile(amp_total, 99) * 1000.0),
    "comp_p99_mm": float(np.percentile(amp_comp, 99) * 1000.0),
    "hex_p99_mm": float(np.percentile(amp_hex, 99) * 1000.0),
    "component_power_share_scene": float(np.mean(p_f0_comp) / np.mean(p_f0_tot)),
}
with open("scratch/e3_active_decomposition.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
print("\nWrote scratch/e3_active_decomposition.json")
