import numpy as np
import json
import sys
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_multiview_wobble import evaluate_3d_trajectories

fps = 20.004025315610498
f0 = 5.690745

e2_data = np.load("runs/turbine_r1_missing_6_w3_e2-fda6282c/trajectories.npz")
e3_data = np.load("runs/turbine_r1_missing_6_w4_e3-10e6ad8b/trajectories.npz")

e2_canon = e2_data.get("canonical_xyz", e2_data["traj"][:, 0, :])
e3_canon = e3_data.get("canonical_xyz", e3_data["traj"][:, 0, :])

e2_traj = e2_data["traj"]
e3_traj = e3_data["traj"]

# Build KDTree on E2 canonical coordinates and query closest E3 points
tree = cKDTree(e2_canon)
dist, idx = tree.query(e3_canon)

# Match within 2 cm (0.02 m)
match_mask = dist < 0.02
matched_e3_traj = e3_traj[match_mask]
matched_e3_canon = e3_canon[match_mask]
matched_e2_idx = idx[match_mask]
matched_e2_traj = e2_traj[matched_e2_idx]
matched_e2_canon = e2_canon[matched_e2_idx]

print(f"Matched {len(matched_e3_traj)} points between E2 and E3 (within 2 cm)")

# Evaluate on matched populations under Convention B (physical 5.69 Hz)
diag_e2_matched = evaluate_3d_trajectories(matched_e2_traj, fps=fps, f0=f0, canonical_xyz=matched_e2_canon)
diag_e3_matched = evaluate_3d_trajectories(matched_e3_traj, fps=fps, f0=f0, canonical_xyz=matched_e3_canon)

print("\n--- MATCHED POPULATION (53,842 points, Physical Timebase) ---")
print(f"E2 Matched: p50={diag_e2_matched.p50_amp*1000:.2f} mm, p90={diag_e2_matched.p90_amp*1000:.2f} mm, p99={diag_e2_matched.p99_amp*1000:.2f} mm, Max={diag_e2_matched.max_amp*1000:.2f} mm, Coherence={diag_e2_matched.phase_coherence_r:.3f}")
print(f"E3 Matched: p50={diag_e3_matched.p50_amp*1000:.2f} mm, p90={diag_e3_matched.p90_amp*1000:.2f} mm, p99={diag_e3_matched.p99_amp*1000:.2f} mm, Max={diag_e3_matched.max_amp*1000:.2f} mm, Coherence={diag_e3_matched.phase_coherence_r:.3f}")
