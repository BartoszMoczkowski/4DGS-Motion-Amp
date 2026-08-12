import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

from pipeline.vendored.host.rigidity_graph import otsu_threshold_log
import numpy as np

rng = np.random.RandomState(42)
n = 100
xyz = rng.randn(n, 3).astype(np.float32)
traj = xyz[:, None, :] + rng.randn(n, 60, 3).astype(np.float32) * 0.001

from pipeline.vendored.host.trajectory_denoise import bandpass, trajectory_energy
traj_denoised, f0 = bandpass(traj, drive_freq=None, harmonics=3)
energy = trajectory_energy(traj_denoised)

thr = otsu_threshold_log(energy)
moving = energy > thr

print('energy min:', energy.min())
print('energy max:', energy.max())
print('energy median:', np.median(energy))
print('thr:', thr)
print('n_moving:', moving.sum())
print('thr >= max:', thr >= energy.max())
