import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

from pipeline.vendored.host.motion_gate import motion_gate
import numpy as np

# Test 1: all-static scene should fall back to all True (degenerate)
rng = np.random.RandomState(42)
n = 100
xyz = rng.randn(n, 3).astype(np.float32)
# Very uniform energy (tiny noise)
traj = xyz[:, None, :] + rng.randn(n, 60, 3).astype(np.float32) * 0.001

roi_mask, snr, info = motion_gate(xyz, traj)
print('Test 1 (all-static):')
print('  degenerate:', info['degenerate'])
print('  all True:', roi_mask.all())
assert info['degenerate'] == True
assert roi_mask.all()

# Test 2: scene with movers should gate out static background
n_moving = 100
n_static = 100
n_frames = 60
drive_freq = 3
t = np.arange(n_frames)

body_pts = rng.randn(n_moving, 3) * 0.1 + np.array([[0.0, 0.0, 0.0]])
motion = np.zeros((n_moving, n_frames, 3))
motion[:, :, 0] = 0.3 * np.sin(2 * np.pi * drive_freq * t / n_frames)[None, :]
body_traj = body_pts[:, None, :] + motion

static_pts = rng.randn(n_static, 3) * 2.0 + np.array([[5.0, 5.0, 5.0]])
static_traj = static_pts[:, None, :] + rng.randn(n_static, n_frames, 3) * 0.005

xyz2 = np.concatenate([body_pts, static_pts]).astype(np.float32)
traj2 = np.concatenate([body_traj, static_traj]).astype(np.float32)

roi_mask2, snr2, info2 = motion_gate(xyz2, traj2, drive_freq=drive_freq)
print('Test 2 (movers + static):')
print('  degenerate:', info2['degenerate'])
print('  movers in ROI:', roi_mask2[:n_moving].mean())
print('  static in ROI:', roi_mask2[n_moving:].mean())
print('  recall:', roi_mask2[:n_moving].mean())
print('  precision:', (roi_mask2[:n_moving].sum() / roi_mask2.sum() if roi_mask2.sum() else 0))
assert info2['degenerate'] == False
assert roi_mask2[:n_moving].mean() >= 0.9  # high recall

print('All smoke tests passed!')
