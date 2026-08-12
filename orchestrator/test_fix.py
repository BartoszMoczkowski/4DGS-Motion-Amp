from pipeline.vendored.host.kabsch_em import _compute_residuals
import numpy as np

N, T = 10, 5
K = 3
traj = np.random.randn(N, T, 3)
R = np.stack([np.eye(3) for _ in range(K * T)]).reshape(K, T, 3, 3)
tau = np.zeros((K, T, 3))
xyz = np.random.randn(N, 3)
r = _compute_residuals(traj, R, tau, xyz)
print('residuals shape:', r.shape, 'OK' if r.shape == (N, K) else 'FAIL')
