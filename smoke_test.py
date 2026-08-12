import sys
sys.path.insert(0, 'orchestrator')
from pipeline.vendored.host.kabsch_em import weighted_kabsch
import numpy as np
pts = np.random.randn(50, 3)
R, tau = weighted_kabsch(pts, pts)
print('identity Kabsch:', 'PASS' if np.allclose(R, np.eye(3)) else 'FAIL')

from pipeline.vendored.host.kabsch_em import _compute_residuals
N, T = 10, 5
K = 3
traj = np.random.randn(N, T, 3)
R2 = np.stack([np.eye(3) for _ in range(K * T)]).reshape(K, T, 3, 3)
tau2 = np.zeros((K, T, 3))
xyz = np.random.randn(N, 3)
r = _compute_residuals(traj, R2, tau2, xyz)
print('residuals shape:', r.shape, 'PASS' if r.shape == (N, K) else 'FAIL')
