import numpy as np
import json
from pxr import Usd, UsdGeom, Gf

usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
stage = Usd.Stage.Open(usd_path)

part_prims = {}
for p in stage.Traverse():
    if p.GetTypeName() == "Xform" and p.GetParent().GetName() == "CONJUNTO_BOMBAS":
        part_prims[p.GetName()] = p

ordered_names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, 107)]
num_frames = 60
time_codes = np.linspace(0, stage.GetEndTimeCode(), num_frames)

# Evaluate SE(3) transforms M(t) (4x4 matrix per frame) for all 107 parts
transforms = np.zeros((107, num_frames, 4, 4))
for idx, name in enumerate(ordered_names):
    prim = part_prims[name]
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    for t_i, tc in enumerate(time_codes):
        if not ops:
            M = np.eye(4)
        else:
            mat_gf = ops[0].Get(Usd.TimeCode(tc))
            M = np.eye(4) if mat_gf is None else np.array(mat_gf, dtype=np.float64).T
        transforms[idx, t_i] = M

# Displacement trajectory relative to t=0 for each part:
# For rigid motion of a part with centroid c_i, the displacement of any point r is:
# d_i(r, t) = R_i(t) (r - c_i) + c_i + T_i(t) - r
# Let's inspect pure translation displacement T_i(t) - T_i(0):
disp_t = transforms[:, :, :3, 3] - transforms[:, :1, :3, 3]  # (107, 60, 3)

# Reshape trajectory into 180D vector per part
traj_vecs = disp_t.reshape(107, -1)  # (107, 180)

# Check norms (amplitude of trajectory)
norms = np.linalg.norm(traj_vecs, axis=1)

print(f"Static parts (norm < 1e-5): {(norms < 1e-5).sum()} parts (label 0: {ordered_names[0]})")

# Let's compute normalized trajectory vectors for moving parts
norm_trajs = np.zeros_like(traj_vecs)
moving_indices = np.where(norms >= 1e-5)[0]
for idx in moving_indices:
    norm_trajs[idx] = traj_vecs[idx] / norms[idx]

# Pairwise cosine distance (1 - cos_sim) on normalized trajectories
cos_sim = norm_trajs @ norm_trajs.T  # (107, 107)

# Check pairs of parts with very high trajectory similarity (cos_sim > threshold, e.g. 0.95 or 0.98 or 0.99)
print("\n--- Pairwise similarity analysis ---")
for thresh in [0.90, 0.95, 0.98, 0.99, 0.995]:
    pairs = []
    for i in range(len(moving_indices)):
        for j in range(i+1, len(moving_indices)):
            idx_i = moving_indices[i]
            idx_j = moving_indices[j]
            if cos_sim[idx_i, idx_j] >= thresh:
                pairs.append((idx_i, idx_j, cos_sim[idx_i, idx_j]))
    print(f"Threshold cos >= {thresh}: {len(pairs)} part pairs exceed threshold")
    if thresh >= 0.99 and len(pairs) > 0:
        for i, j, s in pairs:
            print(f"  {ordered_names[i]} (norm={norms[i]:.2f}) <-> {ordered_names[j]} (norm={norms[j]:.2f}): cos={s:.5f}")

