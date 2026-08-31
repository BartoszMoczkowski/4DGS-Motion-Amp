import numpy as np
from pxr import Usd, UsdGeom
from omniverse_pipeline import add_motion
from scipy.cluster.hierarchy import fcluster, linkage

usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
stage = Usd.Stage.Open(usd_path)

parts = add_motion.collect_parts(stage, "CONJUNTO_BOMBAS", UsdGeom, Usd)
part_prims = {p.GetName(): p for p, _, _ in parts}
ordered_names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, 107)]

num_frames = 60
time_codes = np.linspace(0, stage.GetEndTimeCode(), num_frames)

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

# Trajectories relative to t=0
disp_t = transforms[:, :, :3, 3] - transforms[:, :1, :3, 3] # (107, 60, 3)
traj_vecs = disp_t.reshape(107, -1) # (107, 180)
norms = np.linalg.norm(traj_vecs, axis=1)

is_static = norms < 1e-5

# Normalized trajectories for moving parts
norm_trajs = np.zeros_like(traj_vecs)
moving_indices = np.where(~is_static)[0]
for idx in moving_indices:
    norm_trajs[idx] = traj_vecs[idx] / norms[idx]

# Pairwise cosine similarity matrix
cos_sim = norm_trajs @ norm_trajs.T # (107, 107)

# Distance matrix d_ij = 1 - |cos_sim_ij|
dist_matrix = np.clip(1.0 - np.abs(cos_sim), 0.0, 2.0)

# Build hierarchical linkage on moving parts
dist_condensed = []
for i in range(len(moving_indices)):
    for j in range(i+1, len(moving_indices)):
        idx_i = moving_indices[i]
        idx_j = moving_indices[j]
        dist_condensed.append(dist_matrix[idx_i, idx_j])

Z = linkage(dist_condensed, method="complete")

print(f"Total parts: {len(ordered_names)}")
print(f"Static parts: {is_static.sum()} ({ordered_names[0]})")
print(f"Moving parts: {(~is_static).sum()}")

print("\n--- Clustering results at various distance cutoffs (1 - |cos_sim|) ---")
for cutoff_deg, cutoff in [("1 deg (cos=0.9998)", 1 - np.cos(np.radians(1))),
                           ("5 deg (cos=0.9962)", 1 - np.cos(np.radians(5))),
                           ("10 deg (cos=0.9848)", 1 - np.cos(np.radians(10))),
                           ("15 deg (cos=0.9659)", 1 - np.cos(np.radians(15))),
                           ("20 deg (cos=0.9397)", 1 - np.cos(np.radians(20))),
                           ("25 deg (cos=0.9063)", 1 - np.cos(np.radians(25)))]:
    clusters = fcluster(Z, t=cutoff, criterion="distance")
    n_moving_clusters = len(np.unique(clusters))
    total_classes = 1 + n_moving_clusters # 1 static class + n_moving_clusters
    
    # Map back to 107 labels
    labels = np.zeros(107, dtype=int)
    labels[0] = 0 # static class
    for idx_in_moving, cluster_id in enumerate(clusters):
        part_idx = moving_indices[idx_in_moving]
        labels[part_idx] = cluster_id # 1..n_moving_clusters
        
    counts = np.bincount(labels)
    n_static = counts[0]
    n_singletons = (counts[1:] == 1).sum()
    n_coupled_classes = (counts[1:] > 1).sum()
    n_coupled_parts = counts[1:][counts[1:] > 1].sum()
    
    print(f"\nCutoff {cutoff_deg}:")
    print(f"  Total GT motion classes: {total_classes}")
    print(f"  (a) Static parts: {n_static}")
    print(f"  (b) Independently moving (singletons): {n_singletons}")
    print(f"  (c) Kinematically coupled duplicate parts: {n_coupled_parts} across {n_coupled_classes} coupled groups")
