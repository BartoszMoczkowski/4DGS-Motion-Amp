import numpy as np
from pxr import Usd, UsdGeom
from omniverse_pipeline import add_motion

usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
stage = Usd.Stage.Open(usd_path)

parts = add_motion.collect_parts(stage, "CONJUNTO_BOMBAS", UsdGeom, Usd)
movable = [(p, c, r) for (p, c, r) in parts if p.GetName() != "frame_base"]

base_amp_mm = 20
multiplier = 2
amplified_name = "part_102"
num_frames = 240
fps = 60.0
freq_hz = 10.0

mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
units_per_mm = 1e-3 / mpu
cycles = int(round(freq_hz * (num_frames - 1) / fps))
cfg = dict(
    trans_amp=[0.5 * base_amp_mm * units_per_mm, 1.5 * base_amp_mm * units_per_mm],
    rot_surface=[0.25 * base_amp_mm * units_per_mm, 0.75 * base_amp_mm * units_per_mm],
    rot_deg_max=3.0,
    freq=[cycles, cycles],
)
rng = np.random.default_rng(0)

part_params = {}
part_params["frame_base"] = {
    "trans_amp": 0.0, "trans_dir": np.array([0.,0.,0.]), "trans_freq": 0, "trans_phase": 0.0,
    "rot_amp": 0.0, "rot_dir": np.array([0.,0.,0.]), "rot_freq": 0, "rot_phase": 0.0,
}

ordered_names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, 107)]

for prim, c, r in movable:
    name = prim.GetName()
    mp = add_motion._rand_motion(cfg, rng, r)
    if name == amplified_name:
        mp["trans_amp"] *= multiplier
        mp["rot_amp"] *= multiplier
    part_params[name] = mp

# Let's inspect pairwise similarity of parameters and trajectories!
# Trajectories (60 frames)
num_eval_frames = 60
time_codes = np.linspace(0, stage.GetEndTimeCode(), num_eval_frames)

part_prims = {p.GetName(): p for p, _, _ in parts}
transforms = np.zeros((107, num_eval_frames, 4, 4))
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

disp_t = transforms[:, :, :3, 3] - transforms[:, :1, :3, 3] # (107, 60, 3)
traj_vecs = disp_t.reshape(107, -1)
norms = np.linalg.norm(traj_vecs, axis=1)

# Check pairs of parts:
print("--- Checking duplicate / coupled parts based on trajectory correlation ---")
cos_sim = np.zeros((107, 107))
for i in range(107):
    if norms[i] < 1e-5: continue
    v_i = traj_vecs[i] / norms[i]
    for j in range(107):
        if norms[j] < 1e-5: continue
        v_j = traj_vecs[j] / norms[j]
        cos_sim[i, j] = np.abs(np.dot(v_i, v_j))

for cos_thresh in [0.95, 0.98, 0.99, 0.995, 0.999]:
    # Group parts using connected components where cos_sim >= cos_thresh
    adj = (cos_sim >= cos_thresh)
    np.fill_diagonal(adj, True)
    visited = set()
    groups = []
    for i in range(107):
        if i in visited: continue
        if norms[i] < 1e-5:
            groups.append([i])
            visited.add(i)
            continue
        # find connected component
        comp = []
        q = [i]
        visited.add(i)
        while q:
            curr = q.pop(0)
            comp.append(curr)
            for neighbor in range(107):
                if neighbor not in visited and adj[curr, neighbor]:
                    visited.add(neighbor)
                    q.append(neighbor)
        groups.append(comp)
    
    n_static = sum(1 for g in groups if norms[g[0]] < 1e-5)
    n_coupled_groups = sum(1 for g in groups if len(g) > 1 and norms[g[0]] >= 1e-5)
    n_coupled_parts = sum(len(g) for g in groups if len(g) > 1 and norms[g[0]] >= 1e-5)
    n_singletons = sum(1 for g in groups if len(g) == 1 and norms[g[0]] >= 1e-5)
    
    print(f"Thresh {cos_thresh:.4f} -> Total groups: {len(groups)} | Static: {n_static} | Coupled groups: {n_coupled_groups} ({n_coupled_parts} parts) | Singletons: {n_singletons}")
