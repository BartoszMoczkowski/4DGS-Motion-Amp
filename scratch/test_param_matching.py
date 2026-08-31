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

params_list = []
params_list.append({
    "name": "frame_base", "trans_amp": 0.0, "trans_dir": np.array([0,0,0]),
    "trans_freq": 0, "trans_phase": 0.0, "rot_amp": 0.0, "rot_dir": np.array([0,0,0]),
    "rot_freq": 0, "rot_phase": 0.0
})

for prim, c, r in movable:
    name = prim.GetName()
    mp = add_motion._rand_motion(cfg, rng, r)
    if name == amplified_name:
        mp["trans_amp"] *= multiplier
        mp["rot_amp"] *= multiplier
    mp["name"] = name
    params_list.append(mp)

# Compare both direction and phase (or flipped direction and phase + pi)
matched_pairs = []
for i in range(1, 107):
    for j in range(i+1, 107):
        p_i = params_list[i]
        p_j = params_list[j]
        # dir cos
        dot_d = np.dot(p_i["trans_dir"], p_j["trans_dir"])
        cos_d = np.abs(dot_d)
        
        # phase diff
        if dot_d > 0:
            phase_diff = np.abs(p_i["trans_phase"] - p_j["trans_phase"]) % (2*np.pi)
        else:
            phase_diff = np.abs(p_i["trans_phase"] - (p_j["trans_phase"] + np.pi)) % (2*np.pi)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)
        
        if cos_d > 0.98 and phase_diff < 0.2: # ~11 degrees phase
            matched_pairs.append((i, j, cos_d, phase_diff))

print(f"Matched pairs (cos(trans_dir) > 0.98 AND phase_diff < 0.2 rad): {len(matched_pairs)}")
for i, j, c, pd in matched_pairs:
    print(f"  {params_list[i]['name']} <-> {params_list[j]['name']}: cos(dir)={c:.4f}, phase_diff={pd:.4f} rad")
