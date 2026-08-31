import numpy as np
from pxr import Usd, UsdGeom
from omniverse_pipeline import add_motion

usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
stage = Usd.Stage.Open(usd_path)

parts = add_motion.collect_parts(stage, "CONJUNTO_BOMBAS", UsdGeom, Usd)
movable = [(p, c, r) for (p, c, r) in parts if p.GetName() != "frame_base"]

print(f"Total parts: {len(parts)}, Movable: {len(movable)}")

base_amp_mm = 20
multiplier = 2
amplified_name = "part_102"
num_frames = 240 # stage frame count
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

for prim, c, r in movable:
    name = prim.GetName()
    mp = add_motion._rand_motion(cfg, rng, r)
    if name == amplified_name:
        mp["trans_amp"] *= multiplier
        mp["rot_amp"] *= multiplier
    part_params[name] = mp

print("Generated motion params for all 107 parts!")
print("Sample params for part_001:")
print(part_params["part_001"])
