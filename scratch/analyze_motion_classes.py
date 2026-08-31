import numpy as np
import json
from pxr import Usd, UsdGeom, Gf

usd_path = "omniverse-pipeline/data/scenes/grid/pump_A20mm_M2_animated.usd"
stage = Usd.Stage.Open(usd_path)

parts = []
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
grp = stage.GetPrimAtPath("/World/CONJUNTO_BOMBAS")

# Collect 107 parts in label order: 0: frame_base, 1..106: part_001..part_106
part_prims = {}
for p in stage.Traverse():
    if p.GetTypeName() == "Xform" and p.GetParent().GetName() == "CONJUNTO_BOMBAS":
        part_prims[p.GetName()] = p

print(f"Found {len(part_prims)} part prims in USD.")

# Build ordered list of 107 prims: index 0 is frame_base, 1..106 are part_001..part_106
ordered_names = ["frame_base"] + [f"part_{i:03d}" for i in range(1, 107)]

num_frames = 60
time_codes = np.linspace(0, stage.GetEndTimeCode(), num_frames)

# For each part, evaluate its transform matrix M(t) at all 60 time codes
# M(t) is 4x4
transforms = np.zeros((107, num_frames, 4, 4))
translations = np.zeros((107, num_frames, 3))

for idx, name in enumerate(ordered_names):
    prim = part_prims[name]
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    for t_i, tc in enumerate(time_codes):
        if not ops:
            M = np.eye(4)
        else:
            # Evaluate transform op at timecode tc
            mat_gf = ops[0].Get(Usd.TimeCode(tc))
            if mat_gf is None:
                M = np.eye(4)
            else:
                M = np.array(mat_gf, dtype=np.float64).T  # USD Matrix4d to numpy
        transforms[idx, t_i] = M
        translations[idx, t_i] = M[:3, 3]

# Compute max displacement (motion magnitude) for each part relative to frame 0
motion_mag = np.zeros(107)
for i in range(107):
    disp = np.linalg.norm(translations[i] - translations[i, 0], axis=-1)
    motion_mag[i] = disp.max()

print("Motion magnitudes summary:")
print(f"Static parts (disp max < 1e-5): {(motion_mag < 1e-5).sum()}")
print(f"Part 0 (frame_base) motion mag: {motion_mag[0]}")
print(f"Min moving part motion mag: {motion_mag[1:].min()}")
print(f"Max moving part motion mag: {motion_mag[1:].max()}")
print(f"Mean moving part motion mag: {motion_mag[1:].mean()}")

# Check pairwise trajectory correlation / RMS distance normalized by amplitude
# For parts with nonzero motion, center their translation curves:
centered_trans = translations - translations[:, :1]  # shape (107, 60, 3)

# Compute normalized directional displacement curve d_i(t) in R^3
# d_i(t) = T_i(t) - T_i(0)
# Check if any pairs of parts have parallel/proportional motion vectors across all t
# d_i(t) = a * d_j(t) for all t
pairwise_cos = np.zeros((107, 107))
for i in range(107):
    vec_i = (translations[i] - translations[i, 0]).reshape(-1) # 180-dim
    norm_i = np.linalg.norm(vec_i)
    if norm_i < 1e-5:
        continue
    for j in range(107):
        vec_j = (translations[j] - translations[j, 0]).reshape(-1)
        norm_j = np.linalg.norm(vec_j)
        if norm_j < 1e-5:
            continue
        pairwise_cos[i, j] = np.abs(np.dot(vec_i, vec_j) / (norm_i * norm_j))

print(f"Max off-diagonal pairwise absolute cosine correlation between moving parts:")
moving_cos = pairwise_cos[1:, 1:]
np.fill_diagonal(moving_cos, 0)
print(f"Max cos correlation: {moving_cos.max():.4f}")
print(f"Histogram of cos correlation > 0.9: {(moving_cos > 0.9).sum()}")
print(f"Histogram of cos correlation > 0.99: {(moving_cos > 0.99).sum()}")
