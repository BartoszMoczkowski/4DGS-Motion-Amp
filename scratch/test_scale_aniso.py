import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

# Let's test scale and anisotropy features:
# On M2, earlier we saw M2 Bridge 65-90 had vol = 4.98mm, aniso = 5.7!
# What if:
# w = d_ij * (1.0 + lam_vol * max(0, vol_max - v0))
# or w = d_ij * (1.0 + lam_aniso * max(0, a0 - aniso_min)) (penalizing spherical/low-anisotropy floaters)
print("=== Testing Scale & Anisotropy features ===")
for v0_mm in [1.5, 2.0, 2.5, 3.0]:
    for lam_v in [10.0, 50.0, 100.0, 200.0, 500.0]:
        v0 = v0_mm / 1000.0
        pen_v_2 = np.maximum(0.0, d_m2['vol_max'] - v0)
        w_2 = d_m2['d_ij'] * (1.0 + lam_v * pen_v_2)
        res_m2 = eval_mask(d_m2, w_2 <= 0.003)

        if res_m2['n_catb'] > 20 and res_m2['ari_dil'] >= 0.4035:
            pen_v_8 = np.maximum(0.0, d_m8['vol_max'] - v0)
            w_8 = d_m8['d_ij'] * (1.0 + lam_v * pen_v_8)
            res_m8 = eval_mask(d_m8, w_8 <= 0.003)
            print(f"Scale candidate: v0={v0_mm}mm, lam={lam_v} | M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']}/102 seeds={res_m2['seeds']} | M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']}")

for a0 in [5.0, 10.0, 15.0, 20.0]:
    for lam_a in [1.0, 5.0, 10.0, 20.0]:
        # Spherical bridging Gaussians have low anisotropy (aniso < a0)
        # We penalize low anisotropy: max(0, a0 - aniso)
        pen_a_2 = np.maximum(0.0, a0 - np.minimum(d_m2['aniso'][d_m2['pairs'][:,0]], d_m2['aniso'][d_m2['pairs'][:,1]]))
        w_2 = d_m2['d_ij'] * (1.0 + lam_a * pen_a_2)
        res_m2 = eval_mask(d_m2, w_2 <= 0.003)
        if res_m2['n_catb'] > 20 and res_m2['ari_dil'] >= 0.4035:
            pen_a_8 = np.maximum(0.0, a0 - np.minimum(d_m8['aniso'][d_m8['pairs'][:,0]], d_m8['aniso'][d_m8['pairs'][:,1]]))
            w_8 = d_m8['d_ij'] * (1.0 + lam_a * pen_a_8)
            res_m8 = eval_mask(d_m8, w_8 <= 0.003)
            print(f"Aniso candidate: a0={a0}, lam={lam_a} | M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']}/102 seeds={res_m2['seeds']} | M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']}")
