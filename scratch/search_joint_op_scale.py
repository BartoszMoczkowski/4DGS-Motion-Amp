import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

m2_base = eval_mask(d_m2, np.ones(len(d_m2["pairs"]), dtype=bool))
m8_base = eval_mask(d_m8, np.ones(len(d_m8["pairs"]), dtype=bool))
print(f"BASELINE M2: ARI={m2_base['ari_dil']:.4f}, CatB={m2_base['n_catb']}/102, 141={m2_base['n_141']}/141, seeds={m2_base['seeds']}")
print(f"BASELINE M8: ARI={m8_base['ari_dil']:.4f}, CatB={m8_base['n_catb']}/102, 141={m8_base['n_141']}/141, seeds={m8_base['seeds']}")

candidates = []

# Let's explore:
# w = d_ij * (1 + penalty_op) * (1 + penalty_scale)
# Where:
# penalty_op = lam_op * max(0, tau_op - op_prod)
# penalty_scale = lam_scale * max(0, vol_max - v0)
for tau_op in [0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
    for lam_op in [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        for v0_mm in [2.0, 2.5, 3.0, 4.0]:
            for lam_scale in [0.0, 100.0, 500.0, 1000.0]:
                v0 = v0_mm / 1000.0

                pen_op_2 = lam_op * np.maximum(0.0, tau_op - d_m2['op_prod'])
                pen_sc_2 = lam_scale * np.maximum(0.0, d_m2['vol_max'] - v0)
                w_m2 = d_m2['d_ij'] * (1.0 + pen_op_2) * (1.0 + pen_sc_2)
                res_m2 = eval_mask(d_m2, w_m2 <= 0.003)

                # M2 gate: Cat B split increases, ARI drop <= 0.02
                if res_m2['n_catb'] > 20 and res_m2['ari_dil'] >= 0.4035:
                    pen_op_8 = lam_op * np.maximum(0.0, tau_op - d_m8['op_prod'])
                    pen_sc_8 = lam_scale * np.maximum(0.0, d_m8['vol_max'] - v0)
                    w_m8 = d_m8['d_ij'] * (1.0 + pen_op_8) * (1.0 + pen_sc_8)
                    res_m8 = eval_mask(d_m8, w_m8 <= 0.003)

                    # M8 gate: ARI >= 0.38468
                    if res_m8['ari_dil'] >= 0.3846:
                        candidates.append({
                            'tau_op': tau_op, 'lam_op': lam_op, 'v0_mm': v0_mm, 'lam_scale': lam_scale,
                            'm2': res_m2, 'm8': res_m8
                        })
                        print(f"FOUND! tau_op={tau_op:.3f}, lam_op={lam_op:5.1f}, v0={v0_mm:.1f}mm, lam_sc={lam_scale:5.0f} | "
                              f"M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']:2d}/102 seeds={res_m2['seeds']} | "
                              f"M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']}")

print(f"Total matching candidates: {len(candidates)}")
