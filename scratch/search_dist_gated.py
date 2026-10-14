import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

# Let's test what functions allow M8 to keep ARI >= 0.3847 while improving M2!
print("--- Grid search over weighting functions on M2 and M8 ---")

# Idea 1: Distance-gated opacity penalty: only penalize if d_ij > d_gate
# Since bridging edges have median dist 2.3mm (max 2.95mm), what if d_gate = 2.2mm or 2.5mm?
for d_gate in [0.0020, 0.0022, 0.0024, 0.0025, 0.0026, 0.0027]:
    for tau in [0.005, 0.010, 0.020, 0.050]:
        for lam in [10.0, 20.0, 50.0, 100.0]:
            # penalty only applies to long edges: (d_ij - d_gate)+
            dist_factor = np.maximum(0.0, (d_m2['d_ij'] - d_gate) / (0.003 - d_gate))
            pen_m2 = np.maximum(0.0, tau - d_m2['op_prod']) * dist_factor
            w_m2 = d_m2['d_ij'] * (1.0 + lam * pen_m2)
            res_m2 = eval_mask(d_m2, w_m2 <= 0.003)

            dist_factor_8 = np.maximum(0.0, (d_m8['d_ij'] - d_gate) / (0.003 - d_gate))
            pen_m8 = np.maximum(0.0, tau - d_m8['op_prod']) * dist_factor_8
            w_m8 = d_m8['d_ij'] * (1.0 + lam * pen_m8)
            res_m8 = eval_mask(d_m8, w_m8 <= 0.003)

            m2_pass = (res_m2['ari_dil'] >= 0.4035)
            m8_pass = (res_m8['ari_dil'] >= 0.3840)
            if res_m8['ari_dil'] >= 0.35:
                print(f"d_gate={d_gate*1000:.1f}mm, tau={tau:.3f}, lam={lam:4.1f} | "
                      f"M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']:2d}/102 seeds={res_m2['seeds']} | "
                      f"M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']}")

