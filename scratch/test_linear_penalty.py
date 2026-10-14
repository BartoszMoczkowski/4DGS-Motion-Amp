import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

print("--- Testing tau x lam grid on M2 and M8 ---")
for tau in [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020]:
    for lam in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        pen_m2 = np.maximum(0.0, tau - d_m2['op_prod'])
        w_m2 = d_m2['d_ij'] * (1.0 + lam * pen_m2)
        res_m2 = eval_mask(d_m2, w_m2 <= 0.003)

        pen_m8 = np.maximum(0.0, tau - d_m8['op_prod'])
        w_m8 = d_m8['d_ij'] * (1.0 + lam * pen_m8)
        res_m8 = eval_mask(d_m8, w_m8 <= 0.003)

        m2_pass = (res_m2['ari_dil'] >= 0.4035) and (res_m2['n_catb'] > 20)
        m8_pass = (res_m8['ari_dil'] >= 0.3847)

        if m2_pass or m8_pass:
            print(f"tau={tau:.3f}, lam={lam:5.1f} | "
                  f"M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']:2d}/102 seeds={res_m2['seeds']:2d} (pass={m2_pass}) | "
                  f"M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']:2d} (pass={m8_pass})")
