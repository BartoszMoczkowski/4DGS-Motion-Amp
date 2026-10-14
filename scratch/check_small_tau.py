import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

for tau in [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035]:
    for lam in [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
        pen_2 = np.maximum(0.0, tau - d_m2['op_prod'])
        w_2 = d_m2['d_ij'] * (1.0 + lam * pen_2)
        res_m2 = eval_mask(d_m2, w_2 <= 0.003)

        pen_8 = np.maximum(0.0, tau - d_m8['op_prod'])
        w_8 = d_m8['d_ij'] * (1.0 + lam * pen_8)
        res_m8 = eval_mask(d_m8, w_8 <= 0.003)

        print(f"tau={tau:.4f}, lam={lam:5.1f} | M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']}/102 seeds={res_m2['seeds']} | M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']}")
