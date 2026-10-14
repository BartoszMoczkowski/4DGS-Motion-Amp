import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

med_m2_op2 = np.median(d_m2["op"])**2
med_m8_op2 = np.median(d_m8["op"])**2
print(f"med_m2_op2: {med_m2_op2:.5f}, med_m8_op2: {med_m8_op2:.5f}")

for tau_rel in [0.1, 0.2, 0.3, 0.4, 0.5]:
    tau_m2 = tau_rel * med_m2_op2
    tau_m8 = tau_rel * med_m8_op2
    for lam in [5.0, 10.0, 20.0, 30.0, 50.0]:
        pen_2 = np.maximum(0.0, tau_m2 - d_m2['op_prod'])
        w_2 = d_m2['d_ij'] * (1.0 + lam * pen_2)
        res_m2 = eval_mask(d_m2, w_2 <= 0.003)

        pen_8 = np.maximum(0.0, tau_m8 - d_m8['op_prod'])
        w_8 = d_m8['d_ij'] * (1.0 + lam * pen_8)
        res_m8 = eval_mask(d_m8, w_8 <= 0.003)

        m2_pass = (res_m2['ari_dil'] >= 0.4035) and (res_m2['n_catb'] > 20)
        m8_pass = (res_m8['ari_dil'] >= 0.3847)
        if m2_pass or m8_pass:
            print(f"tau_rel={tau_rel:.2f} (tau2={tau_m2:.4f}, tau8={tau_m8:.4f}), lam={lam:4.1f} | "
                  f"M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']:2d}/102 seeds={res_m2['seeds']} (pass={m2_pass}) | "
                  f"M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']} (pass={m8_pass})")
