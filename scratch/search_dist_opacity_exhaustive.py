import sys
sys.path.append('scratch')
import numpy as np
from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

# Let's search over:
# w = d_ij * (1.0 + lam_op * penalty_op + lam_vol * penalty_vol)
# where:
# penalty_op = max(0, tau_op - op_prod) * (max(0, d_ij - d_min) / (0.003 - d_min))
# penalty_vol = max(0, vol_max - v0) / v0

best_m2_catb = 0
passing_configs = []

for d_min in [0.0, 0.0015, 0.0020, 0.0023, 0.0025]:
    for tau_op in [0.002, 0.005, 0.008, 0.010, 0.015, 0.020]:
        for lam_op in [5.0, 10.0, 20.0, 50.0, 100.0]:
            dist_factor_2 = 1.0 if d_min == 0.0 else np.maximum(0.0, (d_m2['d_ij'] - d_min) / (0.003 - d_min))
            pen_op_2 = np.maximum(0.0, tau_op - d_m2['op_prod']) * dist_factor_2
            w_2 = d_m2['d_ij'] * (1.0 + lam_op * pen_op_2)
            res_m2 = eval_mask(d_m2, w_2 <= 0.003)

            # Check if M2 improves Cat B and keeps ARI >= 0.4035
            if res_m2['n_catb'] > 20 and res_m2['ari_dil'] >= 0.4035:
                dist_factor_8 = 1.0 if d_min == 0.0 else np.maximum(0.0, (d_m8['d_ij'] - d_min) / (0.003 - d_min))
                pen_op_8 = np.maximum(0.0, tau_op - d_m8['op_prod']) * dist_factor_8
                w_8 = d_m8['d_ij'] * (1.0 + lam_op * pen_op_8)
                res_m8 = eval_mask(d_m8, w_8 <= 0.003)

                is_pass_m8 = res_m8['ari_dil'] >= 0.38468
                print(f"M2 Candidate: d_min={d_min*1000:4.1f}mm, tau={tau_op:.3f}, lam={lam_op:5.1f} | "
                      f"M2: ARI={res_m2['ari_dil']:.4f} CatB={res_m2['n_catb']:2d}/102 seeds={res_m2['seeds']} | "
                      f"M8: ARI={res_m8['ari_dil']:.4f} seeds={res_m8['seeds']} | pass_both={is_pass_m8}")
                passing_configs.append({
                    'd_min': d_min, 'tau_op': tau_op, 'lam_op': lam_op,
                    'm2': res_m2, 'm8': res_m8, 'pass_both': is_pass_m8
                })

print(f"Total configs evaluated that improve M2: {len(passing_configs)}")
pass_both = [c for c in passing_configs if c['pass_both']]
print(f"Configs passing both gates: {len(pass_both)}")
