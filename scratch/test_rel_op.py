import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index, compute_pairs_split_rate, PUMP01_BASELINE_MERGED_PAIRS_141

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

from search_joint_gates import load_data, eval_mask

d_m2 = load_data("grid-A20mm_M2")
d_m8 = load_data("grid-A40mm_M8")

# Let's test relative opacity: tau_rel * median(op)^2
med_m2_op2 = np.median(d_m2["op"])**2
med_m8_op2 = np.median(d_m8["op"])**2
print(f"med_m2_op2: {med_m2_op2:.5f}, med_m8_op2: {med_m8_op2:.5f}")

for tau_rel in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    m2_res = eval_mask(d_m2, d_m2["op_prod"] >= tau_rel * med_m2_op2)
    m8_res = eval_mask(d_m8, d_m8["op_prod"] >= tau_rel * med_m8_op2)
    print(f"tau_rel={tau_rel:4.2f} | "
          f"M2: ARI={m2_res['ari_dil']:.4f} CatB={m2_res['catb']:.4f} ({m2_res['n_catb']:2d}/102) seeds={m2_res['seeds']} | "
          f"M8: ARI={m8_res['ari_dil']:.4f} (pass={m8_res['ari_dil']>=0.3847}) seeds={m8_res['seeds']}")
