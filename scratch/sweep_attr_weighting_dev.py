import sys
sys.path.append('scratch')
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index, compute_pairs_split_rate, PUMP01_BASELINE_MERGED_PAIRS_141
from search_joint_gates import load_data, eval_mask

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

d_m2 = load_data("grid-A20mm_M2")
r_cutoff = 0.003  # 0.3 cm

# Baseline on M2
m2_base = eval_mask(d_m2, np.ones(len(d_m2["pairs"]), dtype=bool))
print(f"BASELINE M2: ARI={m2_base['ari_dil']:.4f}, CatB={m2_base['catb']:.4f} ({m2_base['n_catb']}/102), 141={m2_base['p141']:.4f} ({m2_base['n_141']}/141), seeds={m2_base['seeds']}")

results = []

def test_weighting(name, w_edge):
    # prune/down-weight BEFORE reachability: edge retained if w_edge <= r_cutoff
    mask = (w_edge <= r_cutoff)
    res = eval_mask(d_m2, mask)
    ari_drop = m2_base['ari_dil'] - res['ari_dil']
    passes_gate = (ari_drop <= 0.02) and (res['n_catb'] > m2_base['n_catb'])
    print(f"{name:50s} | ARI={res['ari_dil']:.4f} (drop={ari_drop:+.4f}) | CatB={res['catb']:.4f} ({res['n_catb']:2d}/102) | 141={res['p141']:.4f} ({res['n_141']:2d}/141) | seeds={res['seeds']:2d} | pass={passes_gate}")
    return res

print("\n--- Family 1: Power-law opacity w = d_ij / (op_prod)^alpha ---")
for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    w = d_m2["d_ij"] / (d_m2["op_prod"] ** alpha)
    test_weighting(f"Power-law opacity alpha={alpha:.2f}", w)

print("\n--- Family 2: Linear opacity penalty w = d_ij * (1 + lambda * max(0, tau - op_prod)) ---")
for tau in [0.01, 0.02, 0.05, 0.10]:
    for lam in [1.0, 2.0, 5.0, 10.0, 20.0]:
        pen = np.maximum(0.0, tau - d_m2["op_prod"])
        w = d_m2["d_ij"] * (1.0 + lam * pen)
        test_weighting(f"Linear op tau={tau:.2f}, lam={lam:4.1f}", w)

print("\n--- Family 3: Scale volume penalty w = d_ij * (1 + beta * max(0, vol - v0)) ---")
for v0_mm in [1.0, 1.5, 2.0, 2.5]:
    for beta in [100.0, 500.0, 1000.0, 2000.0]:
        v0 = v0_mm / 1000.0
        pen = np.maximum(0.0, d_m2["vol_max"] - v0)
        w = d_m2["d_ij"] * (1.0 + beta * pen)
        test_weighting(f"Scale vol v0={v0_mm:.1f}mm, beta={beta:5.0f}", w)

print("\n--- Family 4: Scale volume ratio w = d_ij * (1 + beta * (vol_max / 0.001)^p) ---")
for p in [1.0, 2.0]:
    for beta in [0.1, 0.2, 0.5, 1.0]:
        w = d_m2["d_ij"] * (1.0 + beta * ((d_m2["vol_max"] / 0.001) ** p))
        test_weighting(f"Scale vol ratio p={p:.1f}, beta={beta:.1f}", w)

print("\n--- Family 5: Joint opacity + scale volume ---")
for alpha in [0.02, 0.03, 0.05]:
    for beta in [200.0, 500.0, 1000.0]:
        v0 = 0.0015
        pen_vol = np.maximum(0.0, d_m2["vol_max"] - v0)
        w = (d_m2["d_ij"] / (d_m2["op_prod"] ** alpha)) * (1.0 + beta * pen_vol)
        test_weighting(f"Joint alpha={alpha:.2f}, beta={beta:.0f}", w)
