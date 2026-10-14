import sys
sys.path.append('scratch')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

from pipeline.vendored.host.metrics import (
    adjusted_rand_index,
    compute_pairs_split_rate,
    PUMP01_BASELINE_MERGED_PAIRS_141
)
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

def fast_best_iou_matching(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    u_true, inv_true = np.unique(labels_true, return_inverse=True)
    u_pred, inv_pred = np.unique(labels_pred, return_inverse=True)
    n_true = len(u_true)
    n_pred = len(u_pred)

    mat = coo_matrix((np.ones(len(labels_true)), (inv_true, inv_pred)), shape=(n_true, n_pred)).tocsr()
    if n_pred > 200:
        cand_cols = np.unique(mat.indices)
        mat_sub = mat[:, cand_cols].toarray()
        pred_sizes = np.bincount(inv_pred)[cand_cols]
    else:
        cand_cols = np.arange(n_pred)
        mat_sub = mat.toarray()
        pred_sizes = np.bincount(inv_pred)

    true_sizes = np.bincount(inv_true)
    union = true_sizes[:, None] + pred_sizes[None, :] - mat_sub
    iou = mat_sub / np.maximum(union, 1)

    row, col = linear_sum_assignment(-iou)
    matched_sum = float(iou[row, col].sum())
    mean_iou = matched_sum / n_true if n_true else 0.0
    return mean_iou

from search_joint_gates import load_data


cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

def load_cubes_data():
    run_dir = Path("runs/cubes-cubes_k2_ring")
    traj = np.load(run_dir / "trajectories.npz")
    xyz = traj["canonical_xyz"]
    gt_file = run_dir / "convert_out/data/multipleview/cubes_k2_ring/gt_segmentation.npz"
    gt = np.load(gt_file)
    _, nn = cKDTree(gt["points"]).query(xyz, k=1)
    gt_on_pred = gt["labels"][nn]

    ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
    ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
    ply_file = ply_files[-1]

    with open(ply_file, "rb") as f:
        header = b""
        while b"end_header\n" not in header: header += f.readline()
        h_len = f.tell()
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    for i in range(45): props.append((f"f_rest_{i}", "<f4"))
    props.extend([("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
                  ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")])
    dt = np.dtype(props)
    with open(ply_file, "rb") as f:
        f.seek(h_len)
        data = np.fromfile(f, dtype=dt)

    raw_op = data["opacity"]
    op = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
    scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1))
    vol = np.cbrt(scales.prod(axis=1))
    aniso = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-8)

    tree = cKDTree(xyz)
    r_base = 0.003
    pairs = tree.query_pairs(r=r_base, output_type="ndarray")
    d_ij = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    op_prod = op[pairs[:, 0]] * op[pairs[:, 1]]
    vol_max = np.maximum(vol[pairs[:, 0]], vol[pairs[:, 1]])
    aniso_max = np.maximum(aniso[pairs[:, 0]], aniso[pairs[:, 1]])

    return {
        "sub_xyz": xyz, "sub_gt": gt_on_pred, "n": len(xyz),
        "pairs": pairs, "d_ij": d_ij, "op_prod": op_prod,
        "vol_max": vol_max, "aniso_max": aniso_max,
        "op": op, "vol": vol, "aniso": aniso,
        "is_cubes": True
    }

def run_evaluation(d, mask, model_name, method_name, formula_str):
    n = d["n"]
    k_pairs = d["pairs"][mask]
    adj = csr_matrix((np.ones(len(k_pairs), dtype=bool), (k_pairs[:, 0], k_pairs[:, 1])), shape=(n, n))
    adj = adj + adj.T
    n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_comp = unique[counts >= 15]

    n_seeds = len(valid_comp)
    n_assigned_seed = int(counts[counts >= 15].sum()) if n_seeds > 0 else 0

    # 1. Dilated 1-NN
    labels_dilated = np.full(n, -1, dtype=np.int64)
    comp_map = {comp: idx for idx, comp in enumerate(valid_comp)}
    for comp, idx in comp_map.items():
        labels_dilated[raw_labels == comp] = idx
    unassigned = np.where(labels_dilated == -1)[0]
    assigned = np.where(labels_dilated >= 0)[0]
    if len(assigned) > 0 and len(unassigned) > 0:
        sub_tree = cKDTree(d["sub_xyz"][assigned])
        _, nearest = sub_tree.query(d["sub_xyz"][unassigned], k=1)
        labels_dilated[unassigned] = labels_dilated[assigned[nearest]]
    elif len(assigned) == 0:
        labels_dilated[:] = 0

    # 2. No dilation: cluster-(-1)
    labels_cminus1 = np.full(n, -1, dtype=np.int64)
    for comp, idx in comp_map.items():
        labels_cminus1[raw_labels == comp] = idx

    # 3. No dilation: singletons
    labels_singletons = labels_cminus1.copy()
    labels_singletons[unassigned] = np.arange(len(valid_comp), len(valid_comp) + len(unassigned))

    modes = [
        ("dilated_1nn", labels_dilated),
        ("no_dilation_cminus1", labels_cminus1),
        ("no_dilation_singletons", labels_singletons),
    ]

    is_cubes = d.get("is_cubes", False)
    rows = []

    for mode_name, pred_labels in modes:
        ari = float(adjusted_rand_index(d["sub_gt"], pred_labels))
        mean_iou = fast_best_iou_matching(d["sub_gt"], pred_labels)
        k_pred = len(np.unique(pred_labels))

        if not is_cubes:
            ps_141 = compute_pairs_split_rate(d["sub_gt"], pred_labels)
            ps_catb = compute_pairs_split_rate(d["sub_gt"], pred_labels, merged_pairs=cat_b_pairs)
            rate_141 = ps_141["pairs_split_rate"]
            n_141 = ps_141["n_pairs_split"]
            rate_catb = ps_catb["pairs_split_rate"]
            n_catb = ps_catb["n_pairs_split"]
        else:
            rate_141 = np.nan
            n_141 = np.nan
            rate_catb = np.nan
            n_catb = np.nan

        rows.append({
            "model": model_name,
            "method": method_name,
            "weighting_formula": formula_str,
            "radius_cm": 0.3,
            "dilation_mode": mode_name,
            "n_scored": n,
            "n_seeds": n_seeds,
            "n_assigned_seed": n_assigned_seed,
            "k_pred": k_pred,
            "ari": ari,
            "mean_iou": mean_iou,
            "pairs_split_rate_141": rate_141,
            "n_pairs_split_141": n_141,
            "cat_b_split_rate": rate_catb,
            "n_cat_b_split": n_catb
        })

    return rows

def main():
    print("Loading data for grid-A20mm_M2, grid-A40mm_M8, cubes-cubes_k2_ring...")
    d_m2 = load_data("grid-A20mm_M2")
    d_m8 = load_data("grid-A40mm_M8")
    d_cubes = load_cubes_data()

    all_rows = []

    # --- 1. BASELINE S6 SPATIAL CC ---
    print("Evaluating Baseline S6 Spatial CC on all models...")
    all_rows.extend(run_evaluation(
        d_m2, np.ones(len(d_m2["pairs"]), dtype=bool),
        "grid-A20mm_M2", "baseline_spatial_cc", "w = d_ij (binary radius)"
    ))
    all_rows.extend(run_evaluation(
        d_m8, np.ones(len(d_m8["pairs"]), dtype=bool),
        "grid-A40mm_M8", "baseline_spatial_cc", "w = d_ij (binary radius)"
    ))
    all_rows.extend(run_evaluation(
        d_cubes, np.ones(len(d_cubes["pairs"]), dtype=bool),
        "cubes-cubes_k2_ring", "baseline_spatial_cc", "w = d_ij (binary radius)"
    ))

    # --- 2. ATTRIBUTE-AWARE CC VARIANTS SWEPT ON DEV MODEL M2 ---
    print("Evaluating Attribute-Aware CC Variants on Dev Model M2...")

    # Variant 1: Linear Opacity (tau=0.010, lam=50) -> Top ARI + split boost
    pen_v1 = np.maximum(0.0, 0.010 - d_m2['op_prod'])
    w_v1 = d_m2['d_ij'] * (1.0 + 50.0 * pen_v1)
    all_rows.extend(run_evaluation(
        d_m2, w_v1 <= 0.003,
        "grid-A20mm_M2", "attr_cc_linear_op_t0.010_l50",
        "w = d_ij * (1 + 50 * max(0, 0.010 - op_prod))"
    ))

    # Variant 2: Linear Opacity (tau=0.020, lam=20) -> Balanced mild penalty
    pen_v2 = np.maximum(0.0, 0.020 - d_m2['op_prod'])
    w_v2 = d_m2['d_ij'] * (1.0 + 20.0 * pen_v2)
    all_rows.extend(run_evaluation(
        d_m2, w_v2 <= 0.003,
        "grid-A20mm_M2", "attr_cc_linear_op_t0.020_l20",
        "w = d_ij * (1 + 20 * max(0, 0.020 - op_prod))"
    ))

    # Variant 3: Linear Opacity (tau=0.015, lam=100) -> Aggressive split boost (24/102 Cat B, 30/141 total)
    pen_v3 = np.maximum(0.0, 0.015 - d_m2['op_prod'])
    w_v3 = d_m2['d_ij'] * (1.0 + 100.0 * pen_v3)
    all_rows.extend(run_evaluation(
        d_m2, w_v3 <= 0.003,
        "grid-A20mm_M2", "attr_cc_linear_op_t0.015_l100",
        "w = d_ij * (1 + 100 * max(0, 0.015 - op_prod))"
    ))

    # Variant 4: Linear Opacity (tau=0.008, lam=100) -> Conservative threshold high gain
    pen_v4 = np.maximum(0.0, 0.008 - d_m2['op_prod'])
    w_v4 = d_m2['d_ij'] * (1.0 + 100.0 * pen_v4)
    all_rows.extend(run_evaluation(
        d_m2, w_v4 <= 0.003,
        "grid-A20mm_M2", "attr_cc_linear_op_t0.008_l100",
        "w = d_ij * (1 + 100 * max(0, 0.008 - op_prod))"
    ))

    # Variant 5: Distance-Gated Opacity (d_min=2.0mm, tau=0.010, lam=100)
    dist_factor_v5 = np.maximum(0.0, (d_m2['d_ij'] - 0.0020) / 0.0010)
    pen_v5 = np.maximum(0.0, 0.010 - d_m2['op_prod']) * dist_factor_v5
    w_v5 = d_m2['d_ij'] * (1.0 + 100.0 * pen_v5)
    all_rows.extend(run_evaluation(
        d_m2, w_v5 <= 0.003,
        "grid-A20mm_M2", "attr_cc_dist_gated_op_d2.0_t0.010_l100",
        "w = d_ij * (1 + 100 * max(0, 0.010 - op_prod) * max(0, d_ij - 0.002)/0.001)"
    ))

    # Variant 6: Joint Opacity + Scale Volume Penalty
    pen_op_v6 = 50.0 * np.maximum(0.0, 0.010 - d_m2['op_prod'])
    pen_sc_v6 = 100.0 * np.maximum(0.0, d_m2['vol_max'] - 0.0025)
    w_v6 = d_m2['d_ij'] * (1.0 + pen_op_v6) * (1.0 + pen_sc_v6)
    all_rows.extend(run_evaluation(
        d_m2, w_v6 <= 0.003,
        "grid-A20mm_M2", "attr_cc_joint_op_scale_t0.010_v2.5",
        "w = d_ij * (1 + 50*max(0, 0.010 - op_prod)) * (1 + 100*max(0, vol - 0.0025))"
    ))

    # --- 3. EVALUATE FROZEN ON HELD-OUT MODEL M8 ---
    print("Evaluating Primary Variants Frozen on Held-Out Model M8...")

    # Frozen Variant 1 on M8
    pen_m8_v1 = np.maximum(0.0, 0.010 - d_m8['op_prod'])
    w_m8_v1 = d_m8['d_ij'] * (1.0 + 50.0 * pen_m8_v1)
    all_rows.extend(run_evaluation(
        d_m8, w_m8_v1 <= 0.003,
        "grid-A40mm_M8", "attr_cc_linear_op_t0.010_l50",
        "w = d_ij * (1 + 50 * max(0, 0.010 - op_prod))"
    ))

    # Frozen Variant 2 on M8
    pen_m8_v2 = np.maximum(0.0, 0.020 - d_m8['op_prod'])
    w_m8_v2 = d_m8['d_ij'] * (1.0 + 20.0 * pen_m8_v2)
    all_rows.extend(run_evaluation(
        d_m8, w_m8_v2 <= 0.003,
        "grid-A40mm_M8", "attr_cc_linear_op_t0.020_l20",
        "w = d_ij * (1 + 20 * max(0, 0.020 - op_prod))"
    ))

    # Frozen Variant 3 on M8
    pen_m8_v3 = np.maximum(0.0, 0.015 - d_m8['op_prod'])
    w_m8_v3 = d_m8['d_ij'] * (1.0 + 100.0 * pen_m8_v3)
    all_rows.extend(run_evaluation(
        d_m8, w_m8_v3 <= 0.003,
        "grid-A40mm_M8", "attr_cc_linear_op_t0.015_l100",
        "w = d_ij * (1 + 100 * max(0, 0.015 - op_prod))"
    ))

    # Frozen Variant 5 on M8
    dist_factor_m8_v5 = np.maximum(0.0, (d_m8['d_ij'] - 0.0020) / 0.0010)
    pen_m8_v5 = np.maximum(0.0, 0.010 - d_m8['op_prod']) * dist_factor_m8_v5
    w_m8_v5 = d_m8['d_ij'] * (1.0 + 100.0 * pen_m8_v5)
    all_rows.extend(run_evaluation(
        d_m8, w_m8_v5 <= 0.003,
        "grid-A40mm_M8", "attr_cc_dist_gated_op_d2.0_t0.010_l100",
        "w = d_ij * (1 + 100 * max(0, 0.010 - op_prod) * max(0, d_ij - 0.002)/0.001)"
    ))

    # --- 4. EVALUATE ON CUBES REGRESSION SCENE ---
    print("Evaluating Primary Variants on Cubes Regression...")

    # Variant 1 on Cubes
    pen_cubes_v1 = np.maximum(0.0, 0.010 - d_cubes['op_prod'])
    w_cubes_v1 = d_cubes['d_ij'] * (1.0 + 50.0 * pen_cubes_v1)
    all_rows.extend(run_evaluation(
        d_cubes, w_cubes_v1 <= 0.003,
        "cubes-cubes_k2_ring", "attr_cc_linear_op_t0.010_l50",
        "w = d_ij * (1 + 50 * max(0, 0.010 - op_prod))"
    ))

    # Variant 2 on Cubes
    pen_cubes_v2 = np.maximum(0.0, 0.020 - d_cubes['op_prod'])
    w_cubes_v2 = d_cubes['d_ij'] * (1.0 + 20.0 * pen_cubes_v2)
    all_rows.extend(run_evaluation(
        d_cubes, w_cubes_v2 <= 0.003,
        "cubes-cubes_k2_ring", "attr_cc_linear_op_t0.020_l20",
        "w = d_ij * (1 + 20 * max(0, 0.020 - op_prod))"
    ))

    # Variant 3 on Cubes
    pen_cubes_v3 = np.maximum(0.0, 0.015 - d_cubes['op_prod'])
    w_cubes_v3 = d_cubes['d_ij'] * (1.0 + 100.0 * pen_cubes_v3)
    all_rows.extend(run_evaluation(
        d_cubes, w_cubes_v3 <= 0.003,
        "cubes-cubes_k2_ring", "attr_cc_linear_op_t0.015_l100",
        "w = d_ij * (1 + 100 * max(0, 0.015 - op_prod))"
    ))

    df = pd.DataFrame(all_rows)
    out_csv = Path("runs/pump01_cc_attr_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSuccessfully generated {out_csv} with {len(df)} rows!")

    # Print summary table
    print("\n--- RESULTS SUMMARY (dilated_1nn) ---")
    dil_df = df[df["dilation_mode"] == "dilated_1nn"][
        ["model", "method", "n_seeds", "ari", "mean_iou", "cat_b_split_rate", "n_cat_b_split", "pairs_split_rate_141", "n_pairs_split_141"]
    ]
    print(dil_df.to_string(index=False))

    print("\n--- NO-DILATION ABLATION SUMMARY (grid-A20mm_M2) ---")
    m2_df = df[df["model"] == "grid-A20mm_M2"][
        ["method", "dilation_mode", "n_seeds", "n_assigned_seed", "ari", "cat_b_split_rate", "n_cat_b_split"]
    ]
    print(m2_df.to_string(index=False))

if __name__ == "__main__":
    main()
