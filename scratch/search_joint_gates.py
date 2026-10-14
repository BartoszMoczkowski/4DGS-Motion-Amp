import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from pipeline.vendored.host.metrics import adjusted_rand_index, compute_pairs_split_rate, PUMP01_BASELINE_MERGED_PAIRS_141

cat_b_pairs = PUMP01_BASELINE_MERGED_PAIRS_141[39:]

def load_data(exp_name):
    run_dir = Path("runs") / exp_name
    traj = np.load(run_dir / "trajectories.npz")
    xyz = traj["canonical_xyz"]
    gt = np.load(list(run_dir.glob("**/gt_segmentation.npz"))[0])
    _, nn = cKDTree(gt["points"]).query(xyz, k=1)
    roi = (gt["labels"][nn] > 0)
    sub_xyz = xyz[roi]
    sub_gt = gt["labels"][nn][roi]

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

    raw_op = data["opacity"][roi]
    op = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
    scales = np.exp(np.stack([data["scale_0"][roi], data["scale_1"][roi], data["scale_2"][roi]], axis=1))
    vol = np.cbrt(scales.prod(axis=1))
    aniso = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-8)

    tree = cKDTree(sub_xyz)
    r_base = 0.003
    pairs = tree.query_pairs(r=r_base, output_type="ndarray")
    d_ij = np.linalg.norm(sub_xyz[pairs[:, 0]] - sub_xyz[pairs[:, 1]], axis=1)
    op_prod = op[pairs[:, 0]] * op[pairs[:, 1]]
    vol_max = np.maximum(vol[pairs[:, 0]], vol[pairs[:, 1]])
    vol_prod = vol[pairs[:, 0]] * vol[pairs[:, 1]]
    aniso_max = np.maximum(aniso[pairs[:, 0]], aniso[pairs[:, 1]])

    return {
        "sub_xyz": sub_xyz, "sub_gt": sub_gt, "n": len(sub_xyz),
        "pairs": pairs, "d_ij": d_ij, "op_prod": op_prod,
        "vol_max": vol_max, "vol_prod": vol_prod, "aniso_max": aniso_max,
        "op": op, "vol": vol, "aniso": aniso,
    }

def eval_mask(d, mask):
    n = d["n"]
    k_pairs = d["pairs"][mask]
    adj = csr_matrix((np.ones(len(k_pairs), dtype=bool), (k_pairs[:, 0], k_pairs[:, 1])), shape=(n, n))
    adj = adj + adj.T
    n_comp, raw_labels = csgraph.connected_components(adj, directed=False)
    unique, counts = np.unique(raw_labels, return_counts=True)
    valid_comp = unique[counts >= 15]

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

    labels_cminus1 = np.full(n, -1, dtype=np.int64)
    for comp, idx in comp_map.items():
        labels_cminus1[raw_labels == comp] = idx
    labels_singletons = labels_cminus1.copy()
    labels_singletons[unassigned] = np.arange(len(valid_comp), len(valid_comp) + len(unassigned))

    ari_dil = float(adjusted_rand_index(d["sub_gt"], labels_dilated))
    ari_c1 = float(adjusted_rand_index(d["sub_gt"], labels_cminus1))
    ari_sin = float(adjusted_rand_index(d["sub_gt"], labels_singletons))
    ps_141 = compute_pairs_split_rate(d["sub_gt"], labels_dilated)
    ps_catb = compute_pairs_split_rate(d["sub_gt"], labels_dilated, merged_pairs=cat_b_pairs)

    return {
        "ari_dil": ari_dil, "ari_c1": ari_c1, "ari_sin": ari_sin,
        "seeds": len(valid_comp), "edges": len(k_pairs),
        "catb": ps_catb["pairs_split_rate"], "n_catb": ps_catb["n_pairs_split"],
        "p141": ps_141["pairs_split_rate"], "n_141": ps_141["n_pairs_split"],
    }

if __name__ == "__main__":
    d_m2 = load_data("grid-A20mm_M2")
    d_m8 = load_data("grid-A40mm_M8")


    # Let's test: What if we prune edges where dist is large (> 2mm) AND op_prod is low?
    print("=== Distance-conditioned opacity pruning: prune if d_ij > d_thr AND op_prod < op_thr ===")
    for d_thr in [0.0015, 0.0020, 0.0022, 0.0025]:
        for op_thr in [0.01, 0.02, 0.03, 0.05]:
            mask_m2 = ~((d_m2["d_ij"] > d_thr) & (d_m2["op_prod"] < op_thr))
            mask_m8 = ~((d_m8["d_ij"] > d_thr) & (d_m8["op_prod"] < op_thr))
            m2_res = eval_mask(d_m2, mask_m2)
            m8_res = eval_mask(d_m8, mask_m8)
            if m8_res["ari_dil"] >= 0.3840 and m2_res["catb"] >= 0.1961:
                print(f"d_thr={d_thr*1000:4.1f}mm, op_thr={op_thr:.2f} | "
                      f"M2: ARI={m2_res['ari_dil']:.4f} CatB={m2_res['catb']:.4f} ({m2_res['n_catb']:2d}/102) | "
                      f"M8: ARI={m8_res['ari_dil']:.4f} (pass={m8_res['ari_dil']>=0.3847}) seeds={m8_res['seeds']}")

    # What if we prune edges with anomalous scale (bloated bridging Gaussians)?
    print("\n=== Scale volume pruning: prune if vol_max > v_thr ===")
    for v_thr in [0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050]:
        mask_m2 = d_m2["vol_max"] <= v_thr
        mask_m8 = d_m8["vol_max"] <= v_thr
        m2_res = eval_mask(d_m2, mask_m2)
        m8_res = eval_mask(d_m8, mask_m8)
        print(f"v_thr={v_thr*1000:4.1f}mm | M2: ARI={m2_res['ari_dil']:.4f} CatB={m2_res['catb']:.4f} ({m2_res['n_catb']:2d}/102) | "
              f"M8: ARI={m8_res['ari_dil']:.4f} (pass={m8_res['ari_dil']>=0.3847}) seeds={m8_res['seeds']}")

    # What if we prune edges where aniso is extreme or vol is extreme?
    print("\n=== Anisotropy pruning: prune if aniso_max > a_thr ===")
    for a_thr in [500.0, 1000.0, 2000.0, 5000.0]:
        mask_m2 = d_m2["aniso_max"] <= a_thr
        mask_m8 = d_m8["aniso_max"] <= a_thr
        m2_res = eval_mask(d_m2, mask_m2)
        m8_res = eval_mask(d_m8, mask_m8)
        print(f"a_thr={a_thr:6.0f} | M2: ARI={m2_res['ari_dil']:.4f} CatB={m2_res['catb']:.4f} ({m2_res['n_catb']:2d}/102) | "
              f"M8: ARI={m8_res['ari_dil']:.4f} (pass={m8_res['ari_dil']>=0.3847}) seeds={m8_res['seeds']}")

