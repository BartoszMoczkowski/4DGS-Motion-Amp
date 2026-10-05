import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix
from scipy.optimize import linear_sum_assignment
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scene-gen"))

from pipeline.vendored.host.metrics import adjusted_rand_index
from pipeline.vendored.host.kabsch_em import _kmeans_plus_plus, _lloyd_kmeans

def fast_iou(labels_true, labels_pred):
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    _, c_idx = np.unique(labels_true, return_inverse=True)
    _, k_idx = np.unique(labels_pred, return_inverse=True)
    np.add.at(contingency, (c_idx, k_idx), 1)
    c_sz = contingency.sum(axis=1, keepdims=True)
    k_sz = contingency.sum(axis=0, keepdims=True)
    unions = c_sz + k_sz - contingency
    valid = unions > 0
    ious = np.zeros_like(contingency, dtype=np.float64)
    ious[valid] = contingency[valid] / unions[valid]
    r, c = linear_sum_assignment(-ious)
    return float(np.mean([ious[a, b] for a, b in zip(r, c)]))

def load_checkpoint_data(exp_name):
    run_dir = REPO_ROOT / "runs" / f"grid-{exp_name}"
    traj = np.load(run_dir / "trajectories.npz")
    xyz = traj["canonical_xyz"]
    op = traj["opacity"]

    gt_file = list(run_dir.glob("**/gt_segmentation.npz"))[0]
    gt_data = np.load(gt_file)
    tree = cKDTree(gt_data["points"])
    _, nn = tree.query(xyz, k=1)
    gt_on_pred = gt_data["labels"][nn]
    oracle_roi = (gt_on_pred > 0)

    # PLY rgb
    ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
    ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
    ply_file = ply_files[-1]
    with open(ply_file, "rb") as f:
        header = b""
        while b"end_header\n" not in header:
            header += f.readline()
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
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
    rgb = np.clip(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0)

    return {
        "xyz": xyz,
        "opacity": op,
        "gt_on_pred": gt_on_pred,
        "oracle_roi": oracle_roi,
        "rgb": rgb,
    }

def main():
    csv_path = REPO_ROOT / "runs" / "pump01_spatial_cc_results.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    print(f"Loaded {len(existing_rows)} existing rows from {csv_path}")

    # Group rows by checkpoint
    data_cache = {}
    for exp_name in ["A20mm_M2", "A20mm_M4", "A40mm_M8"]:
        print(f"Loading data for {exp_name}...")
        data_cache[exp_name] = load_checkpoint_data(exp_name)

    patched_rows = []

    for i, row in enumerate(existing_rows):
        exp_name = row["checkpoint"]
        modality = row["modality"]
        r_cm = float(row["radius_cm"])
        r = r_cm / 100.0
        op_filter = (row["opacity_filter"].lower() == "true")
        method = row["method"]
        d = data_cache[exp_name]

        n_roi = int(row["n_roi_points"])
        n_active = int(row["n_active_points"])
        # Every existing row in pump01_spatial_cc_results.csv was evaluated with roi_mask=oracle_roi
        # So n_scored was exactly n_roi_points
        n_scored = n_roi

        # Determine n_labeled (points in components >= 15)
        if modality == "spatial_xyz":
            if op_filter:
                active_mask = (d["opacity"] > 0.1) & d["oracle_roi"]
            else:
                active_mask = d["oracle_roi"]
            sub_xyz = d["xyz"][active_mask]
            tree = cKDTree(sub_xyz)
            pairs = tree.query_pairs(r=r, output_type="ndarray")
            n_sub = len(sub_xyz)
            if len(pairs) > 0:
                data = np.ones(len(pairs), dtype=bool)
                adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n_sub, n_sub))
                adj = adj + adj.T
            else:
                adj = csr_matrix((n_sub, n_sub), dtype=bool)
            _, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)
            _, counts = np.unique(raw_labels, return_counts=True)
            n_labeled = int(counts[counts >= 15].sum())

        elif modality == "color_gated_spatial":
            c_th = float(row["color_thresh"])
            active_mask = d["oracle_roi"]
            sub_xyz = d["xyz"][active_mask]
            sub_rgb = d["rgb"][active_mask]
            tree = cKDTree(sub_xyz)
            pairs = tree.query_pairs(r=r, output_type="ndarray")
            if len(pairs) > 0:
                c_diff = sub_rgb[pairs[:, 0]] - sub_rgb[pairs[:, 1]]
                c_dist = np.linalg.norm(c_diff, axis=1)
                pairs = pairs[c_dist <= c_th]
            n_sub = len(sub_xyz)
            if len(pairs) > 0:
                data = np.ones(len(pairs), dtype=bool)
                adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n_sub, n_sub))
                adj = adj + adj.T
            else:
                adj = csr_matrix((n_sub, n_sub), dtype=bool)
            _, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)
            _, counts = np.unique(raw_labels, return_counts=True)
            n_labeled = int(counts[counts >= 15].sum())

        elif modality == "appearance_precluster_spatial":
            # Extract n_colors from method name e.g. e0d_precluster_k5_r...
            n_colors = int(method.split("_")[2].replace("k", ""))
            active_mask = d["oracle_roi"]
            sub_xyz = d["xyz"][active_mask]
            sub_rgb = d["rgb"][active_mask]
            n_sub = len(sub_xyz)
            rng = np.random.default_rng(42)
            centers = _kmeans_plus_plus(sub_rgb, n_colors, rng)
            color_labels, _ = _lloyd_kmeans(sub_rgb, centers)
            n_labeled = 0
            for c in range(n_colors):
                c_mask = (color_labels == c)
                if c_mask.sum() == 0: continue
                c_xyz = sub_xyz[c_mask]
                c_tree = cKDTree(c_xyz)
                c_pairs = c_tree.query_pairs(r=r, output_type="ndarray")
                c_n = len(c_xyz)
                if len(c_pairs) > 0:
                    data = np.ones(len(c_pairs), dtype=bool)
                    adj = csr_matrix((data, (c_pairs[:, 0], c_pairs[:, 1])), shape=(c_n, c_n))
                    adj = adj + adj.T
                else:
                    adj = csr_matrix((c_n, c_n), dtype=bool)
                _, c_raw = csgraph.connected_components(adj, directed=False, return_labels=True)
                _, c_cnts = np.unique(c_raw, return_counts=True)
                n_labeled += int(c_cnts[c_cnts >= 15].sum())
        else:
            n_labeled = n_active

        n_discarded = n_scored - n_labeled

        # Construct patched row
        p_row = {
            "checkpoint": row["checkpoint"],
            "method": row["method"],
            "modality": row["modality"],
            "radius_cm": row["radius_cm"],
            "opacity_filter": row["opacity_filter"],
            "color_weight": row["color_weight"],
            "color_thresh": row["color_thresh"],
            "k_pred_roi": row["k_pred_roi"],
            "k_pred_total": row["k_pred_total"],
            "k_gt_roi": row["k_gt_roi"],
            "k_gt_total": row["k_gt_total"],
            "n_roi_points": row["n_roi_points"],
            "n_active_points": row["n_active_points"],
            "n_scored": n_scored,
            "n_labeled": n_labeled,
            "n_discarded": n_discarded,
            "ari_within_roi": row["ari_within_roi"],
            "mean_iou_within_roi": row["mean_iou_within_roi"],
            "ari_global": row["ari_global"],
            "mean_iou_global": row["mean_iou_global"],
        }
        patched_rows.append(p_row)
        if (i + 1) % 15 == 0:
            print(f"Processed {i+1}/{len(existing_rows)} rows...")

    # Step 3: Compute Recomputed Full-Coverage Rows for r=0.3 and r=0.5
    print("\nComputing Recomputed Full-Coverage Rows (r=0.3 and r=0.5) across M2, M4, M8...")
    recomputed_rows = []

    for exp_name in ["A20mm_M2", "A20mm_M4", "A40mm_M8"]:
        d = data_cache[exp_name]
        xyz = d["xyz"]
        gt_on_pred = d["gt_on_pred"]
        oracle_roi = d["oracle_roi"]
        sub_xyz = xyz[oracle_roi]
        sub_gt = gt_on_pred[oracle_roi]
        n_roi = len(sub_xyz)
        tree_sub = cKDTree(sub_xyz)

        for r_cm in [0.3, 0.5]:
            r = r_cm / 100.0
            pairs = tree_sub.query_pairs(r=r, output_type="ndarray")
            if len(pairs) > 0:
                data = np.ones(len(pairs), dtype=bool)
                adj = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n_roi, n_roi))
                adj = adj + adj.T
            else:
                adj = csr_matrix((n_roi, n_roi), dtype=bool)

            n_comp, raw_labels = csgraph.connected_components(adj, directed=False, return_labels=True)
            unique, counts = np.unique(raw_labels, return_counts=True)
            valid = unique[counts >= 15]
            n_labeled = int(counts[counts >= 15].sum())
            n_discarded = n_roi - n_labeled

            # -------------------------------------------------------------
            # Convention 1: Singletons (discarded fragments become unique singletons)
            # -------------------------------------------------------------
            comp_map = {c: i for i, c in enumerate(valid)}
            labels_singletons = np.full(n_roi, -1, dtype=np.int64)
            for c, i in comp_map.items():
                labels_singletons[raw_labels == c] = i
            cur_id = len(valid)
            for idx in np.where(labels_singletons == -1)[0]:
                labels_singletons[idx] = cur_id
                cur_id += 1
            k_singletons = cur_id
            ari_singletons = float(adjusted_rand_index(sub_gt, labels_singletons))
            iou_singletons = fast_iou(sub_gt, labels_singletons)

            full_singletons = np.full(len(xyz), -1, dtype=np.int64)
            full_singletons[oracle_roi] = labels_singletons
            full_singletons[~oracle_roi] = 0
            ari_glob_sing = float(adjusted_rand_index(gt_on_pred, full_singletons))
            iou_glob_sing = fast_iou(gt_on_pred, full_singletons)

            row_sing = {
                "checkpoint": exp_name,
                "method": f"spatial_cc_r{r_cm:04.1f}cm_full_singletons",
                "modality": "spatial_xyz_full_singletons",
                "radius_cm": r_cm,
                "opacity_filter": False,
                "color_weight": 0.0,
                "color_thresh": 0.0,
                "k_pred_roi": k_singletons,
                "k_pred_total": k_singletons + 1,
                "k_gt_roi": len(np.unique(sub_gt)),
                "k_gt_total": len(np.unique(gt_on_pred)),
                "n_roi_points": n_roi,
                "n_active_points": n_roi,
                "n_scored": n_roi,
                "n_labeled": n_labeled,
                "n_discarded": n_discarded,
                "ari_within_roi": ari_singletons,
                "mean_iou_within_roi": iou_singletons,
                "ari_global": ari_glob_sing,
                "mean_iou_global": iou_glob_sing,
            }
            recomputed_rows.append(row_sing)

            # -------------------------------------------------------------
            # Convention 2: Cluster -1 (discarded fragments grouped into cluster -1)
            # -------------------------------------------------------------
            labels_neg1 = np.full(n_roi, -1, dtype=np.int64)
            for c, i in comp_map.items():
                labels_neg1[raw_labels == c] = i
            k_neg1 = len(valid) + (1 if (labels_neg1 == -1).any() else 0)
            ari_neg1 = float(adjusted_rand_index(sub_gt, labels_neg1))
            iou_neg1 = fast_iou(sub_gt, labels_neg1)

            full_neg1 = np.full(len(xyz), -1, dtype=np.int64)
            full_neg1[oracle_roi] = labels_neg1
            full_neg1[~oracle_roi] = 0
            ari_glob_neg1 = float(adjusted_rand_index(gt_on_pred, full_neg1))
            iou_glob_neg1 = fast_iou(gt_on_pred, full_neg1)

            row_neg1 = {
                "checkpoint": exp_name,
                "method": f"spatial_cc_r{r_cm:04.1f}cm_cluster_neg1",
                "modality": "spatial_xyz_cluster_neg1",
                "radius_cm": r_cm,
                "opacity_filter": False,
                "color_weight": 0.0,
                "color_thresh": 0.0,
                "k_pred_roi": k_neg1,
                "k_pred_total": k_neg1 + 1,
                "k_gt_roi": len(np.unique(sub_gt)),
                "k_gt_total": len(np.unique(gt_on_pred)),
                "n_roi_points": n_roi,
                "n_active_points": n_roi,
                "n_scored": n_roi,
                "n_labeled": n_labeled,
                "n_discarded": n_discarded,
                "ari_within_roi": ari_neg1,
                "mean_iou_within_roi": iou_neg1,
                "ari_global": ari_glob_neg1,
                "mean_iou_global": iou_glob_neg1,
            }
            recomputed_rows.append(row_neg1)

            print(f"  [{exp_name}] r={r_cm:04.1f}cm: Singletons ARI={ari_singletons:.6f} (K={k_singletons}) | Cluster-1 ARI={ari_neg1:.6f} (K={k_neg1}) | n_labeled={n_labeled}/{n_roi} ({n_labeled/n_roi*100:.1f}%)")

    all_rows = patched_rows + recomputed_rows
    fieldnames = list(all_rows[0].keys())

    # Write patched CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSuccessfully wrote patched CSV to {csv_path} with {len(all_rows)} rows (135 patched + {len(recomputed_rows)} recomputed)!")

if __name__ == "__main__":
    main()
