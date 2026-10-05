"""multicut.py — Multi-Channel Affinity Graph Fusion & Lifted Multicut Solver.

Fuses:
1. Spatial adjacency & distance (canonical xyz k-NN + radius bounds)
2. Appearance / Color similarity (SH-DC / RGB Euclidean difference)
3. Trajectory & Motion coherence (FFT complex phasors at drive frequency f0 + normalized trajectory correlation)
4. Optional 2D mask / ROI agreement
5. Optional long-range lifted edges for distant points with high motion/appearance similarity

Solves via Hierarchical Greedy Additive Edge Contraction (GAEC) + Kernighan-Lin (KL) boundary search
(with optional RAMA / rama_py GPU delegation if installed).
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

logger = logging.getLogger(__name__)


def extract_motion_features(traj: np.ndarray, f0: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract complex phasor features at drive frequency f0 and normalized trajectory vectors."""
    N, T, _ = traj.shape
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * f0 * t / T)

    traj_centered = traj - traj.mean(axis=1, keepdims=True)

    phasors = np.tensordot(traj_centered, basis, axes=([1], [0]))  # (N, 3) complex
    phasors_6d = np.concatenate([phasors.real, phasors.imag], axis=1).astype(np.float32)

    p_norms = np.linalg.norm(phasors_6d, axis=1, keepdims=True)
    norm_phasors = np.zeros_like(phasors_6d)
    p_valid = p_norms[:, 0] > 1e-9
    norm_phasors[p_valid] = phasors_6d[p_valid] / p_norms[p_valid]

    traj_flat = traj_centered.reshape(N, -1)
    norms = np.linalg.norm(traj_flat, axis=1, keepdims=True)
    norm_traj = np.zeros_like(traj_flat)
    valid = norms[:, 0] > 1e-9
    norm_traj[valid] = traj_flat[valid] / norms[valid]

    return phasors_6d, norm_phasors, norm_traj


def build_multichannel_graph(
    xyz: np.ndarray,
    traj: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    k: int = 12,
    radius_max: float = 0.05,
    f0: float = 10.0,
    include_lifted: bool = False,
    lifted_k: int = 4,
    lifted_min_dist: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build candidate edge list and extract multi-channel edge feature vectors."""
    N = len(xyz)
    tree = cKDTree(xyz)

    # Vectorized k-NN edge extraction
    k_query = min(k + 1, N)
    dists, indices = tree.query(xyz, k=k_query)
    
    d = dists[:, 1:]
    idx = indices[:, 1:]
    valid = (d <= radius_max) & (idx < N)
    
    row_idx = np.broadcast_to(np.arange(N, dtype=np.int64)[:, None], idx.shape)
    u_raw = row_idx[valid]
    v_raw = idx[valid].astype(np.int64)
    
    u_loc = np.minimum(u_raw, v_raw)
    v_loc = np.maximum(u_raw, v_raw)
    
    packed = u_loc * N + v_loc
    unique_packed = np.unique(packed)
    local_edges = np.column_stack([unique_packed // N, unique_packed % N])
    is_lifted_local = np.zeros(len(local_edges), dtype=np.float32)

    # Motion and Appearance Features
    norm_phasors = None
    norm_traj = None
    if traj is not None:
        _, norm_phasors, norm_traj = extract_motion_features(traj, f0=f0)

    # Lifted long-range edges
    if include_lifted and N > 100 and norm_phasors is not None:
        p_tree = cKDTree(norm_phasors)
        _, p_indices = p_tree.query(norm_phasors, k=min(lifted_k + 5, N))
        p_d = p_indices[:, 1:]
        p_row = np.broadcast_to(np.arange(N, dtype=np.int64)[:, None], p_d.shape)
        p_u = np.minimum(p_row.ravel(), p_d.ravel())
        p_v = np.maximum(p_row.ravel(), p_d.ravel())
        
        d_spat = np.linalg.norm(xyz[p_u] - xyz[p_v], axis=1)
        lift_valid = (d_spat >= lifted_min_dist) & (p_u != p_v)
        if lift_valid.any():
            p_packed = np.unique(p_u[lift_valid] * N + p_v[lift_valid])
            local_set = set(unique_packed)
            p_lifted = [p for p in p_packed if p not in local_set]
            if p_lifted:
                p_arr = np.array(p_lifted, dtype=np.int64)
                lifted_edges = np.column_stack([p_arr // N, p_arr % N])
            else:
                lifted_edges = np.zeros((0, 2), dtype=np.int64)
        else:
            lifted_edges = np.zeros((0, 2), dtype=np.int64)
    else:
        lifted_edges = np.zeros((0, 2), dtype=np.int64)

    if len(lifted_edges) > 0:
        all_edges = np.vstack([local_edges, lifted_edges])
        is_lifted = np.concatenate([is_lifted_local, np.ones(len(lifted_edges), dtype=np.float32)])
    else:
        all_edges = local_edges
        is_lifted = is_lifted_local

    E = len(all_edges)
    if E == 0:
        return all_edges, np.zeros((0, 6), dtype=np.float32), {"n_local": 0, "n_lifted": 0}

    u = all_edges[:, 0]
    v = all_edges[:, 1]

    d_xyz = np.linalg.norm(xyz[u] - xyz[v], axis=1).astype(np.float32)

    if rgb is not None:
        d_rgb = np.linalg.norm(rgb[u] - rgb[v], axis=1).astype(np.float32)
    else:
        d_rgb = np.zeros(E, dtype=np.float32)

    if norm_phasors is not None:
        d_phasor = np.linalg.norm(norm_phasors[u] - norm_phasors[v], axis=1).astype(np.float32)
        r_phasor = np.sum(norm_phasors[u] * norm_phasors[v], axis=1).astype(np.float32)
        disc_phasor = np.clip(1.0 - r_phasor, 0.0, 2.0)
    else:
        d_phasor = np.zeros(E, dtype=np.float32)
        disc_phasor = np.zeros(E, dtype=np.float32)

    if norm_traj is not None:
        r_traj = np.sum(norm_traj[u] * norm_traj[v], axis=1).astype(np.float32)
        disc_traj = np.clip(1.0 - r_traj, 0.0, 2.0)
    else:
        disc_traj = np.zeros(E, dtype=np.float32)

    features = np.column_stack([
        d_xyz,
        d_rgb,
        d_phasor,
        disc_phasor,
        disc_traj,
        is_lifted,
    ]).astype(np.float32)

    info = {
        "n_nodes": N,
        "n_edges": E,
        "n_local_edges": len(local_edges),
        "n_lifted_edges": len(lifted_edges),
        "mean_d_xyz": float(np.mean(d_xyz)),
        "mean_d_rgb": float(np.mean(d_rgb)) if rgb is not None else 0.0,
    }
    return all_edges, features, info


DEFAULT_LOGIT_WEIGHTS = {
    "intercept": 4.0,
    "r_scale": 0.005,       # 5 mm spatial characteristic scale
    "c_scale": 0.18,        # 0.18 color boundary scale
    "p_scale": 0.30,        # 0.30 phasor discrepancy scale
    "w_lifted": -1.0,
}


def fit_edge_weights(
    features: np.ndarray,
    edges: np.ndarray,
    gt_labels: np.ndarray,
    max_samples: int = 100000,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Fit edge scale parameters on GT edge labels (same-part vs cross-part)."""
    u = edges[:, 0]
    v = edges[:, 1]
    same = (gt_labels[u] == gt_labels[v])

    d_xyz_same = features[same, 0]
    d_rgb_diff = features[~same, 1]

    # Characteristic 90th percentile within same part for spatial, 25th percentile across parts for color
    r_scale = float(np.percentile(d_xyz_same, 80)) if len(d_xyz_same) > 0 else 0.005
    c_scale = float(np.percentile(d_rgb_diff[d_rgb_diff > 0.01], 25)) if (d_rgb_diff > 0.01).any() else 0.18

    weights_dict = {
        "intercept": 4.0,
        "r_scale": max(0.002, r_scale),
        "c_scale": max(0.05, c_scale),
        "p_scale": 0.30,
        "w_lifted": -1.0,
    }
    logger.info("Calibrated edge scale weights: %s", weights_dict)
    return weights_dict


def compute_edge_logits(
    features: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute scalar logit edge affinities w_e from multi-channel edge features.

    Formula:
      w_e = intercept * (1 - d_xyz / r_scale) - 4.0 * (d_rgb / c_scale)^2 - 2.0 * (disc_phasor / p_scale)
    """
    w = weights if weights is not None else DEFAULT_LOGIT_WEIGHTS

    d_xyz = features[:, 0]
    d_rgb = features[:, 1]
    disc_phasor = features[:, 3]
    is_lifted = features[:, 5]

    intercept = float(w.get("intercept", 4.0))
    r_scale = float(w.get("r_scale", 0.005))
    c_scale = float(w.get("c_scale", 0.18))
    p_scale = float(w.get("p_scale", 0.30))
    w_lifted = float(w.get("w_lifted", -1.0))

    # Spatial affinity: +intercept at d=0, 0 at d=r_scale, negative beyond
    w_spatial = intercept * (1.0 - (d_xyz / r_scale))
    # Appearance penalty
    w_color = -4.0 * np.square(d_rgb / c_scale)
    # Motion penalty
    w_motion = -2.0 * (disc_phasor / p_scale)

    logits = w_spatial + w_color + w_motion + is_lifted * w_lifted
    return logits.astype(np.float32)


class DisjointSet:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, i: int) -> int:
        root = i
        while root != self.parent[root]:
            root = self.parent[root]
        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i: int, j: int) -> int:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return root_i
        if self.rank[root_i] < self.rank[root_j]:
            root_i, root_j = root_j, root_i
        self.parent[root_j] = root_i
        if self.rank[root_i] == self.rank[root_j]:
            self.rank[root_i] += 1
        return root_i


def solve_multicut_gaec(
    n_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Greedy Additive Edge Contraction (GAEC) solver for Correlation Clustering."""
    E = len(edges)
    if n_nodes == 0 or E == 0:
        return np.zeros(n_nodes, dtype=np.int64)

    # Fast supernode contraction of positive edges (w > 0.0) via connected components
    pos_mask = weights > 0.0
    if pos_mask.any():
        pos_edges = edges[pos_mask]
        adj = csr_matrix(
            (np.ones(len(pos_edges), dtype=bool), (pos_edges[:, 0], pos_edges[:, 1])),
            shape=(n_nodes, n_nodes),
        )
        n_super, node_to_super = csgraph.connected_components(adj, directed=False, return_labels=True)
    else:
        n_super = n_nodes
        node_to_super = np.arange(n_nodes, dtype=np.int64)

    if n_super <= 1:
        return np.zeros(n_nodes, dtype=np.int64)

    # Supernode edge accumulation
    u_super = node_to_super[edges[:, 0]]
    v_super = node_to_super[edges[:, 1]]
    inter_mask = u_super != v_super

    if not inter_mask.any():
        unique_labels = np.unique(node_to_super)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in node_to_super], dtype=np.int64)

    u_inter = u_super[inter_mask]
    v_inter = v_super[inter_mask]
    w_inter = weights[inter_mask]

    super_u = np.minimum(u_inter, v_inter)
    super_v = np.maximum(u_inter, v_inter)

    super_adj = csr_matrix(
        (w_inter, (super_u, super_v)),
        shape=(n_super, n_super),
    )
    super_adj.sum_duplicates()
    coo = super_adj.tocoo()
    s_edges = np.column_stack([coo.row, coo.col])
    s_weights = coo.data

    if not (s_weights > 0).any():
        unique_labels = np.unique(node_to_super)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in node_to_super], dtype=np.int64)

    # GAEC priority queue on remaining inter-supernode edges
    adj_dict: List[Dict[int, float]] = [{} for _ in range(n_super)]
    heap: List[Tuple[float, int, int, int]] = []
    edge_versions: Dict[Tuple[int, int], int] = {}

    for (u, v), w in zip(s_edges, s_weights):
        if u != v:
            adj_dict[u][v] = float(w)
            adj_dict[v][u] = float(w)
            edge_versions[(u, v)] = 0
            if w > 0:
                heapq.heappush(heap, (-float(w), u, v, 0))

    dset = DisjointSet(n_super)

    while heap:
        neg_w, u, v, ver = heapq.heappop(heap)
        w = -neg_w
        if w <= 0:
            break

        root_u = dset.find(u)
        root_v = dset.find(v)
        if root_u == root_v:
            continue

        pair = (min(root_u, root_v), max(root_u, root_v))
        if edge_versions.get(pair, -1) != ver:
            continue

        curr_w = adj_dict[root_u].get(root_v, 0.0)
        if curr_w <= 0:
            continue

        target_root = dset.union(root_u, root_v)
        other_root = root_v if target_root == root_u else root_u

        other_neighbors = list(adj_dict[other_root].items())
        adj_dict[other_root].clear()
        if other_root in adj_dict[target_root]:
            del adj_dict[target_root][other_root]

        for nbr, weight in other_neighbors:
            if nbr == target_root or nbr == other_root:
                continue
            nbr_root = dset.find(nbr)
            if nbr_root == target_root:
                continue

            if other_root in adj_dict[nbr_root]:
                del adj_dict[nbr_root][other_root]

            new_w = adj_dict[target_root].get(nbr_root, 0.0) + weight
            adj_dict[target_root][nbr_root] = new_w
            adj_dict[nbr_root][target_root] = new_w

            p = (min(target_root, nbr_root), max(target_root, nbr_root))
            v_num = edge_versions.get(p, 0) + 1
            edge_versions[p] = v_num
            if new_w > 0:
                heapq.heappush(heap, (-new_w, p[0], p[1], v_num))

    super_labels = np.array([dset.find(s) for s in range(n_super)], dtype=np.int64)
    raw_labels = super_labels[node_to_super]

    unique_labels = np.unique(raw_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in raw_labels], dtype=np.int64)


def solve_multicut_kl_refine(
    n_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
    init_labels: np.ndarray,
    max_passes: int = 3,
) -> np.ndarray:
    """Fast vectorized boundary refinement using CSR indptr/indices arrays."""
    labels = init_labels.copy()
    if len(edges) == 0:
        return labels

    u = edges[:, 0]
    v = edges[:, 1]
    
    adj_u = csr_matrix((weights, (u, v)), shape=(n_nodes, n_nodes))
    adj_w = (adj_u + adj_u.T).tocsr()
    indptr = adj_w.indptr
    indices = adj_w.indices
    data = adj_w.data

    for pass_idx in range(max_passes):
        diff_mask = labels[u] != labels[v]
        if not diff_mask.any():
            break

        boundary_nodes = np.unique(np.concatenate([u[diff_mask], v[diff_mask]]))
        if len(boundary_nodes) == 0:
            break

        improved_count = 0
        for i in boundary_nodes:
            curr_c = labels[i]
            start = indptr[i]
            end = indptr[i + 1]
            if start == end:
                continue

            nbr_idx = indices[start:end]
            nbr_w = data[start:end]
            nbr_c_arr = labels[nbr_idx]

            unique_c, inv = np.unique(nbr_c_arr, return_inverse=True)
            c_gains = np.zeros(len(unique_c), dtype=np.float64)
            np.add.at(c_gains, inv, nbr_w)

            curr_idx_match = np.where(unique_c == curr_c)[0]
            curr_gain = c_gains[curr_idx_match[0]] if len(curr_idx_match) > 0 else 0.0
            
            best_idx = np.argmax(c_gains)
            if c_gains[best_idx] > curr_gain + 1e-3:
                labels[i] = unique_c[best_idx]
                improved_count += 1

        if improved_count == 0:
            break

    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in labels], dtype=np.int64)


def solve_multicut(
    n_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
    use_rama_if_available: bool = True,
    refine_kl: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve Multicut / Correlation Clustering on affinity graph."""
    t0 = time.perf_counter()
    solver_name = "gaec_kl"

    if use_rama_if_available:
        try:
            import rama_py  # type: ignore

            u_gpu = edges[:, 0].astype(np.int32)
            v_gpu = edges[:, 1].astype(np.int32)
            w_gpu = weights.astype(np.float32)
            node_labels = rama_py.multicut(u_gpu, v_gpu, w_gpu)
            labels = np.asarray(node_labels, dtype=np.int64)
            solver_name = "rama_gpu"
        except (ImportError, Exception):
            labels = solve_multicut_gaec(n_nodes, edges, weights)
            if refine_kl:
                labels = solve_multicut_kl_refine(n_nodes, edges, weights, labels)
    else:
        labels = solve_multicut_gaec(n_nodes, edges, weights)
        if refine_kl:
            labels = solve_multicut_kl_refine(n_nodes, edges, weights, labels)

    wall_s = time.perf_counter() - t0
    k_pred = len(np.unique(labels))

    if len(edges) > 0:
        is_cut = labels[edges[:, 0]] != labels[edges[:, 1]]
        cut_energy = float(np.sum(weights[is_cut]))
        uncut_energy = float(np.sum(weights[~is_cut]))
    else:
        cut_energy = 0.0
        uncut_energy = 0.0

    meta = {
        "solver": solver_name,
        "wall_time_s": wall_s,
        "k_pred": k_pred,
        "cut_energy": cut_energy,
        "uncut_energy": uncut_energy,
    }
    return labels, meta


def postprocess_min_cluster_size(
    xyz: np.ndarray,
    labels: np.ndarray,
    min_size: int = 15,
) -> np.ndarray:
    """Reassign micro-clusters with count < min_size to nearest valid cluster."""
    if min_size <= 1:
        return labels

    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = set(unique[counts < min_size])
    if not small_clusters or len(small_clusters) == len(unique):
        return labels

    valid_mask = np.isin(labels, list(small_clusters), invert=True)
    if not valid_mask.any():
        return labels

    out_labels = labels.copy()
    valid_indices = np.where(valid_mask)[0]
    small_indices = np.where(~valid_mask)[0]

    tree = cKDTree(xyz[valid_indices])
    _, nn = tree.query(xyz[small_indices], k=1)
    out_labels[small_indices] = labels[valid_indices[nn]]

    unique_labels = np.unique(out_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in out_labels], dtype=np.int64)
