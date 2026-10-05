#!/usr/bin/env python3
"""recompute_gap_histogram_a9.py — Audit A9: resolve the Category-A threshold contradiction.

Gap definition (stated explicitly):
    GAP(p_i, p_j) = MIN SURFACE DISTANCE between the two parts' GT point sets,
    i.e. min over all point pairs (a in P_i, b in P_j) of ||a - b||,
    computed via cKDTree nearest-neighbor query (min over b in P_j of dist(b, P_i)).
    This is NOT a centroid distance. Units: millimeters.

Recomputes the gap for every merged GT part pair in
runs/pump01_spatial_cc_diagnostic_decomposition.json (3 checkpoints x 2 radii),
validates against the stored means/counts, publishes the full histogram split by
Category A/B under BOTH thresholds (<0.2 mm and <2 mm), and prints the amended
taxonomy numbers.

Outputs:
    runs/pump01_gap_histogram_a9.png          — histogram figure
    runs/pump01_gap_histogram_a9.json         — full per-pair gaps + per-threshold counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scene-gen"))

from run_pump01_spatial_cc import spatial_cc  # noqa: E402

THRESHOLDS_MM = [0.2, 2.0]
RADII = [("r0.5cm", 0.005), ("r1.0cm", 0.010)]
CHECKPOINTS = [
    ("grid-A20mm_M2", "A20mm_M2"),
    ("grid-A20mm_M4", "A20mm_M4"),
    ("grid-A40mm_M8", "A40mm_M8"),
]


def merged_pair_gaps_mm(
    gt_on_pred: np.ndarray,
    pred_labels: np.ndarray,
    gt_pts: np.ndarray,
    gt_labs: np.ndarray,
) -> np.ndarray:
    """Replicates analyze_merged_pairs() in run_pump01_spatial_cc.py but returns
    the full per-merged-pair gap array (min surface distance between GT point sets, mm)."""
    dynamic_gt_ids = np.unique(gt_on_pred[gt_on_pred > 0])

    gt_part_pts = {pid: gt_pts[gt_labs == pid] for pid in dynamic_gt_ids}
    gt_part_trees = {pid: cKDTree(pts) for pid, pts in gt_part_pts.items() if len(pts) > 0}

    part_to_cluster = {}
    for pid in dynamic_gt_ids:
        p_mask = gt_on_pred == pid
        if p_mask.sum() > 0:
            clusters, counts = np.unique(pred_labels[p_mask], return_counts=True)
            part_to_cluster[pid] = clusters[np.argmax(counts)]

    gaps = []
    for i in range(len(dynamic_gt_ids)):
        p_i = dynamic_gt_ids[i]
        c_i = part_to_cluster.get(p_i)
        if c_i is None or p_i not in gt_part_trees:
            continue
        for j in range(i + 1, len(dynamic_gt_ids)):
            p_j = dynamic_gt_ids[j]
            if part_to_cluster.get(p_j) != c_i or len(gt_part_pts.get(p_j, [])) == 0:
                continue
            dists, _ = gt_part_trees[p_i].query(gt_part_pts[p_j], k=1)
            gaps.append(float(dists.min()) * 1000.0)
    return np.array(gaps, dtype=np.float64)


def main() -> None:
    ref = json.load(open(REPO_ROOT / "runs" / "pump01_spatial_cc_diagnostic_decomposition.json"))

    results: dict[str, dict] = {}

    for dir_name, exp_name in CHECKPOINTS:
        run_dir = REPO_ROOT / "runs" / dir_name
        traj_data = np.load(run_dir / "trajectories.npz")
        canonical_xyz = traj_data["canonical_xyz"]

        gt_file = next(run_dir.glob("**/gt_segmentation.npz"))
        gt_data = np.load(gt_file)
        gt_pts, gt_labs = gt_data["points"], gt_data["labels"]

        tree = cKDTree(gt_pts)
        _, nn = tree.query(canonical_xyz, k=1)
        gt_on_pred = gt_labs[nn]
        oracle_roi = gt_on_pred > 0

        sub_xyz = canonical_xyz[oracle_roi]
        sub_gt = gt_on_pred[oracle_roi]
        print(f"[{exp_name}] ROI points: {len(sub_xyz)}", flush=True)

        for r_name, r in RADII:
            key = f"{exp_name}_{r_name}"
            pred_labels, _ = spatial_cc(sub_xyz, radius=r, min_cluster_size=15)
            gaps = merged_pair_gaps_mm(sub_gt, pred_labels, gt_pts, gt_labs)

            # Validate against the stored decomposition
            ref_cfg = ref[key]
            stored_mean = ref_cfg["mean_cad_dist_merged_mm"]
            stored_n = ref_cfg["n_merged_pairs"]
            stored_cat_a_2mm = ref_cfg["n_cat_a_joint_contact"]
            ok_n = len(gaps) == stored_n
            ok_mean = abs(float(gaps.mean()) - stored_mean) < 1e-6
            ok_a2 = int((gaps < 2.0).sum()) == stored_cat_a_2mm
            print(
                f"[{key}] n={len(gaps)} (stored {stored_n}, match={ok_n}) "
                f"mean={gaps.mean():.4f} (stored {stored_mean:.4f}, match={ok_mean}) "
                f"catA@2mm={int((gaps < 2.0).sum())} (stored {stored_cat_a_2mm}, match={ok_a2})",
                flush=True,
            )

            per_thr = {}
            for thr in THRESHOLDS_MM:
                n_a = int((gaps < thr).sum())
                per_thr[str(thr)] = {
                    "cat_a": n_a,
                    "cat_b": int(len(gaps) - n_a),
                    "frac_cat_a": n_a / max(1, len(gaps)),
                    "frac_cat_b": 1.0 - n_a / max(1, len(gaps)),
                }
            results[key] = {
                "n_merged": len(gaps),
                "mean_mm": float(gaps.mean()),
                "median_mm": float(np.median(gaps)),
                "min_mm": float(gaps.min()),
                "max_mm": float(gaps.max()),
                "gaps_mm": sorted(gaps.tolist()),
                "per_threshold": per_thr,
                "validated_vs_stored": bool(ok_n and ok_mean and ok_a2),
            }

    # ---- Figure: full histogram per config, log-x, stacked Cat A/B, both threshold lines
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    bins = np.logspace(np.log10(0.05), np.log10(500), 60)
    for ax, (key, res) in zip(axes.flat, results.items()):
        gaps = np.asarray(res["gaps_mm"])
        cat_a = gaps[gaps < 2.0]
        cat_b = gaps[gaps >= 2.0]
        ax.hist([cat_a, cat_b], bins=bins, stacked=True,
                color=["#d62728", "#1f77b4"],
                label=[f"Cat A (gap < 2 mm): n={len(cat_a)}",
                       f"Cat B (gap >= 2 mm): n={len(cat_b)}"])
        ax.set_xscale("log")
        ax.axvline(0.2, color="k", ls="--", lw=1, label="0.2 mm threshold")
        ax.axvline(2.0, color="k", ls="-", lw=1.2, label="2 mm threshold")
        n_a02 = res["per_threshold"]["0.2"]["cat_a"]
        ax.set_title(
            f"{key} — {res['n_merged']} merged pairs | "
            f"Cat A: {n_a02} @<0.2mm vs {len(cat_a)} @<2mm | median {res['median_mm']:.1f} mm",
            fontsize=9,
        )
        ax.set_ylabel("pairs")
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("min surface distance between GT part point sets (mm, log scale)")
    fig.suptitle(
        "Audit A9: pump01 merged-pair gap histograms\n"
        "gap = MIN SURFACE DISTANCE between the two parts' GT point sets (not centroid distance)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = REPO_ROOT / "runs" / "pump01_gap_histogram_a9.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}", flush=True)

    out_json = REPO_ROOT / "runs" / "pump01_gap_histogram_a9.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gap_definition": (
                    "MIN SURFACE DISTANCE between the two parts' GT point sets: "
                    "min over point pairs (a in P_i, b in P_j) of ||a-b||, mm. Not centroid distance."
                ),
                "thresholds_mm": THRESHOLDS_MM,
                "configs": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}", flush=True)

    # ---- Summary table
    print("\n=== Per-threshold Category A/B counts ===")
    print(f"{'config':<24} {'merged':>6} | {'A@0.2':>5} {'B@0.2':>5} {'%A':>6} | {'A@2':>5} {'B@2':>5} {'%A':>6} | median mean")
    for key, res in results.items():
        t02, t2 = res["per_threshold"]["0.2"], res["per_threshold"]["2.0"]
        print(
            f"{key:<24} {res['n_merged']:>6} | "
            f"{t02['cat_a']:>5} {t02['cat_b']:>5} {t02['frac_cat_a']*100:>5.1f}% | "
            f"{t2['cat_a']:>5} {t2['cat_b']:>5} {t2['frac_cat_a']*100:>5.1f}% | "
            f"{res['median_mm']:>6.1f} {res['mean_mm']:>6.1f}"
        )


if __name__ == "__main__":
    main()
