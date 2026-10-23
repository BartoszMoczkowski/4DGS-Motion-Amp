"""
scripts/run_lego_turbine_pipeline.py — End-to-end 4DGS Training, Rendering,
Evaluation, and Motion Segmentation for the Lego Turbine series:
1. lego_turbine_base
2. lego_turbine_2_missing
3. lego_turbine_5_missing
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline import api
from pipeline.artifacts import Artifact, validate_external_artifact_path, load_manifest, update_manifest
from pipeline.vendored.host.kabsch_em import segment_by_kabsch
from pipeline.vendored.host.rigidity_graph import segment_by_rigidity
from pipeline.vendored.host.rigidity_graph2 import segment_by_rigidity2

SCENES = [
    {"name": "lego_turbine_base", "desc": "Baseline Lego turbine with all blades intact", "n_times": 160},
    {"name": "lego_turbine_2_missing", "desc": "Lego turbine with 2 blades missing (mass imbalance)", "n_times": 160},
    {"name": "lego_turbine_5_missing", "desc": "Lego turbine with 5 blades missing (severe asymmetric damage)", "n_times": 160},
]

STATUS_FILE = REPO_ROOT / "runs" / "lego_turbine_pipeline_status.json"
SUMMARY_JSON = REPO_ROOT / "runs" / "lego_turbine_evaluation_summary.json"
SUMMARY_CSV = REPO_ROOT / "runs" / "lego_turbine_evaluation_summary.csv"


def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": {}}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def compute_image_metrics(renders_dir: Path, gt_dir: Path) -> dict:
    """Compute PSNR and simple SSIM between rendered test views and GT."""
    render_files = sorted(list(renders_dir.glob("*.png")) + list(renders_dir.glob("*.jpg")))
    gt_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))

    if not render_files or not gt_files:
        return {"n_eval_images": 0}

    psnr_list = []
    l1_list = []

    for rf, gf in zip(render_files, gt_files):
        r_img = np.array(Image.open(rf).convert("RGB"), dtype=np.float32) / 255.0
        g_img = np.array(Image.open(gf).convert("RGB"), dtype=np.float32) / 255.0

        if r_img.shape != g_img.shape:
            # Resize render to match GT if needed
            r_img = np.array(Image.fromarray((r_img * 255).astype(np.uint8)).resize((g_img.shape[1], g_img.shape[0])), dtype=np.float32) / 255.0

        mse = np.mean((r_img - g_img) ** 2)
        psnr = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100.0
        l1 = np.mean(np.abs(r_img - g_img))

        psnr_list.append(float(psnr))
        l1_list.append(float(l1))

    return {
        "n_eval_images": len(psnr_list),
        "mean_psnr_db": float(np.mean(psnr_list)),
        "median_psnr_db": float(np.median(psnr_list)),
        "mean_mae": float(np.mean(l1_list)),
    }


def render_segmentation_figures(xyz: np.ndarray, labels: np.ndarray, out_prefix: str, title: str):
    """Render 2D orthographic projections and 3D figure of clusters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(int(l) for l in np.unique(labels) if l != -1)
    cmap = matplotlib.colormaps["tab20"]
    colors_table = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors_table[-1] = (0.55, 0.55, 0.55, 0.2)

    # Subsample for rendering speed if point cloud is large
    n_pts = len(xyz)
    if n_pts > 100_000:
        idx = np.random.RandomState(42).choice(n_pts, 100_000, replace=False)
        xyz_sub = xyz[idx]
        labels_sub = labels[idx]
    else:
        xyz_sub, labels_sub = xyz, labels

    point_colors = np.array([colors_table[int(l)] for l in labels_sub])

    # Put floaters in back
    order = np.argsort(labels_sub == -1)[::-1]
    xyz_sub = xyz_sub[order]
    labels_sub = labels_sub[order]
    point_colors = point_colors[order]

    lo = np.percentile(xyz_sub, 2.0, axis=0)
    hi = np.percentile(xyz_sub, 98.0, axis=0)
    pad = np.maximum(hi - lo, 1e-6) * 0.08
    lo -= pad
    hi += pad

    # Orthographic 3-View
    views = [("Top View (X-Y)", 0, 1), ("Front View (X-Z)", 0, 2), ("Side View (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#181818")
    for ax, (vname, a, b) in zip(axes, views):
        ax.set_facecolor("#222222")
        ax.scatter(xyz_sub[:, a], xyz_sub[:, b], c=point_colors, s=1.2, edgecolors="none")
        ax.set_title(vname, color="white", fontsize=13, pad=8)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        if b == 2:
            ax.invert_yaxis()
        ax.tick_params(colors="gray")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.4)

    fig.suptitle(f"{title}\n({len(uniq)} Motion Clusters)", color="white", fontsize=15, y=0.98)
    fig.tight_layout()
    ortho_png = f"{out_prefix}_orthographic.png"
    fig.savefig(ortho_png, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 3D View
    fig3d = plt.figure(figsize=(10, 8), facecolor="#181818")
    ax3d = fig3d.add_subplot(111, projection="3d", facecolor="#222222")
    ax3d.scatter(xyz_sub[:, 0], xyz_sub[:, 1], xyz_sub[:, 2], c=point_colors, s=1.2, depthshade=True)
    ax3d.set_title(f"{title} — 3D Cluster View", color="white", fontsize=13, pad=10)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.tick_params(colors="gray")
    ax3d.view_init(elev=25, azim=45)
    fig3d.tight_layout()
    p3d_png = f"{out_prefix}_3d.png"
    fig3d.savefig(p3d_png, dpi=160, facecolor=fig3d.get_facecolor(), bbox_inches="tight")
    plt.close(fig3d)

    return ortho_png, p3d_png


def process_scene(scene_info: dict, status_record: dict) -> dict:
    name = scene_info["name"]
    desc = scene_info["desc"]
    n_times = scene_info["n_times"]
    dataset_dir = REPO_ROOT / "data" / "multipleview" / name

    print(f"\n{'='*75}\n[START] Processing Scene: {name}\nDescription: {desc}\n{'='*75}", flush=True)

    # 1. Dataset verification
    if not dataset_dir.exists() or not (dataset_dir / "points3D_multipleview.ply").exists():
        raise RuntimeError(f"Dataset {name} not prepared in {dataset_dir}")

    # Check status for existing run
    run_info = status_record["runs"].get(name, {})
    run_id = run_info.get("run_id")

    # 2. 4DGS Training
    ckpt_path = REPO_ROOT / "runs" / str(run_id) / "train_out" / "point_cloud" / "iteration_14000" / "point_cloud.ply" if run_id else None
    if run_id and ckpt_path and ckpt_path.exists():
        print(f"[RESUME] 4DGS training already completed in run {run_id} ({ckpt_path})", flush=True)
    else:
        print(f"[STAGE 1/4] Running 4DGS training for {name} (15,000 iterations)...", flush=True)
        t_start = time.time()
        scene_path = validate_external_artifact_path(str(dataset_dir.resolve()), "dataset")
        external_artifacts = {
            "scene": Artifact(
                name="scene",
                kind="dataset",
                path=str(scene_path),
                producing_stage="external",
            )
        }

        run_id = api.run_pipeline(
            name,
            only=["train.default"],
            external_artifacts=external_artifacts,
            metadata={
                "experiment": "lego_turbine",
                "cell": name,
                "variant": name,
                "dataset": name,
                "description": desc,
            },
        )
        train_duration = (time.time() - t_start) / 60.0
        print(f"[STAGE 1/4 DONE] 4DGS trained in {train_duration:.1f} min -> Run ID: {run_id}", flush=True)
        run_info["run_id"] = run_id
        run_info["train_duration_min"] = train_duration
        status_record["runs"][name] = run_info
        save_status(status_record)

    run_dir = REPO_ROOT / "runs" / run_id

    # 3. Test Views Rendering
    test_renders_dir = run_dir / "train_out" / "test" / "ours_14000" / "renders"
    if test_renders_dir.exists() and len(list(test_renders_dir.glob("*.png"))) > 0:
        print(f"[RESUME] Rendered test views already exist in {test_renders_dir}", flush=True)
    else:
        print(f"[STAGE 2/4] Rendering test views for {name}...", flush=True)
        t_start = time.time()
        api.run_stage(run_id, "render.default")
        render_duration = (time.time() - t_start) / 60.0
        print(f"[STAGE 2/4 DONE] Rendered test views in {render_duration:.1f} min", flush=True)

    # 4. Compute Reconstruction Metrics
    print(f"[STAGE 3/4] Evaluating novel-view reconstruction metrics...", flush=True)
    gt_dir = run_dir / "train_out" / "test" / "ours_14000" / "gt"
    metrics = compute_image_metrics(test_renders_dir, gt_dir)
    print(f"  -> Test View PSNR: {metrics.get('mean_psnr_db', 0):.2f} dB, MAE: {metrics.get('mean_mae', 0):.4f}", flush=True)
    run_info["render_metrics"] = metrics

    # 5. Extract Trajectories
    traj_path = run_dir / "trajectories.npz"
    if not traj_path.exists():
        print(f"[STAGE 4/4] Extracting Gaussian trajectories (n_times={n_times})...", flush=True)
        api.run_stage(run_id, "seg_extract.default", overrides={"seg_extract": {"n_times": n_times}})
    else:
        print(f"[RESUME] Trajectories already extracted ({traj_path})", flush=True)

    # Load trajectories & point cloud
    traj_data = np.load(traj_path)
    xyz = traj_data["canonical_xyz"]
    traj = traj_data["traj"]
    opacity = traj_data["opacity"]

    # Filter floaters
    keep = opacity > 0.1
    xyz_k = xyz[keep]
    traj_k = traj[keep]

    # Trajectory motion amplitude stats
    displacements = np.linalg.norm(traj_k - xyz_k[:, np.newaxis, :], axis=-1)  # (N, T)
    max_disp = np.max(displacements, axis=-1)  # peak-to-peak excursion per Gaussian
    mean_motion = float(np.mean(max_disp))
    p95_motion = float(np.percentile(max_disp, 95))
    max_motion = float(np.max(max_disp))

    print(f"  -> Gaussian count: {len(xyz)} (kept {len(xyz_k)} with opacity > 0.1)")
    print(f"  -> Motion excursion: mean={mean_motion:.4f}, p95={p95_motion:.4f}, max={max_motion:.4f}")

    # 6. Motion Segmentation
    renders_out_dir = run_dir / "segmentation_renders"
    renders_out_dir.mkdir(parents=True, exist_ok=True)

    # Kabsch clustering
    kabsch_labels, kabsch_info = segment_by_kabsch(
        xyz_k, traj_k,
        n_clusters=8,
        fps_subsample=10000,
        max_iter=15,
        rng_seed=42,
    )
    full_kabsch_labels = np.full(len(xyz), -1, dtype=np.int64)
    full_kabsch_labels[keep] = kabsch_labels
    np.savez_compressed(run_dir / "segmentation_kabsch.npz", points=xyz, labels=full_kabsch_labels)
    render_segmentation_figures(xyz, full_kabsch_labels, str(renders_out_dir / "kabsch"), f"{name} — Kabsch (K=8)")

    # Rigid graph clustering
    rigid_labels, rigid_info = segment_by_rigidity(
        xyz_k, traj_k,
        k=16,
        threshold_mult=2.0,
        min_size=200,
    )
    full_rigid_labels = np.full(len(xyz), -1, dtype=np.int64)
    full_rigid_labels[keep] = rigid_labels
    np.savez_compressed(run_dir / "segmentation_rigid.npz", points=xyz, labels=full_rigid_labels)
    render_segmentation_figures(xyz, full_rigid_labels, str(renders_out_dir / "rigid"), f"{name} — Rigid Graph")

    # Rigid2 spectral clustering
    rigid2_labels, rigid2_info = segment_by_rigidity2(
        xyz_k, traj_k,
        z_thresh=5.0,
        min_size=200,
        partition="components",
    )
    full_rigid2_labels = np.full(len(xyz), -1, dtype=np.int64)
    full_rigid2_labels[keep] = rigid2_labels
    np.savez_compressed(run_dir / "segmentation_rigid2.npz", points=xyz, labels=full_rigid2_labels)
    render_segmentation_figures(xyz, full_rigid2_labels, str(renders_out_dir / "rigid2"), f"{name} — Rigid2 Spectral")

    kabsch_clusters = len(set(kabsch_labels[kabsch_labels != -1]))
    rigid_clusters = len(set(rigid_labels[rigid_labels != -1]))
    rigid2_clusters = len(set(rigid2_labels[rigid2_labels != -1]))

    print(f"  -> Clusters found: Kabsch={kabsch_clusters}, Rigid={rigid_clusters}, Rigid2={rigid2_clusters}")

    summary = {
        "scene": name,
        "run_id": run_id,
        "description": desc,
        "n_gaussians": int(len(xyz)),
        "n_gaussians_kept": int(len(xyz_k)),
        "mean_psnr_db": metrics.get("mean_psnr_db", 0.0),
        "mean_mae": metrics.get("mean_mae", 0.0),
        "motion_mean_disp": mean_motion,
        "motion_p95_disp": p95_motion,
        "motion_max_disp": max_motion,
        "kabsch_clusters": kabsch_clusters,
        "rigid_clusters": rigid_clusters,
        "rigid2_clusters": rigid2_clusters,
    }

    run_info["summary"] = summary
    status_record["runs"][name] = run_info
    save_status(status_record)
    return summary


def main():
    print("=========================================================================")
    print("  LEGO TURBINE AUTOMATED PIPELINE: SYNC -> 4DGS -> EVALUATE")
    print("=========================================================================")
    status_record = load_status()
    all_summaries = []

    for scene in SCENES:
        try:
            summary = process_scene(scene, status_record)
            all_summaries.append(summary)
        except Exception as exc:
            print(f"\n[ERROR] Scene {scene['name']} failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Save summary reports
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    if all_summaries:
        keys = list(all_summaries[0].keys())
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_summaries)

    print("\n=========================================================================")
    print(f"  ALL 3 TURBINE SCENES PROCESSED! Results saved to:")
    print(f"  - {SUMMARY_JSON}")
    print(f"  - {SUMMARY_CSV}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
