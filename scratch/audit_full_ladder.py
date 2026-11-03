import json
import statistics
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_multiview_wobble import evaluate_3d_trajectories

fps = 20.004025315610498
f0 = 5.690745

runs = {
    "E1b": "runs/turbine_r1_missing_6_w1_e1b-5a361a54",
    "E2": "runs/turbine_r1_missing_6_w3_e2-fda6282c",
    "E2-residual-L2": "runs/turbine_r1_missing_6_w3_e2_residual_l2-10e6ad8b",
    "E3-Active": "runs/turbine_r1_missing_6_w4_e3-625cdb58",
}

results = {}

# Reference canonical coordinates for matched population: E2
e2_data = np.load(Path(runs["E2"]) / "trajectories.npz")
e2_canon = e2_data.get("canonical_xyz", e2_data["traj"][:, 0, :])
e2_tree = cKDTree(e2_canon)

for name, rpath in runs.items():
    rdir = Path(rpath)
    res = {"name": name, "path": str(rdir)}
    
    # 1. PSNR / SSIM from eval_w2.json or eval_w1.json
    eval_files = list(rdir.glob("eval_*.json"))
    if eval_files:
        with open(eval_files[0], "r", encoding="utf-8") as f:
            edata = json.load(f)
        cams = edata.get("cameras", {})
        psnrs = [v.get("psnr_db", v.get("psnr")) for v in cams.values() if v.get("psnr_db") or v.get("psnr")]
        ssims = [v.get("ssim") for v in cams.values() if v.get("ssim")]
        res["mean_psnr"] = float(statistics.mean(psnrs)) if psnrs else None
        res["mean_ssim"] = float(statistics.mean(ssims)) if ssims else None
    
    # 2. 3D Trajectory metrics (Full population)
    tfile = rdir / "trajectories.npz"
    if tfile.exists():
        tdata = np.load(tfile)
        traj = tdata["traj"]
        canon = tdata.get("canonical_xyz", traj[:, 0, :])
        N, T, _ = traj.shape
        res["N_gaussians"] = int(N)
        
        diag_full = evaluate_3d_trajectories(traj, fps=fps, f0=f0, canonical_xyz=canon)
        res["full"] = {
            "p50_mm": float(diag_full.p50_amp * 1000),
            "p90_mm": float(diag_full.p90_amp * 1000),
            "p99_mm": float(diag_full.p99_amp * 1000),
            "max_mm": float(diag_full.max_amp * 1000),
            "gini": float(diag_full.gini_coefficient),
            "coherence_r": float(diag_full.phase_coherence_r),
            "bg_drift_ratio": float(diag_full.drift_ratio),
        }
        
        # 3. Matched population (points within 2cm of E2 canonical points)
        dist, idx = e2_tree.query(canon)
        mask = dist < 0.02
        res["matched_count"] = int(np.sum(mask))
        if np.sum(mask) > 100:
            diag_matched = evaluate_3d_trajectories(traj[mask], fps=fps, f0=f0, canonical_xyz=canon[mask])
            res["matched"] = {
                "p50_mm": float(diag_matched.p50_amp * 1000),
                "p90_mm": float(diag_matched.p90_amp * 1000),
                "p99_mm": float(diag_matched.p99_amp * 1000),
                "max_mm": float(diag_matched.max_amp * 1000),
                "gini": float(diag_matched.gini_coefficient),
                "coherence_r": float(diag_matched.phase_coherence_r),
                "bg_drift_ratio": float(diag_matched.drift_ratio),
            }
            
    results[name] = res

print(json.dumps(results, indent=2))
with open("scratch/full_ladder_audit.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
