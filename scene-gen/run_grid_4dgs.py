"""Run the captured grid scenes through 4DGS reconstruction via the orchestrator.

Two experiment families, all serial (single GPU):

1. **Grid runs** -- one full reconstruction per complete grid capture cell
   (``capture_pump_A20mm_M2``, ``capture_pump_A20mm_M4``, ``capture_pump_A40mm_M8``
   under ``Q:/Omniverse/renders/``), default ``pump01`` preset config.

2. **Gaussian-count sweep** -- the largest cell (``capture_pump_A40mm_M8``) retrained
   with 10k/25k/50k/100k initial Gaussians and densification frozen
   (``optim.densify_from_iter = 1e9``) so the final count stays at the init count.
   Sub-100k variants get a derived capture dir (``<cell>_g<N>/``) with a subsampled
   ``points3D_gt.ply``/``points3D_labels.npy`` and directory junctions to the original
   per-camera folders (no frame copying).

Usage (from the repo root, workspace venv):

    .venv/Scripts/python.exe scene-gen/run_grid_4dgs.py            # full batch
    .venv/Scripts/python.exe scene-gen/run_grid_4dgs.py --smoke    # one tiny run to validate

Results are appended to ``runs/grid_4dgs_results.csv`` (one row per run, with stage
timings, peak VRAM and the trained Gaussian count parsed from the output PLY header).
Re-running the script skips runs whose manifest already finished successfully.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))

from pipeline.api import _stage_config_for  # noqa: E402
from pipeline.artifacts import Artifact, create_run, update_manifest  # noqa: E402
from pipeline.config import validate_config  # noqa: E402
from pipeline.dag import run_dag  # noqa: E402
from pipeline.vendored.host.convert import _read_ply_xyz_rgb, write_ply  # noqa: E402

RENDERS_ROOT = Path("Q:/Omniverse/renders")
RESULTS_CSV = REPO_ROOT / "runs" / "grid_4dgs_results.csv"

STAGES = ["convert.default", "train.default", "render.default"]

GRID_CELLS = ["capture_pump_A20mm_M2", "capture_pump_A20mm_M4", "capture_pump_A40mm_M8"]
SWEEP_CELL = "capture_pump_A40mm_M8"
SWEEP_TARGETS = [10_000, 25_000, 50_000, 100_000]


def make_capture_variant(src: Path, dst: Path, n_points: int, seed: int = 0) -> Path:
    """Derived capture dir with a subsampled init cloud; camNN dirs are junctions."""
    if dst.exists():
        print(f"[variant] {dst} already exists, reusing")
        return dst
    dst.mkdir(parents=True)
    xyz, rgb = _read_ply_xyz_rgb(str(src / "points3D_gt.ply"))
    labels = np.load(src / "points3D_labels.npy")
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(xyz), size=n_points, replace=False))
    write_ply(str(dst / "points3D_gt.ply"), xyz[idx], rgb[idx].astype(np.float32) / 255.0)
    np.save(dst / "points3D_labels.npy", labels[idx])
    shutil.copy2(src / "cameras_gt.json", dst / "cameras_gt.json")
    for cam_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("cam")):
        link = dst / cam_dir.name
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link).replace("/", "\\"), str(cam_dir).replace("/", "\\")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"mklink {link} -> {cam_dir} failed: {r.stdout}{r.stderr}")
    print(f"[variant] {dst}: {n_points} points ({len(xyz)} available)")
    return dst


def base_resolved(cell: str, sweep: bool, smoke: bool, target: int | None = None) -> dict:
    resolved = validate_config("pump01").model_dump()
    # Unique convert name per variant: the cross-run DAG cache keys stages on their resolved
    # config alone (externally-seeded directory artifacts carry no content hash), so two runs
    # sharing a convert/train config would silently reuse each other's outputs.
    resolved["convert"]["name"] = cell if target is None else f"{cell}_g{target}"
    # The upstream render script accumulates every frame of a set on GPU/host before writing;
    # 2400-view train/test sets OOM the Docker VM (exit 137 at ~26%). The 300-view video set
    # fits comfortably and is what we actually inspect.
    resolved["render"]["skip_train"] = True
    resolved["render"]["skip_test"] = True
    if sweep:
        resolved["optim"]["densify_from_iter"] = 10**9  # freeze densification
    if smoke:
        resolved["optim"]["coarse_iterations"] = 50
        resolved["optim"]["iterations"] = 100
        resolved["train"]["test_iterations"] = []
        resolved["train"]["save_iterations"] = []
    return resolved


def count_gaussians(run_dir: Path) -> int | None:
    plys = sorted(run_dir.glob("train_out/point_cloud/iteration_*/point_cloud.ply"))
    if not plys:
        return None
    with open(plys[-1], "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return None


def already_done(run_id: str) -> bool:
    manifest_path = REPO_ROOT / "runs" / run_id / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        return False
    return manifest.get("stages", {}).get("render.default", {}).get("status") in ("success", "skipped")


def run_one(run_id: str, cell: str, target: int | None, capture_dir: Path, resolved: dict) -> None:
    if already_done(run_id):
        print(f"[skip] {run_id} already finished")
        return
    print(f"[run] {run_id} (capture={capture_dir})")
    create_run(run_id, "pump01", resolved, stage_names=STAGES)
    update_manifest(
        run_id,
        lambda m: m.artifacts.update(
            {"capture": Artifact(name="capture", kind="dataset",
                                 path=str(capture_dir), producing_stage="external")}
        ),
    )
    error = ""
    try:
        # force=True: the cross-run stage cache keys on resolved config only and directory
        # artifacts carry no content hash, so without force a later run whose config matches an
        # earlier run's (same convert name / same train bridge) silently reuses that run's
        # scene/model instead of building its own.
        manifest = run_dag(run_id, STAGES, resolved, preset="pump01", force=True, stage_configs={
            name: _stage_config_for(name, resolved) for name in STAGES
        })
        status = manifest.status
        if status != "success":
            error = "; ".join(
                f"{n}: {manifest.stages[n].error}" for n in STAGES
                if manifest.stages[n].status == "failed"
            )
    except Exception as exc:  # keep the batch going on a single failed run
        manifest = None
        status = "exception"
        error = repr(exc)
        print(f"[FAIL] {run_id}: {error}")

    stages = (manifest.stages if manifest else {})
    row = {
        "run_id": run_id,
        "cell": cell,
        "target_gaussians": target if target is not None else "",
        "actual_gaussians": count_gaussians(REPO_ROOT / "runs" / run_id) or "",
        "status": status,
        "convert_s": getattr(stages.get("convert.default"), "wall_time_s", "") or "",
        "train_s": getattr(stages.get("train.default"), "wall_time_s", "") or "",
        "render_s": getattr(stages.get("render.default"), "wall_time_s", "") or "",
        "train_peak_vram_mb": getattr(stages.get("train.default"), "peak_vram_mb", "") or "",
        "error": error,
    }
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[done] {run_id}: {status} gaussians={row['actual_gaussians']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="single tiny run (sweep g10000, 100 iters) to validate the path")
    args = ap.parse_args()

    if args.smoke:
        capture = RENDERS_ROOT / SWEEP_CELL
        run_one("sweep-A40mm_M8-g10000-smoke", SWEEP_CELL, 10_000,
                make_capture_variant(capture, RENDERS_ROOT / f"{SWEEP_CELL}_g10000", 10_000),
                base_resolved(SWEEP_CELL, sweep=True, smoke=True, target=10_000))
        return

    # Grid runs: default config, one per complete cell.
    for cell in GRID_CELLS:
        run_one(f"grid-{cell.replace('capture_pump_', '')}", cell, None,
                RENDERS_ROOT / cell, base_resolved(cell, sweep=False, smoke=False))

    # Gaussian-count sweep on the largest cell, densification frozen.
    src = RENDERS_ROOT / SWEEP_CELL
    for n in SWEEP_TARGETS:
        capture = src if n == 100_000 else make_capture_variant(src, RENDERS_ROOT / f"{SWEEP_CELL}_g{n}", n)
        run_one(f"sweep-A40mm_M8-g{n}", SWEEP_CELL, n, capture,
                base_resolved(SWEEP_CELL, sweep=True, smoke=False, target=n))


if __name__ == "__main__":
    main()
