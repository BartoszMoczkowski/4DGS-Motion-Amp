"""Run the baseline motion segmentation (Option B: rigidity-graph clustering) over the
already-trained grid/sweep 4DGS models, via the orchestrator's seg stages.

For each run in ``scene-gen/run_grid_4dgs.py``'s batch (3 grid + 4 sweep), this:

1. Seeds the run's manifest with its ``gt_segmentation`` artifact
   (``convert_out/data/multipleview/<name>/gt_segmentation.npz``, written by ``convert.default``
   from the capture's ``points3D_labels.npy``). ``model`` is already in the manifest from
   ``train.default``.
2. Runs ``seg_extract.default`` (GPU: samples the deformation field into ``trajectories.npz``),
   ``segment.rigid`` (host: rigidity-graph clustering) and ``seg_eval.default`` (host: ARI +
   best-match IoU vs GT) on the same run id.
3. Appends one row per run to ``runs/grid_seg_results.csv`` with the eval summary.

Option A (``segment.mbs`` / MultiBodySync) runs with ``--impl mbs`` (results in
``runs/grid_seg_mbs_results.csv``); it reuses the ``trajectories.npz`` already extracted by the
Option-B pass, so only ``segment.mbs`` + ``seg_eval.default`` run. It needs the MotNet
checkpoint at ``submodules/multibody-sync-4dgs/ckpt/mbs_full.pth.tar`` (downloaded per
``orchestrator/planning/WINDOWS_SETUP.md`` §7) and is unverified on real data (see AGENTS.md) —
the first run JIT-compiles the submodule's CUDA ops inside the container, expect a few minutes.

T18's upgraded Option B (``segment.rigid2`` — FFT denoising + calibrated z-scores + spectral
partition, proposal 06) runs with ``--impl rigid2`` (results in
``runs/grid_seg_rigid2_results.csv``); like ``mbs`` it reuses the extracted trajectories, and
it additionally sets ``segment.rigid2.gt_segmentation_path`` per run so each run also emits a
``separability.json`` diagnostic (z-score AUROC vs GT — the go/no-go signal for per-edge
methods, ``docs/proposals/IMPLEMENTATION_PLAN.md`` §3).

T20's Kabsch EM (``segment.kabsch`` — iterative rigid-body fitting, proposal 05) runs with
``--impl kabsch`` (results in ``runs/grid_seg_kabsch_results.csv``); reuses extracted
trajectories.  FPS subsample (default 5k) + FFT-fingerprint init + BIC model selection.

T19's ROI motion gate (``roi.motion_gate`` + ``segment.rigid2``, proposal 01) runs with
``--impl rigid2_roi`` (results in ``runs/grid_seg_rigid2_roi_results.csv``); reuses extracted
trajectories and prepends the ROI stage so the segmentation runs only on the gated region.

Idempotent: a run whose manifest already has a successful ``seg_eval.default`` is skipped.
``force=True`` for the same reason as in ``run_grid_4dgs.py`` — the cross-run cache keys on
resolved config + input content hashes, the seg configs are identical across runs and the
``model`` directory artifact carries no hash, so without force all runs would reuse the first
run's trajectories/segmentation.

Usage (from the repo root, workspace venv):

    .venv\Scripts\python.exe scene-gen/run_grid_seg.py                # Option B (rigid, default)
    .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl mbs     # Option A (MultiBodySync)
    .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl rigid2  # T18 upgraded Option B
    .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl kabsch  # T20 Kabsch EM
    .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl rigid2_roi  # T19 ROI + rigid2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))

from pipeline.api import _stage_config_for  # noqa: E402
from pipeline.artifacts import Artifact, load_manifest, update_manifest  # noqa: E402
from pipeline.config import validate_config  # noqa: E402
from pipeline.dag import run_dag  # noqa: E402

RESULTS_CSV = {
    "rigid": REPO_ROOT / "runs" / "grid_seg_results.csv",
    "mbs": REPO_ROOT / "runs" / "grid_seg_mbs_results.csv",
    "rigid2": REPO_ROOT / "runs" / "grid_seg_rigid2_results.csv",
    "kabsch": REPO_ROOT / "runs" / "grid_seg_kabsch_results.csv",
    "rigid2_roi": REPO_ROOT / "runs" / "grid_seg_rigid2_roi_results.csv",
}

STAGES = {
    "rigid": ["seg_extract.default", "segment.rigid", "seg_eval.default"],
    # Option A, T18's rigid2, and T20's kabsch reuse trajectories.npz already extracted.
    "mbs": ["segment.mbs", "seg_eval.default"],
    "rigid2": ["segment.rigid2", "seg_eval.default"],
    "kabsch": ["segment.kabsch", "seg_eval.default"],
    "rigid2_roi": ["roi.motion_gate", "segment.rigid2", "seg_eval.default"],
}

PRESET = {
    "rigid": "pump01",
    "mbs": "pump01_segA",
    "rigid2": "pump01_segB2",
    "kabsch": "pump01_kabsch",
    "rigid2_roi": "pump01_roi_gate",
}

RUN_IDS = [
    "grid-A20mm_M2",
    "grid-A20mm_M4",
    "grid-A40mm_M8",
    "sweep-A40mm_M8-g10000",
    "sweep-A40mm_M8-g25000",
    "sweep-A40mm_M8-g50000",
    "sweep-A40mm_M8-g100000",
]


def already_done(run_id: str, impl: str) -> bool:
    try:
        manifest = load_manifest(run_id)
    except FileNotFoundError:
        return False
    # seg_eval runs for both impls, so key idempotency off the impl's own segment stage.
    rec = manifest.stages.get(f"segment.{impl}")
    return rec is not None and rec.status in ("success", "skipped")


def seed_gt(run_id: str) -> Path:
    """Add the ``gt_segmentation`` artifact (next to the run's converted scene) if missing.
    Returns the GT path (T18's rigid2 also passes it to the stage config for the
    separability diagnostic)."""
    manifest = load_manifest(run_id)
    scene = Path(manifest.artifacts["scene"].path)
    gt = scene / "gt_segmentation.npz"
    if "gt_segmentation" not in manifest.artifacts:
        if not gt.is_file():
            raise FileNotFoundError(f"{run_id}: no gt_segmentation.npz under {scene}")
        update_manifest(
            run_id,
            lambda m: m.artifacts.update(
                {"gt_segmentation": Artifact(name="gt_segmentation", kind="npz",
                                             path=str(gt), producing_stage="external")}
            ),
        )
    return gt


def run_one(run_id: str, resolved: dict, impl: str) -> None:
    if already_done(run_id, impl):
        print(f"[skip] {run_id} already segmented ({impl})")
        return
    print(f"[run] {run_id} ({impl})")
    gt_path = seed_gt(run_id)
    stages = STAGES[impl]
    run_dir = REPO_ROOT / "runs" / run_id
    eval_path = run_dir / "seg_eval_result.json"
    if impl == "mbs" and eval_path.is_file():
        # seg_eval writes a fixed filename; keep the rigid pass's summary before it's overwritten.
        backup = run_dir / "seg_eval_result_rigid.json"
        if not backup.exists():
            backup.write_bytes(eval_path.read_bytes())
    if impl == "rigid2":
        # Per-run GT for the separability diagnostic; and preserve whichever eval summary is
        # currently on disk (rigid or mbs) before seg_eval overwrites it.
        resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)
        if eval_path.is_file():
            backup = run_dir / "seg_eval_result_before_rigid2.json"
            if not backup.exists():
                backup.write_bytes(eval_path.read_bytes())
    if impl == "rigid2_roi":
        # Same as rigid2: per-run GT for separability + preserve existing eval summary.
        resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)
        if eval_path.is_file():
            backup = run_dir / "seg_eval_result_before_rigid2_roi.json"
            if not backup.exists():
                backup.write_bytes(eval_path.read_bytes())
    if impl == "kabsch":
        # Preserve any existing eval summary before seg_eval overwrites it.
        if eval_path.is_file():
            backup = run_dir / "seg_eval_result_before_kabsch.json"
            if not backup.exists():
                backup.write_bytes(eval_path.read_bytes())
    error = ""
    try:
        manifest = run_dag(run_id, stages, resolved, preset=PRESET[impl], force=True, stage_configs={
            name: _stage_config_for(name, resolved) for name in stages
        })
        status = manifest.status
        if status != "success":
            error = "; ".join(
                f"{n}: {manifest.stages[n].error}" for n in stages
                if manifest.stages[n].status == "failed"
            )
    except Exception as exc:  # keep the batch going on a single failed run
        manifest = None
        status = "exception"
        error = repr(exc)
        print(f"[FAIL] {run_id}: {error}")

    stages_rec = (manifest.stages if manifest else {})
    summary = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
    row = {
        "run_id": run_id,
        "status": status,
        "ari": summary.get("ari", ""),
        "mean_iou": summary.get("mean_iou", ""),
        "n_gt": summary.get("n_gt", ""),
        "n_pred": summary.get("n_pred", ""),
        "ari_within_roi": summary.get("ari_within_roi", ""),
        "n_roi_points": summary.get("n_roi_points", ""),
        "seg_extract_s": getattr(stages_rec.get("seg_extract.default"), "wall_time_s", "") or "",
        "segment_s": getattr(stages_rec.get(f"segment.{impl.replace('_roi', '')}"), "wall_time_s", "") or "",
        "error": error,
    }
    results_csv = RESULTS_CSV[impl]
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[done] {run_id}: {status} ari={row['ari']} mean_iou={row['mean_iou']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["rigid", "mbs", "rigid2", "kabsch", "rigid2_roi"], default="rigid",
                    help="segmentation backend: rigid = Option B rigidity graph (default), "
                         "mbs = Option A MultiBodySync MotNet (needs the downloaded checkpoint), "
                         "rigid2 = T18 upgraded Option B (denoise + calibrated z + spectral), "
                         "kabsch = T20 Kabsch EM (proposal 05), "
                         "rigid2_roi = T19 ROI motion gate + rigid2 (proposal 01)")
    args = ap.parse_args()

    # Same resolved config the training batch used (defaults for the seg sections: n_times=60,
    # rigid k=12/min_size=15, eval drop_floaters=False; pump01_segA adds the mbs checkpoint).
    resolved = validate_config(PRESET[args.impl]).model_dump()
    # Also write a label-colored PLY per run (path is relative to the run dir) for visual checks.
    resolved["seg_eval"]["recolored_ply"] = (
        "segmentation_colored.ply" if args.impl == "rigid"
        else f"segmentation_colored_{args.impl}.ply"
    )
    for run_id in RUN_IDS:
        run_one(run_id, resolved, args.impl)


if __name__ == "__main__":
    main()
