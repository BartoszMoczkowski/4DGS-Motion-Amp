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

T22's oracle-mask ceiling (``roi.mask_oracle`` + ``segment.rigid2``, proposal 02) runs with
``--impl mask_lift_oracle`` (results in ``runs/grid_seg_mask_lift_oracle_results.csv``);
reuses extracted trajectories and uses GT labels directly as a perfect ROI mask.

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
    .venv\Scripts\python.exe scene-gen/run_grid_seg.py --impl mask_lift_oracle  # T22 oracle ceiling
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

#: Which orchestrator stage name provides the segmentation for each --impl.
SEGMENT_STAGE: dict[str, str] = {
    "rigid": "segment.rigid",
    "mbs": "segment.mbs",
    "rigid2": "segment.rigid2",
    "kabsch": "segment.kabsch",
    "rigid2_roi": "segment.rigid2",
    "mask_lift_oracle": "segment.rigid2",
    "mask_lift": "segment.rigid2",
}

RESULTS_CSV = {
    "rigid": REPO_ROOT / "runs" / "grid_seg_results.csv",
    "mbs": REPO_ROOT / "runs" / "grid_seg_mbs_results.csv",
    "rigid2": REPO_ROOT / "runs" / "grid_seg_rigid2_results.csv",
    "kabsch": REPO_ROOT / "runs" / "grid_seg_kabsch_results.csv",
    "rigid2_roi": REPO_ROOT / "runs" / "grid_seg_rigid2_roi_results.csv",
    "mask_lift_oracle": REPO_ROOT / "runs" / "grid_seg_mask_lift_oracle_results.csv",
    "mask_lift": REPO_ROOT / "runs" / "grid_seg_mask_lift_results.csv",
}

STAGES = {
    "rigid": ["seg_extract.default", "segment.rigid", "seg_eval.default"],
    # Option A, T18's rigid2, and T20's kabsch reuse trajectories.npz already extracted.
    "mbs": ["segment.mbs", "seg_eval.default"],
    "rigid2": ["segment.rigid2", "seg_eval.default"],
    "kabsch": ["segment.kabsch", "seg_eval.default"],
    "rigid2_roi": ["roi.motion_gate", "segment.rigid2", "seg_eval.default"],
    "mask_lift_oracle": ["roi.mask_oracle", "segment.rigid2", "seg_eval.default"],
    "mask_lift": ["roi.mask_lift", "segment.rigid2", "seg_eval.default"],
}

PRESET = {
    "rigid": "pump01",
    "mbs": "pump01_segA",
    "rigid2": "pump01_segB2",
    "kabsch": "pump01_kabsch",
    "rigid2_roi": "pump01_roi_gate",
    "mask_lift_oracle": "pump01_mask_oracle",
    "mask_lift": "pump01_mask_lift",
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
    # For ROI-based impls, idempotency is keyed off the ROI stage so that changing the ROI
    # method forces a re-run even if the underlying segment stage already succeeded.
    seg_stage = SEGMENT_STAGE[impl]
    if impl in ("rigid2_roi", "mask_lift_oracle", "mask_lift"):
        roi_stage = {
            "rigid2_roi": "roi.motion_gate",
            "mask_lift_oracle": "roi.mask_oracle",
            "mask_lift": "roi.mask_lift",
        }[impl]
        rec = manifest.stages.get(roi_stage)
        return rec is not None and rec.status in ("success", "skipped")
    rec = manifest.stages.get(seg_stage)
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


def _backup_eval(run_dir: Path, suffix: str) -> None:
    """Preserve seg_eval_result.json before it gets overwritten."""
    eval_path = run_dir / "seg_eval_result.json"
    if not eval_path.is_file():
        return
    backup = run_dir / f"seg_eval_result_before_{suffix}.json"
    if not backup.exists():
        backup.write_bytes(eval_path.read_bytes())


def run_one(run_id: str, resolved: dict, impl: str) -> None:
    if already_done(run_id, impl):
        print(f"[skip] {run_id} already segmented ({impl})")
        return
    print(f"[run] {run_id} ({impl})")
    gt_path = seed_gt(run_id)
    stages = STAGES[impl]
    run_dir = REPO_ROOT / "runs" / run_id
    eval_path = run_dir / "seg_eval_result.json"

    # Backups before seg_eval overwrites the shared result file.
    if impl == "mbs":
        _backup_eval(run_dir, "rigid")
    if impl in ("rigid2", "rigid2_roi", "mask_lift_oracle", "mask_lift"):
        _backup_eval(run_dir, impl)
    if impl == "kabsch":
        _backup_eval(run_dir, "kabsch")

    # Per-run GT for separability diagnostic (rigid2-based impls).
    if impl in ("rigid2", "rigid2_roi", "mask_lift_oracle", "mask_lift"):
        resolved["segment"]["rigid2"]["gt_segmentation_path"] = str(gt_path)

    # For mask_lift, ensure masks_dir is set (user must provide it via CLI or preset).
    if impl == "mask_lift":
        masks_dir = resolved.get("roi", {}).get("mask_lift", {}).get("masks_dir", "")
        if not masks_dir:
            print(f"[SKIP] {run_id}: mask_lift requires roi.mask_lift.masks_dir")
            return

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
    seg_stage = SEGMENT_STAGE[impl]
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
        "segment_s": getattr(stages_rec.get(seg_stage), "wall_time_s", "") or "",
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
    ap.add_argument("--impl",
                    choices=["rigid", "mbs", "rigid2", "kabsch", "rigid2_roi",
                             "mask_lift_oracle", "mask_lift"],
                    default="rigid",
                    help="segmentation backend: rigid = Option B rigidity graph (default), "
                         "mbs = Option A MultiBodySync MotNet, "
                         "rigid2 = T18 upgraded Option B, "
                         "kabsch = T20 Kabsch EM, "
                         "rigid2_roi = T19 ROI motion gate + rigid2, "
                         "mask_lift_oracle = T22 perfect GT ROI ceiling, "
                         "mask_lift = T22 multi-view mask lifting")
    ap.add_argument("--masks-dir", type=str, default="",
                    help="per-camera mask directory (required for --impl mask_lift)")
    args = ap.parse_args()

    resolved = validate_config(PRESET[args.impl]).model_dump()
    if args.masks_dir:
        resolved["roi"]["mask_lift"]["masks_dir"] = args.masks_dir

    # Also write a label-colored PLY per run (path is relative to the run dir) for visual checks.
    resolved["seg_eval"]["recolored_ply"] = (
        "segmentation_colored.ply" if args.impl == "rigid"
        else f"segmentation_colored_{args.impl}.ply"
    )
    for run_id in RUN_IDS:
        run_one(run_id, resolved, args.impl)


if __name__ == "__main__":
    main()
