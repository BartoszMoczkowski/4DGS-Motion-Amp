"""Migration utility: wrap legacy `output/multipleview/<scene>` directories into modern `runs/<run_id>/`.

In early phases before the orchestrator DAG engine, scripts directly executed upstream 4DGaussians
and dumped files into `output/multipleview/<scene>/`.

This script migrates such a directory into the orchestrator convention (`runs/<run_id>/`) with:
- Stage-isolated output directories (`train_out/`, `seg_extract_out/`, `segment_rigid_out/`, etc.)
- A valid `manifest.json` registering the artifacts (`model`, `trajectories`, `segmentation`)
- A basic `config_snapshot.json`
- Compatibility with `amp explore <run_id>`, `amp render <run_id>`, `amp seg eval <run_id>`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure repo root is on sys.path if run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.artifacts.models import Artifact, RunManifest, StageRecord
from pipeline.artifacts.paths import ensure_run_dirs, manifest_path, run_dir, validate_run_id


def migrate_legacy_output(
    legacy_dir: Path,
    run_id: str,
    preset: str = "pump01",
    dry_run: bool = False,
    link: bool = False,
) -> Path:
    validate_run_id(run_id)
    target_dir = run_dir(run_id)

    if not legacy_dir.exists():
        raise FileNotFoundError(f"Legacy directory not found: {legacy_dir}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stages: dict[str, StageRecord] = {}
    artifacts: dict[str, Artifact] = {}

    print(f"Migrating {legacy_dir} -> {target_dir}")
    if dry_run:
        print("[dry-run] Would create target run directory and migrate files.")
        return target_dir

    ensure_run_dirs(run_id)

    def _copy_or_link(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        if link:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # 1. Train outputs: point_cloud
    point_cloud_src = legacy_dir / "point_cloud"
    if point_cloud_src.exists():
        train_out = target_dir / "train_out"
        train_out.mkdir(parents=True, exist_ok=True)
        _copy_or_link(point_cloud_src, train_out / "point_cloud")
        
        cfg_args = legacy_dir / "cfg_args"
        if cfg_args.exists():
            _copy_or_link(cfg_args, train_out / "cfg_args")

        artifacts["model"] = Artifact(
            name="model",
            kind="model",
            path=str(train_out),
            producing_stage="train.default",
            metadata={"legacy_source": str(legacy_dir)},
        )
        stages["train.default"] = StageRecord(
            status="success",
            start_time=now,
            end_time=now,
            artifacts=["model"],
        )

    # 2. Trajectories: trajectories.npz
    traj_src = legacy_dir / "trajectories.npz"
    if traj_src.exists():
        seg_extract_out = target_dir / "seg_extract_out"
        seg_extract_out.mkdir(parents=True, exist_ok=True)
        traj_dst = seg_extract_out / "trajectories.npz"
        _copy_or_link(traj_src, traj_dst)

        artifacts["trajectories"] = Artifact(
            name="trajectories",
            kind="npz",
            path=str(traj_dst),
            producing_stage="seg_extract.default",
            metadata={"legacy_source": str(traj_src)},
        )
        stages["seg_extract.default"] = StageRecord(
            status="success",
            start_time=now,
            end_time=now,
            artifacts=["trajectories"],
        )

    # 3. Rigid segmentation: segmentation.npz
    seg_src = legacy_dir / "segmentation.npz"
    if seg_src.exists():
        seg_out = target_dir / "segment_rigid_out"
        seg_out.mkdir(parents=True, exist_ok=True)
        seg_dst = seg_out / "segmentation.npz"
        _copy_or_link(seg_src, seg_dst)

        artifacts["segmentation"] = Artifact(
            name="segmentation",
            kind="npz",
            path=str(seg_dst),
            producing_stage="segment.rigid",
            metadata={"legacy_source": str(seg_src)},
        )
        
        # Previews if present
        for preview_file in legacy_dir.glob("segmentation_preview*"):
            preview_dst = seg_out / preview_file.name
            _copy_or_link(preview_file, preview_dst)

        stages["segment.rigid"] = StageRecord(
            status="success",
            start_time=now,
            end_time=now,
            artifacts=["segmentation"],
        )

    # 4. MBS segmentation if present
    seg_mbs_src = legacy_dir / "segmentation_mbs.npz"
    if seg_mbs_src.exists():
        seg_mbs_out = target_dir / "segment_mbs_out"
        seg_mbs_out.mkdir(parents=True, exist_ok=True)
        seg_mbs_dst = seg_mbs_out / "segmentation.npz"
        _copy_or_link(seg_mbs_src, seg_mbs_dst)

        artifacts["segmentation_mbs"] = Artifact(
            name="segmentation_mbs",
            kind="npz",
            path=str(seg_mbs_dst),
            producing_stage="segment.mbs",
            metadata={"legacy_source": str(seg_mbs_src)},
        )
        stages["segment.mbs"] = StageRecord(
            status="success",
            start_time=now,
            end_time=now,
            artifacts=["segmentation_mbs"],
        )

    # 5. Write manifest & config snapshot
    manifest = RunManifest(
        run_id=run_id,
        preset=preset,
        resolved_config={"preset": preset, "migrated_from": str(legacy_dir)},
        created_at=now,
        updated_at=now,
        status="success",
        metadata={"migrated": True, "legacy_path": str(legacy_dir)},
        stages=stages,
        artifacts=artifacts,
    )
    manifest_p = manifest_path(run_id)
    manifest_p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    config_snapshot_p = target_dir / "config_snapshot.json"
    config_snapshot_p.write_text(json.dumps(manifest.resolved_config, indent=2), encoding="utf-8")

    print(f"Successfully migrated {len(artifacts)} artifact(s) to {target_dir}")
    print(f"Manifest written: {manifest_p}")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy output/<dir> to runs/<run_id>")
    parser.add_argument("--legacy-dir", type=Path, required=True, help="Path to legacy output dir (e.g. output/multipleview/pump01)")
    parser.add_argument("--run-id", type=str, default="legacy-pump01-00000001", help="Target run ID")
    parser.add_argument("--preset", type=str, default="pump01", help="Preset name associated with run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without copying")
    parser.add_argument("--link", action="store_true", help="Hardlink files instead of copying when possible")

    args = parser.parse_args()
    try:
        migrate_legacy_output(
            legacy_dir=args.legacy_dir,
            run_id=args.run_id,
            preset=args.preset,
            dry_run=args.dry_run,
            link=args.link,
        )
        return 0
    except Exception as e:
        print(f"Migration error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
