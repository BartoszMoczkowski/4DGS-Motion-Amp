"""Experiment runner (T27): expand a spec into cells × sweep variants × segment impls and
drive everything through the Layer-1 API (``run_pipeline`` / ``run_stage``).

Replaces ``scene-gen/run_grid_4dgs.py`` + ``run_grid_seg.py``'s direct ``run_dag`` bypass,
hand-built run IDs, and ``force=True`` usage:

- Run IDs come from the API scheme (``<experiment>-<variant>-<8hex>``).
- Idempotency lives on the manifest layer: a variant is reused when a run exists whose
  ``metadata.{experiment,cell,variant}`` matches and whose pipeline stages all succeeded —
  never a directory-existence check. Segment phases rely on the stage cache (Phase A's
  per-impl keying + content-hashed directory artifacts make same-config reruns skip safely),
  so there is no ``force=True`` anywhere.
- Sweep variant capture dirs are materialized by :mod:`pipeline.experiments.variants`
  (junction-based subsampling, Windows-only).
- Capture provenance (grid-manifest cell params + the cell's ``_motion.json``) lands in
  ``RunManifest.metadata``.

The segmentation entry table mirrors ``run_grid_seg.py``: entry name -> stage list; entries
sharing a stage impl (``rigid2_roi``/``mask_lift_*`` -> ``segment.rigid2``) are kept distinct in
results via ``seg_eval.result_impl``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from ..artifacts import Artifact, load_manifest, update_manifest
from ..artifacts.manifest import ManifestError
from ..config import validate_config
from ..config.resolver import _deep_merge
from . import results as _results
from .spec import ExperimentSpec, SegmentEntry, load_grid_manifest, load_spec
from .variants import count_ply_vertices, make_capture_variant

#: Entry name -> stage list, mirroring scene-gen/run_grid_seg.py's STAGES table. The
#: ``extract: true`` entry prepends ``seg_extract.default``; all others reuse its trajectories.
SEG_ENTRY_STAGES: dict[str, list[str]] = {
    "rigid": ["segment.rigid", "seg_eval.default"],
    "mbs": ["segment.mbs", "seg_eval.default"],
    "rigid2": ["segment.rigid2", "seg_eval.default"],
    "kabsch": ["segment.kabsch", "seg_eval.default"],
    "rigid2_roi": ["roi.motion_gate", "segment.rigid2", "seg_eval.default"],
    "mask_lift_oracle": ["roi.mask_oracle", "segment.rigid2", "seg_eval.default"],
    "mask_lift": ["roi.mask_lift", "segment.rigid2", "seg_eval.default"],
    "multicut": ["segment.multicut", "seg_eval.default"],
}

#: Entries whose stage impl is segment.rigid2 (they get the per-run GT separability config).
_RIGID2_FAMILY = ("rigid2", "rigid2_roi", "mask_lift_oracle", "mask_lift")

#: Stage-config sections taken from the segment entry's own preset (which may differ from the
#: run's pipeline preset — e.g. grid_seg's seg_extract.n_times=240 for 240-frame captures).
_SEG_SECTIONS = ("seg_extract", "segment", "roi", "seg_eval")


class Variant:
    """One expansion unit: a grid cell, or a Gaussian-count sweep variant of one.

    ``capture_dir`` is the cell's *source* capture dir; :meth:`materialize` returns the actual
    pipeline input (a derived subsampled dir for sweep variants, the source itself for the
    100%-target case — see :func:`variants.make_capture_variant`)."""

    def __init__(self, cell: str, variant: str, target: Optional[int], capture_dir: Path,
                 cell_info: dict[str, Any], motion: dict[str, Any], seed: int = 0) -> None:
        self.cell = cell
        self.variant = variant
        self.target = target
        self.capture_dir = capture_dir
        self.cell_info = cell_info
        self.motion = motion
        self.seed = seed

    @property
    def is_sweep(self) -> bool:
        return self.target is not None

    @property
    def sweep_dir(self) -> Path:
        assert self.target is not None
        return self.capture_dir.parent / f"{self.capture_dir.name}_g{self.target}"

    @property
    def convert_name(self) -> str:
        """Unique per variant, matching the historical convention
        (``capture_pump_A40mm_M8_g10000`` for sweeps) — also keeps each converted scene
        self-describing."""
        return f"{self.capture_dir.name}_g{self.target}" if self.is_sweep else self.capture_dir.name

    def materialize(self) -> Path:
        if not self.is_sweep:
            return self.capture_dir
        return make_capture_variant(self.capture_dir, self.sweep_dir, self.target, seed=self.seed)


def _variants(spec: ExperimentSpec, grid_manifest: dict[str, Any]) -> list[Variant]:
    cells = {c["cell"]: c for c in grid_manifest.get("cells", [])}
    manifest_dir = Path(spec.capture.manifest)
    if not manifest_dir.is_absolute():
        from ..artifacts.paths import REPO_ROOT

        manifest_dir = REPO_ROOT / manifest_dir
    manifest_dir = manifest_dir.parent

    def _capture_dir(cell: str) -> Path:
        info = cells[cell]
        if info.get("capture_dir"):
            return Path(info["capture_dir"])
        if spec.capture.renders_root:
            return Path(spec.capture.renders_root) / f"capture_{cell}"
        raise ValueError(
            f"experiment {spec.name!r}: cell {cell!r} has no capture_dir in the grid manifest "
            f"and capture.renders_root is unset"
        )

    def _motion(cell: str) -> dict[str, Any]:
        p = manifest_dir / f"{cell}_motion.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    variants: list[Variant] = []
    for cell in spec.capture.cells:
        variants.append(Variant(cell, cell, None, _capture_dir(cell), cells[cell], _motion(cell)))
    if spec.sweep is not None:
        src = _capture_dir(spec.sweep.cell)
        for target in spec.sweep.targets:
            variants.append(Variant(
                spec.sweep.cell, f"{spec.sweep.cell}_g{target}", target, src,
                cells[spec.sweep.cell], _motion(spec.sweep.cell), seed=spec.sweep.seed,
            ))
    return variants


def _find_runs(spec_name: str, *, runs_root: Optional[Path] = None) -> dict[tuple[str, str], Any]:
    """Manifests indexed by ``(cell, variant)`` for runs tagged with this experiment's name."""
    from ..artifacts import get_runs_root

    root = Path(runs_root) if runs_root is not None else get_runs_root()
    found: dict[tuple[str, str], Any] = {}
    if not root.is_dir():
        return found
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            m = load_manifest(run_dir.name, runs_root=root)
        except (FileNotFoundError, ManifestError):
            continue
        if m.metadata.get("experiment") == spec_name:
            found[(str(m.metadata.get("cell", "")), str(m.metadata.get("variant", "")))] = m
    return found


def _pipeline_done(manifest: Any, stages: list[str]) -> bool:
    return all(
        (rec := manifest.stages.get(s)) is not None and rec.status in ("success", "skipped")
        for s in stages
    )


def _metadata_for(spec: ExperimentSpec, v: Variant) -> dict[str, Any]:
    provenance = {
        k: v.cell_info.get(k)
        for k in ("base_amp_mm", "multiplier", "amplified_peak_mm", "materials")
        if k in v.cell_info
    }
    for k in ("num_frames", "fps", "freq_hz", "amplified_part", "seed"):
        if k in v.motion:
            provenance[k] = v.motion[k]
    provenance["capture_dir"] = str(v.capture_dir)
    return {
        "experiment": spec.name,
        "cell": v.cell,
        "variant": v.variant,
        "target_gaussians": v.target if v.target is not None else "",
        "capture": provenance,
    }


def _seed_gt(run_id: str, spec: ExperimentSpec) -> Optional[Path]:
    """Seed ``gt_segmentation`` from the run's converted scene (correct per variant — sweep
    variants carry subsampled GT), falling back to the spec's ``eval.gt``. Mirrors
    ``run_grid_seg.seed_gt``."""
    manifest = load_manifest(run_id)
    if "gt_segmentation" in manifest.artifacts:
        return Path(manifest.artifacts["gt_segmentation"].path)
    gt: Optional[Path] = None
    scene = manifest.artifacts.get("scene")
    if scene is not None and (Path(scene.path) / "gt_segmentation.npz").is_file():
        gt = Path(scene.path) / "gt_segmentation.npz"
    elif spec.eval is not None:
        candidate = Path(spec.eval.gt)
        if not candidate.is_absolute():
            from ..artifacts.paths import REPO_ROOT

            candidate = REPO_ROOT / candidate
        if candidate.is_file():
            gt = candidate
    if gt is None:
        return None
    gt_path = gt

    def _mut(m: Any) -> None:
        m.artifacts["gt_segmentation"] = Artifact(
            name="gt_segmentation", kind="npz", path=str(gt_path), producing_stage="external"
        )

    update_manifest(run_id, _mut)
    return gt_path


def _segment_overrides(entry: SegmentEntry, gt_path: Optional[Path]) -> dict[str, Any]:
    """The entry preset's seg-relevant sections + entry overrides + per-run wirings, as a
    ``run_stage``-style overrides dict."""
    impl_resolved = validate_config(entry.preset).model_dump()
    overrides: dict[str, Any] = {
        sec: impl_resolved[sec] for sec in _SEG_SECTIONS if sec in impl_resolved
    }
    overrides = _deep_merge(overrides, entry.overrides)
    if entry.impl in _RIGID2_FAMILY and gt_path is not None:
        overrides.setdefault("segment", {}).setdefault("rigid2", {})[
            "gt_segmentation_path"
        ] = str(gt_path)
    # Per-entry result key (T27): entries sharing segment.rigid2 must not overwrite each
    # other's seg_eval result; also a label-colored PLY per entry for visual checks.
    overrides.setdefault("seg_eval", {})["result_impl"] = entry.impl
    overrides["seg_eval"]["recolored_ply"] = f"segmentation_colored_{entry.impl}.ply"
    return overrides


def _entry_stages(entry: SegmentEntry) -> list[str]:
    stages = list(SEG_ENTRY_STAGES[entry.impl])
    if entry.extract and "seg_extract.default" not in stages:
        stages.insert(0, "seg_extract.default")
    return stages


def run_experiment(
    spec_ref: str,
    *,
    dry_run: bool = False,
    only_cells: Optional[list[str]] = None,
    only_impls: Optional[list[str]] = None,
    log: Any = print,
) -> dict[str, Any]:
    """Run (or plan, with ``dry_run=True``) an experiment spec. Returns a summary dict.

    ``dry_run`` prints the full plan — variants, capture dirs, reused vs to-be-created runs,
    pipeline stages, and the segment impl matrix — and touches nothing on disk.
    """
    from .. import api

    spec = load_spec(spec_ref)
    if spec.scene_gen.enabled:
        raise NotImplementedError(
            f"experiment {spec.name!r}: scene_gen is not driven by the T27 runner — generate "
            "scenes first (amp capture gen-grid), then reference them via capture.manifest"
        )
    grid_manifest = load_grid_manifest(spec)
    variants = _variants(spec, grid_manifest)
    if only_cells:
        variants = [v for v in variants if v.cell in only_cells]
    entries = [e for e in spec.segment if only_impls is None or e.impl in only_impls]
    existing = _find_runs(spec.name)

    log(f"experiment {spec.name}: {len(variants)} variant(s), {len(entries)} segment impl(s)")

    summary: dict[str, Any] = {"spec": spec.name, "variants": []}
    for v in variants:
        prev = existing.get((v.cell, v.variant))
        reused = prev is not None and _pipeline_done(prev, spec.stages)
        run_id = prev.run_id if reused else f"{spec.name}-{v.variant}-{uuid.uuid4().hex[:8]}"
        # The pipeline input dir: the source capture, or the (possibly to-be-materialized)
        # sweep variant. make_capture_variant returns the SOURCE dir for a 100% target, so a
        # dry-run can't tell the two apart without reading the source PLY header — cheap, do it.
        pipeline_dir = v.capture_dir
        if v.is_sweep:
            src_ply = v.capture_dir / "points3D_gt.ply"
            available = count_ply_vertices(src_ply) if src_ply.is_file() else None
            pipeline_dir = (
                v.capture_dir if available is not None and v.target >= available else v.sweep_dir
            )

        plan = {
            "variant": v.variant,
            "cell": v.cell,
            "target": v.target,
            "capture_dir": str(pipeline_dir),
            "run_id": run_id,
            "reused": reused,
            "stages": spec.stages,
            "segment": [{"impl": e.impl, "preset": e.preset, "stages": _entry_stages(e)}
                        for e in entries],
        }
        summary["variants"].append(plan)
        log(f"[{'reuse' if reused else 'plan'}] {v.variant}: run_id={run_id} "
            f"capture={pipeline_dir}")
        for e in entries:
            log(f"    seg {e.impl}: {' + '.join(_entry_stages(e))} (preset {e.preset})")
        if dry_run:
            continue

        if not reused:
            if v.is_sweep:
                pipeline_dir = v.materialize()
            if not pipeline_dir.is_dir():
                raise FileNotFoundError(
                    f"capture dir for variant {v.variant!r} not found: {pipeline_dir} "
                    + ("(reuse_existing is set; capture it first, e.g. amp capture run)"
                       if spec.capture.reuse_existing else "")
                )
            overrides = _deep_merge(
                spec.pipeline.overrides, {"convert": {"name": v.convert_name}}
            )
            if v.is_sweep:
                overrides = _deep_merge(overrides, {"optim": {"densify_from_iter": 10**9}})
            api.run_pipeline(
                spec.pipeline.preset,
                run_id=run_id,
                only=spec.stages,
                overrides=overrides,
                external_artifacts={
                    "capture": Artifact(name="capture", kind="dataset",
                                        path=str(pipeline_dir), producing_stage="external"),
                },
                metadata=_metadata_for(spec, v),
            )
            _append_run_rows(run_id)

        if not entries:
            continue
        gt_path = _seed_gt(run_id, spec)
        for entry in entries:
            overrides = _segment_overrides(entry, gt_path)
            if entry.impl == "mask_lift":
                masks_dir = (
                    overrides.get("roi", {}).get("mask_lift", {}).get("masks_dir") or ""
                )
                if not masks_dir:
                    log(f"[skip] {run_id} mask_lift: roi.mask_lift.masks_dir is unset")
                    continue
            log(f"[seg] {run_id} ({entry.impl})")
            try:
                for stage in _entry_stages(entry):
                    api.run_stage(run_id, stage, overrides=overrides)
            except Exception as exc:  # keep the batch going on a single failed entry
                log(f"[FAIL] {run_id} ({entry.impl}): {exc}")
            _append_run_rows(run_id)
    return summary


def _append_run_rows(run_id: str) -> None:
    """Refresh this run's rows in the results store (manifest is the source of truth)."""
    from ..artifacts import get_runs_root, run_dir as _run_dir

    manifest = load_manifest(run_id)
    for row in _results.rows_for_manifest(manifest, _run_dir(run_id)):
        _results.append_result(row)
