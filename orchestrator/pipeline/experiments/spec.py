"""Declarative experiment specs (T27): one YAML = one grid/sweep experiment.

Specs live in top-level ``experiments/*.yaml`` and layer via ``extends:`` exactly like config
presets (``pipeline.config.resolver`` — the ``_deep_merge``/cycle-check logic is reused from
there). A spec is *fully declarative* (no Python): it names the grid cells, an optional
Gaussian-count sweep, the training pipeline preset + overrides, and the segmentation impl
matrix; ``pipeline.experiments.runner`` expands it into orchestrator runs.

Validation is eager: referenced presets must exist, segment entry names must be known (checked
against a static table — importing the stage registry here would pull numpy into a module the
CLI imports, breaking the ``amp --help`` lazy-import rule), and cells must exist in the
referenced grid manifest. Everything raises ``ValueError``/``FileNotFoundError`` with the spec
name in the message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field

from ..artifacts.paths import REPO_ROOT
from ..config.models import StrictModel
from ..config.resolver import _deep_merge, list_presets

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

#: Segment *entry* names the runner knows how to expand into stage lists (mirrors
#: ``scene-gen/run_grid_seg.py``'s STAGES table — several entries, e.g. ``rigid2_roi``, share
#: one stage impl with a prepended ROI stage). Validated statically to keep this module light.
KNOWN_SEGMENT_IMPLS = (
    "rigid", "mbs", "rigid2", "kabsch", "rigid2_roi", "mask_lift_oracle", "mask_lift",
    "multicut",
)


class SceneGenSpec(StrictModel):
    """Synthetic-scene generation (``gen_scenes.py`` params) when the experiment also authors
    its scenes. ``enabled: false`` (the common case) means the scenes/captures already exist."""

    enabled: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class CaptureSpec(StrictModel):
    """Where the per-cell captures live: a ``grid_manifest.json`` (scene-gen output) naming the
    cells and their capture dirs, plus the renders root fallback."""

    manifest: str
    cells: list[str]
    renders_root: str = ""
    reuse_existing: bool = True


class SweepSpec(StrictModel):
    """Gaussian-count sweep on one cell: derived capture dirs with subsampled init clouds
    (``<capture_dir>_g<target>``) and densification frozen."""

    cell: str
    targets: list[int]
    seed: int = 0


class PipelineSpec(StrictModel):
    preset: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class SegmentEntry(StrictModel):
    """One row of the segmentation impl matrix. ``impl`` is the *entry* name (see
    ``KNOWN_SEGMENT_IMPLS``); ``extract: true`` runs ``seg_extract.default`` first (exactly one
    entry should have it — the rest reuse its trajectories)."""

    impl: str
    preset: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    extract: bool = False


class EvalSpec(StrictModel):
    gt: str


class ExperimentSpec(StrictModel):
    name: str
    scene_gen: SceneGenSpec = Field(default_factory=SceneGenSpec)
    capture: CaptureSpec
    sweep: Optional[SweepSpec] = None
    pipeline: PipelineSpec
    stages: list[str] = Field(
        default_factory=lambda: ["convert.default", "train.default", "render.default"]
    )
    segment: list[SegmentEntry] = Field(default_factory=list)
    eval: Optional[EvalSpec] = None


def _spec_path(ref: str) -> Path:
    """Resolve ``ref`` as an explicit path when it looks like one, else as a name under
    ``experiments/``."""
    p = Path(ref)
    if p.suffix in (".yaml", ".yml") or p.parent != Path("."):
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"experiment spec not found: {p}")
        return p
    path = EXPERIMENTS_DIR / f"{ref}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown experiment {ref!r} (looked for {path}); "
            f"available: {list_experiments()}"
        )
    return path


def list_experiments() -> list[str]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: experiment spec must contain a mapping at the top level")
    return data


def _resolve_layers(path: Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """``extends:`` layering, mirroring ``pipeline.config.resolver._resolve_layers`` — parents
    resolve relative to the extending file's directory, then fall back to ``experiments/``."""
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ValueError(f"circular experiment `extends` chain: {chain}")
    data = _load_yaml(path)
    parents = data.pop("extends", None)

    merged: dict[str, Any] = {}
    if parents is not None:
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.suffix:
                parent_path = parent_path.with_suffix(".yaml")
            if not parent_path.is_absolute():
                candidate = path.parent / parent_path
                parent_path = candidate if candidate.is_file() else EXPERIMENTS_DIR / parent_path
            if not parent_path.is_file():
                raise FileNotFoundError(f"{path.name}: extends target not found: {parent!r}")
            merged = _deep_merge(merged, _resolve_layers(parent_path, _seen=(*_seen, path)))
    return _deep_merge(merged, data)


def load_grid_manifest(spec: ExperimentSpec) -> dict[str, Any]:
    """The spec's ``capture.manifest`` (grid_manifest.json), resolved against the repo root."""
    path = Path(spec.capture.manifest)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"experiment {spec.name!r}: grid manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec(ref: str) -> ExperimentSpec:
    """Load, layer, and validate an experiment spec by name (under ``experiments/``) or path."""
    path = _spec_path(ref)
    merged = _resolve_layers(path)
    try:
        spec = ExperimentSpec(**merged)
    except Exception as e:
        raise ValueError(f"invalid experiment spec {path}: {e}") from e

    known_presets = set(list_presets())
    if spec.pipeline.preset not in known_presets:
        raise ValueError(
            f"experiment {spec.name!r}: unknown pipeline preset {spec.pipeline.preset!r} "
            f"(available: {sorted(known_presets)})"
        )
    for entry in spec.segment:
        if entry.impl not in KNOWN_SEGMENT_IMPLS:
            raise ValueError(
                f"experiment {spec.name!r}: unknown segment impl {entry.impl!r} "
                f"(known: {list(KNOWN_SEGMENT_IMPLS)})"
            )
        if entry.preset not in known_presets:
            raise ValueError(
                f"experiment {spec.name!r}: segment entry {entry.impl!r} references unknown "
                f"preset {entry.preset!r}"
            )

    if spec.capture.cells:
        manifest = load_grid_manifest(spec)
        available = {c["cell"] for c in manifest.get("cells", [])}
        missing = [c for c in spec.capture.cells if c not in available]
        if missing:
            raise ValueError(
                f"experiment {spec.name!r}: cell(s) {missing} not in {spec.capture.manifest} "
                f"(available: {sorted(available)})"
            )
    if spec.sweep is not None and spec.sweep.cell not in spec.capture.cells:
        raise ValueError(
            f"experiment {spec.name!r}: sweep cell {spec.sweep.cell!r} is not in capture.cells "
            f"{spec.capture.cells}"
        )
    extract_entries = [e.impl for e in spec.segment if e.extract]
    if len(extract_entries) > 1:
        raise ValueError(
            f"experiment {spec.name!r}: at most one segment entry may set extract: true "
            f"(got {extract_entries})"
        )
    return spec
