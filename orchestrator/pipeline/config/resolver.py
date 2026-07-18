"""Layered preset resolution: ``base <- scene <- experiment``.

Each preset is a YAML file under ``pipeline/config/presets/``. A preset optionally names one or
more parents via a top-level ``extends:`` key (a string or list of strings); parents are resolved
recursively and deep-merged in order, then the preset's own keys are merged on top. The result is
handed to :class:`pipeline.config.models.PipelineConfig` for validation.

Only pure dict/YAML logic lives here — no filesystem side effects beyond reading preset files, no
CUDA/torch/docker imports, so this module stays importable on the CPU-only host/sandbox.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .models import PipelineConfig

PRESETS_DIR = Path(__file__).parent / "presets"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; non-dict values replace)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _preset_path(name: str) -> Path:
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"unknown preset {name!r} (looked for {path}); "
            f"available presets: {list_presets()}"
        )
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: preset file must contain a mapping at the top level")
    return data


def _resolve_layers(name: str, *, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    if name in _seen:
        chain = " -> ".join((*_seen, name))
        raise ValueError(f"circular preset `extends` chain: {chain}")

    data = _load_yaml(_preset_path(name))
    parents = data.pop("extends", None)

    merged: dict[str, Any] = {}
    if parents is not None:
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents:
            merged = _deep_merge(merged, _resolve_layers(parent, _seen=(*_seen, name)))

    return _deep_merge(merged, data)


def list_presets() -> list[str]:
    """Names of every preset file under ``presets/`` (without the ``.yaml`` extension)."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def resolve_preset(name: str) -> dict[str, Any]:
    """Merge a preset's ``extends`` chain into one plain dict. Does not validate the result."""
    return _resolve_layers(name)


def validate_config(name: str) -> PipelineConfig:
    """Resolve ``name``'s preset chain and validate it. Raises on any invalid/unknown setting."""
    merged = resolve_preset(name)
    merged.setdefault("name", name)
    return PipelineConfig(**merged)
