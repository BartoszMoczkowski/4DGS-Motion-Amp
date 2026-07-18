"""Migration helpers: read the *old* scattered config sources and shape them like the new schema.

These are not on the runtime hot path — stages (T04+) read resolved ``PipelineConfig`` objects,
not these legacy files directly. They exist so:

1. The ``pump01`` preset can be checked by test (:mod:`tests.test_config`) against the actual
   legacy files it was migrated from, instead of trusting a hand-typed copy.
2. Migrating the *next* scene's legacy config is a re-run of these helpers, not a repeat of the
   manual field-by-field port documented in ``MIGRATION.md``.

``load_legacy_hyperparams`` executes the target ``.py`` file (via :func:`runpy.run_path`) to read
its ``ModelHiddenParams``/``OptimizationParams`` dict literals — the same trust model 4DGS's own
``mmengine.Config.fromfile`` already uses for these files (they are plain dict-literal modules
checked into this repo, not user-supplied input).
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import yaml

from .resolver import _deep_merge


def load_legacy_capture_yaml(path: str | Path) -> dict[str, Any]:
    """Load an ``omni_capture.py``-style ``capture_config*.yaml`` into ``{"capture": {...}}``.

    The old and new key layouts are identical for the ``scene``/``rig``/``capture``/``output``/
    ``lighting`` sections; only ``app.headless`` is flattened to ``capture.headless`` (it was the
    only key ever nested under ``app``).
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    capture: dict[str, Any] = {}
    app = raw.get("app") or {}
    if "headless" in app:
        capture["headless"] = app["headless"]
    for section in ("scene", "rig", "capture", "output", "lighting"):
        if section in raw:
            capture[section] = raw[section]
    return {"capture": capture}


def load_legacy_hyperparams(path: str | Path) -> dict[str, Any]:
    """Load an ``arguments/<dataset>/<scene>.py``-style file's ``ModelHiddenParams`` /
    ``OptimizationParams`` dict literals into ``{"hidden": {...}, "optim": {...}}``.
    """
    namespace = runpy.run_path(str(path))

    out: dict[str, Any] = {}
    if "ModelHiddenParams" in namespace:
        out["hidden"] = dict(namespace["ModelHiddenParams"])
    if "OptimizationParams" in namespace:
        out["optim"] = dict(namespace["OptimizationParams"])
    return out


def merge_legacy_sources(*fragments: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge migration fragments (e.g. capture yaml + hyperparams py) into one preset dict."""
    merged: dict[str, Any] = {}
    for fragment in fragments:
        merged = _deep_merge(merged, fragment)
    return merged
