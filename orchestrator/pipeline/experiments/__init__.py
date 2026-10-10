"""Declarative experiment specs + runner (T27).

Lazy-import safe: :mod:`pipeline.experiments.spec` imports only pydantic/yaml and the (light)
config resolver; ``runner``/``variants``/``results`` pull numpy and the vendored host code but
are only imported from CLI command bodies and tests — never at CLI module scope.
"""

from __future__ import annotations

from .spec import ExperimentSpec, list_experiments, load_spec

__all__ = ["ExperimentSpec", "list_experiments", "load_spec"]
