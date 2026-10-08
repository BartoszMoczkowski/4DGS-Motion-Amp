"""4DGS motion amplification — workspace root package body.

Placeholder for the T25 `amp` CLI; re-exports the project version only.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("4dgs-motion-amp")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.1.0"
