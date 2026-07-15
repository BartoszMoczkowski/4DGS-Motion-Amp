"""Typed artifact records and the per-run manifest (manifest.json).

The single read surface Layers 2/3 (MCP server, UI) use to discover runs/artifacts/status
(``planning/ARCHITECTURE.md``). See ``planning/tasks/T03-artifacts-and-manifest.md`` for scope.
Deliberately import-light: no torch/CUDA/docker/pynvml anywhere in this package.
"""

from __future__ import annotations

from .hashing import FAST_ALGO, FULL_ALGO, hash_path
from .manifest import (
    ManifestCorruptError,
    ManifestError,
    create_run,
    get_git_sha,
    load_manifest,
    new_manifest,
    record_stage_result,
    record_stage_start,
    save_manifest,
    update_manifest,
)
from .models import Artifact, ArtifactKind, RunManifest, RunState, StageRecord, StageState
from .paths import (
    DEFAULT_RUNS_ROOT,
    config_snapshot_path,
    ensure_run_dirs,
    get_runs_root,
    log_dir,
    manifest_path,
    run_dir,
    stage_log_path,
)
from .store import ArtifactNotFoundError, get_artifact, get_manifest, list_artifacts, list_runs

__all__ = [
    # models
    "Artifact",
    "ArtifactKind",
    "RunManifest",
    "RunState",
    "StageRecord",
    "StageState",
    # hashing
    "hash_path",
    "FAST_ALGO",
    "FULL_ALGO",
    # paths
    "DEFAULT_RUNS_ROOT",
    "get_runs_root",
    "run_dir",
    "manifest_path",
    "config_snapshot_path",
    "log_dir",
    "stage_log_path",
    "ensure_run_dirs",
    # manifest read/write
    "ManifestError",
    "ManifestCorruptError",
    "new_manifest",
    "create_run",
    "save_manifest",
    "load_manifest",
    "update_manifest",
    "record_stage_start",
    "record_stage_result",
    "get_git_sha",
    # store / query helpers
    "list_runs",
    "get_manifest",
    "list_artifacts",
    "get_artifact",
    "ArtifactNotFoundError",
]
