"""Typed artifact records and the per-run manifest schema.

An ``Artifact`` describes one typed output a stage produced (a dataset dir, a trained model
checkpoint, an ``.npz``, a point cloud, a preview image/video, a JSON summary, ...) without
copying it: artifacts reference paths that already exist under the pipeline's canonical output
conventions (``output/multipleview/<name>/``, ``motion_seg``'s ``trajectories.npz`` /
``segmentation.npz`` / ``*_preview.png``, ...) plus a content hash for identity/caching (T05 uses
this; see ``pipeline/artifacts/hashing.py``).

``RunManifest`` is the per-run ``manifest.json`` — the single read surface Layers 2/3 use
(``planning/ARCHITECTURE.md``): resolved config, git SHA, per-stage status/timing/logs/artifacts,
overall run status. See ``pipeline/artifacts/manifest.py`` for how it's built/read/written and
``planning/tasks/T03-artifacts-and-manifest.md`` for scope.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: dataset|model|npz|ply|png|video|json per the T03 task spec, plus ``usd`` (T11: a single USD
#: mesh file — ``prep_split.default``/``prep_motion.default``'s ``segmented_mesh``/
#: ``animated_mesh`` outputs — didn't fit any of the seven original kinds, none of which is a
#: bare single-file mesh format).
ArtifactKind = Literal["dataset", "model", "npz", "ply", "png", "video", "json", "usd"]

StageState = Literal["pending", "running", "success", "failed", "skipped"]

RunState = Literal["pending", "running", "success", "failed", "cancelled"]


class StrictModel(BaseModel):
    """Base for every artifacts/manifest model: unknown fields are a hard error.

    Mirrors ``pipeline.config.models.StrictModel`` (duplicated rather than imported — this
    module must stay independent of ``pipeline.config``; a manifest records the *resolved*
    config as a plain ``dict``, not the pydantic config type, so this package stays a leaf
    dependency other modules can import without pulling in the config schema).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Artifact(StrictModel):
    """One typed output produced by a stage.

    ``path`` is the real path to the existing output (a directory for ``dataset``/``model``
    kinds, a file otherwise) — artifacts reference, they never copy ("don't duplicate large
    artifacts" per the task's notes/gotchas). ``content_hash``/``hash_algo`` are ``None`` until
    computed (:func:`pipeline.artifacts.hashing.hash_path`); a ``dataset`` artifact that's a
    whole directory tree has no single-file hash, and may legitimately stay ``None`` — that's
    the caller's call, not enforced here.
    """

    name: str
    kind: ArtifactKind
    path: str
    producing_stage: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None
    hash_algo: Optional[str] = None


class StageRecord(StrictModel):
    """Per-stage entry inside a run manifest.

    ``peak_vram_mb``/``peak_ram_mb`` are nullable on purpose — out of scope for T03 ("how peak-
    mem is measured" is T12's job); every stage record leaves them ``None`` until T12 lands.

    ``cache_key`` is nullable for the same reason (T03 predates the scheduler): T05's DAG
    scheduler sets it on every ``success``/``skipped`` result — hash(resolved stage config +
    input-artifact content hashes + code version) — and compares it against a stage's *previous*
    ``cache_key`` (in this run's manifest, or T05's cross-run cache index) to decide whether a
    later call can skip re-running it. ``None`` for stages a pre-T05 manifest recorded, or for a
    stage that hasn't run yet.
    """

    status: StageState = "pending"
    start_time: Optional[str] = None  # ISO 8601, UTC
    end_time: Optional[str] = None
    wall_time_s: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    log_path: Optional[str] = None
    #: keys into RunManifest.artifacts for what this stage produced.
    artifacts: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    cache_key: Optional[str] = None


class RunManifest(StrictModel):
    """``manifest.json`` — the single read surface for Layers 2/3.

    ``resolved_config`` is a plain ``dict`` (``PipelineConfig.model_dump()``, T02's schema) so
    this module never imports ``pipeline.config`` and stays a leaf dependency.
    """

    run_id: str
    preset: str
    resolved_config: dict[str, Any]
    git_sha: Optional[str] = None
    created_at: str  # ISO 8601, UTC
    updated_at: str  # ISO 8601, UTC
    status: RunState = "pending"
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
