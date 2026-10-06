"""Round-2 correctness fixes (2026-10-05): focused tests for the new security/safety guards.

Covers the review's bugs 1.1/1.5/1.6/1.10 from ``reviews/orchestrator-correctness-review.md``:

- ``roi.impl: "none"`` must emit NO roi stage in the auto plan (previously emitted the
  unregistered ``roi.none``, breaking ``run_pipeline("base")`` with ``StageNotFoundError``).
- ``run_id``/preset-name path traversal rejection (``../../x`` must not escape the runs root or
  the presets dir).
- ``run_pipeline``'s ``external_artifacts`` seeding gate (absolute, existing, under a known root,
  file/dir shape matching the declared kind).
- ``run_pipeline(run_id=<existing>)`` refusing to clobber the old run's manifest.

Container-manager guards (stale-image rebuild, ``stop_by_id`` label check) live in
``tests/test_containers.py`` next to their fake-client fixtures; MCP serving confinement lives in
``tests/test_mcp_tools.py`` next to the seeded-run fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.artifacts import (
    Artifact,
    ArtifactPathError,
    InvalidRunIdError,
    create_run,
    manifest_path,
    run_dir,
    stage_log_path,
    validate_external_artifact_path,
    validate_run_id,
)
from pipeline.config.resolver import resolve_preset, validate_preset_name


# --- run_id validation (bug 1.6) --------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["base-1a2b3c4d", "seeded-run-0001", "t07-slice", "run_1.v2"])
def test_validate_run_id_accepts_normal_ids(run_id):
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    ["../../x", "..", "..\\x", "a/b", "a\\b", "", "with space", "x:evil", "..x..", ".", "..."],
)
def test_validate_run_id_rejects_traversal(run_id):
    with pytest.raises(InvalidRunIdError):
        validate_run_id(run_id)


def test_run_dir_and_derived_paths_reject_traversal(tmp_path):
    for bad in ("../../x", "..", "a/b"):
        with pytest.raises(InvalidRunIdError):
            run_dir(bad, runs_root=tmp_path)
        with pytest.raises(InvalidRunIdError):
            manifest_path(bad, runs_root=tmp_path)
        with pytest.raises(InvalidRunIdError):
            stage_log_path(bad, "train.default", runs_root=tmp_path)


def test_run_dir_stays_under_runs_root(tmp_path):
    d = run_dir("ok-run", runs_root=tmp_path)
    assert d.resolve().is_relative_to(tmp_path.resolve())


# --- preset-name validation (bug 1.6) ---------------------------------------------------------


@pytest.mark.parametrize("name", ["../../foo", "..", "a/b", "base.yaml/x", "with space"])
def test_resolve_preset_rejects_traversal_names(name):
    with pytest.raises(ValueError, match="invalid preset name"):
        resolve_preset(name)


def test_validate_preset_name_accepts_real_presets():
    for name in ("base", "pump01", "pump01_segB2", "cubes_kabsch"):
        assert validate_preset_name(name) == name


# --- roi.impl: "none" emits no stage (bug 1.1) --------------------------------------------------


def test_auto_stage_plan_skips_roi_when_impl_is_none():
    from pipeline.api import _auto_stage_plan
    from pipeline.config.models import PipelineConfig

    plan = _auto_stage_plan(PipelineConfig(name="t").model_dump())

    assert plan  # the base DAG has real stages
    assert not any(name.startswith("roi.") for name in plan)


def test_auto_stage_plan_includes_roi_when_an_impl_is_selected():
    from pipeline.api import _auto_stage_plan
    from pipeline.config.models import PipelineConfig

    cfg = PipelineConfig(name="t").model_dump()
    cfg["roi"]["impl"] = "motion_gate"

    plan = _auto_stage_plan(cfg)

    assert "roi.motion_gate" in plan


def test_run_pipeline_base_gets_past_stage_resolution(tmp_path, monkeypatch):
    """``run_pipeline("base")`` must no longer die with ``StageNotFoundError: roi.none`` — it now
    gets as far as the DAG's external-input check (``raw_mesh`` was never seeded), which is the
    correct, documented failure for this preset without ``external_artifacts``.
    """
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))

    from pipeline.api import run_pipeline
    from pipeline.dag import MissingDependencyError

    with pytest.raises(MissingDependencyError):
        run_pipeline("base")


# --- external artifact seeding gate (bug 1.5) ---------------------------------------------------


def test_validate_external_artifact_path_accepts_a_real_file_under_assets_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path))
    f = tmp_path / "assets" / "mesh.usd"
    f.parent.mkdir(parents=True)
    f.write_text("placeholder", encoding="utf-8")

    assert validate_external_artifact_path(str(f), "usd") == f.resolve()


def test_validate_external_artifact_path_rejects_relative_path():
    with pytest.raises(ArtifactPathError, match="absolute"):
        validate_external_artifact_path("relative/mesh.usd", "usd")


def test_validate_external_artifact_path_rejects_nonexistent(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path))
    with pytest.raises(ArtifactPathError, match="existing file"):
        validate_external_artifact_path(str(tmp_path / "nope.usd"), "usd")


def test_validate_external_artifact_path_rejects_paths_outside_known_roots(tmp_path):
    """tmp_path is under neither the repo root, the assets root (default ``Q:/Omniverse``), nor
    the runs root here — exactly the "any host path" case the gate exists to stop."""
    f = tmp_path / "secret.txt"
    f.write_text("x", encoding="utf-8")

    with pytest.raises(ArtifactPathError, match="outside every allowed root"):
        validate_external_artifact_path(str(f), "json")


def test_validate_external_artifact_path_enforces_dir_vs_file_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path))
    f = tmp_path / "mesh.usd"
    f.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ArtifactPathError, match="directory"):
        validate_external_artifact_path(str(f), "dataset")  # file where a dir is declared


def test_run_pipeline_rejects_unconfined_external_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))

    from pipeline.api import run_pipeline

    outside = tmp_path / "CONJUNTO_BOMBAS.usd"  # outside every allowed root (no assets override)
    outside.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ArtifactPathError):
        run_pipeline(
            "base",
            external_artifacts={
                "raw_mesh": Artifact(
                    name="raw_mesh", kind="usd", path=str(outside), producing_stage="external"
                )
            },
        )
    # and the failed seeding must not have created a run directory at all
    assert not (tmp_path / "runs").exists()


def test_run_pipeline_accepts_dict_shaped_external_artifacts(tmp_path, monkeypatch):
    """MCP clients hand ``external_artifacts`` over JSON — dicts, not ``Artifact`` objects — so
    the seeding gate coerces and validates those too."""
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("PIPELINE_ASSETS_ROOT", str(tmp_path))
    mesh = tmp_path / "mesh.usd"
    mesh.write_text("placeholder", encoding="utf-8")

    from pipeline.api import run_pipeline
    from pipeline.dag import MissingDependencyError

    with pytest.raises(MissingDependencyError):  # past seeding; gt_segmentation still missing
        run_pipeline(
            "base",
            external_artifacts={
                "raw_mesh": {
                    "name": "raw_mesh",
                    "kind": "usd",
                    "path": str(mesh),
                    "producing_stage": "external",
                }
            },
        )


# --- run_id clobber guard (bug 1.10) ------------------------------------------------------------


def test_run_pipeline_refuses_an_existing_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))
    create_run("existing-run", "base", {"name": "base"}, stage_names=[])

    from pipeline.api import run_pipeline

    with pytest.raises(FileExistsError, match="existing-run"):
        run_pipeline("base", run_id="existing-run")

    # the old manifest must be untouched (still the hand-created one, not an all-pending reset)
    from pipeline.artifacts import get_manifest

    assert get_manifest("existing-run").resolved_config == {"name": "base"}


def test_run_pipeline_rejects_invalid_run_id_before_creating_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path / "runs"))

    from pipeline.api import run_pipeline

    with pytest.raises(InvalidRunIdError):
        run_pipeline("base", run_id="../../escape")
    assert not (tmp_path / "escape").exists()
