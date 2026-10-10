"""T27 Phase A: ``RunManifest.metadata`` — free-form, caller-attached run metadata.

Covers: round-trip through ``create_run``/``load_manifest``, ``run_pipeline(metadata=...)``
landing in the manifest, and back-compat — a manifest.json written before the field existed
(no ``metadata`` key) must still deserialize (``StrictModel`` is ``extra="forbid"``, so this
pins that the field has a default).
"""

from __future__ import annotations

import json


def test_create_run_metadata_roundtrips(tmp_path):
    from pipeline.artifacts import create_run, load_manifest

    meta = {"experiment": "pump_grid_v2", "cell": "pump_A20mm_M2", "variant": None,
            "capture": {"base_amp_mm": 20, "multiplier": 2}}
    create_run("meta-r1", "t27", {"a": 1}, stage_names=[], runs_root=tmp_path, metadata=meta)

    manifest = load_manifest("meta-r1", runs_root=tmp_path)
    assert manifest.metadata == meta
    # Serialized to disk too (not just the returned object).
    raw = json.loads((tmp_path / "meta-r1" / "manifest.json").read_text(encoding="utf-8"))
    assert raw["metadata"] == meta


def test_create_run_without_metadata_defaults_to_empty_dict(tmp_path):
    from pipeline.artifacts import create_run, load_manifest

    create_run("meta-r2", "t27", {}, runs_root=tmp_path)
    assert load_manifest("meta-r2", runs_root=tmp_path).metadata == {}


def test_old_manifest_without_metadata_field_deserializes(tmp_path):
    """A manifest.json written before T27 has no `metadata` key at all."""
    from pipeline.artifacts import create_run, load_manifest, manifest_path

    create_run("meta-old", "t27", {"a": 1}, stage_names=[], runs_root=tmp_path)
    path = manifest_path("meta-old", runs_root=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["metadata"]  # simulate the pre-T27 on-disk shape
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    manifest = load_manifest("meta-old", runs_root=tmp_path)
    assert manifest.metadata == {}
    assert manifest.resolved_config == {"a": 1}


def test_run_pipeline_metadata_lands_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api
    from pipeline.artifacts import get_manifest

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    run_id = api.run_pipeline(
        "base",
        metadata={"experiment": "pump_grid_v2", "cell": "pump_A40mm_M8"},
    )
    assert get_manifest(run_id).metadata == {
        "experiment": "pump_grid_v2",
        "cell": "pump_A40mm_M8",
    }
