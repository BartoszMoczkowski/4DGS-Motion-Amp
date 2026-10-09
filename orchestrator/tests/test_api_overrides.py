"""T26 Phase A: `overrides` deep-merge on `pipeline.api.run_pipeline` / `run_stage`.

Overrides are merged onto the resolved preset config *before* `PipelineConfig` re-validation
(so `extra="forbid"` still catches misspelled keys) and before `_auto_stage_plan` /
`_stage_config_for` see the config. GPU/Docker stages are never executed here: the stage plan
is monkeypatched to `[]` (or `run_dag` is faked to capture what it would have been given),
following `test_dag.py`'s own decoupling pattern.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _empty_plan(monkeypatch):
    from pipeline import api

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])
    return api


def test_run_pipeline_overrides_reach_manifest_and_stage_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api
    from pipeline.artifacts import get_manifest

    captured: dict = {}

    def fake_run_dag(run_id, stage_names, resolved, **kwargs):
        captured["stage_configs"] = kwargs.get("stage_configs", {})

        # Mimic run_dag's manifest write contract minimally: mark nothing, return manifest.
        return get_manifest(run_id)

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: ["segment.rigid"])
    monkeypatch.setattr("pipeline.dag.run_dag", fake_run_dag)

    run_id = api.run_pipeline(
        "base",
        overrides={
            "train": {"port": 6010},
            "segment": {"rigid": {"k": 7}},
        },
    )

    manifest = get_manifest(run_id)
    # Deep-merge: overridden values land, untouched values keep preset/model defaults.
    assert manifest.resolved_config["train"]["port"] == 6010
    assert manifest.resolved_config["segment"]["rigid"]["k"] == 7
    assert manifest.resolved_config["segment"]["impl"] == "rigid"
    assert manifest.resolved_config["train"]["ip"] == "127.0.0.1"

    # And the override reached `_stage_config_for`'s per-stage slice (segment.rigid's section).
    assert captured["stage_configs"]["segment.rigid"]["k"] == 7
    assert "threshold_mult" in captured["stage_configs"]["segment.rigid"]


def test_run_pipeline_invalid_override_key_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    api = _empty_plan(monkeypatch)

    with pytest.raises(ValidationError) as excinfo:
        api.run_pipeline("base", overrides={"train": {"prot": 6010}})
    assert "prot" in str(excinfo.value)

    # Unknown top-level section is rejected too, not silently added.
    with pytest.raises(ValidationError):
        api.run_pipeline("base", overrides={"trane": {"port": 6010}})


def test_run_pipeline_overrides_can_flip_stage_plan(tmp_path, monkeypatch):
    """Overrides merge BEFORE `_auto_stage_plan`, so e.g. a segment.impl override changes the
    planned stage list."""
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api

    seen: dict = {}

    def fake_plan(resolved):
        seen["resolved"] = resolved
        return []

    monkeypatch.setattr(api, "_auto_stage_plan", fake_plan)
    api.run_pipeline("base", overrides={"seg_extract": {"n_times": 12}})
    assert seen["resolved"]["seg_extract"]["n_times"] == 12


def test_run_stage_overrides_apply_without_rewriting_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    from pipeline import api
    from pipeline.artifacts import get_manifest

    api = _empty_plan(monkeypatch)
    run_id = api.run_pipeline("base")

    captured: dict = {}

    def fake_run_dag(rid, stage_names, resolved, **kwargs):
        captured["resolved"] = resolved
        captured["stage_configs"] = kwargs.get("stage_configs", {})
        return get_manifest(rid)

    monkeypatch.setattr("pipeline.dag.run_dag", fake_run_dag)

    # test.echo has no "test" config section, so its slice is the whole resolved config.
    api.run_stage(run_id, "test.echo", overrides={"seg_extract": {"n_times": 5}})
    assert captured["resolved"]["seg_extract"]["n_times"] == 5
    assert captured["stage_configs"]["test.echo"]["seg_extract"]["n_times"] == 5

    # The recorded config is NOT rewritten: a later plain run_stage sees the original value.
    api.run_stage(run_id, "test.echo")
    assert captured["resolved"]["seg_extract"]["n_times"] != 5
    assert get_manifest(run_id).resolved_config["seg_extract"]["n_times"] != 5


def test_run_stage_invalid_override_key_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    api = _empty_plan(monkeypatch)
    run_id = api.run_pipeline("base")

    with pytest.raises(ValidationError) as excinfo:
        api.run_stage(run_id, "test.echo", overrides={"amp": {"chanel": {}}})
    assert "chanel" in str(excinfo.value)


def test_no_overrides_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))
    api = _empty_plan(monkeypatch)
    assert api._apply_overrides({"a": 1}, None) == {"a": 1}
    assert api._apply_overrides({"a": 1}, {}) == {"a": 1}
