"""Tests for T05 (DAG scheduler & caching).

Covers the task's acceptance criteria directly on a toy 3-stage graph
(``test.t05a`` -(x)-> ``test.t05b`` -(y)-> ``test.t05c``): a first run executes every stage, a
second run — even under a *different* run_id — skips all of them via the cross-run cache,
editing one stage's config invalidates exactly that stage and its descendants, ``from_stage``/
``to_stage``/``only``/``force`` behave as specified, and cycle/missing-dependency detection raises
before anything is touched. Also covers ``pipeline.api``'s ``run_pipeline``/``run_stage`` wiring.

The toy stages are registered under role ``"test"`` (like ``pipeline.stages.echo.EchoStage``)
specifically so ``pipeline.api._auto_stage_plan`` — which drives ``run_pipeline`` and explicitly
skips role ``"test"`` — never picks them up. Registering them under a real-looking role would leak
into every other test in this session that calls ``api.run_pipeline`` (the registry is global and
process-wide), which is exactly the failure mode this sidesteps.
"""

from __future__ import annotations

import json

import pytest

#: call counts for the toy stages, reset before every test by the autouse fixture below.
CALLS = {"t05a": 0, "t05b": 0, "t05c": 0}


def _register_toy_chain() -> None:
    from pipeline.artifacts import Artifact
    from pipeline.stages import Stage, StageContext, list_stages, register

    if "test.t05a" in list_stages():
        return  # already registered by an earlier test in this module

    @register("test.t05a")
    class T05AStage(Stage):
        outputs = ("x",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["t05a"] += 1
            out = ctx.run_dir / "x.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"n": ctx.config.get("n", 1)}))
            return {"x": Artifact(name="x", kind="json", path=str(out), producing_stage=ctx.stage_name)}

    @register("test.t05b")
    class T05BStage(Stage):
        inputs = ("x",)
        outputs = ("y",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["t05b"] += 1
            out = ctx.run_dir / "y.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"n": ctx.config.get("n", 1)}))
            return {"y": Artifact(name="y", kind="json", path=str(out), producing_stage=ctx.stage_name)}

    @register("test.t05c")
    class T05CStage(Stage):
        inputs = ("y",)
        outputs = ("z",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["t05c"] += 1
            out = ctx.run_dir / "z.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"n": ctx.config.get("n", 1)}))
            return {"z": Artifact(name="z", kind="json", path=str(out), producing_stage=ctx.stage_name)}


@pytest.fixture(autouse=True)
def _toy_chain():
    _register_toy_chain()
    for k in CALLS:
        CALLS[k] = 0
    yield


_NAMES = ["test.t05a", "test.t05b", "test.t05c"]


def _resolved(n_a: int = 1, n_b: int = 1, n_c: int = 1) -> dict:
    return {"a": {"n": n_a}, "b": {"n": n_b}, "c": {"n": n_c}}


def _stage_configs(resolved: dict) -> dict:
    return {"test.t05a": resolved["a"], "test.t05b": resolved["b"], "test.t05c": resolved["c"]}


# --- graph.py: pure structure ------------------------------------------------------------------


def test_resolve_nodes_and_topo_sort_orders_the_chain():
    from pipeline.dag import external_inputs, resolve_nodes, topo_sort

    nodes = resolve_nodes(_NAMES)
    assert topo_sort(nodes) == ["test.t05a", "test.t05b", "test.t05c"]
    assert external_inputs(nodes) == {}


def test_external_inputs_flags_input_with_no_in_set_producer():
    from pipeline.dag import external_inputs, resolve_nodes

    nodes = resolve_nodes(["test.t05b"])
    assert external_inputs(nodes) == {"x": {"test.t05b"}}


def test_topo_sort_raises_cycle_error_for_a_cyclic_pair():
    from pipeline.artifacts import Artifact
    from pipeline.dag import CycleError, resolve_nodes, topo_sort
    from pipeline.stages import Stage, list_stages, register

    if "test.t05p" not in list_stages():
        @register("test.t05p")
        class PStage(Stage):
            inputs = ("q",)
            outputs = ("p",)

            def run(self, ctx):
                return {"p": Artifact(name="p", kind="json", path="p", producing_stage=ctx.stage_name)}

        @register("test.t05q")
        class QStage(Stage):
            inputs = ("p",)
            outputs = ("q",)

            def run(self, ctx):
                return {"q": Artifact(name="q", kind="json", path="q", producing_stage=ctx.stage_name)}

    nodes = resolve_nodes(["test.t05p", "test.t05q"])
    with pytest.raises(CycleError):
        topo_sort(nodes)


# --- cache.py -----------------------------------------------------------------------------------


def test_compute_cache_key_changes_with_config_inputs_and_code_version():
    from pipeline.dag import compute_cache_key
    from pipeline.stages import get_stage

    cls = get_stage("test.t05b")
    base = compute_cache_key(cls, {"n": 1}, {"x": "hash1"}, "deadbeef")
    assert compute_cache_key(cls, {"n": 1}, {"x": "hash1"}, "deadbeef") == base  # deterministic
    assert compute_cache_key(cls, {"n": 2}, {"x": "hash1"}, "deadbeef") != base  # config differs
    assert compute_cache_key(cls, {"n": 1}, {"x": "hash2"}, "deadbeef") != base  # input differs
    assert compute_cache_key(cls, {"n": 1}, {"x": "hash1"}, "cafebabe") != base  # git sha differs


def test_stage_source_hash_differs_across_files_stable_within_one():
    from pipeline.dag import stage_source_hash
    from pipeline.stages import get_stage

    echo_hash = stage_source_hash(get_stage("test.echo"))
    toy_hash = stage_source_hash(get_stage("test.t05a"))
    assert echo_hash != toy_hash  # defined in different files
    assert stage_source_hash(get_stage("test.echo")) == echo_hash  # deterministic


# --- scheduler.py: the toy-graph acceptance criteria ---------------------------------------------


def test_first_run_executes_all_second_run_skips_all_via_cross_run_cache(tmp_path):
    from pipeline.dag import run_dag

    resolved = _resolved()
    cfgs = _stage_configs(resolved)

    m1 = run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert m1.status == "success"
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}
    assert [m1.stages[n].status for n in _NAMES] == ["success", "success", "success"]

    # a different run_id, same config -> cross-run cache hit, nothing actually re-executes.
    m2 = run_dag("run2", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}
    assert [m2.stages[n].status for n in _NAMES] == ["skipped", "skipped", "skipped"]
    assert m2.status == "success"

    # re-running the *same* run_id is also a no-op (cheap same-run path, no index lookup needed).
    m1b = run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}
    assert [m1b.stages[n].status for n in _NAMES] == ["success", "success", "success"]


def test_editing_a_stage_config_invalidates_it_and_its_descendants(tmp_path):
    from pipeline.dag import run_dag

    resolved = _resolved()
    cfgs = _stage_configs(resolved)
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}

    cfgs2 = dict(cfgs)
    cfgs2["test.t05b"] = {"n": 2}  # only b's own config changes
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs2, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 2, "t05c": 2}  # a untouched; b reran; c reran (new input)


def test_only_restricts_execution_to_the_named_subset(tmp_path):
    from pipeline.dag import run_dag

    resolved = _resolved()
    cfgs = _stage_configs(resolved)
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}

    # only b, forced -> only b re-executes even though nothing about it actually changed.
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, only=["test.t05b"], force=True, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 2, "t05c": 1}


def test_from_stage_and_to_stage_window_the_execution(tmp_path):
    from pipeline.dag import run_dag

    resolved = _resolved()
    cfgs = _stage_configs(resolved)
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS == {"t05a": 1, "t05b": 1, "t05c": 1}

    # from_stage == to_stage == b, forced -> exactly b.
    run_dag(
        "run1", _NAMES, resolved, preset="toy", stage_configs=cfgs,
        from_stage="test.t05b", to_stage="test.t05b", force=True, runs_root=tmp_path,
    )
    assert CALLS == {"t05a": 1, "t05b": 2, "t05c": 1}

    # from_stage == b, no to_stage -> b and c, forced.
    run_dag(
        "run1", _NAMES, resolved, preset="toy", stage_configs=cfgs,
        from_stage="test.t05b", force=True, runs_root=tmp_path,
    )
    assert CALLS == {"t05a": 1, "t05b": 3, "t05c": 2}


def test_only_rejects_a_name_outside_stage_names(tmp_path):
    from pipeline.dag import run_dag

    with pytest.raises(ValueError):
        run_dag("run1", _NAMES, _resolved(), preset="toy", only=["not.a_stage"], runs_root=tmp_path)


def test_missing_dependency_detected_before_running_anything(tmp_path):
    from pipeline.dag import MissingDependencyError, run_dag

    with pytest.raises(MissingDependencyError):
        run_dag("runX", ["test.t05b"], {"b": {}}, preset="toy", runs_root=tmp_path)
    assert CALLS["t05b"] == 0  # never touched — the check runs before any execution


def test_missing_dependency_satisfied_by_a_resumed_runs_existing_artifacts(tmp_path):
    """`only=["test.t05b"]` alone would normally be a missing-dependency error (nothing in the
    given stage_names produces "x") — but it's fine once a *previous* call already produced and
    recorded "x" for this same run_id."""
    from pipeline.dag import run_dag

    resolved = _resolved()
    cfgs = _stage_configs(resolved)
    run_dag("run1", _NAMES, resolved, preset="toy", stage_configs=cfgs, runs_root=tmp_path)

    # a lone-node graph naming just b — its input "x" is satisfiable from run1's own manifest.
    m = run_dag("run1", ["test.t05b"], resolved, preset="toy", stage_configs=cfgs, force=True, runs_root=tmp_path)
    assert m.stages["test.t05b"].status == "success"
    assert CALLS == {"t05a": 1, "t05b": 2, "t05c": 1}


def test_failed_stage_stops_scheduling_descendants_stay_pending(tmp_path):
    from pipeline.artifacts import Artifact
    from pipeline.dag import run_dag
    from pipeline.stages import Stage, list_stages, register

    if "test.t05fail" not in list_stages():
        @register("test.t05fail")
        class FailStage(Stage):
            outputs = ("w",)

            def run(self, ctx):
                raise RuntimeError("boom")

        @register("test.t05after_fail")
        class AfterFailStage(Stage):
            inputs = ("w",)
            outputs = ("v",)

            def run(self, ctx):
                return {"v": Artifact(name="v", kind="json", path="v", producing_stage=ctx.stage_name)}

    m = run_dag(
        "run_fail", ["test.t05fail", "test.t05after_fail"], {}, preset="toy", runs_root=tmp_path
    )
    assert m.status == "failed"
    assert m.stages["test.t05fail"].status == "failed"
    assert m.stages["test.t05fail"].error == "boom"
    assert m.stages["test.t05after_fail"].status == "pending"  # never attempted


# --- pipeline.api wiring ---------------------------------------------------------------------


def test_api_run_pipeline_wiring_produces_a_trivially_successful_empty_run(tmp_path, monkeypatch):
    """`pipeline.api.run_pipeline` (T05) now delegates to `pipeline.dag.run_dag`, the same way
    T02/T03 wired list_presets/validate_config/list_runs/get_status.

    At T05-time no real (non-`test`) role was registered, so `_auto_stage_plan` naturally
    resolved to an empty plan for every preset. T07 registers the first real stages
    (`convert.default`/`segment.rigid`/`seg_eval.default`), which `_auto_stage_plan` now picks up
    for *every* preset (their inputs are all external in Phase 0, so a preset with nothing
    pre-seeded would raise `MissingDependencyError`, not succeed trivially) — and every later task
    that registers more real stages would keep breaking this same assumption. Monkeypatching
    `_auto_stage_plan` back to `[]` decouples this test from whatever stages happen to be
    globally registered, so it keeps testing what it always meant to: the run_pipeline -> run_dag
    -> manifest wiring itself, not the stage registry's current contents.
    """
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    from pipeline import api

    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])

    run_id = api.run_pipeline("base")
    status = api.get_status(run_id)
    assert status["preset"] == "base"
    assert status["status"] == "success"
    assert status["stages"] == {}


def test_api_run_stage_wiring_runs_a_single_registered_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    from pipeline import api

    # See the previous test's docstring: T07+ register real stages that `_auto_stage_plan` would
    # otherwise pick up for "base" too, which isn't what this test is about.
    monkeypatch.setattr(api, "_auto_stage_plan", lambda resolved: [])

    run_id = api.run_pipeline("base")  # empty run, but creates the manifest run_stage needs
    returned = api.run_stage(run_id, "test.echo")
    assert returned == run_id

    status = api.get_status(run_id)
    assert status["stages"]["test.echo"]["status"] == "success"

    # "test.t05b" needs "x", never produced in this run -> a clear failure, not a silent no-op.
    with pytest.raises(Exception):
        api.run_stage(run_id, "test.t05b")


def test_api_run_stage_missing_run_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    from pipeline import api

    with pytest.raises(FileNotFoundError):
        api.run_stage("does-not-exist", "test.echo")
