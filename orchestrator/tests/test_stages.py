"""Tests for T04 (stage base class & registry).

Covers the task's acceptance criteria directly: two impls registered under one role are each
selectable by name (the same lookup a config's ``impl`` field would drive, T02's
``SegmentConfig``), duplicate registration and unknown role/impl lookups raise clear errors, and
the dummy ``EchoStage`` runs through ``run(ctx)`` with a hand-built context to produce a real,
valid ``Artifact``.
"""

from __future__ import annotations

import logging

import pytest


def test_two_impls_one_role_selectable_by_config(tmp_path):
    from pipeline.artifacts import Artifact
    from pipeline.stages import ResourceRequest, Stage, StageContext, get_stage_for_role, register

    # Registered under role "test" (like EchoStage's "test.echo") rather than a fresh made-up
    # role name: any *other* role name would leak into pipeline.api._auto_stage_plan for the rest
    # of the pytest session (the stage registry is global/process-wide and this test never
    # unregisters), which _auto_stage_plan (T05/T07) picks up for every real preset — role "test"
    # is the one name it deliberately excludes.
    @register("test.dummy_alpha")
    class AlphaStage(Stage):
        outputs = ("out",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            return {"out": Artifact(name="out", kind="json", path="a", producing_stage=ctx.stage_name)}

    @register("test.dummy_beta")
    class BetaStage(Stage):
        outputs = ("out",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            return {"out": Artifact(name="out", kind="json", path="b", producing_stage=ctx.stage_name)}

    # This is exactly the lookup a resolved SegmentConfig(impl="alpha"|"beta") would trigger.
    assert get_stage_for_role("test", "dummy_alpha") is AlphaStage
    assert get_stage_for_role("test", "dummy_beta") is BetaStage
    assert AlphaStage.name == "test.dummy_alpha"
    assert AlphaStage.role == "test"
    assert AlphaStage.impl == "dummy_alpha"
    assert isinstance(AlphaStage.resources, ResourceRequest)


def test_duplicate_registration_errors():
    from pipeline.stages import DuplicateStageError, Stage, register

    @register("test.dummy_dup")
    class First(Stage):
        def run(self, ctx):
            return {}

    with pytest.raises(DuplicateStageError):

        @register("test.dummy_dup")
        class Second(Stage):
            def run(self, ctx):
                return {}


def test_unknown_role_and_impl_lookup_errors_clearly():
    from pipeline.stages import StageNotFoundError, get_stage, get_stage_for_role, register, Stage

    with pytest.raises(StageNotFoundError, match="no stage registered"):
        get_stage("does.not_exist")

    with pytest.raises(StageNotFoundError, match="no stages registered for role"):
        get_stage_for_role("totally_unknown_role", "whatever")

    # Role "test" already has several impls registered by this point (echo, dummy_alpha, ...) —
    # still under role "test" (see the note above), just with one more impl added.
    @register("test.dummy_only_impl")
    class OnlyImpl(Stage):
        def run(self, ctx):
            return {}

    with pytest.raises(StageNotFoundError, match="no impl"):
        get_stage_for_role("test", "totally_missing_impl_name")


def test_register_rejects_malformed_names():
    from pipeline.stages import Stage, register

    with pytest.raises(ValueError):

        @register("no_dot_in_this_name")
        class Bad(Stage):
            def run(self, ctx):
                return {}


def test_register_rejects_non_stage_class():
    from pipeline.stages import register

    with pytest.raises(TypeError):

        @register("dummy_role3.not_a_stage")
        class NotAStage:
            pass


def test_echo_stage_runs_through_ctx_and_produces_valid_artifact(tmp_path):
    from pipeline.artifacts import Artifact
    from pipeline.stages import StageContext, get_stage

    echo_cls = get_stage("test.echo")
    assert echo_cls.name == "test.echo"
    assert echo_cls.role == "test"
    assert echo_cls.impl == "echo"

    ctx = StageContext(
        run_id="run001",
        stage_name="test.echo",
        config={"message": "hi from T04"},
        run_dir=tmp_path,
        logger=logging.getLogger("test.echo"),
    )

    stage = echo_cls()
    result = stage.run(ctx)

    assert set(result) == set(echo_cls.outputs) == {"echo"}
    artifact = result["echo"]
    assert isinstance(artifact, Artifact)
    assert artifact.kind == "json"
    assert artifact.producing_stage == "test.echo"

    written = tmp_path / "echo.json"
    assert written.is_file()
    assert written.read_text() == '{"message": "hi from T04"}'


def test_list_roles_and_list_stages_reflect_registrations():
    from pipeline.stages import get_stage, list_roles, list_stages

    # test.echo is always registered by importing pipeline.stages.
    assert "test.echo" in list_stages()
    roles = list_roles()
    assert "echo" in roles.get("test", [])
    assert get_stage("test.echo").role == "test"
