"""Trivial import test — the T01 acceptance bar for the pipeline package."""

import sys


def test_pipeline_imports():
    import pipeline

    assert pipeline.__version__


def test_submodule_stubs_import():
    from pipeline import artifacts, config, containers, dag, resources, stages

    assert all(m is not None for m in (config, stages, dag, artifacts, containers, resources))


def test_api_stubs_raise_not_implemented():
    """``run_pipeline``/``run_stage`` were wired to a real scheduler in T05 (see
    ``tests/test_dag.py``); ``cancel``/``gpu_status``/container controls remain T08/T12 stubs."""
    from pipeline import api

    import pytest

    with pytest.raises(NotImplementedError):
        api.cancel("run001")
    with pytest.raises(NotImplementedError):
        api.gpu_status()


def test_no_heavy_imports_at_module_scope():
    """No torch/CUDA/docker/pynvml should be pulled in just by importing pipeline."""
    banned = {"torch", "torchvision", "docker", "pynvml"}
    before = set(sys.modules)

    import pipeline  # noqa: F401

    after = set(sys.modules)
    newly_imported = after - before
    assert not (newly_imported & banned), newly_imported & banned
