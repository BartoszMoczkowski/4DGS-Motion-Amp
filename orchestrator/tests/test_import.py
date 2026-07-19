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
    ``tests/test_dag.py``); ``list_containers``/``start_container``/``stop_container`` to
    ``pipeline.containers`` in T08; ``gpu_status`` to ``pipeline.resources`` in T12 (see
    ``tests/test_resources.py``). Only ``cancel`` remains a stub — out of scope for every task
    scheduled so far."""
    from pipeline import api

    import pytest

    with pytest.raises(NotImplementedError):
        api.cancel("run001")


def test_no_heavy_imports_at_module_scope():
    """No torch/CUDA/docker/pynvml/psutil should be pulled in just by importing pipeline."""
    banned = {"torch", "torchvision", "docker", "pynvml", "psutil"}
    before = set(sys.modules)

    import pipeline  # noqa: F401

    after = set(sys.modules)
    newly_imported = after - before
    assert not (newly_imported & banned), newly_imported & banned
