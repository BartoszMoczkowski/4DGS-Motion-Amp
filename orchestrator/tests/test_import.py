"""Trivial import test — the T01 acceptance bar for the pipeline package."""

import sys


def test_pipeline_imports():
    import pipeline

    assert pipeline.__version__


def test_submodule_stubs_import():
    from pipeline import artifacts, config, containers, dag, resources, stages

    assert all(m is not None for m in (config, stages, dag, artifacts, containers, resources))


def test_api_cancel_requires_existing_run():
    """Cancel (T17/T30) confirms run exists, raising FileNotFoundError on unknown run."""
    import pytest

    from pipeline import api

    with pytest.raises(FileNotFoundError):
        api.cancel("run001")


def test_no_heavy_imports_at_module_scope():
    """No torch/CUDA/docker/pynvml/psutil should be pulled in just by importing pipeline."""
    banned = {"torch", "torchvision", "docker", "pynvml", "psutil"}
    before = set(sys.modules)

    import pipeline  # noqa: F401

    after = set(sys.modules)
    newly_imported = after - before
    assert not (newly_imported & banned), newly_imported & banned
