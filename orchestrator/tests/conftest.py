"""Shared fixtures across the whole test suite.

T12 wired real VRAM/RAM gating (``pipeline.resources.check_headroom``) and peak-mem monitoring
(``pipeline.resources.ResourceMonitor``) directly into ``pipeline.dag.scheduler.run_dag``'s
per-stage loop. Every pre-existing integration test that runs a ``cuda``/``isaac``-environment
stage through ``run_dag`` (T09/T10/T11's own suites) does so against a *fake*
``ctx.containers``/``exec_in_container`` — no real GPU/Docker/native subprocess ever actually
runs, so there is no real VRAM/RAM footprint to gate against. But this sandbox is a real (if
small, ~4GB) Linux VM with its own incidental system RAM that has nothing to do with what a stage
would need on Bartosz's actual target machine — gating those fake-exec tests against *this*
sandbox's own RAM would fail them for a reason with zero bearing on whether the scheduler/stage
logic under test is correct.

This autouse fixture forces ``pipeline.resources.query.query_gpu_memory``/``query_ram`` to return
``None`` for every test by default — exactly the same "can't measure, don't block" value real
telemetry returns on a GPU-less/psutil-less machine (see that module's own docstring), so gating
is a no-op and peak-mem comes back ``None`` unless a test opts back in. ``tests/test_resources.py``
(which specifically exercises gating/monitoring/adaptive behavior) monkeypatches its own canned
values on top of this, function-scoped, the same way any other test overrides a fixture default.

Patches ``pipeline.resources.gating``'s and ``pipeline.resources.monitor``'s own ``_query`` module
reference (not ``pipeline.resources.query``'s names directly) — both modules do
``from . import query as _query`` specifically so a single patch on the query module's functions
is visible through every caller, mirroring ``pipeline.dag.scheduler``'s own ``from .. import
containers as _containers`` convention.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_resource_telemetry_by_default(monkeypatch: pytest.MonkeyPatch):
    from pipeline.resources import query as query_mod

    monkeypatch.setattr(query_mod, "query_gpu_memory", lambda: None)
    monkeypatch.setattr(query_mod, "query_ram", lambda: None)
    yield
