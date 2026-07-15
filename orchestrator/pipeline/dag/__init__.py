"""DAG scheduler + cache: topo-sort by artifact deps, skip fresh stages.

See ``planning/tasks/T05-dag-scheduler-and-cache.md`` for scope. Three leaf-ish pieces:

- :mod:`pipeline.dag.graph` — pure structure: resolve registered stage names, topo-sort, detect
  cycles, report which inputs aren't produced within a given stage set.
- :mod:`pipeline.dag.cache` — cache key computation (config + input hashes + code version) and a
  small cross-run cache index so a fresh run can still skip a stage another run already did.
- :mod:`pipeline.dag.scheduler` — :func:`run_dag`, the engine that ties those two together with
  the run manifest (T03) and the stage registry (T04): builds the graph, applies
  ``from_stage``/``to_stage``/``only``/``force``, executes each selected stage in order (skipping
  fresh ones), and writes status/timing/artifacts into the manifest as it goes.

``pipeline.api``'s ``run_pipeline``/``run_stage`` (also T05) are the only intended callers from
outside this package; nothing here imports torch/CUDA/docker/pynvml.
"""

from __future__ import annotations

from .cache import compute_cache_key, get_cached, put_cached, stage_source_hash
from .graph import (
    CycleError,
    DAGError,
    DAGNode,
    MissingDependencyError,
    external_inputs,
    producer_map,
    resolve_nodes,
    topo_sort,
)
from .scheduler import run_dag

__all__ = [
    "run_dag",
    # graph
    "DAGNode",
    "DAGError",
    "CycleError",
    "MissingDependencyError",
    "resolve_nodes",
    "producer_map",
    "external_inputs",
    "topo_sort",
    # cache
    "compute_cache_key",
    "stage_source_hash",
    "get_cached",
    "put_cached",
]
