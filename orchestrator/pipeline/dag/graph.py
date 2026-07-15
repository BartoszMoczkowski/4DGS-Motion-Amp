"""Pure graph structure over registered stages: resolve names, topo-sort, spot cycles.

Deliberately config- and manifest-agnostic (a leaf module, same philosophy as
``pipeline.artifacts`` and ``pipeline.stages``): it only knows the ``inputs``/``outputs`` tuples
declared on each :class:`~pipeline.stages.base.Stage` subclass and matches them by name to build
edges. Whether an "input with no producer in this set" is actually an error (nothing upstream will
ever produce it) or fine (it's already sitting in a resumed run's artifacts) depends on
information this module doesn't have — that's :mod:`pipeline.dag.scheduler`'s job, using
:func:`external_inputs` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..stages import Stage, get_stage


class DAGError(Exception):
    """Base class for graph-structure errors."""


class CycleError(DAGError):
    """The given stage set has no valid topological order."""


class MissingDependencyError(DAGError):
    """An input artifact is needed but nothing can produce it.

    Raised by :mod:`pipeline.dag.scheduler`, not this module — deciding whether an
    :func:`external_inputs` entry is actually missing requires knowing what a resumed run's
    manifest already has, which this module has no access to. Defined here anyway so every DAG-
    structure error lives under one :class:`DAGError` hierarchy.
    """


@dataclass(frozen=True)
class DAGNode:
    """One resolved stage in a graph: its registered name plus its declared contract."""

    name: str
    stage_cls: type[Stage]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def resolve_nodes(stage_names: Sequence[str]) -> dict[str, DAGNode]:
    """Look up every name via the registry, in the order given. Raises ``StageNotFoundError``
    (from :mod:`pipeline.stages.registry`) for any name that isn't registered — the registry's own
    error already names what *is* registered, so this doesn't wrap it further."""

    nodes: dict[str, DAGNode] = {}
    for name in stage_names:
        cls = get_stage(name)
        nodes[name] = DAGNode(name=name, stage_cls=cls, inputs=tuple(cls.inputs), outputs=tuple(cls.outputs))
    return nodes


def producer_map(nodes: dict[str, DAGNode]) -> dict[str, set[str]]:
    """Artifact name -> set of node names (within ``nodes``) that declare producing it.

    Normally exactly one producer per name; if a caller registers two stages in the same DAG that
    both claim the same output name, this records both rather than silently picking one — the
    caller gets a graph that depends on either, which is at least honest about the ambiguity.
    """

    producers: dict[str, set[str]] = {}
    for name, node in nodes.items():
        for out in node.outputs:
            producers.setdefault(out, set()).add(name)
    return producers


def external_inputs(nodes: dict[str, DAGNode]) -> dict[str, set[str]]:
    """Input artifact name -> set of node names that need it but no node in ``nodes`` produces it.

    "External" here just means "not produced within this particular stage set" — it says nothing
    about whether the input is actually available (e.g. from a resumed run's existing artifacts).
    The scheduler decides that; this module only reports the structural fact.
    """

    producers = producer_map(nodes)
    ext: dict[str, set[str]] = {}
    for name, node in nodes.items():
        for inp in node.inputs:
            if inp not in producers:
                ext.setdefault(inp, set()).add(name)
    return ext


def topo_sort(nodes: dict[str, DAGNode]) -> list[str]:
    """Kahn's algorithm over internal edges only (a node depends on whatever in ``nodes``
    produces each of its inputs; external inputs have no in-graph producer to order against).

    Deterministic: ties are broken by name so the same node set always yields the same order.
    Raises :class:`CycleError` naming whichever nodes couldn't be ordered if the graph isn't a
    DAG.
    """

    producers = producer_map(nodes)
    # name -> set of names it directly depends on (must run before it).
    depends_on: dict[str, set[str]] = {
        name: {p for inp in node.inputs for p in producers.get(inp, ()) if p != name}
        for name, node in nodes.items()
    }
    remaining = set(nodes)
    ordered: list[str] = []
    while remaining:
        ready = sorted(n for n in remaining if depends_on[n] <= set(ordered))
        if not ready:
            raise CycleError(
                f"cycle detected among stages: {sorted(remaining)} "
                f"(each still waiting on one of: "
                f"{ {n: sorted(depends_on[n] & remaining) for n in sorted(remaining)} })"
            )
        ordered.extend(ready)
        remaining -= set(ready)
    return ordered
