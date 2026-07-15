"""The plugin mechanism: ``@register("role.impl")`` + lookup by role or full name.

This is how "add a new idea" works end to end (``planning/pipeline-orchestration-plan.md``
problem #3): write a ``Stage`` subclass, decorate it with its ``role.impl`` name, and reference
that name in a config preset (``SegmentConfig.impl`` etc., T02). Nothing else has to change —
no core edits, no new registry entry anywhere but the decorator itself.

Registration happens at *import time* of the module defining the stage, so discovery is just
"make sure that module got imported" (directly, via a package ``__init__`` that imports known
stage modules, or via Python entry points for out-of-tree/experimental stages — the entry-point
mechanism itself is left for whichever task first needs an out-of-tree stage; this module only
needs the registry to not care how a class arrived).
"""

from __future__ import annotations

from typing import TypeVar

from .base import Stage

T = TypeVar("T", bound=type[Stage])


class StageRegistryError(Exception):
    """Base class for registry errors."""


class DuplicateStageError(StageRegistryError):
    """Raised when ``@register`` is used twice for the same ``role.impl`` name."""


class StageNotFoundError(StageRegistryError):
    """Raised by lookups for a name/role/impl that was never registered."""


#: full "role.impl" name -> stage class.
_REGISTRY: dict[str, type[Stage]] = {}
#: role -> {impl -> stage class}, the shape config's "role -> impl" selection needs.
_BY_ROLE: dict[str, dict[str, type[Stage]]] = {}


def register(name: str):
    """Class decorator: register a :class:`~pipeline.stages.base.Stage` under ``"role.impl"``.

    Sets ``cls.name``/``cls.role``/``cls.impl`` from the given name so the class knows its own
    registered identity (used e.g. as ``Artifact.producing_stage`` by callers that only have the
    class, not an instance). Raises :class:`DuplicateStageError` if ``name`` is already taken —
    re-registering under the same name (e.g. a module imported twice) is almost always a bug,
    not an intentional override.
    """
    if "." not in name or name.startswith(".") or name.endswith("."):
        raise ValueError(
            f"stage name must be of the form 'role.impl' (got {name!r})"
        )
    role, impl = name.split(".", 1)
    if not role or not impl:
        raise ValueError(f"stage name must be of the form 'role.impl' (got {name!r})")

    def decorator(cls: T) -> T:
        if not (isinstance(cls, type) and issubclass(cls, Stage)):
            raise TypeError(f"@register({name!r}) must decorate a Stage subclass, got {cls!r}")
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise DuplicateStageError(
                f"stage {name!r} already registered to {existing.__module__}.{existing.__qualname__}"
            )
        cls.name = name
        cls.role = role
        cls.impl = impl
        _REGISTRY[name] = cls
        _BY_ROLE.setdefault(role, {})[impl] = cls
        return cls

    return decorator


def get_stage(name: str) -> type[Stage]:
    """Look up a stage class by its full ``"role.impl"`` name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise StageNotFoundError(f"no stage registered as {name!r}; known stages: {known}") from None


def get_stage_for_role(role: str, impl: str) -> type[Stage]:
    """Look up a stage class by role + impl — what ``SegmentConfig.impl``-style config uses."""
    by_impl = _BY_ROLE.get(role)
    if by_impl is None:
        known = ", ".join(sorted(_BY_ROLE)) or "(none registered)"
        raise StageNotFoundError(f"no stages registered for role {role!r}; known roles: {known}")
    try:
        return by_impl[impl]
    except KeyError:
        known = ", ".join(sorted(by_impl))
        raise StageNotFoundError(
            f"role {role!r} has no impl {impl!r}; known impls for {role!r}: {known}"
        ) from None


def list_roles() -> dict[str, list[str]]:
    """Every registered role -> sorted list of its registered impls."""
    return {role: sorted(impls) for role, impls in _BY_ROLE.items()}


def list_stages() -> list[str]:
    """Every registered full ``"role.impl"`` name, sorted."""
    return sorted(_REGISTRY)


def _reset_registry_for_tests() -> None:
    """Test-only escape hatch: clear the registry between tests that register throwaway stages.

    Not exported from ``pipeline.stages`` on purpose — production code should never need to
    unregister a stage; only ``tests/test_stages.py`` reaches in and calls this directly.
    """
    _REGISTRY.clear()
    _BY_ROLE.clear()
