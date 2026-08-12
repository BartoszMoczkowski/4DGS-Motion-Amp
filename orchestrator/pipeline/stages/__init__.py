"""Stage base class + registry (role.impl names, config-selectable implementations).

The plugin mechanism (``planning/tasks/T04-stage-base-and-registry.md``): a ``Stage`` subclass
declares its inputs/outputs/environment/resources and a ``run(ctx)`` body, registers under a
``"role.impl"`` name via ``@register``, and config (T02's ``SegmentConfig.impl`` and friends)
picks which impl fills a role -- no core edits needed to add a new idea.

Importing this package also imports :mod:`pipeline.stages.echo`, which registers the dummy
``EchoStage`` under ``"test.echo"`` -- always available as a registry smoke-test / example,
never a real pipeline role. Concrete real stages (T07/T09/T10/T11) are expected to follow the
same pattern: define the class in its own module, decorate with ``@register``, import that
module here (or from wherever the pipeline is assembled) so it actually registers.

T07 adds the first three *real* stages this way: :mod:`pipeline.stages.convert`
(``convert.default``), :mod:`pipeline.stages.segment_rigid` (``segment.rigid``), and
:mod:`pipeline.stages.seg_eval` (``seg_eval.default``) -- all CPU/``host``-environment stages
that call already-verified logic *ported into* :mod:`pipeline.vendored.host` (copied from
``omniverse-pipeline/omniverse_pipeline/omni_to_4dgs.py``, ``motion-seg/motion_seg/segment_rigid.py``,
``motion-seg/motion_seg/evaluate_segmentation.py`` respectively) -- per ``planning/INSTRUCTIONS.md``'s
"copy the logic in, don't call the original script" rule, never by importing or subprocessing
those reference scripts directly.

T09 adds the four ``cuda``-environment stages: :mod:`pipeline.stages.train` (``train.default``),
:mod:`pipeline.stages.render` (``render.default``), :mod:`pipeline.stages.seg_extract`
(``seg_extract.default``), :mod:`pipeline.stages.amp` (``amp.default``). Unlike the ``host``
stages, these never import their vendored counterpart (:mod:`pipeline.vendored.cuda`) — they exec
it as a separate process *inside* the ``cuda`` container via ``ctx.containers`` (T08); see
:mod:`pipeline.stages.cuda_common` for the shared CLI-building/bridge-file/exec plumbing.

T10 adds :mod:`pipeline.stages.segment_mbs` (``segment.mbs``) — a *second* impl behind the
``segment`` role :mod:`pipeline.stages.segment_rigid` (``segment.rigid``, T07) already occupies,
the concrete demonstration of "add a new idea = register an impl + a preset" (no core edits
anywhere in this package). Follows T09's ``cuda``-stage shape (MotNet needs the GPU), not T07's.

T11 adds the three ``isaac``-environment stages: :mod:`pipeline.stages.prep_split`
(``prep_split.default``), :mod:`pipeline.stages.prep_motion` (``prep_motion.default``), and
:mod:`pipeline.stages.capture_isaac` (``capture.isaac``) — the synthetic-capture front end
(``planning/tasks/T11-wrap-isaac-stages.md``). Same T09 ``cuda``-stage shape (build a CLI
invocation, exec it as a separate process inside a container) but targeting the ``isaac``
container's own bundled interpreter (``/isaac-sim/python.sh``) via
:mod:`pipeline.stages.isaac_common` instead of ``cuda_common``. ``capture.isaac`` produces the
``capture`` artifact :mod:`pipeline.stages.convert` (T07) has always declared as its external
input, closing the loop so a preset's auto-planned DAG now runs end to end from a raw USD asset.
``prep_split``/``prep_motion`` (not ``prep.split``/``prep.motion`` — see
:mod:`pipeline.stages.prep_split`'s module docstring) are each their own single-impl role, one per
top-level config section, avoiding a role-name collision the registry's ``role.impl`` convention
would otherwise create.
"""

from __future__ import annotations

from . import amp  # noqa: F401  (import for its registration side-effect)
from . import capture_isaac  # noqa: F401  (import for its registration side-effect)
from . import convert  # noqa: F401  (import for its registration side-effect)
from . import echo  # noqa: F401  (import for its registration side-effect)
from . import prep_motion  # noqa: F401  (import for its registration side-effect)
from . import prep_split  # noqa: F401  (import for its registration side-effect)
from . import render  # noqa: F401  (import for its registration side-effect)
from . import seg_eval  # noqa: F401  (import for its registration side-effect)
from . import seg_extract  # noqa: F401  (import for its registration side-effect)
from . import roi_motion_gate  # noqa: F401  (import for its registration side-effect)
from . import segment_kabsch  # noqa: F401  (import for its registration side-effect)
from . import segment_mbs  # noqa: F401  (import for its registration side-effect)
from . import segment_rigid  # noqa: F401  (import for its registration side-effect)
from . import segment_rigid2  # noqa: F401  (import for its registration side-effect)
from . import train  # noqa: F401  (import for its registration side-effect)
from . import segment_rigid  # noqa: F401  (import for its registration side-effect)
from . import segment_rigid2  # noqa: F401  (import for its registration side-effect)
from . import train  # noqa: F401  (import for its registration side-effect)
from .base import Environment, ResourceRequest, Stage, StageContext
from .registry import (
    DuplicateStageError,
    StageNotFoundError,
    StageRegistryError,
    get_stage,
    get_stage_for_role,
    list_roles,
    list_stages,
    register,
)

__all__ = [
    "Stage",
    "StageContext",
    "ResourceRequest",
    "Environment",
    "register",
    "get_stage",
    "get_stage_for_role",
    "list_roles",
    "list_stages",
    "StageRegistryError",
    "DuplicateStageError",
    "StageNotFoundError",
]
