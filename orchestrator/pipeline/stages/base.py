"""``Stage`` interface and the small context object every stage's ``run()`` receives.

A concrete stage is a thin orchestration adapter (config, artifacts, logging, ctx) around logic
*ported into this project* from an already-verified reference script (``train.py``,
``omni_capture.py``, ``motion_seg/segment_rigid.py``, ...) — see ``planning/INSTRUCTIONS.md``'s
"copy the logic in, don't call the original script" rule (2026-07-14, superseding "wrap, don't
rewrite"): a stage calls into ``pipeline.vendored.<env>.*``, never the reference script itself
(no reach-outside-the-package import hack, no subprocess to it). This module only defines the
*shape* every such adapter must have; concrete bodies are out of scope for T04 (T07/T09/T10/T11).

Keep ``StageContext`` small and explicit: it is the contract every future stage depends on, so
changing its fields later touches every stage. ``paths``/``containers`` are typed ``Any`` on
purpose — T06 (path translation) and T08 (container manager) haven't landed yet; a stage that
needs them takes whatever those tasks end up returning, this module just reserves the slot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ..artifacts import Artifact

#: Which of the three execution environments (``planning/INSTRUCTIONS.md``) a stage runs in.
Environment = Literal["host", "cuda", "isaac"]


class ResourceRequest(BaseModel):
    """What a stage needs from the resource manager (T12) before it's scheduled.

    Estimates, not hard limits — T12 uses them to avoid co-scheduling stages whose combined
    VRAM would exceed free VRAM, and to pick adaptive knobs (``low_vram_mode``, subsample size,
    ...) from measured headroom. A CPU-only stage is ``needs_gpu=False`` with both estimates at
    their default of ``0``.
    """

    model_config = ConfigDict(extra="forbid")

    needs_gpu: bool = False
    vram_gb: float = 0.0
    ram_gb: float = 0.0


@dataclass
class StageContext:
    """Everything a stage's ``run()`` gets — deliberately small (see module docstring).

    Attributes:
        run_id: identifier of the run this stage execution belongs to.
        stage_name: the resolved ``"role.impl"`` name of *this* stage instance (used as
            ``Artifact.producing_stage`` so stages don't have to know their own registered name).
        config: this stage's resolved, already-validated config as a plain ``dict``
            (``PipelineConfig``'s relevant section, T02, ``model_dump()``'d — kept as a dict
            rather than the pydantic type so this module doesn't depend on ``pipeline.config``).
        run_dir: the run's directory (``runs/<run_id>/``, T03) — where a stage should write
            outputs/logs unless an output path is dictated by legacy script conventions.
        logger: a stdlib ``Logger`` already configured to write to this stage's log file.
        inputs: this stage's declared ``inputs`` (T04), resolved to the actual
            :class:`~pipeline.artifacts.Artifact` each name currently points to in the run's
            manifest — how a stage finds *where* its upstream data actually lives (e.g. a
            capture directory, a ``trajectories.npz`` path). Added in T07: T04/T05 built
            ``config`` and ``run_dir`` but never actually plumbed resolved input artifacts
            through to a stage's ``run(ctx)``, which every real (non-toy) stage needs — see
            ``pipeline/stages/convert.py`` et al. Defaults to ``{}`` so existing callers
            (``EchoStage``, T05's toy test stages) that never read it are unaffected.
        paths: path-translation handle (T06) — since T09, ``pipeline.dag.scheduler.run_dag``
            always sets this to the :mod:`pipeline.paths` module itself, so any stage can call
            ``ctx.paths.to_container(...)``/``ctx.paths.get_roots()`` etc.; a ``host``-environment
            stage just never reads it.
        containers: container-manager handle (T08) — since T09, always set to the
            :mod:`pipeline.containers` module itself (``ctx.containers.exec_in_container(...)``);
            ``cuda``/``isaac`` stages use this to exec their vendored script inside the right
            container (see ``pipeline.stages.cuda_common``).
    """

    run_id: str
    stage_name: str
    config: dict[str, Any]
    run_dir: Path
    logger: Any
    inputs: dict[str, Artifact] = field(default_factory=dict)
    paths: Any = None
    containers: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Base class every stage implementation subclasses.

    Concrete subclasses are registered with :func:`pipeline.stages.registry.register` under a
    ``"role.impl"`` name (e.g. ``"segment.rigid"``, matching ``SegmentConfig.impl`` in
    ``pipeline.config.models`` — the registry resolves a config's ``impl`` string to one of
    these classes). The decorator sets ``name``/``role``/``impl`` on the class; don't set them
    by hand.

    Class-level declarations (``inputs``/``outputs``/``environment``/``resources``) describe the
    stage *before* any instance exists — the scheduler (T05) needs this to build the DAG and the
    resource manager (T12) needs it before deciding what to run next.
    """

    #: Named upstream artifact dependencies this stage consumes (keys into a run's artifact map).
    inputs: ClassVar[tuple[str, ...]] = ()
    #: Named artifacts this stage produces (keys ``run(ctx)``'s return dict must contain exactly).
    outputs: ClassVar[tuple[str, ...]] = ()
    #: Which of host|cuda|isaac this stage must execute in.
    environment: ClassVar[Environment] = "host"
    #: Estimated resource needs; overridden per-stage. Defaults to the cheapest case (CPU, ~0).
    resources: ClassVar[ResourceRequest] = ResourceRequest()

    #: Set by ``@register(...)`` — do not assign directly.
    name: ClassVar[str] = ""
    role: ClassVar[str] = ""
    impl: ClassVar[str] = ""

    @abstractmethod
    def run(self, ctx: StageContext) -> dict[str, Artifact]:
        """Execute the stage, returning exactly the artifacts named in ``self.outputs``.

        Implementations call the vendored, ported-in copy of this stage's logic (see module
        docstring) rather than reimplementing it from scratch, and never import or subprocess
        the original reference script. Raising is how a stage reports failure; the caller
        (T05's scheduler) is responsible for turning that into a ``StageRecord`` with
        ``status="failed"`` and the exception message.
        """
        raise NotImplementedError
