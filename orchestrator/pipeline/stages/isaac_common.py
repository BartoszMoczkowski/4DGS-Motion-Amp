"""Shared plumbing for the three ``isaac``-environment stages (T11): the container-exec call,
targeting Isaac Sim's own bundled interpreter rather than the container's plain ``python``.

Every ``pipeline/vendored/isaac/*.py`` script (``split_mesh``/``add_motion``/``omni_capture``) is
invoked as a separate process inside the ``isaac`` container via ``ctx.containers`` (T08, wired
into ``StageContext`` by ``pipeline.dag.scheduler`` alongside ``ctx.paths`` -- same mechanism T09's
``cuda`` stages use, see ``pipeline.stages.cuda_common``'s module docstring). The one real
difference from ``cuda_common``: the interpreter. ``INSTRUCTIONS.md``'s "Environments" section
names ``isaac``'s entry point as ``/isaac-sim/python.sh`` -- only that interpreter has
``isaacsim``/``omni.*``/``pxr`` wired onto its own ``sys.path``, unlike the container's plain
``python``, which ``split_mesh.py``/``add_motion.py`` (no ``omni.*`` runtime needed, but still
``pxr``) and ``omni_capture.py`` (needs the full running app) all require.

Unlike ``cuda_common.py``, no ``PYTHONPATH=/workspace`` fix is needed for any vendored Isaac
script: ``omni_capture.py``'s own ``import rig as rigmod`` is a *same-directory* import (``rig.py``
lives right alongside it in ``pipeline/vendored/isaac/``, see that package's docstring), which
Python already resolves via the script's own directory on ``sys.path[0]`` -- the exact mechanism
``cuda_common``'s module docstring explains *doesn't* cover a repo-root package import like
``from arguments import ...``, but does cover this case.

The generic CLI-flag builders (``bool_flag``/``list_flag``/``opt_flag``/``flag``) are env-agnostic
argparse conventions, not cuda-specific -- re-exported from :mod:`pipeline.stages.cuda_common`
rather than duplicated here.

**Revised 2026-07-16, first real-hardware run:** ``split_mesh``/``add_motion`` need one more thing
this module's original docstring got wrong -- a bare ``from pxr import ...`` (no ``SimulationApp``
launch) raised a real ``ModuleNotFoundError`` on the actual ``nvcr.io/nvidia/isaac-sim:6.0.1``
image; ``pxr`` turns out to be supplied by Kit's own extension loader at runtime, not a static
``sys.path`` entry ``python.sh`` provides for free -- true for ``omni_capture.py`` (which already
launches ``isaacsim.SimulationApp`` before touching ``pxr``), false for these two CPU-only scripts,
which were ported verbatim with no such launch. See
``pipeline.stages._isaac_kit_bootstrap``'s docstring and
``.claude_notes/NOTES_pipeline_orchestration.md`` for the full story. :func:`run_isaac_script` now
wraps those two script keys' invocation through that bootstrap automatically -- the vendored
``split_mesh.py``/``add_motion.py`` files themselves are untouched (copy-in rule intact).

**Revised again 2026-07-16, same day -- `omni_capture` moved off the `isaac` container entirely.**
NVIDIA confirms Vulkan (what Isaac Sim's Hydra/RTX renderer needs for actual rendering) isn't
supported under WSL2, which is what backs Docker Desktop's Linux containers on Windows -- real
hardware runs of ``capture.isaac`` got as far as Kit fully starting and ``omni_capture.py``'s
``main()`` completing with no exception, but its RTX render products never produced a single
frame (``IHydraTexture ... no GPU foundation`` + Replicator writer-drain timeouts). `split_mesh`/
`add_motion` only need ``pxr``/USD bindings, no rendering, so they're unaffected and keep running
in the `isaac` container via :func:`run_isaac_script` below, unchanged. `omni_capture` now runs as
a **native Windows subprocess** against Bartosz's own real Isaac Sim install instead, via
:func:`run_native_isaac_script` -- see that function's own docstring, and
``.claude_notes/NOTES_pipeline_orchestration.md``'s "adjust the project plan" entry for the full
decision. ``ISAAC_PYTHON`` (``/isaac-sim/python.sh``) remains exactly what it was for the two
scripts still using it; :data:`DEFAULT_NATIVE_ISAAC_PYTHON`/:func:`native_isaac_python` are its
direct Windows-native equivalent, used only by the native path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Optional

from ..containers import CONTAINER_ENV
from .base import StageContext
from .cuda_common import bool_flag, flag, list_flag, opt_flag  # noqa: F401 (re-exported)

__all__ = [
    "bool_flag",
    "flag",
    "list_flag",
    "opt_flag",
    "star_list_flag",
    "run_isaac_script",
    "run_native_isaac_script",
    "script_container_path",
    "native_isaac_python",
    "IsaacStageError",
    "ISAAC_PYTHON",
    "VENDORED_ISAAC_SCRIPTS",
    "KIT_BOOTSTRAP_SCRIPT",
    "NEEDS_KIT_BOOTSTRAP",
    "NATIVE_ISAAC_PYTHON_ENV_VAR",
    "DEFAULT_NATIVE_ISAAC_PYTHON",
]

#: Isaac Sim's own bundled python entrypoint (``INSTRUCTIONS.md``: "isaac ... entry
#: /isaac-sim/python.sh") -- never the container's plain ``python`` (see module docstring). Still
#: used by :func:`run_isaac_script` for ``split_mesh``/``add_motion`` only.
ISAAC_PYTHON = "/isaac-sim/python.sh"

#: Env var overriding the native Isaac Sim install's own bundled-Python launcher -- the direct
#: Windows-native equivalent of ``ISAAC_PYTHON``, used only by :func:`run_native_isaac_script`.
#: Mirrors ``pipeline.paths``'s own "env var override, sensible default, read at call time not
#: import time" convention (``get_roots``), so a test can ``monkeypatch.setenv`` without reloading
#: this module.
NATIVE_ISAAC_PYTHON_ENV_VAR = "PIPELINE_ISAAC_NATIVE_PYTHON"

#: Bartosz's own Isaac Sim install's packman python launcher. **Corrected 2026-07-18** -- the
#: original 2026-07-16 decision (see ``.claude_notes/NOTES_pipeline_orchestration.md``'s "adjust
#: the project plan" entry) used ``omni_capture.py``'s own pre-orchestrator docstring convention
#: (``Q:\Omniverse\ISAAC_SIM\IsaacSim\tools\packman\python.bat``), but that path doesn't exist on
#: Bartosz's real machine -- his actual install lives under a versioned standalone-package
#: directory instead. Overridable via :data:`NATIVE_ISAAC_PYTHON_ENV_VAR` for a differently-laid-out
#: machine.
DEFAULT_NATIVE_ISAAC_PYTHON = r"Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat"


def native_isaac_python() -> Path:
    """The native Isaac Sim Python launcher's path: env-override-then-default, read at call time
    (mirrors :func:`pipeline.paths.get_roots`'s own convention).
    """
    return Path(os.environ.get(NATIVE_ISAAC_PYTHON_ENV_VAR, DEFAULT_NATIVE_ISAAC_PYTHON))


#: repo-root-relative path to each vendored Isaac script (see pipeline.vendored.isaac).
VENDORED_ISAAC_SCRIPTS: dict[str, str] = {
    "split_mesh": "orchestrator/pipeline/vendored/isaac/split_mesh.py",
    "add_motion": "orchestrator/pipeline/vendored/isaac/add_motion.py",
    "omni_capture": "orchestrator/pipeline/vendored/isaac/omni_capture.py",
}

#: repo-root-relative path to the Kit bootstrap wrapper (see
#: ``pipeline.stages._isaac_kit_bootstrap``'s module docstring) -- new orchestrator glue, not a
#: vendored/ported reference-script copy.
KIT_BOOTSTRAP_SCRIPT = "orchestrator/pipeline/stages/_isaac_kit_bootstrap.py"

#: which vendored scripts need the Kit-bootstrap wrapper ahead of them -- everything that touches
#: ``pxr`` without launching its own ``SimulationApp``. ``omni_capture`` launches one itself and
#: must never be double-wrapped.
NEEDS_KIT_BOOTSTRAP: frozenset = frozenset({"split_mesh", "add_motion"})


class IsaacStageError(RuntimeError):
    """A vendored Isaac script exited non-zero. ``log_path`` (if any) has the full output."""


def star_list_flag(name: str, values: Optional[list]) -> list[str]:
    """``nargs="*"`` flags (``add_motion.py``'s ``--exclude``): unlike :func:`list_flag`'s
    ``nargs="+"`` (which requires at least one value and so is omitted entirely when empty), an
    explicit ``--flag`` with zero following values is valid here and means something different
    from not passing the flag at all (the script's own ``default=[...]`` no longer applies) -- so
    this only omits the flag when ``values`` is ``None`` (unset), not when it's an empty list.
    """
    if values is None:
        return []
    return [f"--{name}", *[str(v) for v in values]]


def script_container_path(script_key: str, ctx: StageContext) -> PurePosixPath:
    """The in-container path to ``pipeline/vendored/isaac/<script_key>.py`` (T08's repo bind-mount
    is what makes it visible there -- see that package's docstring)."""
    roots = ctx.paths.get_roots()
    host_path = roots.repo_root_host / VENDORED_ISAAC_SCRIPTS[script_key]
    return ctx.paths.to_container(host_path, env="isaac")


def run_isaac_script(
    ctx: StageContext,
    script_key: str,
    args: list[str],
    *,
    log_name: Optional[str] = None,
) -> None:
    """Exec ``/isaac-sim/python.sh pipeline/vendored/isaac/<script_key>.py <args>`` inside the warm
    ``isaac`` container, streaming output to ``ctx.run_dir/logs/<log_name or script_key>.log``, and
    raise :class:`IsaacStageError` on a non-zero exit (mirrors
    ``pipeline.stages.cuda_common.run_cuda_script`` exactly, minus the ``PYTHONPATH`` fix -- see
    module docstring for why that's not needed here).

    ``script_key in NEEDS_KIT_BOOTSTRAP`` (``split_mesh``/``add_motion``) routes through
    ``pipeline.stages._isaac_kit_bootstrap`` first -- see the module docstring's "Revised
    2026-07-16" note for why.
    """
    script_path = script_container_path(script_key, ctx)
    if script_key in NEEDS_KIT_BOOTSTRAP:
        bootstrap_host = ctx.paths.get_roots().repo_root_host / KIT_BOOTSTRAP_SCRIPT
        bootstrap_path = ctx.paths.to_container(bootstrap_host, env="isaac")
        cmd = [ISAAC_PYTHON, str(bootstrap_path), str(script_path), *args]
    else:
        cmd = [ISAAC_PYTHON, str(script_path), *args]
    log_path = ctx.run_dir / "logs" / f"{log_name or script_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ctx.logger.info("isaac exec: %s", " ".join(cmd))
    result = ctx.containers.exec_in_container(
        "isaac",
        cmd,
        log_path=str(log_path),
        workdir=str(ctx.paths.get_roots().repo_root_container),
    )
    ctx.logger.info("isaac exec finished: exit_code=%s log=%s", result.exit_code, result.log_path)
    if not result.ok:
        raise IsaacStageError(
            f"{script_key} exited with code {result.exit_code}; see log at {result.log_path}"
        )


def run_native_isaac_script(
    ctx: StageContext,
    script_key: str,
    args: list[str],
    *,
    log_name: Optional[str] = None,
) -> None:
    """Run ``pipeline/vendored/isaac/<script_key>.py <args>`` as a **native Windows subprocess**
    against Bartosz's own real Isaac Sim install (:func:`native_isaac_python`), instead of execing
    inside the ``isaac`` Docker container the way :func:`run_isaac_script` does.

    **Why this one path exists (2026-07-16, see
    .claude_notes/NOTES_pipeline_orchestration.md's "adjust the project plan" entry):**
    ``omni_capture.py`` is the one vendored Isaac script that actually needs Isaac Sim's RTX/Hydra
    renderer, and NVIDIA has confirmed Vulkan -- what that renderer needs on Linux -- isn't
    supported under WSL2, which is what backs Docker Desktop's Linux containers on Windows. Real
    hardware runs of ``capture.isaac`` got as far as Kit fully starting and ``main()`` completing
    with no exception, but the RTX render products never produced a single frame. Running
    ``omni_capture.py`` natively sidesteps the WSL2/Vulkan gap entirely -- it's the exact same
    Isaac-Sim-bundled interpreter convention the container uses
    (``ISAAC_PYTHON``/``/isaac-sim/python.sh``), just against Bartosz's own real Windows install
    instead (the direct native equivalent, already documented in ``omni_capture.py``'s own
    original pre-orchestrator docstring: "Windows: ...\\tools\\packman\\python.bat").
    ``split_mesh``/``add_motion`` only need ``pxr``/USD bindings -- no rendering -- so they're
    unaffected and keep using :func:`run_isaac_script` above.

    Only this function currently exists for the native path (not a generalized "run any script
    natively" like :func:`run_isaac_script`'s container version) because ``omni_capture`` is the
    only script that needs to move -- ``script_key`` is accepted for symmetry/logging, not because
    ``split_mesh``/``add_motion`` are expected to route through here too.

    No path translation happens here (unlike :func:`run_isaac_script`): a native subprocess shares
    this orchestrator process's own filesystem, so ``args`` must already carry plain host paths,
    not container ones -- ``capture_isaac.py`` builds them that way specifically for this function.
    No Kit-bootstrap wrapper either -- that's only needed by ``split_mesh``/``add_motion``'s bare
    ``from pxr import ...`` (see :data:`NEEDS_KIT_BOOTSTRAP`'s docstring), and ``omni_capture.py``
    already launches its own ``SimulationApp`` before ever touching ``pxr``.

    Reuses :data:`pipeline.containers.CONTAINER_ENV`'s ``"isaac"`` entry for the EULA/privacy-
    consent env vars Isaac Sim needs to start non-interactively -- that dict is pure data (no
    Docker import), so borrowing it here for a non-Docker subprocess doesn't pull in ``docker``.
    """
    roots = ctx.paths.get_roots()
    script_path = roots.repo_root_host / VENDORED_ISAAC_SCRIPTS[script_key]
    python_path = native_isaac_python()
    if not python_path.is_file():
        raise IsaacStageError(
            f"native Isaac Sim python launcher not found at {python_path} -- set "
            f"{NATIVE_ISAAC_PYTHON_ENV_VAR} to the real tools/packman/python.bat path under your "
            f"Isaac Sim install (see planning/WINDOWS_SETUP.md)"
        )
    cmd = [str(python_path), str(script_path), *args]

    log_path = ctx.run_dir / "logs" / f"{log_name or script_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ctx.logger.info("native isaac exec: %s", " ".join(cmd))
    env = {**os.environ, **CONTAINER_ENV["isaac"]}
    with log_path.open("wb") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(roots.repo_root_host),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    ctx.logger.info("native isaac exec finished: exit_code=%s log=%s", proc.returncode, log_path)
    if proc.returncode != 0:
        raise IsaacStageError(
            f"{script_key} (native) exited with code {proc.returncode}; see log at {log_path}"
        )
