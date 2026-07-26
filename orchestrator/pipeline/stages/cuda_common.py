"""Shared plumbing for the four ``cuda``-environment stages (T09): CLI-flag builders, the
bridge-config file (``pipeline.config.bridge``), and the container-exec call itself.

Every ``pipeline/vendored/cuda/*.py`` script is a plain ``argparse`` CLI (verbatim ports of
``core/train.py``/``core/render.py``/``motion-seg/motion_seg/extract_trajectories.py``/``core/render_amp.py`` — see
``pipeline.vendored.cuda``'s package docstring) invoked as ``python <script> <args...>`` *inside*
the ``cuda`` container via ``ctx.containers`` (T08, wired into ``StageContext`` by
``pipeline.dag.scheduler`` alongside ``ctx.paths`` — see that module's "T09" note). None of these
functions import ``torch``/``docker`` — this module only builds a command line and hands it to
``ctx.containers.exec_in_container``.

Two kinds of config field feed a script's CLI, mirroring how ``utils/params_utils.merge_hparams``
itself splits the world:

- **Bridged** (``model``/``pipeline_params``/``hidden``/``optim``) — written once per stage call
  into a temp ``arguments/multipleview/<name>.py``-style file (:mod:`pipeline.config.bridge`) and
  passed via ``--configs``; these are the fields ``merge_hparams`` (called inside every vendored
  script) applies *unconditionally*, overriding any CLI value with the same name.
- **CLI-only** (``train``/``render``/``seg_extract``/``amp``'s own fields, e.g. ``--port``,
  ``--n-times``, ``--amp_factors``) — plain argparse flags each script defines itself, outside any
  ``merge_hparams`` group; these must be passed as explicit ``--flag value`` entries.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Optional

from ..config.bridge import write_bridge
from .base import StageContext

#: repo-root-relative path to each vendored CUDA script (see pipeline.vendored.cuda).
VENDORED_CUDA_SCRIPTS: dict[str, str] = {
    "train": "orchestrator/pipeline/vendored/cuda/train.py",
    "render": "orchestrator/pipeline/vendored/cuda/render.py",
    "seg_extract": "orchestrator/pipeline/vendored/cuda/seg_extract.py",
    "amp": "orchestrator/pipeline/vendored/cuda/amp.py",
    "mbs_infer": "orchestrator/pipeline/vendored/cuda/mbs_infer.py",
}

#: `PYTHONPATH` a container-side script needs so `from arguments import ...` etc. resolve —
#: Python only puts the *script's own* directory on `sys.path[0]`, not the exec `workdir`, and
#: `pipeline/vendored/cuda/*.py` lives several directories below the repo root those imports need.
#: Since the repo-layout split, the importable top-level packages (`scene`, `utils`,
#: `arguments`, `gaussian_renderer`, `motion_amp`, `lpipsPyTorch`) live under `core/`.
CUDA_EXTRA_ENV: dict[str, str] = {"PYTHONPATH": "/workspace/core"}


class CudaStageError(RuntimeError):
    """A vendored CUDA script exited non-zero. ``log_path`` (if any) has the full output.

    ``log_path`` is a real constructor attribute (not just embedded in the message string) so
    T12's ``pipeline.resources.oom_retry.is_oom_error`` can read the captured output straight off
    the exception to check for a CUDA-OOM signature, without re-parsing it out of ``str(exc)``.
    """

    def __init__(self, message: str, *, log_path: Optional[str] = None) -> None:
        super().__init__(message)
        self.log_path = log_path


# --- CLI-flag builders (mirroring argparse's own store_true/nargs='+'/optional conventions) ----


def bool_flag(name: str, value: bool) -> list[str]:
    """``action="store_true"`` flags: present (no value) if ``True``, omitted if ``False`` —
    never emit ``--flag false`` (these scripts have no such syntax)."""
    return [f"--{name}"] if value else []


def list_flag(name: str, values: Optional[list[Any]]) -> list[str]:
    """``nargs="+"`` flags: omitted entirely if empty/``None`` (matches each script's own
    ``default=[]``/``default=None`` — an explicit empty list on the CLI is a parse error for
    ``nargs="+"``, which requires at least one value)."""
    if not values:
        return []
    return [f"--{name}", *[str(v) for v in values]]


def opt_flag(name: str, value: Any) -> list[str]:
    """A plain ``--flag value``, omitted if ``value`` is ``None`` or ``""`` (matches each
    script's own "unset" sentinel for optional string/int flags)."""
    if value is None or value == "":
        return []
    return [f"--{name}", str(value)]


def flag(name: str, value: Any) -> list[str]:
    """A plain, always-present ``--flag value``."""
    return [f"--{name}", str(value)]


# --- bridge file + path translation -------------------------------------------------------------


def write_stage_bridge(ctx: StageContext) -> PurePosixPath:
    """Write this stage-call's bridge file under ``ctx.run_dir`` and return its *container* path.

    Reads ``ctx.config["_bridge"]`` — the ``{"model": ..., "pipeline_params": ..., "hidden": ...,
    "optim": ...}`` dict ``pipeline.api._stage_config_for`` merges into every ``cuda``-role
    stage's config (T09) — and shapes it exactly like ``pipeline.config.bridge.write_bridge``
    expects (a ``PipelineConfig.model_dump()``-shaped dict, just with only those four keys).

    One bridge file per stage invocation (not shared/cached across stages) — cheap to (re)write
    and keeps every stage call self-contained; ``pipeline.dag.cache``'s cache key already covers
    whether a stage needs to re-run at all, so writing this on every real invocation costs nothing
    extra in the common (cached-skip) case.
    """
    bridge_host = ctx.run_dir / f"{ctx.stage_name.replace('.', '_')}_arguments_bridge.py"
    write_bridge(ctx.config["_bridge"], bridge_host)
    return ctx.paths.to_container(bridge_host, env="cuda")


def script_container_path(script_key: str, ctx: StageContext) -> PurePosixPath:
    """The in-container path to ``pipeline/vendored/cuda/<script_key>.py`` (T08's repo bind-mount
    is what makes it visible there — see that package's docstring)."""
    roots = ctx.paths.get_roots()
    host_path = roots.repo_root_host / VENDORED_CUDA_SCRIPTS[script_key]
    return ctx.paths.to_container(host_path, env="cuda")


def run_cuda_script(
    ctx: StageContext,
    script_key: str,
    args: list[str],
    *,
    log_name: Optional[str] = None,
) -> None:
    """Exec ``pipeline/vendored/cuda/<script_key>.py <args>`` inside the warm ``cuda`` container,
    streaming output to ``ctx.run_dir/logs/<log_name or script_key>.log``, and raise
    :class:`CudaStageError` on a non-zero exit (this is how a ``Stage.run`` reports failure to the
    scheduler, per ``pipeline.stages.base.Stage.run``'s docstring).
    """
    script_path = script_container_path(script_key, ctx)
    cmd = ["python", str(script_path), *args]
    log_path = ctx.run_dir / "logs" / f"{log_name or script_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ctx.logger.info("cuda exec: %s", " ".join(cmd))
    result = ctx.containers.exec_in_container(
        "cuda",
        cmd,
        log_path=str(log_path),
        workdir=str(ctx.paths.get_roots().repo_root_container),
        environment=CUDA_EXTRA_ENV,
    )
    ctx.logger.info("cuda exec finished: exit_code=%s log=%s", result.exit_code, result.log_path)
    if not result.ok:
        raise CudaStageError(
            f"{script_key} exited with code {result.exit_code}; see log at {result.log_path}",
            log_path=result.log_path,
        )
