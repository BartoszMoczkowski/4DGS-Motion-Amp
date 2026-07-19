"""Builds the FastMCP app + auth wrapper, and a ``python -m mcp_server`` entry point.

T13 built the transport/auth skeleton plus one connectivity-proof tool (``gpu_status``). T14 fills
in the rest of ``planning/tasks/T14-mcp-tools-and-resources.md``'s scope: the full control + read
surface so Claude can actually launch, watch, and inspect a real run.

Tools (grouped by ``ARCHITECTURE.md``'s Layer 2 bullet):

- **Discovery**: ``list_presets``, ``validate_config``, ``list_runs``, ``list_artifacts``.
- **Run lifecycle (async)**: ``run_pipeline``/``run_stage`` return a ``run_id`` immediately (see
  ``mcp_server.jobs`` for how — a stage can run for hours, so neither ever blocks the request for
  the run's own duration); ``get_run_status``/``tail_logs`` poll progress; ``cancel_run`` is
  honestly best-effort (Layer 1 has no real cancellation yet, see its own docstring).
- **Artifact inspection**: ``read_artifact`` (text/JSON/npz *summary*, directory listing — never a
  raw blob, see ``mcp_server.artifact_view``), ``get_preview`` (an inline image for a ``png``
  artifact, or a pointer for a ``video`` one — what actually lets Claude *see* results).
- **Resources**: the same manifest/log/artifact data, also reachable as MCP resources
  (``run://<run_id>/manifest``, ``run://<run_id>/log/<stage>``,
  ``run://<run_id>/artifact/<artifact_name>``) for a client that prefers reading over calling a
  tool.
- **Machine control**: ``gpu_status`` (T13), ``list_containers``/``start_container``/
  ``stop_container``.

Every tool body does its own lazy ``from pipeline... import ...`` — mirrors ``pipeline.api``'s own
"lazy import per call" convention, and keeps this module's own load-time imports light (only
``pipeline.artifacts``/``pipeline.paths`` at module scope, both explicitly documented as
docker/torch/pynvml-free leaf packages — needed here, not lazily, so their types are resolvable
when FastMCP inspects each tool's signature at registration time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Image
from starlette.types import ASGIApp

from pipeline.artifacts import Artifact
from pipeline.paths import Env

from .auth import BearerAuthMiddleware
from .config import ServerSettings, load_settings


def build_mcp() -> FastMCP:
    """A fresh :class:`FastMCP` instance with the full T14 tool/resource set.

    ``stateless_http=True``: each HTTP request is self-contained rather than requiring a
    server-held session to persist across calls. This server's own process state (T14's
    ``mcp_server.jobs`` registry) lives independently of any one request/session — a restart loses
    in-flight background-job tracking the same way it always would have (the manifest itself,
    T03, is what actually survives a restart; ``get_run_status`` reflects that either way).
    """

    mcp = FastMCP(
        "4dgs-pipeline",
        instructions=(
            "Layer 2 MCP server for the 4DGS motion-amp orchestrator. Wraps orchestrator/"
            "pipeline/api.py (Layer 1) so Claude can drive the pipeline on a machine with the "
            "GPU/Docker/Isaac Sim the Claude sandbox itself doesn't have. Typical flow: "
            "list_presets -> validate_config -> run_pipeline (returns a run_id right away) -> "
            "poll get_run_status/tail_logs -> list_artifacts -> read_artifact/get_preview once "
            "stages finish."
        ),
        stateless_http=True,
    )

    # --- machine / connectivity (T13) -----------------------------------------------------

    @mcp.tool()
    def gpu_status() -> dict[str, Any]:
        """Current VRAM/RAM usage and free headroom on this machine.

        Delegates to ``pipeline.api.gpu_status`` (T12). ``None`` sub-dicts mean that dimension
        couldn't be measured here (no GPU, or no working ``pynvml``/``nvidia-smi``/``psutil``) —
        not an error.
        """
        from pipeline.api import gpu_status as _gpu_status

        return _gpu_status()

    @mcp.tool()
    def list_containers() -> list[dict[str, Any]]:
        """Every managed ``cuda``/``isaac`` container this server knows about (image, state,
        mounts) — delegates to ``pipeline.api.list_containers``. Running or not; a container this
        server never started (no ``pipeline.managed`` label) never shows up here.
        """
        from pipeline.api import list_containers as _list_containers

        return _list_containers()

    @mcp.tool()
    def start_container(env: Env) -> dict[str, Any]:
        """Start (or reuse a warm) container for ``env`` (``"cuda"`` or ``"isaac"``).

        Rarely needed directly — ``run_pipeline``/``run_stage`` start whatever container a stage
        needs on their own. Useful for warming one up ahead of a run, or checking that Docker
        Desktop is actually reachable before launching something longer.
        """
        from pipeline.api import start_container as _start_container

        return {"env": env, "container_id": _start_container(env)}

    @mcp.tool()
    def stop_container(container_id: str) -> dict[str, Any]:
        """Stop a managed container by id (from ``list_containers``)."""
        from pipeline.api import stop_container as _stop_container

        _stop_container(container_id)
        return {"container_id": container_id, "stopped": True}

    # --- discovery -------------------------------------------------------------------------

    @mcp.tool()
    def list_presets() -> list[str]:
        """Available config presets (``base``, ``pump01``, ...) — see
        ``pipeline/config/presets/*.yaml``."""
        from pipeline.api import list_presets as _list_presets

        return _list_presets()

    @mcp.tool()
    def validate_config(preset: str) -> dict[str, Any]:
        """Resolve + validate ``preset`` without running anything — the fully-resolved config as
        a plain dict, or a validation error if the preset/its overrides don't type-check.
        """
        from pipeline.api import validate_config as _validate_config

        return _validate_config(preset)

    @mcp.tool()
    def list_runs() -> list[dict[str, Any]]:
        """Summaries of every known run (id/preset/status/timestamps), most recently updated
        first."""
        from pipeline.api import list_runs as _list_runs

        return _list_runs()

    @mcp.tool()
    def list_artifacts(run_id: str) -> list[dict[str, Any]]:
        """Every artifact record (name/kind/path/producing stage/hash) ``run_id`` has produced so
        far — call ``read_artifact``/``get_preview`` on a name from here to see its actual
        content."""
        from pipeline.api import list_artifacts as _list_artifacts

        return _list_artifacts(run_id)

    # --- run lifecycle (async — see mcp_server.jobs) ----------------------------------------

    @mcp.tool()
    def run_pipeline(
        preset: str,
        external_artifacts: Optional[dict[str, Artifact]] = None,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        only: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Launch (or resume) ``preset``'s full auto-planned DAG. Returns a ``run_id``
        **immediately** — this does not wait for the run to finish (it can take hours; see
        ``mcp_server.jobs``). Poll ``get_run_status``/``tail_logs`` afterward.

        ``external_artifacts`` pre-seeds a DAG's external inputs the run itself can't produce
        (e.g. ``{"raw_mesh": Artifact(name="raw_mesh", kind="usd", path="...",
        producing_stage="external")}``) — required whenever the selected stages include one that
        declares an input nothing else in this run produces (``prep_split.default``'s
        ``raw_mesh``, most commonly). Raises immediately (before any ``run_id`` is even returned)
        if ``preset`` doesn't resolve/validate; any failure *after* that (a missing external
        input, a stage genuinely failing) is recorded in the run's own manifest instead — see
        ``get_run_status``.
        """
        from .jobs import start_pipeline_run

        run_id = start_pipeline_run(
            preset,
            external_artifacts=external_artifacts,
            from_stage=from_stage,
            to_stage=to_stage,
            only=only,
            force=force,
        )
        return {"run_id": run_id}

    @mcp.tool()
    def run_stage(run_id: str, stage: str, force: bool = False) -> dict[str, Any]:
        """Run one stage (e.g. ``"train.default"``) within an already-existing ``run_id``.
        Returns immediately, same async pattern as ``run_pipeline``. The stage's declared inputs
        must already be present among ``run_id``'s recorded artifacts (from earlier stages of
        that same run) — this never runs anything upstream. Raises immediately if ``run_id``
        doesn't exist yet.
        """
        from .jobs import start_stage_run

        start_stage_run(run_id, stage, force=force)
        return {"run_id": run_id, "stage": stage}

    @mcp.tool()
    def get_run_status(run_id: str) -> dict[str, Any]:
        """Per-stage status/timing/artifacts/peak-mem for ``run_id`` (delegates to
        ``pipeline.api.get_status``), plus ``job_error``: a background-job exception (see
        ``mcp_server.jobs``) if this run was started via this server's own ``run_pipeline``/
        ``run_stage`` and its thread raised *before* the manifest reflected anything useful —
        ``None`` in the overwhelming common case (a stage failure is normally already visible as
        that stage's own ``"failed"`` status/``error`` field below).
        """
        from pipeline.api import get_status as _get_status

        from .jobs import job_error

        status = _get_status(run_id)
        status["job_error"] = job_error(run_id)
        return status

    @mcp.tool()
    def tail_logs(run_id: str, stage: str, max_lines: int = 200) -> dict[str, Any]:
        """The last ``max_lines`` lines of ``stage``'s own log file for ``run_id`` — what a
        currently-``"running"`` stage is actually doing, without waiting for it to finish.
        Returns an empty ``lines`` list (not an error) if the stage hasn't started writing a log
        yet.
        """
        from pipeline.artifacts import get_runs_root, stage_log_path

        path = stage_log_path(run_id, stage, runs_root=get_runs_root())
        if not path.is_file():
            return {"path": str(path), "lines": [], "line_count": 0, "truncated": False}
        text = path.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        tail = all_lines[-max_lines:] if max_lines > 0 else all_lines
        return {
            "path": str(path),
            "lines": tail,
            "line_count": len(all_lines),
            "truncated": len(all_lines) > len(tail),
        }

    @mcp.tool()
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Best-effort cancellation of an in-flight run.

        Layer 1's ``pipeline.api.cancel`` is deliberately unimplemented — T12 explicitly scoped
        cancelling a mid-flight run (interrupting a running container exec / native subprocess)
        out of its own work, and nothing since has picked it up (see
        ``planning/tasks/T12-resource-manager.md``'s scope note). Rather than surface that as an
        opaque tool error, this confirms ``run_id`` actually exists (a real 404 is still a real
        error) and reports the gap honestly: the run keeps going, it's just not stoppable from
        here yet.
        """
        from pipeline.api import cancel as _cancel
        from pipeline.artifacts import get_manifest

        get_manifest(run_id)  # a genuine 404 if run_id doesn't exist — not swallowed
        try:
            _cancel(run_id)
            return {"run_id": run_id, "cancelled": True}
        except NotImplementedError:
            return {
                "run_id": run_id,
                "cancelled": False,
                "reason": (
                    "cancellation isn't implemented in the pipeline engine yet (see T12's scope "
                    "note) — the run will continue to completion; poll get_run_status for its "
                    "outcome."
                ),
            }

    # --- artifact inspection -----------------------------------------------------------------

    @mcp.tool()
    def read_artifact(run_id: str, artifact_name: str) -> dict[str, Any]:
        """A small, JSON-safe summary of one artifact's content — never the raw file.

        Shape depends on the artifact's ``kind``: parsed content for a small ``json`` artifact,
        per-key shape/dtype/min/max/mean for an ``npz`` (not the raw arrays), vertex/face counts
        for a ``ply``, a shallow file listing for a ``dataset``/``model`` directory. See
        ``mcp_server.artifact_view.read_artifact_summary`` for the full per-kind breakdown. Use
        ``get_preview`` instead for a ``png``/``video`` artifact.
        """
        from pipeline.artifacts import get_artifact as _get_artifact

        from .artifact_view import read_artifact_summary

        return read_artifact_summary(_get_artifact(run_id, artifact_name))

    @mcp.tool()
    def get_preview(run_id: str, artifact_name: str):
        """A viewable preview of a ``png``/``video`` artifact — what lets Claude actually *see*
        results (segmentation previews, renders, amp clips), not just read paths about them.

        A ``png`` artifact comes back as an inline image, embedded directly in the tool result
        (small enough to always do this). A ``video`` artifact is **not** inlined — there's no
        standard MCP video content type, and a clip is far too large to base64 into a tool
        response — its path/size and the ``run://<run_id>/artifact/<name>`` resource URI are
        returned instead, for a client that can fetch and play it separately. Raises for any
        other artifact kind (use ``read_artifact`` instead).
        """
        from pipeline.artifacts import get_artifact as _get_artifact

        from .artifact_view import preview_kind

        artifact = _get_artifact(run_id, artifact_name)
        kind = preview_kind(artifact)  # raises ArtifactNotPreviewableError for anything else
        if kind == "image":
            return Image(path=artifact.path)

        path = Path(artifact.path)
        return {
            "kind": "video",
            "path": artifact.path,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "resource_uri": f"run://{run_id}/artifact/{artifact_name}",
            "note": "video content isn't inlined in the tool result — fetch the resource URI above.",
        }

    # --- resources (manifests / logs / artifacts) -------------------------------------------

    @mcp.resource("run://{run_id}/manifest", mime_type="application/json")
    def manifest_resource(run_id: str) -> dict[str, Any]:
        """The full status/manifest for ``run_id`` — same content as ``get_run_status`` (minus
        ``job_error``, which is this *process's* transient state, not part of the manifest
        itself), reachable as a resource for a client that prefers reading over calling a tool.
        """
        from pipeline.api import get_status as _get_status

        return _get_status(run_id)

    @mcp.resource("run://{run_id}/log/{stage}", mime_type="text/plain")
    def log_resource(run_id: str, stage: str) -> str:
        """The full (untruncated) log text for one stage of one run — ``tail_logs`` is the tool
        form of this for a quick look; fetch this resource for the whole thing.
        """
        from pipeline.artifacts import get_runs_root, stage_log_path

        path = stage_log_path(run_id, stage, runs_root=get_runs_root())
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @mcp.resource("run://{run_id}/artifact/{artifact_name}")
    def artifact_resource(run_id: str, artifact_name: str) -> bytes:
        """Raw bytes of one file-kind artifact — e.g. what ``get_preview``'s ``video`` branch
        points a client at to actually fetch the clip, or a ``png`` a client prefers to read as a
        resource rather than receiving inline from ``get_preview``. Only for file artifacts —
        raises for a ``dataset``/``model`` directory artifact (use ``read_artifact`` for those).
        """
        from pipeline.artifacts import get_artifact as _get_artifact

        artifact = _get_artifact(run_id, artifact_name)
        path = Path(artifact.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"artifact {artifact_name!r} (kind={artifact.kind!r}) isn't a single file at "
                f"{path} — directory-kind artifacts have no single-resource form; use "
                "read_artifact for a listing instead."
            )
        return path.read_bytes()

    return mcp


def build_app(settings: ServerSettings) -> ASGIApp:
    """The full ASGI app: FastMCP's streamable-HTTP app wrapped in the bearer-token gate."""

    mcp = build_mcp()
    mcp.settings.host = settings.host
    mcp.settings.port = settings.port
    inner_app = mcp.streamable_http_app()
    return BearerAuthMiddleware(inner_app, settings.token)


def main() -> None:
    """Entry point for ``python -m mcp_server``. Reads settings from the environment."""

    import uvicorn

    settings = load_settings()
    app = build_app(settings)
    print(
        f"4DGS orchestrator MCP server listening on http://{settings.host}:{settings.port}/mcp "
        "(streamable HTTP, bearer-token auth required) — see mcp_server/CONNECTING.md."
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
