"""Builds the FastMCP app + auth wrapper, and a ``python -m mcp_server`` entry point.

T30 MCP consolidation:
Intent-level domain-grouped tools (≤12 tools total), cutting down LLM round-trips:
- `explore_experiment`: unified single-call navigator for experiments, runs, captures, and workspace overview.
- `run_experiment`: launches declarative experiment specs or presets with auto-seeded external artifacts.
- `run_stage`: executes a single stage within an existing run.
- `get_run_status`: per-stage progress, timing, peak VRAM/RAM, MLflow URLs, and optional stage log tailing.
- `list_runs`: quick run summary list.
- `cancel_run`: real cancellation (container stop, manifest cancellation, MLflow update).
- `get_run_artifact`: JSON-safe summaries of artifacts (never big binary blobs).
- `get_run_preview`: visual previews (FastMCP Image for PNG or typed VideoPreview for videos).
- `get_run_report`: report bundle location (T29/T30), index markdown, and MLflow URL.
- `manage_preset`: list, validate, or save new preset variants extending base presets.
- `gpu_status`: machine VRAM/RAM headroom.
- `manage_container`: list, start, or stop managed Docker containers.

Resources:
- `run://{run_id}/manifest`: raw JSON manifest.
- `run://{run_id}/log/{stage}`: full raw stage log.
- `run://{run_id}/artifact/{artifact_name}`: raw bytes of a file artifact.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP, Image
from starlette.types import ASGIApp

from pipeline.artifacts import Artifact
from pipeline.paths import Env

from .artifact_view import VideoPreview
from .auth import BearerAuthMiddleware
from .config import ServerSettings, load_settings


def build_mcp() -> FastMCP:
    """A fresh :class:`FastMCP` instance with the consolidated T30 tool/resource set."""

    mcp = FastMCP(
        "4dgs-pipeline",
        instructions=(
            "Layer 2 MCP server for the 4DGS motion-amp orchestrator.\n"
            "Conventions:\n"
            "1. Navigator First: Use explore_experiment() as your primary inspection tool. It returns "
            "complete provenance (experiment spec, linked runs, config diffs, stage timelines, metrics, "
            "and MLflow URLs) in ONE call, eliminating multiple sequential list/view calls.\n"
            "2. Experiments: Experiment specs live under experiments/*.yaml. Run IDs follow "
            "'<experiment>-<variant>-<8hex>' (e.g. pump_grid_v2-pump_A20mm_M2-9104a031) or '<preset>-<8hex>' "
            "(e.g. pump01-15e286ff).\n"
            "3. Status Before Artifacts: Check get_run_status() before requesting previews or artifacts to "
            "verify that the producing stage has finished successfully.\n"
            "4. Robust Execution: run_experiment() automatically discovers and seeds required external "
            "artifacts (such as raw USD meshes, capture datasets, and ground-truth segmentation) and validates configs.\n"
            "5. Tracking: All runs log to local MLflow (default http://127.0.0.1:5000); inspect backlinks in status "
            "or explore_experiment payloads."
        ),
        stateless_http=True,
    )

    # --- 1. unified navigator -------------------------------------------------------------

    @mcp.tool()
    def explore_experiment(
        target: Optional[str] = None,
        view: str = "default",
        include_unregistered: bool = False,
    ) -> dict[str, Any]:
        """Unified experiment, run, capture, and workspace inspection tool (T30).

        Returns complete provenance in ONE call, eliminating 10-30 sequential tool round-trips:
        - When ``target`` is omitted / None: returns workspace overview (active experiments, recent runs,
          status breakdown, available capture datasets).
        - When ``target`` is an experiment (e.g. ``"pump_grid_v2"``): returns spec details, all cells,
          capture paths, linked run IDs, stage statuses, and pivoted metrics (ARI, IoU).
        - When ``target`` is a run ID (e.g. ``"pump01-15e286ff"``): returns preset, metadata, config diff
          against base, stage timeline with peak VRAM/time, all registered artifacts, metrics, and
          optional un-manifested file detection.
        - When ``target`` is a capture / dataset path: returns frame count, camera list, sync residuals,
          poses-bounds stats, and COLMAP status.

        Args:
            target: Experiment name/spec, run ID, or capture directory path. Omit for workspace overview.
            view: Detail level ('default', 'full', 'diff', or 'metrics').
            include_unregistered: If True, scans the run directory for files not registered in the manifest.

        Examples:
            explore_experiment()
            explore_experiment(target="pump_grid_v2")
            explore_experiment(target="pump_grid_v2-pump_A20mm_M2-9104a031", view="full")
            explore_experiment(target="data/bouncingballs")
        """
        from pipeline.api import explore as _explore

        return _explore(target=target, view=view, include_unregistered=include_unregistered)

    # --- 2. execution ---------------------------------------------------------------------

    @mcp.tool()
    def run_experiment(
        target: str,
        stages: Optional[list[str]] = None,
        overrides: Optional[dict[str, Any]] = None,
        external_artifacts: Optional[dict[str, Artifact]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Launch an experiment spec (e.g. ``"pump_grid_v2"``) OR a preset with auto-seeded external artifacts.

        Returns a job/run ID **immediately** (asynchronous background execution; poll ``get_run_status``).

        Unlike raw pipeline execution, this tool handles external-artifact seeding internally:
        it checks and auto-locates required external inputs (like ``raw_mesh``, ``capture``, ``scene``,
        and ``gt_segmentation``). If any required asset is missing, it raises an actionable structured
        error immediately rather than failing inside an opaque background thread.

        Args:
            target: An experiment spec name (e.g. "pump_grid_v2") or config preset (e.g. "pump01").
            stages: Optional restricted list of stages to run (e.g. ["train.default", "render.default"]).
            overrides: Optional nested config overrides dict.
            external_artifacts: Optional pre-constructed Artifact objects for external inputs.
            force: If True, forces re-execution of stages regardless of cache.

        Examples:
            run_experiment(target="pump_grid_v2")
            run_experiment(target="pump01", stages=["train.default", "render.default"])
        """
        from .jobs import start_experiment_run

        run_id = start_experiment_run(
            target,
            stages=stages,
            overrides=overrides,
            external_artifacts=external_artifacts,
            force=force,
        )
        return {"run_id": run_id, "target": target, "status": "started"}

    @mcp.tool()
    def run_stage(
        run_id: str,
        stage: str,
        force: bool = False,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a single stage (e.g. ``"render.default"``) within an existing ``run_id``.

        Returns immediately (asynchronous background execution). The stage's declared inputs must
        already be present in ``run_id``'s recorded artifacts.

        Args:
            run_id: The run ID to execute within.
            stage: Stage name (e.g. "render.default", "segment.rigid", "seg_eval.default").
            force: Force rerun even if cached.
            overrides: Optional stage-scoped config overrides.

        Example:
            run_stage(run_id="pump01-15e286ff", stage="render.default")
        """
        from .jobs import start_stage_run

        start_stage_run(run_id, stage, force=force)
        return {"run_id": run_id, "stage": stage, "status": "started"}

    # --- 3. monitoring & status -----------------------------------------------------------

    @mcp.tool()
    def get_run_status(
        run_id: str,
        include_logs: bool = False,
        stage_log: Optional[str] = None,
        max_log_lines: int = 100,
    ) -> dict[str, Any]:
        """Per-stage status, execution progress, peak VRAM/RAM, MLflow URLs, and optional stage logs.

        Args:
            run_id: Target run ID.
            include_logs: If True and stage_log is given (or latest running stage), attaches log tail.
            stage_log: Specific stage whose log lines should be tailed.
            max_log_lines: Maximum log lines to return (default: 100).

        Example:
            get_run_status(run_id="pump01-15e286ff", include_logs=True, stage_log="train.default")
        """
        from pipeline.api import get_status as _get_status
        from pipeline.artifacts import get_runs_root, stage_log_path

        from .jobs import job_error

        status = _get_status(run_id)
        status["job_error"] = job_error(run_id)

        # Attach MLflow URL if present
        from pipeline.artifacts import get_manifest

        try:
            m = get_manifest(run_id)
            status["mlflow_url"] = m.metadata.get("mlflow", {}).get("url")
        except Exception:
            pass

        if include_logs or stage_log:
            target_stage = stage_log
            if not target_stage:
                for sname, srec in status.get("stages", {}).items():
                    if srec.get("status") in ("running", "failed"):
                        target_stage = sname
                        break
            if target_stage:
                lpath = stage_log_path(run_id, target_stage, runs_root=get_runs_root())
                if lpath.is_file():
                    lines = lpath.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = lines[-max_log_lines:] if max_log_lines > 0 else lines
                    status["log_tail"] = {
                        "stage": target_stage,
                        "lines": tail,
                        "line_count": len(lines),
                    }
        return status

    @mcp.tool()
    def list_runs(limit: int = 25) -> list[dict[str, Any]]:
        """List known runs (most recently updated first) with status, preset, and timestamps.

        Args:
            limit: Maximum runs to return (default: 25).
        """
        from pipeline.api import list_runs as _list_runs

        return _list_runs()[:limit]

    @mcp.tool()
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Safely cancel an in-flight pipeline run (T17/T30).

        Stops running Docker containers (the verified fast cancellation approach), marks the
        run manifest and stages as 'cancelled', and updates MLflow tracking.

        Args:
            run_id: Target run ID.
        """
        from pipeline.api import cancel as _cancel

        return _cancel(run_id)

    # --- 4. artifacts & previews ----------------------------------------------------------

    @mcp.tool()
    def get_run_artifact(run_id: str, artifact_name: str) -> dict[str, Any]:
        """A small, JSON-safe summary of one artifact's content — never raw giant binary blobs.

        Summarizes based on kind:
        - json: parsed structured data (inlined up to 64KB).
        - npz: per-array shape, dtype, min, max, mean, nan_count.
        - ply: vertex and face counts parsed from PLY header.
        - dataset/model: directory listing with entry sizes and file counts.

        Args:
            run_id: Target run ID.
            artifact_name: Name of artifact (from explore_experiment or get_run_status).
        """
        from pipeline.artifacts import get_artifact as _get_artifact

        from .artifact_view import read_artifact_summary

        artifact = _get_artifact(run_id, artifact_name)
        return read_artifact_summary(artifact)

    @mcp.tool()
    def get_run_preview(run_id: str, artifact_name: str):
        """Visual preview of a previewable artifact (PNG image or video clip).

        - PNG images: returns an inline FastMCP Image embedded directly in the tool response.
        - MP4 video clips: returns a typed VideoPreview pointer containing path, size, and resource URI.

        Raises an error for non-previewable kinds (use get_run_artifact for NPZ/PLY/JSON/dataset).

        Args:
            run_id: Target run ID.
            artifact_name: Artifact name (e.g. "comparison_png", "amp_video", "render_video").
        """
        from pipeline.artifacts import get_artifact as _get_artifact
        from pipeline.artifacts import resolve_servable_artifact_path

        from .artifact_view import VideoPreview, preview_kind

        artifact = _get_artifact(run_id, artifact_name)
        kind = preview_kind(artifact)
        path = resolve_servable_artifact_path(artifact.path)

        if kind == "image":
            return Image(path=str(path))

        return VideoPreview(
            kind="video",
            path=artifact.path,
            size_bytes=path.stat().st_size if path.is_file() else None,
            resource_uri=f"run://{run_id}/artifact/{artifact_name}",
        )

    @mcp.tool()
    def get_run_report(run_id: str) -> dict[str, Any]:
        """Fetch report bundle info (T29/T30) and MLflow tracking URL for ``run_id``.

        Returns the report bundle path (``runs/<run_id>/report/``), index markdown text,
        generated chart names, and direct MLflow run link.

        Args:
            run_id: Target run ID.
        """
        from pipeline.artifacts import get_manifest, get_runs_root, run_dir

        m = get_manifest(run_id)
        r_dir = run_dir(run_id, runs_root=get_runs_root())
        rep_dir = r_dir / "report"

        mlflow_info = m.metadata.get("mlflow", {})
        mlflow_url = mlflow_info.get("url")

        index_path = rep_dir / "index.md"
        index_md = None
        if index_path.is_file():
            try:
                index_md = index_path.read_text(encoding="utf-8")
            except Exception:
                pass

        charts = [p.name for p in rep_dir.glob("*.png")] if rep_dir.is_dir() else []

        return {
            "run_id": run_id,
            "report_dir": str(rep_dir),
            "exists": rep_dir.is_dir(),
            "has_index": index_path.is_file(),
            "index_markdown": index_md,
            "charts": charts,
            "mlflow_url": mlflow_url,
        }

    # --- 5. presets & config management ---------------------------------------------------

    @mcp.tool()
    def manage_preset(
        action: Literal["list", "validate", "save"] = "list",
        preset: Optional[str] = None,
        name: Optional[str] = None,
        base_preset: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Inspect, validate, or create configuration presets (T30).

        Actions:
        - action='list': returns list of all available presets.
        - action='validate': resolves and type-checks ``preset``, returning fully resolved config dict.
        - action='save': saves a new preset YAML ``name`` extending ``base_preset`` with ``overrides``.

        Examples:
            manage_preset(action="list")
            manage_preset(action="validate", preset="pump01")
            manage_preset(action="save", name="pump01_high_amp", base_preset="pump01",
                          overrides={"amp": {"channels": {"pos3d": {"factor": 5.0}}}})
        """
        from pipeline.api import list_presets as _list_presets
        from pipeline.api import save_preset_variant as _save_variant
        from pipeline.api import validate_config as _validate_config

        if action == "list":
            return {"presets": _list_presets()}
        if action == "validate":
            if not preset:
                raise ValueError("action='validate' requires 'preset'")
            return {"preset": preset, "config": _validate_config(preset)}
        if action == "save":
            if not name or not base_preset:
                raise ValueError("action='save' requires 'name' and 'base_preset'")
            path = _save_variant(name, base_preset, overrides=overrides, overwrite=overwrite)
            return {"name": name, "path": path, "saved": True}
        raise ValueError(f"unknown action: {action!r}")

    # --- 6. machine & container control ---------------------------------------------------

    @mcp.tool()
    def gpu_status() -> dict[str, Any]:
        """Current VRAM and RAM allocation, total limits, and free headroom on the machine."""
        from pipeline.api import gpu_status as _gpu_status

        return _gpu_status()

    @mcp.tool()
    def manage_container(
        action: Literal["list", "start", "stop"] = "list",
        env: Optional[Env] = None,
        container_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Inspect or control managed Docker containers ('cuda' or 'isaac') (T30).

        Actions:
        - action='list': returns all running/exited pipeline-managed containers.
        - action='start': starts (or warms up) a container for ``env`` ('cuda' or 'isaac').
        - action='stop': stops a container by container_id.

        Examples:
            manage_container(action="list")
            manage_container(action="start", env="cuda")
            manage_container(action="stop", container_id="pipeline-cuda")
        """
        from pipeline.api import list_containers as _list_containers
        from pipeline.api import start_container as _start_container
        from pipeline.api import stop_container as _stop_container

        if action == "list":
            return {"containers": _list_containers()}
        if action == "start":
            if not env:
                raise ValueError("action='start' requires 'env' ('cuda' or 'isaac')")
            cid = _start_container(env)
            return {"env": env, "container_id": cid, "started": True}
        if action == "stop":
            if not container_id:
                raise ValueError("action='stop' requires 'container_id'")
            _stop_container(container_id)
            return {"container_id": container_id, "stopped": True}
        raise ValueError(f"unknown action: {action!r}")

    # --- Resources (unchanged, 100% backwards-compatible) ---------------------------------

    @mcp.resource("run://{run_id}/manifest", mime_type="application/json")
    def manifest_resource(run_id: str) -> dict[str, Any]:
        """Raw JSON manifest for ``run_id``."""
        from pipeline.api import get_status as _get_status

        return _get_status(run_id)

    @mcp.resource("run://{run_id}/log/{stage}", mime_type="text/plain")
    def log_resource(run_id: str, stage: str) -> str:
        """Full untruncated stage log text."""
        from pipeline.artifacts import get_runs_root, stage_log_path

        path = stage_log_path(run_id, stage, runs_root=get_runs_root())
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @mcp.resource("run://{run_id}/artifact/{artifact_name}")
    def artifact_resource(run_id: str, artifact_name: str) -> bytes:
        """Raw bytes of a single file artifact."""
        from pipeline.artifacts import get_artifact as _get_artifact
        from pipeline.artifacts import resolve_servable_artifact_path

        artifact = _get_artifact(run_id, artifact_name)
        path = resolve_servable_artifact_path(artifact.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"artifact {artifact_name!r} (kind={artifact.kind!r}) isn't a single file at {path}"
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
