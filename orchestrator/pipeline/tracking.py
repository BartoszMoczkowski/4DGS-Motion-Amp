"""T28 — local MLflow tracking integration. This module owns **all** MLflow interaction.

The run manifest (``runs/<id>/manifest.json``) stays the source of truth; MLflow is a *view*
over it, keyed by the ``pipeline.run_id`` tag (the join key). Every public function here is
degradation-safe: any failure (server down, mlflow uninstalled, store locked) is caught, logged
as a warning, and swallowed — tracking must never fail a pipeline run.

Light-import rule (``orchestrator/planning/INSTRUCTIONS.md``): **no** ``import mlflow`` at
module scope anywhere in ``pipeline/`` — all mlflow imports happen inside functions (see
:func:`_mlflow`), so Layer 1 stays importable in the sandbox and `amp --help` stays light.
``tests/test_tracking.py`` enforces this with an AST scan.

Storage layout (owner decision, locked):
    ``mlflow server --backend-store-uri sqlite:///<runs>/mlflow/mlflow.db
    --default-artifact-root <runs>/mlflow/artifacts --no-serve-artifacts
    --host 127.0.0.1 --port 5000``
The SQLite db is a single file — see ``docs/orchestrator.md`` for the backup discipline.
``runs/`` (and therefore ``runs/mlflow/``) is already gitignored.

Metrics-file contract (declaration only — T29 implements producers and the metric registry):
a stage may drop ``<runs>/<run_id>/<stage with '.'→'_'>_out/metrics.json`` shaped like
``{"metrics": {name: float}, "step": int?}``; :func:`log_stage_metrics` ingests it when
present, prefixing every key with ``stage.<name>.``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Env var overriding the tracking URI (read at call time, mirroring ``PIPELINE_RUNS_ROOT``).
#: Set to a ``file:///<dir>`` URI for tests/offline use; set ``PIPELINE_MLFLOW_DISABLE=1`` to
#: turn tracking off entirely.
TRACKING_URI_ENV = "PIPELINE_MLFLOW_TRACKING_URI"
DISABLE_ENV = "PIPELINE_MLFLOW_DISABLE"

#: Host-side default: the local server ``amp mlflow serve`` starts.
DEFAULT_HOST_URI = "http://127.0.0.1:5000"
#: Container-side default: Docker Desktop's host gateway from inside a container.
DEFAULT_CONTAINER_URI = "http://host.docker.internal:5000"

#: MLflow tag names.
TAG_RUN_ID = "pipeline.run_id"  # the join key back to runs/<id>/
TAG_PRESET = "pipeline.preset"
TAG_GIT_SHA = "git_sha"
TAG_MANIFEST_PATH = "manifest_path"
TAG_STATUS = "pipeline.status"
TAG_LOGGED_STAGES = "pipeline.logged_stages"  # CSV of stage names already logged (idempotency)

#: Metrics-file ingest contract (see module docstring).
STAGE_METRICS_FILENAME = "metrics.json"

#: Artifact policy: only small preview/summary files are logged as MLflow artifacts;
#: checkpoints/PLY stay path-referenced via the ``manifest_path`` tag — never uploaded.
ARTIFACT_SUFFIXES = frozenset({".mp4", ".png", ".jpg", ".json"})
ARTIFACT_MAX_BYTES = 50 * 1024 * 1024

#: mlflow param values are length-limited (historically 250 chars); truncate with a marker.
_PARAM_VALUE_LIMIT = 250
#: ``log_params``/``log_metrics`` batch limits; chunk to stay under.
_BATCH_LIMIT = 100


def default_tracking_uri() -> str:
    """Host-side tracking URI: ``$PIPELINE_MLFLOW_TRACKING_URI`` if set, else the local server."""
    return os.environ.get(TRACKING_URI_ENV, DEFAULT_HOST_URI)


def container_tracking_uri() -> str:
    """Container-side tracking URI — ``host.docker.internal`` reaches the host's server from
    inside a Docker Desktop container (why the container URI differs from the host URI)."""
    return os.environ.get(TRACKING_URI_ENV, DEFAULT_CONTAINER_URI)


def is_enabled() -> bool:
    """False when tracking is explicitly disabled or the URI is empty."""
    if os.environ.get(DISABLE_ENV) == "1":
        return False
    return bool(default_tracking_uri())


def _mlflow():
    """Import mlflow lazily (light-import rule) and point it at the resolved tracking URI."""
    # Windows consoles default to charmap/cp1252; mlflow prints emoji status lines
    # ("🏃 View run ...") that would otherwise raise UnicodeEncodeError mid-call.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — non-reconfigurable stream (redirected, closed)
            pass
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(default_tracking_uri())
    # Degradation budget: mlflow's default HTTP timeout is 120s — with the server down, every
    # tracking call would stall a real pipeline run for minutes. Cap it low unless the caller
    # explicitly chose a timeout.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    return mlflow


def get_mlflow_root() -> Path:
    """Where the MLflow store lives on disk — re-exported from
    :func:`pipeline.artifacts.paths.get_mlflow_root` so all MLflow-facing helpers have one
    import site.
    """
    from .artifacts.paths import get_mlflow_root as _impl

    return _impl()


def _experiment_name(preset: str, metadata: Optional[dict[str, Any]]) -> str:
    """T27 experiment runs group under ``metadata["experiment"]``; plain runs under the preset."""
    if metadata and metadata.get("experiment"):
        return str(metadata["experiment"])
    return preset


def _flatten_params(config: dict[str, Any]) -> dict[str, str]:
    """Flatten a nested resolved-config dict into ``a.b.c=value`` mlflow params (strings)."""
    flat: dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            s = json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
            if len(s) > _PARAM_VALUE_LIMIT:
                s = s[: _PARAM_VALUE_LIMIT - 14] + "...[truncated]"
            flat[prefix] = s

    _walk("", config)
    return flat


def _client_and_experiment(preset: str, metadata: Optional[dict[str, Any]]):
    mlflow = _mlflow()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    name = _experiment_name(preset, metadata)
    exp = client.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp is not None else client.create_experiment(name)
    return mlflow, client, exp_id


def _find_run(client: Any, exp_id: str, run_id: str):
    """The existing MLflow run tagged with this manifest ``run_id``, if any (idempotent reuse)."""
    runs = client.search_runs(
        [exp_id], filter_string=f'tags."{TAG_RUN_ID}" = \'{run_id}\'', max_results=1
    )
    return runs[0] if runs else None


def start_run(
    run_id: str,
    preset: str,
    resolved_config: dict[str, Any],
    metadata: Optional[dict[str, Any]],
) -> Optional[str]:
    """Start (or reuse) the MLflow run for manifest ``run_id``. Returns the MLflow run id,
    or ``None`` when tracking is disabled/unavailable — never raises."""
    if not is_enabled():
        return None
    try:
        mlflow, client, exp_id = _client_and_experiment(preset, metadata)
        existing = _find_run(client, exp_id, run_id)
        if existing is not None:
            return existing.info.run_id

        from .artifacts import get_git_sha, manifest_path

        tags = {
            TAG_RUN_ID: run_id,
            TAG_PRESET: preset,
            TAG_MANIFEST_PATH: str(manifest_path(run_id).resolve()),
        }
        sha = get_git_sha()
        if sha:
            tags[TAG_GIT_SHA] = sha
        for key in ("experiment", "cell", "variant"):
            if metadata and metadata.get(key) is not None:
                tags[f"pipeline.{key}"] = str(metadata[key])
        run = client.create_run(exp_id, run_name=run_id, tags=tags)
        mlflow_run_id = run.info.run_id

        flat = _flatten_params(resolved_config)
        for i in range(0, len(flat), _BATCH_LIMIT):
            chunk = dict(list(flat.items())[i : i + _BATCH_LIMIT])
            client.log_batch(mlflow_run_id, params=[_param(k, v) for k, v in chunk.items()])
        return mlflow_run_id
    except Exception as exc:  # noqa: BLE001 — degradation-safe by design
        logger.warning("mlflow tracking unavailable; run %r proceeds untracked: %r", run_id, exc)
        return None


def _param(key: str, value: str):
    from mlflow.entities import Param

    return Param(key, value)


def _metric(key: str, value: float, step: int):
    from mlflow.entities import Metric

    return Metric(key, float(value), timestamp=0, step=step)


def _logged_stages(client: Any, mlflow_run_id: str) -> list[str]:
    run = client.get_run(mlflow_run_id)
    raw = run.data.tags.get(TAG_LOGGED_STAGES, "")
    return [s for s in raw.split(",") if s]


def log_stage_metrics(
    run_id: str, manifest: Any, stage_order: Optional[list[str]] = None
) -> None:
    """Log every new stage's metrics/params from the manifest onto the run's MLflow run.

    Stage metrics are prefixed ``stage.<name>.`` (``wall_time_s``, ``peak_vram_mb``,
    ``peak_ram_mb``; ``None`` values skipped) with ``step`` = the stage's index in
    ``stage_order`` (the DAG's planned order — pass it whenever known, since the manifest's
    JSON serialization sorts stage keys alphabetically and is NOT the execution order;
    defaults to sorted stage names). Status/error land as tags. Already-logged stages are
    tracked via the ``pipeline.logged_stages`` MLflow run tag so repeated calls
    (``run_stage`` re-entry) stay incremental and idempotent. Also ingests each stage's
    ``metrics.json`` when present (see :data:`STAGE_METRICS_FILENAME` and the module
    docstring). Never raises.
    """
    if not is_enabled():
        return
    try:
        mlflow, client, exp_id = _client_and_experiment(manifest.preset, manifest.metadata)
        existing = _find_run(client, exp_id, run_id)
        if existing is None:
            return
        mlflow_run_id = existing.info.run_id
        done = set(_logged_stages(client, mlflow_run_id))

        from .artifacts.paths import run_dir

        metrics = []
        newly_done: list[str] = []
        order = stage_order if stage_order is not None else sorted(manifest.stages)
        for index, name in enumerate(order):
            rec = manifest.stages.get(name)
            if rec is None or name in done or rec.status in ("pending", "running"):
                continue
            prefix = f"stage.{name}."
            for field in ("wall_time_s", "peak_vram_mb", "peak_ram_mb"):
                value = getattr(rec, field)
                if value is not None:
                    metrics.append(_metric(prefix + field, value, index))
            metrics_file = (
                run_dir(run_id) / f"{name.replace('.', '_')}_out" / STAGE_METRICS_FILENAME
            )
            if metrics_file.is_file():
                try:
                    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
                    step = int(payload.get("step", index))
                    for key, value in payload.get("metrics", {}).items():
                        if not re.fullmatch(r"[A-Za-z0-9_.\-/ ]+", str(key)):
                            logger.warning(
                                "skipping odd metrics.json key %r for stage %s", key, name
                            )
                            continue
                        metrics.append(_metric(prefix + str(key), float(value), step))
                except (OSError, ValueError, TypeError) as exc:
                    logger.warning("ignoring unreadable %s: %r", metrics_file, exc)
            client.set_tag(mlflow_run_id, f"stage.{name}.status", rec.status)
            if rec.error:
                client.set_tag(mlflow_run_id, f"stage.{name}.error", rec.error[:5000])
            newly_done.append(name)

        if newly_done:
            for i in range(0, len(metrics), _BATCH_LIMIT):
                batch = metrics[i : i + _BATCH_LIMIT]
                if batch:
                    client.log_batch(mlflow_run_id, metrics=batch)
            client.set_tag(
                mlflow_run_id, TAG_LOGGED_STAGES, ",".join(sorted(done | set(newly_done)))
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow stage-metric logging failed for run %r: %r", run_id, exc)


def log_artifacts(run_id: str, manifest: Any) -> None:
    """Log each stage artifact passing the artifact policy (suffix ∈
    :data:`ARTIFACT_SUFFIXES`, size < :data:`ARTIFACT_MAX_BYTES`) as an MLflow artifact.
    Checkpoints/PLY/oversized files stay path-referenced via the ``manifest_path`` tag.
    Never raises.
    """
    if not is_enabled():
        return
    try:
        mlflow, client, exp_id = _client_and_experiment(manifest.preset, manifest.metadata)
        existing = _find_run(client, exp_id, run_id)
        if existing is None:
            return
        mlflow_run_id = existing.info.run_id

        logged_dir = f"stage_artifacts"
        seen: set[str] = set()
        for name, rec in manifest.stages.items():
            for art_name in rec.artifacts:
                if art_name in seen:
                    continue
                seen.add(art_name)
                art = manifest.artifacts.get(art_name)
                if art is None:
                    continue
                path = Path(art.path)
                if path.suffix.lower() not in ARTIFACT_SUFFIXES or not path.is_file():
                    continue
                try:
                    if path.stat().st_size >= ARTIFACT_MAX_BYTES:
                        continue
                except OSError:
                    continue
                client.log_artifact(mlflow_run_id, str(path), artifact_path=logged_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow artifact logging failed for run %r: %r", run_id, exc)


def finish_run(run_id: str, manifest: Any, status: str) -> None:
    """Set the ``pipeline.status`` tag and terminate the MLflow run when the manifest's status
    is terminal: ``failed``/``cancelled`` → MLflow ``FAILED``, ``success`` → ``FINISHED``.
    ``running``/``pending`` (e.g. a ``run_stage`` call mid-pipeline) only updates the tag and
    leaves the MLflow run active. Never raises.
    """
    if not is_enabled():
        return
    try:
        mlflow, client, exp_id = _client_and_experiment(manifest.preset, manifest.metadata)
        existing = _find_run(client, exp_id, run_id)
        if existing is None:
            return
        mlflow_run_id = existing.info.run_id
        client.set_tag(mlflow_run_id, TAG_STATUS, status)
        if status in ("failed", "cancelled"):
            client.set_terminated(mlflow_run_id, status="FAILED")
        elif status == "success":
            client.set_terminated(mlflow_run_id, status="FINISHED")
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow finish_run failed for run %r: %r", run_id, exc)


def record_backlink(run_id: str, mlflow_run_id: str, tracking_uri: Optional[str] = None) -> None:
    """Write ``metadata["mlflow"] = {"run_id": ..., "url": ...}`` into the run manifest (the
    manifest→MLflow backlink; ``metadata`` is free-form, so no RunManifest schema change).
    Never raises.
    """
    if not mlflow_run_id:
        return
    try:
        from .artifacts import update_manifest

        uri = tracking_uri or default_tracking_uri()
        exp_id = "0"
        try:
            from mlflow.tracking import MlflowClient

            _mlflow()
            exp_id = MlflowClient().get_run(mlflow_run_id).info.experiment_id
        except Exception:  # noqa: BLE001 — cosmetic detail; fall back to the generic URL
            pass
        url = f"{uri}/#/experiments/{exp_id}/runs/{mlflow_run_id}"

        def _mutate(m: Any) -> None:
            m.metadata["mlflow"] = {"run_id": mlflow_run_id, "url": url, "tracking_uri": uri}

        update_manifest(run_id, _mutate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow backlink write failed for run %r: %r", run_id, exc)


def build_serve_command(host: str = "127.0.0.1", port: int = 5000) -> list[str]:
    """The locked server command line (see module docstring). ``sys.executable -m mlflow``
    so it works regardless of console-script PATH setup."""
    import sys

    root = get_mlflow_root()
    db = (root / "mlflow.db").as_posix()
    artifacts = (root / "artifacts").as_posix()
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{db}",
        "--default-artifact-root",
        artifacts,
        "--no-serve-artifacts",
        # mlflow 3.x validates Host headers (DNS-rebinding protection); containers reach this
        # server as host.docker.internal, which is not in the default allowlist (403 without it).
        # Explicit entries are matched literally against the Host header, port included — so
        # list both bare and :port forms.
        "--allowed-hosts",
        ",".join(
            f"{h}{suffix}"
            for h in ("localhost", "127.0.0.1", "host.docker.internal")
            for suffix in ("", f":{port}")
        ),
        "--host",
        host,
        "--port",
        str(port),
    ]


def prune(older_than_days: int) -> int:
    """Hard-delete MLflow runs older than ``older_than_days`` days (and their artifact dirs
    under the artifact root). Never touches ``runs/<id>/`` manifests. Returns the count."""
    if older_than_days < 0:
        raise ValueError("older_than_days must be >= 0")
    import time
    import shutil

    from mlflow.tracking import MlflowClient

    _mlflow()  # sets the tracking URI
    client = MlflowClient()
    cutoff_ms = int((time.time() - older_than_days * 86400) * 1000)
    count = 0
    for exp in client.search_experiments():
        for run in client.search_runs(
            [exp.experiment_id], filter_string=f"attributes.end_time < {cutoff_ms}"
        ):
            info = run.info
            client.delete_run(info.run_id)
            count += 1
            # File-store artifact dirs (runs/mlflow/artifacts/<exp>/<run>/artifacts) are not
            # removed by delete_run; clean them up ourselves when under our artifact root.
            artifact_uri = getattr(info, "artifact_uri", "") or ""
            if artifact_uri.startswith("file:"):
                from mlflow.utils.file_utils import local_file_uri_to_path

                try:
                    art_dir = Path(local_file_uri_to_path(artifact_uri))
                    if art_dir.is_dir() and art_dir.resolve().is_relative_to(
                        get_mlflow_root().resolve()
                    ):
                        shutil.rmtree(art_dir, ignore_errors=True)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    logger.warning("could not remove artifact dir %s", artifact_uri)
    return count


def track_run_completion(
    run_id: str, manifest: Any = None, stage_order: Optional[list[str]] = None
) -> None:
    """Post-hoc ingestion entry point used by ``pipeline.api`` after ``run_dag``/``run_stage``
    returns (or raises): reload the manifest if not given, then log stage metrics, artifacts,
    and terminate the MLflow run with the manifest's status. ``stage_order`` is the DAG's
    planned stage order (used as metric ``step``; see :func:`log_stage_metrics`). Safe in a
    ``finally:`` block — every sub-step already swallows its own errors, and this wrapper
    does too.
    """
    try:
        if manifest is None:
            from .artifacts import load_manifest

            manifest = load_manifest(run_id)
        log_stage_metrics(run_id, manifest, stage_order)
        log_artifacts(run_id, manifest)
        finish_run(run_id, manifest, manifest.status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow completion tracking failed for run %r: %r", run_id, exc)


def server_status() -> dict[str, Any]:
    """For ``amp mlflow status``: resolved URI, whether a server is reachable, run count."""
    uri = default_tracking_uri()
    info: dict[str, Any] = {"tracking_uri": uri, "enabled": is_enabled(), "reachable": False,
                            "run_count": None, "mlflow_root": str(get_mlflow_root())}
    if not info["enabled"]:
        return info
    try:
        from mlflow.tracking import MlflowClient

        _mlflow()
        client = MlflowClient()
        exps = client.search_experiments()
        total = 0
        for exp in exps:
            total += len(client.search_runs([exp.experiment_id], max_results=50000))
        info["reachable"] = True
        info["run_count"] = total
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info
