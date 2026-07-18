"""GPU/Isaac-Sim integration tests for T11 — the runnable counterpart to that task's own
acceptance criteria (`planning/tasks/T11-wrap-isaac-stages.md`):

  1. `run_capture.sh`'s smoke test (`--n-cameras 2 --frames 2`) reproduced via `run_stage`.
  2. A full run from a raw asset through amp completes; artifacts + manifest populated; warm
     Isaac container avoids repeated cold-start.

Same story as `test_containers_gpu.py` (T08): these talk to a *real* Docker daemon and a real,
already-pulled `isaac` image, so they can't run in the sandbox and auto-skip there. Run for real
on Bartosz's machine, with Docker Desktop + GPU support set up and `planning/WINDOWS_SETUP.md`'s
step 8 done (`trimesh` installed in the `isaac` container):

    cd orchestrator
    $env:PIPELINE_TEST_ISAAC = "1"; pytest -q -s tests/test_stages_isaac_gpu.py

(Not gated separately from Isaac like `test_containers_gpu.py`'s cuda-only checks are — every
test in this file needs the `isaac` image, so `PIPELINE_TEST_ISAAC=1` gates the whole module.)

Needs real Omniverse assets on disk (relative to `PIPELINE_ASSETS_ROOT`, default `Q:/Omniverse`,
per `pipeline.paths`):

  - The pre-existing animated pump (`assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd`, override
    via `PIPELINE_TEST_ANIMATED_MESH`) for criterion 1's capture-only smoke test — the exact asset
    `omniverse_pipeline/.devcontainer/run_capture.sh` itself asserts exists.
  - The raw, unsplit fused mesh (`assets/pump_radnom/CONJUNTO BOMBAS.usd` — note the literal space
    in the filename, matching `split_mesh.py`'s own docstring example and
    `.claude_notes/NOTES_omniverse_pipeline.md`; override via `PIPELINE_TEST_RAW_MESH`) for
    criterion 2's full prep-through-amp chain.

Each test skips independently, with a clear reason, if its own asset isn't found at the expected
path — one missing file doesn't block the other test. A separate `trimesh`-importability check
runs first so a missing `pip install` (WINDOWS_SETUP.md step 8.1) fails fast and legibly instead
of surfacing as a confusing mid-script `ImportError` from inside `split_mesh.py`.

**Criterion 2's run is deliberately trimmed** (`capture.rig.n_cameras`/`capture.capture.
num_frames`/`optim.iterations`/`optim.coarse_iterations`/`seg_extract.n_times`, all cut to a
handful) — a real multi-hour 4DGS training run isn't what "does the wiring work end to end" needs
to prove, and the task's own acceptance criteria only ask for the run to *complete*, not to
produce a publication-quality reconstruction. These are first-guess values, not validated against
a real training run (this logic was written but never executed on real hardware before today, per
every other GPU-touching task's own honest status note) — expect to adjust them if training
chokes on too-small iteration counts. `seg_eval.default` is deliberately excluded from the stage
list: it needs an external `gt_segmentation` artifact this project never wires up automatically
(a pre-existing T07-era gap, see `pipeline.stages.seg_eval`'s docstring) and isn't part of T11's
own acceptance criteria ("prep through amp"), so pulling it in would only add an unrelated
manual-asset requirement to this test.

Each run writes a real run directory under the repo's own `runs/` (no `runs_root` override — same
place a real `pipeline.api.run_pipeline` call would write), same as any real pipeline invocation;
not auto-cleaned up by this test (unlike `test_containers_gpu.py`'s managed-container teardown —
there's no equivalent "stop and remove" operation for a run directory, and leaving it is exactly
what a real run would do too).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


def _docker_reachable() -> bool:
    try:
        import docker  # local import — this module must stay importable with no `docker` package
    except ImportError:
        return False
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


_RUN_ISAAC = os.environ.get("PIPELINE_TEST_ISAAC") == "1"

pytestmark = [
    pytest.mark.skipif(
        not _docker_reachable(),
        reason=(
            "no reachable Docker daemon — real-GPU/real-Isaac integration tests for Bartosz's "
            "Windows + Docker Desktop machine, not the sandbox/CI"
        ),
    ),
    pytest.mark.skipif(
        not _RUN_ISAAC,
        reason="set PIPELINE_TEST_ISAAC=1 to run these (needs the isaac image + real assets)",
    ),
]


def _animated_mesh_path() -> Path:
    from pipeline.paths import get_roots

    override = os.environ.get("PIPELINE_TEST_ANIMATED_MESH")
    if override:
        return Path(override)
    # matches capture_config_pump.yaml's / pump01.yaml's own usd_path convention: assets live one
    # level deeper than the assets root itself, under "assets/pump_radnom/...".
    return get_roots().assets_root_host / "assets" / "pump_radnom" / "CONJUNTO_BOMBAS_animated.usd"


def _raw_mesh_path() -> Path:
    from pipeline.paths import get_roots

    override = os.environ.get("PIPELINE_TEST_RAW_MESH")
    if override:
        return Path(override)
    return get_roots().assets_root_host / "assets" / "pump_radnom" / "CONJUNTO BOMBAS.usd"


# --- 0. trimesh sanity check (WINDOWS_SETUP.md step 8.1) ----------------------------------------


def test_trimesh_is_importable_in_the_isaac_container(tmp_path):
    from pipeline.containers import exec_in_container

    log_path = tmp_path / "trimesh_check.log"
    result = exec_in_container(
        "isaac",
        ["/isaac-sim/python.sh", "-c", "import trimesh; print('trimesh ok')"],
        log_path=log_path,
    )
    output = log_path.read_text() if log_path.is_file() else ""
    assert result.exit_code == 0, (
        f"trimesh not importable in the isaac container (needed by prep_split.default's "
        f"split_mesh.py) -- run `/isaac-sim/python.sh -m pip install trimesh` inside it first, "
        f"see planning/WINDOWS_SETUP.md's step 8.1. Output:\n{output}"
    )
    assert "trimesh ok" in output


# --- 1. run_capture.sh's smoke test, reproduced via run_stage -----------------------------------


def test_capture_isaac_smoke_reproduces_run_capture_sh(tmp_path):
    animated = _animated_mesh_path()
    if not animated.is_file():
        pytest.skip(f"animated pump not found at {animated} -- set PIPELINE_TEST_ANIMATED_MESH")

    from pipeline.api import _stage_config_for
    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.config import validate_config
    from pipeline.dag import run_dag

    resolved = validate_config("pump01").model_dump()
    # The literal --n-cameras 2 --frames 2 smoke-test overrides run_capture.sh's own docstring
    # names as its example invocation.
    resolved["capture"]["rig"]["n_cameras"] = 2
    resolved["capture"]["capture"]["num_frames"] = 2

    stage_names = ["capture.isaac"]
    stage_configs = {"capture.isaac": _stage_config_for("capture.isaac", resolved)}
    run_id = f"t11-capture-smoke-{int(time.time())}"

    create_run(run_id, "pump01", resolved, stage_names=stage_names)
    update_manifest(
        run_id,
        lambda m: m.artifacts.update(
            {
                "animated_mesh": Artifact(
                    name="animated_mesh", kind="usd", path=str(animated), producing_stage="external"
                )
            }
        ),
    )

    t0 = time.time()
    manifest = run_dag(run_id, stage_names, resolved, preset="pump01", stage_configs=stage_configs)
    elapsed = time.time() - t0
    print(f"\ncapture.isaac smoke run ({run_id}) took {elapsed:.1f}s")

    assert manifest.status == "success", manifest.stages["capture.isaac"].error

    capture_dir = Path(manifest.artifacts["capture"].path)
    gt = json.loads((capture_dir / "cameras_gt.json").read_text())
    assert len(gt["cameras"]) == 2  # --n-cameras 2
    cam_dirs = sorted(p.name for p in capture_dir.iterdir() if p.is_dir() and p.name.startswith("cam"))
    assert len(cam_dirs) == 2  # --frames 2 -> 2 captured cameras' worth of output dirs


# --- 2. full prep-through-amp chain, from a raw asset -------------------------------------------


FULL_CHAIN_STAGES = [
    "prep_split.default",
    "prep_motion.default",
    "capture.isaac",
    "convert.default",
    "train.default",
    "render.default",
    "seg_extract.default",
    "segment.rigid",
    "amp.default",
]


def test_pump01_prep_through_amp_completes(tmp_path):
    raw_mesh = _raw_mesh_path()
    if not raw_mesh.is_file():
        pytest.skip(f"raw fused mesh not found at {raw_mesh} -- set PIPELINE_TEST_RAW_MESH")

    from pipeline.api import _stage_config_for
    from pipeline.artifacts import Artifact, create_run, update_manifest
    from pipeline.config import validate_config
    from pipeline.containers import start_container
    from pipeline.dag import run_dag

    resolved = validate_config("pump01").model_dump()
    # Trimmed for a fast smoke pass, not a real reconstruction -- see module docstring.
    resolved["capture"]["rig"]["n_cameras"] = 3
    resolved["capture"]["capture"]["num_frames"] = 4
    resolved["capture"]["capture"]["rt_subframes"] = 4
    resolved["prep_motion"]["num_frames"] = 4
    resolved["optim"]["coarse_iterations"] = 50
    resolved["optim"]["iterations"] = 100
    resolved["train"]["test_iterations"] = []
    resolved["train"]["save_iterations"] = []
    resolved["seg_extract"]["n_times"] = 4

    stage_configs = {name: _stage_config_for(name, resolved) for name in FULL_CHAIN_STAGES}
    run_id = f"t11-full-smoke-{int(time.time())}"

    create_run(run_id, "pump01", resolved, stage_names=FULL_CHAIN_STAGES)
    update_manifest(
        run_id,
        lambda m: m.artifacts.update(
            {
                "raw_mesh": Artifact(
                    name="raw_mesh", kind="usd", path=str(raw_mesh), producing_stage="external"
                )
            }
        ),
    )

    t0 = time.time()
    manifest = run_dag(
        run_id, FULL_CHAIN_STAGES, resolved, preset="pump01", stage_configs=stage_configs
    )
    elapsed = time.time() - t0
    print(f"\nfull prep-through-amp smoke run ({run_id}) took {elapsed:.1f}s")

    failures = {
        name: manifest.stages[name].error
        for name in FULL_CHAIN_STAGES
        if manifest.stages[name].status == "failed"
    }
    assert manifest.status == "success", failures
    for name in FULL_CHAIN_STAGES:
        # "skipped" (cross-run cache hit, `pipeline.dag.cache`) is just as valid a terminal state
        # as "success" here -- found 2026-07-19 when an *earlier* successful run's
        # prep_split.default/prep_motion.default/capture.isaac/train.default got correctly reused
        # via cache on this run (their inputs/config hadn't changed), and this assertion's original
        # `== "success"` wording failed on that, even though `manifest.status == "success"` above
        # (which already treats "skipped" as fine) had just passed. Re-running this test against
        # unchanged upstream stages is exactly when caching is supposed to kick in.
        status = manifest.stages[name].status
        assert status in ("success", "skipped"), f"{name}: {status} / {manifest.stages[name].error}"

    assert Path(manifest.artifacts["amp_video"].path).is_file()

    # Criterion 2's "warm Isaac container avoids repeated cold-start" -- printed, not hard-asserted
    # against a portable threshold (same call as test_containers_gpu.py's own warm-reuse check).
    t1 = time.time()
    start_container("isaac")
    print(f"\nisaac container re-start right after the full run took {time.time() - t1:.3f}s")
