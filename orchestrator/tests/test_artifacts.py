"""Tests for T03 (artifact store & run manifest).

Covers the task's acceptance criteria directly: a hand-constructed run dir round-trips through
the manifest reader/writer, concurrent writes never produce a torn file (write-temp-rename), and
a corrupt/partial manifest is handled gracefully (a clear exception type, not a crash). Also
covers the query helpers (`list_runs`/`get_manifest`/`list_artifacts`/`get_artifact`), hashing,
and that `pipeline.api`'s stubs delegate here correctly.
"""

from __future__ import annotations

import json
import threading

import pytest


def test_create_run_and_manifest_round_trip(tmp_path):
    from pipeline.artifacts import create_run, load_manifest

    manifest = create_run(
        "run001",
        "pump01",
        {"name": "pump01"},
        stage_names=["prep.split", "train"],
        runs_root=tmp_path,
    )
    assert manifest.status == "pending"
    assert set(manifest.stages) == {"prep.split", "train"}

    loaded = load_manifest("run001", runs_root=tmp_path)
    assert loaded == manifest
    assert (tmp_path / "run001" / "manifest.json").is_file()
    assert (tmp_path / "run001" / "config_snapshot.json").is_file()
    assert (tmp_path / "run001" / "logs").is_dir()


def test_hand_constructed_run_dir_round_trips(tmp_path):
    """A run dir built by hand (not via create_run) still reads back correctly — the task's
    acceptance criterion #1."""
    from pipeline.artifacts import Artifact, RunManifest, StageRecord, load_manifest, save_manifest

    manifest = RunManifest(
        run_id="handmade",
        preset="base",
        resolved_config={"name": "base"},
        git_sha="deadbeef",
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:00+00:00",
        status="success",
        stages={"convert": StageRecord(status="success", wall_time_s=1.5)},
        artifacts={
            "scene.npz": Artifact(
                name="scene.npz", kind="npz", path="/tmp/scene.npz", producing_stage="convert"
            )
        },
    )
    (tmp_path / "handmade").mkdir()
    save_manifest(manifest, runs_root=tmp_path)

    loaded = load_manifest("handmade", runs_root=tmp_path)
    assert loaded == manifest


def test_stage_start_and_result_lifecycle(tmp_path):
    from pipeline.artifacts import (
        Artifact,
        create_run,
        load_manifest,
        record_stage_result,
        record_stage_start,
    )

    create_run("run002", "base", {"name": "base"}, stage_names=["convert"], runs_root=tmp_path)
    record_stage_start("run002", "convert", runs_root=tmp_path)

    manifest = load_manifest("run002", runs_root=tmp_path)
    assert manifest.stages["convert"].status == "running"
    assert manifest.status == "running"

    art = Artifact(name="scene.npz", kind="npz", path="/tmp/scene.npz", producing_stage="convert")
    record_stage_result("run002", "convert", status="success", artifacts=[art], runs_root=tmp_path)

    manifest = load_manifest("run002", runs_root=tmp_path)
    rec = manifest.stages["convert"]
    assert rec.status == "success"
    assert rec.wall_time_s is not None and rec.wall_time_s >= 0
    assert rec.artifacts == ["scene.npz"]
    assert manifest.artifacts["scene.npz"].path == "/tmp/scene.npz"
    assert manifest.status == "success"  # all stages terminal + successful


def test_failed_stage_marks_run_failed(tmp_path):
    from pipeline.artifacts import create_run, load_manifest, record_stage_result

    create_run("run003", "base", {}, stage_names=["train"], runs_root=tmp_path)
    record_stage_result("run003", "train", status="failed", error="boom", runs_root=tmp_path)

    manifest = load_manifest("run003", runs_root=tmp_path)
    assert manifest.stages["train"].status == "failed"
    assert manifest.stages["train"].error == "boom"
    assert manifest.status == "failed"


def test_corrupt_manifest_raises_manifest_corrupt_error(tmp_path):
    from pipeline.artifacts import ManifestCorruptError, create_run, load_manifest

    create_run("run004", "base", {}, runs_root=tmp_path)
    (tmp_path / "run004" / "manifest.json").write_text("{not valid json")

    with pytest.raises(ManifestCorruptError):
        load_manifest("run004", runs_root=tmp_path)


def test_schema_invalid_manifest_raises_manifest_corrupt_error(tmp_path):
    from pipeline.artifacts import ManifestCorruptError, create_run, load_manifest

    create_run("run005", "base", {}, runs_root=tmp_path)
    path = tmp_path / "run005" / "manifest.json"
    data = json.loads(path.read_text())
    data["status"] = "not-a-real-status"
    path.write_text(json.dumps(data))

    with pytest.raises(ManifestCorruptError):
        load_manifest("run005", runs_root=tmp_path)


def test_missing_manifest_raises_file_not_found(tmp_path):
    from pipeline.artifacts import load_manifest

    with pytest.raises(FileNotFoundError):
        load_manifest("does-not-exist", runs_root=tmp_path)


def test_concurrent_writes_never_produce_a_torn_file(tmp_path):
    """Hammer save_manifest/load_manifest from many threads at once — the task's acceptance
    criterion #2. Every read observed mid-storm must be valid JSON / a valid RunManifest, i.e.
    write-temp-rename is atomic under concurrency."""
    from pipeline.artifacts import create_run, load_manifest, save_manifest

    manifest = create_run("run006", "base", {}, stage_names=["convert"], runs_root=tmp_path)

    errors: list[Exception] = []
    stop = threading.Event()

    def writer(i: int) -> None:
        m = manifest.model_copy(deep=True)
        for n in range(50):
            m.updated_at = f"2026-07-13T00:{i:02d}:{n:02d}+00:00"
            try:
                save_manifest(m, runs_root=tmp_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def reader() -> None:
        while not stop.is_set():
            try:
                load_manifest("run006", runs_root=tmp_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert not errors
    load_manifest("run006", runs_root=tmp_path)  # sanity: still valid after the storm


def test_replace_with_retry_recovers_from_transient_permission_error(tmp_path, monkeypatch):
    """Direct unit test for the 2026-07-19 Windows fix -- `test_concurrent_writes_never_produce_a_
    torn_file` failed for real on Bartosz's machine with a raw `PermissionError(13, 'Access is
    denied')` from `os.replace` (Windows's mandatory file locking can transiently deny a rename
    over a path another thread has open, unlike POSIX). This sandbox can't reproduce the real race
    (Linux `os.replace` doesn't have this failure mode), so this fakes `os.replace` itself to
    exercise the retry loop deterministically."""
    from pipeline.artifacts import manifest as manifest_mod

    calls = {"n": 0}
    real_replace = manifest_mod.os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(manifest_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(manifest_mod.time, "sleep", lambda s: None)  # don't actually wait in tests

    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")
    manifest_mod._replace_with_retry(str(src), dst)

    assert calls["n"] == 3
    assert dst.read_text() == "hello"
    assert not src.exists()


def test_replace_with_retry_reraises_after_exhausting_attempts(tmp_path, monkeypatch):
    from pipeline.artifacts import manifest as manifest_mod

    def always_denied(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(manifest_mod.os, "replace", always_denied)
    monkeypatch.setattr(manifest_mod.time, "sleep", lambda s: None)

    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")
    with pytest.raises(PermissionError):
        manifest_mod._replace_with_retry(str(src), dst)


def test_lock_for_is_identical_per_path_and_distinct_across_paths(tmp_path):
    """2026-07-19 follow-up fix: `_replace_with_retry` alone wasn't enough under real concurrent
    load on Windows (both writers and readers hit transient sharing-violation `PermissionError`s
    faster than the retry budget could absorb) -- `save_manifest`/`load_manifest` now serialize
    through a per-path lock instead, which deterministically prevents this process's own threads
    from ever having the same manifest path open at once. This is the direct unit test for that
    registry's own correctness (same lock object for the same path, so callers actually block on
    each other; distinct objects for different paths, so unrelated runs never contend)."""
    from pipeline.artifacts import manifest as manifest_mod

    a = tmp_path / "runs" / "run1" / "manifest.json"
    b = tmp_path / "runs" / "run2" / "manifest.json"

    assert manifest_mod._lock_for(a) is manifest_mod._lock_for(a)
    assert manifest_mod._lock_for(a) is not manifest_mod._lock_for(b)


def test_hash_path_fast_stable_and_changes_with_content(tmp_path):
    from pipeline.artifacts import hash_path

    f = tmp_path / "a.npz"
    f.write_bytes(b"hello world")
    h1 = hash_path(f)
    h2 = hash_path(f)
    assert h1 == h2

    f.write_bytes(b"hello world!!")
    h3 = hash_path(f)
    assert h3 != h1


def test_hash_path_full_mode_matches_manual_sha256(tmp_path):
    import hashlib

    from pipeline.artifacts import hash_path
    from pipeline.artifacts.hashing import FULL_ALGO

    f = tmp_path / "b.npz"
    f.write_bytes(b"some content" * 1000)
    expected = hashlib.sha256(f.read_bytes()).hexdigest()

    assert hash_path(f, fast=False) == f"{FULL_ALGO}:{expected}"


def test_hash_path_missing_file_raises(tmp_path):
    from pipeline.artifacts import hash_path

    with pytest.raises(FileNotFoundError):
        hash_path(tmp_path / "nope.npz")


def test_artifact_rejects_unknown_kind():
    from pipeline.artifacts import Artifact

    with pytest.raises(Exception):
        Artifact(name="x", kind="not-a-kind", path="/tmp/x", producing_stage="convert")


def test_store_list_runs_most_recent_first(tmp_path):
    from pipeline.artifacts import save_manifest
    from pipeline.artifacts.store import list_runs

    from pipeline.artifacts import create_run

    create_run("older", "base", {}, runs_root=tmp_path)
    newer = create_run("newer", "base", {}, runs_root=tmp_path)
    newer.updated_at = "2099-01-01T00:00:00+00:00"  # deterministically "most recent"
    save_manifest(newer, runs_root=tmp_path)

    runs = list_runs(runs_root=tmp_path)
    assert [r["run_id"] for r in runs] == ["newer", "older"]


def test_store_skips_corrupt_run_when_listing(tmp_path):
    from pipeline.artifacts import create_run
    from pipeline.artifacts.store import list_runs

    create_run("good", "base", {}, runs_root=tmp_path)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{broken")

    runs = list_runs(runs_root=tmp_path)
    assert [r["run_id"] for r in runs] == ["good"]


def test_store_list_and_get_artifact(tmp_path):
    from pipeline.artifacts import Artifact, create_run, record_stage_result
    from pipeline.artifacts.store import ArtifactNotFoundError, get_artifact, list_artifacts

    create_run("run008", "base", {}, stage_names=["convert"], runs_root=tmp_path)
    art = Artifact(name="scene.npz", kind="npz", path="/tmp/scene.npz", producing_stage="convert")
    record_stage_result("run008", "convert", status="success", artifacts=[art], runs_root=tmp_path)

    artifacts = list_artifacts("run008", runs_root=tmp_path)
    assert [a.name for a in artifacts] == ["scene.npz"]

    fetched = get_artifact("run008", "scene.npz", runs_root=tmp_path)
    assert fetched.path == "/tmp/scene.npz"

    with pytest.raises(ArtifactNotFoundError):
        get_artifact("run008", "does-not-exist", runs_root=tmp_path)


def test_api_wiring_delegates_to_artifacts(tmp_path, monkeypatch):
    """`pipeline.api`'s list_runs/list_artifacts/get_artifact/get_status stubs (T01) now
    delegate to `pipeline.artifacts` (T03), the same way T02 wired list_presets/validate_config."""
    monkeypatch.setenv("PIPELINE_RUNS_ROOT", str(tmp_path))

    from pipeline import api
    from pipeline.artifacts import Artifact, create_run, record_stage_result

    create_run("run009", "base", {"name": "base"}, stage_names=["convert"], runs_root=tmp_path)
    art = Artifact(name="scene.npz", kind="npz", path="/tmp/scene.npz", producing_stage="convert")
    record_stage_result("run009", "convert", status="success", artifacts=[art], runs_root=tmp_path)

    runs = api.list_runs()
    assert [r["run_id"] for r in runs] == ["run009"]

    status = api.get_status("run009")
    assert status["status"] == "success"
    assert status["stages"]["convert"]["status"] == "success"

    artifacts = api.list_artifacts("run009")
    assert artifacts[0]["name"] == "scene.npz"

    fetched = api.get_artifact("run009", "scene.npz")
    assert fetched["path"] == "/tmp/scene.npz"

    # `cancel` is the one stub left untouched by T03/T05/T08/T12 — out of scope for every task
    # scheduled so far (see tests/test_import.py). `run_pipeline`/`run_stage` were wired for real
    # in T05 (tests/test_dag.py); `gpu_status` in T12 (tests/test_resources.py) — neither is part
    # of this "still a stub" check anymore.
    with pytest.raises(NotImplementedError):
        api.cancel("run009")
