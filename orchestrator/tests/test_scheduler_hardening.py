"""Tests for the 2026-10-05 scheduler hardening fixes (reviews/orchestrator-correctness-review.md
bugs 1.3 / 1.4 / 1.8 / 1.9):

- **1.3** — directory artifacts now carry a real content fingerprint
  (``pipeline.artifacts.hash_directory``), so a forced rerun of a directory-producing stage
  invalidates its downstream stages' cache keys instead of leaving them at a constant ``""``.
- **1.4** — a cache hit (same-run ``_already_recorded`` or the cross-run index) revalidates
  that the recorded artifact paths still exist, and that file hashes still match; a stale entry
  is treated as a miss and the stage re-runs.
- **1.8** — a stage that exits 0 but never wrote a declared output is recorded ``failed``
  (``StageOutputError``), not ``success``, and is never cached.
- **1.9** — an OOM-fallback success is recorded/cached under the *effective* (reduced-memory)
  config's cache key, never the original config's.

Toy stages register under role ``"test"`` so ``pipeline.api._auto_stage_plan`` never picks them
up (same pattern as ``tests/test_dag.py``).
"""

from __future__ import annotations

import json

import pytest

CALLS = {"dir_src": 0, "dir_use": 0, "file_src": 0, "oom": 0}


def _register_toy_stages() -> None:
    from pipeline.artifacts import Artifact
    from pipeline.stages import Stage, StageContext, list_stages, register

    if "test.h_dir_src" in list_stages():
        return

    @register("test.h_dir_src")
    class DirSrcStage(Stage):
        """Produces a DIRECTORY artifact whose content changes every execution (stand-in for
        capture.isaac/train.default)."""

        outputs = ("d",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["dir_src"] += 1
            out = ctx.run_dir / "d_out"
            out.mkdir(parents=True, exist_ok=True)
            (out / "payload.txt").write_text(f"content-v{CALLS['dir_src']}")
            return {"d": Artifact(name="d", kind="dataset", path=str(out), producing_stage=ctx.stage_name)}

    @register("test.h_dir_use")
    class DirUseStage(Stage):
        inputs = ("d",)
        outputs = ("u",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["dir_use"] += 1
            out = ctx.run_dir / "u.json"
            out.write_text(json.dumps({"n": CALLS["dir_use"]}))
            return {"u": Artifact(name="u", kind="json", path=str(out), producing_stage=ctx.stage_name)}

    @register("test.h_file_src")
    class FileSrcStage(Stage):
        outputs = ("f",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["file_src"] += 1
            out = ctx.run_dir / "f.json"
            out.write_text(json.dumps({"n": CALLS["file_src"], "cfg": ctx.config.get("n", 1)}))
            return {"f": Artifact(name="f", kind="json", path=str(out), producing_stage=ctx.stage_name)}

    @register("test.h_oom")
    class OomOnceStage(Stage):
        outputs = ("o",)

        def run(self, ctx: StageContext) -> dict[str, Artifact]:
            CALLS["oom"] += 1
            if CALLS["oom"] == 1:
                exc = RuntimeError("simulated OOM")
                exc._toy_oom = True  # see fake_is_oom_error in the D test
                raise exc
            out = ctx.run_dir / "o.json"
            out.write_text(json.dumps({"cfg": ctx.config.get("n", 1)}))
            return {"o": Artifact(name="o", kind="json", path=str(out), producing_stage=ctx.stage_name)}


@pytest.fixture(autouse=True)
def _toy_stages():
    _register_toy_stages()
    for k in CALLS:
        CALLS[k] = 0
    yield


# --- 1.3: directory artifacts are content-hashed ------------------------------------------------


def test_hash_directory_fingerprint_tracks_tree_content(tmp_path):
    from pipeline.artifacts import hash_directory

    d = tmp_path / "tree"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("alpha")
    (d / "sub" / "b.txt").write_text("beta")

    h1 = hash_directory(d)
    assert hash_directory(d) == h1  # deterministic

    (d / "a.txt").write_text("ALPHA")  # same size, different content -> content-hashed
    h2 = hash_directory(d)
    assert h2 != h1

    (d / "sub" / "c.txt").write_text("gamma")  # file added
    assert hash_directory(d) != h2

    with pytest.raises(FileNotFoundError):
        hash_directory(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        hash_directory(d / "a.txt")  # files go through hash_path, not hash_directory


def test_forced_producer_rerun_invalidates_downstream_directory_consumers(tmp_path):
    """The 1.3 scenario: force-rerun the directory-producing stage, resume the run — the
    downstream consumer must re-run because the directory's content fingerprint changed."""
    from pipeline.dag import run_dag

    names = ["test.h_dir_src", "test.h_dir_use"]
    m1 = run_dag("run1", names, {}, preset="toy", runs_root=tmp_path)
    assert m1.status == "success"
    assert CALLS["dir_src"] == 1 and CALLS["dir_use"] == 1
    # The directory artifact now carries a real fingerprint, not None/"".
    assert m1.artifacts["d"].content_hash

    # Force-rerun just the producer (same config — the key wouldn't change on its own);
    # the stage writes different directory content this time.
    m2 = run_dag("run1", names, {}, preset="toy", only=["test.h_dir_src"], force=True, runs_root=tmp_path)
    assert m2.stages["test.h_dir_src"].status == "success"
    assert CALLS["dir_src"] == 2
    assert m2.artifacts["d"].content_hash != m1.artifacts["d"].content_hash

    # Resume the full run: the consumer's cache key input is the NEW directory fingerprint,
    # so it re-runs instead of being silently skipped against stale data.
    m3 = run_dag("run1", names, {}, preset="toy", runs_root=tmp_path)
    assert CALLS["dir_src"] == 2  # producer itself is fresh, stays put
    assert CALLS["dir_use"] == 2  # consumer re-ran
    assert m3.stages["test.h_dir_use"].status == "success"


# --- 1.4: cache hits revalidate artifact existence/hashes ----------------------------------------


def test_cross_run_cache_hit_with_deleted_output_reruns(tmp_path):
    from pipeline.dag import run_dag

    cfgs = {"test.h_file_src": {"n": 1}}
    m1 = run_dag("run1", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert m1.stages["test.h_file_src"].status == "success"
    assert CALLS["file_src"] == 1

    # The producing run's output is deleted externally — the cache index entry is still valid
    # JSON, but its path is dead.
    (tmp_path / "run1" / "f.json").unlink()

    m2 = run_dag("run2", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS["file_src"] == 2  # treated as a cache miss, re-ran
    assert m2.stages["test.h_file_src"].status == "success"
    assert m2.artifacts["f"].path != m1.artifacts["f"].path or (tmp_path / "run2" / "f.json").is_file()


def test_cross_run_cache_hit_with_modified_output_reruns(tmp_path):
    from pipeline.dag import run_dag

    cfgs = {"test.h_file_src": {"n": 1}}
    m1 = run_dag("run1", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert m1.stages["test.h_file_src"].status == "success"

    f = tmp_path / "run1" / "f.json"
    f.write_text(f.read_text() + " tampered")  # external modification, path still exists

    m2 = run_dag("run2", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS["file_src"] == 2  # hash mismatch -> cache miss -> re-ran
    assert m2.stages["test.h_file_src"].status == "success"


def test_same_run_resume_with_deleted_output_reruns(tmp_path):
    """The same hole in the same-run path: `_already_recorded` must not trust the manifest
    record over the filesystem."""
    from pipeline.dag import run_dag

    cfgs = {"test.h_file_src": {"n": 1}}
    run_dag("run1", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS["file_src"] == 1

    (tmp_path / "run1" / "f.json").unlink()

    m = run_dag("run1", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS["file_src"] == 2  # record existed but the file didn't -> re-ran
    assert m.stages["test.h_file_src"].status == "success"


def test_intact_cache_hit_still_skips(tmp_path):
    """Revalidation must not break the normal case: unchanged artifacts still skip."""
    from pipeline.dag import run_dag

    cfgs = {"test.h_file_src": {"n": 1}}
    run_dag("run1", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    m2 = run_dag("run2", ["test.h_file_src"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    assert CALLS["file_src"] == 1
    assert m2.stages["test.h_file_src"].status == "skipped"


# --- 1.8: declared outputs must exist after a success exit ---------------------------------------


def test_missing_declared_output_fails_loudly_instead_of_caching_success(tmp_path):
    from pipeline.artifacts import Artifact
    from pipeline.dag import get_cached, compute_cache_key, run_dag
    from pipeline.stages import Stage, get_stage, list_stages, register

    if "test.h_nooutput" not in list_stages():

        @register("test.h_nooutput")
        class NoOutputStage(Stage):
            outputs = ("ghost",)

            def run(self, ctx):
                # Exits "successfully" without writing anything — the pre-fix silent-poison case.
                return {"ghost": Artifact(name="ghost", kind="json", path=str(ctx.run_dir / "ghost.json"),
                                          producing_stage=ctx.stage_name)}

    m = run_dag("run_ghost", ["test.h_nooutput"], {}, preset="toy", runs_root=tmp_path)
    rec = m.stages["test.h_nooutput"]
    assert m.status == "failed"
    assert rec.status == "failed"
    assert "ghost" in rec.error and "does not exist" in rec.error
    # ...and nothing was cached under its key (the poisoned-cache half of the bug).
    key = compute_cache_key(get_stage("test.h_nooutput"), {}, {}, m.git_sha)
    assert get_cached(key, runs_root=tmp_path) is None


def test_empty_directory_output_fails_loudly(tmp_path):
    from pipeline.artifacts import Artifact
    from pipeline.dag import run_dag
    from pipeline.stages import Stage, list_stages, register

    if "test.h_emptydir" not in list_stages():

        @register("test.h_emptydir")
        class EmptyDirStage(Stage):
            outputs = ("d",)

            def run(self, ctx):
                out = ctx.run_dir / "empty_out"
                out.mkdir(parents=True, exist_ok=True)
                return {"d": Artifact(name="d", kind="dataset", path=str(out), producing_stage=ctx.stage_name)}

    m = run_dag("run_empty", ["test.h_emptydir"], {}, preset="toy", runs_root=tmp_path)
    assert m.stages["test.h_emptydir"].status == "failed"
    assert "empty directory" in m.stages["test.h_emptydir"].error


def test_stage_not_returning_a_declared_output_fails_loudly(tmp_path):
    from pipeline.dag import run_dag
    from pipeline.stages import Stage, list_stages, register

    if "test.h_wrongkeys" not in list_stages():

        @register("test.h_wrongkeys")
        class WrongKeysStage(Stage):
            outputs = ("x",)

            def run(self, ctx):
                return {}  # declared "x" but returned nothing

    m = run_dag("run_keys", ["test.h_wrongkeys"], {}, preset="toy", runs_root=tmp_path)
    rec = m.stages["test.h_wrongkeys"]
    assert rec.status == "failed"
    assert "declared" in rec.error


# --- 1.9: OOM-fallback results are cached under the effective config's key -----------------------


def test_oom_fallback_recorded_and_cached_under_effective_config_key(tmp_path, monkeypatch):
    from pipeline.dag import compute_cache_key, get_cached, run_dag
    from pipeline.resources import oom_retry as oom_retry_mod
    from pipeline.stages import get_stage

    real_is_oom_error = oom_retry_mod.is_oom_error

    def fake_is_oom_error(exc):
        return getattr(exc, "_toy_oom", False) or real_is_oom_error(exc)

    monkeypatch.setattr(oom_retry_mod, "is_oom_error", fake_is_oom_error)
    monkeypatch.setattr(
        oom_retry_mod,
        "reduced_memory_config",
        lambda name, cfg: {**cfg, "n": 0} if name == "test.h_oom" and cfg.get("n") != 0 else None,
    )

    cls = get_stage("test.h_oom")
    original_cfg = {"n": 5}
    cfgs = {"test.h_oom": dict(original_cfg)}

    m = run_dag("run_oom", ["test.h_oom"], {}, preset="toy", stage_configs=cfgs, runs_root=tmp_path)
    rec = m.stages["test.h_oom"]
    assert rec.status == "success"
    assert rec.oom_fallback == {"reason": "cuda_oom", "changed": {"n": 0}}

    original_key = compute_cache_key(cls, original_cfg, {}, m.git_sha)
    effective_key = compute_cache_key(cls, {"n": 0}, {}, m.git_sha)
    assert original_key != effective_key
    # Recorded and cached under the EFFECTIVE key only — the original config's key must NOT
    # resolve to fallback-produced artifacts.
    assert rec.cache_key == effective_key
    assert get_cached(original_key, runs_root=tmp_path) is None
    assert get_cached(effective_key, runs_root=tmp_path) is not None
