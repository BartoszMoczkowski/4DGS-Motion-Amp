# T26 — CLI command migration: task log (2026-10-06)

Status: **done**. Every recurring operation in the old AGENTS.md §5 catalog now has an `amp`
equivalent; raw `python path/to/script.py` survives only as the in-container exec path the
Docker stages use (`pipeline/stages/cuda_common.py`), not the user path. The full AGENTS.md
§5 rewrite landed here (decision recorded in the plan: here, not T31).

## Old → new command mapping

| Old invocation | New command |
|---|---|
| `uv run --package 4dgs-core python core/train.py -s ... -m ... --configs ...` | `amp train <preset> [--scene DIR] [--dry-run] [--force]` |
| `python core/render.py -m <model> --iteration ... --skip_train` | `amp render <run_id>` |
| `python core/render_amp.py -m ... --amp_factors 2 -1 ... --method ... --video_path ...` | `amp render <run_id> --amp-factors "2 -1 ..." --method ... --video ... [--low-vram] [--frozen-cam]` |
| `python -m motion_seg.extract_trajectories --model_path ... --configs ...` | `amp seg extract <run_id> [--n-times N]` |
| `python -m motion_seg.segment_rigid --trajectories ... --out ... [--k ...]` | `amp seg rigid <run_id> [--k --min-size --threshold-mult --opacity-thresh]` |
| (rigid2 / kabsch / multicut / mbs raw module runs) | `amp seg rigid2|kabsch|multicut|mbs <run_id> [impl flags]` |
| `python -m motion_seg.evaluate_segmentation --pred ... --gt ...` | `amp seg eval <run_id> [--drop-floaters --bg-label X]` |
| `./motion-seg/motion_seg/run.sh <scene>` (+ `SKIP_EXTRACT=1`, `EXTRACT_ARGS`) | `amp seg run <preset> [--impl ...] [--skip-extract]` — **run.sh deleted** |
| `python omniverse-pipeline/omniverse_pipeline/split_mesh.py --in ... --out ...` | `amp capture split --in ... --out ... [--group --min-faces --preview --no-usd]` |
| `python .../add_motion.py --in ... --out ...` | `amp capture motion --in ... --out ... [--num-frames --fps --trans-amp-mm A B --rot-surface-mm A B --rot-deg-max D --freq LO HI --groups K --exclude ... --seed --plot]` |
| `python scene-gen/gen_scenes.py [--base-amp-mm ... --multipliers ...]` | `amp capture gen-grid [same flags]` |
| `<isaac-python.bat> omni_capture.py --config ...` (native) | `amp capture run --config YAML [--usd --out --n-cameras --frames --headless]` — still native-Windows-subprocess-only, via `run_native_isaac_script` |
| `python .../omni_to_4dgs.py --capture ... --out . --name pump01` | `amp capture convert --capture DIR --out DIR --name N [--near --far --target-radius]` |
| `python scene-gen/frames_to_mp4.py --capture ...` | `amp capture preview --capture DIR [--out-dir --fps --tile-cols --tile-width --no-per-cam]` |
| ad-hoc `python -c` frame counting (capture post-mortem) | `amp capture verify <dir> [--cams N --frames M \| --config YAML]` |
| `python scene-gen/check_capture_progress.py [--watch]` | unchanged (now a thin wrapper over `omniverse_pipeline.capture_check`) |
| `python -m mcp_server` (+ manual env) | `amp orch mcp [--host --port]` (refuses to start without `PIPELINE_MCP_TOKEN`) |
| `streamlit run orchestrator/ui/app.py` / `amp-ui/amp_ui/ampUI.py` | `amp orch ui [--app orchestrator\|amp]` |
| `python amp-ui/amp_ui/run_renders_auto.py` (hardcoded lists, cwd results.csv) | `amp bench renders [--models --methods --repeats --vram-modes --out-csv --no-registry]` |
| scene-gen one-off analysis scripts (10 + `identity/`) | moved to `scratch/` — unsupported, not installed, frozen as-is |

## Design decisions (in-task)

- **Config overrides API** (`pipeline.api`, Phase A): `run_pipeline()`/`run_stage()` take
  `overrides: dict | None`, deep-merged onto the resolved config and **re-validated through
  `PipelineConfig`** so `extra="forbid"` still rejects misspelled keys. In `run_pipeline` the
  merge happens before `_auto_stage_plan` (an override flipping `segment.impl` changes the
  plan). In `run_stage` the manifest's recorded `resolved_config` is deliberately **not**
  rewritten — overrides apply per-invocation (and the changed slice changes the cache key, so
  the stage re-runs instead of cache-hitting on stale parameters).
- **`amp train`/`amp render` are commands, not groups** — a mounted Typer group can't take a
  positional argument, so `core_cli.train`/`render` are registered as commands on the root app.
  (T25 exported stub `train_app`/`render_app` Typers; replaced.)
- **Amp-factor validation is reimplemented locally in `core_cli.py`** (8 values, −1 only
  negative; freqs len 1/8) rather than importing torch-heavy `render_amp` — keeps even error
  paths light; render_amp re-validates in-container anyway.
- **amp-factors → channels mapping**: positional slots in render_amp order
  `[means3D, means2D, scales, rotations, opacity, shs, colors, cov3D]` map to channel keys
  `("pos3d","pos2d","scale","rotation","opacity","SHs","color","cov3D")` (per
  `ampUI.py`'s `chanels_list`). CLI flags default to `None` and are pruned, so unset flags
  never clobber preset values.
- **Dry-run** on every pipeline-executing command: prints stage name + JSON `_stage_config_for`
  slice, creates no run dir. This is the sandbox-testable surface for GPU/Isaac commands.
- **argparse-`main()` reference modules** (split_mesh/add_motion/frames_to_mp4/gen_scenes) are
  called in-process via a temporary `sys.argv` swap — no subprocess, no re-porting of flag
  wiring. `omni_to_4dgs.convert()` / `capture_check.verify_capture_dir()` are real callables.
- **capture_check relocation**: capture QA moved from scene-gen to `omniverse_pipeline` (correct
  dependency direction); heuristics parameterized (`expected_cams`/`expected_frames`, or parsed
  from the capture YAML's `rig.n_cameras`/`capture.num_frames`); legacy 16/60 kept as defaults;
  per-camera counts added (`cam_frames`). `scene-gen/check_capture_progress.py` keeps its CLI as
  a thin wrapper.
- **`amp capture run` uses a minimal `StageContext`** (`create_run` + `StageContext(run_dir,
  logger, paths=pipeline.paths)`) and calls `run_native_isaac_script(ctx, "omni_capture", ...)` —
  no fallback to bare subprocess needed; launcher check happens before the run dir is created.
- **bench registry artifact**: `run_renders_auto` refactored into `run_benchmark(...)` (module
  constants are now `DEFAULT_*`; no-arg call reproduces old behavior incl. atomic per-combo CSV).
  The CLI additionally records a fresh `bench-*` run (`create_run`) with a `bench_results.json`
  summary artifact (kind `json`, manifest status success); CSV write unchanged.
- **scene-gen reorg**: moved to `scratch/` (git mv): `analyze_complexity_ladder.py`,
  `partition_swap.py`, `re_render_grid_runs.py`, `render_seg_visualizations.py`,
  `run_grid_identity.py`, `identity/`, `run_grid_multicut_calibrated.py`,
  `run_partition_swap_exp.py`, `run_pump01_boundary_refine.py`, `run_pump01_spatial_cc.py`,
  `show_cubes_seg.py`. Import audit first: only other moved/scratch files import them.
  `scene-gen/pyproject.toml` py-modules trimmed; `scripts/smoke_workspace_imports.py` updated;
  `.venv/Scripts/python.exe` docstring instructions stripped from `run_grid_4dgs.py` /
  `run_grid_seg.py` (interim `uv run python ...`, `amp exp` lands in T27).

## ⚠️ Follow-up for Bartosz (pre-existing bug found in Phase A)

`pipeline/config/models.py:AMP_CHANNELS` = `(pos3d, pos2d, rotation, scale, ...)` puts
**rotation before scale**, but `core/render_amp.py`'s positional `amp_factors` slot order is
`[means3D, means2D, scales, rotations, ...]` — **scale before rotation** (confirmed by
`ampUI.py`'s `chanels_list`). `pipeline/stages/amp.py` emits `--amp_factors` positionally in
AMP_CHANNELS order, so the `amp.default` stage currently swaps the scale and rotation factors
(and freq cutoffs) relative to what render_amp expects. The new CLI maps by key and is
unaffected. Verify on real hardware (render with scale-only amplification and check the video),
then fix either AMP_CHANNELS order or `AmpStage`'s emission loop.

## Gotchas

- click can't give an option argparse-style `nargs="+"`: `--amp-factors 2 -1 -1 ...` fails
  ("No such option: -1"). The CLI accepts a **quoted** space/comma-separated list
  (`--amp-factors "2 -1 ..."`) or repeated flags; a quoted list *starting with a negative* needs
  the `=` spelling (`--amp-factors="-2 ..."`). Same for `--freq-low/--freq-high`, `--models`,
  `--vram-modes`, `--exclude`, `--base-amp-mm`, `--multipliers`. Exception: `--methods` is
  comma-split only (labels contain spaces: "base segmented").
- Rich wraps long error messages at 80 cols — tests assert on short substrings.
- Windows test-host caveat: two suite failures are environmental (they assert hardware absence):
  `test_mcp_server.py::test_gpu_status_over_real_http_with_valid_token` (GPU present),
  `test_mcp_tools.py::test_list_containers_without_a_docker_daemon...` (Docker daemon running).

## Verification (2026-10-06, this machine)

- `uv lock --check` / `uv sync` clean.
- `cd orchestrator && uv run --group test pytest -q`: **377 passed, 9 skipped, 2 failed** —
  both failures environmental (above); baseline at T26 start was 315 passed. New test files:
  `test_api_overrides.py` (6), `test_amp_cli_commands.py` (16), `test_amp_cli_seg_bench.py` (21),
  `test_amp_cli_capture.py` (20). `test_amp_cli.py`'s clean-subprocess heavy-import probe passes
  unchanged.
- `uv run python scripts/smoke_workspace_imports.py`: 28/28 required, 10 expected skips, 0 failures.
- `amp --help` + every group/subcommand `--help` exit 0; manual spot checks:
  `amp seg run pump01 --dry-run`, `amp render nonexistent-run --amp-factors "2 -1 -1 -1 -1 -1 -1 -1"`
  (unknown-run error path), `amp capture verify --help`.
- `grep -rn "sys.path" scene-gen/ motion-seg/ amp-ui/` — clean (one comment referencing the
  sanctioned `mbs_bootstrap` mechanism); scratch/ exempt.
- Not sandbox-verifiable (Bartosz's real-hardware follow-up): `amp train`/`amp render`/`amp seg
  extract` execution, `amp capture run` against real Isaac Sim, `amp bench renders` full run,
  `amp orch mcp` with a token, `amp orch ui`.

## Follow-up resolution (2026-10-06/09)

### AMP_CHANNELS scale/rotation swap — fixed

`pipeline/config/models.py:AMP_CHANNELS` reordered to the canonical
`core/render_amp.py` slot order `(pos3d, pos2d, scale, rotation, opacity, SHs, color, cov3D)`
(was rotation-before-scale, so `AmpStage`'s positional `--amp_factors` emission swapped the two
slots). Consumer sweep:

- `pipeline/stages/amp.py` — docstring claim ("matches render_amp's positional lists") is now
  true; emission loop unchanged (it follows AMP_CHANNELS).
- `pipeline/config/MIGRATION.md` §"Positional -> named channels" — order corrected + fix noted.
- `core/core_cli.py:30-36` — NOTE comment updated: mismatch fixed; the keyed `_AMP_FACTOR_CHANNELS`
  mapping stays so the CLI is immune to future reordering.
- `orchestrator/ui/app.py:154` — iterates AMP_CHANNELS by name for display only; unaffected.
- Presets key channels by name (order-independent); only `radius_scale` matched the sweep, no
  positional reliance anywhere.

New regression test `test_amp_stage_emits_factors_in_render_amp_slot_order`
(`orchestrator/tests/test_stages_cuda.py`): distinct factor 2..9 per channel, asserts the emitted
`--amp_factors` is exactly `[2,3,4,5,6,7,8,9]` with scale at index 2 and rotation at index 3.

`cd orchestrator && uv run --group test pytest -q`: **378 passed, 9 skipped, 2 failed** — the two
failures are the known environmental ones (GPU present / Docker daemon up).

### Real-hardware verification (this machine, RTX 3090 + Docker Desktop)

Root-cause detour: the first `amp render` attempt hung 60 min with no container activity.
`ensure_image("cuda")` compares the image's `pipeline.cuda_build_hash` label against current
inputs — the label (Sept-4 build) no longer matched (`uv.lock`/Dockerfile drift since), so it
silently started a **full 25 GB image rebuild**, and with **no `.dockerignore`** in the repo the
docker-py build streams the entire repo (runs/, output/, data/, LLFF/, .venv/...) as build
context — hours of silent work. As a verification expedient the existing image was re-labeled in
place (`FROM 4dgs-motion-amp-cuda:latest` + corrected `pipeline.cuda_build_hash`, 1-second shim
build); in-container code paths come from the bind-mounted live repo, so stage behavior is
current. **A proper clean rebuild is still pending** (see new issues below).

| Check | Result | Evidence |
|---|---|---|
| `amp render cubes-k1-smoke --amp-factors "2 -1 -1 -1 -1 -1 -1 -1" --method eulerian --video t26_verify.mp4` | **pass** | `runs/cubes-k1-smoke/train_out/video/t26_verify.mp4` (2.7 MB, 300 frames @ ~60 it/s in-container); also exercises the fixed channel order end-to-end |
| `amp render cubes-k1-smoke --force` (render.default) | **pass** | stage finished, exit 0 |
| `amp seg extract cubes-k1-smoke --n-times 10` | **pass** | fresh `runs/cubes-k1-smoke/trajectories.npz` (2.0 MB) |
| `amp seg rigid` | **pass** | fresh `segmentation.npz` on cubes-k1-smoke; also forced fresh run on cubes-k1 via `--k 8` |
| `amp seg eval` | **pass** | cubes-k1 (has `gt_segmentation`): fresh `seg_eval_result.json` (k=8 → ARI 0.086, 18 clusters); on cubes-k1-smoke correctly raises clean `MissingDependencyError` for missing GT |
| `amp train cubes --scene ...` | **pass (dry-run + smoke)** | `--dry-run` resolves `train.default`; full 15k-iter run >20 min, so smoke via `run_stage('cubes-k1-smoke','train.default',overrides={'optim':{'iterations':110,'coarse_iterations':50}})` → fresh `point_cloud/iteration_110/{point_cloud.ply,deformation.pth,...}`, "Training complete." The CLI has no iterations-override flag — candidate for a future flag |
| `amp bench renders --models multipleview/test1 --methods base --repeats 1 --vram-modes full` | **skip (architectural)** | fails cleanly: "the render benchmark needs the Linux/CUDA 4DGS environment (import failed: No module named 'simple_knn')" — `run_renders_auto`/`render_amp` import natively and are Linux/CUDA-only (known §9 gotcha). Pre-existing limitation of the old harness, not a T26 regression; bench needs containerization to work on Windows |
| `amp orch mcp` (token set) | **pass** | `curl /mcp` → 401 without bearer; with bearer full MCP initialize handshake (`serverInfo: 4dgs-pipeline 1.28.1`); server stopped after |
| `amp orch ui --app orchestrator` | **pass** | http://localhost:8501 → HTTP 200 (Streamlit shell); stopped after |
| `amp capture run --config capture_config_cubes.yaml --frames 2 --out Q:/Omniverse/renders/t26_verify_capture --headless 1` | **pass** | real native Isaac Sim subprocess; 10 cams × 2 frames written (`cam01/rgb/rgb_0000.png`, `rgb_0001.png`, ... + `points3D_gt.ply`), run dir `runs/capture-ea2c902c`; ~2 min end-to-end |
| `amp capture verify Q:/Omniverse/renders/capture_cubes_k2_ring --cams 16 --frames 120` | **pass** | "verify: OK"; with the mismatched cubes YAML it correctly reports per-cam MISMATCH lines (negative path works too) |

Cleanup: `pipeline-cuda` warm container stopped (none were running before); MCP/UI background
processes killed; test token deleted. `stupefied_newton` (unrelated VS Code devcontainer of
another project) was left alone. The `t26_verify.mp4` and `Q:/Omniverse/renders/t26_verify_capture`
artifacts were left in place for inspection.

### New issues found

1. **Silent full-context image rebuild.** `ensure_image("cuda")` rebuilds when the build-hash label
   is stale, but (a) there is **no `.dockerignore`**, so the build context is the whole repo
   (100+ GB incl. `runs/`, `output/`, `.venv/`), and (b) the build streams for hours with zero
   stage-log output — it looks exactly like a hang. Recommended: add a `.dockerignore` (exclude
   `runs/`, `output/`, `data/`, `.venv/`, `LLFF/`, `viseron/`, ...) and log a "rebuilding cuda
   image, this may take a while" line before `images.build`. Until a proper rebuild is done, the
   `pipeline.cuda_build_hash` label on `4dgs-motion-amp-cuda:latest` is the shim value — the image
   *content* still dates from 2026-09-04.
2. **`amp bench renders` is unusable on Windows** — the harness imports `render_amp` (torch +
   `simple_knn`) natively. Either document as Linux-only or route through the `cuda` container
   like the other GPU stages.

### Issue-1 resolution (2026-10-09): build context + streaming progress + clean rebuild

- **`.dockerignore` (new, repo root).** The `cuda` Dockerfile COPYs only `pyproject.toml`,
  `uv.lock`, `submodules/`, `core/`, and the six other workspace members' `pyproject.toml`
  manifests — the file excludes everything else (`.git`, `.venv`, `data/`, `output/`, `runs/`,
  `assets/`, `LLFF/`, `viseron/`, `papers/`, `docs/`, `scratch/`, `weights/`, `results_visual/`,
  `reviews/`, `test_tmp/`, `.claude_notes/`, `.devcontainer/`, `**/__pycache__`, `**/*.egg-info`,
  `*.zip/*.mp4/*.jpg/*.pdf`, member trees via `member/*` + `!member/pyproject.toml` negations,
  and `submodules/multibody-sync-4dgs/ckpt/`). CUDA extension sources stay in. Context dropped
  from the whole repo (100+ GB, hours of silent client-side tar streaming) to ~55 MB (core 5 MB +
  submodules 50 MB + manifests); the build reached `Step 6/24` within seconds.
- **Streaming build progress** in `orchestrator/pipeline/containers/manager.py::ensure_image`:
  the cuda build now uses the low-level `client.api.build(..., decode=True)` chunk iterator,
  printing an up-front `[cuda-build] rebuilding image ...: <no local image|hash mismatch>` header
  and every step line as it arrives (docker-py's high-level `images.build` buffers the whole
  stream — that was the silent hour). A failed `RUN` step arrives as an `error` chunk and raises
  `ImageNotAvailableError` with the persisted-log pointer, same contract as before; the full log
  still lands in `runs/.cache/cuda_build.log` on success and failure. `decode=True` and the
  chunk shape are covered by the reworked `_FakeImages.stream_build` fake in
  `orchestrator/tests/test_containers.py`, plus a new
  `test_ensure_image_emits_build_progress_lines` (header + `[cuda-build] Step x/y` lines on a
  fake build; silence when up-to-date). All 35 container tests pass.
- **Clean rebuild for real:** deleted the shim label + old image, ran `ensure_image("cuda")` —
  full rebuild in **~10.5 min** (layer cache up to Step 5, fresh `uv sync` + CUDA-extension
  compiles), image 25.1 GB, `pipeline.cuda_build_hash` label = current `982bbc5b…` (hash drift
  will now correctly trigger rebuilds again, but visibly and with a small context).
  Post-build smoke: `uv run amp render cubes-k1-smoke --amp-factors "2 -1 -1 -1 -1 -1 -1 -1"
  --method eulerian --video t26_verify2.mp4 --force` → fresh 2.76 MB video, manifest stage
  status `success`.
- **Suite:** `cd orchestrator && uv run --group test pytest -q` → **379 passed, 9 skipped,
  2 failed** (the two known environmental ones).

### New issues found (round 2)

3. **Stage failure is recorded, not raised, and the CLI says "finished" anyway.** `run_dag`
   catches a failing stage (by design, "must not crash the scheduler"), records
   `status="failed"` in the manifest, and returns; `core_cli.render`/`train` then print
   `<stage> finished; run_id: ...` unconditionally. During this session an `amp.default --force`
   run silently failed the T12 RAM pre-flight ("needs 10.0 GB, only 0.6 GB free" — transient
   host pressure) and the CLI reported success. The CLI (and ideally `run_stage`) should read
   the returned manifest's stage status and exit non-zero on `failed`.

### Issue-3 + bench resolution (2026-10-09): failure propagation + bench docs

- **Stage failure propagation (issue 3 above).** New `pipeline.api.StageFailedError`:
  `run_dag` keeps its deliberate record-and-return semantics (the manifest stays the durable
  record; later selected stages remain `pending`), and `run_pipeline`/`run_stage` now raise
  `StageFailedError` after `run_dag` returns when any stage *this call* executed has
  `status="failed"` — the message names the stage(s), the error, and the stage log path.
  Layers 2/3 needed no changes: the MCP server already captures background-job exceptions into
  `job_error` (`mcp_server/jobs.py`), and the UI displays caught exceptions (`ui/app.py`).
- **CLI exit codes.** `core/core_cli.py` (`train`, `render`) and `motion-seg/motion_seg/cli.py`
  (`_run_seg_stage`, `run`) catch `api.StageFailedError` → red one-line error + `typer.Exit(1)`,
  no traceback. Verified against the observed live failure mode (RAM-preflight rejection would
  now read `amp.default failed: run ...: stage(s) failed: ...` and exit 1).
- **Tests:** `test_dag.py` gains two API tests (`run_stage` raises `StageFailedError` matching
  the stage name while the manifest still records `failed` with the original error; `run_pipeline`
  raises on a failing plan and the dependent stage stays `pending`).
  `test_amp_cli_commands.py` gains a CLI-level test (monkeypatched `api.run_stage` raising
  `StageFailedError` → exit 1, stage name in output, no traceback). No existing test asserted the
  old non-raising behavior.
- **Suite:** `cd orchestrator && uv run --group test pytest -q` → **382 passed, 9 skipped,
  2 failed** (the two known environmental ones); the `test_amp_cli.py` heavy-import probe passes.
- **`amp bench renders` (issue 2 above): documented.** Confirmed on this Windows host:
  `uv run amp bench renders --methods base --repeats 1` exits **1** with the clean one-liner
  "the render benchmark needs the Linux/CUDA 4DGS environment (import failed: No module named
  'simple_knn')" — no traceback, no partial `results.csv`/`bench-*` run written. AGENTS.md §5.6
  now states the Linux/CUDA requirement explicitly; containerizing the bench remains future work.
