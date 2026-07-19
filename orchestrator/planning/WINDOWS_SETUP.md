# First-time Windows setup (for the container manager / GPU stages)

One-time machine setup so `orchestrator/pipeline/containers` can actually build/pull images and
run `cuda`/`isaac` containers with GPU passthrough — **run directly from Windows** (PowerShell or
cmd), no WSL2 distro needed. Do this once; after that, `ensure_image`/`start_container`/
`exec_in_container` (and `tests/test_containers_gpu.py`) just work.

**Revised 2026-07-14:** this replaces the earlier `WSL_SETUP.md`, which assumed the orchestrator
had to run *from inside* a WSL2 Linux distro. That was never actually required — Docker Desktop is
reachable directly from Windows (it's the same engine either way), so running natively is simpler
and has fewer moving parts. Bundling a proper WSL2/Docker setup as an alternative is deferred
future work (`planning/tasks/T16-wsl-docker-bundling.md`), not something you need today.

## 1. Windows prerequisites

1. **Windows 11, or Windows 10 21H2+.**
2. **Update your NVIDIA driver** (the regular Windows GeForce/Studio/RTX driver — nothing special
   to download for GPU-in-Docker; a current driver is enough).
3. **Install Docker Desktop.** The installer sets up its own internal WSL2 engine automatically —
   you don't need to open a WSL2 shell or manage a distro yourself for any of this.
4. **Turn on GPU passthrough**, if it isn't already: Docker Desktop Settings → Resources →
   Advanced → GPU Passthrough (Docker Desktop 4.34+). Recent Docker Desktop + a current driver is
   often enough with `--gpus all` even without touching that setting.

**Verify GPU passthrough works at all**, from a normal Windows terminal, before touching this
repo:
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
If that doesn't print a normal `nvidia-smi` table, stop here and fix that first — nothing in
`pipeline/containers` can work around broken GPU passthrough.

## 2. NGC login (needed to pull the `isaac` image)

`nvcr.io/nvidia/isaac-sim` requires a free NVIDIA NGC account:

```powershell
docker login nvcr.io
# username: $oauthtoken
# password: <your NGC API key>
```
Get a key at https://ngc.nvidia.com → Setup → Generate API Key. Do this once; Docker remembers
the login. `ensure_image("cuda")` (this repo's own image) doesn't need NGC — only `isaac` does.

## 3. Submodules

The `cuda` image build needs `submodules/` checked out — the repo `Dockerfile` does
`COPY submodules/ ./submodules/`:
```powershell
cd C:\path\to\4DGS-Motion-Amp
git submodule update --init --recursive
```

## 4. Omniverse assets (`Q:` drive)

Both containers bind-mount your Omniverse assets to `/omniverse`, from whatever Windows path
`PIPELINE_ASSETS_ROOT` points at (default `Q:\Omniverse`, matching the existing
`capture_config_pump.yaml` convention). If your assets live somewhere else, set the env var:
```powershell
$env:PIPELINE_ASSETS_ROOT = "D:\Somewhere\Omniverse"
```

## 5. Python environment for the orchestrator

You do **not** need the full CUDA/PyTorch training env (`torch`, `mmcv`, ...) just to run the
container manager or its tests — `orchestrator/pipeline` is a separate, lightweight uv workspace
member (`pydantic`, `pyyaml`, `docker`, `pynvml`, `numpy`/`scipy`/`Pillow`). From the repo root, in
a normal Windows Python (3.10+) environment:

```powershell
# with uv (repo already uses it):
uv sync --extra orchestrator

# or plain pip, in any venv:
pip install -e '.[orchestrator]'
```

Sanity-check the CPU-only suite first (no Docker/GPU needed for this part):
```powershell
cd orchestrator
pytest -q
# expect "151 passed, 9 skipped" (as of T11's real-hardware fixup pass, 2026-07-16 -- this count
# grows as later tasks add tests) -- the 9 skips are test_containers_gpu.py (6) and
# tests/test_stages_isaac_gpu.py (3), both auto-skipping until a real Docker daemon is reachable
# (see step 6 / step 8.4).
```

## 6. Run the real GPU/Isaac checks

Now that Docker + GPU + NGC + submodules + assets are all in place. **These need
`PIPELINE_TEST_GPU=1` explicitly set** (2026-07-19 — plain Docker-reachability isn't a safe
default gate on a real machine, where it's always true; without this flag, a normal `pytest -q`
run over the whole repo skips this file instantly instead of silently triggering a real, possibly
20+ minute `cuda` image build):

```powershell
cd orchestrator
$env:PIPELINE_TEST_GPU = "1"; pytest -q -s tests/test_containers_gpu.py   # cuda build, GPU, mounts, warm-reuse, teardown
$env:PIPELINE_TEST_GPU = "1"; $env:PIPELINE_TEST_ISAAC = "1"; pytest -q -s tests/test_containers_gpu.py   # + isaac pull/EULA/cache (slow first run, ~10GB+ pull)
```

If something fails, `pipeline/containers/MANUAL_CHECKLIST.md` has the same 6 checks spelled out
as individual Python snippets, useful for reproducing/debugging one specific step interactively.

## 7. Option-A segmentation (MBS) setup — optional, only for `segment.mbs`

`segment.rigid` (Option B, the default `segment.impl`) needs nothing beyond step 5. This step is
only needed to run `segment.mbs` (Option A — MultiBodySync MotNet inference,
`planning/tasks/T10-wrap-option-a-segmentation.md`), e.g. via the `pump01_segA` preset.

1. **CUDA ops build automatically — no separate build step.** `submodules/multibody-sync-4dgs/
   ext/` JIT-compiles the first time anything imports `utils.pointnet2_util` from that package
   (`torch.utils.cpp_extension.load(...)` at import time) — this happens inside the `cuda`
   container the first time `segment.mbs` actually runs, using the same `nvcc` the `cuda` image
   already ships (it's a `-devel`, not `-runtime`, CUDA base image — see the repo `Dockerfile`).
   Expect a one-time compile delay (a couple of minutes) on the very first `segment.mbs` run;
   every run after that is instant (the compiled extension is cached inside the container's
   filesystem layer for as long as that warm container lives).
2. **Download a pretrained MotNet checkpoint — this part IS manual, there's nothing to automate.**
   `submodules/multibody-sync-4dgs/ckpt/` ships empty (gitignored — no checkpoint is vendored).
   `hubconf.py` in that submodule points at a Google-Drive-hosted checkpoint for the full
   pipeline (flow+conf+mot combined):
   ```
   https://drive.google.com/uc?export=download&id=1bomD88-6N1iGsTtftfGvAm9JeOw8gKwb
   ```
   Download it manually in a browser (Google Drive's large-file virus-scan redirect often breaks
   scripted/`torch.hub` downloads of it) and place it at:
   ```
   submodules\multibody-sync-4dgs\ckpt\mbs_full.pth.tar
   ```
   (matching `pump01_segA.yaml`'s `segment.mbs.checkpoint` default — a repo-root-relative path,
   resolved by `pipeline/stages/segment_mbs.py`; point `segment.mbs.checkpoint` at wherever you
   actually put it if that's not where you downloaded it). The checkpoint's state dict has
   `mot_net.`-prefixed keys mixed in with `flow_net.`/`conf_net.` ones —
   `pipeline/vendored/cuda/mbs_infer.py` strips the prefix and loads only those into a standalone
   `MotNet`; you don't need to separate them yourself.
3. **Known out-of-distribution risk (not a setup problem, a modeling one).** MotNet was trained
   on MBS's own noisy FlowNet-predicted flow at roughly unit/meter point-cloud scale, not the
   exact zero-noise 4DGS trajectories at `pump01`'s `target_radius=4.0`-normalized scale — results
   may need fine-tuning to look good. If `segment.mbs` runs successfully but segments look wrong,
   this is the likely reason, not a bug in the port — see
   `.claude_notes/NOTES_4dgs_motion_segmentation.md` §6d and
   `planning/tasks/T10-wrap-option-a-segmentation.md`'s "out of scope" note (fine-tuning MotNet
   itself is research, not infra, and explicitly not this task's job).
4. **First real run is expected to need debugging.** The port was never executed before T10 (no
   GPU/compiled `ext/`/checkpoint were available while writing it) — expect to iterate on real
   shape/behavior mismatches the first time `segment.mbs` actually runs against real data, exactly
   like `seg_extract`/`train`/`render`/`amp` (T09) did before their first real run.

## 8. Isaac prep/capture stages setup

Step 2 (NGC login) already covers pulling the `isaac` image; this step covers the extra things
`prep_split.default`/`prep_motion.default`/`capture.isaac` each need beyond that. **Revised
2026-07-16:** `capture.isaac` no longer runs inside the `isaac` Docker container at all — see step
8.5 below — so steps 8.1–8.4 (the `trimesh` install, the cache-permission fixup, etc.) now only
matter for `prep_split.default`/`prep_motion.default`.

1. **Install `trimesh` in the `isaac` container — this part IS manual.** Isaac Sim's bundled
   interpreter (`/isaac-sim/python.sh`) already ships `pxr` (usd-core), but not `trimesh`, which
   `split_mesh.py` needs for mesh loading/connected-component splitting. One-time, inside a warm
   `isaac` container (e.g. via `ContainerManager.exec` or a manual `docker exec`):
   ```
   /isaac-sim/python.sh -m pip install trimesh
   ```
2. **The raw fused-mesh asset (`raw_mesh`, e.g. `CONJUNTO_BOMBAS.usd`) has no in-repo producer —
   pre-seed it yourself.** Same pattern as `gt_segmentation` (step 7 already covers a different
   external input): `prep_split.default` needs a real CAD-exported mesh as an external artifact
   before `pipeline.api.run_pipeline`'s `external_artifacts` parameter can seed it, e.g.:
   ```python
   from pipeline.api import run_pipeline
   from pipeline.artifacts import Artifact

   run_pipeline(
       "pump01",
       external_artifacts={
           "raw_mesh": Artifact(name="raw_mesh", kind="usd", path="Q:/Omniverse/assets/.../CONJUNTO_BOMBAS.usd", producing_stage="external"),
           "gt_segmentation": Artifact(name="gt_segmentation", kind="npz", path="...", producing_stage="external"),
       },
   )
   ```
3. **`pump01.yaml`'s `capture.scene.usd_path`/`capture.output.capture_dir` are now just fallback
   values, not the real ones used.** `capture.isaac` (T11) always overrides both at runtime from
   the DAG's own artifact wiring (`prep_motion.default`'s `animated_mesh` output / this run's own
   capture directory) via `omni_capture.py`'s own `--usd`/`--out` CLI flags — the preset's static
   values are effectively unused once the full chain runs, kept only as human-readable defaults.
4. **T11's two acceptance criteria have a runnable test file**, mirroring `test_containers_gpu.py`
   (T08)'s own real-hardware pattern — `tests/test_stages_isaac_gpu.py`: criterion 1
   (`run_capture.sh`'s `--n-cameras 2 --frames 2` smoke test, via `run_stage`) and criterion 2 (a
   full, deliberately-trimmed prep-through-amp run from a raw asset). Gated behind the same
   `PIPELINE_TEST_ISAAC=1` flag as the Isaac checks in `test_containers_gpu.py`, plus real-asset
   env vars (`PIPELINE_TEST_ANIMATED_MESH`/`PIPELINE_TEST_RAW_MESH`, each falling back to the
   documented `Q:/Omniverse/assets/pump_radnom/...` convention if unset — see that file's own
   module docstring for exact defaults/overrides):
   ```powershell
   cd orchestrator
   $env:PIPELINE_TEST_ISAAC = "1"; pytest -q -s tests/test_stages_isaac_gpu.py
   ```
   Each of the two real-asset tests skips independently (with a clear reason) if its own file
   isn't found, and a `trimesh`-importability check runs first so a missed step 1 above fails fast
   with a pointer back here instead of a confusing mid-script `ImportError`.

   **First real run (2026-07-16) found two real bugs, both now fixed — re-run to verify:**
   - `prep_split.default`/`prep_motion.default` failed with `ModuleNotFoundError: No module named
     'pxr'`: `split_mesh.py`/`add_motion.py` do a bare `from pxr import ...` with no Kit runtime
     bootstrap, which this Isaac Sim 6.0.1 image doesn't support (`pxr` is supplied by Kit's own
     extension loader at `SimulationApp` init, not a static `python.sh` `sys.path` entry). Fixed by
     routing those two scripts through a new `pipeline/stages/_isaac_kit_bootstrap.py` wrapper that
     launches a headless do-nothing `SimulationApp` first — the vendored scripts themselves are
     untouched.
   - `capture.isaac` reported manifest "success" but never wrote `cameras_gt.json`: the persisted
     `isaac-cache` volume (`/isaac-sim/.cache`) wasn't writable by whatever UID `exec` runs as,
     causing a silent `PermissionError` inside Kit's startup that cascaded into
     `omni.replicator.core` failing to load (so `BasicWriter` was never registered) — while Kit
     still exited 0. Fixed with a best-effort `chmod -R 0777` on the three cache-volume mount
     points, run once right after a fresh `isaac` container is created
     (`ContainerManager._fixup_isaac_cache_permissions`).

   See `.claude_notes/NOTES_pipeline_orchestration.md`'s "T11 real-hardware fixup" section for the
   full diagnosis.

5. **`capture.isaac` needs a real, native Isaac Sim install on Windows — not the `isaac` Docker
   container.** Revised 2026-07-16: a second real-hardware run showed the *actual* rendering
   still failing even after the two fixes above (`IHydraTexture ... no GPU foundation` + Replicator
   writer-drain timeouts) — root cause is that Vulkan (what Isaac Sim's Hydra/RTX renderer needs)
   [isn't supported under WSL2](https://forums.developer.nvidia.com/t/isaac-sim-x86-64-headless-docker-wsl2-support/278252),
   which is what backs Docker Desktop's Linux containers on Windows. No Docker/env-var
   configuration can work around this — it's confirmed by NVIDIA as a hard platform limitation.
   `capture.isaac` now execs `omni_capture.py` as a native Windows subprocess against a real Isaac
   Sim install instead (`pipeline.stages.isaac_common.run_native_isaac_script`):
   - Set `PIPELINE_ISAAC_NATIVE_PYTHON` to your Isaac Sim install's own bundled-Python launcher —
     the `python.bat` under wherever Isaac Sim actually lives, e.g. for Bartosz's real machine
     `Q:\Omniverse\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat` (this is also the default
     if the env var is unset, **corrected 2026-07-18** — the originally-documented
     `Q:\Omniverse\ISAAC_SIM\IsaacSim\tools\packman\python.bat` convention, from `omni_capture.py`'s
     own pre-orchestrator docstring, turned out not to match his actual install layout; override
     the default only if your install lives somewhere else still, e.g. the
     `Physical-AI-Learning-KAT\tools\packman\python.bat` convention some earlier scripts used).
   - No `trimesh`/cache-permission setup needed for this path — those were specific to the
     Docker/Isaac-Sim-6.0.1-image combination in steps 8.1/8.2 above; the native install is
     whatever Bartosz already has configured and working outside this project.
   - `prep_split.default`/`prep_motion.default` are unaffected by any of this — they stay on the
     `isaac` Docker container path (steps 8.1–8.4 above still apply to them).
   - See `INSTRUCTIONS.md`'s locked decision and
     `.claude_notes/NOTES_pipeline_orchestration.md`'s "adjust the project plan" entry for the full
     write-up (including the original diagnosis of the Vulkan/WSL2 failure).

## Troubleshooting quick-reference

- `docker run --gpus all ... nvidia-smi` fails → fix step 1 before anything else; nothing else
  here can work around broken GPU passthrough.
- `ensure_image("isaac")` / pull fails with an auth error → step 2 (`docker login nvcr.io`).
- `ensure_image("cuda")` builds but is missing files / build fails partway → step 3
  (`git submodule update --init --recursive`).
- `exec_in_container("cuda", ["ls", "/omniverse"])` is empty or errors → step 4 (assets root not
  actually where `PIPELINE_ASSETS_ROOT` says, or the env var isn't set for wherever yours live).
- `import pipeline` fails → step 5 (`uv sync --extra orchestrator` / `pip install -e
  '.[orchestrator]'` not run, or run in the wrong environment).
