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
# expect "120 passed, 6 skipped" -- the 6 skips are test_containers_gpu.py,
# auto-skipping until a real Docker daemon is reachable (see step 6).
```

## 6. Run the real GPU/Isaac checks

Now that Docker + GPU + NGC + submodules + assets are all in place:

```powershell
cd orchestrator
pytest -q -s tests/test_containers_gpu.py                        # cuda build, GPU, mounts, warm-reuse, teardown
$env:PIPELINE_TEST_ISAAC = "1"; pytest -q -s tests/test_containers_gpu.py   # + isaac pull/EULA/cache (slow first run, ~10GB+ pull)
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
