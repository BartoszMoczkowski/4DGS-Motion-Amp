# Isaac Sim 5.1 + Claude Code dev container

Runs a **local Claude Code agent inside a container that has the Isaac Sim runtime and your
RTX GPU**, so the agent can execute `omniverse_pipeline/omni_capture.py` on your hardware.
(The hosted Cowork session can't do this — it has no GPU/Docker. This is the way to have *an*
agent run the capture: you launch Claude Code locally against this container.)

How it works (per the [Claude Code dev container docs](https://code.claude.com/docs/en/devcontainer)):
the repo is bind-mounted into the container, Claude Code runs inside it, and every command the
agent runs executes in the container — which here is the Isaac Sim image with `--gpus all`.

## Prerequisites

1. **RTX GPU + driver.** On Windows: Docker Desktop with the **WSL2** backend and NVIDIA GPU
   support enabled. On Linux: the **NVIDIA Container Toolkit** (`nvidia-ctk`) so `--gpus all`
   works. Verify: `docker run --rm --gpus all nvcr.io/nvidia/isaac-sim:5.1.0 nvidia-smi`.
2. **NGC access** for the base image:
   ```bash
   docker login nvcr.io        # username: $oauthtoken   password: <your NGC API key>
   docker pull nvcr.io/nvidia/isaac-sim:5.1.0
   ```
   Get an API key at https://ngc.nvidia.com (Setup → API Key). First pull accepts the EULA
   (the `ACCEPT_EULA=Y` env in devcontainer.json).
3. **VS Code + Dev Containers extension** (or the `devcontainer` CLI), and Claude Code.

## Install

1. Copy this folder to the repo's dev-container location (it can't live in `.devcontainer`
   from the Cowork session — protected path — so copy it yourself):
   ```
   cp -r omniverse_pipeline/devcontainer_isaacsim  .devcontainer/isaacsim
   ```
2. **Point it at your Omniverse assets.** The container expects them at `/omniverse`. Set the
   host path via the `OMNIVERSE_DIR` env var before opening (the `mounts` line reads it):
   - Linux: `export OMNIVERSE_DIR=/data/Omniverse`
   - Windows/WSL2: make `Q:` reachable from WSL2, then `export OMNIVERSE_DIR=/mnt/q/Omniverse`
     (set it in your WSL shell / VS Code `terminal.integrated.env`). If `Q:` is a network
     drive, copy the assets into the WSL2 filesystem instead and point there.
   Or hard-code the `source=` on that `mounts` line to an absolute path.
3. In VS Code: **Dev Containers: Reopen in Container** → pick "Isaac Sim 5.1 + Claude Code".
   First build pulls ~20 GB and warms the shader cache (slow once; cached after).

## Run the capture

Inside the container (VS Code terminal, or `claude` running in it):

```bash
# smoke test FIRST (2 cams, 2 frames) — shakes out any Replicator/API mismatch fast
./.devcontainer/isaacsim/run_capture.sh --n-cameras 2 --frames 2

# full capture (10 cams, 60 frames)
./.devcontainer/isaacsim/run_capture.sh

# convert to 4DGS multipleview
python3 omniverse_pipeline/omni_to_4dgs.py \
    --capture /omniverse/renders/capture_pump --out /workspace --name pump01
```

Or just tell the local Claude Code agent: *"run the pump capture (smoke test first), then
convert it to 4DGS."* — it has the tools and GPU inside this container.

## Notes / caveats

- **Not tested here.** These files were authored in the hosted session (no GPU/Docker), so the
  first `Reopen in Container` is the real test. Most likely edits: the `OMNIVERSE_DIR` mount,
  and (if the Isaac Sim image later changes its user) `containerUser`/`remoteUser` from `1234`.
- **First run of `omni_capture.py`** is still an Isaac 5.1 API smoke test (semantics helper,
  `BasicWriter` subfolder naming, light aim) — hence the `--n-cameras 2 --frames 2` step.
- **Auth:** run `claude` in the container and sign in; the `claude-config` volume persists it
  across rebuilds.
- **Caches** are named Docker volumes (`isaac-cache`, `isaac-compute`, `isaac-ovdata`) so the
  slow shader warm-up only happens once.
- This is separate from the repo's existing `.devcontainer/` (the CUDA/PyTorch container for
  4DGS training / MBS). Use that one for training, this one for capture.
