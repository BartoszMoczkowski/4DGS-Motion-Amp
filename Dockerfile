# Use a development-optimized CUDA base image (includes nvcc, headers, etc.)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for Python, git, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    python3-dev \
    build-essential \
    libgl1 \
    libgomp1 \
    libegl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libice6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Build the venv (and the editable-installed CUDA extensions it pulls in) in /opt/build, NOT
# /workspace. /workspace gets bind-mounted with the *live* host repo at container runtime
# (pipeline/containers/config.py's mounts_for("cuda")/T06's container_mounts), which completely
# replaces whatever this image wrote there during the build -- bind mounts shadow the underlying
# image layer entirely, they don't merge with it. Found on T11's real-hardware run (2026-07-18):
# `docker exec pipeline-cuda ls /workspace/.venv/bin/` came back "No such file or directory" even
# though the build had completed successfully and the running container was confirmed to be using
# the freshly-built image (not a stale one) -- the venv was built, then immediately shadowed the
# moment the container actually started. The same would happen to `diff-gaussian-rasterization`/
# `simple-knn`'s compiled `.so` files, since `editable = true` (pyproject.toml's
# `[tool.uv.sources]`) builds those in place inside `submodules/...`, which is also under
# `/workspace`. Building here instead -- entirely outside the bind-mounted path -- means both the
# venv and the compiled extensions survive into the running container.
WORKDIR /opt/build

# Copy the dependency files first to leverage Docker caching
COPY pyproject.toml uv.lock  ./
COPY submodules/ ./submodules/
RUN ls -la /*

# `docker build` never has GPU passthrough (only `docker run --gpus` does), so torch can't
# auto-detect a device's compute capability while compiling the `diff-gaussian-rasterization`/
# `simple-knn` CUDA extensions below -- without this, torch.utils.cpp_extension's arch-detection
# finds zero devices and crashes with `IndexError: list index out of range` trying to append
# "+PTX" to an empty arch list. Set explicitly for Bartosz's RTX 3090 (compute capability 8.6);
# update this if the build ever needs to target a different GPU.
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

# Create the virtual environment and install dependencies
# --frozen ensures uv uses the exact versions in uv.lock without updating it
RUN uv venv .venv && \
    uv sync --frozen

# Expose the virtual environment's binary folder to the PATH
# This fulfills the requirement to expose the obtained python file/binaries
ENV PATH="/opt/build/.venv/bin:$PATH"

# Enable external access to uv's cache if needed, and force color output
ENV UV_LINK_MODE=copy

# Back to /workspace as the default working directory: this is where the *live* repo bind mount
# lands at container runtime, and where a stage's exec() workdir/script paths expect to be (e.g.
# pipeline.stages.cuda_common's PYTHONPATH=/workspace) -- only the build itself needed to happen
# somewhere else.
WORKDIR /workspace
