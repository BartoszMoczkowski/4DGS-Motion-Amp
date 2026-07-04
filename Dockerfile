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
    libx11-6 \
    libxext6 \
    libxrender1 \
    libice6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up the working directory inside the container
WORKDIR /workspace

# Copy the dependency files first to leverage Docker caching
COPY pyproject.toml uv.lock ./

# Create the virtual environment and install dependencies
# --frozen ensures uv uses the exact versions in uv.lock without updating it
RUN uv venv .venv && \
    uv sync --frozen

# Expose the virtual environment's binary folder to the PATH
# This fulfills the requirement to expose the obtained python file/binaries
ENV PATH="/workspace/.venv/bin:$PATH"

# Enable external access to uv's cache if needed, and force color output
ENV UV_LINK_MODE=copy
