"""Static per-environment container config: images, mounts, GPU/env-var/keep-alive settings.

Pure data — nothing here talks to Docker (``manager.py`` is the only module that does). Mirrors
the two existing devcontainer definitions and ``run_capture.sh`` exactly, per
``planning/tasks/T08-container-manager.md``'s "Notes/gotchas": *"Reuse the devcontainer
definitions as the source of truth rather than re-specifying images/mounts."*

- ``.devcontainer/devcontainer.json`` — the ``cuda`` image (built from the repo ``Dockerfile``),
  ``--gpus=all --ipc=host``, repo + Omniverse-assets bind mounts.
- ``omniverse_pipeline/.devcontainer/devcontainer.json`` — the ``isaac`` image
  (``nvcr.io/nvidia/isaac-sim:6.0.1``), ``--gpus all``, the EULA/privacy-consent env vars Isaac
  Sim needs to start non-interactively, the same repo/assets binds, plus persisted shader/compute/
  asset cache *volumes* (Isaac's slow first-run warm-up) and an overridden entrypoint so the
  container stays up instead of running (and exiting after) the Kit app.

Per ``planning/ARCHITECTURE.md``'s "Vendored stage logic": there is no separate mount for
``pipeline/vendored/cuda|isaac`` — it already lives under the repo root, which T06's
``container_mounts`` binds to ``/workspace``, so a container-side stage sees it there for free.
"""

from __future__ import annotations

from typing import Optional

from ..paths import Env, MountSpec, container_mounts

#: matches ``.devcontainer/devcontainer.json``'s ``build.dockerfile`` (relative to its
#: ``build.context: ".."``, i.e. the repo root).
CUDA_DOCKERFILE = "Dockerfile"

#: local image tag the manager builds and looks for — never pulled from a registry (there is no
#: registry copy of this image; it's built fresh from the repo ``Dockerfile``).
CUDA_IMAGE = "4dgs-motion-amp-cuda:latest"

#: matches ``omniverse_pipeline/.devcontainer/devcontainer.json``'s ``image`` — pulled from NGC.
ISAAC_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"

IMAGES: dict[Env, str] = {"cuda": CUDA_IMAGE, "isaac": ISAAC_IMAGE}

#: matches each devcontainer.json's ``runArgs: ["--gpus", "all", ...]`` — both environments need
#: full GPU passthrough (the two GPU images never run concurrently, see ``ARCHITECTURE.md``).
GPU_ALL: dict[Env, bool] = {"cuda": True, "isaac": True}

#: matches ``.devcontainer/devcontainer.json``'s extra ``--ipc=host`` (PyTorch's shared-memory
#: dataloader workers need it); Isaac's devcontainer.json has no such arg.
IPC_MODE: dict[Env, Optional[str]] = {"cuda": "host", "isaac": None}

#: matches ``omniverse_pipeline/.devcontainer/devcontainer.json``'s ``containerEnv`` — required
#: for Isaac Sim/Omniverse Kit to accept the EULA and start non-interactively (no vars needed for
#: the plain CUDA/PyTorch image).
CONTAINER_ENV: dict[Env, dict[str, str]] = {
    "cuda": {},
    "isaac": {
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    },
}

#: both devcontainer.json files set ``"overrideCommand": true`` so the container stays alive for
#: VS Code to attach to instead of running (and, for Isaac, exiting right after) its image
#: ENTRYPOINT. The manager needs the same "stay up, exec into it" model for a warm long-lived
#: container it can run several stages against — a plain sleep loop replicates that.
KEEP_ALIVE_CMD: list[str] = ["sleep", "infinity"]

#: Isaac Sim's shader/compute/OV-data caches, persisted across container recreation so only the
#: very first run pays the slow warm-up (``omniverse_pipeline/.devcontainer/devcontainer.json``'s
#: ``mounts``). The ``claude-config`` volume in that same file is a devcontainer-only convenience
#: for running Claude Code *inside* the container interactively — irrelevant here, not replicated.
CACHE_VOLUMES: dict[Env, list[MountSpec]] = {
    "cuda": [],
    "isaac": [
        MountSpec(source="isaac-cache", target="/isaac-sim/.cache", type="volume", consistency=None),
        MountSpec(
            source="isaac-compute", target="/isaac-sim/.nv/ComputeCache", type="volume", consistency=None
        ),
        MountSpec(
            source="isaac-ovdata",
            target="/isaac-sim/.local/share/ov/data",
            type="volume",
            consistency=None,
        ),
    ],
}


def container_name(env: Env) -> str:
    """Deterministic container name for ``env`` — how a fresh ``ContainerManager`` (e.g. after a
    process restart) finds and reuses an already-warm container instead of starting a duplicate,
    without having to remember a container id anywhere.
    """
    return f"pipeline-{env}"


def mounts_for(env: Env) -> list[MountSpec]:
    """Every bind/volume mount ``env``'s container needs: T06's repo+assets binds (also how
    ``pipeline/vendored/cuda|isaac`` gets into the container, see the module docstring) plus this
    env's persisted cache volumes, if any.
    """
    return [*container_mounts(env), *CACHE_VOLUMES[env]]
