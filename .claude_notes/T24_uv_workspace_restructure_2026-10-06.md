# T24 — uv workspace restructure & environment cleanup (2026-10-06)

Task: `orchestrator/planning/tasks/T24-uv-workspace-restructure.md`. Done.

## What changed

- **Root is a real thin package** (`4dgs-motion-amp`, hatchling, body `src/amp/__init__.py` —
  T25 puts the CLI there) whose `dependencies` list all 13 workspace members. A plain
  `uv sync` at the root now installs the whole project; `--package` syncs remain for Docker.
- **New members**: `scene-gen` (setuptools flat layout: 24 `py-modules` + `identity` package
  with new `__init__.py`; deps pipeline/omniverse-pipeline/4dgs-core/motion-seg + numpy/scipy/
  matplotlib/opencv/plyfile/sklearn) and `mbs-bootstrap` (see below). Dead
  `depth-diff-gaussian-rasterization` source alias removed (real member: `diff-gaussian-rasterization`).
- **`requires-python` unified to `==3.12.12`** in orchestrator, 4 camera_sync packages, both CUDA
  submodules, scene-gen, mbs shim.
- **opencv floors unified to `>=4.9.0`** (core/amp-ui previously `>=5.0.0.93`). Rationale: nothing
  in the repo uses 5.x-only APIs; camera_sync already ran on 4.9; `uv lock` resolves cleanly
  (opencv-python 4.12.x). If a 5.x-only feature is ever needed, bump all three packages together.
- **PEP 735 dependency-groups**: root `test = ["pytest>=8"]`, `dev = [{include-group="test"}]`;
  orchestrator mirrors `test = ["pytest>=8", "mcp>=1.28"]` because `uv run --group test` resolves
  groups against the *current* project, and the MCP tests import httpx/mcp. pytest was previously
  undeclared (ambient).
- **sys.path hacks removed**: all 11 scene-gen scripts (orchestrator/core/scene-gen inserts),
  `motion-seg/motion_seg/mbs_infer.py` (double bootstrap → `import mbs_bootstrap; register()`),
  and the same core-dir hack in `amp-ui/amp_ui/ampUI.py` + `run_renders_auto.py` (render_amp is an
  installed top-level module now).
- **Deletions**: root `requirements.txt`, `submodules/multibody-sync-4dgs/requirements.txt`,
  all `__pycache__` trees containing cpython-37 artifacts (root, core/*, submodule rasterizer,
  omniverse-pipeline, scene-gen). Reference docs updated (`README_4DGS.md`, mbs `README.md`).
- **Dockerfile**: COPY lines for the previously missing member manifests (camera_sync ×4,
  scene-gen; the mbs pyproject rides along with the existing `COPY submodules/`).
- **Import safety**: `scene-gen/re_render_grid_runs.py` and `scene-gen/show_cubes_seg.py` ran their
  whole workload at module import (the first smoke run actually launched 7 docker render jobs!).
  Both got `main()` + `__main__` guards. NOTE: `show_cubes_seg.py` has a pre-existing data bug
  (`int()` on a float CSV field in runs/cubes_seg_summary.csv) — untouched, only guarded.
- **Smoke test**: `scripts/smoke_workspace_imports.py` — 40 required imports, platform-aware
  (CUDA-extension-dependent modules are required on linux, expected-skip elsewhere), Isaac-only
  `pxr` modules are expected skips. mbs_bootstrap.register() idempotency asserted.

## mbs packaging decision (shim, not member-of-record)

`submodules/multibody-sync-4dgs` is upstream MultiBodySync code with bare top-level modules
(`models`, `utils`, `ext`, `config`). Installing it as flat top-level modules was rejected because
its `utils` package **collides with 4dgs-core's installed top-level `utils`**
(`core/pyproject.toml` `packages.find` includes `utils*`) — last-one-installed-wins site-packages
corruption. Repackaging under a namespace was rejected (upstream divergence, `ext/__init__.py`
JIT-compiles CUDA at import). Chosen: shim package `mbs-bootstrap` (`mbs_bootstrap.py` +
pyproject inside the submodule) that prepends the submodule dir to `sys.path` with the same
insert-at-0 semantics as the old hack, now dependency-managed. Its deps carry only what the
`mbs_infer` MotNet path needs (pyquaternion/tqdm/tensorboardx/pyyaml); the stale `open3d==0.11.2`
pin was dropped (nothing in the MotNet path imports open3d).
**Follow-up**: `pyproject.toml` + `mbs_bootstrap.py` live inside the git submodule — they must be
committed there (fork `BartoszMoczkowski/multibody-sync-4dgs`) or a fresh clone breaks `uv lock`.

## mmcv decision — kept cu121/torch2.4 wheel, documented as known risk

Checked https://download.openmmlab.com/mmcv/dist/ (2026-10-06): the newest cp312 manylinux wheels
are cu121/torch2.4.0 and cu118/torch2.4.0 — **no cu126 or torch≥2.5 build exists**. The existing
URL pin stays. Risk (unvalidated): mmcv 2.2.0 built against torch 2.4 loaded under torch
2.13.0+cu126 in the CUDA container may ABI-break at runtime. mmcv is linux-only
(`sys_platform == 'linux'` marker); Windows venvs never see it. Follow-up decision for Bartosz:
test `import mmcv` in the cuda image; if broken, options are building mmcv from source in the
image or dropping the mmcv dependency (mmengine covers the config loading render_amp/train need).

## torch / cpu-gpu design — DEVIATION from plan

The plan called for a `cpu` extra with a source override + empty `gpu` marker extra +
`[tool.uv] conflicts`, keeping cu126 as the unconditional default. **uv 0.9.13 cannot express
this**: a source entry qualified by extra/group plus an unqualified entry for the same package is
a hard "conflicting indexes" error inside the fork (verified empirically for both extras and
groups; the uv PyTorch guide's extras pattern requires *both* sources extra-qualified, leaving no
GPU default — plain sync falls back to PyPI, whose Windows torch is CPU-only, and
`uv sync --package 4dgs-core` resolves against 4dgs-core's own groups, not the root's, so the
Docker path would silently get PyPI torch 2.14.x+cu13). Also tried: cu126 as `default = true`
index (uv still preferred PyPI's plain 2.14.1 over 2.14.1+cu126) — rejected.

Final design: **torch/torchvision/torchaudio stay unconditionally sourced from the cu126 index**
(pre-T24 behavior, GPU default on every sync path, Dockerfile unchanged). CPU-only envs are a
documented manual swap: `uv sync && uv pip install torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/cpu` (AGENTS.md §4.1). The one-lockfile-cpu-fork goal is dropped.

Two torch-side fixes that WERE needed:
1. **torchvision win32 hashless wheel**: the cu126 index lists the cp312 win_amd64 wheel of
   0.28.0+cu126 without a `#sha256=` fragment → uv hash-mismatch on Windows sync. Worked around by
   direct-URL source entries (win32 + linux x86_64/aarch64); the pins also keep torch at the
   validated 2.13.0+cu126 / torchvision 0.28.0+cu126 uniformly on all desktop platforms (darwin
   and exotic arches fork to whatever the indexes have — unused here).
2. **CUDA extensions are linux-only deps**: `diff-gaussian-rasterization` / `simple-knn` now carry
   `sys_platform == 'linux'` markers in `core/pyproject.toml`. There is no CUDA toolkit (nvcc) on
   the Windows host, so they can never build there (pre-T24 this meant 4dgs-core simply couldn't
   sync on Windows at all). Consequence: core's renderer modules are importable on linux only;
   the smoke test encodes this. The Docker build (linux) is unaffected.

## Verification results (2026-10-06, Windows host)

- `uv lock` + `uv lock --check`: pass (166 packages).
- Root `uv sync`: clean; `uv run python scripts/smoke_workspace_imports.py`: **40/40 required
  imports green**, 13 expected skips (10 CUDA-gated on win32, pxr-only modules; pxr is actually
  importable here via usd-core, reported as informational).
- Scratch env (`UV_PROJECT_ENVIRONMENT`) `uv sync --frozen --package 4dgs-core` (Docker path):
  succeeds, installs torch 2.13.0+cu126 + torchvision 0.28.0+cu126 (verified by import).
- `cd orchestrator && uv run --group test pytest -q`: **313 passed, 9 skipped, 2 failed** — both
  failures are environmental, not packaging regressions:
  `test_gpu_status_over_real_http_with_valid_token` asserts this machine has NO GPU (it has one),
  `test_list_containers_without_a_docker_daemon...` asserts no Docker daemon (Docker Desktop is
  running). Both would fail identically pre-T24 on this machine.
- `grep -rn "sys.path.insert" scene-gen/ motion-seg/ amp-ui/`: clean.
- No live references to the deleted requirements.txt files (doc/historical notes only).
